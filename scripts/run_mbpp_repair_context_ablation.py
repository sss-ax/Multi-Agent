"""Run a controlled MBPP repair signal-quality ablation.

The experiment fixes, per sample, the same initial Solver0 code and the same
deterministic execution failure.  It then changes only the quality of the
feedback signal visible to the repair Solver:

    binary
    typed_error
    concrete_failure
    diagnostic
    full_context

All arms still receive the task seed context (problem statement and tests);
the ablation isolates which failure signal is sufficient.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_domain_workflow import read  # noqa: E402
from workflow_runtime.domain_executors import execute_python_tests_detailed, strip_top_level_candidate_tests  # noqa: E402
from workflow_runtime.model_backend import TransformersModel  # noqa: E402


ARMS = (
    "binary",
    "typed_error",
    "concrete_failure",
    "diagnostic",
    "full_context",
)


@dataclass
class TestRun:
    passed: bool
    detail: dict[str, Any]


def extract_python_code(text: str) -> str:
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    cleaned = str(text).strip()
    if cleaned.startswith("{"):
        try:
            payload = json.loads(cleaned)
            code = payload.get("code") if isinstance(payload, dict) else None
            if isinstance(code, str):
                return code.strip()
        except json.JSONDecodeError:
            pass
    return strip_top_level_candidate_tests(cleaned)


def run_tests(code: str, tests: list[str], *, setup: str = "", timeout: int = 5) -> TestRun:
    result = execute_python_tests_detailed(code=code, tests=tests, setup=setup, timeout=timeout)
    return TestRun(result.success, dict(result.execution))


def build_diagnosis(test_run: TestRun) -> dict[str, Any]:
    detail = dict(test_run.detail)
    failed_test = str(detail.get("failed_test", "") or "")
    kind = str(detail.get("kind") or "")
    exception_type = str(detail.get("exception_type") or "")
    if kind == "assertion_failure":
        error_type = "wrong_value"
        function_name = str(detail.get("function_name") or "the target function")
        location = failed_test or f"{function_name} return value"
        expected = detail.get("expected")
        actual = detail.get("actual")
        if expected is not None or actual is not None:
            instruction = (
                f"For {failed_test}, {function_name} returned {actual}; "
                f"it must return {expected}. Fix the logic for this case."
            )
        else:
            instruction = "Change the implementation so the failing assertion and the other provided tests pass."
    elif kind == "syntax_error":
        error_type = "syntax_error"
        location = "code syntax"
        instruction = "Fix the syntax while preserving the required function signature."
    elif kind == "import_error":
        error_type = "import_error"
        location = "imports"
        instruction = "Fix missing or invalid imports without changing the required behavior."
    elif kind == "name_error":
        error_type = "name_error"
        location = failed_test or "undefined name"
        instruction = "Fix undefined names while preserving the required function signature."
    elif kind == "type_error":
        error_type = "type_error"
        location = failed_test or "function call"
        instruction = "Fix the implementation so it handles the argument types used by the failing test."
    elif kind == "timeout":
        error_type = "timeout"
        location = "function execution"
        instruction = "Remove non-terminating behavior while preserving the intended algorithm."
    else:
        error_type = "runtime_error"
        location = failed_test or "function execution"
        instruction = "Fix the implementation so it handles the failing test without raising an exception."
    return {
        "status": "need_fix",
        "error_type": error_type,
        "error_location": location,
        "reason": detail.get("traceback", detail.get("exception", "")),
        "repair_instruction": instruction,
        "preserve": ["function signature", "working behavior already covered by passing tests"],
        "concrete_failure": {
            key: detail.get(key)
            for key in (
                "kind", "function_name", "test_id", "failed_test", "test_expression",
                "input_repr", "expected", "actual", "exception_type", "exception",
            )
            if detail.get(key) not in (None, "")
        },
        "requested_fragments": ["code#function_body", "execution#execution_detail"],
    }


def seed_context(row: dict[str, Any]) -> str:
    tests = "\n".join(str(test.get("text", test)) if isinstance(test, dict) else str(test) for test in row["tests"])
    return (
        "Task seed context:\n"
        f"{row['question']}\n\n"
        "Provided tests:\n"
        f"{tests}\n"
    )


def diagnosis_text(diagnosis: dict[str, Any]) -> str:
    keys = ("error_type", "error_location", "reason", "repair_instruction", "preserve", "concrete_failure")
    return json.dumps({key: diagnosis.get(key) for key in keys}, ensure_ascii=False, indent=2)


def execution_detail_text(test_run: TestRun) -> str:
    return json.dumps(test_run.detail, ensure_ascii=False, indent=2)


def solver0_prompt(row: dict[str, Any]) -> str:
    return (
        f"{seed_context(row)}\n"
        "Write a corrected Python solution. Return only executable Python code, with no Markdown."
    )


def repair_prompt(
    row: dict[str, Any],
    *,
    arm: str,
    diagnosis: dict[str, Any],
    previous_code: str,
    execution_detail: dict[str, Any],
    solver0_output: str,
) -> str:
    parts = [
        seed_context(row),
        "You are repairing a previous MBPP solution. Return only executable Python code, with no Markdown.",
    ]
    concrete = diagnosis.get("concrete_failure", {}) if isinstance(diagnosis.get("concrete_failure"), dict) else {}
    if arm == "binary":
        parts.extend(["Feedback signal:", json.dumps({"status": "need_fix"}, ensure_ascii=False, indent=2)])
    elif arm == "typed_error":
        parts.extend([
            "Feedback signal:",
            json.dumps({
                "status": "need_fix",
                "error_type": diagnosis.get("error_type"),
            }, ensure_ascii=False, indent=2),
        ])
    elif arm == "concrete_failure":
        parts.extend([
            "FAILED TEST:",
            "\n".join([
                f"function: {concrete.get('function_name', '')}",
                f"input: {concrete.get('input_repr', '')}",
                f"expected: {concrete.get('expected', '')}",
                f"actual: {concrete.get('actual', '')}",
                f"test: {concrete.get('failed_test', '')}",
            ]),
            "REVISION:",
            "Fix only the logic causing this mismatch. Return only the corrected function.",
        ])
    elif arm == "diagnostic":
        parts.extend([
            "FAILED TEST:",
            "\n".join([
                f"function: {concrete.get('function_name', '')}",
                f"input: {concrete.get('input_repr', '')}",
                f"expected: {concrete.get('expected', '')}",
                f"actual: {concrete.get('actual', '')}",
            ]),
            "REVISION:",
            diagnosis.get("repair_instruction", "Fix the candidate."),
        ])
    elif arm == "full_context":
        parts.extend(["Critic diagnosis:", diagnosis_text(diagnosis)])
        parts.extend(["Previous code:", previous_code])
        parts.extend(["Execution detail:", json.dumps(execution_detail, ensure_ascii=False, indent=2)])
        parts.extend([
            "Original Solver0 raw output:",
            solver0_output,
            "Repair directive:",
            "Preserve the function signature. Fix only the faulty behavior implied by the tests and execution detail.",
        ])
    return "\n\n".join(parts)


def token_count(model: TransformersModel, text: str) -> int:
    return int(len(model.tokenizer(str(text), add_special_tokens=False)["input_ids"]))


def call_model(model: TransformersModel, prompt: str) -> tuple[str, int, int]:
    output = model(type("Request", (), {"prompt": prompt, "session_prompt": prompt})())
    metrics = dict(getattr(model, "last_call_metrics", {}) or {})
    return output, int(metrics.get("physical_input_tokens", token_count(model, prompt))), int(metrics.get("output_tokens", 0))


def evaluate_rows(model: TransformersModel, rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "arms": {
            arm: {
                "attempted": 0,
                "wrong_to_correct": 0,
                "repair_input_tokens": 0,
                "repair_output_tokens": 0,
                "repair_model_tokens": 0,
                "records": [],
            }
            for arm in ARMS
        },
        "records": [],
        "solver0_correct_count": 0,
        "solver0_wrong_count": 0,
        "assertion_failure_count": 0,
        "concrete_failure_complete_count": 0,
    }
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", f"mbpp_{index}"))
        raw0, solver0_input_tokens, solver0_output_tokens = call_model(model, solver0_prompt(row))
        code0 = extract_python_code(raw0)
        initial = run_tests(code0, [test["text"] for test in row["tests"]])
        if initial.passed:
            report["solver0_correct_count"] += 1
            report["records"].append({
                "sample_id": sample_id,
                "solver0_correct": True,
                "solver0_input_tokens": solver0_input_tokens,
                "solver0_output_tokens": solver0_output_tokens,
            })
            print(f"{index + 1}/{len(rows)} {sample_id} solver0_correct=True skip_repair", flush=True)
            continue
        report["solver0_wrong_count"] += 1
        diagnosis = build_diagnosis(initial)
        concrete = diagnosis.get("concrete_failure", {})
        if concrete.get("kind") == "assertion_failure":
            report["assertion_failure_count"] += 1
            if concrete.get("failed_test") and concrete.get("expected") is not None and concrete.get("actual") is not None:
                report["concrete_failure_complete_count"] += 1
        sample_record: dict[str, Any] = {
            "sample_id": sample_id,
            "solver0_correct": False,
            "solver0_input_tokens": solver0_input_tokens,
            "solver0_output_tokens": solver0_output_tokens,
            "initial_execution": initial.detail,
            "diagnosis": diagnosis,
            "arms": {},
        }
        for arm in ARMS:
            prompt = repair_prompt(
                row,
                arm=arm,
                diagnosis=diagnosis,
                previous_code=code0,
                execution_detail=initial.detail,
                solver0_output=raw0,
            )
            output, input_tokens, output_tokens = call_model(model, prompt)
            code = extract_python_code(output)
            repaired = run_tests(code, [test["text"] for test in row["tests"]])
            arm_summary = report["arms"][arm]
            arm_summary["attempted"] += 1
            arm_summary["repair_input_tokens"] += input_tokens
            arm_summary["repair_output_tokens"] += output_tokens
            arm_summary["repair_model_tokens"] += input_tokens + output_tokens
            if repaired.passed:
                arm_summary["wrong_to_correct"] += 1
            arm_record = {
                "correct": repaired.passed,
                "repair_input_tokens": input_tokens,
                "repair_output_tokens": output_tokens,
                "repair_model_tokens": input_tokens + output_tokens,
                "repair_output_limit_hit": bool(output_tokens >= int(getattr(model, "max_new_tokens", 0) or 0))
                if getattr(model, "max_new_tokens", None) else False,
                "execution": repaired.detail,
            }
            arm_summary["records"].append({"sample_id": sample_id, **arm_record})
            sample_record["arms"][arm] = arm_record
        report["records"].append(sample_record)
        compact = {
            arm: int(bool(sample_record["arms"][arm]["correct"]))
            for arm in ARMS
        }
        print(f"{index + 1}/{len(rows)} {sample_id} solver0_correct=False repairs={compact}", flush=True)
    for arm, summary in report["arms"].items():
        attempted = int(summary["attempted"])
        successes = int(summary["wrong_to_correct"])
        repair_input = int(summary["repair_input_tokens"])
        summary["repair_success_rate"] = successes / attempted if attempted else 0.0
        summary["repair_token_efficiency"] = successes / repair_input if repair_input else 0.0
        summary["avg_repair_input_tokens"] = repair_input / attempted if attempted else 0.0
        summary["avg_repair_model_tokens"] = int(summary["repair_model_tokens"]) / attempted if attempted else 0.0
        summary["repair_output_limit_hit_count"] = sum(
            int(bool(record.get("repair_output_limit_hit")))
            for record in summary["records"]
        )
    assertion_failures = int(report["assertion_failure_count"])
    report["concrete_failure_completeness_rate"] = (
        int(report["concrete_failure_complete_count"]) / assertion_failures
        if assertion_failures else 0.0
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/mbpp/mbpp.jsonl")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read(Path(args.data_path), "mbpp", args.limit)
    model = TransformersModel(args.model_path, max_new_tokens=args.max_new_tokens)
    report = evaluate_rows(model, rows)
    report.update({
        "domain": "mbpp",
        "data_path": args.data_path,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "arms_order": list(ARMS),
    })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
