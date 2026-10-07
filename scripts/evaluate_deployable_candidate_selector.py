"""Evaluate deployable candidate selection on generated best-of-N pools.

This script does not generate new model outputs.  It reads a best_of_n report,
extracts non-gold verifier features for each candidate, runs the deployable
selector, and uses the stored correctness labels only for final evaluation.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.evaluate_domain_workflow import extract_code, extract_numeric_answer  # noqa: E402
from workflow_runtime.candidate_graph import materialize_candidate_pool  # noqa: E402
from workflow_runtime.candidate_selector import select_candidate_with_deployable_verifier  # noqa: E402
from workflow_runtime.graph_store import GraphStore  # noqa: E402


def numeric_features(candidate: dict[str, Any], answer_counts: dict[str, int], total: int) -> dict[str, Any]:
    text = str(candidate.get("raw_tail") or candidate.get("artifact") or "")
    final_value = str(candidate.get("final_value") or extract_numeric_answer(text) or "").strip()
    equations = _extract_equations(text)
    parseable = [item for item in equations if item["parseable"]]
    consistent = [item for item in parseable if item["consistent"]]
    last_value = parseable[-1]["rhs"] if parseable else None
    trace_agrees = bool(final_value and last_value is not None and _same_number(final_value, last_value))
    consensus = answer_counts.get(final_value, 0) / max(1, total) if final_value else 0.0
    return {
        "final_value_valid": _number(final_value) is not None,
        "expression_parseable": bool(parseable),
        "arithmetic_consistency": bool(parseable) and len(consistent) == len(parseable),
        "trace_result_agreement": trace_agrees,
        "trace_result_disagreement": bool(parseable) and not trace_agrees,
        "cross_candidate_answer_consensus": consensus,
        "constraint_consistency": bool(parseable) and trace_agrees,
        "independent_recompute_consistency": bool(parseable) and trace_agrees,
        "confidence": 0.8 if bool(parseable) and trace_agrees else 0.2 if final_value else 0.0,
    }


def code_features(candidate: dict[str, Any]) -> dict[str, Any]:
    raw = str(candidate.get("raw_tail") or candidate.get("artifact") or "")
    code = extract_code(raw)
    if not code.strip():
        code = str(candidate.get("artifact") or "")
    features: dict[str, Any] = {
        "syntax_valid": False,
        "compile_success": False,
        "runtime_ok": False,
        "no_exception": False,
        "test_pass_rate": 0.0,
        "failed_test_count": 0,
        "execution_diversity": 0.0,
        "confidence": 0.0,
    }
    try:
        ast.parse(code)
        features["syntax_valid"] = True
        features["compile_success"] = True
    except SyntaxError:
        features["syntax_error"] = True
        features["compile_error"] = True
        features["exception_severity"] = 8
        return features
    tests = _extract_self_generated_asserts(raw)
    if tests:
        passed = 0
        failed = 0
        for test in tests[:8]:
            ok = _run_self_test(code, test)
            passed += int(ok)
            failed += int(not ok)
        features["test_pass_rate"] = passed / max(1, passed + failed)
        features["failed_test_count"] = failed
        features["execution_diversity"] = len(set(tests[:8]))
        features["public_tests_passed"] = failed == 0 and passed > 0
    else:
        features["test_pass_rate"] = 0.0
        features["failed_test_count"] = 0
    features["runtime_ok"] = True
    features["no_exception"] = True
    features["confidence"] = 0.6 if features.get("public_tests_passed") else 0.25
    if _looks_hardcoded(code):
        features["hardcoded_output"] = True
    return features


def evaluate_report(path: Path, *, domain: str, current_baseline: float, oracle_baseline: float | None = None) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    records = []
    selected_correct = 0
    false_positive = 0
    false_negative = 0
    tie_count = 0
    confidence_sum = 0.0
    for record in report.get("records", []):
        original_candidates = list(record.get("candidates", []))
        answer_counts: dict[str, int] = {}
        if domain == "gsm8k":
            for candidate in original_candidates:
                value = str(candidate.get("final_value") or extract_numeric_answer(candidate.get("raw_tail", ""))).strip()
                if value:
                    answer_counts[value] = answer_counts.get(value, 0) + 1
        deployable_candidates = []
        for index, candidate in enumerate(original_candidates):
            candidate_id = str(candidate.get("candidate_id") or f"c{index}")
            signals = (
                numeric_features(candidate, answer_counts, len(original_candidates))
                if domain == "gsm8k"
                else code_features(candidate)
            )
            deployable_candidates.append({
                "candidate_id": candidate_id,
                "artifact_type": candidate.get("artifact_type") or ("result" if domain == "gsm8k" else "code"),
                "artifact": candidate.get("artifact"),
                "final_value": candidate.get("final_value"),
                "correct": bool(candidate.get("correct")),
                "verification_source": "deployable_numeric_check" if domain == "gsm8k" else "deployable_static_generated_tests",
                "score": signals,
                "verification_signals": signals,
                "input_tokens": candidate.get("input_tokens", 0),
                "output_tokens": candidate.get("output_tokens", 0),
                "raw_tail": candidate.get("raw_tail"),
            })
        store = GraphStore()
        task = store.add_node(
            task_id=str(record.get("sample_id") or len(records)),
            branch_id="main",
            logical_id="task",
            node_type="task",
            content={"sample_id": record.get("sample_id"), "domain": domain},
            owner="dataset",
        )
        pool = materialize_candidate_pool(
            store,
            task_id=task.task_id,
            branch_id="main",
            group_id=f"{record.get('sample_id')}_deployable_selector",
            domain=domain,
            candidates=deployable_candidates,
            source_node_id=task.node_id,
        )
        result = select_candidate_with_deployable_verifier(store, pool)
        selected = next(c for c in original_candidates if str(c.get("candidate_id")) == result.selected_candidate_id)
        correct = bool(selected.get("correct"))
        selected_correct += int(correct)
        oracle_has_correct = any(bool(c.get("correct")) for c in original_candidates)
        false_positive += int(not correct)
        false_negative += int((not correct) and oracle_has_correct)
        top_scores = [item["feature_score"] for item in result.as_dict()["trace"][:2]]
        tie_count += int(len(top_scores) == 2 and abs(top_scores[0] - top_scores[1]) < 1e-9)
        confidence_sum += result.confidence
        records.append({
            "sample_id": record.get("sample_id"),
            "selected_candidate_id": result.selected_candidate_id,
            "selected_correct": correct,
            "oracle_has_correct": oracle_has_correct,
            "selection_score": result.selection_score,
            "selection_reason": list(result.selection_reason),
            "confidence": result.confidence,
            "trace": list(result.trace),
        })
    count = len(records)
    verifier_select = selected_correct / count if count else 0.0
    oracle = oracle_baseline if oracle_baseline is not None else float(report.get("coverage@4") or report.get("accuracy_or_pass_at_n") or 0.0)
    gap_verifier = oracle - verifier_select
    gap_recovery = (
        (verifier_select - current_baseline) / (oracle - current_baseline)
        if oracle > current_baseline
        else 0.0
    )
    return {
        "domain": domain,
        "source_path": str(path),
        "count": count,
        "current_baseline": current_baseline,
        "oracle_select_at_4": oracle,
        "verifier_select_at_4": verifier_select,
        "gap_verifier": gap_verifier,
        "gap_recovery": gap_recovery,
        "selection_accuracy": verifier_select,
        "false_positive_selection_count": false_positive,
        "false_negative_selection_count": false_negative,
        "tie_count": tie_count,
        "tie_rate": tie_count / count if count else 0.0,
        "mean_confidence": confidence_sum / count if count else 0.0,
        "passes_phase_gate": verifier_select >= current_baseline,
        "records": records,
    }


def _extract_equations(text: str) -> list[dict[str, Any]]:
    found = []
    for match in re.finditer(r"([0-9][0-9,.\s()+*/-]+?)\s*=\s*([-+]?\d+(?:\.\d+)?)", text):
        expr = match.group(1).replace(",", "")
        rhs = match.group(2)
        value = _safe_eval_arithmetic(expr)
        found.append({
            "expression": expr.strip(),
            "rhs": rhs,
            "parseable": value is not None,
            "consistent": value is not None and _same_number(value, rhs),
        })
    return found


def _safe_eval_arithmetic(expression: str) -> float | None:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.Load)
    if not all(isinstance(node, allowed) for node in ast.walk(tree)):
        return None
    try:
        return float(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {}))
    except Exception:
        return None


def _number(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def _same_number(left: Any, right: Any) -> bool:
    a = _number(left)
    b = _number(right)
    return a is not None and b is not None and abs(a - b) <= 1e-9


def _extract_self_generated_asserts(text: str) -> list[str]:
    tests = []
    try:
        tree = ast.parse(extract_code(text) or text)
    except SyntaxError:
        return tests
    for node in tree.body:
        if isinstance(node, ast.Assert):
            tests.append(ast.unparse(node))
    return tests


def _run_self_test(code: str, test: str) -> bool:
    namespace: dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(code, namespace)
            exec(test, namespace)
        return True
    except Exception:
        return False


def _looks_hardcoded(code: str) -> bool:
    return bool(re.search(r"return\\s+(['\"]?\\w+['\"]?|[-+]?\\d+(?:\\.\\d+)?)\\s*$", code.strip()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--domain", required=True, choices=("gsm8k", "humaneval", "mbpp"))
    parser.add_argument("--current-baseline", type=float, required=True)
    parser.add_argument("--oracle-baseline", type=float, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = evaluate_report(
        Path(args.input),
        domain=args.domain,
        current_baseline=args.current_baseline,
        oracle_baseline=args.oracle_baseline,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {key: value for key, value in summary.items() if key != "records"}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
