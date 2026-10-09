"""Run reasoning upper-bound experiments for repairable benchmark domains.

This script intentionally bypasses the graph communication workflow.  It asks:

* can the backbone produce a correct candidate if sampled N times?
* can a deterministic/oracle selector pick a correct candidate?
* can structured verifier feedback repair an initially wrong candidate?
* how far can repeated full-context repair go before communication compression?

Supported domains are the current repairable set: gsm8k, humaneval, mbpp.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.evaluate_domain_workflow import (  # noqa: E402
    extract_code,
    extract_numeric_answer,
    read,
    render_native_task,
    score_prediction,
)
from workflow_runtime.domain_executors import execute_python_tests_detailed  # noqa: E402
from workflow_runtime.candidate_graph import (  # noqa: E402
    add_candidate_selection_decision,
    add_verifier_guided_selection,
    coverage_at_k,
    materialize_candidate_pool,
)
from workflow_runtime.candidate_selector import select_candidate_with_deployable_verifier  # noqa: E402
from workflow_runtime.candidate_selector_communication import (  # noqa: E402
    build_candidate_fragments,
    rendered_fragment_tokens,
    run_selector_with_fragments,
    select_fragment_ids,
)
from workflow_runtime.graph_store import GraphStore  # noqa: E402
from scripts.evaluate_deployable_candidate_selector import code_features, numeric_features  # noqa: E402


DOMAINS = ("gsm8k", "humaneval", "mbpp")
SELECTOR_COMMUNICATION_POLICIES = (
    "deployable_full",
    "selector_send_all",
    "selector_core_only",
    "selector_verifier_only",
    "selector_receiver_aware",
    "selector_random_same_budget",
)


@dataclass
class Candidate:
    text: str
    score: dict[str, Any]
    input_tokens: int
    output_tokens: int
    latency_sec: float
    seed: int
    repair_round: int = 0
    feedback: dict[str, Any] | None = None


@dataclass(frozen=True)
class AdaptiveDecision:
    action: str
    reason: str
    should_stop: bool = False


class SamplingModel:
    def __init__(self, model_path: str, *, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
        signature = inspect.signature(AutoModelForCausalLM.from_pretrained)
        kwargs["dtype" if "dtype" in signature.parameters else "torch_dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **kwargs,
        )
        self.model.eval()
        self.max_new_tokens = int(max_new_tokens)

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> tuple[str, dict[str, int | float]]:
        self.torch.manual_seed(int(seed))
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(int(seed))
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        device = next(self.model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": bool(do_sample),
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            kwargs["temperature"] = max(1e-5, float(temperature))
            kwargs["top_p"] = float(top_p)
        started = time.time()
        with self.torch.inference_mode():
            out = self.model.generate(**kwargs)
        elapsed = time.time() - started
        new_ids = out[0, input_ids.shape[-1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return text, {
            "input_tokens": int(input_ids.shape[-1]),
            "output_tokens": int(new_ids.shape[-1]),
            "latency_sec": elapsed,
        }


def task_prompt(row: dict[str, Any], domain: str) -> str:
    if domain == "gsm8k":
        return (
            "Solve the math word problem. Show concise reasoning, then write the final answer "
            "as 'Final answer: <number>'.\n\n"
            f"Problem:\n{row['question']}"
        )
    return render_native_task(row)


def score_candidate(row: dict[str, Any], domain: str, text: str) -> dict[str, Any]:
    if domain in {"humaneval", "mbpp"}:
        return score_prediction(row, extract_code(text))
    return score_prediction(row, text)


def candidate_artifact(domain: str, text: str) -> tuple[str, Any, Any]:
    if domain in {"humaneval", "mbpp"}:
        code = extract_code(text)
        return "code", code, None
    final_value = extract_numeric_answer(text)
    return "result", final_value, final_value


def tests_for_row(row: dict[str, Any]) -> tuple[list[str], str]:
    tests = row.get("tests", [])
    setup = "\n".join(str(item.get("setup", "")) for item in tests if isinstance(item, dict))
    test_text = [str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in tests]
    return test_text, setup


def code_failure_detail(row: dict[str, Any], text: str) -> dict[str, Any]:
    tests, setup = tests_for_row(row)
    code = extract_code(text)
    result = execute_python_tests_detailed(code=code, tests=tests, setup=setup, timeout=5)
    return dict(result.execution)


def build_repair_feedback(row: dict[str, Any], domain: str, candidate: Candidate) -> dict[str, Any]:
    if domain in {"humaneval", "mbpp"}:
        detail = code_failure_detail(row, candidate.text)
        kind = str(detail.get("kind") or "execution_failure")
        failed_test = str(detail.get("failed_test") or "")
        expected = detail.get("expected")
        actual = detail.get("actual")
        function_name = str(detail.get("function_name") or "the target function")
        if kind == "assertion_failure" and (expected is not None or actual is not None):
            instruction = (
                f"For this failing test, {function_name} returned {actual}; it must return "
                f"{expected}. Fix the implementation for this concrete case while preserving "
                "the function signature."
            )
        else:
            instruction = (
                "Fix the implementation so all provided tests pass. Preserve the required "
                "function signature and return only executable Python code."
            )
        return {
            "feedback_type": "oracle_repair_signal",
            "kind": kind,
            "failed_test": failed_test,
            "test_expression": detail.get("test_expression"),
            "function_name": function_name,
            "input": detail.get("input_repr"),
            "expected": expected,
            "actual": actual,
            "exception_type": detail.get("exception_type"),
            "exception": detail.get("exception"),
            "repair_instruction": instruction,
        }
    predicted = extract_numeric_answer(candidate.text)
    return {
        "feedback_type": "oracle_repair_signal",
        "kind": "numeric_answer_incorrect",
        "predicted_answer": predicted,
        "repair_instruction": (
            "The previous final numeric answer is incorrect. Recompute the word problem from "
            "the quantities and relationships, check units, and provide a new final answer. "
            "Do not simply repeat the previous answer."
        ),
    }


def repair_prompt(row: dict[str, Any], domain: str, previous: Candidate, feedback: dict[str, Any], *, round_id: int) -> str:
    if domain in {"humaneval", "mbpp"}:
        tests = "\n".join(tests_for_row(row)[0])
        return (
            "You are repairing a Python solution. Return only executable Python code, no Markdown.\n\n"
            f"Task:\n{row['question']}\n\n"
            f"Provided tests:\n{tests}\n\n"
            f"Previous candidate:\n{extract_code(previous.text)}\n\n"
            f"Verifier feedback:\n{json.dumps(feedback, ensure_ascii=False, indent=2)}\n\n"
            "Repair only the faulty behavior and preserve correct behavior."
        )
    return (
        "You are repairing a math solution. Give concise corrected reasoning and end with "
        "'Final answer: <number>'.\n\n"
        f"Problem:\n{row['question']}\n\n"
        f"Previous solution:\n{previous.text}\n\n"
        f"Verifier feedback:\n{json.dumps(feedback, ensure_ascii=False, indent=2)}\n"
    )


def generate_candidate(
    model: SamplingModel,
    row: dict[str, Any],
    domain: str,
    prompt: str,
    *,
    seed: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    repair_round: int = 0,
    feedback: dict[str, Any] | None = None,
) -> Candidate:
    text, metrics = model.generate(
        prompt,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )
    return Candidate(
        text=text,
        score=score_candidate(row, domain, text),
        input_tokens=int(metrics["input_tokens"]),
        output_tokens=int(metrics["output_tokens"]),
        latency_sec=float(metrics["latency_sec"]),
        seed=seed,
        repair_round=repair_round,
        feedback=feedback,
    )


def run_best_of_n(
    model: SamplingModel,
    rows: list[dict[str, Any]],
    *,
    domain: str,
    n: int,
    seed: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    records = []
    for index, row in enumerate(rows, start=1):
        prompt = task_prompt(row, domain)
        candidates = [
            generate_candidate(
                model,
                row,
                domain,
                prompt,
                seed=seed + index * 1009 + attempt,
                temperature=temperature,
                top_p=top_p,
                do_sample=attempt > 0,
            )
            for attempt in range(n)
        ]
        correct_any = any(c.score.get("correct") for c in candidates)
        correct_count = sum(int(bool(c.score.get("correct"))) for c in candidates)
        candidate_records = [candidate_record(c, domain=domain, candidate_id=f"c{attempt}") for attempt, c in enumerate(candidates)]
        coverage = coverage_at_k(candidate_records, (1, 2, 4, 8, 16))
        records.append({
            "sample_id": row.get("sample_id"),
            "correct": correct_any,
            "correct_count": correct_count,
            "candidate_count": len(candidates),
            "best_of_n": n,
            **coverage,
            "candidates": candidate_records,
            "candidate_graph": record_candidate_graph(
                row,
                domain=domain,
                group_id=f"{row.get('sample_id', index)}_best_of_{n}",
                candidates=candidate_records,
                selector_policy="verifier_guided_first_verified",
                selected_candidate_id=None,
                metadata={"experiment": "best_of_n", "n": n},
            ),
        })
        print(f"{index}/{len(rows)} {row.get('sample_id')} best_of_{n} correct={correct_any} correct_count={correct_count}", flush=True)
    return summarize(records, domain=domain, experiment="best_of_n", n=n)


def run_oracle_select(
    model: SamplingModel,
    rows: list[dict[str, Any]],
    *,
    domain: str,
    n: int,
    seed: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    report = run_best_of_n(
        model,
        rows,
        domain=domain,
        n=n,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
    )
    report["experiment"] = "oracle_select"
    report["selector"] = "unit_tests" if domain in {"humaneval", "mbpp"} else "gold_numeric_oracle"
    report["uses_gold_selector"] = domain == "gsm8k"
    return report


def run_verifier_select(
    model: SamplingModel,
    rows: list[dict[str, Any]],
    *,
    domain: str,
    n: int,
    seed: int,
    temperature: float,
    top_p: float,
    current_baseline: float = 0.0,
) -> dict[str, Any]:
    records = []
    for index, row in enumerate(rows, start=1):
        prompt = task_prompt(row, domain)
        candidates = [
            generate_candidate(
                model,
                row,
                domain,
                prompt,
                seed=seed + index * 1009 + attempt,
                temperature=temperature,
                top_p=top_p,
                do_sample=attempt > 0,
            )
            for attempt in range(n)
        ]
        candidate_records = [
            candidate_record(candidate, domain=domain, candidate_id=f"c{attempt}")
            for attempt, candidate in enumerate(candidates)
        ]
        deployable_records = deployable_candidate_records(candidate_records, domain=domain, row=row)
        selected, candidate_graph = record_deployable_selection_graph(
            row,
            domain=domain,
            group_id=f"{row.get('sample_id', index)}_verifier_select_{n}",
            candidates=deployable_records,
        )
        selected_original = next(
            item for item in candidate_records
            if item.get("candidate_id") == selected.selected_candidate_id
        )
        correct_any = any(c.get("correct") for c in candidate_records)
        selected_correct = bool(selected_original.get("correct"))
        coverage = coverage_at_k(candidate_records, (1, 2, 4, 8, 16))
        records.append({
            "sample_id": row.get("sample_id"),
            "correct": selected_correct,
            "oracle_has_correct": correct_any,
            "selected_candidate_id": selected.selected_candidate_id,
            "selected_candidate_score": selected.selection_score,
            "selected_candidate_confidence": selected.confidence,
            "selection_reason": list(selected.selection_reason),
            "candidate_count": len(candidate_records),
            "best_of_n": n,
            **coverage,
            "candidates": candidate_records,
            "deployable_candidates": deployable_records,
            "candidate_graph": candidate_graph,
            "selection_trace": list(selected.trace),
        })
        print(
            f"{index}/{len(rows)} {row.get('sample_id')} verifier_select_{n} "
            f"correct={selected_correct} oracle_has_correct={correct_any} selected={selected.selected_candidate_id}",
            flush=True,
        )
    report = summarize(records, domain=domain, experiment="verifier_select", n=n)
    selected_correct = sum(int(bool(record.get("correct"))) for record in records)
    false_positive = sum(int(not bool(record.get("correct"))) for record in records)
    false_negative = sum(
        int((not bool(record.get("correct"))) and bool(record.get("oracle_has_correct")))
        for record in records
    )
    count = len(records)
    oracle = float(report.get(f"coverage@{n}", report.get("accuracy_or_pass_at_n", 0.0)) or 0.0)
    verifier = selected_correct / count if count else 0.0
    report.update({
        "current_baseline": current_baseline,
        "oracle_select_at_n": oracle,
        "verifier_select_at_n": verifier,
        "gap_verifier": oracle - verifier,
        "gap_recovery": (
            (verifier - current_baseline) / (oracle - current_baseline)
            if oracle > current_baseline else 0.0
        ),
        "selection_accuracy": verifier,
        "false_positive_selection_count": false_positive,
        "false_negative_selection_count": false_negative,
        "mean_selection_confidence": (
            sum(float(record.get("selected_candidate_confidence", 0.0) or 0.0) for record in records) / count
            if count else 0.0
        ),
        "passes_phase_gate": verifier >= current_baseline if current_baseline else None,
    })
    return report


def run_select_repair(
    model: SamplingModel,
    rows: list[dict[str, Any]],
    *,
    domain: str,
    n: int,
    seed: int,
    temperature: float,
    top_p: float,
    verifier_baseline: float = 0.0,
    experiment_name: str = "select_repair",
    min_repair_candidates: int = 4,
    selector_communication_policy: str = "deployable_full",
) -> dict[str, Any]:
    records = []
    for index, row in enumerate(rows, start=1):
        prompt = task_prompt(row, domain)
        candidates: list[Candidate] = []
        candidate_records: list[dict[str, Any]] = []
        deployable_records: list[dict[str, Any]] = []
        selected = None
        candidate_graph: dict[str, Any] = {}
        selected_index = 0
        action = "GENERATE"
        actions: list[str] = []
        controller_reasons: list[str] = []
        generate_count = 0
        selector_communication_steps: list[dict[str, Any]] = []

        while len(candidates) < max(1, n):
            attempt = len(candidates)
            generated = generate_candidate(
                model,
                row,
                domain,
                prompt,
                seed=seed + index * 1009 + attempt,
                temperature=temperature,
                top_p=top_p,
                do_sample=attempt > 0,
            )
            generate_count += 1
            actions.append("GENERATE")
            candidates.append(generated)
            candidate_records = [
                candidate_record(candidate, domain=domain, candidate_id=f"c{candidate_index}")
                for candidate_index, candidate in enumerate(candidates)
            ]
            deployable_records = deployable_candidate_records(candidate_records, domain=domain, row=row)
            selected, candidate_graph, selector_comm = select_candidate_for_adaptive_controller(
                row=row,
                domain=domain,
                group_id=f"{row.get('sample_id', index)}_select_repair_{n}_step_{len(candidates)}",
                candidates=deployable_records,
                policy=selector_communication_policy,
                seed=seed + index * 1009 + attempt,
            )
            selector_communication_steps.append(selector_comm)
            selected_index = _candidate_index_from_id(selected.selected_candidate_id, len(candidates))
            selected_deployable = deployable_records[selected_index]
            decision = adaptive_controller_decision(
                domain=domain,
                selected_candidate=selected_deployable,
                selection=selected,
                candidate_count=len(candidates),
                max_candidates=n,
                min_repair_candidates=min_repair_candidates,
            )
            action = decision.action
            controller_reasons.append(decision.reason)
            if action == "REQUEST":
                actions.append("REQUEST")
                action = "GENERATE"
            if action == "SELECT":
                actions.extend(["SELECT", "STOP"])
                break
            if action == "REPAIR":
                actions.append("REPAIR")
                break

        assert selected is not None
        selected_index = _candidate_index_from_id(selected.selected_candidate_id, len(candidates))
        selected_candidate = candidates[selected_index]
        selected_record = candidate_records[selected_index]
        selected_deployable = deployable_records[selected_index]
        selected_correct = bool(selected_record.get("correct"))
        repair_attempt: Candidate | None = None
        repair_feedback: dict[str, Any] | None = None
        final_candidate = selected_candidate
        final_record = selected_record

        if action == "REPAIR" and _has_actionable_failure(domain, selected_deployable):
            repair_feedback = build_repair_feedback(row, domain, selected_candidate)
            repair_attempt = generate_candidate(
                model,
                row,
                domain,
                repair_prompt(row, domain, selected_candidate, repair_feedback, round_id=1),
                seed=seed + index * 1009 + n + 1,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repair_round=1,
                feedback=repair_feedback,
            )
            final_candidate = repair_attempt
            final_record = candidate_record(repair_attempt, domain=domain, candidate_id=f"{selected.selected_candidate_id}_repair_1")
        elif action != "SELECT":
            action = "SELECT"
            actions.extend(["SELECT", "STOP"])

        before_correct = selected_correct
        after_correct = bool(final_record.get("correct"))
        transition = repair_transition(before_correct, after_correct)
        coverage = coverage_at_k(candidate_records, (1, 2, 4, 8, 16))
        record = {
            "sample_id": row.get("sample_id"),
            "correct": after_correct,
            "selected_candidate_id": selected.selected_candidate_id,
            "selected_correct": before_correct,
            "final_candidate_id": final_record.get("candidate_id"),
            "transition": transition,
            "action": action,
            "actions": actions,
            "controller_reasons": controller_reasons,
            "repair_attempted": repair_attempt is not None,
            "repair_success": transition == "W->C",
            "repair_regression": transition == "C->W",
            "extra_generation": generate_count > 1,
            "extra_generation_count": max(0, generate_count - 1),
            "generation_attempt_count": generate_count,
            "candidate_reused": final_record.get("candidate_id") == selected_record.get("candidate_id"),
            "candidate_count": len(candidate_records),
            "max_candidate_budget": n,
            **coverage,
            "selected": selected.as_dict(),
            "candidate_graph": candidate_graph,
            "candidates": candidate_records,
            "feedback": repair_feedback,
            "selector_communication_steps": selector_communication_steps,
        }
        if repair_attempt:
            record["repair"] = candidate_record(repair_attempt, domain=domain, candidate_id=f"{selected.selected_candidate_id}_repair_1")
        records.append(record)
        print(
            f"{index}/{len(rows)} {row.get('sample_id')} select_repair_{n} "
            f"actions={','.join(actions)} transition={transition} correct={after_correct}",
            flush=True,
        )
    report = summarize(records, domain=domain, experiment=experiment_name, n=n)
    correct_count = int(report.get("correct_count", 0) or 0)
    repair_attempt_count = sum(int(record.get("repair_attempted", False)) for record in records)
    successful_repair_count = sum(int(record.get("repair_success", False)) for record in records)
    repair_regression_count = sum(int(record.get("repair_regression", False)) for record in records)
    extra_generation_count = sum(int(record.get("extra_generation_count", 0) or 0) for record in records)
    candidate_reuse_count = sum(int(record.get("candidate_reused", False)) for record in records)
    wrong_to_correct = sum(int(record.get("transition") == "W->C") for record in records)
    correct_to_wrong = sum(int(record.get("transition") == "C->W") for record in records)
    controller_action_counts: dict[str, int] = {}
    for record in records:
        for controller_action in record.get("actions", []):
            controller_action_counts[controller_action] = controller_action_counts.get(controller_action, 0) + 1
    report.update({
        "verifier_baseline": verifier_baseline,
        "min_repair_candidates": min_repair_candidates,
        "select_repair_accuracy": report.get("accuracy_or_pass_at_n", 0.0),
        "adaptive_controller_accuracy": report.get("accuracy_or_pass_at_n", 0.0),
        "controller_action_counts": controller_action_counts,
        "select_action_count": controller_action_counts.get("SELECT", 0),
        "request_action_count": controller_action_counts.get("REQUEST", 0),
        "generate_action_count": controller_action_counts.get("GENERATE", 0),
        "repair_action_count": controller_action_counts.get("REPAIR", 0),
        "stop_action_count": controller_action_counts.get("STOP", 0),
        "phase5_gate_select_repair_ge_verifier": (
            float(report.get("accuracy_or_pass_at_n", 0.0) or 0.0) >= verifier_baseline
            if verifier_baseline else None
        ),
        "repair_attempt_count": repair_attempt_count,
        "successful_repair_count": successful_repair_count,
        "repair_regression_count": repair_regression_count,
        "extra_generation_count": extra_generation_count,
        "generation_attempt_count": sum(int(record.get("generation_attempt_count", 0) or 0) for record in records),
        "candidate_reuse_count": candidate_reuse_count,
        "candidate_reuse_rate": candidate_reuse_count / len(records) if records else 0.0,
        "wrong_to_correct_count": wrong_to_correct,
        "correct_to_wrong_count": correct_to_wrong,
        "net_repair_gain": wrong_to_correct - correct_to_wrong,
        "cw_regression_rate": correct_to_wrong / max(1, sum(int(record.get("selected_correct", False)) for record in records)),
        "tokens_per_correct": (
            int(report.get("total_model_tokens", 0) or 0) / correct_count
            if correct_count else None
        ),
        "observed_candidate_model_tokens": sum(
            sum(int(candidate.get("model_tokens", 0) or 0) for candidate in record.get("candidates", []))
            for record in records
        ),
        "estimated_bruteforce_best_of_n_model_tokens": 0,
        "bruteforce_tokens_per_oracle_correct": None,
        "selector_communication_policy": selector_communication_policy,
        "selector_input_tokens": sum(
            sum(int(step.get("selector_input_tokens", 0) or 0) for step in record.get("selector_communication_steps", []))
            for record in records
        ),
        "selector_send_all_equivalent_tokens": sum(
            sum(int(step.get("selector_send_all_equivalent_tokens", 0) or 0) for step in record.get("selector_communication_steps", []))
            for record in records
        ),
    })
    report["selector_token_saving_vs_send_all"] = (
        1.0 - report["selector_input_tokens"] / report["selector_send_all_equivalent_tokens"]
        if report["selector_send_all_equivalent_tokens"] else 0.0
    )
    generated_candidate_tokens = [
        int(candidate.get("model_tokens", 0) or 0)
        for record in records
        for candidate in record.get("candidates", [])
    ]
    avg_candidate_tokens = (
        sum(generated_candidate_tokens) / len(generated_candidate_tokens)
        if generated_candidate_tokens else 0.0
    )
    report["estimated_bruteforce_best_of_n_model_tokens"] = avg_candidate_tokens * n * len(records)
    oracle_correct = sum(int(any(candidate.get("correct") for candidate in record.get("candidates", []))) for record in records)
    if oracle_correct:
        report["bruteforce_tokens_per_oracle_correct"] = report["estimated_bruteforce_best_of_n_model_tokens"] / oracle_correct
        report["phase5_gate_tokens_per_correct_better_than_bruteforce"] = (
            report["tokens_per_correct"] is not None
            and report["tokens_per_correct"] <= report["bruteforce_tokens_per_oracle_correct"]
        )
    else:
        report["phase5_gate_tokens_per_correct_better_than_bruteforce"] = None
    return report


def run_oracle_repair(
    model: SamplingModel,
    rows: list[dict[str, Any]],
    *,
    domain: str,
    seed: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    records = []
    for index, row in enumerate(rows, start=1):
        initial = generate_candidate(
            model,
            row,
            domain,
            task_prompt(row, domain),
            seed=seed + index * 1009,
            temperature=temperature,
            top_p=top_p,
            do_sample=False,
        )
        if initial.score.get("correct"):
            records.append({
                "sample_id": row.get("sample_id"),
                "initial_correct": True,
                "correct": True,
                "wrong_to_correct": False,
                "candidate": candidate_record(initial, domain=domain, candidate_id="initial"),
                "candidate_graph": record_candidate_graph(
                    row,
                    domain=domain,
                    group_id=f"{row.get('sample_id', index)}_oracle_repair_initial",
                    candidates=[candidate_record(initial, domain=domain, candidate_id="initial")],
                    selector_policy="verifier_guided_first_verified",
                    selected_candidate_id=None,
                    metadata={"experiment": "oracle_repair", "initial_correct": True},
                ),
            })
            print(f"{index}/{len(rows)} {row.get('sample_id')} initial_correct=True", flush=True)
            continue
        feedback = build_repair_feedback(row, domain, initial)
        repaired = generate_candidate(
            model,
            row,
            domain,
            repair_prompt(row, domain, initial, feedback, round_id=1),
            seed=seed + index * 1009 + 1,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repair_round=1,
            feedback=feedback,
        )
        records.append({
            "sample_id": row.get("sample_id"),
            "initial_correct": False,
            "correct": bool(repaired.score.get("correct")),
            "wrong_to_correct": bool(repaired.score.get("correct")),
            "initial": candidate_record(initial, domain=domain, candidate_id="initial"),
            "repair": candidate_record(repaired, domain=domain, candidate_id="repair_1"),
            "feedback": feedback,
            "candidate_graph": record_candidate_graph(
                row,
                domain=domain,
                group_id=f"{row.get('sample_id', index)}_oracle_repair",
                candidates=[
                    candidate_record(initial, domain=domain, candidate_id="initial"),
                    candidate_record(repaired, domain=domain, candidate_id="repair_1"),
                ],
                selector_policy="verifier_guided_first_verified",
                selected_candidate_id=None,
                metadata={"experiment": "oracle_repair", "initial_correct": False},
            ),
        })
        print(f"{index}/{len(rows)} {row.get('sample_id')} oracle_repair correct={bool(repaired.score.get('correct'))}", flush=True)
    return summarize(records, domain=domain, experiment="oracle_repair")


def run_iterative_repair(
    model: SamplingModel,
    rows: list[dict[str, Any]],
    *,
    domain: str,
    max_rounds: int,
    seed: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    records = []
    for index, row in enumerate(rows, start=1):
        attempts: list[Candidate] = []
        current = generate_candidate(
            model,
            row,
            domain,
            task_prompt(row, domain),
            seed=seed + index * 1009,
            temperature=temperature,
            top_p=top_p,
            do_sample=False,
        )
        attempts.append(current)
        first_success_round = 0 if current.score.get("correct") else None
        for round_id in range(1, max_rounds + 1):
            if first_success_round is not None:
                break
            feedback = build_repair_feedback(row, domain, current)
            current = generate_candidate(
                model,
                row,
                domain,
                repair_prompt(row, domain, current, feedback, round_id=round_id),
                seed=seed + index * 1009 + round_id,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repair_round=round_id,
                feedback=feedback,
            )
            attempts.append(current)
            if current.score.get("correct"):
                first_success_round = round_id
                break
        records.append({
            "sample_id": row.get("sample_id"),
            "correct": first_success_round is not None,
            "first_success_round": first_success_round,
            "attempt_count": len(attempts),
            "attempts": [
                candidate_record(c, domain=domain, candidate_id=f"round_{attempt_index}")
                for attempt_index, c in enumerate(attempts)
            ],
            "candidate_graph": record_candidate_graph(
                row,
                domain=domain,
                group_id=f"{row.get('sample_id', index)}_iterative",
                candidates=[
                    candidate_record(c, domain=domain, candidate_id=f"round_{attempt_index}")
                    for attempt_index, c in enumerate(attempts)
                ],
                selector_policy="verifier_guided_first_verified",
                selected_candidate_id=None,
                metadata={"experiment": "iterative_repair", "max_rounds": max_rounds},
            ),
        })
        print(f"{index}/{len(rows)} {row.get('sample_id')} iterative correct={first_success_round is not None} round={first_success_round}", flush=True)
    return summarize(records, domain=domain, experiment="iterative_repair", max_rounds=max_rounds)


def candidate_record(candidate: Candidate, *, domain: str | None = None, candidate_id: str | None = None) -> dict[str, Any]:
    artifact_type = "text"
    artifact = candidate.text
    final_value = None
    if domain is not None:
        artifact_type, artifact, final_value = candidate_artifact(domain, candidate.text)
    return {
        "candidate_id": candidate_id,
        "correct": bool(candidate.score.get("correct")),
        "score": candidate.score,
        "artifact_type": artifact_type,
        "artifact": artifact,
        "final_value": final_value,
        "input_tokens": candidate.input_tokens,
        "output_tokens": candidate.output_tokens,
        "model_tokens": candidate.input_tokens + candidate.output_tokens,
        "latency_sec": candidate.latency_sec,
        "seed": candidate.seed,
        "repair_round": candidate.repair_round,
        "raw_tail": candidate.text[-1000:],
        "feedback": candidate.feedback,
    }


def first_correct_candidate_id(candidates: list[dict[str, Any]]) -> str | None:
    for candidate in candidates:
        if candidate.get("correct"):
            return str(candidate.get("candidate_id"))
    return None


def deployable_candidate_records(
    candidates: list[dict[str, Any]],
    *,
    domain: str,
    row: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    answer_counts: dict[str, int] = {}
    if domain == "gsm8k":
        for candidate in candidates:
            value = str(candidate.get("final_value") or extract_numeric_answer(candidate.get("raw_tail", ""))).strip()
            if value:
                answer_counts[value] = answer_counts.get(value, 0) + 1
    records = []
    for candidate in candidates:
        if domain == "gsm8k":
            signals = numeric_features(candidate, answer_counts, len(candidates))
        else:
            signals = code_features(candidate)
            if domain == "mbpp" and row is not None:
                signals = {**signals, **mbpp_allowed_test_features(row, candidate)}
        records.append({
            **candidate,
            "score": signals,
            "verification_signals": signals,
            "verification_source": (
                "deployable_numeric_check"
                if domain == "gsm8k"
                else "deployable_static_generated_tests"
            ),
        })
    return records


def mbpp_allowed_test_features(row: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    code = str(candidate.get("artifact") or candidate.get("raw_tail") or "")
    tests, setup = tests_for_row(row)
    if not tests:
        return {}
    result = execute_python_tests_detailed(code=code, tests=tests, setup=setup, timeout=5)
    detail = dict(result.execution)
    if result.success:
        return {
            "tests_passed": True,
            "public_tests_passed": True,
            "verifier_pass": True,
            "failed_test_count": 0,
            "test_pass_rate": 1.0,
            "confidence": 0.9,
        }
    kind = str(detail.get("kind") or "execution_failure")
    features: dict[str, Any] = {
        "tests_passed": False,
        "public_tests_passed": False,
        "tests_failed": True,
        "failed_test_count": 1,
        "test_pass_rate": 0.0,
        "confidence": 0.1,
        "failure_kind": kind,
        "failed_test": detail.get("failed_test"),
        "test_expression": detail.get("test_expression"),
        "function_name": detail.get("function_name"),
        "input_repr": detail.get("input_repr"),
        "expected": detail.get("expected"),
        "actual": detail.get("actual"),
        "exception_type": detail.get("exception_type"),
        "exception": detail.get("exception"),
    }
    if kind == "assertion_failure":
        features["assertion_failure"] = True
    elif kind == "timeout":
        features["timeout"] = True
    elif kind == "syntax_error":
        features["syntax_error"] = True
        features["compile_error"] = True
    elif kind in {"type_error", "name_error", "import_error", "runtime_exception"}:
        features["runtime_exception"] = True
    return features


def repair_transition(before_correct: bool, after_correct: bool) -> str:
    if before_correct and after_correct:
        return "C->C"
    if before_correct and not after_correct:
        return "C->W"
    if not before_correct and after_correct:
        return "W->C"
    return "W->W"


def _candidate_index_from_id(candidate_id: str, candidate_count: int) -> int:
    try:
        index = int(str(candidate_id).lstrip("c") or 0)
    except ValueError:
        index = 0
    return max(0, min(index, max(0, candidate_count - 1)))


def _is_high_confidence_verifier_pass(domain: str, candidate: dict[str, Any], selection: Any, *, candidate_count: int) -> bool:
    score = candidate.get("score") if isinstance(candidate.get("score"), dict) else {}
    if domain == "gsm8k":
        consensus = float(score.get("cross_candidate_answer_consensus") or 0.0)
        passed = bool(
            candidate_count >= 2
            and
            consensus > 0.5
            and (
                score.get("trace_result_agreement")
                or (
                    score.get("final_value_valid")
                    and score.get("arithmetic_consistency")
                    and score.get("constraint_consistency")
                )
            )
        )
    else:
        passed = bool(
            score.get("tests_passed")
            or score.get("verifier_pass")
            or score.get("public_tests_passed")
        ) and not bool(
            score.get("syntax_error")
            or score.get("compile_error")
            or score.get("runtime_exception")
            or score.get("timeout")
        )
    feature_confidence = float(score.get("confidence") or 0.0)
    selector_confidence = float(getattr(selection, "confidence", 0.0) or 0.0)
    return passed and max(feature_confidence, selector_confidence) >= 0.2


def adaptive_controller_decision(
    *,
    domain: str,
    selected_candidate: dict[str, Any],
    selection: Any,
    candidate_count: int,
    max_candidates: int,
    min_repair_candidates: int = 4,
) -> AdaptiveDecision:
    """Rule-based controller over SELECT/REQUEST/REPAIR/GENERATE/STOP.

    The controller is intentionally deployable: it only reads verifier features
    and selector metadata, never gold correctness.  Domain priors reflect the
    observed Phase 5 boundary: GSM8K/HumanEval are selection-heavy, while MBPP
    can benefit from targeted repair when a concrete failure signal exists.
    """

    if _is_high_confidence_verifier_pass(
        domain,
        selected_candidate,
        selection,
        candidate_count=candidate_count,
    ):
        return AdaptiveDecision("SELECT", "high_confidence_verifier_pass", should_stop=True)

    if (
        domain == "mbpp"
        and candidate_count >= min_repair_candidates
        and _has_actionable_failure(domain, selected_candidate)
    ):
        return AdaptiveDecision("REPAIR", "mbpp_actionable_failure")

    if candidate_count < max_candidates:
        if _candidate_disagreement_high(domain, selected_candidate, selection):
            return AdaptiveDecision("REQUEST", "insufficient_discrimination_request_more_evidence")
        return AdaptiveDecision("GENERATE", "insufficient_confidence_generate_more")

    if domain == "mbpp" and _has_actionable_failure(domain, selected_candidate):
        return AdaptiveDecision("REPAIR", "mbpp_budget_exhausted_actionable_failure")

    return AdaptiveDecision("SELECT", "budget_exhausted_select_best", should_stop=True)


def _candidate_disagreement_high(domain: str, candidate: dict[str, Any], selection: Any) -> bool:
    score = candidate.get("score") if isinstance(candidate.get("score"), dict) else {}
    if domain == "gsm8k":
        return float(score.get("cross_candidate_answer_consensus") or 0.0) < 0.75
    if domain in {"humaneval", "mbpp"}:
        return float(getattr(selection, "confidence", 0.0) or 0.0) < 0.35
    return False


def _has_actionable_failure(domain: str, candidate: dict[str, Any]) -> bool:
    score = candidate.get("score") if isinstance(candidate.get("score"), dict) else {}
    if domain == "gsm8k":
        return bool(
            score.get("trace_result_disagreement")
            or score.get("constraint_inconsistency")
            or score.get("final_value_valid")
        )
    if score.get("tests_passed") or score.get("verifier_pass") or score.get("public_tests_passed"):
        return False
    return bool(
        score.get("assertion_failure")
        or score.get("runtime_exception")
        or score.get("timeout")
        or score.get("wrong_return_type")
        or score.get("syntax_error")
        or score.get("compile_error")
        or score.get("failed_test_count")
    )


def record_candidate_graph(
    row: dict[str, Any],
    *,
    domain: str,
    group_id: str,
    candidates: list[dict[str, Any]],
    selector_policy: str,
    selected_candidate_id: str | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    store = GraphStore()
    task_id = str(row.get("sample_id") or group_id)
    task = store.add_node(
        task_id=task_id,
        branch_id="main",
        logical_id="task",
        node_type="task",
        content={
            "sample_id": row.get("sample_id"),
            "domain": domain,
            "question": row.get("question"),
        },
        owner="dataset",
        created_by_role="dataset",
    )
    result = materialize_candidate_pool(
        store,
        task_id=task_id,
        branch_id="main",
        group_id=group_id,
        domain=domain,
        candidates=candidates,
        selector_policy=selector_policy if selected_candidate_id is not None else None,
        selected_candidate_id=selected_candidate_id,
        source_node_id=task.node_id,
        metadata=metadata or {},
    )
    if selected_candidate_id is None:
        result = add_verifier_guided_selection(
            store,
            result,
            selection_policy=selector_policy,
            metadata={"source": "candidate_verification_nodes"},
        )
    state = store.snapshot()
    node_type_counts: dict[str, int] = {}
    for node in state.nodes.values():
        node_type_counts[node.type] = node_type_counts.get(node.type, 0) + 1
    return {
        **result.as_dict(),
        "node_count": len(state.nodes),
        "edge_count": len(state.edges),
        "node_type_counts": node_type_counts,
    }


def record_deployable_selection_graph(
    row: dict[str, Any],
    *,
    domain: str,
    group_id: str,
    candidates: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    store = GraphStore()
    task_id = str(row.get("sample_id") or group_id)
    task = store.add_node(
        task_id=task_id,
        branch_id="main",
        logical_id="task",
        node_type="task",
        content={
            "sample_id": row.get("sample_id"),
            "domain": domain,
            "question": row.get("question"),
        },
        owner="dataset",
        created_by_role="dataset",
    )
    pool = materialize_candidate_pool(
        store,
        task_id=task_id,
        branch_id="main",
        group_id=group_id,
        domain=domain,
        candidates=candidates,
        source_node_id=task.node_id,
        metadata={"experiment": "verifier_select"},
    )
    selection = select_candidate_with_deployable_verifier(store, pool)
    result = add_candidate_selection_decision(
        store,
        pool,
        selected_candidate_id=selection.selected_candidate_id,
        selection_policy="deployable_feature_score",
        reason="; ".join(selection.selection_reason),
        metadata=selection.as_dict(),
    )
    state = store.snapshot()
    node_type_counts: dict[str, int] = {}
    for node in state.nodes.values():
        node_type_counts[node.type] = node_type_counts.get(node.type, 0) + 1
    return selection, {
        **result.as_dict(),
        "node_count": len(state.nodes),
        "edge_count": len(state.edges),
        "node_type_counts": node_type_counts,
    }


def select_candidate_for_adaptive_controller(
    *,
    row: dict[str, Any],
    domain: str,
    group_id: str,
    candidates: list[dict[str, Any]],
    policy: str,
    seed: int,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    fragments = build_candidate_fragments(candidates)
    send_all_ids = select_fragment_ids(fragments, policy="selector_send_all", seed=seed)
    send_all_tokens = rendered_fragment_tokens(fragments, send_all_ids)
    if policy == "deployable_full":
        selection, candidate_graph = record_deployable_selection_graph(
            row,
            domain=domain,
            group_id=group_id,
            candidates=candidates,
        )
        return selection, candidate_graph, {
            "policy": policy,
            "selected_candidate_id": selection.selected_candidate_id,
            "selector_input_tokens": send_all_tokens,
            "selector_send_all_equivalent_tokens": send_all_tokens,
            "token_saving_vs_send_all": 0.0,
            "visible_fragment_ids": list(send_all_ids),
        }

    budget_tokens = None
    if policy == "selector_random_same_budget":
        receiver_aware_ids = select_fragment_ids(fragments, policy="selector_receiver_aware", seed=seed)
        budget_tokens = rendered_fragment_tokens(fragments, receiver_aware_ids)
    selection = run_selector_with_fragments(
        candidates,
        policy=policy,
        seed=seed,
        budget_tokens=budget_tokens,
    )
    candidate_graph = record_candidate_graph(
        row,
        domain=domain,
        group_id=group_id,
        candidates=candidates,
        selector_policy=policy,
        selected_candidate_id=selection.selected_candidate_id,
        metadata=selection.as_dict(),
    )
    selector_input_tokens = int(selection.rendered_tokens)
    return selection, candidate_graph, {
        **selection.as_dict(),
        "selector_input_tokens": selector_input_tokens,
        "selector_send_all_equivalent_tokens": send_all_tokens,
        "token_saving_vs_send_all": (
            1.0 - selector_input_tokens / send_all_tokens
            if send_all_tokens else 0.0
        ),
    }


def summarize(records: list[dict[str, Any]], **metadata: Any) -> dict[str, Any]:
    count = len(records)
    correct = sum(int(bool(r.get("correct"))) for r in records)
    initial_correct = sum(int(bool(r.get("initial_correct"))) for r in records)
    wrong_to_correct = sum(int(bool(r.get("wrong_to_correct"))) for r in records)
    coverage_summary: dict[str, float] = {}
    for k in (1, 2, 4, 8, 16):
        key = f"coverage@{k}"
        values = [float(record[key]) for record in records if key in record]
        if values:
            coverage_summary[key] = sum(values) / len(values)
    input_tokens = 0
    output_tokens = 0
    for record in records:
        candidates = []
        if "candidates" in record:
            candidates.extend(record["candidates"])
        if "candidate" in record:
            candidates.append(record["candidate"])
        if "initial" in record:
            candidates.append(record["initial"])
        if "repair" in record:
            candidates.append(record["repair"])
        if "attempts" in record:
            candidates.extend(record["attempts"])
        for candidate in candidates:
            if not candidate:
                continue
            input_tokens += int(candidate.get("input_tokens", 0) or 0)
            output_tokens += int(candidate.get("output_tokens", 0) or 0)
    return {
        **metadata,
        "count": count,
        "correct_count": correct,
        "accuracy_or_pass_at_n": correct / count if count else 0.0,
        "initial_correct_count": initial_correct,
        "wrong_to_correct_count": wrong_to_correct,
        **coverage_summary,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_model_tokens": input_tokens + output_tokens,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=DOMAINS)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--experiment", required=True, choices=("best_of_n", "oracle_select", "verifier_select", "select_repair", "adaptive_controller", "oracle_repair", "iterative_repair"))
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--current-baseline", type=float, default=0.0)
    parser.add_argument("--verifier-baseline", type=float, default=0.0)
    parser.add_argument("--min-repair-candidates", type=int, default=4)
    parser.add_argument(
        "--selector-communication-policy",
        choices=SELECTOR_COMMUNICATION_POLICIES,
        default="deployable_full",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read(Path(args.data_path), args.domain, args.limit)
    model = SamplingModel(args.model_path, max_new_tokens=args.max_new_tokens)
    if args.experiment == "best_of_n":
        report = run_best_of_n(
            model,
            rows,
            domain=args.domain,
            n=args.n,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    elif args.experiment == "oracle_select":
        report = run_oracle_select(
            model,
            rows,
            domain=args.domain,
            n=args.n,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    elif args.experiment == "verifier_select":
        report = run_verifier_select(
            model,
            rows,
            domain=args.domain,
            n=args.n,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            current_baseline=args.current_baseline,
        )
    elif args.experiment in {"select_repair", "adaptive_controller"}:
        report = run_select_repair(
            model,
            rows,
            domain=args.domain,
            n=args.n,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            verifier_baseline=args.verifier_baseline,
            experiment_name=args.experiment,
            min_repair_candidates=args.min_repair_candidates,
            selector_communication_policy=args.selector_communication_policy,
        )
    elif args.experiment == "oracle_repair":
        report = run_oracle_repair(
            model,
            rows,
            domain=args.domain,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    else:
        report = run_iterative_repair(
            model,
            rows,
            domain=args.domain,
            max_rounds=args.max_rounds,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    report.update({
        "data_path": args.data_path,
        "model_path": args.model_path,
        "limit": args.limit,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {key: value for key, value in report.items() if key != "records"}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
