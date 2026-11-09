#!/usr/bin/env python3
"""Audit three collaboration features from an AFlow replay result."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


def _load_tokenizer(model_path: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("transformers is required; run this script with the project .venv") from exc
    try:
        return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception as exc:
        raise SystemExit(f"cannot load local tokenizer from {model_path!r}: {exc}") from exc


def _token_ids(tokenizer: Any, text: Any) -> list[int]:
    encoded = tokenizer(str(text or ""), add_special_tokens=False)
    if hasattr(encoded, "get"):
        encoded = encoded.get("input_ids", [])
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    return [int(value) for value in encoded]


def _matched_target_positions(
    sources: Iterable[list[int]],
    target: list[int],
    *,
    min_match_tokens: int,
) -> set[int]:
    """Return target-token positions copied from any source token sequence."""
    matched: set[int] = set()
    for source in sources:
        if len(source) < min_match_tokens or len(target) < min_match_tokens:
            continue
        # SequenceMatcher returns one preferred alignment when the same
        # source occurs repeatedly in a prompt.  AFlow's ensemble prompt can
        # intentionally contain the same candidate several times, so count
        # every exact occurrence first.
        exact_match_count = 0
        for start in range(0, len(target) - len(source) + 1):
            if target[start:start + len(source)] == source:
                matched.update(range(start, start + len(source)))
                exact_match_count += 1
        if exact_match_count:
            continue
        matcher = SequenceMatcher(None, source, target, autojunk=False)
        for block in matcher.get_matching_blocks():
            if block.size >= min_match_tokens:
                matched.update(range(block.b, block.b + block.size))
    return matched


def _iter_runs(payload: dict[str, Any]) -> Iterable[tuple[str, str, dict[str, Any]]]:
    runs = payload.get("runs")
    if not isinstance(runs, dict):
        raise SystemExit("input is not an AFlow reproduction result: missing runs")
    for dataset, dataset_runs in runs.items():
        if not isinstance(dataset_runs, dict):
            continue
        for run_id, run in dataset_runs.items():
            if isinstance(run, dict):
                yield str(dataset), str(run_id), run


def _audit_run(
    dataset: str,
    run_id: str,
    run: dict[str, Any],
    tokenizer: Any,
    *,
    min_match_tokens: int,
) -> list[dict[str, Any]]:
    mode = str(run.get("communication_mode", "original"))
    records = run.get("model_call_records", [])
    if not isinstance(records, list) or not records:
        return []
    missing_trace = [record for record in records if "prompt" not in record or "output" not in record]
    if missing_trace:
        raise SystemExit(
            f"{dataset} run {run_id} predates AFlow trace instrumentation. "
            "Rerun scripts/reproduce_aflow_paper.py without --resume."
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        problem_id = str(record.get("problem_id") or f"problem_index:{record.get('problem_index', index)}")
        grouped[problem_id].append(record)

    audited: list[dict[str, Any]] = []
    for problem_id, calls in grouped.items():
        prior_outputs: list[list[int]] = []
        downstream_tokens = 0
        repeated_output_tokens = 0
        total_output_tokens = 0
        phase_counts = {
            "initial_code_generation_calls": 0,
            "programmer_calls": 0,
            "refine_calls": 0,
            "detailed_calls": 0,
            "generate_calls": 0,
            "ensemble_calls": 0,
            "reflection_calls": 0,
            "custom_fix_calls": 0,
            "other_calls": 0,
        }
        for call in calls:
            prompt_text = str(call.get("prompt", ""))
            is_native_ensemble_prompt = (
                "solution_letter" in prompt_text
                and (
                    "Several solutions have been generated" in prompt_text
                    or "Several candidate Python solutions" in prompt_text
                    or "Allowed candidate IDs" in prompt_text
                )
            )
            is_compact_weighted_prompt = (
                "Candidates:" in prompt_text
                and "Reply with one ID only" in prompt_text
            )
            math_refine = dataset == "MATH" and prompt_text.lstrip().startswith(
                "Given the mathematical problem and the output from the code execution,"
            )
            math_detailed = dataset == "MATH" and prompt_text.lstrip().startswith(
                "Provide a comprehensive, step-by-step solution to the given mathematical problem."
            )
            math_generate = dataset == "MATH" and prompt_text.lstrip().startswith(
                "Please solve the given mathematical problem step by step."
            )
            math_programmer = dataset == "MATH" and prompt_text.lstrip().startswith(
                "You are a professional Python programmer."
            )
            humaneval_reflection = (
                dataset == "HumanEval"
                and prompt_text.lstrip().startswith(
                    "Given a code problem and a python code solution which failed to pass test or execute,"
                )
            )
            humaneval_repair = (
                dataset == "HumanEval"
                and prompt_text.lstrip().startswith(
                    "The previous solution failed some test cases."
                )
            )
            if is_native_ensemble_prompt or is_compact_weighted_prompt:
                phase_counts["ensemble_calls"] += 1
            elif math_refine:
                phase_counts["refine_calls"] += 1
            elif math_detailed:
                phase_counts["detailed_calls"] += 1
            elif math_generate:
                phase_counts["generate_calls"] += 1
            elif math_programmer:
                phase_counts["programmer_calls"] += 1
                phase_counts["initial_code_generation_calls"] += 1
            elif humaneval_reflection:
                phase_counts["reflection_calls"] += 1
            elif humaneval_repair:
                phase_counts["custom_fix_calls"] += 1
            elif dataset == "HumanEval":
                # HumanEval's initial CustomCodeGenerate prompt is the native
                # problem text itself.  Its two exceptional prompt families
                # are classified above, so remaining calls are initial code
                # generation rather than generic "other" calls.
                phase_counts["initial_code_generation_calls"] += 1
            elif "Generate a Python function" in prompt_text and "Function signature" in prompt_text:
                phase_counts["initial_code_generation_calls"] += 1
            elif "Failed solution:" in prompt_text or "FIX_CODE_PROMPT" in prompt_text:
                phase_counts["custom_fix_calls"] += 1
            elif "reflection" in prompt_text.casefold() or "test_fail" in prompt_text.casefold():
                phase_counts["reflection_calls"] += 1
            else:
                phase_counts["other_calls"] += 1
            prompt_ids = _token_ids(tokenizer, call.get("prompt", ""))
            output_ids = _token_ids(tokenizer, call.get("output", ""))
            downstream_tokens += len(
                _matched_target_positions(prior_outputs, prompt_ids, min_match_tokens=min_match_tokens)
            )
            repeated_output_tokens += len(
                _matched_target_positions(prior_outputs, output_ids, min_match_tokens=min_match_tokens)
            )
            total_output_tokens += len(output_ids)
            prior_outputs.append(output_ids)
        audited.append({
            "dataset": dataset,
            "run": run_id,
            "communication_mode": mode,
            "problem_id": problem_id,
            "llm_calls": len(calls),
            "intermediate_output_tokens_in_downstream_prompts": downstream_tokens,
            "cross_node_repeated_tokens": repeated_output_tokens,
            "cross_node_output_tokens": total_output_tokens,
            "cross_node_duplicate_semantic_ratio": (
                repeated_output_tokens / total_output_tokens if total_output_tokens else 0.0
            ),
            "phase_call_counts": phase_counts,
        })
    return audited


def _aflow_telemetry(payload: dict[str, Any]) -> dict[str, int]:
    fields = (
        "aflow_operator_invocations",
        "aflow_actual_llm_calls",
        "aflow_invocation_reuse_hits",
        "aflow_candidate_occurrences",
        "aflow_canonical_candidates",
        "aflow_duplicate_candidate_occurrences",
        "aflow_candidate_tokens_before_dedup",
        "aflow_candidate_tokens_after_dedup",
        "aflow_operator_argument_tokens_before_dedup",
        "aflow_operator_argument_tokens_after_dedup",
        "aflow_delivered_candidate_count",
        "aflow_candidate_occurrence_count",
        "aflow_dedup_saved_tokens",
        "aflow_weighted_ensemble_calls",
        "aflow_weighted_unique_candidates",
        "aflow_weighted_candidate_occurrences",
        "aflow_weighted_ensemble_fallbacks",
        "aflow_native_ensemble_prompt_tokens",
        "aflow_weighted_ensemble_prompt_tokens",
        "aflow_weighted_candidate_content_tokens",
        "aflow_weighted_metadata_tokens",
        "aflow_weighted_prompt_saved_tokens",
    )
    totals = {field: 0 for field in fields}
    for _, _, run in _iter_runs(payload):
        telemetry = run.get("graph_telemetry", [])
        if not isinstance(telemetry, list):
            continue
        for summary in telemetry:
            if not isinstance(summary, dict):
                continue
            for field in fields:
                totals[field] += int(summary.get(field, 0) or 0)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="results.json from scripts/reproduce_aflow_paper.py")
    parser.add_argument("--output", type=Path, help="optional audit JSON path")
    parser.add_argument(
        "--min-match-tokens",
        type=int,
        default=8,
        help="minimum contiguous token span counted as transmitted/repeated content",
    )
    args = parser.parse_args()
    if args.min_match_tokens < 1:
        raise SystemExit("--min-match-tokens must be positive")
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    model_path = str(payload.get("model_path", ""))
    if not model_path:
        raise SystemExit("AFlow result is missing model_path")
    tokenizer = _load_tokenizer(model_path)

    tasks: list[dict[str, Any]] = []
    for dataset, run_id, run in _iter_runs(payload):
        tasks.extend(_audit_run(dataset, run_id, run, tokenizer, min_match_tokens=args.min_match_tokens))
    if not tasks:
        raise SystemExit("no completed AFlow model-call records found")

    total_downstream = sum(item["intermediate_output_tokens_in_downstream_prompts"] for item in tasks)
    total_repeated = sum(item["cross_node_repeated_tokens"] for item in tasks)
    total_output = sum(item["cross_node_output_tokens"] for item in tasks)
    calls = [item["llm_calls"] for item in tasks]
    phase_totals = {
        key: sum(item.get("phase_call_counts", {}).get(key, 0) for item in tasks)
        for key in (
            "initial_code_generation_calls",
            "programmer_calls",
            "refine_calls",
            "detailed_calls",
            "generate_calls",
            "ensemble_calls",
            "reflection_calls",
            "custom_fix_calls",
            "other_calls",
        )
    }
    report = {
        "subject": "AFlow",
        "communication_mode": sorted({item["communication_mode"] for item in tasks}),
        "method": {
            "downstream_tokens": "unique downstream prompt token positions matching an earlier operator output",
            "duplicate_ratio": (
                "later-output token positions matching an earlier operator output / all operator output tokens; "
                "deterministic repeated-content proxy for semantic duplication"
            ),
            "min_contiguous_match_tokens": args.min_match_tokens,
        },
        "three_metrics": {
            "questions": len(tasks),
            "llm_calls_per_question": {
                "mean": statistics.mean(calls),
                "median": statistics.median(calls),
                "min": min(calls),
                "max": max(calls),
            },
            "intermediate_output_tokens_in_downstream_prompts": {
                "total": total_downstream,
                "mean_per_question": total_downstream / len(tasks),
            },
            "cross_node_duplicate_semantic_ratio": (
                total_repeated / total_output if total_output else 0.0
            ),
        },
        "aflow_telemetry": _aflow_telemetry(payload),
        "phase_call_counts": phase_totals,
        "tasks": tasks,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
