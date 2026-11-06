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
        for call in calls:
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
        })
    return audited


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
        "tasks": tasks,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
