"""Audit token-cost accounting across experiment reports.

This script keeps billed model tokens separate from logical communication
payloads.  It intentionally does not add selector/logical communication tokens
to billed totals unless those tokens are also present as model input/output in
the source report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    records = report.get("records", [])
    model_input_tokens = _int(report.get("total_input_tokens"))
    model_output_tokens = _int(report.get("total_output_tokens"))
    if not model_input_tokens and records:
        model_input_tokens = sum(_record_input_tokens(record) for record in records)
    if not model_output_tokens and records:
        model_output_tokens = sum(_record_output_tokens(record) for record in records)

    selector_input_tokens = _int(report.get("selector_input_tokens"))
    if not selector_input_tokens and records:
        selector_input_tokens = sum(
            sum(_int(step.get("selector_input_tokens")) for step in record.get("selector_communication_steps", []))
            for record in records
        )

    logical_comm_tokens = _first_int(
        report,
        (
            "logical_comm_tokens",
            "logical_communication_tokens",
            "communication_cost_tokens",
            "semantic_communication_cost_tokens",
        ),
    )
    if not logical_comm_tokens and records:
        logical_comm_tokens = sum(
            _first_int(
                record,
                (
                    "logical_comm_tokens",
                    "logical_communication_tokens",
                    "communication_tokens",
                    "communication_cost_tokens",
                ),
            )
            for record in records
        )

    repeated_context_tokens = _first_int(
        report,
        (
            "repeated_context_tokens",
            "repeated_prefix_tokens",
            "estimated_reused_input_tokens",
        ),
    )
    if not repeated_context_tokens and records:
        repeated_context_tokens = sum(
            _first_int(record, ("repeated_context_tokens", "repeated_prefix_tokens", "estimated_reused_input_tokens"))
            for record in records
        )

    total_billed_tokens = model_input_tokens + model_output_tokens
    reported_total_model_tokens = _int(report.get("total_model_tokens"))
    if not reported_total_model_tokens:
        reported_total_model_tokens = _int(report.get("inference_prefill_cost_tokens")) + _int(
            report.get("inference_decode_cost_tokens")
        )

    recomputed = _recompute_from_records(records)
    total_model_delta = (
        reported_total_model_tokens - total_billed_tokens
        if reported_total_model_tokens
        else 0
    )
    selector_likely_in_billed_input = bool(
        selector_input_tokens
        and model_input_tokens
        and selector_input_tokens <= model_input_tokens
        and abs(total_model_delta) <= max(1, int(total_billed_tokens * 0.01))
    )

    count = _int(report.get("count")) or len(records)
    wall_time_seconds = _float(report.get("wall_time_seconds")) or _float(report.get("total_latency_sec"))
    if not wall_time_seconds and records:
        wall_time_seconds = sum(_float(record.get("latency_sec")) for record in records)

    normalized = {
        "path": str(path),
        "method": report.get("method") or report.get("experiment") or report.get("baseline") or "",
        "domain": report.get("domain") or "",
        "count": count,
        "quality": _quality(report),
        "model_input_tokens": model_input_tokens,
        "model_output_tokens": model_output_tokens,
        "selector_input_tokens": selector_input_tokens,
        "logical_comm_tokens": logical_comm_tokens,
        "repeated_context_tokens": repeated_context_tokens,
        "total_billed_tokens": total_billed_tokens,
        "reported_total_model_tokens": reported_total_model_tokens,
        "num_model_calls": _first_int(report, ("forward_calls", "num_model_calls", "model_calls")),
        "wall_time_seconds": wall_time_seconds,
        "analysis_total_model_plus_logical_comm": total_billed_tokens + logical_comm_tokens,
        "per_sample": _per_sample(
            count,
            {
                "model_input_tokens": model_input_tokens,
                "model_output_tokens": model_output_tokens,
                "selector_input_tokens": selector_input_tokens,
                "logical_comm_tokens": logical_comm_tokens,
                "repeated_context_tokens": repeated_context_tokens,
                "total_billed_tokens": total_billed_tokens,
            },
        ),
        "audit": {
            "reported_total_model_delta": total_model_delta,
            "selector_likely_already_in_billed_input": selector_likely_in_billed_input,
            "do_not_sum_selector_into_billed_total": selector_likely_in_billed_input,
            "records_recomputed": recomputed,
            "record_recompute_delta": {
                "model_input_tokens": model_input_tokens - recomputed["model_input_tokens"],
                "model_output_tokens": model_output_tokens - recomputed["model_output_tokens"],
                "selector_input_tokens": selector_input_tokens - recomputed["selector_input_tokens"],
                "logical_comm_tokens": logical_comm_tokens - recomputed["logical_comm_tokens"],
            },
        },
    }
    return normalized


def aggregate_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "count": sum(_int(item.get("count")) for item in audits),
        "model_input_tokens": sum(_int(item.get("model_input_tokens")) for item in audits),
        "model_output_tokens": sum(_int(item.get("model_output_tokens")) for item in audits),
        "selector_input_tokens": sum(_int(item.get("selector_input_tokens")) for item in audits),
        "logical_comm_tokens": sum(_int(item.get("logical_comm_tokens")) for item in audits),
        "repeated_context_tokens": sum(_int(item.get("repeated_context_tokens")) for item in audits),
        "total_billed_tokens": sum(_int(item.get("total_billed_tokens")) for item in audits),
        "num_model_calls": sum(_int(item.get("num_model_calls")) for item in audits),
        "wall_time_seconds": sum(_float(item.get("wall_time_seconds")) for item in audits),
    }
    weighted_correct = sum(_float(item.get("quality")) * _int(item.get("count")) for item in audits)
    totals["quality"] = weighted_correct / totals["count"] if totals["count"] else 0.0
    totals["per_sample"] = _per_sample(totals["count"], totals)
    return totals


def _recompute_from_records(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "model_input_tokens": sum(_record_input_tokens(record) for record in records),
        "model_output_tokens": sum(_record_output_tokens(record) for record in records),
        "selector_input_tokens": sum(
            sum(_int(step.get("selector_input_tokens")) for step in record.get("selector_communication_steps", []))
            for record in records
        ),
        "logical_comm_tokens": sum(
            _first_int(
                record,
                (
                    "logical_comm_tokens",
                    "logical_communication_tokens",
                    "communication_tokens",
                    "communication_cost_tokens",
                ),
            )
            for record in records
        ),
    }


def _record_input_tokens(record: dict[str, Any]) -> int:
    direct = _first_int(record, ("input_tokens", "total_input_tokens", "physical_input_tokens"))
    candidate_sum = sum(_int(candidate.get("input_tokens")) for candidate in record.get("candidates", []))
    step_sum = sum(_int(step.get("input_tokens")) for step in record.get("aflow_step_records", []))
    call_sum = sum(_int(step.get("input_tokens")) for step in record.get("aflow_call_records", []))
    return max(direct, candidate_sum + step_sum + call_sum)


def _record_output_tokens(record: dict[str, Any]) -> int:
    direct = _first_int(record, ("output_tokens", "total_output_tokens"))
    candidate_sum = sum(_int(candidate.get("output_tokens")) for candidate in record.get("candidates", []))
    step_sum = sum(_int(step.get("output_tokens")) for step in record.get("aflow_step_records", []))
    call_sum = sum(_int(step.get("output_tokens")) for step in record.get("aflow_call_records", []))
    return max(direct, candidate_sum + step_sum + call_sum)


def _quality(report: dict[str, Any]) -> float:
    for key in ("accuracy", "accuracy_or_pass_at_1", "pass_at_1", "exact_match", "correct_rate"):
        if key in report:
            return _float(report.get(key))
    count = _int(report.get("count")) or len(report.get("records", []))
    if not count:
        return 0.0
    correct = report.get("correct_count")
    if correct is None:
        correct = sum(int(bool(record.get("correct") or record.get("selected_correct"))) for record in report.get("records", []))
    return _int(correct) / count


def _per_sample(count: int, values: dict[str, Any]) -> dict[str, float]:
    if not count:
        return {key: 0.0 for key in values if key not in {"count", "quality"}}
    return {
        key: _float(value) / count
        for key, value in values.items()
        if key not in {"count", "quality"} and isinstance(value, (int, float))
    }


def _first_int(payload: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        if key in payload:
            value = _int(payload.get(key))
            if value:
                return value
    return 0


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize and audit token-cost accounting for experiment reports.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Report JSON files to audit.")
    parser.add_argument("--output", required=True, help="Output audit JSON path.")
    args = parser.parse_args()

    audits = [audit_report(Path(path)) for path in args.inputs]
    payload = {
        "schema": {
            "model_input_tokens": "Actual input tokens sent to model calls.",
            "model_output_tokens": "Actual output tokens produced by model calls.",
            "selector_input_tokens": "Selector-visible input payload; keep separate unless excluded from model_input_tokens.",
            "logical_comm_tokens": "Agent-to-agent message payload length before prompt reuse/re-rendering.",
            "repeated_context_tokens": "Repeated/reused context injected across calls.",
            "total_billed_tokens": "model_input_tokens + model_output_tokens.",
        },
        "reports": audits,
        "aggregate": aggregate_audits(audits),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
