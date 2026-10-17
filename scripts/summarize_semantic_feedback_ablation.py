"""Summarize semantic feedback recovery metrics from workflow reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    initial_nack = _int(report.get("semantic_initial_nack_count"))
    repaired = _int(report.get("semantic_missing_repaired_count"))
    unresolved = _int(report.get("semantic_nack_unresolved_count"))
    if not unresolved:
        unresolved = _count_events(report, "semantic_nack_unresolved")
    return {
        "path": str(path),
        "domain": report.get("domain", ""),
        "policy": report.get("communication_policy") or report.get("policy") or _policy_from_path(path),
        "count": _int(report.get("count")) or len(report.get("records", [])),
        "quality": _quality(report),
        "model_input_tokens": _int(report.get("total_input_tokens")),
        "model_output_tokens": _int(report.get("total_output_tokens")),
        "total_model_tokens": _int(report.get("total_model_tokens")) or _int(report.get("total_input_tokens")) + _int(report.get("total_output_tokens")),
        "logical_comm_tokens": _int(report.get("communication_cost_tokens")) or _int(report.get("graph_delta_sent_tokens")),
        "graph_delta_sent_tokens": _int(report.get("graph_delta_sent_tokens")),
        "semantic_initial_nack_count": initial_nack,
        "semantic_missing_repaired_count": repaired,
        "semantic_nack_unresolved_count": unresolved,
        "semantic_recovery_rate": repaired / initial_nack if initial_nack else 0.0,
        "feedback_sent_tokens": _int(report.get("feedback_sent_tokens")),
        "feedback_newly_rendered_tokens": _int(report.get("feedback_newly_rendered_tokens")),
        "feedback_tokens_per_repair": _int(report.get("feedback_sent_tokens")) / repaired if repaired else None,
        "communication_round_count": _int(report.get("communication_round_count")),
    }


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(str(item.get("policy", "")), []).append(item)
    return {
        policy: {
            "count": sum(_int(item.get("count")) for item in rows),
            "quality": _weighted(rows, "quality"),
            "total_model_tokens": sum(_int(item.get("total_model_tokens")) for item in rows),
            "logical_comm_tokens": sum(_int(item.get("logical_comm_tokens")) for item in rows),
            "semantic_initial_nack_count": sum(_int(item.get("semantic_initial_nack_count")) for item in rows),
            "semantic_missing_repaired_count": sum(_int(item.get("semantic_missing_repaired_count")) for item in rows),
            "semantic_recovery_rate": (
                sum(_int(item.get("semantic_missing_repaired_count")) for item in rows)
                / sum(_int(item.get("semantic_initial_nack_count")) for item in rows)
                if sum(_int(item.get("semantic_initial_nack_count")) for item in rows)
                else 0.0
            ),
            "feedback_sent_tokens": sum(_int(item.get("feedback_sent_tokens")) for item in rows),
        }
        for policy, rows in sorted(grouped.items())
    }


def _count_events(report: dict[str, Any], key: str) -> int:
    total = 0
    for record in report.get("records", []):
        for action in record.get("runtime_summary", {}).get("records", []):
            for event in action.get("communication", []):
                total += int(bool(event.get(key, False)))
    return total


def _quality(report: dict[str, Any]) -> float:
    for key in ("accuracy_or_pass_at_1", "accuracy", "exact_match", "pass_at_1"):
        if key in report:
            return _float(report.get(key))
    count = _int(report.get("count")) or len(report.get("records", []))
    correct = sum(int(bool(record.get("correct"))) for record in report.get("records", []))
    return correct / count if count else 0.0


def _policy_from_path(path: Path) -> str:
    name = path.stem
    for prefix in ("gsm8k_", "humaneval_", "mbpp_", "mmlu_pro_", "hotpotqa_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.split("_limit", 1)[0]


def _weighted(rows: list[dict[str, Any]], key: str) -> float:
    total = sum(_int(item.get("count")) for item in rows)
    return sum(_float(item.get(key)) * _int(item.get("count")) for item in rows) / total if total else 0.0


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
    parser = argparse.ArgumentParser(description="Summarize semantic feedback ablations.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reports = [summarize_report(Path(path)) for path in args.inputs]
    payload = {"reports": reports, "aggregate_by_policy": aggregate(reports)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate_by_policy"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
