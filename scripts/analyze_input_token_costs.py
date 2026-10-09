#!/usr/bin/env python3
"""Analyze per-agent input token cost distribution from workflow telemetry."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


CATEGORIES = ("system", "task", "graph", "history", "feedback", "protocol")


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


@dataclass
class TokenStats:
    calls: int = 0
    totals: Counter[str] = field(default_factory=Counter)
    physical_input_tokens: int = 0
    logical_input_tokens: int = 0
    unassigned_tokens: int = 0

    def add(self, split: Mapping[str, int], *, physical: int, logical: int) -> None:
        self.calls += 1
        self.physical_input_tokens += physical
        self.logical_input_tokens += logical
        for key in CATEGORIES:
            self.totals[key] += _as_int(split.get(key))
        basis = physical or logical
        self.unassigned_tokens += max(0, basis - sum(_as_int(split.get(key)) for key in CATEGORIES))

    def as_dict(self) -> dict[str, Any]:
        basis = self.physical_input_tokens or self.logical_input_tokens
        return {
            "calls": self.calls,
            "physical_input_tokens": self.physical_input_tokens,
            "logical_input_tokens": self.logical_input_tokens,
            "unassigned_tokens": self.unassigned_tokens,
            "categories": {
                key: {
                    "tokens": int(self.totals[key]),
                    "share": (self.totals[key] / basis if basis else 0.0),
                }
                for key in CATEGORIES
            },
        }


def iter_records(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        if not path.exists():
            raise SystemExit(
                f"{path}: file not found. Run the evaluation command successfully first, "
                "or pass an existing workflow.jsonl/report JSON file."
            )
        if path.suffix == ".json":
            yield from iter_json_report_records(path)
            continue
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_number}: invalid JSONL record: {exc}") from exc
                if isinstance(record, dict):
                    record.setdefault("_source_path", str(path))
                    record.setdefault("_source_line", line_number)
                    yield record


def iter_json_report_records(path: Path) -> Iterable[dict[str, Any]]:
    """Yield telemetry-like records from evaluate_domain_workflow JSON reports."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for index, item in enumerate(data):
            if isinstance(item, dict):
                item.setdefault("_source_path", str(path))
                item.setdefault("_source_line", index + 1)
                yield item
        return
    if not isinstance(data, dict):
        return
    if str(data.get("event") or ""):
        data.setdefault("_source_path", str(path))
        yield data
        return
    for record_index, record in enumerate(data.get("records", []) if isinstance(data.get("records"), list) else []):
        if not isinstance(record, dict):
            continue
        runtime_summary = record.get("runtime_summary")
        logs = runtime_summary.get("records", []) if isinstance(runtime_summary, dict) else []
        for log_index, log in enumerate(logs if isinstance(logs, list) else []):
            if not isinstance(log, dict):
                continue
            telemetry = log.get("telemetry")
            if isinstance(telemetry, dict) and telemetry:
                event = {
                    **telemetry,
                    "event": "model_action",
                    "role": telemetry.get("role", log.get("role")),
                    "mode": telemetry.get("mode", log.get("mode", "normal")),
                    "action_index": telemetry.get("action_index", log.get("action_index", log_index)),
                    "rendered_context_node_ids": log.get("rendered_context_node_ids", []),
                    "rendered_context_fragment_ids": log.get("rendered_context_fragment_ids", []),
                    "_source_path": str(path),
                    "_source_line": f"records[{record_index}].runtime_summary.records[{log_index}].telemetry",
                }
                yield event
            for comm_index, communication in enumerate(
                log.get("communication", []) if isinstance(log.get("communication"), list) else []
            ):
                if isinstance(communication, dict):
                    yield {
                        **communication,
                        "event": "graph_delta_communication",
                        "_source_path": str(path),
                        "_source_line": (
                            f"records[{record_index}].runtime_summary.records[{log_index}]"
                            f".communication[{comm_index}]"
                        ),
                    }


def input_split(record: Mapping[str, Any]) -> dict[str, int]:
    """Return T_system/T_task/T_graph/T_history/T_feedback/T_protocol.

    Explicit category fields win.  Current optimized workflow telemetry does
    not yet emits all six named fields, so the fallback mapping is:
    persistent context -> task, incremental context -> graph, prompt wrapper
    -> protocol, system prompt -> system.  This keeps the script useful before
    deeper prompt instrumentation lands.
    """

    explicit_field_present = any(f"{key}_input_tokens" in record for key in CATEGORIES)
    explicit = {
        "system": _as_int(record.get("system_input_tokens")),
        "task": _as_int(record.get("task_input_tokens")),
        "graph": _as_int(record.get("graph_input_tokens")),
        "history": _as_int(record.get("history_input_tokens")),
        "feedback": _as_int(record.get("feedback_input_tokens")),
        "protocol": _as_int(record.get("protocol_input_tokens")),
    }
    if explicit_field_present:
        if explicit["system"] == 0 and "system_input_tokens" not in record:
            explicit["system"] = _as_int(record.get("system_prompt_tokens"))
        if explicit["task"] == 0 and "task_input_tokens" not in record:
            explicit["task"] = _as_int(record.get("task_context_tokens"))
        if explicit["graph"] == 0 and "graph_input_tokens" not in record:
            explicit["graph"] = _as_int(record.get("graph_read_context_tokens"))
        if explicit["protocol"] == 0 and "protocol_input_tokens" not in record:
            explicit["protocol"] = _as_int(record.get("prompt_wrapper_tokens"))
        return explicit

    graph_total = _as_int(record.get("graph_read_context_tokens", record.get("context_tokens")))
    task_tokens = _as_int(record.get("persistent_context_tokens"))
    graph_tokens = _as_int(record.get("incremental_context_tokens"))
    if task_tokens == 0 and graph_tokens == 0:
        graph_tokens = graph_total
    elif graph_total:
        graph_tokens = min(graph_tokens, graph_total)
        task_tokens = min(task_tokens, max(0, graph_total - graph_tokens))

    return {
        "system": _as_int(record.get("system_prompt_tokens")),
        "task": task_tokens,
        "graph": graph_tokens,
        "history": _as_int(record.get("history_context_tokens")),
        "feedback": _as_int(record.get("feedback_context_tokens")),
        "protocol": _as_int(record.get("prompt_wrapper_tokens")),
    }


def call_key(record: Mapping[str, Any], index: int) -> tuple[str, int, str]:
    role = str(record.get("role") or record.get("agent_id") or "unknown")
    action_index = _as_int(record.get("action_index", index))
    mode = str(record.get("mode") or "normal")
    return role, action_index, mode


def analyze(paths: Iterable[Path]) -> dict[str, Any]:
    overall = TokenStats()
    by_agent: dict[str, TokenStats] = defaultdict(TokenStats)
    by_round: dict[str, TokenStats] = defaultdict(TokenStats)
    calls: list[dict[str, Any]] = []
    seen_nodes_by_agent: dict[str, set[str]] = defaultdict(set)
    repeated_context_nodes: Counter[str] = Counter()
    graph_duplicate_events: list[dict[str, Any]] = []
    comm_totals = Counter()

    action_index = 0
    for record in iter_records(paths):
        event = str(record.get("event") or "")
        if event in {"model_action", "model_generation_error", "native_model_call"}:
            split = input_split(record)
            role, round_index, mode = call_key(record, action_index)
            physical = _as_int(record.get("physical_input_tokens"))
            if physical == 0:
                metrics = record.get("call_metrics")
                if isinstance(metrics, dict):
                    physical = _as_int(metrics.get("physical_input_tokens"))
            logical = _as_int(record.get("logical_input_tokens")) or sum(split.values()) or physical
            overall.add(split, physical=physical, logical=logical)
            by_agent[role].add(split, physical=physical, logical=logical)
            by_round[f"{role}#{round_index}:{mode}"].add(split, physical=physical, logical=logical)

            node_ids = _as_list(record.get("rendered_context_node_ids")) + _as_list(
                record.get("persistent_context_node_ids")
            )
            repeated = [node_id for node_id in dict.fromkeys(node_ids) if node_id in seen_nodes_by_agent[role]]
            for node_id in repeated:
                repeated_context_nodes[f"{role}:{node_id}"] += 1
            seen_nodes_by_agent[role].update(node_ids)

            calls.append({
                "role": role,
                "round": round_index,
                "mode": mode,
                "physical_input_tokens": physical,
                "logical_input_tokens": logical,
                **{f"t_{key}": split[key] for key in CATEGORIES},
                "unassigned_tokens": max(0, (physical or logical) - sum(split.values())),
                "repeated_context_node_count": len(repeated),
            })
            action_index += 1
            continue

        if event == "graph_delta_communication":
            repeated_tokens = _as_int(record.get("repeated_comm_tokens"))
            receiver_seen_hits = _as_int(record.get("receiver_seen_hit_count"))
            sent_nodes = set(_as_list(record.get("sent_node_ids")))
            visible_before = set(_as_list(record.get("receiver_visible_before_node_ids")))
            repeated_sent_nodes = sorted(sent_nodes & visible_before)
            comm_totals["events"] += 1
            comm_totals["sent_tokens"] += _as_int(record.get("sent_tokens"))
            comm_totals["repeated_comm_tokens"] += repeated_tokens
            comm_totals["receiver_seen_hit_count"] += receiver_seen_hits
            comm_totals["repeated_sent_node_count"] += len(repeated_sent_nodes)
            if repeated_tokens or receiver_seen_hits or repeated_sent_nodes:
                graph_duplicate_events.append({
                    "sender": record.get("sender"),
                    "receiver": record.get("receiver"),
                    "policy": record.get("policy"),
                    "sent_tokens": _as_int(record.get("sent_tokens")),
                    "repeated_comm_tokens": repeated_tokens,
                    "receiver_seen_hit_count": receiver_seen_hits,
                    "repeated_sent_node_ids": repeated_sent_nodes,
                })

    category_totals = overall.totals
    dominant = category_totals.most_common(1)[0][0] if category_totals else None
    recommendation = recommendation_for(dominant, overall.as_dict(), comm_totals)
    return {
        "overall": overall.as_dict(),
        "by_agent": {key: value.as_dict() for key, value in sorted(by_agent.items())},
        "by_round": {key: value.as_dict() for key, value in sorted(by_round.items())},
        "calls": calls,
        "graph_duplicates": {
            "communication": dict(comm_totals),
            "events": graph_duplicate_events,
            "repeated_context_nodes": dict(repeated_context_nodes),
        },
        "dominant_category": dominant,
        "recommendation": recommendation,
    }


def recommendation_for(dominant: str | None, summary: Mapping[str, Any], comm_totals: Mapping[str, int]) -> str:
    repeated_tokens = _as_int(comm_totals.get("repeated_comm_tokens"))
    sent_tokens = _as_int(comm_totals.get("sent_tokens"))
    if sent_tokens and repeated_tokens / sent_tokens >= 0.20:
        return "大量通信 token 是重复可见内容，优先检查 Graph 节点重复注入，并考虑 Adaptive Compute Gate。"
    if dominant in {"graph", "task", "history", "feedback"}:
        return "GraphStore/History/Feedback 类上下文占比最高，优先做 Receiver-Aware Subgraph Retrieval。"
    if dominant in {"system", "protocol"}:
        return "System/Protocol 占比最高，优先做 Compact Prompt/Action Protocol。"
    return "没有明显主导项；先扩大样本或补充更细粒度 prompt 字段后再决定优化方向。"


def write_csv(path: Path, calls: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "role",
        "round",
        "mode",
        "physical_input_tokens",
        "logical_input_tokens",
        *[f"t_{key}" for key in CATEGORIES],
        "unassigned_tokens",
        "repeated_context_node_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(calls)


def print_summary(report: Mapping[str, Any]) -> None:
    overall = report["overall"]
    print("Input token cost distribution")
    print("=============================")
    print(f"calls: {overall['calls']}")
    print(f"physical_input_tokens: {overall['physical_input_tokens']}")
    print(f"logical_input_tokens: {overall['logical_input_tokens']}")
    print("")
    for key in CATEGORIES:
        item = overall["categories"][key]
        print(f"{key:>8}: {item['tokens']:>8} ({item['share']:.1%})")
    print("")
    print(f"dominant_category: {report['dominant_category']}")
    print(f"recommendation: {report['recommendation']}")
    dup = report["graph_duplicates"]["communication"]
    if dup:
        print("")
        print("Graph duplicate injection checks")
        print("--------------------------------")
        print(f"communication_events: {dup.get('events', 0)}")
        print(f"repeated_comm_tokens: {dup.get('repeated_comm_tokens', 0)}")
        print(f"receiver_seen_hit_count: {dup.get('receiver_seen_hit_count', 0)}")
        print(f"repeated_sent_node_count: {dup.get('repeated_sent_node_count', 0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", nargs="+", type=Path, help="Workflow telemetry JSONL file(s).")
    parser.add_argument("--json-out", type=Path, help="Write full JSON report.")
    parser.add_argument("--csv-out", type=Path, help="Write per-call CSV table.")
    args = parser.parse_args()

    report = analyze(args.telemetry)
    print_summary(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.csv_out:
        write_csv(args.csv_out, report["calls"])


if __name__ == "__main__":
    main()
