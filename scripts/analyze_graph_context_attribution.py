#!/usr/bin/env python3
"""Attribute Graph input cost from evaluate_domain_workflow reports.

This script works with existing report JSON files.  Because those reports store
context node ids and total rendered graph tokens, but not each node's rendered
content, per-node/type attribution is an equal-share estimate within each model
call.  It is still useful for identifying repeated whole-graph reads and the
node types that dominate context slices.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SOURCE_TYPES = {
    "task",
    "query_spec",
    "facts",
    "fact",
    "requirements",
    "test",
    "choice",
    "entity",
    "supporting_fact",
    "evidence_link",
    "table",
    "table_cell",
    "evidence",
}
HISTORY_TYPES = {"error", "tool_request", "tool_result", "execution"}
DERIVED_TYPES = {"plan", "plan_step", "plan_steps", "calculation", "result", "verification", "final_answer", "code"}


def node_type(node_id: str) -> str:
    logical = str(node_id).split("@", 1)[0]
    if logical.startswith("fact_"):
        return "fact"
    if logical.startswith("plan_step_"):
        return "plan_step"
    if logical.startswith("supporting_fact_"):
        return "supporting_fact"
    if logical.startswith("evidence_link_"):
        return "evidence_link"
    if logical.startswith("entity_"):
        return "entity"
    if logical.startswith("choice_"):
        return "choice"
    if logical.startswith("test_"):
        return "test"
    if logical.startswith("table_cell_"):
        return "table_cell"
    if logical.startswith("evidence_"):
        return "evidence"
    if logical.startswith("tool_request_"):
        return "tool_request"
    if logical.startswith("tool_result_"):
        return "tool_result"
    return logical


def category_for_type(kind: str) -> str:
    if kind in SOURCE_TYPES:
        return "source_payload"
    if kind in HISTORY_TYPES:
        return "history_or_tool"
    if kind in DERIVED_TYPES:
        return "derived_reasoning"
    return "other"


def read_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_logs(report: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for record in report.get("records", []) if isinstance(report.get("records"), list) else []:
        sample_id = str(record.get("sample_id", ""))
        runtime = record.get("runtime_summary")
        logs = runtime.get("records", []) if isinstance(runtime, dict) else []
        for log in logs if isinstance(logs, list) else []:
            if isinstance(log, dict):
                yield sample_id, log


def analyze_report(path: Path) -> dict[str, Any]:
    report = read_report(path)
    dataset = str(report.get("domain") or path.stem)
    total_graph_tokens = 0.0
    calls = 0
    node_instances = 0
    repeated_node_instances = 0
    repeated_token_estimate = 0.0
    source_repeated_token_estimate = 0.0
    type_tokens: Counter[str] = Counter()
    category_tokens: Counter[str] = Counter()
    role_tokens: Counter[str] = Counter()
    role_node_counts: Counter[str] = Counter()
    max_context = {"tokens": 0, "sample_id": "", "role": "", "node_count": 0}
    seen_by_sample: dict[str, set[str]] = defaultdict(set)
    seen_by_sample_role: dict[tuple[str, str], set[str]] = defaultdict(set)
    repeat_by_type: Counter[str] = Counter()

    for sample_id, log in iter_logs(report):
        nodes = [str(item) for item in log.get("rendered_context_node_ids", []) if str(item)]
        graph_tokens = int(log.get("context_tokens") or 0)
        if graph_tokens <= 0:
            telemetry = log.get("telemetry")
            if isinstance(telemetry, dict):
                graph_tokens = int(telemetry.get("graph_read_context_tokens") or 0)
        if graph_tokens <= 0 or not nodes:
            continue
        calls += 1
        total_graph_tokens += graph_tokens
        role = str(log.get("role") or "unknown")
        role_tokens[role] += graph_tokens
        role_node_counts[role] += len(nodes)
        node_instances += len(nodes)
        per_node = graph_tokens / len(nodes)
        if graph_tokens > max_context["tokens"]:
            max_context = {
                "tokens": graph_tokens,
                "sample_id": sample_id,
                "role": role,
                "node_count": len(nodes),
            }
        sample_seen = seen_by_sample[sample_id]
        role_seen = seen_by_sample_role[(sample_id, role)]
        for node_id in nodes:
            kind = node_type(node_id)
            type_tokens[kind] += per_node
            category_tokens[category_for_type(kind)] += per_node
            if node_id in sample_seen:
                repeated_node_instances += 1
                repeated_token_estimate += per_node
                repeat_by_type[kind] += per_node
                if category_for_type(kind) == "source_payload":
                    source_repeated_token_estimate += per_node
            sample_seen.add(node_id)
            role_seen.add(node_id)

    top_types = [
        {"node_type": kind, "estimated_tokens": round(value), "share": value / total_graph_tokens if total_graph_tokens else 0.0}
        for kind, value in type_tokens.most_common(12)
    ]
    return {
        "dataset": dataset,
        "path": str(path),
        "calls": calls,
        "graph_tokens": round(total_graph_tokens),
        "graph_tokens_per_sample": (
            total_graph_tokens / int(report.get("count") or 1)
            if int(report.get("count") or 0) else 0.0
        ),
        "node_instances": node_instances,
        "avg_nodes_per_call": node_instances / calls if calls else 0.0,
        "unique_node_instances_repeated": repeated_node_instances,
        "repeated_read_token_estimate": round(repeated_token_estimate),
        "repeated_read_share_estimate": repeated_token_estimate / total_graph_tokens if total_graph_tokens else 0.0,
        "source_repeated_token_estimate": round(source_repeated_token_estimate),
        "category_tokens": {
            key: {
                "estimated_tokens": round(value),
                "share": value / total_graph_tokens if total_graph_tokens else 0.0,
            }
            for key, value in category_tokens.most_common()
        },
        "role_tokens": {
            key: {
                "tokens": int(value),
                "share": value / total_graph_tokens if total_graph_tokens else 0.0,
                "avg_nodes_per_call": role_node_counts[key] / max(1, sum(1 for _, log in iter_logs(report) if str(log.get("role") or "unknown") == key)),
            }
            for key, value in role_tokens.most_common()
        },
        "top_node_types": top_types,
        "top_repeated_types": [
            {"node_type": kind, "estimated_tokens": round(value), "share": value / repeated_token_estimate if repeated_token_estimate else 0.0}
            for kind, value in repeat_by_type.most_common(12)
        ],
        "max_context_call": max_context,
    }


def print_summary(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(f"\n{row['dataset']}")
        print("-" * len(row["dataset"]))
        print(f"graph_tokens: {row['graph_tokens']}  per_sample: {row['graph_tokens_per_sample']:.1f}")
        print(f"calls: {row['calls']}  avg_nodes_per_call: {row['avg_nodes_per_call']:.1f}")
        print(
            "repeated_read_estimate: "
            f"{row['repeated_read_token_estimate']} ({row['repeated_read_share_estimate']:.1%})"
        )
        print("category_tokens:")
        for key, value in row["category_tokens"].items():
            print(f"  {key}: {value['estimated_tokens']} ({value['share']:.1%})")
        print("top_node_types:")
        for item in row["top_node_types"][:8]:
            print(f"  {item['node_type']}: {item['estimated_tokens']} ({item['share']:.1%})")
        print(
            "max_context_call: "
            f"{row['max_context_call']['tokens']} tokens, "
            f"{row['max_context_call']['node_count']} nodes, "
            f"{row['max_context_call']['role']}, {row['max_context_call']['sample_id']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    rows = [analyze_report(path) for path in args.reports]
    print_summary(rows)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"reports": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
