#!/usr/bin/env python3
"""Evaluator-only audit of semantic proof evidence for HotpotQA.

Gold supporting facts are read only here and are never passed to GraphStore or
model prompts.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _run(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return next(x for ds in data.get("runs", {}).values() for x in ds.values() if isinstance(x, dict) and "score" in x)


def _id(row: dict, index: int) -> str:
    for key in ("id", "_id", "task_id", "question_id"):
        if key in row:
            return f"{key}:{row[key]}"
    return f"row:{index}"


def _selected_facts(sentence_ids):
    facts = set()
    for node_id in sentence_ids:
        match = re.search(r"sentence_source_(\d+)_(\d+)@v\d+", str(node_id))
        if match:
            facts.add((int(match.group(1)) - 1, int(match.group(2)) - 1))
    return facts


def _all_public_facts(row):
    return {(doc_index, sentence_index)
            for doc_index, item in enumerate(row.get("context", []))
            if isinstance(item, (list, tuple)) and len(item) >= 2
            for sentence_index, _ in enumerate(item[1] if isinstance(item[1], list) else [])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.dataset_path).read_text(encoding="utf-8").splitlines() if line.strip()][: args.limit]
    result = _run(args.result)
    plans = {}
    for summary in result.get("graph_telemetry", []) or []:
        sample_id = summary.get("sample_id")
        plan_list = summary.get("aflow_context_plans", []) or []
        if sample_id and plan_list:
            plans[str(sample_id)] = plan_list[-1]
    records = []
    doc_recalls = []; sent_recalls = []; sent_precisions = []
    for index, row in enumerate(rows):
        sample_id = _id(row, index)
        plan = plans.get(sample_id, {})
        level = str(plan.get("effective_level", ""))
        # Full-context recovery deliberately has no selected node grant.  It
        # is nevertheless complete public evidence for evaluator-only recall.
        selected = _all_public_facts(row) if level in {"S2", "L2"} else _selected_facts(plan.get("selected_node_ids", []))
        gold = set()
        title_to_index = {str(item[0]): doc_index for doc_index, item in enumerate(row.get("context", [])) if isinstance(item, (list, tuple)) and len(item) >= 2}
        for fact in row.get("supporting_facts", []) or []:
            if isinstance(fact, (list, tuple)) and len(fact) >= 2 and str(fact[0]) in title_to_index:
                gold.add((title_to_index[str(fact[0])], int(fact[1])))
        doc_gold = {doc for doc, _ in gold}; doc_selected = {doc for doc, _ in selected}
        doc_recalls.append(len(doc_gold & doc_selected) / max(1, len(doc_gold)))
        sent_recalls.append(len(gold & selected) / max(1, len(gold)))
        sent_precisions.append(len(gold & selected) / max(1, len(selected)))
        records.append({
            "sample_id": sample_id,
            "effective_level": plan.get("effective_level", "unknown"),
            "selected_facts": sorted(selected),
            "gold_facts": sorted(gold),
            "document_recall": doc_recalls[-1],
            "sentence_recall": sent_recalls[-1],
            "sentence_precision": sent_precisions[-1],
        })
    report = {
        "sample_count": len(records),
        "document_recall": sum(doc_recalls) / max(1, len(doc_recalls)),
        "sentence_recall": sum(sent_recalls) / max(1, len(sent_recalls)),
        "sentence_precision": sum(sent_precisions) / max(1, len(sent_precisions)),
        "s0_count": sum(item["effective_level"] == "S0" for item in records),
        "s1_count": sum(item["effective_level"] == "S1" for item in records),
        "s2_count": sum(item["effective_level"] == "S2" for item in records),
        "records": records,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output: Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__": main()
