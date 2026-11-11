#!/usr/bin/env python3
"""Offline audit of HotpotQA L0/L1/L2 recovery decisions (no model calls)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.aflow_graph_adapter import AFlowGraphSession


def _load_result(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    dataset_runs = next(iter(data["runs"].values()))
    return next(iter(dataset_runs.values()))


def _original_input(row: dict) -> str:
    context = "\n".join(" ".join(item[1]) for item in row.get("context", []) if isinstance(item[1], list))
    return f"Context: {context}\n\nQuestion: {row.get('question', '')}\n\nAnswer:"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--compressed", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.dataset_path).read_text(encoding="utf-8").splitlines() if line.strip()][: args.limit]
    baseline = _load_result(args.baseline)
    compressed = _load_result(args.compressed)
    baseline_scores = {str(i): float(row[4]) for i, row in zip(baseline.get("record_ids", []), baseline.get("records", []))}
    compressed_scores = {str(i): float(row[4]) for i, row in zip(compressed.get("record_ids", []), compressed.get("records", []))}
    output = []
    for index, row in enumerate(rows):
        if "id" in row:
            sample_id = f"id:{row['id']}"
        elif "_id" in row:
            sample_id = f"_id:{row['_id']}"
        elif "task_id" in row:
            sample_id = f"task_id:{row['task_id']}"
        elif "question_id" in row:
            sample_id = f"question_id:{row['question_id']}"
        else:
            sample_id = f"row:{index}"
        session = AFlowGraphSession(
            task_id=sample_id,
            task=str(row.get("question", "")),
            dataset="HotpotQA",
            workflow_round=3,
            profile_explicit=True,
            context_mode="graph_minctx",
            context_policy="dependency_minimal",
            task_metadata={"question": row.get("question", ""), "context": row.get("context", [])},
        )
        original_tokens = session._count_tokens(_original_input(row))
        attempts = []
        for level, top_k in (("L0", 4), ("L1", 6)):
            selection = session._hotpot_select_source_nodes("dependency_minimal", top_k)
            rendered = session.render_hotpot_context(selection.selected_node_ids)
            risk = session._hotpot_context_risk(
                selection,
                original_tokens=original_tokens,
                selected_tokens=session._count_tokens(rendered),
                level=level,
            )
            attempts.append({
                "level": level,
                "top_k_documents": top_k,
                "selected_document_ids": list(selection.selected_document_ids),
                "selected_sentence_ids": list(selection.selected_sentence_ids),
                "input_tokens": session._count_tokens(rendered),
                "critical_reasons": list(risk.critical_reasons),
                "soft_reasons": list(risk.soft_reasons),
                "risk_score": risk.score,
                "bridge_paths": list(selection.bridge_paths),
                "accepted": not risk.critical_reasons and len(risk.soft_reasons) < (2 if level == "L0" else 3),
            })
            if attempts[-1]["accepted"]:
                break
        effective_level = attempts[-1]["level"] if attempts[-1]["accepted"] else "L2"
        output.append({
            "sample_id": sample_id,
            "baseline_score": baseline_scores.get(sample_id),
            "compressed_score": compressed_scores.get(sample_id),
            "correct_to_wrong": bool((baseline_scores.get(sample_id) or 0) > 0 and (compressed_scores.get(sample_id) or 0) == 0),
            "attempts": attempts,
            "effective_level": effective_level,
            "original_input_tokens": original_tokens,
        })
    report = {"sample_count": len(output), "samples": output}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
