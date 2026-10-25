"""Offline oracle-injection audit for HotpotQA candidate bottlenecks.

This is a diagnostic only.  Gold evidence/bridge information is never used by
the normal evaluator; the report estimates whether the current verifier could
solve a sample if a missing retrieval stage were repaired.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from scripts.evaluate_domain_workflow import hotpot_normalize, read
from scripts.run_reasoning_upper_bounds import (
    _hotpot_entities_in_text,
    _hotpot_fact_semantics,
    _hotpot_facts,
    _hotpot_topology_classification,
    _hotpot_verify_semantic_chain,
)


def _gold_fact_ids(row: dict[str, Any]) -> list[str]:
    links = row.get("evidence_links", [])
    wanted = {(str(item.get("entity", "")), int(item.get("sent_id", -1))) for item in links if isinstance(item, dict)}
    return [str(fact.get("fact_id", "")) for fact in _hotpot_facts(row) if (str(fact.get("title", "")), int(fact.get("sent_id", -1))) in wanted]


def _bridge_candidates(row: dict[str, Any], facts: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for fact in facts:
        values.extend([str(fact.get("title", "")), *_hotpot_entities_in_text(str(fact.get("text", "")))])
    result: list[str] = []
    for value in values:
        value = value.strip()
        if value and value not in result:
            result.append(value)
    return result


def _gold_path_verification(row: dict[str, Any], gold_ids: list[str]) -> dict[str, Any]:
    facts = {str(fact.get("fact_id", "")): fact for fact in _hotpot_facts(row)}
    selected = [facts[fact_id] for fact_id in gold_ids if fact_id in facts]
    if len(selected) < 2:
        return {"gold_fact_count": len(selected), "gold_bridge_pass": False, "gold_path_pass": False, "gold_answer_supported": False}
    classification = _hotpot_topology_classification(row)
    reasoning_types = [str(item.get("topology")) for item in classification.get("topology_distribution", []) if item.get("topology")]
    if not reasoning_types:
        reasoning_types = [str(classification.get("reasoning_type", "bridge_entity"))]
    best = None
    for left in selected:
        for right in selected:
            if left is right:
                continue
            for bridge in _bridge_candidates(row, selected):
                for reasoning_type in reasoning_types:
                    verdict = _hotpot_verify_semantic_chain(row, bridge, left, right, reasoning_type_override=reasoning_type)
                    gates = verdict.get("programmatic_gates", {})
                    details = verdict.get("type_specific_details", {}) or {}
                    candidate = {
                        "bridge": bridge,
                        "pass": bool(gates.get("pass")),
                        "evidence_valid": bool(verdict.get("evidence_valid")),
                        "answer_supported": bool(verdict.get("answer_supported")),
                        "semantic_answer_supported": bool(details.get("answer_supported")),
                        "semantic_chain_complete": bool(details.get("chain_complete")),
                        "score": float(verdict.get("score", 0.0) or 0.0),
                    }
                    if best is None or (candidate["pass"], candidate["answer_supported"], candidate["score"]) > (best["pass"], best["answer_supported"], best["score"]):
                        best = candidate
    best = best or {"pass": False, "evidence_valid": False, "answer_supported": False, "semantic_answer_supported": False, "semantic_chain_complete": False, "score": 0.0, "bridge": ""}
    return {
        "gold_fact_count": len(selected),
        "gold_bridge_pass": bool(best.get("evidence_valid")),
        "gold_path_pass": bool(best.get("pass")),
        "gold_answer_supported": bool(best.get("answer_supported")),
        "gold_semantic_answer_supported": bool(best.get("semantic_answer_supported")),
        "gold_semantic_chain_complete": bool(best.get("semantic_chain_complete")),
        "oracle_bridge": best.get("bridge", ""),
        "oracle_score": best.get("score", 0.0),
    }


def audit(eval_json: Path, data_path: Path, limit: int) -> dict[str, Any]:
    evaluation = json.loads(eval_json.read_text(encoding="utf-8"))
    rows = read(data_path, "hotpotqa", limit)
    records = {str(record.get("sample_id", "")): record for record in evaluation.get("records", [])}
    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        record = records.get(sample_id, {})
        gold_ids = _gold_fact_ids(row)
        candidates = record.get("candidates", []) or []
        observed = []
        for candidate in candidates:
            pool_ids = {
                str(item.get("fact_id", ""))
                for key in ("hop1_pool", "hop2_pool")
                for item in (candidate.get(key, []) or []) if isinstance(item, dict)
            }
            observed.append({"correct": bool(candidate.get("correct")), "gold_facts_in_pool": set(gold_ids) <= pool_ids})
        path = _gold_path_verification(row, gold_ids)
        diagnostics.append({
            "sample_id": sample_id,
            "question": row.get("question", ""),
            "gold_fact_ids": gold_ids,
            "baseline_correct": any(item["correct"] for item in observed),
            "baseline_gold_facts_in_pool": any(item["gold_facts_in_pool"] for item in observed),
            **path,
        })
    count = len(diagnostics) or 1
    return {
        "experiment": "hotpotqa_oracle_injection_audit",
        "eval_json": str(eval_json),
        "data_path": str(data_path),
        "count": len(diagnostics),
        "summary": {
            "baseline_accuracy_or_pass_at_n": sum(item["baseline_correct"] for item in diagnostics) / count,
            "baseline_full_gold_facts_in_pool": sum(item["baseline_gold_facts_in_pool"] for item in diagnostics) / count,
            "gold_bridge_oracle": sum(item["gold_bridge_pass"] for item in diagnostics) / count,
            "gold_path_oracle": sum(item["gold_path_pass"] for item in diagnostics) / count,
            "gold_answer_supported_oracle": sum(item["gold_answer_supported"] for item in diagnostics) / count,
            "gold_semantic_answer_supported_oracle": sum(item["gold_semantic_answer_supported"] for item in diagnostics) / count,
            "gold_semantic_chain_complete_oracle": sum(item["gold_semantic_chain_complete"] for item in diagnostics) / count,
        },
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit(Path(args.eval_json), Path(args.data_path), args.limit)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
