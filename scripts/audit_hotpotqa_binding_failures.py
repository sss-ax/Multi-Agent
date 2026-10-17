"""Inspect HotpotQA cases where gold facts are available but not bound.

Gold labels are used only for offline diagnostics.  This audit narrows the
failure mode:

    both_gold_facts_available == True
    gold_evidence_binding_constructed == False

For each such sample it records the gold hops, relation endpoints, inferred
bridge from hop1, candidate bridges, and the Stage9 failure code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.evaluate_domain_workflow import read  # noqa: E402
from scripts import run_reasoning_upper_bounds as rub  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage9-audit-json", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    audit = json.loads(Path(args.stage9_audit_json).read_text(encoding="utf-8"))
    diagnostics = list(audit.get("diagnostics", []))
    rows = read(Path(args.data_path), "hotpotqa", args.limit or len(diagnostics))
    rows_by_id = {str(row.get("sample_id", "")): row for row in rows}
    cases = [
        diagnose_binding_failure(item, rows_by_id.get(str(item.get("sample_id", ""))))
        for item in diagnostics
        if item.get("both_gold_facts_available") is True
        and item.get("gold_evidence_binding_constructed") is False
    ]
    cases = [case for case in cases if case]
    summary = summarize(cases)
    payload = {
        "stage9_audit_json": args.stage9_audit_json,
        "data_path": args.data_path,
        "count": len(cases),
        "summary": summary,
        "cases": cases,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "cases"}, ensure_ascii=False, indent=2))


def diagnose_binding_failure(diag: dict[str, Any], row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    facts = rub._hotpot_facts(row)
    fact_by_id = {str(fact.get("fact_id", "")): fact for fact in facts}
    gold_ids = [str(item) for item in diag.get("gold_fact_ids_ordered", []) if str(item)]
    if not gold_ids:
        gold_ids = [str(item) for item in diag.get("gold_fact_ids", []) if str(item)]
    hop1_id = gold_ids[0] if gold_ids else ""
    hop2_id = gold_ids[1] if len(gold_ids) > 1 else ""
    hop1 = fact_by_id.get(hop1_id, {})
    hop2 = fact_by_id.get(hop2_id, {})
    best_candidate = best_candidate_for_gold(diag.get("candidates", []), set(gold_ids))
    candidate_bridges = list(dict.fromkeys(
        str(candidate.get("bridge_entity", "")).strip()
        for candidate in diag.get("candidates", [])
        if str(candidate.get("bridge_entity", "")).strip()
    ))[:8]
    inferred_bridge = rub._hotpot_inferred_intermediate_from_fact(row, hop1) if hop1 else ""
    hop1_relation = relation_view(row, inferred_bridge, hop1, role="hop1") if hop1 else {}
    hop2_relation = relation_view(row, inferred_bridge, hop2, role="hop2") if hop2 else {}
    return {
        "sample_id": diag.get("sample_id"),
        "reasoning_type": diag.get("reasoning_type"),
        "question": diag.get("question"),
        "gold_answer": diag.get("gold_answer"),
        "gold_hop1": fact_payload(hop1_id, hop1),
        "gold_hop2": fact_payload(hop2_id, hop2),
        "candidate_bridge": best_candidate.get("bridge_entity", ""),
        "candidate_bridges_top": candidate_bridges,
        "best_candidate": {
            "candidate_id": best_candidate.get("candidate_id"),
            "bridge_entity": best_candidate.get("bridge_entity", ""),
            "cited_fact_ids": best_candidate.get("cited_fact_ids", []),
            "gold_fact_overlap": best_candidate.get("gold_fact_overlap", []),
            "answer": best_candidate.get("answer", ""),
            "answer_correct": bool(best_candidate.get("answer_correct")),
            "constructor_family": best_candidate.get("constructor_family", ""),
            "allocation_policy": best_candidate.get("allocation_policy", ""),
        },
        "inferred_bridge_from_hop1": inferred_bridge,
        "hop1_relation": hop1_relation,
        "hop2_relation": hop2_relation,
        "direction_diagnostics": {
            "hop1_subject_equals_inferred_bridge": entity_eq(hop1_relation.get("source_entity"), inferred_bridge),
            "hop1_object_equals_inferred_bridge": entity_eq(hop1_relation.get("target_entity"), inferred_bridge),
            "hop2_subject_equals_inferred_bridge": entity_eq(hop2_relation.get("source_entity"), inferred_bridge),
            "hop2_object_equals_inferred_bridge": entity_eq(hop2_relation.get("target_entity"), inferred_bridge),
        },
        "failure_code": diag.get("failure_stage"),
        "stage_flags": {
            "gold_support_fact1_available": bool(diag.get("gold_support_fact1_available")),
            "gold_support_fact2_available": bool(diag.get("gold_support_fact2_available")),
            "gold_reasoning_structure_constructible": bool(diag.get("gold_reasoning_structure_constructible")),
            "gold_pair_constructed": bool(diag.get("gold_pair_constructed")),
            "answer_correct_given_gold_evidence": bool(diag.get("answer_correct_given_gold_evidence")),
        },
    }


def relation_view(row: dict[str, Any], bridge: str, fact: dict[str, Any], *, role: str) -> dict[str, Any]:
    view = rub._hotpot_fact_relation_view(row, bridge, fact, role=role)
    return {
        "subject": str(view.get("source_entity", "")),
        "relation": str(view.get("relation", "")),
        "object": str(view.get("target_entity", "")),
        "object_role": str(view.get("target_role", "")),
        "predicate": str(view.get("predicate", "")),
        "direction": str(view.get("direction", "")),
        "referenced_entities": [str(item) for item in view.get("referenced_entities", [])],
    }


def fact_payload(fact_id: str, fact: dict[str, Any]) -> dict[str, Any]:
    return {
        "fact_id": fact_id,
        "title": str(fact.get("title", "")),
        "sent_id": fact.get("sent_id"),
        "text": str(fact.get("text", "")),
    }


def best_candidate_for_gold(candidates: Any, gold_ids: set[str]) -> dict[str, Any]:
    if not isinstance(candidates, list):
        return {}
    def score(candidate: dict[str, Any]) -> tuple[int, int, int]:
        overlap = len(set(str(item) for item in candidate.get("cited_fact_ids", [])) & gold_ids)
        pool_overlap = 0
        for key in ("hop1_pool", "hop2_pool"):
            for item in candidate.get(key, []) if isinstance(candidate.get(key), list) else []:
                if str(item.get("fact_id", "")) in gold_ids:
                    pool_overlap += 1
        return (overlap, pool_overlap, int(bool(candidate.get("answer_correct"))))
    if not candidates:
        return {}
    return max((c for c in candidates if isinstance(c, dict)), key=score, default={})


def entity_eq(left: Any, right: Any) -> bool:
    return bool(rub.normalize_bridge(left) and rub.normalize_bridge(left) == rub.normalize_bridge(right))


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    by_failure: dict[str, int] = {}
    direction_patterns: dict[str, int] = {}
    for case in cases:
        by_type[str(case.get("reasoning_type", ""))] = by_type.get(str(case.get("reasoning_type", "")), 0) + 1
        by_failure[str(case.get("failure_code", ""))] = by_failure.get(str(case.get("failure_code", "")), 0) + 1
        direction = case.get("direction_diagnostics", {})
        if direction.get("hop2_subject_equals_inferred_bridge"):
            pattern = "hop2_subject_matches_bridge"
        elif direction.get("hop2_object_equals_inferred_bridge"):
            pattern = "hop2_object_matches_bridge_direction_reversed"
        else:
            pattern = "hop2_not_aligned_to_bridge"
        direction_patterns[pattern] = direction_patterns.get(pattern, 0) + 1
    return {
        "by_reasoning_type": dict(sorted(by_type.items())),
        "by_failure_code": dict(sorted(by_failure.items())),
        "direction_patterns": dict(sorted(direction_patterns.items())),
    }


if __name__ == "__main__":
    main()
