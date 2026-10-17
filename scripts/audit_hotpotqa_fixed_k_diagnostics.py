"""Offline HotpotQA fixed-K candidate diagnostics.

Gold labels are used only for analysis.  The audit separates:

* bridge-pool coverage: whether gold evidence entities appear as candidate bridges;
* chain binding: whether candidate cited facts match gold supporting facts;
* answer derivation: whether candidates derive the correct short answer;
* grounded reasoning availability: whether an ordered candidate is both answer
  correct and fully grounded in the gold supporting facts.
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

from scripts.evaluate_domain_workflow import hotpot_normalize, read  # noqa: E402
from scripts.run_reasoning_upper_bounds import _hotpot_reasoning_type  # noqa: E402
from workflow_runtime.hotpot_topology import REASONING_TYPES  # noqa: E402
from workflow_runtime.hotpot_bridge import bridge_matches_any, normalize_bridge  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    report = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    records = list(report.get("records", []))
    rows = read(Path(args.data_path), "hotpotqa", args.limit or len(records))
    by_id = {str(row.get("sample_id", "")): row for row in rows}
    diagnostics = [
        diagnose_record(record, by_id.get(str(record.get("sample_id", ""))))
        for record in records
    ]
    diagnostics = [item for item in diagnostics if item]
    summary = summarize(diagnostics)
    payload = {
        "eval_json": args.eval_json,
        "data_path": args.data_path,
        "count": len(diagnostics),
        "summary": summary,
        "diagnostics": diagnostics,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "diagnostics"}, ensure_ascii=False, indent=2))


def diagnose_record(record: dict[str, Any], row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    candidates = list(record.get("candidates", []))
    gold_ordered = gold_fact_ids_ordered(row)
    gold_facts = set(gold_ordered)
    gold_entities = gold_bridge_values(row)
    candidates_by_id = []
    reasoning_type = _hotpot_reasoning_type(row)
    bridge_pool = _candidate_bridge_pool(candidates)
    bridge_pool_audit = audit_bridge_pool(bridge_pool, gold_entities)
    for candidate in candidates:
        bridge = str(candidate.get("bridge_entity") or "").strip()
        cited = [str(item) for item in candidate.get("cited_fact_ids", []) if str(item).strip()]
        cited_gold = sorted(set(cited) & gold_facts)
        answer = str(candidate.get("final_value") or candidate.get("artifact") or "").strip()
        evidence_correct = gold_facts.issubset(set(cited)) if gold_facts else False
        answer_correct = hotpot_normalize(answer) == hotpot_normalize(row.get("gold_answer", ""))
        candidates_by_id.append({
            "candidate_id": candidate.get("candidate_id"),
            "bridge_entity": bridge,
            "bridge_matches_gold": bridge_matches_any(bridge, gold_entities),
            "cited_fact_ids": cited,
            "gold_fact_overlap": cited_gold,
            "gold_fact_overlap_count": len(cited_gold),
            "covers_all_gold_facts": evidence_correct,
            "evidence_correct": evidence_correct,
            "answer": answer,
            "answer_correct": answer_correct,
            "grounded_correct": answer_correct and evidence_correct,
            "reasoning_type": candidate.get("reasoning_type") or reasoning_type,
            "allocation_policy": candidate.get("allocation_policy", ""),
            "constructor_family": candidate.get("constructor_family", ""),
            "bridge_entity_telemetry": candidate.get("bridge_entity_telemetry", {}),
            "bridge_attribute_telemetry": candidate.get("bridge_attribute_telemetry", {}),
            "multi_constraint_telemetry": candidate.get("multi_constraint_telemetry", {}),
            "parallel_retrieval_telemetry": candidate.get("parallel_retrieval_telemetry", {}),
            "hop1_pool": candidate.get("hop1_pool", []),
            "hop2_pool": candidate.get("hop2_pool", []),
            "chain_pair_pool": candidate.get("chain_pair_pool", []),
        })
    hop_audit = audit_hop_binding(candidates_by_id, gold_ordered)
    construction_audit = audit_topology_construction(candidates_by_id, gold_ordered, reasoning_type)
    bridge_entity_audit = audit_bridge_entity_construction(candidates_by_id, gold_ordered, reasoning_type)
    bridge_attribute_audit = audit_bridge_attribute_hop2_retrieval(candidates_by_id, gold_ordered, reasoning_type)
    multi_constraint_audit = audit_multi_constraint_target_intersection(candidates_by_id, reasoning_type)
    parallel_audit = audit_parallel_entity_side_retrieval(candidates_by_id, reasoning_type)
    bridge_pool_gold_any = any(item["bridge_matches_gold"] for item in candidates_by_id)
    bridge_pool_gold_count = len({
        normalize_bridge(item["bridge_entity"])
        for item in candidates_by_id
        if item["bridge_matches_gold"]
    })
    answer_evidence_audit = audit_answer_evidence_coverage(candidates_by_id)
    any_gold_fact_bound = any(item["gold_fact_overlap_count"] > 0 for item in candidates_by_id)
    any_full_gold_chain_bound = bool(answer_evidence_audit["evidence_oracle_coverage"])
    any_correct_answer = bool(answer_evidence_audit["answer_oracle_coverage"])
    any_correct_with_gold_fact = any(item["answer_correct"] and item["gold_fact_overlap_count"] > 0 for item in candidates_by_id)
    any_correct_with_full_gold_chain = bool(answer_evidence_audit["grounded_answer_oracle_coverage"])
    best_gold_overlap = max((item["gold_fact_overlap_count"] for item in candidates_by_id), default=0)
    failure_stage = "ok"
    if not construction_audit["gold_support_fact1_available"] and not construction_audit["gold_support_fact2_available"]:
        failure_stage = "both_evidence_missing"
    elif not construction_audit["gold_support_fact1_available"]:
        failure_stage = "evidence_fact1_missing"
    elif not construction_audit["gold_support_fact2_available"]:
        failure_stage = "evidence_fact2_missing"
    elif not construction_audit["gold_evidence_binding_constructed"]:
        failure_stage = _topology_binding_failure(reasoning_type)
    elif not construction_audit["answer_correct_given_gold_evidence"]:
        failure_stage = "evidence_valid_but_answer_wrong"
    elif not any_correct_answer:
        failure_stage = "derived_decision_miss"
    return {
        "sample_id": record.get("sample_id"),
        "question": row.get("question"),
        "reasoning_type": reasoning_type,
        "gold_answer": row.get("gold_answer"),
        "gold_entities": sorted(gold_entities),
        "gold_fact_ids": sorted(gold_facts),
        "gold_fact_ids_ordered": gold_ordered,
        "candidate_count": len(candidates),
        "unique_bridge_count": len({normalize_bridge(item["bridge_entity"]) for item in candidates_by_id if item["bridge_entity"]}),
        **bridge_pool_audit,
        **hop_audit,
        **construction_audit,
        **bridge_entity_audit,
        **bridge_attribute_audit,
        **multi_constraint_audit,
        **parallel_audit,
        **answer_evidence_audit,
        "bridge_pool_gold_any": bridge_pool_gold_any,
        "bridge_pool_gold_count": bridge_pool_gold_count,
        "any_gold_fact_bound": any_gold_fact_bound,
        "any_full_gold_chain_bound": any_full_gold_chain_bound,
        "best_gold_fact_overlap": best_gold_overlap,
        "any_correct_answer": any_correct_answer,
        "any_correct_with_gold_fact": any_correct_with_gold_fact,
        "any_correct_with_full_gold_chain": any_correct_with_full_gold_chain,
        "failure_stage": failure_stage,
        "candidates": candidates_by_id,
    }


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(items)
    def rate(key: str) -> float:
        return sum(int(bool(item.get(key))) for item in items) / count if count else 0.0
    stages: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for item in items:
        stage = str(item.get("failure_stage") or "unknown")
        stages[stage] = stages.get(stage, 0) + 1
        source = str(item.get("bridge_source_type_of_gold") or "")
        if source:
            source_counts[source] = source_counts.get(source, 0) + 1
    topology_summary = summarize_by_topology(items)
    return {
        "gold_in_raw_bridge_pool": rate("gold_in_raw_pool"),
        "gold_in_ranked_top2": rate("gold_in_top2"),
        "gold_in_ranked_top4": rate("gold_in_top4"),
        "gold_in_ranked_top8": rate("gold_in_top8"),
        "bridge_pool_gold_recall": rate("bridge_pool_gold_any"),
        "gold_fact_binding_recall": rate("any_gold_fact_bound"),
        "gold_hop1_in_pool": rate("gold_hop1_in_pool"),
        "gold_hop2_in_pool": rate("gold_hop2_in_pool"),
        "gold_hop1_in_top3": rate("gold_hop1_in_top3"),
        "gold_hop2_in_top3": rate("gold_hop2_in_top3"),
        "gold_hop1_selected": rate("gold_hop1_selected"),
        "gold_hop2_selected": rate("gold_hop2_selected"),
        "gold_hop1_hop2_both_in_pool": rate("gold_hop1_hop2_both_in_pool"),
        "gold_pair_constructed": rate("gold_pair_constructed"),
        "pair_recall_given_hop1_hop2": _conditional_rate(items, "gold_pair_constructed", "gold_hop1_hop2_both_in_pool"),
        "bridge_entity_gold_fact1_recall": rate("bridge_entity_gold_fact1_recall"),
        "bridge_entity_inferred_bridge_correct": rate("bridge_entity_inferred_bridge_correct"),
        "bridge_entity_gold_fact2_recall_given_correct_bridge": _conditional_rate(
            items,
            "bridge_entity_gold_fact2_recall_after_correct_bridge",
            "bridge_entity_inferred_bridge_correct",
        ),
        "bridge_entity_binding_success_given_both_facts": _conditional_rate(
            items,
            "gold_evidence_binding_constructed",
            "both_gold_facts_available",
        ),
        "bridge_attribute_hop2_retrieved_any_about_bridge": rate("bridge_attribute_hop2_retrieved_any_about_bridge"),
        "bridge_attribute_hop2_retrieved_requested_relation": rate("bridge_attribute_hop2_retrieved_requested_relation"),
        "bridge_attribute_hop2_retrieved_gold_fact": rate("bridge_attribute_hop2_retrieved_gold_fact"),
        "multi_constraint_constraint_1_target_retrieved": rate("multi_constraint_constraint_1_target_retrieved"),
        "multi_constraint_constraint_2_target_retrieved": rate("multi_constraint_constraint_2_target_retrieved"),
        "multi_constraint_shared_target_found": rate("multi_constraint_shared_target_found"),
        "parallel_entity_a_retrieved": rate("parallel_entity_a_retrieved"),
        "parallel_entity_b_retrieved": rate("parallel_entity_b_retrieved"),
        "parallel_both_entities_retrieved": rate("parallel_both_entities_retrieved"),
        "parallel_both_relation_compatible": rate("parallel_both_relation_compatible"),
        "parallel_both_values_present": rate("parallel_both_values_present"),
        "full_gold_chain_binding_recall": rate("any_full_gold_chain_bound"),
        "gold_candidate_available": rate("gold_candidate_available"),
        "gold_candidate_in_full_pool": rate("gold_candidate_in_full_pool"),
        "gold_candidate_in_top8": rate("gold_candidate_in_top8"),
        "gold_candidate_in_top4": rate("gold_candidate_in_top4"),
        "gold_candidate_selected": rate("gold_candidate_selected"),
        "gold_candidate_top1": rate("gold_candidate_top1"),
        "gold_candidate_top2": rate("gold_candidate_top2"),
        "gold_candidate_top4": rate("gold_candidate_top4"),
        "gold_candidate_top8": rate("gold_candidate_top8"),
        "R_construct": rate("gold_candidate_in_full_pool"),
        "R_top8": rate("gold_candidate_in_top8"),
        "R_top4": rate("gold_candidate_in_top4"),
        "R_selector": rate("gold_candidate_selected"),
        "P_top8_given_pool": _conditional_rate(items, "gold_candidate_in_top8", "gold_candidate_in_full_pool"),
        "P_top4_given_top8": _conditional_rate(items, "gold_candidate_in_top4", "gold_candidate_in_top8"),
        "P_selected_given_top4": _conditional_rate(items, "gold_candidate_selected", "gold_candidate_in_top4"),
        "answer_oracle_coverage": rate("answer_oracle_coverage"),
        "evidence_oracle_coverage": rate("evidence_oracle_coverage"),
        "grounded_answer_oracle_coverage": rate("grounded_answer_oracle_coverage"),
        "answer_oracle_recall": rate("any_correct_answer"),
        "answer_given_gold_fact_recall": rate("any_correct_with_gold_fact"),
        "answer_given_full_gold_chain_recall": rate("any_correct_with_full_gold_chain"),
        "avg_candidate_count": (
            sum(int(item.get("candidate_count", 0) or 0) for item in items) / count if count else 0.0
        ),
        "avg_unique_bridge_count": (
            sum(int(item.get("unique_bridge_count", 0) or 0) for item in items) / count if count else 0.0
        ),
        "avg_best_gold_fact_overlap": (
            sum(int(item.get("best_gold_fact_overlap", 0) or 0) for item in items) / count if count else 0.0
        ),
        "avg_gold_pair_rank": _avg_present(item.get("gold_pair_rank") for item in items),
        "failure_stage_counts": dict(sorted(stages.items())),
        "topology_summary": topology_summary,
        "bridge_source_type_of_gold_counts": dict(sorted(source_counts.items())),
    }


def audit_hop_binding(candidates: list[dict[str, Any]], gold_ordered: list[str]) -> dict[str, Any]:
    gold_hop1 = gold_ordered[0] if gold_ordered else ""
    gold_hop2 = gold_ordered[1] if len(gold_ordered) > 1 else (gold_ordered[0] if gold_ordered else "")
    gold_set = set(gold_ordered)
    hop1_pool_ids = _pool_ids(candidates, "hop1_pool")
    hop2_pool_ids = _pool_ids(candidates, "hop2_pool")
    hop1_top3_ids = _pool_ids(candidates, "hop1_pool", top_k=3)
    hop2_top3_ids = _pool_ids(candidates, "hop2_pool", top_k=3)
    selected_hop1 = {str(item["cited_fact_ids"][0]) for item in candidates if item.get("cited_fact_ids")}
    selected_hop2 = {str(item["cited_fact_ids"][1]) for item in candidates if len(item.get("cited_fact_ids", [])) > 1}
    pair_ranks = []
    for candidate in candidates:
        for index, pair in enumerate(candidate.get("chain_pair_pool", []), start=1):
            if not isinstance(pair, dict):
                continue
            pair_ids = {str(pair.get("hop1_fact_id")), str(pair.get("hop2_fact_id"))}
            if gold_set and gold_set.issubset(pair_ids):
                pair_ranks.append(index)
    return {
        "gold_hop1_id": gold_hop1,
        "gold_hop2_id": gold_hop2,
        "gold_hop1_in_pool": bool(gold_hop1 and gold_hop1 in hop1_pool_ids),
        "gold_hop2_in_pool": bool(gold_hop2 and gold_hop2 in hop2_pool_ids),
        "gold_hop1_in_top3": bool(gold_hop1 and gold_hop1 in hop1_top3_ids),
        "gold_hop2_in_top3": bool(gold_hop2 and gold_hop2 in hop2_top3_ids),
        "gold_hop1_selected": bool(gold_hop1 and gold_hop1 in selected_hop1),
        "gold_hop2_selected": bool(gold_hop2 and gold_hop2 in selected_hop2),
        "gold_hop1_hop2_both_in_pool": bool(
            gold_hop1 and gold_hop2 and gold_hop1 in hop1_pool_ids and gold_hop2 in hop2_pool_ids
        ),
        "gold_pair_constructed": bool(pair_ranks),
        "gold_pair_rank": min(pair_ranks) if pair_ranks else None,
    }


def audit_topology_construction(candidates: list[dict[str, Any]], gold_ordered: list[str], reasoning_type: str) -> dict[str, Any]:
    gold_fact1 = gold_ordered[0] if gold_ordered else ""
    gold_fact2 = gold_ordered[1] if len(gold_ordered) > 1 else ""
    all_pool_ids = _pool_ids(candidates, "hop1_pool") | _pool_ids(candidates, "hop2_pool")
    hop1_pool_ids = _pool_ids(candidates, "hop1_pool")
    hop2_pool_ids = _pool_ids(candidates, "hop2_pool")
    selected_pairs = [
        [str(item) for item in candidate.get("cited_fact_ids", [])]
        for candidate in candidates
    ]
    pair_pool_pairs = [
        pair_ids
        for candidate in candidates
        for pair_ids in _candidate_pair_pool_ids(candidate)
    ]
    gold_set = {fact for fact in (gold_fact1, gold_fact2) if fact}
    fact1_available = bool(gold_fact1 and gold_fact1 in all_pool_ids)
    fact2_available = bool(gold_fact2 and gold_fact2 in all_pool_ids)
    if not gold_fact2 and gold_fact1:
        fact2_available = fact1_available
    both_available = fact1_available and fact2_available
    role_constructible = False
    if gold_fact1 and gold_fact2:
        role_constructible = (
            gold_fact1 in hop1_pool_ids and gold_fact2 in hop2_pool_ids
        ) or (
            _topology_allows_swapped_facts(reasoning_type)
            and gold_fact2 in hop1_pool_ids and gold_fact1 in hop2_pool_ids
        )
    elif gold_fact1:
        role_constructible = gold_fact1 in all_pool_ids
    pair_constructed = _contains_gold_fact_set(pair_pool_pairs, gold_set) or _contains_gold_fact_set(selected_pairs, gold_set)
    answer_given_gold_evidence = any(
        bool(candidate.get("evidence_correct")) and bool(candidate.get("answer_correct"))
        for candidate in candidates
    )
    return {
        "gold_support_fact1_available": fact1_available,
        "gold_support_fact2_available": fact2_available,
        "both_gold_facts_available": both_available,
        "gold_reasoning_structure_constructible": role_constructible,
        "gold_evidence_binding_constructed": pair_constructed,
        "answer_correct_given_gold_evidence": answer_given_gold_evidence,
    }


def audit_bridge_entity_construction(
    candidates: list[dict[str, Any]],
    gold_ordered: list[str],
    reasoning_type: str,
) -> dict[str, Any]:
    if reasoning_type != "bridge_entity":
        return {
            "bridge_entity_gold_fact1_recall": False,
            "bridge_entity_inferred_bridge_correct": False,
            "bridge_entity_gold_fact2_recall_after_correct_bridge": False,
        }
    gold_fact1 = gold_ordered[0] if gold_ordered else ""
    gold_fact2 = gold_ordered[1] if len(gold_ordered) > 1 else ""
    telemetry_items = [
        item.get("bridge_entity_telemetry", {})
        for item in candidates
        if isinstance(item.get("bridge_entity_telemetry", {}), dict)
    ]
    gold_fact1_used = [
        item for item in telemetry_items
        if gold_fact1 and str(item.get("source_fact_id", "")) == gold_fact1
    ]
    inferred_from_gold_fact1 = [
        item for item in gold_fact1_used
        if str(item.get("inferred_bridge", "")).strip()
    ]
    fact2_after_correct_bridge = any(
        gold_fact2 and gold_fact2 in {str(fid) for fid in item.get("hop2_pool_fact_ids", [])}
        for item in inferred_from_gold_fact1
    )
    return {
        "bridge_entity_gold_fact1_recall": bool(gold_fact1_used),
        "bridge_entity_inferred_bridge_correct": bool(inferred_from_gold_fact1),
        "bridge_entity_gold_fact2_recall_after_correct_bridge": bool(fact2_after_correct_bridge),
    }


def audit_bridge_attribute_hop2_retrieval(
    candidates: list[dict[str, Any]],
    gold_ordered: list[str],
    reasoning_type: str,
) -> dict[str, Any]:
    if reasoning_type != "bridge_attribute":
        return {
            "bridge_attribute_hop2_retrieved_any_about_bridge": False,
            "bridge_attribute_hop2_retrieved_requested_relation": False,
            "bridge_attribute_hop2_retrieved_gold_fact": False,
        }
    gold_fact2 = gold_ordered[1] if len(gold_ordered) > 1 else ""
    telemetry_items = [
        item.get("bridge_attribute_telemetry", {})
        for item in candidates
        if isinstance(item.get("bridge_attribute_telemetry", {}), dict)
    ]
    return {
        "bridge_attribute_hop2_retrieved_any_about_bridge": any(
            bool(item.get("hop2_retrieved_any_about_bridge")) for item in telemetry_items
        ),
        "bridge_attribute_hop2_retrieved_requested_relation": any(
            bool(item.get("hop2_retrieved_requested_relation")) for item in telemetry_items
        ),
        "bridge_attribute_hop2_retrieved_gold_fact": any(
            gold_fact2 and gold_fact2 in {str(fid) for fid in item.get("hop2_pool_fact_ids", [])}
            for item in telemetry_items
        ),
    }


def audit_multi_constraint_target_intersection(
    candidates: list[dict[str, Any]],
    reasoning_type: str,
) -> dict[str, Any]:
    if reasoning_type != "multi_constraint_lookup":
        return {
            "multi_constraint_constraint_1_target_retrieved": False,
            "multi_constraint_constraint_2_target_retrieved": False,
            "multi_constraint_shared_target_found": False,
        }
    telemetry_items = [
        item.get("multi_constraint_telemetry", {})
        for item in candidates
        if isinstance(item.get("multi_constraint_telemetry", {}), dict)
    ]
    return {
        "multi_constraint_constraint_1_target_retrieved": any(
            int((item.get("target_set_sizes") or [0, 0])[0] or 0) > 0 for item in telemetry_items
        ),
        "multi_constraint_constraint_2_target_retrieved": any(
            len(item.get("target_set_sizes") or []) > 1 and int((item.get("target_set_sizes") or [0, 0])[1] or 0) > 0
            for item in telemetry_items
        ),
        "multi_constraint_shared_target_found": any(
            int(item.get("shared_target_count", 0) or 0) > 0 for item in telemetry_items
        ),
    }


def audit_parallel_entity_side_retrieval(
    candidates: list[dict[str, Any]],
    reasoning_type: str,
) -> dict[str, Any]:
    if reasoning_type not in {"comparison_attribute", "shared_property_yesno", "set_intersection"}:
        return {
            "parallel_entity_a_retrieved": False,
            "parallel_entity_b_retrieved": False,
            "parallel_both_entities_retrieved": False,
            "parallel_both_relation_compatible": False,
            "parallel_both_values_present": False,
        }
    telemetry_items = [
        item.get("parallel_retrieval_telemetry", {})
        for item in candidates
        if isinstance(item.get("parallel_retrieval_telemetry", {}), dict)
    ]
    entity_a = any(int(item.get("entity_a_fact_count", 0) or 0) > 0 for item in telemetry_items)
    entity_b = any(int(item.get("entity_b_fact_count", 0) or 0) > 0 for item in telemetry_items)
    relation = any(
        int(item.get("entity_a_relation_compatible_count", 0) or 0) > 0
        and int(item.get("entity_b_relation_compatible_count", 0) or 0) > 0
        for item in telemetry_items
    )
    values = any(
        int(item.get("entity_a_value_count", 0) or 0) > 0
        and int(item.get("entity_b_value_count", 0) or 0) > 0
        for item in telemetry_items
    )
    return {
        "parallel_entity_a_retrieved": entity_a,
        "parallel_entity_b_retrieved": entity_b,
        "parallel_both_entities_retrieved": entity_a and entity_b,
        "parallel_both_relation_compatible": relation,
        "parallel_both_values_present": values,
    }


def summarize_by_topology(items: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for reasoning_type in REASONING_TYPES:
        bucket = [item for item in items if item.get("reasoning_type") == reasoning_type]
        if not bucket:
            result[reasoning_type] = {
                "count": 0,
                "gold_support_fact1_available": 0.0,
                "gold_support_fact2_available": 0.0,
                "both_gold_facts_available": 0.0,
                "gold_reasoning_structure_constructible": 0.0,
                "gold_evidence_binding_constructed": 0.0,
                "answer_correct_given_gold_evidence": 0.0,
                "bridge_entity_gold_fact1_recall": 0.0,
                "bridge_entity_inferred_bridge_correct": 0.0,
                "bridge_entity_gold_fact2_recall_given_correct_bridge": None,
                "bridge_entity_binding_success_given_both_facts": None,
                "bridge_attribute_hop2_retrieved_any_about_bridge": 0.0,
                "bridge_attribute_hop2_retrieved_requested_relation": 0.0,
                "bridge_attribute_hop2_retrieved_gold_fact": 0.0,
                "multi_constraint_constraint_1_target_retrieved": 0.0,
                "multi_constraint_constraint_2_target_retrieved": 0.0,
                "multi_constraint_shared_target_found": 0.0,
                "parallel_entity_a_retrieved": 0.0,
                "parallel_entity_b_retrieved": 0.0,
                "parallel_both_entities_retrieved": 0.0,
                "parallel_both_relation_compatible": 0.0,
                "parallel_both_values_present": 0.0,
                "grounded_candidate": 0.0,
                "failure_stage_counts": {},
            }
            continue
        count = len(bucket)
        def rate(key: str) -> float:
            return sum(int(bool(item.get(key))) for item in bucket) / count if count else 0.0
        failures: dict[str, int] = {}
        for item in bucket:
            stage = str(item.get("failure_stage") or "unknown")
            failures[stage] = failures.get(stage, 0) + 1
        result[reasoning_type] = {
            "count": count,
            "gold_support_fact1_available": rate("gold_support_fact1_available"),
            "gold_support_fact2_available": rate("gold_support_fact2_available"),
            "both_gold_facts_available": rate("both_gold_facts_available"),
            "gold_reasoning_structure_constructible": rate("gold_reasoning_structure_constructible"),
            "gold_evidence_binding_constructed": rate("gold_evidence_binding_constructed"),
            "answer_correct_given_gold_evidence": rate("answer_correct_given_gold_evidence"),
            "bridge_entity_gold_fact1_recall": rate("bridge_entity_gold_fact1_recall"),
            "bridge_entity_inferred_bridge_correct": rate("bridge_entity_inferred_bridge_correct"),
            "bridge_entity_gold_fact2_recall_given_correct_bridge": _conditional_rate(
                bucket,
                "bridge_entity_gold_fact2_recall_after_correct_bridge",
                "bridge_entity_inferred_bridge_correct",
            ),
            "bridge_entity_binding_success_given_both_facts": _conditional_rate(
                bucket,
                "gold_evidence_binding_constructed",
                "both_gold_facts_available",
            ),
            "bridge_attribute_hop2_retrieved_any_about_bridge": rate("bridge_attribute_hop2_retrieved_any_about_bridge"),
            "bridge_attribute_hop2_retrieved_requested_relation": rate("bridge_attribute_hop2_retrieved_requested_relation"),
            "bridge_attribute_hop2_retrieved_gold_fact": rate("bridge_attribute_hop2_retrieved_gold_fact"),
            "multi_constraint_constraint_1_target_retrieved": rate("multi_constraint_constraint_1_target_retrieved"),
            "multi_constraint_constraint_2_target_retrieved": rate("multi_constraint_constraint_2_target_retrieved"),
            "multi_constraint_shared_target_found": rate("multi_constraint_shared_target_found"),
            "parallel_entity_a_retrieved": rate("parallel_entity_a_retrieved"),
            "parallel_entity_b_retrieved": rate("parallel_entity_b_retrieved"),
            "parallel_both_entities_retrieved": rate("parallel_both_entities_retrieved"),
            "parallel_both_relation_compatible": rate("parallel_both_relation_compatible"),
            "parallel_both_values_present": rate("parallel_both_values_present"),
            "grounded_candidate": rate("grounded_answer_oracle_coverage"),
            "failure_stage_counts": dict(sorted(failures.items())),
        }
    return result


def _candidate_pair_pool_ids(candidate: dict[str, Any]) -> list[list[str]]:
    result = []
    pool = candidate.get("chain_pair_pool", [])
    if not isinstance(pool, list):
        return result
    for pair in pool:
        if not isinstance(pair, dict):
            continue
        result.append([str(pair.get("hop1_fact_id", "")), str(pair.get("hop2_fact_id", ""))])
    return result


def _contains_gold_fact_set(pairs: list[list[str]], gold_set: set[str]) -> bool:
    if not gold_set:
        return False
    return any(gold_set.issubset({str(item) for item in pair if str(item)}) for pair in pairs)


def _topology_allows_swapped_facts(reasoning_type: str) -> bool:
    return reasoning_type in {"comparison_attribute", "shared_property_yesno", "set_intersection"}


def _topology_binding_failure(reasoning_type: str) -> str:
    if reasoning_type in {"bridge_entity", "bridge_attribute"}:
        return "relation_binding_failed"
    if reasoning_type == "multi_constraint_lookup":
        return "constraint_binding_failed"
    if reasoning_type in {"comparison_attribute", "shared_property_yesno", "set_intersection"}:
        return "comparison_binding_failed"
    return "topology_binding_failed"


def audit_answer_evidence_coverage(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    grounded_ranks = [
        index for index, candidate in enumerate(candidates, start=1)
        if bool(candidate.get("grounded_correct"))
    ]
    evidence_ranks = [
        index for index, candidate in enumerate(candidates, start=1)
        if bool(candidate.get("evidence_correct"))
    ]
    answer_ranks = [
        index for index, candidate in enumerate(candidates, start=1)
        if bool(candidate.get("answer_correct"))
    ]
    first_grounded = min(grounded_ranks) if grounded_ranks else None
    first_evidence = min(evidence_ranks) if evidence_ranks else None
    first_answer = min(answer_ranks) if answer_ranks else None
    return {
        "gold_candidate_available": bool(first_grounded),
        "gold_candidate_in_full_pool": bool(first_grounded),
        "gold_candidate_in_top8": bool(first_grounded and first_grounded <= 8),
        "gold_candidate_in_top4": bool(first_grounded and first_grounded <= 4),
        "gold_candidate_selected": bool(first_grounded and first_grounded == 1),
        "gold_candidate_rank": first_grounded,
        "gold_candidate_top1": bool(first_grounded and first_grounded <= 1),
        "gold_candidate_top2": bool(first_grounded and first_grounded <= 2),
        "gold_candidate_top4": bool(first_grounded and first_grounded <= 4),
        "gold_candidate_top8": bool(first_grounded and first_grounded <= 8),
        "answer_oracle_coverage": bool(first_answer),
        "answer_oracle_rank": first_answer,
        "evidence_oracle_coverage": bool(first_evidence),
        "evidence_oracle_rank": first_evidence,
        "grounded_answer_oracle_coverage": bool(first_grounded),
        "grounded_answer_oracle_rank": first_grounded,
    }


def _conditional_rate(items: list[dict[str, Any]], numerator_key: str, denominator_key: str) -> float | None:
    denom = [item for item in items if bool(item.get(denominator_key))]
    if not denom:
        return None
    return sum(int(bool(item.get(numerator_key))) for item in denom) / len(denom)


def _pool_ids(candidates: list[dict[str, Any]], key: str, *, top_k: int | None = None) -> set[str]:
    result = set()
    for candidate in candidates:
        pool = candidate.get(key)
        if not isinstance(pool, list):
            continue
        items = pool[:top_k] if top_k is not None else pool
        for item in items:
            if isinstance(item, dict) and str(item.get("fact_id", "")).strip():
                result.add(str(item["fact_id"]))
    return result


def _avg_present(values: Any) -> float | None:
    items = [float(value) for value in values if value is not None]
    return sum(items) / len(items) if items else None


def _candidate_bridge_pool(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for candidate in candidates:
        pool = candidate.get("bridge_candidate_pool")
        if isinstance(pool, list) and pool:
            return [item for item in pool if isinstance(item, dict)]
    return []


def audit_bridge_pool(pool: list[dict[str, Any]], gold_entities: set[str]) -> dict[str, Any]:
    gold_ranks = [
        index for index, item in enumerate(pool, start=1)
        if bridge_matches_any(item.get("name", ""), gold_entities)
    ]
    first_rank = min(gold_ranks) if gold_ranks else None
    first = pool[first_rank - 1] if first_rank else {}
    return {
        "raw_bridge_pool_size": len(pool),
        "gold_in_raw_pool": bool(gold_ranks),
        "gold_rank": first_rank,
        "gold_in_top2": bool(first_rank and first_rank <= 2),
        "gold_in_top4": bool(first_rank and first_rank <= 4),
        "gold_in_top8": bool(first_rank and first_rank <= 8),
        "bridge_source_type_of_gold": first.get("source_type", ""),
        "bridge_connectivity_score": first.get("connectivity_score", 0.0),
        "bridge_question_relevance": first.get("question_relevance", 0.0),
        "bridge_evidence_support": first.get("evidence_support", 0.0),
    }


def gold_bridge_values(row: dict[str, Any]) -> set[str]:
    values = set()
    for link in row.get("evidence_links", []):
        if isinstance(link, dict):
            value = str(link.get("entity", link.get("title", ""))).strip()
            if value:
                values.add(value)
    return values


def gold_fact_ids(row: dict[str, Any]) -> set[str]:
    return set(gold_fact_ids_ordered(row))


def gold_fact_ids_ordered(row: dict[str, Any]) -> list[str]:
    links = [
        (normalize_bridge(link.get("entity", link.get("title", ""))), str(link.get("sent_id")))
        for link in row.get("evidence_links", [])
        if isinstance(link, dict)
    ]
    result = []
    for index, fact in enumerate(row.get("supporting_facts", []), start=1):
        if not isinstance(fact, dict):
            continue
        title = normalize_bridge(fact.get("title", fact.get("entity", "")))
        sent_id = str(fact.get("sent_id"))
        if (title, sent_id) in links:
            result.append(str(fact.get("fact_id") or f"supporting_fact_{index}"))
    return result


if __name__ == "__main__":
    main()
