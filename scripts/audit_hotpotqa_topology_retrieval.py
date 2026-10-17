"""Stage 9A HotpotQA topology-specific evidence retrieval audit.

Gold labels are used only offline.  This script bypasses candidate construction
and asks a narrower question:

    Did each gold supporting fact enter the topology-specific retrieved pool?

The retrieval runs over the provided HotpotQA distractor context only.  It uses
lightweight indexes (title/entity/relation/entity+relation) plus the runtime
topology retrievers, and reports fact1/fact2/both recall@k by topology.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.audit_hotpotqa_fixed_k_diagnostics import gold_fact_ids_ordered  # noqa: E402
from scripts.evaluate_domain_workflow import read  # noqa: E402
from scripts import run_reasoning_upper_bounds as rub  # noqa: E402
from workflow_runtime.hotpot_bridge import normalize_bridge  # noqa: E402
from workflow_runtime.hotpot_topology import REASONING_TYPES  # noqa: E402


KS = (1, 3, 5, 10)


@dataclass
class RetrievedFact:
    fact: dict[str, Any]
    score: float
    retrieval_source: str
    retrieval_query_type: str
    anchor: str = ""
    relation_hint: str = ""
    sources: list[str] = field(default_factory=list)

    @property
    def fact_id(self) -> str:
        return str(self.fact.get("fact_id", ""))

    def as_dict(self, rank: int) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "title": str(self.fact.get("title", "")),
            "retrieval_source": self.retrieval_source,
            "retrieval_sources": self.sources or [self.retrieval_source],
            "retrieval_query_type": self.retrieval_query_type,
            "anchor": self.anchor,
            "relation_hint": self.relation_hint,
            "rank": rank,
            "score": round(float(self.score), 4),
        }


@dataclass
class EvidenceIndex:
    by_title: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    by_entity: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    by_relation: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    by_entity_relation: dict[tuple[str, str], list[dict[str, Any]]] = field(default_factory=dict)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows = read(Path(args.data_path), "hotpotqa", args.limit)
    diagnostics = [diagnose_row(row) for row in rows]
    summary = summarize(diagnostics)
    payload = {
        "data_path": args.data_path,
        "count": len(diagnostics),
        "ks": list(KS),
        "summary": summary,
        "diagnostics": diagnostics,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "diagnostics"}, ensure_ascii=False, indent=2))


def diagnose_row(row: dict[str, Any]) -> dict[str, Any]:
    facts = rub._hotpot_facts(row)
    index = build_evidence_index(row, facts)
    reasoning_type = rub._hotpot_reasoning_type(row)
    gold_ordered = gold_fact_ids_ordered(row)
    fact1 = gold_ordered[0] if gold_ordered else ""
    fact2 = gold_ordered[1] if len(gold_ordered) > 1 else ""
    pool1, pool2 = topology_retrieved_pools(row, facts, index, reasoning_type)
    rank1 = fact_rank(pool1, fact1)
    rank2 = fact_rank(pool2, fact2)
    return {
        "sample_id": row.get("sample_id"),
        "question": row.get("question"),
        "reasoning_type": reasoning_type,
        "gold_fact1": fact1,
        "gold_fact2": fact2,
        "fact1_rank": rank1,
        "fact2_rank": rank2,
        "gold_fact1_source": fact_source(pool1, fact1),
        "gold_fact2_source": fact_source(pool2, fact2),
        "fact1_pool": [item.as_dict(rank) for rank, item in enumerate(pool1[:10], start=1)],
        "fact2_pool": [item.as_dict(rank) for rank, item in enumerate(pool2[:10], start=1)],
        **recall_at_k(rank1, rank2),
    }


def build_evidence_index(row: dict[str, Any], facts: list[dict[str, Any]]) -> EvidenceIndex:
    return rub._hotpot_evidence_index(row, facts)


def add_index(mapping: dict[Any, list[dict[str, Any]]], key: Any, fact: dict[str, Any]) -> None:
    if not key:
        return
    bucket = mapping.setdefault(key, [])
    if all(str(item.get("fact_id", "")) != str(fact.get("fact_id", "")) for item in bucket):
        bucket.append(fact)


def topology_retrieved_pools(
    row: dict[str, Any],
    facts: list[dict[str, Any]],
    index: EvidenceIndex,
    reasoning_type: str,
) -> tuple[list[RetrievedFact], list[RetrievedFact]]:
    if reasoning_type == "bridge_entity":
        source, _ = rub._hotpot_question_entities(row)
        pool1 = merge_retrieved([
            from_ranked(rub._hotpot_rank_bridge_entity_source_facts(row, facts, source_entity=source, limit=20), "question_entity_anchor", "bridge_discovery", source, ""),
            from_ranked(rub._hotpot_retrieve_by_anchor(index, anchor=source, query_type="bridge_discovery", topk=40), "anchor_preserving_raw", "bridge_discovery", source, ""),
        ])
        inferred = []
        for item in pool1[:20]:
            inferred.extend(rub._hotpot_inferred_intermediate_hypotheses_from_fact(row, item.fact, limit=3))
        pool2_parts = []
        for bridge in [item for item in inferred if item]:
            relation = rub._hotpot_question_relation_label(row)
            pool2_parts.append(from_ranked(rub._hotpot_rank_relation_specific_hop2(row, bridge, facts, reasoning_type=reasoning_type, limit=20), "bridge_derived", "answer_lookup", bridge, relation))
            pool2_parts.append(from_ranked(rub._hotpot_retrieve_by_anchor_relation(index, anchor=bridge, relation=relation, query_type="answer_lookup", topk=40), "anchor_preserving_raw", "answer_lookup", bridge, relation))
        return pool1, merge_retrieved(pool2_parts)

    if reasoning_type == "bridge_attribute":
        source, _ = rub._hotpot_question_entities(row)
        pool1 = merge_retrieved([
            from_ranked(rub._hotpot_rank_source_side_facts(row, facts, limit=20), "question_entity_anchor", "bridge_discovery", source, ""),
            from_ranked(rub._hotpot_retrieve_by_anchor(index, anchor=source, query_type="bridge_discovery", topk=40), "anchor_preserving_raw", "bridge_discovery", source, ""),
        ])
        inferred = []
        for item in pool1[:16]:
            inferred.extend(rub._hotpot_inferred_intermediate_hypotheses_from_fact(row, item.fact, limit=3))
        relation = rub._hotpot_question_relation_label(row)
        pool2_parts = []
        for bridge in [item for item in inferred if item]:
            pool2_parts.append(from_ranked(rub._hotpot_rank_relation_specific_hop2(row, bridge, facts, reasoning_type=reasoning_type, limit=20), "bridge_derived", "answer_lookup", bridge, relation))
            pool2_parts.append(from_ranked(rub._hotpot_retrieve_by_anchor_relation(index, anchor=bridge, relation=relation, query_type="answer_lookup", topk=40), "anchor_preserving_raw", "answer_lookup", bridge, relation))
        return pool1, merge_retrieved(pool2_parts)

    if reasoning_type in {"comparison_attribute", "shared_property_yesno", "set_intersection"}:
        entity_a, entity_b = rub._hotpot_question_entities(row)
        relation = rub._hotpot_question_relation_label(row)
        return (
            parallel_pool(row, facts, index, entity_a, relation, "entity_a_lookup"),
            parallel_pool(row, facts, index, entity_b, relation, "entity_b_lookup"),
        )

    if reasoning_type == "multi_constraint_lookup":
        constraints = rub._hotpot_question_constraints(row)
        if len(constraints) >= 2:
            return (
                constraint_pool(row, facts, index, constraints[0], "constraint_1"),
                constraint_pool(row, facts, index, constraints[1], "constraint_2"),
            )

    hop1, hop2 = rub._hotpot_topology_fact_pools(row, facts=facts, bridge="", reasoning_type=reasoning_type)
    return (
        from_ranked(hop1, "dataset_context_scan", "fallback_hop1", "", ""),
        from_ranked(hop2, "dataset_context_scan", "fallback_hop2", "", ""),
    )


def parallel_pool(
    row: dict[str, Any],
    facts: list[dict[str, Any]],
    index: EvidenceIndex,
    entity: str,
    relation: str,
    query_type: str,
) -> list[RetrievedFact]:
    return merge_retrieved([
        from_ranked(rub._hotpot_rank_parallel_fact_candidates(row, entity, facts, role=query_type, limit=20), "question_entity_anchor", query_type, entity, relation),
        from_ranked(rub._hotpot_retrieve_by_anchor_relation(index, anchor=entity, relation=relation, query_type=query_type, topk=40), "anchor_preserving_raw", query_type, entity, relation),
        from_ranked(rub._hotpot_retrieve_by_anchor(index, anchor=entity, query_type=query_type, relation_hint=relation, topk=40), "anchor_preserving_raw", query_type, entity, relation),
    ])


def constraint_pool(
    row: dict[str, Any],
    facts: list[dict[str, Any]],
    index: EvidenceIndex,
    constraint: dict[str, str],
    query_type: str,
) -> list[RetrievedFact]:
    anchor = str(constraint.get("anchor", ""))
    relation = str(constraint.get("relation", ""))
    target_items = rub._hotpot_retrieve_relation_targets(row, constraint, facts, role=query_type, limit=40, evidence_index=index)
    ranked = [rub._hotpot_constraint_target_as_fact(item, role=query_type) for item in target_items]
    return merge_retrieved([
        from_ranked(ranked, "constraint_anchor", query_type, anchor, relation),
        from_ranked(rub._hotpot_retrieve_by_anchor_relation(index, anchor=anchor, relation=relation, query_type=query_type, topk=40), "anchor_preserving_raw", query_type, anchor, relation),
        from_ranked(rub._hotpot_retrieve_by_anchor(index, anchor=anchor, query_type=query_type, relation_hint=relation, topk=40), "anchor_preserving_raw", query_type, anchor, relation),
    ])


def from_ranked(
    facts: list[dict[str, Any]],
    source: str,
    query_type: str,
    anchor: str,
    relation_hint: str,
) -> list[RetrievedFact]:
    return [
        RetrievedFact(
            fact=fact,
            score=float(fact.get("retrieval_score", fact.get("hop_score", 0.0)) or 0.0),
            retrieval_source=source,
            retrieval_query_type=query_type,
            anchor=anchor,
            relation_hint=relation_hint,
        )
        for fact in facts
    ]


def from_index(
    index: EvidenceIndex,
    kind: str,
    anchor: str,
    source: str,
    query_type: str,
    query_anchor: str,
    relation_hint: str,
) -> list[RetrievedFact]:
    anchor_key = normalize_bridge(anchor)
    relation = rub._hotpot_canonical_relation(relation_hint)
    if kind == "title":
        facts = index.by_title.get(anchor_key, [])
    elif kind == "entity":
        facts = index.by_entity.get(anchor_key, [])
    elif kind == "entity_relation":
        facts = index.by_entity_relation.get((anchor_key, relation), [])
    else:
        facts = []
    base = 4.0 if kind == "entity_relation" else 2.0 if kind == "title" else 1.5
    return [
        RetrievedFact(
            fact=fact,
            score=base,
            retrieval_source=source,
            retrieval_query_type=query_type,
            anchor=query_anchor,
            relation_hint=relation_hint,
        )
        for fact in facts
    ]


def merge_retrieved(parts: list[list[RetrievedFact]]) -> list[RetrievedFact]:
    by_id: dict[str, RetrievedFact] = {}
    for part in parts:
        for item in part:
            fact_id = item.fact_id
            if not fact_id:
                continue
            existing = by_id.get(fact_id)
            if existing is None or item.score > existing.score:
                item.sources = sorted({item.retrieval_source, *(existing.sources if existing else [])})
                by_id[fact_id] = item
            elif existing is not None:
                existing.sources = sorted({existing.retrieval_source, *existing.sources, item.retrieval_source})
    return sorted(by_id.values(), key=lambda item: (-float(item.score), str(item.fact.get("fact_id", ""))))


def fact_rank(pool: list[RetrievedFact], fact_id: str) -> int | None:
    if not fact_id:
        return None
    for rank, item in enumerate(pool, start=1):
        if item.fact_id == fact_id:
            return rank
    return None


def fact_source(pool: list[RetrievedFact], fact_id: str) -> dict[str, Any]:
    rank = fact_rank(pool, fact_id)
    if rank is None:
        return {"rank": None, "retrieval_source": "", "retrieval_sources": []}
    item = pool[rank - 1]
    return {
        "rank": rank,
        "retrieval_source": item.retrieval_source,
        "retrieval_sources": item.sources or [item.retrieval_source],
        "retrieval_query_type": item.retrieval_query_type,
        "anchor": item.anchor,
        "relation_hint": item.relation_hint,
        "score": round(float(item.score), 4),
    }


def recall_at_k(rank1: int | None, rank2: int | None) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for k in KS:
        f1 = bool(rank1 is not None and rank1 <= k)
        f2 = bool(rank2 is not None and rank2 <= k)
        result[f"fact1_recall@{k}"] = f1
        result[f"fact2_recall@{k}"] = f2
        result[f"both_fact_recall@{k}"] = f1 and f2
    return result


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": summarize_bucket(items),
        "by_topology": {
            reasoning_type: summarize_bucket([item for item in items if item.get("reasoning_type") == reasoning_type])
            for reasoning_type in REASONING_TYPES
        },
    }


def summarize_bucket(items: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(items)
    result: dict[str, Any] = {"count": count}
    for k in KS:
        result[f"fact1_recall@{k}"] = rate(items, f"fact1_recall@{k}")
        result[f"fact2_recall@{k}"] = rate(items, f"fact2_recall@{k}")
        result[f"both_fact_recall@{k}"] = rate(items, f"both_fact_recall@{k}")
    result["gold_fact1_source_counts"] = source_counts(items, "gold_fact1_source")
    result["gold_fact2_source_counts"] = source_counts(items, "gold_fact2_source")
    result["gate"] = {
        "fact1_recall@10>=0.80": result["fact1_recall@10"] >= 0.80,
        "fact2_recall@10>=0.75": result["fact2_recall@10"] >= 0.75,
        "both_fact_recall@10>=0.65": result["both_fact_recall@10"] >= 0.65,
        "pass": result["fact1_recall@10"] >= 0.80
        and result["fact2_recall@10"] >= 0.75
        and result["both_fact_recall@10"] >= 0.65,
    }
    return result


def rate(items: list[dict[str, Any]], key: str) -> float:
    return sum(int(bool(item.get(key))) for item in items) / len(items) if items else 0.0


def source_counts(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        source = str((item.get(key) or {}).get("retrieval_source") or "miss")
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()
