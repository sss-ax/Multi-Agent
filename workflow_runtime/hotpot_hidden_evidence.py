"""Leakage-safe local hidden-evidence environment for HotpotQA Phase 2B.

The corpus contains only question/context data.  Gold answers and
``supporting_facts`` annotations are deliberately not accepted by this module.
Evidence enters the graph only after a RETRIEVE action, so merely taking a
snapshot cannot expose the withheld sentences to a policy feature extractor.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .communication_action import (
    CommunicationAction,
    CommunicationActionType,
    CommunicationPreview,
)
from .communication_replay import CommunicationDecisionSnapshot
from .hotpot_graph_retriever import GraphSeedRetriever, QueryGuidedGraphRetriever


@dataclass(frozen=True)
class HotpotHiddenEvidenceEnvironment:
    question: str
    visible_facts: tuple[dict[str, Any], ...]
    hidden_facts: tuple[dict[str, Any], ...]
    split_method: str = "question_only_title_ranking"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HotpotHiddenEvidenceEnvironment":
        forbidden = {"gold_answer", "answer", "supporting_facts"} & set(payload)
        if forbidden:
            raise ValueError(f"hidden evidence environment contains oracle fields: {sorted(forbidden)}")
        return cls(
            question=str(payload.get("question", "")),
            visible_facts=tuple(copy.deepcopy(list(payload.get("visible_facts", ())))),
            hidden_facts=tuple(copy.deepcopy(list(payload.get("hidden_facts", ())))),
            split_method=str(payload.get("split_method", "question_only_title_ranking")),
        )


def build_hotpot_hidden_environment(
    question: str,
    facts: Iterable[Mapping[str, Any]],
    *,
    initial_title_count: int = 1,
) -> HotpotHiddenEvidenceEnvironment:
    """Expose the most question-related document titles and withhold the rest.

    Ranking sees only the question and raw context.  It never sees HotpotQA's
    gold supporting-fact pairs, which remain evaluator-only labels.
    """
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(facts):
        fact = copy.deepcopy(dict(raw))
        fact_id = str(fact.get("fact_id") or f"hotpot_fact_{index + 1}")
        normalized.append({
            "fact_id": fact_id,
            "title": str(fact.get("title", fact.get("entity", ""))),
            "sent_id": int(fact.get("sent_id", index) or 0),
            "text": str(fact.get("text", "")),
        })
    if not normalized:
        return HotpotHiddenEvidenceEnvironment(str(question), (), ())

    seeds = GraphSeedRetriever().select(str(question), normalized, limit=len(normalized))
    score_by_id = {seed.fact_id: seed.score for seed in seeds}
    title_order: dict[str, tuple[float, int]] = {}
    for index, fact in enumerate(normalized):
        title = fact["title"]
        score = score_by_id.get(fact["fact_id"], 0.0)
        previous = title_order.get(title)
        if previous is None or score > previous[0]:
            title_order[title] = (score, index)
    ranked_titles = sorted(title_order, key=lambda title: (-title_order[title][0], title_order[title][1], title))
    visible_titles = set(ranked_titles[: max(0, int(initial_title_count))])
    visible = tuple(fact for fact in normalized if fact["title"] in visible_titles)
    hidden = tuple(fact for fact in normalized if fact["title"] not in visible_titles)
    return HotpotHiddenEvidenceEnvironment(str(question), visible, hidden)


class HotpotLocalInformationAdapter:
    """Implement REQUEST/RETRIEVE/REVISE against a local hidden corpus."""

    def __init__(self, environment: HotpotHiddenEvidenceEnvironment, *, retrieval_limit: int = 4) -> None:
        self.environment = environment
        self.retrieval_limit = max(1, int(retrieval_limit))
        self.retriever = QueryGuidedGraphRetriever(beam_size=max(8, self.retrieval_limit * 2))

    @classmethod
    def from_snapshot(
        cls,
        snapshot: CommunicationDecisionSnapshot,
        *,
        retrieval_limit: int = 4,
    ) -> "HotpotLocalInformationAdapter":
        payload = snapshot.external_state.get("hotpot_hidden_evidence")
        if not isinstance(payload, Mapping):
            raise ValueError("snapshot has no HotpotQA hidden evidence environment")
        return cls(HotpotHiddenEvidenceEnvironment.from_dict(payload), retrieval_limit=retrieval_limit)

    def preview(self, snapshot: CommunicationDecisionSnapshot, action: CommunicationAction) -> CommunicationPreview:
        if action.action_type == CommunicationActionType.RETRIEVE:
            if not action.query.strip():
                return CommunicationPreview(action=action, legal=False, reason="RETRIEVE requires a non-empty query")
            facts, metadata = self._retrieve(snapshot, action.query)
            if not facts:
                return CommunicationPreview(action=action, legal=False, reason="no new local evidence matched the query")
            tokens = sum(_token_cost(f"{fact['title']}: {fact['text']}") for fact in facts)
            return CommunicationPreview(
                action=action,
                legal=True,
                reason=f"local hidden corpus returned {len(facts)} facts via {metadata['retrieval_method']}",
                communication_tokens=tokens,
            )
        if action.action_type == CommunicationActionType.REQUEST:
            if not action.requested_semantics:
                return CommunicationPreview(action=action, legal=False, reason="REQUEST requires requested_semantics")
            return CommunicationPreview(action=action, legal=True, reason="information request recorded")
        if action.action_type == CommunicationActionType.REVISE:
            if not action.target.strip():
                return CommunicationPreview(action=action, legal=False, reason="REVISE requires a target")
            return CommunicationPreview(action=action, legal=True, reason="revision request recorded")
        return CommunicationPreview(action=action, legal=False, reason="adapter supports REQUEST/RETRIEVE/REVISE only")

    def apply(self, snapshot, action, *, store, views) -> CommunicationPreview:
        preview = self.preview(snapshot, action)
        if not preview.legal:
            return preview
        if action.action_type == CommunicationActionType.RETRIEVE:
            facts, metadata = self._retrieve(snapshot, action.query)
            node_ids: list[str] = []
            edge_ids: list[str] = []
            task = store.latest_valid(snapshot.task_id, snapshot.branch_id, "task")
            for fact in facts:
                safe_fact_id = re.sub(r"[^a-zA-Z0-9_]+", "_", fact["fact_id"]).strip("_")
                safe_title = re.sub(r"[^a-zA-Z0-9_]+", "_", fact["title"]).strip("_") or "untitled"
                entity_logical_id = f"retrieved_entity_{safe_title}"
                entity = store.latest_valid(snapshot.task_id, snapshot.branch_id, entity_logical_id)
                if entity is None:
                    entity = store.add_node(
                        task_id=snapshot.task_id,
                        branch_id=snapshot.branch_id,
                        logical_id=entity_logical_id,
                        node_type="entity",
                        content=json.dumps({"name": fact["title"]}, ensure_ascii=False),
                        owner="retriever",
                        created_by_role="retriever",
                        validation={"schema_valid": True},
                        provenance={
                            "source": "hotpot_local_hidden_corpus",
                            "query": action.query,
                            "retrieval_method": metadata["retrieval_method"],
                        },
                    )
                    node_ids.append(entity.node_id)
                logical_id = "retrieved_" + safe_fact_id
                existing = store.latest_valid(snapshot.task_id, snapshot.branch_id, logical_id)
                node = existing or store.add_node(
                    task_id=snapshot.task_id,
                    branch_id=snapshot.branch_id,
                    logical_id=logical_id,
                    node_type="supporting_fact",
                    content=json.dumps({k: fact[k] for k in ("title", "sent_id", "text")}, ensure_ascii=False),
                    owner="retriever",
                    created_by_role="retriever",
                    validation={"schema_valid": True},
                    provenance={
                        "source": "hotpot_local_hidden_corpus",
                        "corpus_fact_id": fact["fact_id"],
                        "query": action.query,
                        "retrieval_method": metadata["retrieval_method"],
                    },
                )
                node_ids.append(node.node_id)
                link_logical_id = "retrieved_evidence_link_" + safe_fact_id
                link = store.latest_valid(snapshot.task_id, snapshot.branch_id, link_logical_id)
                if link is None:
                    link = store.add_node(
                        task_id=snapshot.task_id,
                        branch_id=snapshot.branch_id,
                        logical_id=link_logical_id,
                        node_type="evidence_link",
                        content=json.dumps({
                            "entity": fact["title"],
                            "sent_id": fact["sent_id"],
                            "source": "retrieved_context_index",
                        }, ensure_ascii=False),
                        owner="retriever",
                        created_by_role="retriever",
                        validation={"schema_valid": True},
                        provenance={
                            "source": "hotpot_local_hidden_corpus",
                            "corpus_fact_id": fact["fact_id"],
                            "query": action.query,
                            "retrieval_method": metadata["retrieval_method"],
                        },
                    )
                    node_ids.append(link.node_id)
                if task is not None:
                    for retrieved_node in (entity, node, link):
                        edge = store.add_edge(
                            source=task.node_id,
                            target=retrieved_node.node_id,
                            relation="depends_on",
                            created_by_role="retriever",
                            metadata={"retrieval_query": action.query},
                        )
                        edge_ids.append(edge.edge_id)
            views.grant(
                snapshot.receiver,
                node_ids=node_ids,
                edge_ids=edge_ids,
                delta_id=f"retrieve:{snapshot.snapshot_id}",
                packet_id=f"retrieve-packet:{snapshot.snapshot_id}",
            )
            return CommunicationPreview(
                action=action,
                legal=True,
                reason=preview.reason,
                closure_node_ids=tuple(node_ids),
                closure_edge_ids=tuple(edge_ids),
                communication_tokens=preview.communication_tokens,
            )

        kind = "information_request" if action.action_type == CommunicationActionType.REQUEST else "revision_request"
        content = (
            {"requested_semantics": list(action.requested_semantics)}
            if action.action_type == CommunicationActionType.REQUEST
            else {"target": action.target}
        )
        index = 1 + sum(1 for node in store.snapshot().nodes.values() if node.type == kind)
        node = store.add_node(
            task_id=snapshot.task_id,
            branch_id=snapshot.branch_id,
            logical_id=f"{kind}_{index}",
            node_type=kind,
            content=json.dumps(content, ensure_ascii=False),
            owner=snapshot.receiver,
            created_by_role=snapshot.receiver,
            provenance={"source": "phase2b_action", "action_type": action.action_type.value},
        )
        views.grant(snapshot.receiver, node_ids=[node.node_id], local=True)
        return CommunicationPreview(action=action, legal=True, reason=preview.reason, closure_node_ids=(node.node_id,))

    def _retrieve(self, snapshot: CommunicationDecisionSnapshot, query: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        already = {
            str(node.provenance.get("corpus_fact_id"))
            for node in snapshot.graph_state.nodes.values()
            if node.provenance.get("source") == "hotpot_local_hidden_corpus"
        }
        available = [dict(fact) for fact in self.environment.hidden_facts if fact["fact_id"] not in already]
        combined_query = f"{self.environment.question}\nInformation needed: {query}"
        result = self.retriever.retrieve(combined_query, available, limit=max(self.retrieval_limit, 2))
        by_id = {fact["fact_id"]: fact for fact in available}
        selected_ids: list[str] = []
        for path in result["paths"]:
            for fact_id in path["evidence_ids"]:
                if fact_id not in selected_ids:
                    selected_ids.append(fact_id)
                if len(selected_ids) >= self.retrieval_limit:
                    break
            if len(selected_ids) >= self.retrieval_limit:
                break
        if not selected_ids and available:
            # A zero-overlap query must still have a deterministic, non-oracle
            # corpus fallback so RETRIEVE remains an executable action.  The
            # original context order is dataset-provided, not a gold label.
            selected_ids = [fact["fact_id"] for fact in available[: self.retrieval_limit]]
            result = {**result, "retrieval_method": "query_guided_ppr_beam:context_order_fallback"}
        return [by_id[fact_id] for fact_id in selected_ids if fact_id in by_id], result


def _token_cost(text: str) -> int:
    return max(1, len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)))
