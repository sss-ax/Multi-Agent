"""Closure-aware graph-structural communication policy."""

from __future__ import annotations

from .base import GraphCommunicationPolicy
from ..agent_graph_view import AgentGraphView
from ..graph_delta import DeltaCandidate
from ..models import GraphState


class ClosureAwareHeuristicPolicy(GraphCommunicationPolicy):
    name = "closure_aware_heuristic"

    def select_roots(
        self,
        *,
        state: GraphState,
        sender_view: AgentGraphView,
        receiver_view: AgentGraphView,
        candidates: list[DeltaCandidate],
        task: str = "",
        budget_tokens: int | None = None,
    ) -> list[str]:
        scored = sorted(
            (
                candidate for candidate in candidates
                if candidate.exportable and candidate.scope == "OPTIONAL_POLICY"
            ),
            key=self._score,
            reverse=True,
        )
        selected: list[str] = []
        used = 0
        for candidate in scored:
            if candidate.novelty_score <= 0:
                continue
            if budget_tokens is not None and used + candidate.token_cost > budget_tokens:
                continue
            selected.append(candidate.node_id)
            used += candidate.token_cost
        return selected

    @staticmethod
    def _score(candidate: DeltaCandidate) -> float:
        features = candidate.structural_features
        return (
            2.0 * candidate.novelty_score
            + 0.5 * float(features.get("reachability_to_answer", False))
            + 0.05 * float(features.get("descendant_count", 0))
            - 0.5 * candidate.redundancy_score
            - 0.01 * candidate.token_cost
        )
