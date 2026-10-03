"""Send every exportable candidate root."""

from __future__ import annotations

from .base import GraphCommunicationPolicy
from ..agent_graph_view import AgentGraphView
from ..graph_delta import DeltaCandidate
from ..models import GraphState


class SendAllPolicy(GraphCommunicationPolicy):
    name = "send_all"

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
        return [
            candidate.node_id
            for candidate in candidates
            if candidate.exportable and candidate.scope == "OPTIONAL_POLICY"
        ]
