"""Policy interface for graph-structured inter-agent communication."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..agent_graph_view import AgentGraphView
from ..graph_delta import DeltaCandidate
from ..models import GraphState


class GraphCommunicationPolicy(ABC):
    name = "base"
    include_dependency_closure = True

    @abstractmethod
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
        """Return selected root node IDs before receiver-relative closure."""
