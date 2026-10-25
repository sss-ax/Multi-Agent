"""Communication policy whose optional roots come from model send_hints."""

from __future__ import annotations

from ..agent_graph_view import AgentGraphView
from ..graph_delta import DeltaCandidate
from ..models import GraphState
from .base import GraphCommunicationPolicy


class MandatoryOnlyPolicy(GraphCommunicationPolicy):
    name = "mandatory_only"
    include_dependency_closure = True

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
        return []


class ModelSelectedPolicy(GraphCommunicationPolicy):
    name = "model_selected"
    include_dependency_closure = True

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
        # LangGraphWorkflow supplies the model's validated send_hints.  Empty
        # here is intentional: a missing/invalid hint must never become an
        # implicit send-all fallback.
        return []
