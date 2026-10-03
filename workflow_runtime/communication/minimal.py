"""Minimal initial semantic packet policy without feedback repair."""

from __future__ import annotations

from .base import GraphCommunicationPolicy
from ..agent_graph_view import AgentGraphView
from ..graph_delta import DeltaCandidate
from ..models import GraphState


class MinimalNoFeedbackPolicy(GraphCommunicationPolicy):
    name = "minimal_no_feedback"

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


class MinimalSendAllFallbackPolicy(GraphCommunicationPolicy):
    name = "minimal_sendall_fallback"

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


class MinimalTargetedFeedbackPolicy(GraphCommunicationPolicy):
    name = "minimal_targeted_feedback"

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
