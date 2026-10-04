"""Deterministic targeted refinement planning for semantic feedback requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .agent_graph_view import AgentGraphView
from .incremental_planner import IncrementalRequestPlanner
from .models import GraphState
from .semantic_feedback import SemanticNACK

REFINEMENT_TARGETED = "TARGETED"
REFINEMENT_UNRESOLVABLE = "UNRESOLVABLE"


@dataclass(frozen=True)
class RefinementPlan:
    sender: str
    receiver: str
    missing_semantics: tuple[str, ...]
    root_node_ids: tuple[str, ...]
    reason_by_node: dict[str, str]
    requested_fragment_ids: tuple[str, ...] = ()
    candidate_fragment_ids: tuple[str, ...] = ()
    selected_fragment_ids: tuple[str, ...] = ()
    rejected_fragment_ids: tuple[str, ...] = ()
    reason_by_fragment: dict[str, str] | None = None
    request_level: str = ""
    target_requirements: tuple[str, ...] = ()
    token_cost: int = 0
    estimated_cost: int = 0
    estimated_utility: float = 0.0
    utility_per_token: float = 0.0
    status: str = REFINEMENT_TARGETED

    @property
    def is_empty(self) -> bool:
        return not self.root_node_ids

    @property
    def is_unresolvable(self) -> bool:
        return self.status == REFINEMENT_UNRESOLVABLE

    def as_dict(self) -> dict[str, object]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "missing_semantics": list(self.missing_semantics),
            "root_node_ids": list(self.root_node_ids),
            "reason_by_node": dict(self.reason_by_node),
            "requested_fragment_ids": list(self.requested_fragment_ids),
            "candidate_fragments": list(self.candidate_fragment_ids),
            "selected_fragments": list(self.selected_fragment_ids),
            "rejected_fragments": list(self.rejected_fragment_ids),
            "reason_by_fragment": dict(self.reason_by_fragment or {}),
            "reason": dict(self.reason_by_fragment or {}),
            "request_level": self.request_level,
            "target_requirements": list(self.target_requirements),
            "token_cost": self.token_cost,
            "estimated_cost": self.estimated_cost,
            "estimated_utility": self.estimated_utility,
            "utility_per_token": self.utility_per_token,
            "status": self.status,
            "is_unresolvable": self.is_unresolvable,
            "is_empty": self.is_empty,
        }


class RefinementPlanner:
    """Choose minimal sender-side roots for missing receiver semantics."""

    def __init__(
        self,
        state: GraphState,
        *,
        sender: str,
        receiver: str,
        sender_view: AgentGraphView,
        receiver_view: AgentGraphView,
    ) -> None:
        self.state = state
        self.sender = sender
        self.receiver = receiver
        self.sender_view = sender_view
        self.receiver_view = receiver_view

    def plan_for_nack(self, nack: SemanticNACK) -> RefinementPlan:
        return self.plan(nack.missing_semantics)

    def plan(self, missing_semantics: Iterable[str]) -> RefinementPlan:
        semantics = tuple(dict.fromkeys(str(item) for item in missing_semantics))
        incremental = IncrementalRequestPlanner(
            self.state,
            sender=self.sender,
            receiver=self.receiver,
            sender_view=self.sender_view,
            receiver_view=self.receiver_view,
        ).plan(semantics, level=_level_for_semantics(semantics))
        status = REFINEMENT_UNRESOLVABLE if incremental.is_unresolvable else REFINEMENT_TARGETED
        return RefinementPlan(
            sender=self.sender,
            receiver=self.receiver,
            missing_semantics=semantics,
            root_node_ids=incremental.root_node_ids,
            reason_by_node=incremental.reason_by_node,
            requested_fragment_ids=incremental.requested_fragment_ids,
            candidate_fragment_ids=incremental.candidate_fragment_ids,
            selected_fragment_ids=incremental.selected_fragment_ids,
            rejected_fragment_ids=incremental.rejected_fragment_ids,
            reason_by_fragment=incremental.reason_by_fragment,
            request_level=incremental.level,
            target_requirements=incremental.target_requirements,
            token_cost=incremental.token_cost,
            estimated_cost=incremental.estimated_cost,
            estimated_utility=incremental.estimated_utility,
            utility_per_token=incremental.utility_per_token,
            status=status,
        )


def plan_refinement(
    state: GraphState,
    *,
    sender: str,
    receiver: str,
    sender_view: AgentGraphView,
    receiver_view: AgentGraphView,
    missing_semantics: Iterable[str],
) -> RefinementPlan:
    return RefinementPlanner(
        state,
        sender=sender,
        receiver=receiver,
        sender_view=sender_view,
        receiver_view=receiver_view,
    ).plan(missing_semantics)


def _level_for_semantics(semantics: tuple[str, ...]) -> str:
    if any(item in {"support_dependencies"} for item in semantics):
        return "verification"
    if any(item in {"full_feedback", "critic_confidence_low", "insufficient_derivation"} for item in semantics):
        return "quality"
    return "hard"
