"""Receiver-predictive innovation detection for newly created graph nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .agent_graph_view import AgentGraphView
from .delta_extractor import communication_scope
from .models import GraphState, NodeVersion
from .semantic_contract import SemanticRequirement, SemanticStatus
from .semantic_resolver import SemanticResolver


@dataclass(frozen=True)
class InnovationDecision:
    node_id: str
    semantic_kind: str
    scope: str
    already_satisfied: bool
    recoverable: bool
    innovative: bool
    requirement: SemanticRequirement
    satisfying_node_ids: tuple[str, ...] = ()
    recoverable_from_node_ids: tuple[str, ...] = ()
    reason: str = ""
    confidence: float = 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "semantic_kind": self.semantic_kind,
            "scope": self.scope,
            "already_satisfied": self.already_satisfied,
            "recoverable": self.recoverable,
            "innovative": self.innovative,
            "requirement": self.requirement.as_dict(),
            "satisfying_node_ids": list(self.satisfying_node_ids),
            "recoverable_from_node_ids": list(self.recoverable_from_node_ids),
            "reason": self.reason,
            "confidence": self.confidence,
        }


class InnovationDetector:
    """Classify sender nodes by receiver-relative semantic innovation."""

    def __init__(
        self,
        state: GraphState,
        *,
        sender: str,
        receiver: str,
        receiver_view: AgentGraphView,
    ) -> None:
        self.state = state
        self.sender = sender
        self.receiver = receiver
        self.receiver_view = receiver_view
        self.resolver = SemanticResolver(state, receiver_view)

    def classify(self, node_ids: Iterable[str]) -> tuple[InnovationDecision, ...]:
        decisions: list[InnovationDecision] = []
        for node_id in dict.fromkeys(str(item) for item in node_ids):
            node = self.state.nodes.get(node_id)
            if node is None or not node.is_operationally_valid():
                continue
            decisions.append(self.classify_node(node))
        return tuple(decisions)

    def classify_node(self, node: NodeVersion) -> InnovationDecision:
        scope = communication_scope(node, sender=self.sender, receiver=self.receiver)
        requirement = semantic_requirement_for_node(node)
        semantic_kind = requirement.kind

        if scope == "LOCAL":
            return InnovationDecision(
                node_id=node.node_id,
                semantic_kind=semantic_kind,
                scope=scope,
                already_satisfied=False,
                recoverable=False,
                innovative=False,
                requirement=requirement,
                reason="local node is never part of semantic communication",
                confidence=1.0,
            )
        if scope == "MANDATORY":
            return InnovationDecision(
                node_id=node.node_id,
                semantic_kind=semantic_kind,
                scope=scope,
                already_satisfied=True,
                recoverable=False,
                innovative=False,
                requirement=requirement,
                reason="global mandatory semantic is baseline context",
                confidence=1.0,
            )

        result = self.resolver.resolve(requirement)
        already_satisfied = result.status == SemanticStatus.SATISFIED
        recoverable = result.status == SemanticStatus.RECOVERABLE
        innovative = result.status == SemanticStatus.MISSING
        return InnovationDecision(
            node_id=node.node_id,
            semantic_kind=semantic_kind,
            scope=scope,
            already_satisfied=already_satisfied,
            recoverable=recoverable,
            innovative=innovative,
            requirement=requirement,
            satisfying_node_ids=result.satisfying_node_ids,
            recoverable_from_node_ids=result.recoverable_from_node_ids,
            reason=result.reason,
            confidence=1.0,
        )


def detect_innovations(
    state: GraphState,
    *,
    sender: str,
    receiver: str,
    receiver_view: AgentGraphView,
    node_ids: Iterable[str],
) -> tuple[InnovationDecision, ...]:
    return InnovationDetector(
        state,
        sender=sender,
        receiver=receiver,
        receiver_view=receiver_view,
    ).classify(node_ids)


def innovative_node_ids(decisions: Iterable[InnovationDecision]) -> tuple[str, ...]:
    return tuple(decision.node_id for decision in decisions if decision.innovative)


def semantic_requirement_for_node(node: NodeVersion) -> SemanticRequirement:
    kind = _semantic_kind_for_node(node)
    return SemanticRequirement(
        kind=kind,
        target_logical_id=node.logical_id,
        required_type=node.type,
        description=f"semantic represented by {node.logical_id}@v{node.version}",
    )


def _semantic_kind_for_node(node: NodeVersion) -> str:
    if node.type in {"task", "query_spec"}:
        return "global_context"
    if node.type in {"fact", "facts"}:
        return "task_inputs"
    if node.type in {"requirements", "test", "table", "table_cell", "evidence", "entity", "supporting_fact", "evidence_link", "choice"}:
        return "task_inputs"
    if node.type in {"plan", "plan_steps"}:
        return "plan_or_operation"
    if node.type in {"calculation", "execution", "test_result"}:
        return "support_dependencies"
    if node.type == "result":
        return "candidate_answer_or_code"
    if node.type == "code":
        return "candidate_code"
    if node.type == "verification":
        return "validation_signal"
    if node.type == "final_answer":
        return "final_candidate"
    if node.type in {"tool_request", "tool_result", "error"}:
        return "local_runtime"
    return node.type
