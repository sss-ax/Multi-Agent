"""Semantic packet structures and deterministic initial packet builder."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from .delta_scoring import estimate_node_tokens
from .graph_delta import GraphDelta
from .models import GraphState
from .semantic_contract import SemanticContract
from .semantic_innovation import InnovationDecision


DECISION_SUFFICIENT = 0
VERIFICATION_SUFFICIENT = 1
RECONSTRUCTION_SUFFICIENT = 2
FULL_SUPPORT = 3


@dataclass(frozen=True)
class SemanticPacket:
    packet_id: str
    sender: str
    receiver: str
    level: int
    round_index: int
    packet_type: str
    target_requirements: tuple[str, ...]
    root_node_ids: tuple[str, ...]
    closure_node_ids: tuple[str, ...] = ()
    edge_ids: tuple[str, ...] = ()
    token_cost: int = 0
    is_initial: bool = False
    is_fallback: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "packet_id": self.packet_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "level": self.level,
            "round_index": self.round_index,
            "packet_type": self.packet_type,
            "target_requirements": list(self.target_requirements),
            "root_node_ids": list(self.root_node_ids),
            "closure_node_ids": list(self.closure_node_ids),
            "edge_ids": list(self.edge_ids),
            "token_cost": self.token_cost,
            "is_initial": self.is_initial,
            "is_fallback": self.is_fallback,
        }

    @property
    def payload_node_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys([*self.root_node_ids, *self.closure_node_ids]))

    def with_delta(self, delta: GraphDelta) -> "SemanticPacket":
        """Attach the actual dependency-closed payload selected for delivery."""
        root_set = set(self.root_node_ids)
        closure_node_ids = tuple(node_id for node_id in delta.node_ids if node_id not in root_set)
        return SemanticPacket(
            packet_id=self.packet_id,
            sender=self.sender,
            receiver=self.receiver,
            level=self.level,
            round_index=self.round_index,
            packet_type=self.packet_type,
            target_requirements=self.target_requirements,
            root_node_ids=self.root_node_ids,
            closure_node_ids=closure_node_ids,
            edge_ids=delta.edge_ids,
            token_cost=delta.token_cost,
            is_initial=self.is_initial,
            is_fallback=self.is_fallback,
        )

    def with_roots(self, root_node_ids: Iterable[str]) -> "SemanticPacket":
        """Return a packet whose audited roots match the runtime send roots."""
        return replace(self, root_node_ids=tuple(dict.fromkeys(str(node_id) for node_id in root_node_ids)))


class SemanticPacketBuilder:
    """Build deterministic semantic packets from innovation decisions."""

    def __init__(self, state: GraphState):
        self.state = state

    def build_initial(
        self,
        *,
        sender: str,
        receiver: str,
        contract: SemanticContract,
        innovation_decisions: Iterable[InnovationDecision],
        round_index: int = 0,
    ) -> SemanticPacket:
        decisions = tuple(innovation_decisions)
        selected = self._select_initial_roots(contract, decisions)
        token_cost = sum(estimate_node_tokens(self.state.nodes[node_id]) for node_id in selected if node_id in self.state.nodes)
        semantic_by_node = {decision.node_id: decision.semantic_kind for decision in decisions}
        target_requirements = tuple(dict.fromkeys(semantic_by_node[node_id] for node_id in selected if node_id in semantic_by_node))
        return SemanticPacket(
            packet_id=_packet_id(sender, receiver, round_index, DECISION_SUFFICIENT, selected),
            sender=sender,
            receiver=receiver,
            level=DECISION_SUFFICIENT,
            round_index=round_index,
            packet_type="initial",
            target_requirements=target_requirements,
            root_node_ids=selected,
            closure_node_ids=(),
            edge_ids=(),
            token_cost=token_cost,
            is_initial=True,
            is_fallback=False,
        )

    def _select_initial_roots(
        self,
        contract: SemanticContract,
        innovation_decisions: Iterable[InnovationDecision],
    ) -> tuple[str, ...]:
        contract_kinds = {requirement.kind for requirement in contract.requirements}
        candidates = [
            decision
            for decision in innovation_decisions
            if decision.innovative and decision.node_id in self.state.nodes
        ]
        relevant = [
            decision for decision in candidates
            if (
                not contract_kinds
                or (
                    decision.semantic_kind in contract_kinds
                    and decision.semantic_kind in _INITIAL_PACKET_KINDS
                )
            )
        ]
        if not contract_kinds:
            relevant = []
        ordered = sorted(
            relevant,
            key=lambda decision: (
                _semantic_priority(decision.semantic_kind),
                estimate_node_tokens(self.state.nodes[decision.node_id]),
                decision.node_id,
            ),
        )
        return tuple(dict.fromkeys(decision.node_id for decision in ordered))


def build_initial_semantic_packet(
    state: GraphState,
    *,
    sender: str,
    receiver: str,
    contract: SemanticContract,
    innovation_decisions: Iterable[InnovationDecision],
    round_index: int = 0,
) -> SemanticPacket:
    return SemanticPacketBuilder(state).build_initial(
        sender=sender,
        receiver=receiver,
        contract=contract,
        innovation_decisions=innovation_decisions,
        round_index=round_index,
    )


def _semantic_priority(kind: str) -> int:
    priorities = {
        "final_candidate": 0,
        "candidate_answer_or_code": 1,
        "candidate_code": 1,
        "validation_signal": 2,
        "plan_or_operation": 3,
        "task_inputs": 4,
        "support_dependencies": 5,
    }
    return priorities.get(kind, 100)


_INITIAL_PACKET_KINDS = frozenset({
    "final_candidate",
    "candidate_answer_or_code",
    "candidate_code",
    "validation_signal",
    "plan_or_operation",
})


def _packet_id(sender: str, receiver: str, round_index: int, level: int, root_node_ids: tuple[str, ...]) -> str:
    suffix = "-".join(root_node_ids) if root_node_ids else "empty"
    safe_suffix = suffix.replace("@", "_").replace(":", "_").replace("/", "_")
    return f"packet:{sender}->{receiver}:r{round_index}:l{level}:{safe_suffix}"
