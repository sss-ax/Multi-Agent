"""ACK/NACK protocol events for semantic communication stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .semantic_contract import SemanticCheckResult, SemanticRequirement, SemanticStatus


class SemanticFeedbackType(str, Enum):
    ACK = "ACK"
    NACK = "NACK"


@dataclass(frozen=True)
class SemanticACK:
    sender: str
    receiver: str
    round_index: int
    satisfied_requirements: tuple[str, ...] = ()
    recoverable_requirements: tuple[str, ...] = ()

    @property
    def type(self) -> SemanticFeedbackType:
        return SemanticFeedbackType.ACK

    def as_dict(self) -> dict[str, object]:
        return {
            "type": self.type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "round_index": self.round_index,
            "satisfied_requirements": list(self.satisfied_requirements),
            "recoverable_requirements": list(self.recoverable_requirements),
        }


@dataclass(frozen=True)
class SemanticNACK:
    sender: str
    receiver: str
    round_index: int
    missing_requirement_ids: tuple[str, ...] = ()
    missing_semantics: tuple[str, ...] = ()
    available_support_node_ids: tuple[str, ...] = ()

    @property
    def type(self) -> SemanticFeedbackType:
        return SemanticFeedbackType.NACK

    def as_dict(self) -> dict[str, object]:
        return {
            "type": self.type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "round_index": self.round_index,
            "missing_requirement_ids": list(self.missing_requirement_ids),
            "missing_semantics": list(self.missing_semantics),
            "available_support_node_ids": list(self.available_support_node_ids),
        }


SemanticFeedback = SemanticACK | SemanticNACK


def build_semantic_feedback(
    *,
    sender: str,
    receiver: str,
    round_index: int,
    results: Iterable[SemanticCheckResult],
) -> SemanticFeedback:
    """Build an ACK when every requirement is sufficient, otherwise a NACK."""

    checked = tuple(results)
    missing = tuple(result for result in checked if result.status == SemanticStatus.MISSING)
    if missing:
        return SemanticNACK(
            sender=sender.strip(),
            receiver=receiver.strip(),
            round_index=round_index,
            missing_requirement_ids=tuple(_requirement_id(result.requirement) for result in missing),
            missing_semantics=tuple(result.requirement.kind for result in missing),
            available_support_node_ids=_available_support_node_ids(checked),
        )
    return SemanticACK(
        sender=sender.strip(),
        receiver=receiver.strip(),
        round_index=round_index,
        satisfied_requirements=tuple(
            result.requirement.kind
            for result in checked
            if result.status == SemanticStatus.SATISFIED
        ),
        recoverable_requirements=tuple(
            result.requirement.kind
            for result in checked
            if result.status == SemanticStatus.RECOVERABLE
        ),
    )


def is_ack(feedback: SemanticFeedback) -> bool:
    return isinstance(feedback, SemanticACK)


def is_nack(feedback: SemanticFeedback) -> bool:
    return isinstance(feedback, SemanticNACK)


def _requirement_id(requirement: SemanticRequirement) -> str:
    if requirement.target_logical_id:
        return f"{requirement.kind}:{requirement.target_logical_id}"
    if requirement.required_type:
        return f"{requirement.kind}:{requirement.required_type}"
    return requirement.kind


def _available_support_node_ids(results: Iterable[SemanticCheckResult]) -> tuple[str, ...]:
    node_ids: list[str] = []
    for result in results:
        node_ids.extend(result.satisfying_node_ids)
        node_ids.extend(result.recoverable_from_node_ids)
    return tuple(dict.fromkeys(node_ids))
