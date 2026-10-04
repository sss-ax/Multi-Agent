"""ACK/NACK protocol events for semantic communication stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .receiver_need import ReceiverNeed, ReceiverNeedLevel, diagnose_receiver_need
from .semantic_contract import (
    SemanticCheckResult,
    SemanticRequirement,
    SemanticRequirementSeverity,
    SemanticStatus,
)


class SemanticFeedbackType(str, Enum):
    NONE = "NONE"
    HARD_NACK = "HARD_NACK"
    VERIFICATION_REQUEST = "VERIFICATION_REQUEST"
    QUALITY_REQUEST = "QUALITY_REQUEST"


@dataclass(frozen=True)
class SemanticACK:
    sender: str
    receiver: str
    round_index: int
    satisfied_requirements: tuple[str, ...] = ()
    recoverable_requirements: tuple[str, ...] = ()
    level: str = ReceiverNeedLevel.SATISFIED.value

    @property
    def type(self) -> SemanticFeedbackType:
        return SemanticFeedbackType.NONE

    def as_dict(self) -> dict[str, object]:
        return {
            "type": self.type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "round_index": self.round_index,
            "level": self.level,
            "satisfied_requirements": list(self.satisfied_requirements),
            "recoverable_requirements": list(self.recoverable_requirements),
        }


@dataclass(frozen=True)
class SemanticNACK:
    sender: str
    receiver: str
    round_index: int
    level: str = ReceiverNeedLevel.HARD.value
    missing_requirement_ids: tuple[str, ...] = ()
    missing_semantics: tuple[str, ...] = ()
    hard_missing_requirement_ids: tuple[str, ...] = ()
    hard_missing_semantics: tuple[str, ...] = ()
    verification_missing_requirement_ids: tuple[str, ...] = ()
    verification_missing_semantics: tuple[str, ...] = ()
    quality_gap_requirement_ids: tuple[str, ...] = ()
    quality_gaps: tuple[str, ...] = ()
    soft_missing_requirement_ids: tuple[str, ...] = ()
    soft_missing_semantics: tuple[str, ...] = ()
    available_support_node_ids: tuple[str, ...] = ()
    requested_fragments: tuple[str, ...] = ()
    receiver_need: ReceiverNeed | None = None

    @property
    def type(self) -> SemanticFeedbackType:
        if self.hard_missing_semantics:
            return SemanticFeedbackType.HARD_NACK
        if self.verification_missing_semantics:
            return SemanticFeedbackType.VERIFICATION_REQUEST
        return SemanticFeedbackType.QUALITY_REQUEST

    def as_dict(self) -> dict[str, object]:
        return {
            "type": self.type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "round_index": self.round_index,
            "level": self.level,
            "missing_requirement_ids": list(self.missing_requirement_ids),
            "missing_semantics": list(self.missing_semantics),
            "hard_missing_requirement_ids": list(self.hard_missing_requirement_ids),
            "hard_missing_semantics": list(self.hard_missing_semantics),
            "verification_missing_requirement_ids": list(self.verification_missing_requirement_ids),
            "verification_missing_semantics": list(self.verification_missing_semantics),
            "quality_gap_requirement_ids": list(self.quality_gap_requirement_ids),
            "quality_gaps": list(self.quality_gaps),
            "soft_missing_requirement_ids": list(self.soft_missing_requirement_ids),
            "soft_missing_semantics": list(self.soft_missing_semantics),
            "has_hard_missing": bool(self.hard_missing_semantics),
            "has_soft_missing": bool(self.soft_missing_semantics),
            "available_support_node_ids": list(self.available_support_node_ids),
            "requested_fragments": list(self.requested_fragments),
            "receiver_need": self.receiver_need.as_dict() if self.receiver_need is not None else None,
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
    need = diagnose_receiver_need(
        sender=sender,
        receiver=receiver,
        round_index=round_index,
        results=checked,
    )
    missing = tuple(result for result in checked if result.status == SemanticStatus.MISSING)
    if missing:
        hard_missing = tuple(result for result in missing if _requirement_severity(result.requirement) == SemanticRequirementSeverity.HARD.value)
        verification_missing = tuple(result for result in missing if _requirement_severity(result.requirement) == SemanticRequirementSeverity.VERIFICATION.value)
        quality_missing = tuple(result for result in missing if _requirement_severity(result.requirement) == SemanticRequirementSeverity.REFINEMENT.value)
        soft_missing = (*verification_missing, *quality_missing)
        return SemanticNACK(
            sender=sender.strip(),
            receiver=receiver.strip(),
            round_index=round_index,
            level=need.level.value,
            missing_requirement_ids=tuple(_requirement_id(result.requirement) for result in missing),
            missing_semantics=tuple(result.requirement.kind for result in missing),
            hard_missing_requirement_ids=tuple(_requirement_id(result.requirement) for result in hard_missing),
            hard_missing_semantics=tuple(result.requirement.kind for result in hard_missing),
            verification_missing_requirement_ids=tuple(_requirement_id(result.requirement) for result in verification_missing),
            verification_missing_semantics=tuple(result.requirement.kind for result in verification_missing),
            quality_gap_requirement_ids=tuple(_requirement_id(result.requirement) for result in quality_missing),
            quality_gaps=tuple(result.requirement.kind for result in quality_missing),
            soft_missing_requirement_ids=tuple(_requirement_id(result.requirement) for result in soft_missing),
            soft_missing_semantics=tuple(result.requirement.kind for result in soft_missing),
            available_support_node_ids=need.available_support_node_ids,
            requested_fragments=need.requested_fragments,
            receiver_need=need,
        )
    return SemanticACK(
        sender=sender.strip(),
        receiver=receiver.strip(),
        round_index=round_index,
        level=need.level.value,
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
    return is_hard_nack(feedback)


def is_hard_nack(feedback: SemanticFeedback) -> bool:
    return isinstance(feedback, SemanticNACK) and bool(feedback.hard_missing_semantics)


def is_soft_nack(feedback: SemanticFeedback) -> bool:
    return isinstance(feedback, SemanticNACK) and bool(feedback.soft_missing_semantics)


def is_verification_nack(feedback: SemanticFeedback) -> bool:
    return isinstance(feedback, SemanticNACK) and bool(feedback.verification_missing_semantics)


def is_quality_nack(feedback: SemanticFeedback) -> bool:
    return isinstance(feedback, SemanticNACK) and bool(feedback.quality_gaps)


def is_feedback_request(feedback: SemanticFeedback) -> bool:
    return isinstance(feedback, SemanticNACK)


def nack_missing_semantics(
    feedback: SemanticFeedback,
    *,
    include_hard: bool = True,
    include_soft: bool = True,
) -> tuple[str, ...]:
    if not isinstance(feedback, SemanticNACK):
        return ()
    values: list[str] = []
    if include_hard:
        values.extend(feedback.hard_missing_semantics)
    if include_soft:
        values.extend(feedback.soft_missing_semantics)
    return tuple(dict.fromkeys(values))


def nack_level(feedback: SemanticFeedback) -> str:
    if isinstance(feedback, SemanticNACK):
        return feedback.level
    return ReceiverNeedLevel.SATISFIED.value


def _requirement_id(requirement: SemanticRequirement) -> str:
    if requirement.target_logical_id:
        return f"{requirement.kind}:{requirement.target_logical_id}"
    if requirement.required_type:
        return f"{requirement.kind}:{requirement.required_type}"
    return requirement.kind


def _requirement_severity(requirement: SemanticRequirement) -> str:
    severity = str(getattr(requirement, "severity", "hard")).strip().lower()
    if severity in {item.value for item in SemanticRequirementSeverity}:
        return severity
    return SemanticRequirementSeverity.HARD.value


def _available_support_node_ids(results: Iterable[SemanticCheckResult]) -> tuple[str, ...]:
    node_ids: list[str] = []
    for result in results:
        node_ids.extend(result.satisfying_node_ids)
        node_ids.extend(result.recoverable_from_node_ids)
    return tuple(dict.fromkeys(node_ids))
