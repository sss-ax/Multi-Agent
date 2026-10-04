"""Receiver-side semantic need diagnosis.

This module turns concrete resolver results into a receiver-owned description
of what is still missing.  It is intentionally read-only: it does not choose a
sender packet, mutate graph state, or grant visibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .semantic_contract import (
    SemanticCheckResult,
    SemanticRequirement,
    SemanticRequirementSeverity,
    SemanticStatus,
)


class ReceiverNeedLevel(str, Enum):
    SATISFIED = "satisfied"
    HARD = "hard"
    VERIFICATION = "verification"
    QUALITY = "quality"


@dataclass(frozen=True)
class ReceiverNeed:
    sender: str
    receiver: str
    stage: str
    round_index: int
    hard_missing: tuple[str, ...] = ()
    verification_missing: tuple[str, ...] = ()
    quality_gaps: tuple[str, ...] = ()
    satisfied_semantics: tuple[str, ...] = ()
    recoverable_semantics: tuple[str, ...] = ()
    available_support_node_ids: tuple[str, ...] = ()
    available_capabilities: tuple[str, ...] = ()
    requested_fragments: tuple[str, ...] = ()
    need_sources: dict[str, str] | None = None
    uncertainty: float = 0.0

    @property
    def is_satisfied(self) -> bool:
        return not self.hard_missing and not self.verification_missing and not self.quality_gaps

    @property
    def has_hard_missing(self) -> bool:
        return bool(self.hard_missing)

    @property
    def has_verification_missing(self) -> bool:
        return bool(self.verification_missing)

    @property
    def has_quality_gap(self) -> bool:
        return bool(self.quality_gaps)

    @property
    def level(self) -> ReceiverNeedLevel:
        if self.hard_missing:
            return ReceiverNeedLevel.HARD
        if self.verification_missing:
            return ReceiverNeedLevel.VERIFICATION
        if self.quality_gaps:
            return ReceiverNeedLevel.QUALITY
        return ReceiverNeedLevel.SATISFIED

    def as_dict(self) -> dict[str, object]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "stage": self.stage,
            "round_index": self.round_index,
            "level": self.level.value,
            "is_satisfied": self.is_satisfied,
            "hard_missing": list(self.hard_missing),
            "verification_missing": list(self.verification_missing),
            "quality_gaps": list(self.quality_gaps),
            "satisfied_semantics": list(self.satisfied_semantics),
            "recoverable_semantics": list(self.recoverable_semantics),
            "available_support_node_ids": list(self.available_support_node_ids),
            "available_capabilities": list(self.available_capabilities),
            "requested_fragments": list(self.requested_fragments),
            "need_sources": dict(self.need_sources or {}),
            "uncertainty": self.uncertainty,
        }


def diagnose_receiver_need(
    *,
    sender: str,
    receiver: str,
    round_index: int,
    results: Iterable[SemanticCheckResult],
    stage: str = "",
    requested_fragments: Iterable[str] = (),
    uncertainty: float = 0.0,
) -> ReceiverNeed:
    """Build a receiver-owned need description from resolver results."""

    checked = tuple(results)
    hard_missing: list[str] = []
    verification_missing: list[str] = []
    quality_gaps: list[str] = []
    satisfied: list[str] = []
    recoverable: list[str] = []
    support_nodes: list[str] = []
    available_capabilities: list[str] = []
    need_sources: dict[str, str] = {}

    for result in checked:
        requirement = result.requirement
        if result.status == SemanticStatus.SATISFIED:
            satisfied.append(requirement.kind)
            available_capabilities.extend(_requirement_capabilities(requirement))
        elif result.status == SemanticStatus.RECOVERABLE:
            recoverable.append(requirement.kind)
            available_capabilities.extend(_requirement_capabilities(requirement))
        elif result.status == SemanticStatus.MISSING:
            severity = _severity(requirement)
            if severity == SemanticRequirementSeverity.HARD.value:
                hard_missing.append(requirement.kind)
                need_sources[requirement.kind] = "hard_contract"
            elif severity == SemanticRequirementSeverity.VERIFICATION.value:
                verification_missing.append(requirement.kind)
                need_sources[requirement.kind] = "verification_contract"
            else:
                quality_gaps.append(requirement.kind)
                need_sources[requirement.kind] = "quality_contract"
        support_nodes.extend(result.satisfying_node_ids)
        support_nodes.extend(result.recoverable_from_node_ids)

    return ReceiverNeed(
        sender=sender.strip(),
        receiver=receiver.strip(),
        stage=stage.strip() or f"{sender.strip()}->{receiver.strip()}",
        round_index=max(0, int(round_index)),
        hard_missing=tuple(dict.fromkeys(hard_missing)),
        verification_missing=tuple(dict.fromkeys(verification_missing)),
        quality_gaps=tuple(dict.fromkeys(quality_gaps)),
        satisfied_semantics=tuple(dict.fromkeys(satisfied)),
        recoverable_semantics=tuple(dict.fromkeys(recoverable)),
        available_support_node_ids=tuple(dict.fromkeys(support_nodes)),
        available_capabilities=tuple(dict.fromkeys(available_capabilities)),
        requested_fragments=tuple(dict.fromkeys(str(item) for item in requested_fragments if str(item).strip())),
        need_sources=need_sources,
        uncertainty=max(0.0, min(1.0, float(uncertainty))),
    )


def _severity(requirement: SemanticRequirement) -> str:
    severity = str(getattr(requirement, "severity", SemanticRequirementSeverity.HARD.value)).strip().lower()
    if severity in {item.value for item in SemanticRequirementSeverity}:
        return severity
    return SemanticRequirementSeverity.HARD.value


def _requirement_capabilities(requirement: SemanticRequirement) -> tuple[str, ...]:
    values = [requirement.kind, requirement.requirement_id]
    values.extend(requirement.candidate_logical_ids())
    values.extend(requirement.candidate_types())
    return tuple(dict.fromkeys(value for value in values if value))
