"""Topology-specific HotpotQA reasoning candidates and verifier results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


REASONING_TYPES = (
    "bridge_entity",
    "bridge_attribute",
    "comparison_attribute",
    "shared_property_yesno",
    "set_intersection",
    "multi_constraint_lookup",
)


@dataclass(frozen=True)
class ReasoningCandidate:
    candidate_id: str
    reasoning_type: str
    evidence_bindings: dict[str, Any]
    derived_decision: dict[str, Any]
    construction_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReasoningIR:
    """Unified semantic proposal consumed by the programmatic verifier."""

    topology: dict[str, Any]
    question: dict[str, Any]
    constraints: list[dict[str, Any]]
    facts: dict[str, Any]
    bindings: dict[str, Any]
    derived: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationResult:
    reasoning_type: str
    evidence_valid: bool
    reasoning_complete: bool
    answer_supported: bool
    derived_answer: str
    confidence: float
    score: float
    failure_code: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))
