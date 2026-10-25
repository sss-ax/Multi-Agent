"""Leakage-aware schemas for Phase 2A and Phase 2B training artifacts."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..communication_action import CommunicationAction


@dataclass(frozen=True)
class CurrentAnswerLabel:
    """Oracle training label for the candidate visible at this decision point."""

    correct: bool | None
    candidate_node_id: str = ""
    label_source: str = ""
    error_type: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "correct": self.correct,
            "candidate_node_id": self.candidate_node_id,
            "label_source": self.label_source,
            "error_type": self.error_type,
        }


@dataclass(frozen=True)
class FinalOutcomeLabel:
    """Policy-conditional terminal success label, distinct from candidate truth."""

    success: bool | None
    policy_name: str
    evaluator_source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "policy_name": self.policy_name,
            "evaluator_source": self.evaluator_source,
        }


@dataclass(frozen=True)
class CounterfactualOutcome:
    action: CommunicationAction
    legal: bool
    current_answer: CurrentAnswerLabel
    final_outcome: FinalOutcomeLabel
    communication_tokens: int = 0
    model_input_tokens: int = 0
    model_output_tokens: int = 0
    round_count: int = 0
    policy_cost_tokens: int = 0
    final_answer: Any = None
    failure_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_cost_tokens(self) -> int:
        return (
            self.communication_tokens
            + self.model_input_tokens
            + self.model_output_tokens
            + self.policy_cost_tokens
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.as_dict(),
            "legal": self.legal,
            "current_answer": self.current_answer.as_dict(),
            "final_outcome": self.final_outcome.as_dict(),
            "communication_tokens": self.communication_tokens,
            "model_input_tokens": self.model_input_tokens,
            "model_output_tokens": self.model_output_tokens,
            "round_count": self.round_count,
            "policy_cost_tokens": self.policy_cost_tokens,
            "total_cost_tokens": self.total_cost_tokens,
            "final_answer": copy.deepcopy(self.final_answer),
            "failure_reason": self.failure_reason,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class DualRiskExample:
    """One state with separate Verify-Head and Value-Head targets."""

    snapshot_id: str
    task_id: str
    state_features: dict[str, Any]
    policy_features: dict[str, Any]
    current_answer: CurrentAnswerLabel
    final_outcome: FinalOutcomeLabel

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "task_id": self.task_id,
            "state_features": copy.deepcopy(self.state_features),
            "policy_features": copy.deepcopy(self.policy_features),
            "labels": {
                "current_answer": self.current_answer.as_dict(),
                "final_outcome": self.final_outcome.as_dict(),
            },
        }


@dataclass(frozen=True)
class TeacherSearchResult:
    snapshot_id: str
    phase: str
    status: str
    teacher_outcomes: tuple[CounterfactualOutcome, ...]
    evaluated_outcomes: tuple[CounterfactualOutcome, ...]
    task_id: str = ""
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "task_id": self.task_id,
            "phase": self.phase,
            "status": self.status,
            "reason": self.reason,
            "teacher_outcomes": [item.as_dict() for item in self.teacher_outcomes],
            "evaluated_outcomes": [item.as_dict() for item in self.evaluated_outcomes],
        }


class CounterfactualRolloutRunner:
    """Structural protocol implemented by end-to-end continuation runners."""

    def run(self, action: CommunicationAction) -> CounterfactualOutcome:  # pragma: no cover - protocol stub
        raise NotImplementedError


def ensure_same_current_label(outcomes: tuple[CounterfactualOutcome, ...]) -> CurrentAnswerLabel:
    """A pre-action candidate label must not change across counterfactual actions."""
    if not outcomes:
        return CurrentAnswerLabel(correct=None)
    first = outcomes[0].current_answer
    for outcome in outcomes[1:]:
        if outcome.current_answer != first:
            raise ValueError("counterfactual actions changed the pre-action current-answer label")
    return first
