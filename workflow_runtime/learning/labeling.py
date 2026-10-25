"""Explicit oracle labelers that never feed their outputs back into state features."""

from __future__ import annotations

from typing import Any, Callable

from ..communication_replay import CommunicationDecisionSnapshot
from .schemas import CurrentAnswerLabel


CandidateEvaluator = Callable[[Any], bool | None]


def current_candidate(snapshot: CommunicationDecisionSnapshot):
    """Return the latest pre-action candidate, excluding any finalizer output."""
    preferred_types = {"result", "code", "execution"}
    candidates = [
        node for node in snapshot.graph_state.nodes.values()
        if node.task_id == snapshot.task_id
        and node.branch_id == snapshot.branch_id
        and node.type in preferred_types
        and node.is_operationally_valid()
    ]
    return max(
        candidates,
        key=lambda node: (node.created_at, node.version, node.node_id),
        default=None,
    )


def label_current_answer(
    snapshot: CommunicationDecisionSnapshot,
    evaluator: CandidateEvaluator,
    *,
    label_source: str,
    error_type: str = "",
) -> CurrentAnswerLabel:
    candidate = current_candidate(snapshot)
    if candidate is None:
        return CurrentAnswerLabel(
            correct=None,
            label_source=label_source,
            error_type="candidate_unavailable",
        )
    content = candidate.content
    value = content.get("value") if isinstance(content, dict) and "value" in content else content
    correct = evaluator(value)
    return CurrentAnswerLabel(
        correct=None if correct is None else bool(correct),
        candidate_node_id=candidate.node_id,
        label_source=label_source,
        error_type=error_type if correct is False else "",
    )
