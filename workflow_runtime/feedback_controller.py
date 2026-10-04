"""Protocol-level routing for typed semantic feedback."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .semantic_feedback import (
    SemanticFeedback,
    is_ack,
    is_hard_nack,
    is_quality_nack,
    is_verification_nack,
    nack_level,
    nack_missing_semantics,
)


class FeedbackAction(str, Enum):
    STOP = "STOP"
    TARGETED_REFINEMENT = "TARGETED_REFINEMENT"
    SEND_ALL_FALLBACK = "SEND_ALL_FALLBACK"
    REPORT_INSUFFICIENT = "REPORT_INSUFFICIENT"


@dataclass(frozen=True)
class FeedbackDecision:
    action: FeedbackAction
    policy: str
    level: str
    round_index: int
    target_semantics: tuple[str, ...] = ()
    allow_fallback: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "policy": self.policy,
            "level": self.level,
            "round_index": self.round_index,
            "target_semantics": list(self.target_semantics),
            "allow_fallback": self.allow_fallback,
            "reason": self.reason,
        }


def decide_feedback_action(
    feedback: SemanticFeedback,
    *,
    policy_name: str,
    round_index: int,
    max_rounds: int,
) -> FeedbackDecision:
    """Map typed feedback to the next protocol action.

    Hard missing semantics are correctness failures: targeted feedback may try
    to repair them, and send-all fallback is allowed only for them.
    Verification and quality requests are soft control signals: they may drive
    targeted packets, but they never authorize send-all fallback.
    """

    level = nack_level(feedback)
    targets = nack_missing_semantics(feedback)
    if is_ack(feedback):
        return FeedbackDecision(
            action=FeedbackAction.STOP,
            policy=policy_name,
            level=level,
            round_index=round_index,
            reason="semantic contract satisfied",
        )
    if policy_name == "minimal_no_feedback":
        return FeedbackDecision(
            action=FeedbackAction.REPORT_INSUFFICIENT,
            policy=policy_name,
            level=level,
            round_index=round_index,
            target_semantics=targets,
            reason="policy has no feedback channel",
        )
    if policy_name == "minimal_sendall_fallback":
        if is_hard_nack(feedback):
            return FeedbackDecision(
                action=FeedbackAction.SEND_ALL_FALLBACK,
                policy=policy_name,
                level=level,
                round_index=round_index,
                target_semantics=targets,
                allow_fallback=True,
                reason="hard missing semantics authorize explicit send-all fallback",
            )
        return FeedbackDecision(
            action=FeedbackAction.REPORT_INSUFFICIENT,
            policy=policy_name,
            level=level,
            round_index=round_index,
            target_semantics=targets,
            reason="soft feedback request does not authorize send-all fallback",
        )
    if policy_name == "minimal_targeted_feedback":
        if round_index < max_rounds:
            if is_hard_nack(feedback):
                return FeedbackDecision(
                    action=FeedbackAction.TARGETED_REFINEMENT,
                    policy=policy_name,
                    level=level,
                    round_index=round_index,
                    target_semantics=targets,
                    allow_fallback=True,
                    reason="hard missing semantics require targeted deterministic repair",
                )
            if is_verification_nack(feedback):
                return FeedbackDecision(
                    action=FeedbackAction.TARGETED_REFINEMENT,
                    policy=policy_name,
                    level=level,
                    round_index=round_index,
                    target_semantics=targets,
                    reason="verification request may be repaired by targeted fragments",
                )
            if is_quality_nack(feedback):
                return FeedbackDecision(
                    action=FeedbackAction.TARGETED_REFINEMENT,
                    policy=policy_name,
                    level=level,
                    round_index=round_index,
                    target_semantics=targets,
                    reason="quality request may be repaired by utility-controlled targeted fragments",
                )
        if is_hard_nack(feedback):
            return FeedbackDecision(
                action=FeedbackAction.SEND_ALL_FALLBACK,
                policy=policy_name,
                level=level,
                round_index=round_index,
                target_semantics=targets,
                allow_fallback=True,
                reason="hard missing semantics exceeded targeted rounds",
            )
        return FeedbackDecision(
            action=FeedbackAction.REPORT_INSUFFICIENT,
            policy=policy_name,
            level=level,
            round_index=round_index,
            target_semantics=targets,
            reason="soft feedback request exhausted targeted rounds without send-all fallback",
        )
    return FeedbackDecision(
        action=FeedbackAction.REPORT_INSUFFICIENT,
        policy=policy_name,
        level=level,
        round_index=round_index,
        target_semantics=targets,
        reason="policy does not implement typed feedback handling",
    )
