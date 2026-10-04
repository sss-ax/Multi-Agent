from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_runtime import (
    FeedbackAction,
    SemanticCheckResult,
    SemanticRequirement,
    SemanticRequirementSeverity,
    SemanticStatus,
    build_semantic_feedback,
    decide_feedback_action,
)


def _feedback(kind: str, *, severity: str = "hard"):
    return build_semantic_feedback(
        sender="solver",
        receiver="critic",
        round_index=0,
        results=[
            SemanticCheckResult(
                requirement=SemanticRequirement(kind=kind, severity=severity),
                status=SemanticStatus.MISSING,
            )
        ],
    )


def test_phase4_hard_nack_authorizes_sendall_only_for_fallback_policy() -> None:
    feedback = _feedback("candidate_answer_or_code")

    decision = decide_feedback_action(
        feedback,
        policy_name="minimal_sendall_fallback",
        round_index=0,
        max_rounds=2,
    )

    assert decision.action == FeedbackAction.SEND_ALL_FALLBACK
    assert decision.allow_fallback is True
    assert decision.target_semantics == ("candidate_answer_or_code",)


def test_phase4_verification_request_never_authorizes_sendall_fallback() -> None:
    feedback = _feedback("support_dependencies", severity=SemanticRequirementSeverity.VERIFICATION.value)

    fallback_decision = decide_feedback_action(
        feedback,
        policy_name="minimal_sendall_fallback",
        round_index=0,
        max_rounds=2,
    )
    targeted_decision = decide_feedback_action(
        feedback,
        policy_name="minimal_targeted_feedback",
        round_index=0,
        max_rounds=2,
    )

    assert fallback_decision.action == FeedbackAction.REPORT_INSUFFICIENT
    assert fallback_decision.allow_fallback is False
    assert targeted_decision.action == FeedbackAction.TARGETED_REFINEMENT
    assert targeted_decision.allow_fallback is False


def test_phase4_quality_request_is_utility_controlled_not_sendall_controlled() -> None:
    feedback = _feedback("full_feedback", severity=SemanticRequirementSeverity.REFINEMENT.value)

    fallback_decision = decide_feedback_action(
        feedback,
        policy_name="minimal_sendall_fallback",
        round_index=0,
        max_rounds=2,
    )
    targeted_decision = decide_feedback_action(
        feedback,
        policy_name="minimal_targeted_feedback",
        round_index=0,
        max_rounds=2,
    )

    assert fallback_decision.action == FeedbackAction.REPORT_INSUFFICIENT
    assert fallback_decision.allow_fallback is False
    assert targeted_decision.action == FeedbackAction.TARGETED_REFINEMENT
    assert targeted_decision.allow_fallback is False


def test_phase4_soft_request_exhaustion_reports_insufficient_without_fallback() -> None:
    feedback = _feedback("support_dependencies", severity=SemanticRequirementSeverity.VERIFICATION.value)

    decision = decide_feedback_action(
        feedback,
        policy_name="minimal_targeted_feedback",
        round_index=2,
        max_rounds=2,
    )

    assert decision.action == FeedbackAction.REPORT_INSUFFICIENT
    assert decision.allow_fallback is False


def test_phase4_hard_request_exhaustion_degrades_explicitly_to_sendall() -> None:
    feedback = _feedback("candidate_answer_or_code")

    decision = decide_feedback_action(
        feedback,
        policy_name="minimal_targeted_feedback",
        round_index=2,
        max_rounds=2,
    )

    assert decision.action == FeedbackAction.SEND_ALL_FALLBACK
    assert decision.allow_fallback is True
