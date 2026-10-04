from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_runtime import (
    SemanticACK,
    SemanticCheckResult,
    SemanticNACK,
    SemanticRequirement,
    SemanticRequirementSeverity,
    SemanticStatus,
    build_semantic_feedback,
    is_ack,
    is_quality_nack,
    is_nack,
    is_verification_nack,
)


def test_build_semantic_feedback_returns_ack_when_all_requirements_are_sufficient() -> None:
    result = build_semantic_feedback(
        sender="solver",
        receiver="critic",
        round_index=0,
        results=[
            SemanticCheckResult(
                requirement=SemanticRequirement(kind="candidate_answer_or_code"),
                status=SemanticStatus.SATISFIED,
                satisfying_node_ids=("result@v1",),
            ),
            SemanticCheckResult(
                requirement=SemanticRequirement(kind="support_dependencies"),
                status=SemanticStatus.RECOVERABLE,
                recoverable_from_node_ids=("calculation@v1",),
            ),
        ],
    )

    assert isinstance(result, SemanticACK)
    assert is_ack(result)
    assert not is_nack(result)
    assert result.satisfied_requirements == ("candidate_answer_or_code",)
    assert result.recoverable_requirements == ("support_dependencies",)
    assert result.as_dict()["type"] == "NONE"


def test_build_semantic_feedback_returns_nack_for_missing_semantics() -> None:
    result = build_semantic_feedback(
        sender="critic",
        receiver="final_solver",
        round_index=1,
        results=[
            SemanticCheckResult(
                requirement=SemanticRequirement(kind="final_candidate", target_logical_id="result"),
                status=SemanticStatus.MISSING,
            ),
            SemanticCheckResult(
                requirement=SemanticRequirement(kind="validation_signal", target_logical_id="verification"),
                status=SemanticStatus.SATISFIED,
                satisfying_node_ids=("verification@v1",),
            ),
        ],
    )

    assert isinstance(result, SemanticNACK)
    assert is_nack(result)
    assert not is_ack(result)
    assert result.missing_requirement_ids == ("final_candidate:result",)
    assert result.missing_semantics == ("final_candidate",)
    assert result.level == "hard"
    assert result.available_support_node_ids == ("verification@v1",)
    data = result.as_dict()
    assert data == {
        "type": "HARD_NACK",
        "sender": "critic",
        "receiver": "final_solver",
        "round_index": 1,
        "level": "hard",
        "missing_requirement_ids": ["final_candidate:result"],
        "missing_semantics": ["final_candidate"],
        "hard_missing_requirement_ids": ["final_candidate:result"],
        "hard_missing_semantics": ["final_candidate"],
        "verification_missing_requirement_ids": [],
        "verification_missing_semantics": [],
        "quality_gap_requirement_ids": [],
        "quality_gaps": [],
        "soft_missing_requirement_ids": [],
        "soft_missing_semantics": [],
        "has_hard_missing": True,
        "has_soft_missing": False,
        "available_support_node_ids": ["verification@v1"],
        "requested_fragments": [],
        "receiver_need": {
            "sender": "critic",
            "receiver": "final_solver",
            "stage": "critic->final_solver",
            "round_index": 1,
            "level": "hard",
            "is_satisfied": False,
            "hard_missing": ["final_candidate"],
            "verification_missing": [],
            "quality_gaps": [],
            "satisfied_semantics": ["validation_signal"],
            "recoverable_semantics": [],
            "available_support_node_ids": ["verification@v1"],
            "available_capabilities": [
                "validation_signal",
                "validation_signal:verification",
                "verification",
            ],
            "requested_fragments": [],
            "need_sources": {"final_candidate": "hard_contract"},
            "uncertainty": 0.0,
        },
    }


def test_build_semantic_feedback_returns_verification_nack_for_verification_missing() -> None:
    result = build_semantic_feedback(
        sender="solver",
        receiver="critic",
        round_index=0,
        results=[
            SemanticCheckResult(
                requirement=SemanticRequirement(kind="candidate_answer_or_code"),
                status=SemanticStatus.SATISFIED,
                satisfying_node_ids=("result@v1",),
            ),
            SemanticCheckResult(
                requirement=SemanticRequirement(
                    kind="support_dependencies",
                    severity=SemanticRequirementSeverity.VERIFICATION.value,
                ),
                status=SemanticStatus.MISSING,
            ),
        ],
    )

    assert isinstance(result, SemanticNACK)
    assert result.as_dict()["type"] == "VERIFICATION_REQUEST"
    assert result.level == "verification"
    assert not is_nack(result)
    assert is_verification_nack(result)
    assert result.verification_missing_semantics == ("support_dependencies",)
    assert result.soft_missing_semantics == ("support_dependencies",)
    assert result.hard_missing_semantics == ()
    assert result.as_dict()["receiver_need"]["level"] == "verification"


def test_build_semantic_feedback_returns_quality_nack_for_quality_gap() -> None:
    result = build_semantic_feedback(
        sender="critic",
        receiver="final_solver",
        round_index=0,
        results=[
            SemanticCheckResult(
                requirement=SemanticRequirement(
                    kind="full_feedback",
                    severity=SemanticRequirementSeverity.REFINEMENT.value,
                ),
                status=SemanticStatus.MISSING,
            ),
        ],
    )

    assert isinstance(result, SemanticNACK)
    assert result.as_dict()["type"] == "QUALITY_REQUEST"
    assert result.level == "quality"
    assert not is_nack(result)
    assert is_quality_nack(result)
    assert result.quality_gaps == ("full_feedback",)
    assert result.soft_missing_semantics == ("full_feedback",)


def test_code_missing_is_hard_nack() -> None:
    result = build_semantic_feedback(
        sender="solver",
        receiver="critic",
        round_index=0,
        results=[
            SemanticCheckResult(
                requirement=SemanticRequirement(kind="candidate_code", target_logical_id="code", required_type="code"),
                status=SemanticStatus.MISSING,
            )
        ],
    )

    assert isinstance(result, SemanticNACK)
    assert result.as_dict()["type"] == "HARD_NACK"
    assert is_nack(result)
    assert result.hard_missing_semantics == ("candidate_code",)


def test_critic_low_confidence_is_quality_request() -> None:
    result = build_semantic_feedback(
        sender="critic",
        receiver="solver",
        round_index=0,
        results=[
            SemanticCheckResult(
                requirement=SemanticRequirement(
                    kind="critic_confidence_low",
                    severity=SemanticRequirementSeverity.REFINEMENT.value,
                ),
                status=SemanticStatus.MISSING,
            )
        ],
    )

    assert isinstance(result, SemanticNACK)
    assert result.as_dict()["type"] == "QUALITY_REQUEST"
    assert is_quality_nack(result)
    assert not is_nack(result)


def test_nack_core_protocol_names_semantics_not_missing_node_ids() -> None:
    result = build_semantic_feedback(
        sender="planner",
        receiver="solver",
        round_index=0,
        results=[
            SemanticCheckResult(
                requirement=SemanticRequirement(kind="task_inputs", required_type="facts"),
                status=SemanticStatus.MISSING,
            )
        ],
    )

    assert isinstance(result, SemanticNACK)
    assert result.missing_semantics == ("task_inputs",)
    assert result.missing_requirement_ids == ("task_inputs:facts",)
    assert result.available_support_node_ids == ()
