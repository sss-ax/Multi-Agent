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
    SemanticStatus,
    build_semantic_feedback,
    is_ack,
    is_nack,
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
    assert result.as_dict()["type"] == "ACK"


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
    assert result.available_support_node_ids == ("verification@v1",)
    assert result.as_dict() == {
        "type": "NACK",
        "sender": "critic",
        "receiver": "final_solver",
        "round_index": 1,
        "missing_requirement_ids": ["final_candidate:result"],
        "missing_semantics": ["final_candidate"],
        "available_support_node_ids": ["verification@v1"],
    }


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
