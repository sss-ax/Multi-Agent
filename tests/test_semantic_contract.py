from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from workflow_runtime import (
    SemanticCheckResult,
    SemanticRequirement,
    SemanticStatus,
    default_semantic_contract,
    semantic_requirements_for,
)


def test_semantic_requirement_cleans_and_deduplicates_candidates() -> None:
    requirement = SemanticRequirement(
        kind=" final_candidate ",
        target_logical_id=" result ",
        required_type=" result ",
        acceptable_logical_ids=("code", " result ", "code", ""),
        acceptable_types=("code", " result ", "code", ""),
    )

    assert requirement.kind == "final_candidate"
    assert requirement.target_logical_id == "result"
    assert requirement.required_type == "result"
    assert requirement.candidate_logical_ids() == ("result", "code")
    assert requirement.candidate_types() == ("result", "code")


def test_semantic_requirement_rejects_empty_kind_and_bad_count() -> None:
    with pytest.raises(ValueError, match="kind"):
        SemanticRequirement(kind="")
    with pytest.raises(ValueError, match="min_count"):
        SemanticRequirement(kind="candidate", min_count=0)


def test_default_semantic_contract_for_numeric_stage_pairs() -> None:
    planner_solver = default_semantic_contract("planner", "solver", task_type="numeric_solve")
    assert [item.kind for item in planner_solver.requirements] == ["plan_or_operation", "task_inputs"]

    solver_critic = semantic_requirements_for("solver", "critic", task_type="numeric_solve")
    assert [item.kind for item in solver_critic] == ["candidate_answer_or_code", "support_dependencies"]

    critic_final = default_semantic_contract("critic", "final_solver", task_type="numeric_solve")
    assert [item.kind for item in critic_final.requirements] == ["final_candidate", "validation_signal"]


def test_code_generation_contract_uses_code_candidate() -> None:
    solver_critic = default_semantic_contract("solver", "critic", task_type="code_generation")
    assert [item.kind for item in solver_critic.requirements] == ["candidate_code"]
    assert solver_critic.requirements[0].candidate_logical_ids() == ("code",)
    assert solver_critic.requirements[0].candidate_types() == ("code",)

    critic_final = default_semantic_contract("critic", "final_solver", task_type="code_generation")
    assert critic_final.requirements[0].kind == "final_candidate"
    assert critic_final.requirements[0].candidate_logical_ids() == ("code",)
    assert critic_final.requirements[1].kind == "validation_signal"


def test_planner_solver_contract_is_task_schema_specific() -> None:
    multiple_choice = default_semantic_contract("planner", "solver", task_type="multiple_choice")
    assert multiple_choice.requirements[1].candidate_types() == ("choice",)

    multihop = default_semantic_contract("planner", "solver", task_type="multihop_qa")
    assert set(multihop.requirements[1].candidate_types()) == {
        "supporting_fact",
        "entity",
        "evidence_link",
        "evidence",
    }

    code = default_semantic_contract("planner", "solver", task_type="code_generation")
    assert code.requirements[1].candidate_logical_ids() == ("requirements",)
    assert code.requirements[1].candidate_types() == ("requirements", "test")


def test_semantic_check_result_sufficiency_is_strictly_status_based() -> None:
    requirement = SemanticRequirement(kind="candidate_answer_or_code")
    assert SemanticCheckResult(requirement, SemanticStatus.SATISFIED).is_sufficient
    assert SemanticCheckResult(requirement, SemanticStatus.RECOVERABLE).is_sufficient
    assert not SemanticCheckResult(requirement, SemanticStatus.MISSING).is_sufficient
