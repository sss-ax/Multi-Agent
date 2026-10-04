from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_runtime import (
    AgentGraphViewManager,
    GraphStore,
    ReceiverNeedLevel,
    SemanticCheckResult,
    SemanticRequirement,
    SemanticRequirementSeverity,
    SemanticResolver,
    SemanticStatus,
    default_semantic_contract,
    diagnose_receiver_need,
)


def test_receiver_need_classifies_hard_verification_and_quality_missing() -> None:
    need = diagnose_receiver_need(
        sender="solver",
        receiver="critic",
        stage="solver->critic",
        round_index=2,
        results=[
            SemanticCheckResult(
                requirement=SemanticRequirement(kind="candidate_answer_or_code"),
                status=SemanticStatus.MISSING,
            ),
            SemanticCheckResult(
                requirement=SemanticRequirement(
                    kind="support_dependencies",
                    severity=SemanticRequirementSeverity.VERIFICATION.value,
                ),
                status=SemanticStatus.MISSING,
            ),
            SemanticCheckResult(
                requirement=SemanticRequirement(
                    kind="full_feedback",
                    severity=SemanticRequirementSeverity.REFINEMENT.value,
                ),
                status=SemanticStatus.MISSING,
            ),
        ],
    )

    assert need.level == ReceiverNeedLevel.HARD
    assert need.hard_missing == ("candidate_answer_or_code",)
    assert need.verification_missing == ("support_dependencies",)
    assert need.quality_gaps == ("full_feedback",)
    assert need.need_sources == {
        "candidate_answer_or_code": "hard_contract",
        "support_dependencies": "verification_contract",
        "full_feedback": "quality_contract",
    }
    assert not need.is_satisfied


def test_receiver_need_reports_verification_level_when_only_soft_missing() -> None:
    need = diagnose_receiver_need(
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

    assert need.level == ReceiverNeedLevel.VERIFICATION
    assert need.hard_missing == ()
    assert need.verification_missing == ("support_dependencies",)
    assert need.available_support_node_ids == ("result@v1",)
    assert "candidate_answer_or_code" in need.available_capabilities
    assert need.as_dict()["stage"] == "solver->critic"
    assert need.as_dict()["need_sources"] == {"support_dependencies": "verification_contract"}


def test_receiver_need_is_deterministic_for_identical_results() -> None:
    results = [
        SemanticCheckResult(
            requirement=SemanticRequirement(kind="final_candidate"),
            status=SemanticStatus.RECOVERABLE,
            recoverable_from_node_ids=("result@v1",),
        )
    ]

    first = diagnose_receiver_need(sender="critic", receiver="final_solver", round_index=1, results=results)
    second = diagnose_receiver_need(sender="critic", receiver="final_solver", round_index=1, results=results)

    assert first == second
    assert first.level == ReceiverNeedLevel.SATISFIED
    assert first.is_satisfied
    assert first.recoverable_semantics == ("final_candidate",)
    assert "final_candidate" in first.available_capabilities


def test_required_core_missing_becomes_hard_missing() -> None:
    need = diagnose_receiver_need(
        sender="critic",
        receiver="final_solver",
        round_index=0,
        results=[
            SemanticCheckResult(
                requirement=SemanticRequirement(kind="final_candidate", target_logical_id="result"),
                status=SemanticStatus.MISSING,
            )
        ],
    )

    assert need.level == ReceiverNeedLevel.HARD
    assert need.hard_missing == ("final_candidate",)
    assert need.need_sources == {"final_candidate": "hard_contract"}


def test_optional_full_feedback_missing_defaults_to_quality_not_hard() -> None:
    need = diagnose_receiver_need(
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
            )
        ],
    )

    assert need.level == ReceiverNeedLevel.QUALITY
    assert need.hard_missing == ()
    assert need.quality_gaps == ("full_feedback",)
    assert need.need_sources == {"full_feedback": "quality_contract"}


def test_multiple_choice_seed_choice_does_not_create_false_facts_hard_need() -> None:
    store = GraphStore()
    plan = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="plan",
        node_type="plan",
        content={"goal": "choose answer"},
        owner="planner",
    )
    choice = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="choice_1",
        node_type="choice",
        content={"label": "A", "text": "option"},
        owner="user",
    )
    views = AgentGraphViewManager(store, task_id="t", branch_id="main", agents=("solver",))
    views.grant("solver", node_ids=[plan.node_id, choice.node_id])

    contract = default_semantic_contract("planner", "solver", task_type="multiple_choice")
    results = SemanticResolver(store.snapshot(), views.view("solver")).check_contract(contract)
    need = diagnose_receiver_need(sender="planner", receiver="solver", round_index=0, results=results)

    assert need.is_satisfied
    assert need.hard_missing == ()
    assert "facts" not in need.hard_missing
    assert "choice" in need.available_capabilities


def test_multihop_seed_evidence_does_not_create_false_facts_hard_need() -> None:
    store = GraphStore()
    plan = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="plan",
        node_type="plan",
        content={"goal": "use evidence"},
        owner="planner",
    )
    evidence = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="supporting_fact_1",
        node_type="supporting_fact",
        content={"title": "T", "text": "evidence"},
        owner="user",
    )
    views = AgentGraphViewManager(store, task_id="t", branch_id="main", agents=("solver",))
    views.grant("solver", node_ids=[plan.node_id, evidence.node_id])

    contract = default_semantic_contract("planner", "solver", task_type="multihop_qa")
    results = SemanticResolver(store.snapshot(), views.view("solver")).check_contract(contract)
    need = diagnose_receiver_need(sender="planner", receiver="solver", round_index=0, results=results)

    assert need.is_satisfied
    assert need.hard_missing == ()
    assert "facts" not in need.hard_missing
    assert "supporting_fact" in need.available_capabilities
