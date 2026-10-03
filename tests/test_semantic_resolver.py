from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_runtime import (
    AgentGraphViewManager,
    GraphStore,
    SemanticRequirement,
    SemanticStatus,
    all_requirements_sufficient,
    check_semantic_contract,
    default_semantic_contract,
    missing_requirements,
)


def _store_and_views() -> tuple[GraphStore, AgentGraphViewManager]:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="2+3", owner="user")
    views = AgentGraphViewManager(store, task_id="t", branch_id="main", agents=("solver", "critic", "final_solver"))
    return store, views


def test_visible_matching_node_satisfies_requirement() -> None:
    store, views = _store_and_views()
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 5},
        owner="solver",
    )
    views.grant("critic", node_ids=[result.node_id])

    contract = default_semantic_contract("solver", "critic", task_type="numeric_solve")
    results = check_semantic_contract(store.snapshot(), views.view("critic"), contract)

    candidate = next(item for item in results if item.requirement.kind == "candidate_answer_or_code")
    assert candidate.status == SemanticStatus.SATISFIED
    assert candidate.satisfying_node_ids == (result.node_id,)


def test_existing_but_invisible_node_is_missing() -> None:
    store, views = _store_and_views()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 5},
        owner="solver",
    )

    results = check_semantic_contract(
        store.snapshot(),
        views.view("critic"),
        default_semantic_contract("solver", "critic", task_type="numeric_solve"),
    )
    candidate = next(item for item in results if item.requirement.kind == "candidate_answer_or_code")
    assert candidate.status == SemanticStatus.MISSING


def test_candidate_answer_can_be_deterministically_recovered_from_visible_calculation() -> None:
    store, views = _store_and_views()
    calculation = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="calculation",
        node_type="calculation",
        content={"id": "R1", "expression": "2+3", "value": 5},
        owner="solver",
    )
    views.grant("critic", node_ids=[calculation.node_id])

    contract = default_semantic_contract("solver", "critic", task_type="numeric_solve")
    results = check_semantic_contract(store.snapshot(), views.view("critic"), contract)
    candidate = next(item for item in results if item.requirement.kind == "candidate_answer_or_code")

    assert candidate.status == SemanticStatus.RECOVERABLE
    assert candidate.recoverable_from_node_ids == (calculation.node_id,)


def test_plan_does_not_make_result_recoverable() -> None:
    store, views = _store_and_views()
    plan = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="plan",
        node_type="plan",
        content={"goal": "compute 2+3"},
        owner="planner",
    )
    views.grant("critic", node_ids=[plan.node_id])

    contract = default_semantic_contract("solver", "critic", task_type="numeric_solve")
    results = check_semantic_contract(store.snapshot(), views.view("critic"), contract)
    candidate = next(item for item in results if item.requirement.kind == "candidate_answer_or_code")

    assert candidate.status == SemanticStatus.MISSING


def test_multiple_choice_seed_choice_satisfies_planner_solver_task_inputs() -> None:
    store, views = _store_and_views()
    plan = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="plan",
        node_type="plan",
        content={"goal": "choose option"},
        owner="planner",
    )
    choice = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="choice_1",
        node_type="choice",
        content={"label": "A", "text": "alpha"},
        owner="dataset",
        created_by_role="dataset",
    )
    views.grant("solver", node_ids=[plan.node_id, choice.node_id])

    contract = default_semantic_contract("planner", "solver", task_type="multiple_choice")
    results = check_semantic_contract(store.snapshot(), views.view("solver"), contract)
    task_inputs = next(item for item in results if item.requirement.kind == "task_inputs")

    assert task_inputs.status == SemanticStatus.SATISFIED
    assert task_inputs.satisfying_node_ids == (choice.node_id,)


def test_multihop_seed_evidence_satisfies_planner_solver_task_inputs() -> None:
    store, views = _store_and_views()
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
        owner="dataset",
        created_by_role="dataset",
    )
    views.grant("solver", node_ids=[plan.node_id, evidence.node_id])

    contract = default_semantic_contract("planner", "solver", task_type="multihop_qa")
    results = check_semantic_contract(store.snapshot(), views.view("solver"), contract)
    task_inputs = next(item for item in results if item.requirement.kind == "task_inputs")

    assert task_inputs.status == SemanticStatus.SATISFIED
    assert task_inputs.satisfying_node_ids == (evidence.node_id,)


def test_missing_requirements_and_sufficiency_helpers() -> None:
    store, views = _store_and_views()
    contract = default_semantic_contract("critic", "final_solver", task_type="numeric_solve")
    results = check_semantic_contract(store.snapshot(), views.view("final_solver"), contract)

    assert not all_requirements_sufficient(results)
    assert [item.kind for item in missing_requirements(results)] == ["final_candidate", "validation_signal"]
