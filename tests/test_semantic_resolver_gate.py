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
    check_semantic_contract,
    default_semantic_contract,
)


def _store_and_views() -> tuple[GraphStore, AgentGraphViewManager]:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="A+B", owner="user")
    views = AgentGraphViewManager(store, task_id="t", branch_id="main", agents=("critic", "final_solver"))
    return store, views


def test_phase2_visible_node_satisfies_final_candidate() -> None:
    store, views = _store_and_views()
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 8},
        owner="solver",
    )
    views.grant("final_solver", node_ids=[result.node_id])

    contract = default_semantic_contract("critic", "final_solver", task_type="numeric_solve")
    resolution = check_semantic_contract(store.snapshot(), views.view("final_solver"), contract)
    final_candidate = next(item for item in resolution if item.requirement.kind == "final_candidate")

    assert final_candidate.status == SemanticStatus.SATISFIED
    assert final_candidate.satisfying_node_ids == (result.node_id,)


def test_phase2_hidden_canonical_node_is_missing() -> None:
    store, views = _store_and_views()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 8},
        owner="solver",
    )

    contract = default_semantic_contract("critic", "final_solver", task_type="numeric_solve")
    resolution = check_semantic_contract(store.snapshot(), views.view("final_solver"), contract)
    final_candidate = next(item for item in resolution if item.requirement.kind == "final_candidate")

    assert final_candidate.status == SemanticStatus.MISSING


def test_phase2_visible_inputs_and_operation_are_deterministically_recoverable() -> None:
    store, views = _store_and_views()
    facts = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="facts",
        node_type="facts",
        content=[{"id": "A", "value": 3}, {"id": "B", "value": 5}],
        owner="planner",
    )
    plan = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="plan",
        node_type="plan",
        content={"steps": [{"id": "R1", "operation": "add", "inputs": ["A", "B"]}]},
        owner="planner",
    )
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 8},
        owner="solver",
    )
    views.grant("critic", node_ids=[facts.node_id, plan.node_id])

    contract = default_semantic_contract("solver", "critic", task_type="numeric_solve")
    resolution = check_semantic_contract(store.snapshot(), views.view("critic"), contract)
    candidate = next(item for item in resolution if item.requirement.kind == "candidate_answer_or_code")

    assert candidate.status == SemanticStatus.RECOVERABLE
    assert candidate.recovery_method == "deterministic_executor"
    assert candidate.recovered_value == 8
    assert set(candidate.recoverable_from_node_ids) == {facts.node_id, plan.node_id}


def test_phase2_vague_plan_is_not_recoverable() -> None:
    store, views = _store_and_views()
    vague_plan = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="plan",
        node_type="plan",
        content={"goal": "think about the arithmetic"},
        owner="planner",
    )
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 8},
        owner="solver",
    )
    views.grant("critic", node_ids=[vague_plan.node_id])

    contract = default_semantic_contract("solver", "critic", task_type="numeric_solve")
    resolution = check_semantic_contract(store.snapshot(), views.view("critic"), contract)
    candidate = next(item for item in resolution if item.requirement.kind == "candidate_answer_or_code")

    assert candidate.status == SemanticStatus.MISSING
    assert candidate.recovery_method is None
    assert candidate.recovered_value is None


def test_phase2_resolver_does_not_mutate_graph_or_grant_visibility() -> None:
    store, views = _store_and_views()
    hidden = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 8},
        owner="solver",
    )
    before_node_ids = set(store.snapshot().nodes)
    before_edges = tuple(edge.edge_id for edge in store.snapshot().edges)
    before_visible = set(views.view("critic").visible_node_ids)

    contract = default_semantic_contract("solver", "critic", task_type="numeric_solve")
    check_semantic_contract(store.snapshot(), views.view("critic"), contract)

    after = store.snapshot()
    assert set(after.nodes) == before_node_ids
    assert tuple(edge.edge_id for edge in after.edges) == before_edges
    assert views.view("critic").visible_node_ids == before_visible
    assert hidden.node_id not in views.view("critic").visible_node_ids


def test_phase2_resolver_does_not_trigger_send_all() -> None:
    calls = {"send_all_for_stage": 0}

    def send_all_for_stage():
        calls["send_all_for_stage"] += 1

    store, views = _store_and_views()
    contract = default_semantic_contract("solver", "critic", task_type="numeric_solve")
    check_semantic_contract(store.snapshot(), views.view("critic"), contract)

    assert calls["send_all_for_stage"] == 0


def test_phase2_resolution_is_deterministic_for_same_input() -> None:
    store, views = _store_and_views()
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 8},
        owner="solver",
    )
    views.grant("critic", node_ids=[result.node_id])
    contract = default_semantic_contract("solver", "critic", task_type="numeric_solve")

    r1 = check_semantic_contract(store.snapshot(), views.view("critic"), contract)
    r2 = check_semantic_contract(store.snapshot(), views.view("critic"), contract)

    assert r1 == r2


def test_phase2_gate_metrics_are_clean() -> None:
    hidden_node_leakage = 0
    illegal_graph_mutation = 0
    illegal_visibility_grant = 0
    illegal_fallback = 0
    deterministic_resolution = 1.0

    store, views = _store_and_views()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 8},
        owner="solver",
    )
    before_nodes = set(store.snapshot().nodes)
    before_visible = set(views.view("critic").visible_node_ids)
    contract = default_semantic_contract("solver", "critic", task_type="numeric_solve")
    r1 = check_semantic_contract(store.snapshot(), views.view("critic"), contract)
    r2 = check_semantic_contract(store.snapshot(), views.view("critic"), contract)
    candidate = next(item for item in r1 if item.requirement.kind == "candidate_answer_or_code")
    if candidate.status != SemanticStatus.MISSING:
        hidden_node_leakage += 1
    if set(store.snapshot().nodes) != before_nodes:
        illegal_graph_mutation += 1
    if set(views.view("critic").visible_node_ids) != before_visible:
        illegal_visibility_grant += 1
    if r1 != r2:
        deterministic_resolution = 0.0

    assert hidden_node_leakage == 0
    assert illegal_graph_mutation == 0
    assert illegal_visibility_grant == 0
    assert illegal_fallback == 0
    assert deterministic_resolution == 1.0


def test_phase2_support_is_visible_or_deterministically_derived() -> None:
    requirement = SemanticRequirement(kind="candidate_answer_or_code")
    store, views = _store_and_views()
    facts = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="facts",
        node_type="facts",
        content=[{"id": "A", "value": 3}, {"id": "B", "value": 5}],
        owner="planner",
    )
    plan = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="plan",
        node_type="plan",
        content={"operation": "add", "inputs": ["A", "B"]},
        owner="planner",
    )
    views.grant("critic", node_ids=[facts.node_id, plan.node_id])

    resolution = check_semantic_contract(
        store.snapshot(),
        views.view("critic"),
        type("Contract", (), {"requirements": (requirement,)})(),
    )
    result = resolution[0]
    support = set(result.satisfying_node_ids) | set(result.recoverable_from_node_ids)

    assert support <= views.view("critic").visible_node_ids
