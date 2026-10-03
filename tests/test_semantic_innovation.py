from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_runtime import (
    AgentGraphViewManager,
    GraphStore,
    detect_innovations,
    innovative_node_ids,
    semantic_requirement_for_node,
)


def _store_and_views() -> tuple[GraphStore, AgentGraphViewManager]:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="2+3", owner="user")
    views = AgentGraphViewManager(store, task_id="t", branch_id="main", agents=("solver", "critic"))
    return store, views


def test_detects_visible_equivalent_result_as_already_satisfied() -> None:
    store, views = _store_and_views()
    old_result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result_seen",
        node_type="result",
        content={"id": "R1", "value": 5},
        owner="solver",
    )
    views.grant("critic", node_ids=[old_result.node_id])
    new_result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 5},
        owner="solver",
    )

    decision = detect_innovations(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        receiver_view=views.view("critic"),
        node_ids=[new_result.node_id],
    )[0]

    assert decision.semantic_kind == "candidate_answer_or_code"
    assert decision.already_satisfied
    assert not decision.recoverable
    assert not decision.innovative
    assert decision.satisfying_node_ids == (old_result.node_id,)


def test_detects_candidate_result_as_recoverable_from_visible_calculation() -> None:
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
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 5},
        owner="solver",
    )

    decision = detect_innovations(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        receiver_view=views.view("critic"),
        node_ids=[result.node_id],
    )[0]

    assert not decision.already_satisfied
    assert decision.recoverable
    assert not decision.innovative
    assert decision.recoverable_from_node_ids == (calculation.node_id,)


def test_missing_shareable_result_is_innovative() -> None:
    store, views = _store_and_views()
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 5},
        owner="solver",
    )

    decisions = detect_innovations(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        receiver_view=views.view("critic"),
        node_ids=[result.node_id],
    )

    assert decisions[0].innovative
    assert innovative_node_ids(decisions) == (result.node_id,)


def test_local_and_global_nodes_are_not_innovative() -> None:
    store, views = _store_and_views()
    query = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="query_spec",
        node_type="query_spec",
        content={"question": "2+3"},
        owner="planner",
    )
    error = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="error",
        node_type="error",
        content="internal",
        owner="solver",
    )

    decisions = detect_innovations(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        receiver_view=views.view("critic"),
        node_ids=[query.node_id, error.node_id],
    )

    assert [decision.scope for decision in decisions] == ["MANDATORY", "LOCAL"]
    assert [decision.innovative for decision in decisions] == [False, False]


def test_semantic_requirement_for_node_uses_node_identity() -> None:
    store, _views = _store_and_views()
    verification = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="verification",
        node_type="verification",
        content={"target": "result", "status": "verified"},
        owner="critic",
    )

    requirement = semantic_requirement_for_node(verification)

    assert requirement.kind == "validation_signal"
    assert requirement.target_logical_id == "verification"
    assert requirement.required_type == "verification"
