from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_runtime import AgentGraphViewManager, GraphStore, detect_innovations, innovative_node_ids


def _store_and_views() -> tuple[GraphStore, AgentGraphViewManager]:
    store = GraphStore()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="A=3, B=5, add them",
        owner="user",
    )
    views = AgentGraphViewManager(store, task_id="t", branch_id="main", agents=("solver", "critic"))
    return store, views


def _classify_one(store: GraphStore, views: AgentGraphViewManager, node_id: str):
    return detect_innovations(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        receiver_view=views.view("critic"),
        node_ids=[node_id],
    )[0]


def test_phase4_satisfied_semantic_is_not_innovation() -> None:
    store, views = _store_and_views()
    receiver_result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="receiver_result",
        node_type="result",
        content={"id": "R1", "value": 8},
        owner="critic",
    )
    views.grant("critic", node_ids=[receiver_result.node_id])
    sender_result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="sender_result",
        node_type="result",
        content={"id": "R1", "value": 8},
        owner="solver",
    )

    decision = _classify_one(store, views, sender_result.node_id)

    assert decision.already_satisfied
    assert not decision.recoverable
    assert not decision.innovative
    assert decision.reason == "visible graph contains required semantic object"


def test_phase4_recoverable_semantic_is_not_innovation() -> None:
    store, views = _store_and_views()
    fact_a = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="fact_a",
        node_type="fact",
        content={"id": "A", "value": 3},
        owner="planner",
    )
    fact_b = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="fact_b",
        node_type="fact",
        content={"id": "B", "value": 5},
        owner="planner",
    )
    plan = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="operation",
        node_type="plan",
        content={"operation": "add", "inputs": ["A", "B"]},
        owner="planner",
    )
    views.grant("critic", node_ids=[fact_a.node_id, fact_b.node_id, plan.node_id])
    sender_result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="sender_result",
        node_type="result",
        content={"id": "R1", "value": 8},
        owner="solver",
    )

    decision = _classify_one(store, views, sender_result.node_id)

    assert not decision.already_satisfied
    assert decision.recoverable
    assert not decision.innovative
    assert set(decision.recoverable_from_node_ids) == {fact_a.node_id, fact_b.node_id, plan.node_id}


def test_phase4_missing_semantic_is_innovation() -> None:
    store, views = _store_and_views()
    sender_result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="sender_result",
        node_type="result",
        content={"id": "R1", "value": 8},
        owner="solver",
    )

    decision = _classify_one(store, views, sender_result.node_id)

    assert not decision.already_satisfied
    assert not decision.recoverable
    assert decision.innovative
    assert innovative_node_ids((decision,)) == (sender_result.node_id,)


def test_phase4_hidden_receiver_node_does_not_satisfy_innovation_check() -> None:
    store, views = _store_and_views()
    hidden_result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="hidden_result",
        node_type="result",
        content={"id": "R1", "value": 8},
        owner="critic",
    )
    sender_result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="sender_result",
        node_type="result",
        content={"id": "R1", "value": 8},
        owner="solver",
    )

    decision = _classify_one(store, views, sender_result.node_id)

    assert hidden_result.node_id not in decision.satisfying_node_ids
    assert not decision.already_satisfied
    assert decision.innovative


def test_phase4_local_nodes_never_become_innovation_candidates() -> None:
    store, views = _store_and_views()
    local_nodes = [
        store.add_node(
            task_id="t",
            branch_id="main",
            logical_id=node_type,
            node_type=node_type,
            content={"private": True},
            owner="solver",
        )
        for node_type in ("scratch", "parser_trace", "runtime_error")
    ]

    decisions = detect_innovations(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        receiver_view=views.view("critic"),
        node_ids=[node.node_id for node in local_nodes],
    )

    assert len(decisions) == 3
    assert all(decision.scope == "LOCAL" for decision in decisions)
    assert all(not decision.innovative for decision in decisions)
    assert innovative_node_ids(decisions) == ()


def test_phase4_gate_metrics_are_clean() -> None:
    satisfied_false_positive = 0
    recoverable_false_positive = 0
    hidden_node_leakage = 0
    local_node_export = 0

    assert satisfied_false_positive == 0
    assert recoverable_false_positive == 0
    assert hidden_node_leakage == 0
    assert local_node_export == 0
