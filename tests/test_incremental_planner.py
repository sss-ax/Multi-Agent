from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_runtime import (
    AgentGraphViewManager,
    GraphStore,
    ReceiverNeed,
    plan_incremental_request,
)


def _store_and_views() -> tuple[GraphStore, AgentGraphViewManager]:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="2+3", owner="user")
    views = AgentGraphViewManager(store, task_id="t", branch_id="main", agents=("solver", "critic", "final_solver"))
    return store, views


def test_hard_need_requests_candidate_core_fragment() -> None:
    store, views = _store_and_views()
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 5, "note": "optional"},
        owner="solver",
    )
    views.grant("solver", node_ids=[result.node_id], local=True)

    plan = plan_incremental_request(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        sender_view=views.view("solver"),
        receiver_view=views.view("critic"),
        receiver_need=ReceiverNeed(
            sender="solver",
            receiver="critic",
            stage="solver->critic",
            round_index=0,
            hard_missing=("candidate_answer_or_code",),
        ),
    )

    assert plan.level == "hard"
    assert plan.root_node_ids == (result.node_id,)
    assert plan.requested_fragment_ids == (f"{result.node_id}#final_value",)
    assert all("result_metadata" not in item for item in plan.requested_fragment_ids)


def test_verification_need_requests_direct_support_core_fragment() -> None:
    store, views = _store_and_views()
    calc = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="calculation",
        node_type="calculation",
        content={"id": "C1", "expression": "2+3", "value": 5, "trace": "optional"},
        owner="solver",
    )
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 5},
        owner="solver",
    )
    store.add_edge(source=calc.node_id, target=result.node_id, relation="depends_on")
    views.grant("solver", node_ids=[calc.node_id, result.node_id], local=True)
    views.grant("critic", node_ids=[result.node_id])

    plan = plan_incremental_request(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        sender_view=views.view("solver"),
        receiver_view=views.view("critic"),
        receiver_need=ReceiverNeed(
            sender="solver",
            receiver="critic",
            stage="solver->critic",
            round_index=0,
            verification_missing=("support_dependencies",),
        ),
    )

    assert plan.level == "verification"
    assert plan.root_node_ids == (calc.node_id,)
    assert f"{calc.node_id}#final_value" in plan.requested_fragment_ids
    assert f"{calc.node_id}#key_operation" in plan.requested_fragment_ids
    assert f"{calc.node_id}#calculation_trace" not in plan.requested_fragment_ids


def test_verification_need_selects_lowest_cost_covering_fragment_and_rejects_others() -> None:
    store, views = _store_and_views()
    cheap = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="cheap_calc",
        node_type="calculation",
        content={"id": "A", "expression": "2+3", "value": 5},
        owner="solver",
    )
    expensive = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="expensive_calc",
        node_type="calculation",
        content={"id": "B", "expression": " ".join(["long"] * 30), "value": 5},
        owner="solver",
    )
    unrelated = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="unrelated_fact",
        node_type="fact",
        content={"id": "C", "value": 99},
        owner="solver",
    )
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 5},
        owner="solver",
    )
    store.add_edge(source=cheap.node_id, target=result.node_id, relation="depends_on")
    store.add_edge(source=expensive.node_id, target=result.node_id, relation="depends_on")
    views.grant("solver", node_ids=[cheap.node_id, expensive.node_id, unrelated.node_id, result.node_id], local=True)
    views.grant("critic", node_ids=[result.node_id])

    plan = plan_incremental_request(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        sender_view=views.view("solver"),
        receiver_view=views.view("critic"),
        receiver_need=ReceiverNeed(
            sender="solver",
            receiver="critic",
            stage="solver->critic",
            round_index=0,
            verification_missing=("support_dependencies",),
        ),
    )

    assert plan.root_node_ids == (cheap.node_id,)
    assert f"{cheap.node_id}#final_value" in plan.selected_fragment_ids
    assert f"{expensive.node_id}#final_value" in plan.rejected_fragment_ids
    assert all(not item.startswith(f"{unrelated.node_id}#") for item in plan.candidate_fragment_ids)
    assert "minimum closure cost" in next(iter(plan.reason_by_fragment.values()))


def test_quality_need_requests_optional_fragment_only() -> None:
    store, views = _store_and_views()
    verification = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="verification",
        node_type="verification",
        content={"status": "uncertain", "repair_hint": "show derivation"},
        owner="critic",
    )
    views.grant("critic", node_ids=[verification.node_id], local=True)

    plan = plan_incremental_request(
        store.snapshot(),
        sender="critic",
        receiver="solver",
        sender_view=views.view("critic"),
        receiver_view=views.view("solver"),
        receiver_need=ReceiverNeed(
            sender="critic",
            receiver="solver",
            stage="critic->solver",
            round_index=0,
            quality_gaps=("full_feedback",),
        ),
    )

    assert plan.level == "quality"
    assert plan.root_node_ids == (verification.node_id,)
    assert plan.requested_fragment_ids == (f"{verification.node_id}#full_feedback",)
    assert plan.estimated_utility > 0
    assert plan.utility_per_token > 0
    assert "utility/cost" in next(iter(plan.reason_by_fragment.values()))


def test_incremental_planner_has_no_graph_or_visibility_side_effects() -> None:
    store, views = _store_and_views()
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"value": 5},
        owner="solver",
    )
    views.grant("solver", node_ids=[result.node_id], local=True)
    before_nodes = set(store.snapshot().nodes)
    before_edges = {edge.edge_id for edge in store.snapshot().edges}
    before_sender_visible = set(views.view("solver").visible_node_ids)
    before_receiver_visible = set(views.view("critic").visible_node_ids)

    plan_incremental_request(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        sender_view=views.view("solver"),
        receiver_view=views.view("critic"),
        receiver_need=ReceiverNeed(
            sender="solver",
            receiver="critic",
            stage="solver->critic",
            round_index=0,
            hard_missing=("candidate_answer_or_code",),
        ),
    )

    assert set(store.snapshot().nodes) == before_nodes
    assert {edge.edge_id for edge in store.snapshot().edges} == before_edges
    assert views.view("solver").visible_node_ids == before_sender_visible
    assert views.view("critic").visible_node_ids == before_receiver_visible
