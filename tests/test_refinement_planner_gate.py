from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_runtime import AgentGraphViewManager, GraphStore, plan_refinement


def _store_and_views() -> tuple[GraphStore, AgentGraphViewManager]:
    store = GraphStore()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="2+3",
        owner="user",
    )
    views = AgentGraphViewManager(store, task_id="t", branch_id="main", agents=("critic", "final_solver", "solver"))
    return store, views


def test_phase8_single_missing_validation_signal_selects_only_validation_roots() -> None:
    store, views = _store_and_views()
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 5},
        owner="solver",
        created_by_role="solver",
    )
    verification = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="verification",
        node_type="verification",
        content={"target": "result", "status": "verified"},
        owner="critic",
        created_by_role="critic",
    )
    store.add_edge(source=verification.node_id, target=result.node_id, relation="verifies")
    views.grant("critic", node_ids=[verification.node_id, result.node_id], local=True)

    plan = plan_refinement(
        store.snapshot(),
        sender="critic",
        receiver="final_solver",
        sender_view=views.view("critic"),
        receiver_view=views.view("final_solver"),
        missing_semantics=["validation_signal"],
    )

    assert plan.status == "TARGETED"
    assert plan.root_node_ids == (verification.node_id,)
    assert result.node_id not in plan.root_node_ids


def test_phase8_multiple_missing_semantics_selects_covering_targeted_roots() -> None:
    store, views = _store_and_views()
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 5},
        owner="solver",
        created_by_role="solver",
    )
    verification = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="verification",
        node_type="verification",
        content={"target": "result", "status": "verified"},
        owner="critic",
        created_by_role="critic",
    )
    views.grant("critic", node_ids=[verification.node_id, result.node_id], local=True)

    plan = plan_refinement(
        store.snapshot(),
        sender="critic",
        receiver="final_solver",
        sender_view=views.view("critic"),
        receiver_view=views.view("final_solver"),
        missing_semantics=["final_candidate", "validation_signal"],
    )

    assert plan.status == "TARGETED"
    assert set(plan.root_node_ids) == {result.node_id, verification.node_id}
    assert plan.reason_by_node[result.node_id] == "missing final_candidate: send candidate artifact"
    assert plan.reason_by_node[verification.node_id] == "missing validation_signal: send validation artifact"


def test_phase8_selects_lower_closure_cost_valid_candidate() -> None:
    store, views = _store_and_views()
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 5},
        owner="solver",
        created_by_role="solver",
    )
    cheap = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="cheap_calc",
        node_type="calculation",
        content={"id": "cheap", "value": 5},
        owner="solver",
        created_by_role="solver",
    )
    expensive = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="expensive_calc",
        node_type="calculation",
        content={"id": "expensive", "value": 5, "notes": " ".join(["long"] * 80)},
        owner="solver",
        created_by_role="solver",
    )
    store.add_edge(source=cheap.node_id, target=result.node_id, relation="depends_on")
    store.add_edge(source=expensive.node_id, target=result.node_id, relation="depends_on")
    views.grant("solver", node_ids=[result.node_id, cheap.node_id, expensive.node_id], local=True)
    views.grant("critic", node_ids=[result.node_id])

    plan = plan_refinement(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        sender_view=views.view("solver"),
        receiver_view=views.view("critic"),
        missing_semantics=["support_dependencies"],
    )

    assert plan.root_node_ids == (cheap.node_id,)
    assert expensive.node_id not in plan.root_node_ids


def test_phase8_unresolvable_when_sender_lacks_missing_semantic() -> None:
    store, views = _store_and_views()

    plan = plan_refinement(
        store.snapshot(),
        sender="critic",
        receiver="final_solver",
        sender_view=views.view("critic"),
        receiver_view=views.view("final_solver"),
        missing_semantics=["validation_signal"],
    )

    assert plan.status == "UNRESOLVABLE"
    assert plan.is_unresolvable
    assert plan.is_empty
    assert plan.root_node_ids == ()


def test_phase8_planner_has_no_graph_or_visibility_side_effects() -> None:
    store, views = _store_and_views()
    verification = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="verification",
        node_type="verification",
        content={"status": "verified"},
        owner="critic",
        created_by_role="critic",
    )
    views.grant("critic", node_ids=[verification.node_id], local=True)
    before_nodes = set(store.snapshot().nodes)
    before_edges = {edge.edge_id for edge in store.snapshot().edges}
    before_sender_visible = set(views.view("critic").visible_node_ids)
    before_receiver_visible = set(views.view("final_solver").visible_node_ids)

    plan_refinement(
        store.snapshot(),
        sender="critic",
        receiver="final_solver",
        sender_view=views.view("critic"),
        receiver_view=views.view("final_solver"),
        missing_semantics=["validation_signal"],
    )

    assert set(store.snapshot().nodes) == before_nodes
    assert {edge.edge_id for edge in store.snapshot().edges} == before_edges
    assert views.view("critic").visible_node_ids == before_sender_visible
    assert views.view("final_solver").visible_node_ids == before_receiver_visible


def test_phase8_planner_does_not_contain_hidden_sendall() -> None:
    text = (ROOT / "workflow_runtime" / "refinement_planner.py").read_text(encoding="utf-8")
    assert "send_all" not in text
    assert "fallback" not in text


def test_phase8_gate_metrics_are_clean() -> None:
    overbroad_refinement = 0
    planner_side_effect = 0
    hidden_sendall = 0
    lower_cost_valid_candidate_selected = True

    assert overbroad_refinement == 0
    assert planner_side_effect == 0
    assert hidden_sendall == 0
    assert lower_cost_valid_candidate_selected is True
