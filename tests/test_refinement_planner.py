from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_runtime import AgentGraphViewManager, GraphStore, SemanticNACK, plan_refinement
from workflow_runtime.refinement_planner import RefinementPlanner


def _store_and_views() -> tuple[GraphStore, AgentGraphViewManager]:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="2+3", owner="user")
    views = AgentGraphViewManager(store, task_id="t", branch_id="main", agents=("solver", "critic", "final_solver"))
    return store, views


def test_plan_refinement_sends_candidate_result_for_missing_final_candidate() -> None:
    store, views = _store_and_views()
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 5},
        owner="solver",
    )
    views.grant("solver", node_ids=[result.node_id], local=True)

    plan = plan_refinement(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        sender_view=views.view("solver"),
        receiver_view=views.view("critic"),
        missing_semantics=["final_candidate"],
    )

    assert plan.root_node_ids == (result.node_id,)
    assert plan.reason_by_node[result.node_id] == "missing final_candidate: send candidate artifact"


def test_plan_refinement_sends_code_for_missing_candidate_code() -> None:
    store, views = _store_and_views()
    code = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="code",
        node_type="code",
        content="def f():\n    return 1\n",
        owner="solver",
    )
    views.grant("solver", node_ids=[code.node_id], local=True)

    plan = plan_refinement(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        sender_view=views.view("solver"),
        receiver_view=views.view("critic"),
        missing_semantics=["candidate_code"],
    )

    assert plan.root_node_ids == (code.node_id,)


def test_refinement_planner_sends_validation_and_target_for_missing_validation_signal() -> None:
    store, views = _store_and_views()
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 5},
        owner="solver",
    )
    verification = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="verification",
        node_type="verification",
        content={"target": "result", "status": "verified"},
        owner="critic",
        status="verified",
    )
    store.add_edge(source=verification.node_id, target=result.node_id, relation="verifies", created_by_role="critic")
    views.grant("critic", node_ids=[verification.node_id, result.node_id], local=True)

    nack = SemanticNACK(
        sender="critic",
        receiver="final_solver",
        round_index=0,
        missing_semantics=("validation_signal",),
    )
    plan = RefinementPlanner(
        store.snapshot(),
        sender="critic",
        receiver="final_solver",
        sender_view=views.view("critic"),
        receiver_view=views.view("final_solver"),
    ).plan_for_nack(nack)

    assert plan.root_node_ids == (verification.node_id,)
    assert plan.reason_by_node[verification.node_id] == "missing validation_signal: send validation artifact"
    assert result.node_id not in plan.reason_by_node


def test_refinement_planner_sends_direct_dependencies_for_support_dependencies() -> None:
    store, views = _store_and_views()
    facts = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="facts",
        node_type="facts",
        content=[{"id": "A", "value": 2}, {"id": "B", "value": 3}],
        owner="planner",
    )
    calculation = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="calculation",
        node_type="calculation",
        content={"id": "R1", "expression": "2+3", "value": 5},
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
    store.add_edge(source=facts.node_id, target=calculation.node_id, relation="depends_on", created_by_role="planner")
    store.add_edge(source=calculation.node_id, target=result.node_id, relation="depends_on", created_by_role="solver")
    views.grant("solver", node_ids=[facts.node_id, calculation.node_id, result.node_id], local=True)
    views.grant("critic", node_ids=[result.node_id])

    plan = plan_refinement(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        sender_view=views.view("solver"),
        receiver_view=views.view("critic"),
        missing_semantics=["support_dependencies"],
    )

    assert plan.root_node_ids == (calculation.node_id,)
    assert facts.node_id not in plan.root_node_ids


def test_refinement_plan_is_empty_when_sender_lacks_matching_semantics() -> None:
    store, views = _store_and_views()

    plan = plan_refinement(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        sender_view=views.view("solver"),
        receiver_view=views.view("critic"),
        missing_semantics=["candidate_code"],
    )

    assert plan.is_empty
    assert plan.is_unresolvable
    assert plan.status == "UNRESOLVABLE"
    assert plan.as_dict()["is_empty"] is True
