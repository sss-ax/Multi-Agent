import json

import pytest

from workflow_runtime import GraphStore
from workflow_runtime.agent_graph_view import AgentGraphViewManager
from workflow_runtime.communication_replay import (
    CommunicationDecisionSnapshot,
    CommunicationReplayBuffer,
    graph_state_from_dict,
    graph_state_to_dict,
)
from workflow_runtime.langgraph_workflow import LangGraphWorkflow


def _store_with_task() -> GraphStore:
    store = GraphStore()
    store.add_node(
        task_id="replay-task",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="solve",
        owner="user",
        created_by_role="user",
    )
    return store


def test_agent_views_snapshot_restore_all_visibility_fields():
    store = _store_with_task()
    result = store.add_node(
        task_id="replay-task",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"value": 7},
        owner="solver",
        created_by_role="solver",
    )
    views = AgentGraphViewManager(
        store,
        task_id="replay-task",
        branch_id="main",
        agents=("solver", "critic"),
    )
    views.grant(
        "critic",
        node_ids=[result.node_id],
        fragment_ids=[f"{result.node_id}#final_value"],
        delta_id="solver->critic:1",
        packet_id="packet-1",
    )
    expected = views.snapshot()

    views.view("critic").visible_node_ids.clear()
    views.view("critic").visible_fragment_ids.clear()
    views.restore(expected)

    assert views.snapshot() == expected


def test_graph_state_json_round_trip_preserves_runtime_models():
    store = _store_with_task()
    state = store.snapshot()

    restored = graph_state_from_dict(json.loads(json.dumps(graph_state_to_dict(state))))

    assert graph_state_to_dict(restored) == graph_state_to_dict(state)


def test_workflow_records_pre_decision_snapshot_and_restores_from_jsonl(tmp_path):
    store = _store_with_task()
    plan = store.add_node(
        task_id="replay-task",
        branch_id="main",
        logical_id="plan_steps",
        node_type="plan_steps",
        content=[{"id": "R1", "operation": "solve", "inputs": ["task"]}],
        owner="planner",
        created_by_role="planner",
    )
    path = tmp_path / "communication-decisions.jsonl"
    replay = CommunicationReplayBuffer(path)
    workflow = LangGraphWorkflow(
        store=store,
        model=lambda request: {},
        task_id="replay-task",
        communication_budget_tokens=64,
        communication_replay_buffer=replay,
        replay_external_state_provider=lambda: {"model_seed": 11, "environment": {"turn": 2}},
    )
    workflow._graph_source_seen_hashes = {"before-decision"}
    workflow.agent_views.grant("planner", node_ids=[plan.node_id], local=True)

    events = workflow._communicate_nodes(
        sender="planner",
        node_ids=[plan.node_id],
        branch_id="main",
        workflow_state={"round_id": 3, "action_count": 4, "status": "running"},
        decision_context={"mode": "normal", "action_index": 1},
    )

    assert len(replay) == len(events) == 1
    snapshot = replay.snapshots()[0]
    assert events[0]["decision_snapshot_id"] == snapshot.snapshot_id
    assert snapshot.stage == "planner->solver"
    assert snapshot.remaining_budget == 64
    assert snapshot.workflow_state["round_id"] == 3
    assert snapshot.workflow_state["communication_decision_context"]["action_index"] == 1
    assert snapshot.external_state["model_seed"] == 11
    assert snapshot.candidates
    assert snapshot.fragments
    assert plan.node_id not in snapshot.agent_views["views"]["solver"]["visible_node_ids"]
    assert plan.node_id in workflow.agent_views.view("solver").visible_node_ids

    loaded = CommunicationReplayBuffer.from_jsonl(path).get(snapshot.snapshot_id)
    extra = store.add_node(
        task_id="replay-task",
        branch_id="main",
        logical_id="later",
        node_type="result",
        content=99,
        owner="solver",
        created_by_role="solver",
    )
    workflow.agent_views.grant("solver", node_ids=[extra.node_id], local=True)
    workflow._graph_source_seen_hashes = {"after-decision"}
    restored_external = []
    restored_workflow_state = workflow.restore_communication_snapshot(
        loaded,
        restore_external=lambda value: restored_external.append(dict(value)),
    )

    assert extra.node_id not in store.snapshot().nodes
    assert plan.node_id not in workflow.agent_views.view("solver").visible_node_ids
    assert plan.node_id in workflow.agent_views.view("planner").visible_node_ids
    assert workflow._graph_source_seen_hashes == {"before-decision"}
    assert restored_workflow_state["action_count"] == 4
    assert restored_external == [{"model_seed": 11, "environment": {"turn": 2}}]


def test_snapshot_rejects_restore_into_another_task_boundary():
    left_store = _store_with_task()
    left_views = AgentGraphViewManager(
        left_store,
        task_id="replay-task",
        branch_id="main",
        agents=("solver",),
    )
    snapshot = CommunicationDecisionSnapshot(
        snapshot_id="one",
        task_id="replay-task",
        branch_id="main",
        task_type="numeric_solve",
        stage="planner->solver",
        sender="planner",
        receiver="solver",
        decision_index=0,
        round_index=0,
        remaining_budget=None,
        policy_name="send_all",
        graph_state=left_store.snapshot(),
        agent_views=left_views.snapshot(),
    )
    other_store = GraphStore()
    other_views = AgentGraphViewManager(
        other_store,
        task_id="other",
        branch_id="main",
        agents=("solver",),
    )

    with pytest.raises(ValueError, match="different task boundary"):
        snapshot.restore_runtime(other_store, other_views)
