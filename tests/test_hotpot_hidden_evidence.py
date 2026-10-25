import json

import main as workflow_helpers
from scripts.evaluate_domain_workflow import add_source_nodes, hotpot_hidden_source_row
from workflow_runtime.context_slicer import build_context_slice

from workflow_runtime import (
    CommunicationAction,
    CommunicationActionExecutor,
    CommunicationActionType,
    CommunicationReplayBuffer,
    GraphStore,
    HotpotLocalInformationAdapter,
    LangGraphWorkflow,
    build_hotpot_hidden_environment,
)


def _facts():
    return [
        {"title": "Corliss Archer", "sent_id": 0, "text": "Corliss Archer was portrayed by Shirley Temple."},
        {"title": "Shirley Temple", "sent_id": 0, "text": "Shirley Temple served as Chief of Protocol."},
        {"title": "Noise", "sent_id": 0, "text": "Noise is an unrelated film."},
    ]


def _snapshot(environment):
    store = GraphStore()
    store.add_node(
        task_id="hotpot-hidden",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content=environment.question,
        owner="user",
        created_by_role="user",
    )
    replay = CommunicationReplayBuffer()
    workflow = LangGraphWorkflow(
        store=store,
        model=lambda request: {},
        task_id="hotpot-hidden",
        task_type="multihop_qa",
        communication_replay_buffer=replay,
        replay_external_state_provider=lambda: {"hotpot_hidden_evidence": environment.as_dict()},
    )
    node = store.add_node(
        task_id="hotpot-hidden",
        branch_id="main",
        logical_id="plan_steps",
        node_type="plan_steps",
        content=[{"id": "R1", "operation": "answer"}],
        owner="planner",
        created_by_role="planner",
    )
    workflow.agent_views.grant("planner", node_ids=[node.node_id], local=True)
    workflow._communicate_nodes(
        sender="planner",
        node_ids=[node.node_id],
        branch_id="main",
        workflow_state={"status": "running", "round_id": 0},
    )
    return replay.snapshots()[0], workflow


def test_hidden_split_uses_no_oracle_fields_and_withholds_other_titles():
    environment = build_hotpot_hidden_environment(
        "What position was held by the woman who portrayed Corliss Archer?",
        _facts(),
        initial_title_count=1,
    )

    assert environment.visible_facts
    assert environment.hidden_facts
    serialized = json.dumps(environment.as_dict())
    assert "gold_answer" not in serialized
    assert "supporting_facts" not in serialized
    assert {fact["fact_id"] for fact in environment.visible_facts}.isdisjoint(
        fact["fact_id"] for fact in environment.hidden_facts
    )


def test_retrieve_materializes_hidden_facts_only_for_receiving_agent():
    environment = build_hotpot_hidden_environment(
        "What position was held by the woman who portrayed Corliss Archer?",
        _facts(),
        initial_title_count=1,
    )
    snapshot, workflow = _snapshot(environment)
    hidden_text = {fact["text"] for fact in environment.hidden_facts}
    graph_text_before = {str(node.content) for node in snapshot.graph_state.nodes.values()}
    assert not any(text in content for text in hidden_text for content in graph_text_before)

    action = CommunicationAction(
        CommunicationActionType.RETRIEVE,
        query="second hop position held by Shirley Temple",
    )
    executor = CommunicationActionExecutor(
        information_adapter=HotpotLocalInformationAdapter.from_snapshot(snapshot, retrieval_limit=2)
    )
    workflow.restore_communication_snapshot(snapshot)
    result = executor.apply(snapshot, action, store=workflow.store, views=workflow.agent_views)

    assert result.legal
    assert result.closure_node_ids
    solver_visible = workflow.agent_views.view(snapshot.receiver).visible_node_ids
    planner_visible = workflow.agent_views.view("planner").visible_node_ids
    assert set(result.closure_node_ids) <= solver_visible
    assert set(result.closure_node_ids).isdisjoint(planner_visible)
    retrieved = [workflow.store.node(node_id) for node_id in result.closure_node_ids]
    assert all(node.provenance["source"] == "hotpot_local_hidden_corpus" for node in retrieved)


def test_hidden_adapter_rejects_empty_retrieve_and_oracle_environment():
    environment = build_hotpot_hidden_environment("question", _facts(), initial_title_count=1)
    snapshot, _ = _snapshot(environment)
    adapter = HotpotLocalInformationAdapter.from_snapshot(snapshot)
    preview = adapter.preview(snapshot, CommunicationAction(CommunicationActionType.RETRIEVE))
    assert not preview.legal

    bad = snapshot.to_dict()
    bad["external_state"]["hotpot_hidden_evidence"]["gold_answer"] = "leak"
    from workflow_runtime.communication_replay import CommunicationDecisionSnapshot

    contaminated = CommunicationDecisionSnapshot.from_dict(bad)
    try:
        HotpotLocalInformationAdapter.from_snapshot(contaminated)
    except ValueError as exc:
        assert "oracle" in str(exc)
    else:
        raise AssertionError("oracle-contaminated hidden environment was accepted")


def test_visible_hidden_mode_has_non_oracle_evidence_links_required_by_context_contract():
    environment = build_hotpot_hidden_environment("question", _facts(), initial_title_count=1)
    original = {
        "task_type": "multihop_qa",
        "entities": ["oracle inventory must be replaced"],
        "supporting_facts": list(_facts()),
        "evidence_links": [{"entity": "gold support", "sent_id": 9}],
    }
    source_row = hotpot_hidden_source_row(original, environment)
    store = workflow_helpers.init_workflow_graph(
        "[domain=multihop_qa] question",
        task_id="hidden-source",
        task_type="multihop_qa",
    )
    add_source_nodes(workflow_helpers, store, source_row, "hidden-source")

    sentences = [node for node in store.snapshot().nodes.values() if node.type == "sentence_source"]
    assert sentences
    assert "gold support" not in json.dumps([node.content for node in sentences])
    assert not any(node.type == "evidence_link" for node in store.snapshot().nodes.values())
    runtime = LangGraphWorkflow(
        store=store,
        model=lambda request: {},
        task_id="hidden-source",
        task_type="multihop_qa",
    )
    context = build_context_slice(
        runtime.agent_views.visible_store("planner"),
        task_id="hidden-source",
        branch_id="main",
        role="planner",
    )
    assert any(store.node(node_id).type == "sentence_source" for node_id in context.visible_node_ids)
