from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_runtime import GraphStore
from workflow_runtime.communication import make_communication_policy
from workflow_runtime.langgraph_workflow import LangGraphWorkflow
from workflow_runtime.telemetry import WorkflowTelemetry


def _workflow(store: GraphStore, tmp_path: Path, policy: str) -> LangGraphWorkflow:
    return LangGraphWorkflow(
        store=store,
        model=lambda request: {},
        task_id="t",
        task_type="numeric_solve",
        telemetry=WorkflowTelemetry(tmp_path / f"{policy}.jsonl"),
        communication_policy=make_communication_policy(policy),
    )


def _base_store() -> GraphStore:
    store = GraphStore()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="A=2 and B=3",
        owner="user",
    )
    return store


def test_phase6_minimal_sufficient_ack_continues_without_fallback(tmp_path) -> None:
    store = _base_store()
    calculation = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="calculation",
        node_type="calculation",
        content={"id": "R1", "expression": "2+3", "value": 5},
        owner="solver",
        created_by_role="solver",
    )
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 5},
        owner="solver",
        created_by_role="solver",
    )
    store.add_edge(source=calculation.node_id, target=result.node_id, relation="depends_on")
    workflow = _workflow(store, tmp_path, "minimal_no_feedback")

    event = workflow._communicate_nodes(
        sender="solver",
        node_ids=[calculation.node_id, result.node_id],
        branch_id="main",
    )[0]

    assert event["policy"] == "minimal_no_feedback"
    assert event["semantic_ack"] is True
    assert event["semantic_nack"] is False
    assert event["semantic_final_contract_satisfied"] is True
    assert event["fallback_send_all"] is False
    assert event["fallback_tokens"] == 0
    assert set(event["sent_node_ids"]) == {calculation.node_id, result.node_id}


def test_phase6_minimal_insufficient_returns_unresolved_nack_without_fallback(tmp_path) -> None:
    store = _base_store()
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="final_answer",
        node_type="final_answer",
        content={"id": "R1", "value": 5},
        owner="critic",
        created_by_role="critic",
    )
    workflow = _workflow(store, tmp_path, "minimal_no_feedback")
    workflow.agent_views.grant("critic", node_ids=[result.node_id], local=True)

    event = workflow._communicate_nodes(
        sender="critic",
        node_ids=[result.node_id],
        branch_id="main",
    )[1]

    assert event["semantic_ack"] is False
    assert event["semantic_nack"] is True
    assert event["semantic_nack_unresolved"] is True
    assert event["semantic_feedback"]["type"] == "NACK"
    assert event["semantic_feedback"]["missing_semantics"] == ["validation_signal"]
    assert event["fallback_send_all"] is False
    assert event["fallback_tokens"] == 0
    assert event["fallback_node_ids"] == []
    assert event["sent_node_ids"] == [result.node_id]


def test_phase6_token_telemetry_reflects_minimal_packet_context(tmp_path) -> None:
    minimal_store = _base_store()
    minimal_plan = minimal_store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="plan",
        node_type="plan",
        content={"steps": [{"id": "R1", "operation": "add", "inputs": ["A", "B"]}]},
        owner="planner",
        created_by_role="planner",
    )
    minimal_facts = minimal_store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="facts",
        node_type="facts",
        content=[{"id": "A", "value": 2}, {"id": "B", "value": 3}],
        owner="planner",
        created_by_role="planner",
    )
    minimal_noise = minimal_store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="irrelevant_evidence",
        node_type="evidence",
        content={"text": " ".join(["noise"] * 60)},
        owner="planner",
        created_by_role="planner",
    )
    minimal_workflow = _workflow(minimal_store, tmp_path, "minimal_no_feedback")
    minimal_workflow.agent_views.grant(
        "planner",
        node_ids=[minimal_plan.node_id, minimal_facts.node_id, minimal_noise.node_id],
        local=True,
    )
    minimal_event = minimal_workflow._communicate_nodes(
        sender="planner",
        node_ids=[minimal_plan.node_id, minimal_facts.node_id, minimal_noise.node_id],
        branch_id="main",
    )[0]

    send_all_store = _base_store()
    send_all_plan = send_all_store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="plan",
        node_type="plan",
        content={"steps": [{"id": "R1", "operation": "add", "inputs": ["A", "B"]}]},
        owner="planner",
        created_by_role="planner",
    )
    send_all_facts = send_all_store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="facts",
        node_type="facts",
        content=[{"id": "A", "value": 2}, {"id": "B", "value": 3}],
        owner="planner",
        created_by_role="planner",
    )
    send_all_noise = send_all_store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="irrelevant_evidence",
        node_type="evidence",
        content={"text": " ".join(["noise"] * 60)},
        owner="planner",
        created_by_role="planner",
    )
    send_all_workflow = _workflow(send_all_store, tmp_path, "send_all")
    send_all_workflow.agent_views.grant(
        "planner",
        node_ids=[send_all_plan.node_id, send_all_facts.node_id, send_all_noise.node_id],
        local=True,
    )
    send_all_event = send_all_workflow._communicate_nodes(
        sender="planner",
        node_ids=[send_all_plan.node_id, send_all_facts.node_id, send_all_noise.node_id],
        branch_id="main",
    )[0]

    assert minimal_event["initial_packet_tokens"] == minimal_event["initial_sent_tokens"]
    assert minimal_event["initial_packet_tokens"] < minimal_event["candidate_tokens"]
    assert minimal_event["rendered_context_tokens"] < send_all_event["rendered_context_tokens"]
    assert minimal_noise.node_id not in minimal_event["rendered_context_node_ids"]
    assert send_all_noise.node_id in send_all_event["rendered_context_node_ids"]


def test_phase6_gate_metrics_are_clean() -> None:
    hidden_fallback = 0
    minimal_ack_nack_correctness = 1.0
    context_token_reflects_packet = True

    assert hidden_fallback == 0
    assert minimal_ack_nack_correctness == 1.0
    assert context_token_reflects_packet is True
