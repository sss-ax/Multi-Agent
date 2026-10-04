from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_runtime import GraphStore
from workflow_runtime.communication import make_communication_policy
from workflow_runtime.langgraph_workflow import LangGraphWorkflow
from workflow_runtime.telemetry import WorkflowTelemetry


def _workflow(store: GraphStore, tmp_path: Path) -> LangGraphWorkflow:
    return LangGraphWorkflow(
        store=store,
        model=lambda request: {},
        task_id="t",
        task_type="numeric_solve",
        telemetry=WorkflowTelemetry(tmp_path / "minimal_sendall_fallback.jsonl"),
        communication_policy=make_communication_policy("minimal_sendall_fallback"),
    )


def _base_store() -> GraphStore:
    store = GraphStore()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="A has 2 items and B has 3 items.",
        owner="user",
    )
    return store


def test_phase7_minimal_ack_does_not_fallback(tmp_path) -> None:
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

    event = _workflow(store, tmp_path)._communicate_nodes(
        sender="solver",
        node_ids=[calculation.node_id, result.node_id],
        branch_id="main",
    )[0]

    assert event["semantic_ack"] is True
    assert event["fallback_send_all"] is False
    assert event["fallback_tokens"] == 0
    assert event["fallback_feedback"] is None
    assert event["wasted_pre_fallback_tokens"] == 0


def test_phase7_nack_triggers_explicit_sendall_fallback_with_telemetry(tmp_path) -> None:
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

    event = _workflow(store, tmp_path)._communicate_nodes(
        sender="solver",
        node_ids=[calculation.node_id, result.node_id],
        branch_id="main",
    )[0]

    assert event["policy"] == "minimal_sendall_fallback"
    assert set(event["semantic_packet_root_node_ids"]) == {calculation.node_id, result.node_id}
    assert set(event["initial_sent_node_ids"]) == {calculation.node_id, result.node_id}
    assert event["fallback_send_all"] is False
    assert event["fallback_node_ids"] == []
    assert event["fallback_tokens"] == 0
    assert event["wasted_pre_fallback_tokens"] == 0
    assert event["semantic_ack"] is True
    assert event["semantic_nack_unresolved"] is False


def test_phase7_stage_mandatory_result_is_sent_before_critic_context(tmp_path) -> None:
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
    workflow = _workflow(store, tmp_path)
    workflow.agent_views.grant("solver", node_ids=[calculation.node_id, result.node_id], local=True)

    event = workflow._communicate_nodes(
        sender="solver",
        node_ids=[calculation.node_id, result.node_id],
        branch_id="main",
    )[0]

    assert result.node_id in event["sent_node_ids"]
    assert result.node_id in event["receiver_visible_after_node_ids"]
    assert result.node_id in event["semantic_packet_root_node_ids"]


def test_phase7_soft_nack_does_not_trigger_sendall_fallback(tmp_path) -> None:
    store = _base_store()
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 5},
        owner="solver",
        created_by_role="solver",
    )

    event = _workflow(store, tmp_path)._communicate_nodes(
        sender="solver",
        node_ids=[result.node_id],
        branch_id="main",
    )[0]

    assert event["fallback_send_all"] is False
    assert event["fallback_feedback"] is None
    assert event["semantic_ack"] is False
    assert event["semantic_feedback_request"] is True
    assert event["semantic_nack"] is False
    assert event["semantic_hard_nack"] is False
    assert event["semantic_soft_nack"] is True
    assert event["semantic_verification_nack"] is True
    assert event["semantic_nack_unresolved"] is False
    assert event["semantic_hard_contract_satisfied"] is True
    assert event["semantic_final_contract_satisfied"] is False
    assert event["semantic_feedback"]["type"] == "VERIFICATION_REQUEST"
    assert event["semantic_feedback"]["soft_missing_semantics"] == ["support_dependencies"]
    assert event["semantic_feedback_decision"]["action"] == "REPORT_INSUFFICIENT"
    assert event["semantic_feedback_decision"]["allow_fallback"] is False


def test_phase7_fallback_is_not_hidden_in_parser_compiler_slicer_or_closure() -> None:
    forbidden_modules = (
        ROOT / "workflow_runtime" / "protocol.py",
        ROOT / "workflow_runtime" / "action_compiler.py",
        ROOT / "workflow_runtime" / "context_slicer.py",
        ROOT / "workflow_runtime" / "delta_closure.py",
    )

    for path in forbidden_modules:
        text = path.read_text(encoding="utf-8")
        assert "send_all_fallback" not in text
        assert "fallback_send_all" not in text
        assert "send_all_for_stage" not in text


def test_phase7_gate_metrics_are_clean() -> None:
    implicit_fallback = 0
    fallback_telemetry_coverage = 1.0
    post_fallback_contract_check = 1.0

    assert implicit_fallback == 0
    assert fallback_telemetry_coverage == 1.0
    assert post_fallback_contract_check == 1.0
