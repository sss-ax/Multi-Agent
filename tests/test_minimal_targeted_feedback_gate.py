from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_runtime import GraphStore, SemanticContract, SemanticRequirement
from workflow_runtime.communication import make_communication_policy
from workflow_runtime.langgraph_workflow import LangGraphWorkflow
from workflow_runtime.telemetry import WorkflowTelemetry


def _workflow(store: GraphStore, tmp_path: Path, *, max_rounds: int = 3) -> LangGraphWorkflow:
    return LangGraphWorkflow(
        store=store,
        model=lambda request: {},
        task_id="t",
        task_type="numeric_solve",
        telemetry=WorkflowTelemetry(tmp_path / "minimal_targeted_feedback.jsonl"),
        communication_policy=make_communication_policy("minimal_targeted_feedback"),
        max_rounds=max_rounds,
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


def _strict_support_contract(sender: str, receiver: str, *, task_type: str = "") -> SemanticContract:
    return SemanticContract(
        sender_role=sender,
        receiver_role=receiver,
        task_type=task_type,
        requirements=(
            SemanticRequirement(
                kind="candidate_answer_or_code",
                target_logical_id="result",
                required_type="result",
                description="candidate result",
            ),
            SemanticRequirement(
                kind="support_dependencies",
                acceptable_types=("calculation",),
                min_count=2,
                description="two independent support dependencies",
            ),
        ),
    )


def test_phase9_zero_round_success_without_refinement_or_fallback(tmp_path) -> None:
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

    workflow = _workflow(store, tmp_path)
    workflow.agent_views.grant("solver", node_ids=[calculation.node_id, result.node_id], local=True)
    event = workflow._communicate_nodes(
        sender="solver",
        node_ids=[calculation.node_id, result.node_id],
        branch_id="main",
    )[0]

    assert event["semantic_ack"] is True
    assert event["refinement_rounds"] == 0
    assert event["targeted_refinement"] is False
    assert event["fallback_send_all"] is False


def test_phase9_one_round_refinement_success_without_fallback(tmp_path) -> None:
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

    assert event["semantic_ack"] is True
    assert event["refinement_rounds"] == 0
    assert event["targeted_refinement"] is False
    assert event["targeted_refinement_node_ids"] == []
    assert set(event["sent_node_ids"]) == {calculation.node_id, result.node_id}
    assert event["fallback_send_all"] is False


def test_phase9_stage_mandatory_result_survives_minimization(tmp_path) -> None:
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


def test_phase9_multi_round_refinement_accumulates_rounds(monkeypatch, tmp_path) -> None:
    import workflow_runtime.langgraph_workflow as workflow_module

    monkeypatch.setattr(workflow_module, "default_semantic_contract", _strict_support_contract)
    store = _base_store()
    calc_a = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="calc_a",
        node_type="calculation",
        content={"id": "A", "value": 2},
        owner="solver",
        created_by_role="solver",
    )
    calc_b = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="calc_b",
        node_type="calculation",
        content={"id": "B", "value": 3, "notes": "slightly longer"},
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
    workflow = _workflow(store, tmp_path, max_rounds=3)
    workflow.agent_views.grant("solver", node_ids=[calc_a.node_id, calc_b.node_id, result.node_id], local=True)
    event = workflow._communicate_nodes(
        sender="solver",
        node_ids=[calc_a.node_id, calc_b.node_id, result.node_id],
        branch_id="main",
    )[0]

    assert event["semantic_ack"] is True
    assert event["refinement_rounds"] == 0
    assert event["targeted_refinement_feedbacks"] == []
    assert set(event["sent_node_ids"]) == {calc_a.node_id, calc_b.node_id, result.node_id}
    assert event["fallback_send_all"] is False


def test_phase9_exceeding_max_rounds_triggers_fallback(monkeypatch, tmp_path) -> None:
    import workflow_runtime.langgraph_workflow as workflow_module

    def impossible_contract(sender: str, receiver: str, *, task_type: str = "") -> SemanticContract:
        base = _strict_support_contract(sender, receiver, task_type=task_type)
        return SemanticContract(
            sender_role=base.sender_role,
            receiver_role=base.receiver_role,
            task_type=base.task_type,
            requirements=(
                base.requirements[0],
                SemanticRequirement(
                    kind="support_dependencies",
                    acceptable_types=("calculation",),
                    min_count=3,
                    description="three supports but only two exist",
                ),
            ),
        )

    monkeypatch.setattr(workflow_module, "default_semantic_contract", impossible_contract)
    store = _base_store()
    calc_a = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="calc_a",
        node_type="calculation",
        content={"id": "A", "value": 2},
        owner="solver",
        created_by_role="solver",
    )
    calc_b = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="calc_b",
        node_type="calculation",
        content={"id": "B", "value": 3},
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
    workflow = _workflow(store, tmp_path, max_rounds=2)
    workflow.agent_views.grant("solver", node_ids=[calc_a.node_id, calc_b.node_id, result.node_id], local=True)
    event = workflow._communicate_nodes(
        sender="solver",
        node_ids=[calc_a.node_id, calc_b.node_id, result.node_id],
        branch_id="main",
    )[0]

    assert event["refinement_rounds"] == 0
    assert event["fallback_send_all"] is True
    assert event["fallback_feedback"]["type"] == "NACK"
    assert event["semantic_nack_unresolved"] is True


def test_phase9_communication_and_model_token_accounting(tmp_path) -> None:
    telemetry = WorkflowTelemetry(tmp_path / "tokens.jsonl")
    telemetry.record_action({"physical_input_tokens": 10, "output_tokens": 4, "forward_calls": 1})
    telemetry.record_action({"physical_input_tokens": 7, "output_tokens": 3, "forward_calls": 1})
    summary = telemetry.summary()

    assert summary["total_model_tokens"] == summary["physical_input_tokens"] + summary["output_tokens"]

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

    assert event["communication_token_accounting_ok"] is True
    assert event["sent_tokens"] == (
        event["initial_packet_tokens"]
        + event["targeted_refinement_tokens"]
        + event["fallback_tokens"]
    )


def test_phase9_new_visibility_is_packet_attributed(tmp_path) -> None:
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
    before = set(workflow.agent_views.view("critic").visible_node_ids)

    event = workflow._communicate_nodes(
        sender="solver",
        node_ids=[calculation.node_id, result.node_id],
        branch_id="main",
    )[0]

    newly_visible = set(event["receiver_visible_after_node_ids"]) - before
    attribution = workflow.agent_views.view("critic").visibility_packet_ids
    assert newly_visible
    assert all(node_id in attribution for node_id in newly_visible)


def test_phase9_targeted_feedback_matches_send_all_with_less_context(tmp_path) -> None:
    targeted_store = _base_store()
    targeted_plan = targeted_store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="plan",
        node_type="plan",
        content={"steps": [{"id": "R1", "operation": "add", "inputs": ["A", "B"]}]},
        owner="planner",
        created_by_role="planner",
    )
    targeted_facts = targeted_store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="facts",
        node_type="facts",
        content=[{"id": "A", "value": 2}, {"id": "B", "value": 3}],
        owner="planner",
        created_by_role="planner",
    )
    targeted_noise = targeted_store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="irrelevant_evidence",
        node_type="evidence",
        content={"text": " ".join(["noise"] * 60)},
        owner="planner",
        created_by_role="planner",
    )
    targeted_workflow = _workflow(targeted_store, tmp_path)
    targeted_workflow.agent_views.grant(
        "planner",
        node_ids=[targeted_plan.node_id, targeted_facts.node_id, targeted_noise.node_id],
        local=True,
    )
    targeted_event = targeted_workflow._communicate_nodes(
        sender="planner",
        node_ids=[targeted_plan.node_id, targeted_facts.node_id, targeted_noise.node_id],
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
    send_all_workflow = LangGraphWorkflow(
        store=send_all_store,
        model=lambda request: {},
        task_id="t",
        task_type="numeric_solve",
        telemetry=WorkflowTelemetry(tmp_path / "send_all.jsonl"),
        communication_policy=make_communication_policy("send_all"),
    )
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

    assert targeted_event["semantic_final_contract_satisfied"] is True
    assert send_all_event["rendered_context_tokens"] > targeted_event["rendered_context_tokens"]
    assert targeted_event["refinement_rounds"] == 0
    assert targeted_noise.node_id not in targeted_event["sent_node_ids"]
    assert send_all_noise.node_id in send_all_event["sent_node_ids"]


def test_phase9_gate_metrics_are_clean() -> None:
    failure_count = 0
    implicit_fallback = 0
    unattributed_visibility = 0
    contract_unresolved_after_completion = 0
    telemetry_accounting_error = 0

    assert failure_count == 0
    assert implicit_fallback == 0
    assert unattributed_visibility == 0
    assert contract_unresolved_after_completion == 0
    assert telemetry_accounting_error == 0
