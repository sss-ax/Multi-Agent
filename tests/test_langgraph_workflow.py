from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from workflow_runtime import GraphStore
from workflow_runtime.communication import make_communication_policy
from workflow_runtime.langgraph_workflow import LangGraphWorkflow, NativeLangGraphWorkflow
from workflow_runtime.telemetry import WorkflowTelemetry


def test_langgraph_dependency_is_reported_at_compile_time():
    store = GraphStore()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="solve",
        owner="user",
    )
    workflow = LangGraphWorkflow(store=store, model=lambda request: {}, task_id="t")
    try:
        workflow.compile()
    except RuntimeError as exc:
        assert "langgraph" in str(exc).lower()
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"unexpected LangGraph error: {exc}")


def test_workflow_contract_requests_one_strict_action():
    workflow = LangGraphWorkflow(store=GraphStore(), model=lambda request: {}, task_id="t")
    contract = workflow._system_contract("planner", "normal")
    assert "one Action" in contract
    assert "GraphStore nodes" in contract


def test_minimal_no_feedback_records_unresolved_semantic_nack(tmp_path):
    store = GraphStore()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="A has 2 items and B has 3 items.",
        owner="user",
    )
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 5},
        owner="solver",
    )
    telemetry = WorkflowTelemetry(tmp_path / "workflow.jsonl")
    workflow = LangGraphWorkflow(
        store=store,
        model=lambda request: {},
        task_id="t",
        task_type="numeric_solve",
        telemetry=telemetry,
        communication_policy=make_communication_policy("minimal_no_feedback"),
    )

    events = workflow._communicate_nodes(sender="solver", node_ids=[result.node_id], branch_id="main")

    assert len(events) == 1
    event = events[0]
    assert event["policy"] == "minimal_no_feedback"
    assert event["semantic_packet_root_node_ids"] == [result.node_id]
    assert event["semantic_feedback_request"] is True
    assert event["semantic_nack"] is False
    assert event["semantic_nack_unresolved"] is False
    assert event["semantic_hard_nack"] is False
    assert event["semantic_soft_nack"] is True
    assert event["semantic_verification_nack"] is True
    assert event["semantic_hard_contract_satisfied"] is True
    assert event["semantic_final_contract_satisfied"] is False
    assert event["semantic_feedback"]["type"] == "VERIFICATION_REQUEST"
    assert event["semantic_feedback"]["missing_semantics"] == ["support_dependencies"]
    assert event["semantic_feedback"]["soft_missing_semantics"] == ["support_dependencies"]
    assert event["semantic_initial_receiver_need"]["level"] == "verification"
    assert event["semantic_initial_receiver_need"]["verification_missing"] == ["support_dependencies"]
    assert event["semantic_receiver_need"]["level"] == "verification"
    assert result.node_id in event["sent_node_ids"]


def test_minimal_sendall_fallback_satisfies_contract_after_nack(tmp_path):
    store = GraphStore()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="A has 2 items and B has 3 items.",
        owner="user",
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
    telemetry = WorkflowTelemetry(tmp_path / "workflow.jsonl")
    workflow = LangGraphWorkflow(
        store=store,
        model=lambda request: {},
        task_id="t",
        task_type="numeric_solve",
        telemetry=telemetry,
        communication_policy=make_communication_policy("minimal_sendall_fallback"),
    )

    events = workflow._communicate_nodes(
        sender="solver",
        node_ids=[calculation.node_id, result.node_id],
        branch_id="main",
    )

    assert len(events) == 1
    event = events[0]
    assert event["policy"] == "minimal_sendall_fallback"
    assert set(event["semantic_packet_root_node_ids"]) == {calculation.node_id, result.node_id}
    assert event["fallback_send_all"] is False
    assert event["fallback_node_ids"] == []
    assert event["fallback_tokens"] == 0
    assert event["wasted_pre_fallback_tokens"] == 0
    assert event["semantic_ack"] is True
    assert event["semantic_nack"] is False
    assert event["semantic_nack_unresolved"] is False
    assert event["semantic_final_contract_satisfied"] is True
    assert set(event["sent_node_ids"]) == {result.node_id, calculation.node_id}


def test_minimal_targeted_feedback_satisfies_contract_without_fallback(tmp_path):
    store = GraphStore()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="A has 2 items and B has 3 items.",
        owner="user",
    )
    facts = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="facts",
        node_type="facts",
        content=[{"id": "A", "value": 2}, {"id": "B", "value": 3}],
        owner="planner",
    )
    plan = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="plan",
        node_type="plan",
        content={"steps": [{"id": "R1", "operation": "add", "inputs": ["A", "B"]}]},
        owner="planner",
    )
    telemetry = WorkflowTelemetry(tmp_path / "workflow.jsonl")
    workflow = LangGraphWorkflow(
        store=store,
        model=lambda request: {},
        task_id="t",
        task_type="numeric_solve",
        telemetry=telemetry,
        communication_policy=make_communication_policy("minimal_targeted_feedback"),
    )
    workflow.agent_views.grant("planner", node_ids=[facts.node_id, plan.node_id], local=True)

    events = workflow._communicate_nodes(
        sender="planner",
        node_ids=[facts.node_id, plan.node_id],
        branch_id="main",
    )

    event = events[0]
    assert event["policy"] == "minimal_targeted_feedback"
    assert set(event["semantic_packet_root_node_ids"]) == {facts.node_id, plan.node_id}
    assert event["targeted_refinement"] is False
    assert event["targeted_refinement_node_ids"] == []
    assert event["targeted_refinement_tokens"] == 0
    assert event["fallback_send_all"] is False
    assert event["fallback_tokens"] == 0
    assert event["semantic_ack"] is True
    assert event["semantic_nack_unresolved"] is False
    assert set(event["sent_node_ids"]) == {plan.node_id, facts.node_id}


def test_minimal_targeted_feedback_falls_back_when_targeted_refinement_is_empty(tmp_path):
    store = GraphStore()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="A has 2 items and B has 3 items.",
        owner="user",
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
    telemetry = WorkflowTelemetry(tmp_path / "workflow.jsonl")
    workflow = LangGraphWorkflow(
        store=store,
        model=lambda request: {},
        task_id="t",
        task_type="numeric_solve",
        telemetry=telemetry,
        communication_policy=make_communication_policy("minimal_targeted_feedback"),
    )

    events = workflow._communicate_nodes(
        sender="solver",
        node_ids=[calculation.node_id, result.node_id],
        branch_id="main",
    )

    event = events[0]
    assert event["targeted_refinement"] is False
    assert event["targeted_refinement_node_ids"] == []
    assert event["targeted_refinement_plan"] is None
    assert event["fallback_send_all"] is False
    assert event["fallback_node_ids"] == []
    assert event["semantic_ack"] is True
    assert event["semantic_nack_unresolved"] is False


def test_langgraph_runs_incremental_actions_to_final_answer(tmp_path):
    store = GraphStore()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="A has 2 items and B has 3 items.",
        owner="user",
    )
    queues = {
        "planner": [
            {"op": "declare_query", "content": {"question": "total"}},
            {"op": "add_fact", "id": "A", "value": 2},
            {"op": "add_fact", "id": "B", "value": 3},
            {"op": "add_plan_step", "id": "R1", "operation": "add", "inputs": ["A", "B"]},
            {"op": "done"},
        ],
        "solver": [
            {"op": "calculate", "id": "R1", "expression": "2+3", "value": 5},
            {"op": "set_result", "id": "R1", "value": 5},
            {"op": "done"},
        ],
        "critic": [
            {"op": "verify", "target": "result", "status": "verified"},
            {"op": "done"},
        ],
        "final_solver": [
            {"op": "answer", "source": "result", "value": 5},
            {"op": "done"},
        ],
    }
    requests = []

    def model(request):
        requests.append(request)
        return queues[request.role].pop(0)

    telemetry = WorkflowTelemetry(tmp_path / "workflow.jsonl")
    workflow = LangGraphWorkflow(
        store=store,
        model=model,
        task_id="t",
        max_actions_per_role=8,
        telemetry=telemetry,
    )
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:  # pragma: no cover
        pytest.skip("LangGraph is not installed")
    result = workflow.compile(checkpointer=MemorySaver()).invoke(
        workflow.initial_state(),
        {"configurable": {"thread_id": "t"}},
    )
    assert result["status"] == "running"
    assert store.latest_valid("t", "main", "final_answer").content == 5
    assert requests
    assert {request.session_id for request in requests} == {"workflow:t:main"}
    assert requests[0].session_reset is True
    assert requests[0].action_constraint.allowed_ops == ("declare_query",)
    assert requests[1].action_constraint.allowed_ops == ("add_fact",)
    assert all(request.session_prompt for request in requests)
    assert all(request.session_reset is True for request in requests)
    assert telemetry.summary()["action_events"] == len(requests)
    assert telemetry.summary()["graph_communication_events"] > 0
    assert telemetry.summary()["graph_delta_sent_nodes"] > 0
    assert (tmp_path / "workflow.jsonl").read_text().count('"event":"model_action"') == len(requests)
    assert '"event":"graph_delta_communication"' in (tmp_path / "workflow.jsonl").read_text()


def test_optimized_workflow_runs_solver_revision_after_need_fix_critic(tmp_path):
    store = GraphStore()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="A has 2 items and B has 3 items.",
        owner="user",
    )
    queues = {
        ("planner", "normal"): [
            {"op": "declare_query", "content": {"question": "total", "task_type": "numeric_solve"}},
            {"op": "add_fact", "id": "A", "value": 2},
            {"op": "add_fact", "id": "B", "value": 3},
            {"op": "add_plan_step", "id": "R1", "operation": "add", "inputs": ["A", "B"]},
            {"op": "done"},
        ],
        ("solver", "normal"): [
            {"op": "calculate", "id": "R1", "expression": "2+3", "value": 6},
            {"op": "set_result", "id": "R1", "value": 6},
            {"op": "done"},
        ],
        ("critic", "normal"): [
            {
                "op": "verify",
                "target": "result",
                "status": "need_fix",
                "error_type": "arithmetic",
                "error_location": "calculation R1",
                "reason": "2+3 was recorded as 6.",
                "repair_instruction": "Recompute R1 as 2+3=5 and update the result.",
                "preserve": ["facts", "plan"],
                "requested_fragments": ["calculation#final_value"],
            },
            {"op": "verify", "target": "result", "status": "verified"},
        ],
        ("solver", "repair"): [
            {"op": "set_result", "id": "R1", "value": 5},
            {"op": "done"},
        ],
        ("final_solver", "finalization"): [
            {"op": "answer", "source": "result", "value": 5},
            {"op": "done"},
        ],
    }
    requests = []

    def model(request):
        requests.append((request.role, request.mode, request.action_constraint.allowed_ops))
        return queues[(request.role, request.mode)].pop(0)

    telemetry = WorkflowTelemetry(tmp_path / "workflow.jsonl")
    workflow = LangGraphWorkflow(
        store=store,
        model=model,
        task_id="t",
        task_type="numeric_solve",
        max_rounds=3,
        max_actions_per_role=8,
        telemetry=telemetry,
    )
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:  # pragma: no cover
        pytest.skip("LangGraph is not installed")

    result = workflow.compile(checkpointer=MemorySaver()).invoke(
        workflow.initial_state(),
        {"configurable": {"thread_id": "t-repair"}},
    )

    assert result["status"] == "running"
    assert store.latest_valid("t", "main", "result").content == {"id": "R1", "value": 5}
    assert store.latest_valid("t", "main", "verification").status == "verified"
    assert store.latest_valid("t", "main", "final_answer").content == 5
    assert ("solver", "repair", ("set_result",)) in requests
    assert [item for item in requests if item[0] == "critic"] == [
        ("critic", "normal", ("verify",)),
        ("critic", "normal", ("verify",)),
    ]
    assert result["round_id"] >= 2
    assert telemetry.summary()["reasoning_round_count"] == len(requests)


def test_numeric_solver_with_existing_result_deterministically_completes_without_done(tmp_path):
    store = GraphStore()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="A has 2 items and B has 3 items.",
        owner="user",
    )
    queues = {
        ("planner", "normal"): [
            {"op": "declare_query", "content": {"question": "total", "task_type": "numeric_solve"}},
            {"op": "add_fact", "id": "A", "value": 2},
            {"op": "add_fact", "id": "B", "value": 3},
            {"op": "add_plan_step", "id": "R1", "operation": "add", "inputs": ["A", "B"]},
            {"op": "done"},
        ],
        ("solver", "normal"): [
            {"op": "calculate", "id": "R1", "expression": "2+3", "value": 5},
            {"op": "set_result", "id": "R1", "value": 5},
        ],
        ("critic", "normal"): [
            {"op": "verify", "target": "result", "status": "verified"},
        ],
        ("final_solver", "finalization"): [
            {"op": "answer", "source": "result", "value": 5},
        ],
    }
    requests = []

    def model(request):
        requests.append((request.role, request.mode, request.action_constraint.allowed_ops))
        return queues[(request.role, request.mode)].pop(0)

    workflow = LangGraphWorkflow(
        store=store,
        model=model,
        task_id="t",
        task_type="numeric_solve",
        max_rounds=1,
        max_actions_per_role=8,
        telemetry=WorkflowTelemetry(tmp_path / "workflow.jsonl"),
    )
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:  # pragma: no cover
        pytest.skip("LangGraph is not installed")

    result = workflow.compile(checkpointer=MemorySaver()).invoke(
        workflow.initial_state(),
        {"configurable": {"thread_id": "t-result-stop"}},
    )

    assert result["status"] == "running"
    assert store.latest_valid("t", "main", "final_answer").content == 5
    assert [item for item in requests if item[0] == "solver"] == [
        ("solver", "normal", ("calculate",)),
        ("solver", "normal", ("set_result",)),
    ]


def test_solver_null_result_is_recovered_before_generation_validation(tmp_path):
    store = GraphStore()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="A has 2 items and B has 3 items.",
        owner="user",
    )
    queues = {
        ("planner", "normal"): [
            {"op": "declare_query", "content": {"question": "total", "task_type": "numeric_solve"}},
            {"op": "add_fact", "id": "A", "value": 2},
            {"op": "add_fact", "id": "B", "value": 3},
            {"op": "add_plan_step", "id": "R1", "operation": "add", "inputs": ["A", "B"]},
            {"op": "done"},
        ],
        ("solver", "normal"): [
            {"op": "calculate", "id": "R1", "expression": "2+3", "value": 5},
            {"op": "set_result", "id": "R1", "value": None},
        ],
        ("critic", "normal"): [
            {"op": "verify", "target": "result", "status": "verified"},
        ],
        ("final_solver", "finalization"): [
            {"op": "answer", "source": "result", "value": 5},
        ],
    }

    def model(request):
        return queues[(request.role, request.mode)].pop(0)

    workflow = LangGraphWorkflow(
        store=store,
        model=model,
        task_id="t",
        task_type="numeric_solve",
        max_rounds=1,
        max_actions_per_role=8,
        telemetry=WorkflowTelemetry(tmp_path / "workflow.jsonl"),
    )
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:  # pragma: no cover
        pytest.skip("LangGraph is not installed")

    workflow.compile(checkpointer=MemorySaver()).invoke(
        workflow.initial_state(),
        {"configurable": {"thread_id": "t-null-recover"}},
    )

    assert store.latest_valid("t", "main", "result").content == {"id": "R1", "value": 5}
    assert store.latest_valid("t", "main", "final_answer").content == 5


def test_numeric_solver_exhaustion_auto_sets_result_from_latest_calculation(tmp_path):
    store = GraphStore()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="A has 2 items and B has 3 items.",
        owner="user",
    )
    queues = {
        ("planner", "normal"): [
            {"op": "declare_query", "content": {"question": "total", "task_type": "numeric_solve"}},
            {"op": "add_fact", "id": "A", "value": 2},
            {"op": "add_fact", "id": "B", "value": 3},
            {"op": "add_plan_step", "id": "R1", "operation": "add", "inputs": ["A", "B"]},
            {"op": "done"},
        ],
        ("solver", "normal"): [
            {"op": "calculate", "id": "R1", "expression": "2+3", "value": 5},
            {"op": "calculate", "id": "R1", "expression": "2+3", "value": 5},
            {"op": "calculate", "id": "R1", "expression": "2+3", "value": 5},
            {"op": "calculate", "id": "R1", "expression": "2+3", "value": 5},
            {"op": "calculate", "id": "R1", "expression": "2+3", "value": 5},
        ],
        ("critic", "normal"): [
            {"op": "verify", "target": "result", "status": "verified"},
        ],
        ("final_solver", "finalization"): [
            {"op": "answer", "source": "result", "value": 5},
        ],
    }

    def model(request):
        return queues[(request.role, request.mode)].pop(0)

    workflow = LangGraphWorkflow(
        store=store,
        model=model,
        task_id="t",
        task_type="numeric_solve",
        max_rounds=1,
        max_actions_per_role=5,
        telemetry=WorkflowTelemetry(tmp_path / "workflow.jsonl"),
    )
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:  # pragma: no cover
        pytest.skip("LangGraph is not installed")

    workflow.compile(checkpointer=MemorySaver()).invoke(
        workflow.initial_state(),
        {"configurable": {"thread_id": "t-calc-exhaustion"}},
    )

    assert store.latest_valid("t", "main", "result").content == {"id": "R1", "value": 5}
    assert store.latest_valid("t", "main", "final_answer").content == 5


def test_task_verifier_overrides_false_positive_code_verified(tmp_path):
    store = GraphStore()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="Write f so that f() returns 1.",
        owner="user",
    )
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="requirements",
        node_type="requirements",
        content={"text": "def f():\n"},
        owner="dataset",
    )
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="test_1",
        node_type="test",
        content={"setup": "", "text": "assert f() == 1"},
        owner="dataset",
    )
    queues = {
        ("planner", "normal"): [
            {"op": "declare_query", "content": {"question": "return one", "task_type": "code_generation"}},
            {"op": "add_plan_step", "id": "M1", "operation": "synthesize", "inputs": ["task"]},
            {"op": "done"},
        ],
        ("solver", "normal"): [
            {"op": "emit_code", "code": "def f():\n    return 2\n"},
            {"op": "done"},
        ],
        ("critic", "normal"): [
            {"op": "verify", "target": "result", "status": "verified"},
            {"op": "verify", "target": "result", "status": "verified"},
        ],
        ("solver", "repair"): [
            {"op": "emit_code", "code": "def f():\n    return 1\n"},
            {"op": "done"},
        ],
        ("final_solver", "finalization"): [
            {"op": "answer", "source": "code", "value": "ignored"},
            {"op": "done"},
        ],
    }

    def model(request):
        return queues[(request.role, request.mode)].pop(0)

    telemetry = WorkflowTelemetry(tmp_path / "workflow.jsonl")
    workflow = LangGraphWorkflow(
        store=store,
        model=model,
        task_id="t",
        task_type="code_generation",
        max_rounds=3,
        max_actions_per_role=6,
        telemetry=telemetry,
    )
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:  # pragma: no cover
        pytest.skip("LangGraph is not installed")

    workflow.compile(checkpointer=MemorySaver()).invoke(
        workflow.initial_state(),
        {"configurable": {"thread_id": "t-code-verifier"}},
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "workflow.jsonl").read_text().splitlines()
        if json.loads(line).get("event") == "model_action"
    ]
    critic_actions = [item["action"] for item in records if item.get("role") == "critic"]
    assert critic_actions[0]["status"] == "need_fix"
    assert critic_actions[0]["error_type"] == "wrong_value"
    assert "must return 1" in critic_actions[0]["repair_instruction"]
    assert critic_actions[1]["status"] == "verified"
    assert telemetry.summary()["critic_need_fix_count"] == 1
    assert telemetry.summary()["solver_revision_round_count"] == 1
    assert store.latest_valid("t", "main", "result").validation["execution_valid"] is True


def test_native_langgraph_accepts_free_form_messages_without_action_protocol(tmp_path):
    calls = []

    def model(request):
        calls.append(request)
        if request.role == "planner":
            return "```text\nFirst add the two quantities.\n```"
        if request.role == "solver":
            return "The total is 5."
        if request.role == "critic":
            return "The candidate is correct and complete."
        if request.role == "final_solver":
            return "**5**"
        raise AssertionError(request.role)

    telemetry = WorkflowTelemetry(tmp_path / "native.jsonl")
    workflow = NativeLangGraphWorkflow(
        model=model,
        task_id="native-t",
        task="A has 2 items and B has 3 items. What is the total?",
        task_type="numeric_solve",
        telemetry=telemetry,
    )
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:  # pragma: no cover
        pytest.skip("LangGraph is not installed")
    result = workflow.compile(checkpointer=MemorySaver()).invoke(
        workflow.initial_state(),
        {"configurable": {"thread_id": "native-t"}},
    )

    assert result["final_answer"] == "**5**"
    assert [request.role for request in calls] == ["planner", "solver", "critic", "final_solver"]
    assert all(request.action_constraint is None for request in calls)
    assert all(request.session_prompt for request in calls)
    assert telemetry.summary()["native_model_calls"] == 4
    assert '"event":"native_model_call"' in (tmp_path / "native.jsonl").read_text()
    transcript = workflow.message_bus.transcript()
    assert result["messages"] == transcript
    assert any(
        item["sender_id"] == "native:native-t:planner"
        and item["recipient_ids"] == ["native:native-t:solver"]
        and item["message_type"] == "plan"
        for item in transcript
    )
    assert any(
        item["sender_id"] == "native:native-t:final_solver"
        and item["recipient_ids"] == ["user"]
        and item["message_type"] == "final_answer"
        for item in transcript
    )
    assert all(
        status == "acknowledged"
        for item in transcript
        if item["message_type"] != "final_answer"
        for status in item["delivery"].values()
    )
    assert transcript[-1]["delivery"] == {"user": "pending"}


def test_native_renders_each_role_as_an_independent_chat_request():
    class Tokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            self.calls.append((messages, tokenize, add_generation_prompt))
            return f"rendered-{len(self.calls)}"

    class Model:
        tokenizer = Tokenizer()

    workflow = NativeLangGraphWorkflow(
        model=Model(),
        task_id="native-template",
        task="2 + 3",
        task_type="numeric_solve",
    )

    first = workflow._render_prompt("planner", "task text")
    second = workflow._render_prompt("solver", "task text\nPLANNER: plan")
    assert (first, second) == ("rendered-1", "rendered-2")
    calls = workflow.model.tokenizer.calls
    assert calls[0][1:] == (False, True)
    assert calls[0][0][0]["role"] == "system"
    assert calls[0][0][1] == {"role": "user", "content": "task text"}
    assert calls[1][0][1]["content"] == "task text\nPLANNER: plan"


def test_native_has_independent_agent_identity_memory_policy_and_lifecycle():
    calls = []

    def model(request):
        calls.append(request)
        if request.role == "planner":
            return "Plan: add the quantities."
        if request.role == "solver":
            return "The total is 5."
        if request.role == "critic":
            return "The candidate is correct and complete."
        if request.role == "final_solver":
            return "5"
        raise AssertionError(request.role)

    workflow = NativeLangGraphWorkflow(
        model=model,
        task_id="native-agents",
        task="A has 2 items and B has 3 items.",
        task_type="numeric_solve",
    )

    assert set(workflow.agents) == {"planner", "solver", "critic", "final_solver"}
    assert len({agent.agent_id for agent in workflow.agents.values()}) == 4
    assert workflow.agents["solver"].can_use_tool("calculator") is True
    assert workflow.agents["planner"].can_use_tool("calculator") is False
    assert workflow.agents["solver"].execute_tool("calculator", {"expression": "2 + 3"}) == 5
    with pytest.raises(ValueError, match="not allowed"):
        workflow.agents["planner"].execute_tool("calculator", {"expression": "2 + 3"})

    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:  # pragma: no cover
        pytest.skip("LangGraph is not installed")
    result = workflow.compile(checkpointer=MemorySaver()).invoke(
        workflow.initial_state(),
        {"configurable": {"thread_id": "native-agents"}},
    )

    assert result["final_answer"] == "5"
    assert {request.agent_id for request in calls} == {
        "native:native-agents:planner",
        "native:native-agents:solver",
        "native:native-agents:critic",
        "native:native-agents:final_solver",
    }
    assert all(request.agent_config is workflow.agents[request.role].config for request in calls)
    assert workflow.agents["planner"].memory.messages
    assert workflow.agents["planner"].memory.messages != workflow.agents["solver"].memory.messages
    assert all(agent.lifecycle.state == "completed" for agent in workflow.agents.values())
    assert result["agent_snapshots"]["solver"]["tool_permissions"] == [
        "calculator",
        "json_validate",
        "python_syntax_check",
    ]


def test_native_can_inject_distinct_models_per_agent():
    model_calls = []

    def make_model(role):
        def model(request):
            model_calls.append((role, request.agent_id))
            return role
        return model

    models = {role: make_model(role) for role in ("planner", "solver", "critic", "final_solver")}
    workflow = NativeLangGraphWorkflow(
        model=lambda request: "default",
        agent_models=models,
        task_id="native-models",
        task="task",
        task_type="numeric_solve",
    )

    for role, agent in workflow.agents.items():
        assert agent.model is models[role]

    assert workflow.agents["planner"].model is not workflow.agents["solver"].model


def test_native_tool_node_executes_explicit_call_and_updates_environment_state():
    calls = []

    def model(request):
        calls.append(request.role)
        if request.role == "planner":
            return "Use a deterministic calculation."
        if request.role == "solver":
            return '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 3 * 4"}}</tool_call> The total is 14.'
        if request.role == "critic":
            return "The tool result verifies the candidate."
        if request.role == "final_solver":
            return "14"
        raise AssertionError(request.role)

    workflow = NativeLangGraphWorkflow(
        model=model,
        task_id="native-tool-success",
        task="Calculate 2 + 3 * 4.",
        task_type="numeric_solve",
    )
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:  # pragma: no cover
        pytest.skip("LangGraph is not installed")
    result = workflow.compile(checkpointer=MemorySaver()).invoke(
        workflow.initial_state(),
        {"configurable": {"thread_id": "native-tool-success"}},
    )

    successful = [item for item in result["tool_results"] if item["ok"]]
    assert result["tool_requests"][0]["name"] == "calculator"
    assert successful[-1]["name"] == "calculator"
    assert successful[-1]["output"] == 14
    assert result["environment_state"]["tool_call_count"] == 1
    assert result["environment_state"]["last_tool_result"]["output"] == 14
    assert result["tool_failures"] == 0
    assert calls == ["planner", "solver", "critic", "final_solver"]


def test_native_tool_failure_sends_error_to_solver_and_retries():
    solver_calls = 0

    def model(request):
        nonlocal solver_calls
        if request.role == "planner":
            return "Use the calculator."
        if request.role == "solver":
            solver_calls += 1
            if solver_calls == 1:
                return '<tool_call>{"name":"calculator","arguments":{"expression":"1 / 0"}}</tool_call>'
            return '<tool_call>{"name":"calculator","arguments":{"expression":"10 / 2"}}</tool_call>'
        if request.role == "critic":
            return "The tool result is correct."
        if request.role == "final_solver":
            return "5"
        raise AssertionError(request.role)

    workflow = NativeLangGraphWorkflow(
        model=model,
        task_id="native-tool-retry",
        task="Calculate 10 / 2.",
        task_type="numeric_solve",
        max_tool_retries=2,
    )
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:  # pragma: no cover
        pytest.skip("LangGraph is not installed")
    result = workflow.compile(checkpointer=MemorySaver()).invoke(
        workflow.initial_state(),
        {"configurable": {"thread_id": "native-tool-retry"}},
    )

    assert solver_calls == 2
    assert result["tool_attempts"] == 2
    assert result["tool_failures"] == 1
    assert result["tool_results"][0]["ok"] is False
    assert result["tool_results"][-1]["output"] == 5
    assert len(result["tool_requests"]) == 2
    assert any(item["message_type"] == "tool_error" for item in workflow.message_bus.transcript())


def test_native_critic_structured_verdict_controls_repair_route():
    solver_calls = 0
    critic_calls = 0

    def model(request):
        nonlocal solver_calls, critic_calls
        if request.role == "planner":
            return "Plan."
        if request.role == "solver":
            solver_calls += 1
            return "Initial candidate." if solver_calls == 1 else "Repaired candidate."
        if request.role == "critic":
            critic_calls += 1
            if critic_calls == 1:
                return (
                    '<critic_verdict>{"verdict":"needs_repair","confidence":0.95,'
                    '"target":"candidate","issues":["missing evidence"],'
                    '"repair_instructions":"Add the missing evidence."}</critic_verdict>'
                )
            return (
                '<critic_verdict>{"verdict":"pass","confidence":0.98,'
                '"target":"candidate","issues":[],"repair_instructions":""}</critic_verdict>'
            )
        if request.role == "final_solver":
            return "Final answer."
        raise AssertionError(request.role)

    workflow = NativeLangGraphWorkflow(
        model=model,
        task_id="native-critic-verdict",
        task="Task.",
        task_type="numeric_solve",
    )
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:  # pragma: no cover
        pytest.skip("LangGraph is not installed")
    result = workflow.compile(checkpointer=MemorySaver()).invoke(
        workflow.initial_state(),
        {"configurable": {"thread_id": "native-critic-verdict"}},
    )

    assert result["final_answer"] == "Final answer."
    assert solver_calls == 2
    assert critic_calls == 2
    assert result["critic_verdict"]["verdict"] == "pass"
