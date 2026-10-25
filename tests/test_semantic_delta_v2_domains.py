import pytest

from workflow_runtime import GraphStore, NodeOrigin, VisibilityClass
from workflow_runtime.communication import make_communication_policy
from workflow_runtime.langgraph_workflow import LangGraphWorkflow


CASES = (
    ("gsm8k", "numeric_solve"),
    ("mmlu_pro", "multiple_choice"),
    ("mbpp", "code_generation"),
    ("humaneval", "code_generation"),
    ("hotpotqa", "multihop_qa"),
)


def _source_graph(dataset: str, task_type: str):
    task_id = f"v2-{dataset}"
    store = GraphStore()

    def add(logical_id, node_type, content):
        return store.add_node(
            task_id=task_id,
            branch_id="main",
            logical_id=logical_id,
            node_type=node_type,
            content=content,
            owner="dataset",
            created_by_role="dataset",
            provenance={
                "origin": NodeOrigin.SOURCE,
                "trust": "trusted_input",
                "visibility_class": VisibilityClass.PUBLIC,
                "source_dataset": dataset,
            },
        )

    if task_type == "numeric_solve":
        task = add("task", "task_source", "Alice has 2 apples and Bob has 3 apples.")
    elif task_type == "multiple_choice":
        task = add("task", "question_source", "Which option equals four?")
        add("choice_source_A", "choice_source", {"label": "A", "text": "4"})
        add("choice_source_B", "choice_source", {"label": "B", "text": "5"})
        add("choice_schema", "choice_schema", {"allowed_labels": ["A", "B"]})
    elif task_type == "code_generation":
        task = add("task", "task_source", "Write solve() so it returns 1.")
        add("requirements_source", "requirements_source", {"task_ref": task.node_id})
        add("entry_point_source", "entry_point_source", {"entry_point": "solve"})
        add("public_test_source_1", "public_test_source", {"test_id": "public_1"})
    else:
        task = add("task", "question_source", "Where was the bridge person born?")
        add("sentence_source_1", "sentence_source", {"title": "A", "sent_id": 0, "text": "The bridge person was born in Paris."})
        add("sentence_source_2", "sentence_source", {"title": "B", "sent_id": 0, "text": "Paris is in France."})
    query = store.add_node(
        task_id=task_id,
        branch_id="main",
        logical_id="query_spec",
        node_type="query_spec",
        content={"task_type": task_type, "task_ref": task.node_id},
        owner="runtime",
        created_by_role="runtime",
        provenance={
            "origin": NodeOrigin.RUNTIME,
            "visibility_class": VisibilityClass.PUBLIC,
            "communication_scope": "MANDATORY",
        },
    )
    store.add_edge(source=task.node_id, target=query.node_id, relation="depends_on", created_by_role="runtime")
    return store, task_id, task


def _planner_delta(task_type: str, task_node):
    if task_type not in {"numeric_solve", "numeric_comparison"}:
        return {"op": "plan", "facts": [], "strategy": "solve from the provided source data"}
    return {
        "op": "plan", "facts": [{"content": "there are 2 apples", "source": "task", "quote": "2 apples"}],
        "strategy": "add the quantities",
    }


def _solver_delta(task_type: str):
    if task_type == "numeric_solve":
        return {"op": "solve_numeric", "expression": "2+3", "value": "5"}
    elif task_type == "multiple_choice":
        return {"op": "choose_option", "answer": "A", "slot_reasons": [{"slot": 1, "reason": "A fits"}], "eliminated_options": {"B": "B does not fit"}, "confidence": 1}
    elif task_type == "code_generation":
        return {"op": "submit_code", "code": "def solve():\n    return 1\n"}
    return {
        "op": "answer_with_evidence", "answer": "Paris", "bridge_entity": "Paris",
        "citations": ["sentence_source_1", "sentence_source_2"],
        "reasoning": ["sentence_source_1 -> Paris", "sentence_source_2 confirms France"],
    }


def _critic_delta(task_type: str):
    if task_type in {"numeric_solve", "numeric_comparison"}:
        return {
            "op": "judge_numeric", "model_verdict": "consistent", "error_type": "none",
            "error_location": "", "reason": "calculation is consistent", "repair_instruction": "",
        }
    if task_type == "multiple_choice":
        return {
            "op": "audit_choice",
            "selected_option_check": {"status": "supported", "reason": "The selected option fits."},
            "slot_checks": [{"slot": 1, "status": "supported", "reason": "The selected term fits."}],
            "eliminated_checks": [{"option": "B", "status": "supported", "reason": "The challenger conflicts."}],
            "summary": "The candidate claims were checked.",
        }
    if task_type == "code_generation":
        return {"op": "diagnose_code", "diagnosis": "implementation is consistent", "repair_instruction": ""}
    return {
        "op": "judge_evidence", "answer_support": "supported", "bridge_status": "valid",
        "citation_status": "valid", "repair_target": "none", "reason": "both hops support Paris",
    }


@pytest.mark.parametrize("dataset,task_type", CASES)
def test_five_dataset_source_delta_commit_receiver_view_and_final_boundary(dataset, task_type):
    store, task_id, task = _source_graph(dataset, task_type)
    workflow = LangGraphWorkflow(
        store=store,
        model=lambda request: {},
        task_id=task_id,
        task_type=task_type,
        action_protocol="semantic_delta_v2",
        communication_policy=make_communication_policy("send_all"),
    )

    workflow.compiler.set_node_ref_context({"visible_node_ids": workflow.agent_views.view("planner").visible_node_ids})
    planned = workflow.compiler.apply("planner", _planner_delta(task_type, task))
    workflow.agent_views.grant("planner", node_ids=planned.node_ids, edge_ids=planned.edge_ids, local=True)
    workflow._communicate_nodes(sender="planner", node_ids=planned.node_ids, branch_id="main")

    workflow.compiler.set_node_ref_context({"visible_node_ids": workflow.agent_views.view("solver").visible_node_ids})
    solved = workflow.compiler.apply("solver", _solver_delta(task_type))
    workflow.agent_views.grant("solver", node_ids=solved.node_ids, edge_ids=solved.edge_ids, local=True)
    workflow._communicate_nodes(sender="solver", node_ids=solved.node_ids, branch_id="main")

    target = "code" if task_type == "code_generation" else "result"
    critic_action = _critic_delta(task_type)
    verified = workflow.compiler.compile_critic_verification("critic", critic_action, {
        "op": "verify", "target": target, "status": "verified", "error_type": "",
        "error_location": "", "reason": "", "repair_instruction": "",
        "preserve": [], "requested_fragments": [],
    })
    workflow.agent_views.grant("critic", node_ids=verified.node_ids, local=True)
    workflow._communicate_nodes(sender="critic", node_ids=verified.node_ids, branch_id="main")

    source = store.latest_valid(task_id, "main", target)
    assert source is not None
    if task_type == "code_generation":
        final_value = source.content["code"]
    else:
        final_value = source.content["value"]
    workflow.compiler.copy_verified_result()

    final = store.latest_valid(task_id, "main", "final_answer")
    assert final is not None and final.content == final_value
    assert store.latest_valid(task_id, "main", "semantic_delta_solver_boundary") is not None
    assert target in {node.logical_id for node in store.snapshot().nodes.values()}


def test_numeric_v2_runs_through_the_real_langgraph_control_flow():
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:  # pragma: no cover
        pytest.skip("LangGraph is not installed")

    store, task_id, task = _source_graph("gsm8k", "numeric_solve")
    actions = {
        ("planner", "normal"): _planner_delta("numeric_solve", task),
        ("solver", "normal"): _solver_delta("numeric_solve"),
        ("critic", "normal"): {
            "op": "judge_numeric", "model_verdict": "consistent", "error_type": "none",
            "error_location": "", "reason": "calculation is consistent", "repair_instruction": "",
        },
    }

    def model(request):
        assert request.role != "final_solver"
        return actions[(request.role, request.mode)]

    workflow = LangGraphWorkflow(
        store=store,
        model=model,
        task_id=task_id,
        task_type="numeric_solve",
        action_protocol="semantic_delta_v2",
        communication_policy=make_communication_policy("send_all"),
    )
    state = workflow.compile(checkpointer=MemorySaver()).invoke(
        workflow.initial_state(),
        {"configurable": {"thread_id": "numeric-semantic-delta-v2"}},
    )
    assert not state.get("error")
    assert store.latest_valid(task_id, "main", "final_answer").content == "5"
