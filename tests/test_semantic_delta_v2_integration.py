import json

import pytest

import main as workflow_main
from scripts.evaluate_domain_workflow import add_source_nodes
from workflow_runtime.communication import make_communication_policy
from workflow_runtime.context_slicer import build_context_slice
from workflow_runtime.evaluation_context import EvaluationContext
from workflow_runtime.langgraph_workflow import LangGraphWorkflow
from workflow_runtime.telemetry import WorkflowTelemetry


def _case(domain):
    if domain == "gsm8k":
        return {
            "domain": domain,
            "task_type": "numeric_solve",
            "question": "Alice has 2 apples and Bob has 3 apples. How many apples are there?",
            "gold_answer": "5",
        }
    if domain == "mmlu_pro":
        return {
            "domain": domain,
            "task_type": "multiple_choice",
            "question": "Which number equals two plus two?",
            "question_text": "Which number equals two plus two?",
            "choices": [{"label": "A", "text": "4"}, {"label": "B", "text": "5"}],
            "gold_answer": "A",
        }
    if domain == "mbpp":
        return {
            "domain": domain,
            "task_type": "code_generation",
            "question": "Write add_one(x).",
            "requirements": {"text": "Return x plus one.", "entry_point": "add_one"},
            "tests": [{"setup": "", "text": "assert add_one(1) == 2"}],
            "gold_answer": {"tests": 1},
        }
    if domain == "humaneval":
        return {
            "domain": domain,
            "task_type": "code_generation",
            "question": "Complete identity(x).",
            "requirements": {"text": "Return x.", "entry_point": "identity"},
            "tests": [{"setup": "", "text": "assert identity(7) == 7"}],
            "gold_answer": {"entry_point": "identity"},
        }
    return {
        "domain": "hotpotqa",
        "task_type": "multihop_qa",
        "question": "Where was Alice born?",
        "context_titles": ["Alice", "Paris"],
        "context_entities": ["Alice", "Paris"],
        "supporting_facts": [
            {"title": "Alice", "sent_id": 0, "text": "Alice was born in Paris."},
            {"title": "Paris", "sent_id": 0, "text": "Paris is in France."},
        ],
        "evidence_links": [],
        "gold_answer": "Paris",
    }


def _planner_action(task_type, *, bad_quote=False):
    if task_type == "numeric_solve":
        quote = "Alice has two apples" if bad_quote else "Alice has 2 apples"
        return {
            "op": "plan",
            "facts": [{"content": "the stated quantity is 2", "source": "task", "quote": quote}],
            "strategy": "add the stated quantities",
        }
    operation = {
        "multiple_choice": "compare_options",
        "code_generation": "implement_and_check",
        "multihop_qa": "connect_two_evidence_hops",
    }[task_type]
    return {"op": "plan", "facts": [], "strategy": operation}


def _solver_action(domain, task_type):
    if task_type == "numeric_solve":
        return {"op": "solve_numeric", "expression": "2+3", "value": "5"}
    elif task_type == "multiple_choice":
        return {"op": "choose_option", "answer": "A", "slot_reasons": [{"slot": 1, "reason": "A fits"}], "eliminated_options": {"B": "B does not fit"}, "confidence": 1}
    elif task_type == "code_generation":
        code = "def add_one(x):\n    return x + 1\n" if domain == "mbpp" else "def identity(x):\n    return x\n"
        return {"op": "submit_code", "code": code}
    return {
        "op": "answer_with_evidence", "answer": "Paris", "bridge_entity": "Paris",
        "citations": ["sentence_source_1", "sentence_source_2"],
        "reasoning": ["Alice was born in Paris", "Paris is the answer"],
    }


def _critic_action(task_type):
    if task_type == "numeric_solve":
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
        return {"op": "diagnose_code", "diagnosis": "the implementation matches the task", "repair_instruction": ""}
    reason = "both evidence hops support the answer"
    if task_type == "multihop_qa":
        reason = json.dumps({
            "hop1_conclusion": "Alice was born in Paris.",
            "hop2_conclusion": "The requested birthplace is Paris.",
            "derived_answer": "Paris", "candidate_answer": "Paris",
            "match": True, "contradiction": False, "answers_question": True,
        })
    return {
        "op": "judge_evidence", "answer_support": "supported", "bridge_status": "valid",
        "citation_status": "valid", "repair_target": "none", "reason": reason,
    }


@pytest.mark.parametrize("domain", ["gsm8k", "mmlu_pro", "mbpp", "humaneval", "hotpotqa"])
def test_five_domain_adapters_run_through_semantic_delta_v2(domain):
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import MemorySaver

    row = _case(domain)
    task_id = f"integration-{domain}"
    task_text = row.get("question_text", row["question"])
    store = workflow_main.init_workflow_graph(
        f"[domain={row['task_type']}]\n{task_text}",
        task_id=task_id,
        task_type=row["task_type"],
        action_protocol="semantic_delta_v2",
    )
    add_source_nodes(workflow_main, store, row, task_id, action_protocol="semantic_delta_v2")
    solver_action = _solver_action(domain, row["task_type"])

    def model(request):
        if request.role == "planner":
            return _planner_action(row["task_type"])
        if request.role == "solver":
            return solver_action
        if request.role == "critic":
            return _critic_action(row["task_type"])
        raise AssertionError("semantic_delta_v2 finalization must not invoke the model")

    telemetry = WorkflowTelemetry()
    runtime = LangGraphWorkflow(
        store=store,
        model=model,
        task_id=task_id,
        task_type=row["task_type"],
        action_protocol="semantic_delta_v2",
        communication_policy=make_communication_policy("send_all"),
        telemetry=telemetry,
        evaluation_context=EvaluationContext(
            task_id=task_id,
            task_type=row["task_type"],
            gold_answer=row.get("gold_answer"),
            official_tests=tuple(row.get("tests", ())),
            private_tests=tuple(row.get("tests", ())),
        ),
    )
    state = runtime.compile(checkpointer=MemorySaver()).invoke(
        runtime.initial_state(), {"configurable": {"thread_id": task_id}}
    )

    assert state.get("error") is None
    assert store.latest_valid(task_id, "main", "semantic_delta_planner_boundary") is not None
    assert store.latest_valid(task_id, "main", "semantic_delta_solver_boundary") is not None
    assert store.latest_valid(task_id, "main", "final_answer") is not None


def test_semantic_grounding_error_is_retried_before_atomic_commit():
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import MemorySaver

    row = _case("gsm8k")
    task_id = "integration-grounding-retry"
    store = workflow_main.init_workflow_graph(
        row["question"], task_id=task_id, task_type="numeric_solve", action_protocol="semantic_delta_v2"
    )
    planner_calls = 0
    planner_requests = []

    def model(request):
        nonlocal planner_calls
        if request.role == "planner":
            planner_calls += 1
            planner_requests.append(request)
            return _planner_action("numeric_solve", bad_quote=planner_calls == 1)
        if request.role == "solver":
            return _solver_action("gsm8k", "numeric_solve")
        if request.role == "critic":
            return _critic_action("numeric_solve")
        raise AssertionError("semantic_delta_v2 finalization must not invoke the model")

    runtime = LangGraphWorkflow(
        store=store, model=model, task_id=task_id, task_type="numeric_solve",
        action_protocol="semantic_delta_v2", telemetry=WorkflowTelemetry(),
        communication_policy=make_communication_policy("send_all"),
    )
    state = runtime.compile(checkpointer=MemorySaver()).invoke(
        runtime.initial_state(), {"configurable": {"thread_id": task_id}}
    )

    assert planner_calls == 2
    assert "REFERENCED_GRAPH_NODES" in planner_requests[1].session_prompt
    assert "Alice has 2 apples" in planner_requests[1].session_prompt
    assert len(planner_requests[1].session_prompt) < len(
        planner_requests[0].session_prompt
        + json.dumps(_planner_action("numeric_solve", bad_quote=True))
        + runtime._retry_session_prompt("planner", "normal", ValueError("grounding quote does not occur"))
    )
    planner_log = next(item for item in state["logs"] if item["role"] == "planner")
    assert planner_log["generation"]["attempts"] == 2
    assert "quote does not occur" in planner_log["generation"]["validation_errors"][0]
    assert store.latest_valid(task_id, "main", "final_answer").content == "5"


def test_role_aware_planner_sees_v2_source_payload_content():
    row = _case("hotpotqa")
    task_id = "integration-source-render"
    store = workflow_main.init_workflow_graph(
        row["question"], task_id=task_id, task_type="multihop_qa",
        action_protocol="semantic_delta_v2",
    )
    add_source_nodes(workflow_main, store, row, task_id, action_protocol="semantic_delta_v2")
    runtime = LangGraphWorkflow(
        store=store, model=lambda request: {}, task_id=task_id,
        task_type="multihop_qa", action_protocol="semantic_delta_v2",
        graph_context_mode="role_aware",
    )
    visible_store = runtime.agent_views.visible_store("planner")
    context = build_context_slice(
        visible_store, task_id=task_id, branch_id="main", role="planner",
        policy="planner_state",
    )

    rendered = runtime._render_graph_context(
        context, visible_store.snapshot(), role="planner", mode="normal", action_index=0,
    )

    assert "sentence_source" in rendered
    assert "Alice was born in Paris." in rendered
    assert "ref=state" not in rendered


def test_role_aware_multihop_context_limits_sentences_without_gold_labels():
    task_id = "integration-hotpot-context-limit"
    row = {
        "domain": "hotpotqa",
        "task_type": "multihop_qa",
        "question": "What position did the woman portrayed by Alice hold?",
        "context_titles": ["Alice Film", "Minister Jane"] + [f"Distractor {index}" for index in range(20)],
        "context_entities": ["Alice Film", "Minister Jane"],
        "supporting_facts": [
            {"title": "Alice Film", "sent_id": 0, "text": "Alice portrayed Minister Jane in the film."},
            {"title": "Minister Jane", "sent_id": 0, "text": "Minister Jane held the position of Chief of Protocol."},
            *[
                {"title": f"Distractor {index}", "sent_id": 0, "text": f"Unrelated background material number {index}."}
                for index in range(20)
            ],
        ],
        "evidence_links": [],
        "gold_answer": "Chief of Protocol",
    }
    store = workflow_main.init_workflow_graph(
        row["question"], task_id=task_id, task_type="multihop_qa",
        action_protocol="semantic_delta_v2",
    )
    add_source_nodes(workflow_main, store, row, task_id, action_protocol="semantic_delta_v2")
    runtime = LangGraphWorkflow(
        store=store, model=lambda request: {}, task_id=task_id,
        task_type="multihop_qa", action_protocol="semantic_delta_v2",
        graph_context_mode="role_aware", multihop_sentence_limit=6,
    )
    visible_store = runtime.agent_views.visible_store("planner")
    context = build_context_slice(
        visible_store, task_id=task_id, branch_id="main", role="planner",
        policy="planner_state", multihop_sentence_limit=6,
    )
    sentence_nodes = [
        visible_store.snapshot().nodes[node_id]
        for node_id in context.visible_node_ids
        if visible_store.snapshot().nodes[node_id].type == "sentence_source"
    ]
    rendered = runtime._render_graph_context(
        context, visible_store.snapshot(), role="planner", mode="normal", action_index=0,
    )

    assert len(sentence_nodes) <= 6
    assert "Alice portrayed Minister Jane" in rendered
    assert "Chief of Protocol" in rendered
    assert row["gold_answer"] not in str(row.get("evidence_links", []))
