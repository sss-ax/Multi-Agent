from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from workflow_runtime import GraphStore
from workflow_runtime.action_compiler import ActionCompilationError, ActionCompiler
from workflow_runtime.domain_executors import execute_domain


def make_compiler() -> tuple[GraphStore, ActionCompiler]:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="A plus B", owner="user")
    return store, ActionCompiler(store, task_id="t", task_type="numeric_solve")


def test_planner_actions_compile_to_graph_nodes_without_direct_graph_mutation() -> None:
    store, compiler = make_compiler()
    compiler.apply("planner", {"op": "declare_query", "content": {"question": "total"}})
    compiler.apply("planner", {"op": "add_fact", "id": "A", "value": 2})
    compiler.apply("planner", {"op": "add_fact", "id": "B", "value": 3})
    compiler.apply("planner", {"op": "add_plan_step", "id": "R1", "operation": "add", "inputs": ["A", "B"]})

    assert store.latest_valid("t", "main", "query_spec") is not None
    assert store.latest_valid("t", "main", "facts").content == [{"id": "A", "value": 2}, {"id": "B", "value": 3}]
    assert store.latest_valid("t", "main", "plan") is not None
    assert store.latest_valid("t", "main", "plan_steps") is not None
    assert store.latest_valid("t", "main", "task").content == "A plus B"


def test_solver_critic_and_final_actions_compile() -> None:
    store, compiler = make_compiler()
    compiler.apply("planner", {"op": "declare_query", "content": {"question": "total"}})
    compiler.apply("planner", {"op": "add_fact", "id": "A", "value": 2})
    compiler.apply("planner", {"op": "add_fact", "id": "B", "value": 3})
    compiler.apply("planner", {"op": "add_plan_step", "id": "R1", "operation": "add", "inputs": ["A", "B"]})
    compiler.apply("solver", {"op": "calculate", "id": "R1", "expression": "2+3", "value": 5})
    compiler.apply("solver", {"op": "set_result", "id": "R1", "value": 5})
    compiler.apply("critic", {"op": "verify", "target": "result", "status": "verified"})
    compiler.apply("final_solver", {"op": "answer", "source": "result", "value": 5})

    assert store.latest_valid("t", "main", "verification").status == "verified"
    assert store.latest_valid("t", "main", "final_answer").content == 5


def test_critic_verify_accepts_answer_alias_for_result() -> None:
    store = GraphStore()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "answer", "value": "yes"},
        owner="solver",
    )
    compiler = ActionCompiler(store, task_id="t", task_type="multihop_qa")
    compiler.apply("critic", {"op": "verify", "target": "answer", "status": "verified"})
    verification = store.latest_valid("t", "main", "verification")
    assert verification is not None
    assert verification.content == {"target": "result", "status": "verified"}


def test_final_answer_falls_back_from_pseudo_source_alias() -> None:
    store, compiler = make_compiler()
    compiler.apply("solver", {"op": "set_result", "id": "R1", "value": 5})
    result = store.latest_valid("t", "main", "result")

    compiler.apply("final_solver", {"op": "answer", "source": "n10", "value": 5})

    final = store.latest_valid("t", "main", "final_answer")
    assert final.content == 5
    assert final.provenance["declared_answer_source"] == "n10"
    assert final.provenance["resolved_answer_source"] == result.node_id
    assert final.provenance["answer_source_fallback"] is True


def test_final_answer_falls_back_from_hotpot_pseudo_source() -> None:
    store = GraphStore()
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "answer", "value": "bridge entity"},
        owner="solver",
    )
    compiler = ActionCompiler(store, task_id="t", task_type="multihop_qa")

    compiler.apply("final_solver", {"op": "answer", "source": "R.id.n3", "value": "ignored"})

    final = store.latest_valid("t", "main", "final_answer")
    assert final.content == "bridge entity"
    assert final.provenance["declared_answer_source"] == "R.id.n3"
    assert final.provenance["resolved_answer_source"] == result.node_id
    assert final.provenance["answer_source_fallback"] is True


def test_solver_null_result_value_reuses_latest_valid_result() -> None:
    store, compiler = make_compiler()
    compiler.apply("solver", {"op": "set_result", "id": "R1", "value": 9})

    compiler.apply("solver", {"op": "set_result", "id": "R2", "value": None})

    result = store.latest_valid("t", "main", "result")
    assert result.content == {"id": "R2", "value": 9}


def test_final_answer_resolves_fragment_alias_to_canonical_node() -> None:
    store, compiler = make_compiler()
    compiler.apply("solver", {"op": "set_result", "id": "R1", "value": 5})
    result = store.latest_valid("t", "main", "result")
    compiler.set_node_ref_context({
        "fragment_index": {
            "result@v1#final_value": {
                "fragment_id": "result@v1#final_value",
                "fragment_type": "final_value",
                "source_node_id": result.node_id,
                "source_logical_id": "result",
                "source_node_type": "result",
                "local_node_id": "n4",
            },
            "final_value": {
                "fragment_id": "result@v1#final_value",
                "fragment_type": "final_value",
                "source_node_id": result.node_id,
                "source_logical_id": "result",
                "source_node_type": "result",
                "local_node_id": "n4",
            },
        },
        "local_node_index": {
            "n4": {
                "source_node_id": result.node_id,
                "source_logical_id": "result",
                "source_node_type": "result",
                "local_node_id": "n4",
            },
        },
    })

    compiler.apply("final_solver", {"op": "answer", "source": "final_value", "value": 5})

    final = store.latest_valid("t", "main", "final_answer")
    assert final.content == 5
    assert final.provenance["resolved_answer_source"] == result.node_id


def test_final_answer_redirects_verification_to_value_bearing_target() -> None:
    store, compiler = make_compiler()
    compiler.apply("solver", {"op": "set_result", "id": "R1", "value": "I"})
    result = store.latest_valid("t", "main", "result")
    compiler.apply("critic", {"op": "verify", "target": "result", "status": "verified"})
    verification = store.latest_valid("t", "main", "verification")

    compiler.apply("final_solver", {"op": "answer", "source": "verification", "value": "verified"})

    final = store.latest_valid("t", "main", "final_answer")
    assert final.content == "I"
    assert final.provenance["resolved_answer_source"] == verification.node_id
    assert final.provenance["value_answer_source"] == result.node_id
    assert final.provenance["value_answer_source_type"] == "result"


def test_final_answer_rejects_non_value_source_without_redirect() -> None:
    store, compiler = make_compiler()
    compiler.apply("planner", {"op": "add_plan_step", "id": "R1", "operation": "add", "inputs": []})

    with pytest.raises(ActionCompilationError, match="invalid final answer source type"):
        compiler.apply("final_solver", {"op": "answer", "source": "plan", "value": "ignored"})


def test_code_generation_uses_code_as_answer_boundary() -> None:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="code", node_type="code", content="def f():\n    return 1\n", owner="solver")
    compiler = ActionCompiler(store, task_id="t", task_type="code_generation")

    compiler.apply("critic", {"op": "verify", "target": "answer", "status": "verified"})
    compiler.apply("final_solver", {"op": "answer", "source": "code", "value": "ignored"})

    assert store.latest_valid("t", "main", "verification").content == {"target": "code", "status": "verified"}
    assert store.latest_valid("t", "main", "final_answer").content == "def f():\n    return 1\n"


def test_need_fix_verify_stores_structured_repair_feedback() -> None:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="code", node_type="code", content="def f():\n    return 2\n", owner="solver")
    compiler = ActionCompiler(store, task_id="t", task_type="code_generation")

    compiler.apply(
        "critic",
        {
            "op": "verify",
            "target": "code",
            "status": "need_fix",
            "error_type": "wrong_return",
            "error_location": "return statement",
            "reason": "The function returns 2 instead of 1.",
            "repair_instruction": "Change the return value to 1 and preserve the signature.",
            "preserve": ["function signature"],
            "requested_fragments": ["code#function_body"],
        },
    )

    verification = store.latest_valid("t", "main", "verification")
    assert verification.status == "need_fix"
    assert verification.content["error_type"] == "wrong_return"
    assert verification.content["repair_instruction"].startswith("Change the return value")
    assert verification.content["preserve"] == ["function signature"]


def test_code_generation_final_answer_falls_back_from_local_source_alias() -> None:
    store = GraphStore()
    code = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="code",
        node_type="code",
        content="def f():\n    return 2\n",
        owner="solver",
    )
    compiler = ActionCompiler(store, task_id="t", task_type="code_generation")

    compiler.apply("final_solver", {"op": "answer", "source": "n8", "value": "ignored"})

    final = store.latest_valid("t", "main", "final_answer")
    assert final.content == "def f():\n    return 2\n"
    assert final.provenance["declared_answer_source"] == "n8"
    assert final.provenance["resolved_answer_source"] == code.node_id
    assert final.provenance["answer_source_fallback"] is True


def test_code_generation_final_answer_prefers_code_over_execution_for_bad_source() -> None:
    store = GraphStore()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="code",
        node_type="code",
        content="def f():\n    return 3\n",
        owner="solver",
    )
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="execution",
        node_type="execution",
        content={"tests_passed": True, "stdout": ""},
        owner="tool",
    )
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "execution", "value": {"tests_passed": True}},
        owner="tool",
    )
    compiler = ActionCompiler(store, task_id="t", task_type="code_generation")

    compiler.apply("final_solver", {"op": "answer", "source": "n4", "value": "ignored"})

    assert store.latest_valid("t", "main", "final_answer").content == "def f():\n    return 3\n"


def test_duplicate_fact_and_plan_step_are_noops() -> None:
    store, compiler = make_compiler()
    first_fact = compiler.apply("planner", {"op": "add_fact", "id": "A", "value": 2})
    second_fact = compiler.apply("planner", {"op": "add_fact", "id": "A", "value": 2})
    first_step = compiler.apply("planner", {"op": "add_plan_step", "id": "R1", "operation": "add", "inputs": ["A"]})
    second_step = compiler.apply("planner", {"op": "add_plan_step", "id": "R1", "operation": "add", "inputs": ["A"]})

    assert first_fact.node_ids
    assert second_fact.node_ids == ()
    assert first_step.node_ids
    assert second_step.node_ids == ()
    assert store.latest_valid("t", "main", "facts").content == [{"id": "A", "value": 2}]


def test_final_action_dereferences_source_value_and_unknown_target_is_rejected() -> None:
    store, compiler = make_compiler()
    compiler.apply("solver", {"op": "set_result", "id": "R1", "value": 5})
    compiler.apply("final_solver", {"op": "answer", "source": "result", "value": "The answer is 5."})
    assert store.latest_valid("t", "main", "final_answer").content == 5
    with pytest.raises(ActionCompilationError, match="does not exist"):
        compiler.apply("critic", {"op": "verify", "target": "missing", "status": "verified"})


def test_solver_tool_action_is_allowlisted_and_compiled_with_result():
    store, compiler = make_compiler()
    result = compiler.apply(
        "solver",
        {
            "op": "call_tool",
            "call_id": "calc-1",
            "name": "calculator",
            "arguments": {"expression": "2 + 3 * 4"},
        },
    )
    assert len(result.node_ids) == 2
    tool_result = store.latest_valid("t", "main", "tool_result_calc-1")
    assert tool_result.status == "verified"
    assert tool_result.content["output"] == 14


def test_unknown_tool_is_recorded_as_a_recoverable_tool_error():
    store, compiler = make_compiler()
    compiler.apply(
        "solver",
        {
            "op": "call_tool",
            "call_id": "bad-1",
            "name": "not-registered",
            "arguments": {},
        },
    )
    tool_result = store.latest_valid("t", "main", "tool_result_bad-1")
    assert tool_result.status == "need_fix"
    assert tool_result.content["ok"] is False


def test_code_executor_prepends_imports_from_requirements() -> None:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="write first", owner="user")
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="requirements",
        node_type="requirements",
        content={"text": "from typing import List\n\ndef first(xs: List[int]) -> int:\n"},
        owner="dataset",
    )
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="test_1",
        node_type="test",
        content={"setup": "", "text": "assert first([3, 4]) == 3"},
        owner="dataset",
    )

    execution = execute_domain(
        "code_generation",
        store,
        {"code": "def first(xs: List[int]) -> int:\n    return xs[0]\n"},
    )

    assert execution.success
