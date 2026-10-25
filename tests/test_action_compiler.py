import pytest
import json

from workflow_runtime.action_compiler import ActionCompilationError, ActionCompiler
from workflow_runtime.domain_executors import execute_domain
from workflow_runtime.graph_store import GraphStore
from workflow_runtime.choice_audit import build_choice_audit_manifest


def _numeric_compiler():
    store = GraphStore()
    task = store.add_node(
        task_id="t", branch_id="main", logical_id="task", node_type="task_source",
        content="Alice has 2 apples and Bob has 3 apples.", owner="dataset",
        provenance={"origin": "source", "visibility_class": "public"},
    )
    store.add_node(
        task_id="t", branch_id="main", logical_id="query_spec", node_type="query_spec",
        content={"task_type": "numeric_solve", "task_ref": task.node_id}, owner="runtime",
    )
    compiler = ActionCompiler(store, task_id="t", task_type="numeric_solve")
    compiler.set_node_ref_context({"visible_node_ids": tuple(store.snapshot().nodes)})
    return store, compiler


def test_typed_actions_compile_and_kernel_finalizes_verified_result():
    store, compiler = _numeric_compiler()
    planned = compiler.apply("planner", {
        "op": "plan",
        "facts": [{"content": "Alice has 2 apples", "source": "task", "quote": "2 apples"}],
        "strategy": "add both quantities",
    })
    assert planned.node_ids
    compiler.set_node_ref_context({"visible_node_ids": tuple(store.snapshot().nodes)})
    solved = compiler.apply("solver", {"op": "solve_numeric", "expression": "2+3", "value": "5"})
    assert solved.node_ids
    judged = compiler.apply("critic", {
        "op": "judge_numeric", "model_verdict": "consistent", "error_type": "none",
        "error_location": "", "reason": "2+3=5", "repair_instruction": "",
    })
    assert judged.node_ids
    finalized = compiler.copy_verified_result()
    assert finalized.node_ids
    assert store.latest_valid("t", "main", "final_answer").content == "5"


def test_choice_solver_reasons_are_resolved_against_source_option_slots():
    store = GraphStore()
    store.add_node(
        task_id="choice", branch_id="main", logical_id="choice_schema", node_type="choice_schema",
        content={"allowed_labels": ["A", "I"], "slot_values": {
            "A": ["Safe practices", "Fear", "Jealousy", "Trivial"],
            "I": ["Unsafe practices", "Distress", "Fear", "Serious"],
        }}, owner="dataset",
    )
    compiler = ActionCompiler(store, task_id="choice", task_type="multiple_choice")
    action = {
        "op": "choose_option", "answer": "I",
        "slot_reasons": [{"slot": i, "reason": f"slot {i} supports I"} for i in range(1, 5)],
        "option_reviews": [
            {"label": "A", "assessment": "rejected", "reason": "slot 1 and slot 4 conflict"},
            {"label": "I", "assessment": "selected", "reason": "all slots fit"},
        ],
        "confidence": 0.9,
    }
    compiler.apply("solver", action)
    value = store.latest_valid("choice", "main", "result").content["value"]
    assert [item["selected_term"] for item in value["slot_analysis"]] == [
        "Unsafe practices", "Distress", "Fear", "Serious"
    ]
    assert value["slot_reasons"] == action["slot_reasons"]


def test_old_model_actions_are_not_compilable():
    _store, compiler = _numeric_compiler()
    for role, action in (
        ("planner", {"op": "add_fact", "id": "A", "value": 2}),
        ("solver", {"op": "set_result", "id": "R1", "value": 5}),
        ("critic", {"op": "verify", "target": "result", "status": "verified"}),
        ("final_solver", {"op": "answer", "source": "result", "value": 5}),
    ):
        with pytest.raises(ActionCompilationError):
            compiler.apply(role, action)


def test_choice_critic_disagreement_compiles_need_fix():
    store = GraphStore()
    store.add_node(
        task_id="t", branch_id="main", logical_id="task", node_type="task_source",
        content="Choose the right term.", owner="dataset",
    )
    store.add_node(task_id="t", branch_id="main", logical_id="choice_schema", node_type="choice_schema", content={"allowed_labels": ["A", "B"], "slot_values": {"A": ["wrong"], "B": ["right"]}}, owner="dataset")
    for label, text in (("A", "wrong"), ("B", "right")):
        store.add_node(task_id="t", branch_id="main", logical_id=f"choice_source_{label}", node_type="choice_source", content={"label": label, "text": text}, owner="dataset")
    compiler = ActionCompiler(store, task_id="t", task_type="multiple_choice")
    compiler.apply("solver", {"op": "choose_option", "answer": "A", "slot_reasons": [{"slot": 1, "reason": "wrong"}], "option_reviews": [{"label": "A", "assessment": "selected", "reason": "wrong"}, {"label": "B", "assessment": "rejected", "reason": "right"}], "confidence": 1})
    manifest = build_choice_audit_manifest(store, task_id="t")
    result = compiler.apply("critic", {
        "op": "audit_choice",
        "candidate_version": manifest["candidate_version"], "manifest_digest": manifest["manifest_digest"],
        "selected_option_check": {"assessment": "violates", "reason": "wrong conflicts with the question"},
        "slot_checks": [{"claim_id": manifest["slot_claims"][0]["claim_id"], "assessment": "violates", "reason": "wrong conflicts with the question"}],
        "option_checks": [{"claim_id": item["claim_id"], "assessment": "violates", "reason": item["option_text"]} for item in manifest["elimination_claims"]],
        "summary": "The selected claim is contradicted.",
    })
    assert result.node_ids
    verification = store.latest_valid("t", "main", "verification")
    assert verification.status == "need_fix"
    assert verification.content["error_type"] == "contradicted_selected_option"


def test_choice_critic_preserves_structured_repair_packet():
    store = GraphStore()
    store.add_node(
        task_id="t", branch_id="main", logical_id="task", node_type="task_source", content="Choose the right term.", owner="dataset",
    )
    store.add_node(task_id="t", branch_id="main", logical_id="choice_schema", node_type="choice_schema", content={"allowed_labels": ["A", "I"], "slot_values": {"A": ["wrong"], "I": ["right"]}}, owner="dataset")
    for label, text in (("A", "wrong"), ("I", "right")):
        store.add_node(task_id="t", branch_id="main", logical_id=f"choice_source_{label}", node_type="choice_source", content={"label": label, "text": text}, owner="dataset")
    compiler = ActionCompiler(store, task_id="t", task_type="multiple_choice")
    compiler.apply("solver", {"op": "choose_option", "answer": "A", "slot_reasons": [{"slot": 1, "reason": "wrong"}], "option_reviews": [{"label": "A", "assessment": "selected", "reason": "wrong"}, {"label": "I", "assessment": "rejected", "reason": "right"}], "confidence": 1})
    manifest = build_choice_audit_manifest(store, task_id="t")
    compiler.apply("critic", {
        "op": "audit_choice",
        "candidate_version": manifest["candidate_version"], "manifest_digest": manifest["manifest_digest"],
        "selected_option_check": {"assessment": "violates", "reason": "wrong is not supported"},
        "slot_checks": [{"claim_id": manifest["slot_claims"][0]["claim_id"], "assessment": "violates", "reason": "wrong is wrong"}],
        "option_checks": [{"claim_id": item["claim_id"], "assessment": "fits" if item["option"] == "I" else "violates", "reason": item["option_text"]} for item in manifest["elimination_claims"]],
        "summary": "candidate is contradicted",
    })
    verification = store.latest_valid("t", "main", "verification")
    packet = json.loads(verification.content["reason"])["repair_packet"]
    assert packet["forbidden_answers"] == ["A"]
    assert packet["contradicted_slots"][0]["slot"] == 1
    assert packet["viable_challengers"] == ["I"]


def test_code_critic_uses_kernel_execution_status():
    store = GraphStore()
    store.add_node(
        task_id="t", branch_id="main", logical_id="code", node_type="code",
        content={"code": "def solve():\n    return 1\n"}, owner="solver",
    )
    store.add_node(
        task_id="t", branch_id="main", logical_id="execution", node_type="execution",
        content={"tests_passed": True}, owner="tool", status="verified",
        validation={"schema_valid": True, "execution_valid": True},
    )
    compiler = ActionCompiler(store, task_id="t", task_type="code_generation")
    compiler.apply("critic", {
        "op": "diagnose_code", "diagnosis": "tests pass", "repair_instruction": "",
    })
    assert store.latest_valid("t", "main", "verification").status == "verified"


def test_code_executor_prepends_imports_from_requirements() -> None:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="write first", owner="user")
    store.add_node(
        task_id="t", branch_id="main", logical_id="requirements", node_type="requirements",
        content={"text": "from typing import List\n\ndef first(xs: List[int]) -> int:\n"}, owner="dataset",
    )
    store.add_node(
        task_id="t", branch_id="main", logical_id="test_1", node_type="test",
        content={"setup": "", "text": "assert first([3, 4]) == 3"}, owner="dataset",
    )
    execution = execute_domain(
        "code_generation", store, {"code": "def first(xs: List[int]) -> int:\n    return xs[0]\n"},
    )
    assert execution.success
