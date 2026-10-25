import pytest

from workflow_runtime.protocol import (
    PROTOCOL_VERSION,
    action_contract,
    parse_action,
    render_role_context,
    validate_action,
)


def test_protocol_accepts_only_typed_planner_action() -> None:
    action = {
        "op": "plan",
        "facts": [{"content": "Alice has 2 apples", "source": "task", "quote": "2 apples"}],
        "strategy": "add the stated quantities",
    }
    assert validate_action("planner", action, task_type="numeric_solve") == []
    assert parse_action(__import__("json").dumps(action), role="planner", task_type="numeric_solve") == action
    assert PROTOCOL_VERSION == "semantic-delta/2.0"


def test_protocol_rejects_old_actions_and_wrapped_json() -> None:
    assert validate_action("planner", {"op": "add_fact", "id": "A", "value": 2}, task_type="numeric_solve")
    assert validate_action("critic", {"op": "verify", "target": "result", "status": "verified"}, task_type="numeric_solve")
    assert validate_action("final_solver", {"op": "answer", "source": "result", "value": 5}, task_type="numeric_solve")
    with pytest.raises(ValueError):
        parse_action('```json\n{"op":"done"}\n```', role="planner", task_type="numeric_solve")


def test_domain_solver_and_critic_actions_are_role_scoped() -> None:
    choice = {
        "op": "choose_option", "answer": "B",
        "slot_reasons": [{"slot": 1, "reason": "B fits"}],
        "eliminated_options": {"A": "A contradicts the question"}, "confidence": 0.75,
    }
    judgment = {
        "op": "audit_choice",
        "selected_option_check": {"status": "supported", "reason": "B is supported"},
        "slot_checks": [{"slot": 1, "status": "supported", "reason": "B fits"}],
        "eliminated_checks": [{"option": "A", "status": "contradicted", "reason": "A conflicts"}],
        "summary": "The candidate claims were checked.",
    }
    assert validate_action("solver", choice, task_type="multiple_choice") == []
    assert validate_action("critic", judgment, task_type="multiple_choice") == []
    assert validate_action("planner", choice, task_type="multiple_choice")
    assert validate_action("solver", judgment, task_type="multiple_choice")


def test_multihop_repair_patch_has_bounded_shape() -> None:
    valid = {
        "op": "repair_patch",
        "action": "PATCH_CITATIONS",
        "patch": {
            "selected_fact_ids": ["sentence_source_1", "sentence_source_2"],
            "preserve_answer": True,
            "reasoning_chain": ["first hop", "second hop"],
        },
    }
    assert validate_action("solver", valid, task_type="multihop_qa") == []
    one_citation = {**valid, "patch": {**valid["patch"], "selected_fact_ids": ["sentence_source_1"]}}
    assert validate_action("solver", one_citation, task_type="multihop_qa")


def test_action_contract_exposes_only_semantic_fields() -> None:
    contract = action_contract("critic", "code_generation")
    assert '"op":"diagnose_code"' in contract
    assert "verification nodes" not in contract
    assert "temporary IDs" in contract
    assert "exactly one JSON object" in contract


def test_role_contexts_use_canonical_runtime_grammar() -> None:
    planner_context = render_role_context("planner", {"task": "Find the total."})
    assert planner_context == "<GRAPH>\n<TASK>Find the total.</TASK>\n</GRAPH>"
    solver_context = render_role_context("solver", {
        "query_spec": {"task_type": "numeric_solve"},
        "facts": [],
        "plan": "add",
    })
    assert "<PLAN>add</PLAN>" in solver_context
