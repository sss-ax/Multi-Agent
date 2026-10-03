from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from workflow_runtime.protocol import (
    PROTOCOL_VERSION,
    action_contract,
    parse_action,
    render_role_context,
    require_exact_copy,
    validate_action,
)


def test_action_protocol_accepts_one_planner_action() -> None:
    action = {"op": "add_fact", "id": "A", "value": 2}
    assert validate_action("planner", action, task_type="numeric_solve") == []
    assert parse_action('{"op":"add_fact","id":"A","value":2}', role="planner", task_type="numeric_solve") == action
    assert PROTOCOL_VERSION == "workflow-action/1.0"


def test_action_protocol_rejects_non_action_objects_and_wrong_roles() -> None:
    assert validate_action("planner", {"protocol_version": PROTOCOL_VERSION}, task_type="numeric_solve")
    assert validate_action("planner", {"op": "verify", "target": "result", "status": "verified"}, task_type="numeric_solve")
    with pytest.raises(ValueError):
        parse_action("```json\n{\"op\":\"done\"}\n```", role="planner", task_type="numeric_solve")


def test_action_fields_are_exact_and_task_is_not_an_operation() -> None:
    assert validate_action("planner", {"op": "add_fact", "id": "A", "value": 2, "version": 1}, task_type="numeric_solve")
    assert validate_action("planner", {"op": "add_node", "node_type": "task"}, task_type="numeric_solve")
    assert validate_action("solver", {"op": "set_execution", "content": {}, "success": "true"}, task_type="code_generation")


def test_action_contract_is_role_scoped() -> None:
    contract = action_contract("planner", "numeric_solve")
    assert "add_fact" in contract
    assert "GraphStore node IDs" in contract
    assert "exactly one JSON object" in contract


def test_final_answer_must_be_exactly_executor_value() -> None:
    require_exact_copy("18", 18)
    with pytest.raises(ValueError):
        require_exact_copy("44", 18)


def test_role_contexts_use_canonical_runtime_grammar() -> None:
    planner_context = render_role_context("planner", {"task": "Find the total."})
    assert planner_context == "<GRAPH>\n<TASK>Find the total.</TASK>\n</GRAPH>"
    solver_context = render_role_context("solver", {
        "query_spec": {"task_type": "numeric_solve"},
        "facts": [],
        "plan": "add",
        "plan_steps": [],
    })
    assert "<PLAN_STEPS>[]</PLAN_STEPS>" in solver_context
