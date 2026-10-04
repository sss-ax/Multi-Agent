"""Strict incremental Action IR used by the workflow runtime.

The model never writes GraphStore nodes or edges directly. It emits one small,
role-scoped action at a time; the runtime validates and compiles that action
into the versioned graph.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping

PROTOCOL_VERSION = "workflow-action/1.0"
TASK_TYPES = frozenset({
    "numeric_solve", "numeric_comparison", "table_qa", "multihop_qa", "multiple_choice", "code_generation",
    "marble_research", "marble_bargaining", "marble_database",
})

ROLE_SYSTEM_PROMPTS = {
    "planner": "Read the visible task graph and emit one planner Action. Do not write GraphStore nodes directly and do not solve the task.",
    "solver": "Read the visible dependency-aware graph and emit one solver Action that executes the plan or produces a domain artifact.",
    "critic": "Read the visible result and dependency graph and emit one verification or error Action.",
    "final_solver": "Read the verified result boundary and emit one answer Action that copies the verified value.",
}

ROLE_ACTIONS = {
    "planner": frozenset({
        "declare_query", "add_fact", "add_evidence", "add_entity", "add_supporting_fact",
        "add_evidence_link", "add_table", "add_table_cell", "add_requirement", "add_test",
        "add_plan_step", "done",
    }),
    "solver": frozenset({
        "calculate", "set_result", "emit_code", "set_execution", "set_test_result",
        "call_tool", "report_error", "done",
    }),
    "critic": frozenset({"verify", "report_error", "done"}),
    "final_solver": frozenset({"answer", "done"}),
}

ACTION_FIELDS = {
    "declare_query": frozenset({"op", "content"}),
    "add_fact": frozenset({"op", "id", "value"}),
    "add_evidence": frozenset({"op", "id", "content"}),
    "add_entity": frozenset({"op", "id", "content"}),
    "add_supporting_fact": frozenset({"op", "id", "content"}),
    "add_evidence_link": frozenset({"op", "id", "content"}),
    "add_table": frozenset({"op", "content"}),
    "add_table_cell": frozenset({"op", "id", "content"}),
    "add_requirement": frozenset({"op", "content"}),
    "add_test": frozenset({"op", "id", "content"}),
    "add_plan_step": frozenset({"op", "id", "operation", "inputs"}),
    "calculate": frozenset({"op", "id", "expression", "value"}),
    "set_result": frozenset({"op", "id", "value"}),
    "emit_code": frozenset({"op", "code"}),
    "set_execution": frozenset({"op", "content", "success"}),
    "set_test_result": frozenset({"op", "id", "content"}),
    "call_tool": frozenset({"op", "call_id", "name", "arguments"}),
    "verify": frozenset({
        "op", "target", "status", "error_type", "error_location",
        "reason", "repair_instruction", "preserve", "requested_fragments",
    }),
    "report_error": frozenset({"op", "content"}),
    "answer": frozenset({"op", "source", "value"}),
    "done": frozenset({"op"}),
}


def normalize_action_payload(payload: Any) -> Any:
    """Normalize harmless model variations before strict validation.

    Code models often include an ``id`` or ``language`` field with emitted
    code.  The runtime stores code under a fixed logical boundary, so those
    fields do not carry semantics and can be discarded safely.
    """
    if not isinstance(payload, dict):
        return payload
    op = payload.get("op")
    if op == "emit_code":
        if "code" in payload:
            return {"op": "emit_code", "code": payload["code"]}
        if "content" in payload:
            return {"op": "emit_code", "code": payload["content"]}
    if op == "verify":
        normalized = dict(payload)
        normalized.setdefault("error_type", "")
        normalized.setdefault("error_location", "")
        normalized.setdefault("reason", "")
        normalized.setdefault("repair_instruction", "")
        normalized.setdefault("preserve", [])
        normalized.setdefault("requested_fragments", [])
        return normalized
    return payload


def validate_action(role: str, payload: Any, *, task_type: str) -> List[str]:
    payload = normalize_action_payload(payload)
    errors: List[str] = []
    if role not in ROLE_ACTIONS:
        return [f"unknown action role: {role}"]
    if task_type not in TASK_TYPES:
        errors.append(f"unsupported task_type: {task_type}")
    if not isinstance(payload, dict):
        return ["action must be exactly one JSON object"]
    op = payload.get("op")
    if not isinstance(op, str) or not op:
        return ["action op must be a non-empty string"]
    if op not in ROLE_ACTIONS[role]:
        errors.append(f"{role} cannot emit action {op}")
    expected = ACTION_FIELDS.get(op)
    if expected is None:
        errors.append(f"unknown action op: {op}")
        return errors
    if set(payload) != expected:
        errors.append(f"{op} keys must be exactly {', '.join(sorted(expected))}")
    for key, value in payload.items():
        if key != "op" and value is None:
            errors.append(f"{op}.{key} must not be null")
    for key in ("id", "call_id"):
        if key in payload and (not isinstance(payload[key], str) or not payload[key].strip()):
            errors.append(f"{op}.{key} must be a non-empty string")
    for key in ("target", "source"):
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, str):
            if not value.strip():
                errors.append(f"{op}.{key} must be a non-empty string")
        elif isinstance(value, dict):
            allowed_ref_keys = {"logical_id", "type", "expected_type", "version"}
            if not set(value).issubset(allowed_ref_keys):
                errors.append(f"{op}.{key} NodeRef keys must be drawn from logical_id, type, expected_type, version")
            if "logical_id" in value and (not isinstance(value["logical_id"], str) or not value["logical_id"].strip()):
                errors.append(f"{op}.{key}.logical_id must be a non-empty string when present")
            expected = value.get("expected_type", value.get("type", ""))
            if expected and not isinstance(expected, str):
                errors.append(f"{op}.{key}.expected_type must be a string")
            if "version" in value and not isinstance(value["version"], int):
                errors.append(f"{op}.{key}.version must be an integer when present")
            if "logical_id" not in value and not expected:
                errors.append(f"{op}.{key} NodeRef must include logical_id or expected_type/type")
        else:
            errors.append(f"{op}.{key} must be a string or NodeRef object")
    if op == "add_plan_step":
        if not isinstance(payload.get("operation"), str) or not payload["operation"].strip():
            errors.append("add_plan_step.operation must be a non-empty string")
        if not isinstance(payload.get("inputs"), list) or not all(isinstance(item, str) and item for item in payload["inputs"]):
            errors.append("add_plan_step.inputs must be an array of non-empty strings")
    if op == "set_execution" and not isinstance(payload.get("success"), bool):
        errors.append("set_execution.success must be boolean")
    if op == "call_tool":
        if not isinstance(payload.get("name"), str) or not payload["name"].strip():
            errors.append("call_tool.name must be a non-empty string")
        if not isinstance(payload.get("arguments"), dict):
            errors.append("call_tool.arguments must be an object")
    if op == "verify" and payload.get("status") not in {"verified", "need_fix", "uncertain"}:
        errors.append("verify.status must be verified, need_fix, or uncertain")
    if op == "verify":
        for key in ("error_type", "error_location", "reason", "repair_instruction"):
            if not isinstance(payload.get(key), str):
                errors.append(f"verify.{key} must be a string")
        for key in ("preserve", "requested_fragments"):
            if not isinstance(payload.get(key), list) or not all(isinstance(item, str) for item in payload.get(key, [])):
                errors.append(f"verify.{key} must be an array of strings")
        if payload.get("status") == "need_fix":
            for key in ("error_type", "error_location", "repair_instruction"):
                if not str(payload.get(key, "")).strip():
                    errors.append(f"need_fix verify.{key} must be non-empty")
    return errors


def parse_action(text: str, *, role: str, task_type: str) -> Dict[str, Any]:
    try:
        value = normalize_action_payload(json.loads(text))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"{role} output must be exactly one Action JSON object") from exc
    errors = validate_action(role, value, task_type=task_type)
    if errors:
        raise ValueError(f"invalid {role} Action: {'; '.join(errors)}")
    return value


def action_contract(role: str, task_type: str) -> str:
    allowed = ", ".join(sorted(ROLE_ACTIONS[role]))
    examples = {
        "planner": '{"op":"add_fact","id":"A","value":2}',
        "solver": '{"op":"calculate","id":"R1","expression":"2+3","value":5}',
        "critic": '{"op":"verify","target":"result","status":"verified","error_type":"","error_location":"","reason":"","repair_instruction":"","preserve":[],"requested_fragments":[]}',
        "final_solver": '{"op":"answer","source":"result","value":5}',
    }
    solver_rule = (
        "Solver must use calculate and set_result for numeric work; it may call an allowlisted tool with call_tool; it must not emit planner Actions."
    )
    if task_type == "multiple_choice":
        solver_rule = "Solver must choose one option and emit set_result with a single letter value such as A; it must not emit planner Actions."
    elif task_type == "multihop_qa":
        solver_rule = "Solver must answer from the provided evidence and emit set_result with a concise answer; it must not emit planner Actions."
    elif task_type == "code_generation":
        solver_rule = "Solver must emit executable Python code with emit_code; it must not emit planner Actions."
    elif task_type in {"marble_research", "marble_bargaining", "marble_database"}:
        solver_rule = "Solver must emit the domain artifact with set_result; it must not emit planner Actions."
    stage_rules = {
        "planner": (
            "Planner may use add_fact only for facts explicitly present in the task. "
            "For numeric tasks, do not put a derived answer into add_fact; emit add_plan_step "
            "for the operation and operands before finishing."
        ),
        "solver": solver_rule,
        "critic": (
            "Critic must verify an existing result or report an error; it must not create a result. "
            "For status=need_fix, include actionable diagnostic fields: error_type, "
            "error_location, repair_instruction, preserve, and requested_fragments."
        ),
        "final_solver": "Finalizer must copy an existing verified result using answer; it must not recalculate.",
    }[role]
    return (
        f"Action protocol {PROTOCOL_VERSION}. Task type={task_type}. "
        f"Emit exactly one JSON object per call. Allowed op values: {allowed}. "
        "The object must contain exactly the fields required by its op. "
        "Do not emit GraphStore node IDs, versions, edges, task nodes, markdown, "
        "explanations, or a second object. Emit {\"op\":\"done\"} when the role has no more Actions. "
        "The runtime also ends the role when its required graph boundary is complete. "
        f"{stage_rules} "
        f"Example shape for {role}: {examples[role]}"
    )


def render_role_context(role: str, values: Mapping[str, Any]) -> str:
    def content(name: str) -> str:
        value = values.get(name, [])
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if role == "planner":
        return f"<GRAPH>\n<TASK>{content('task')}</TASK>\n</GRAPH>"
    if role == "solver":
        lines = ["<GRAPH>", f"<QUERY_SPEC>{content('query_spec')}</QUERY_SPEC>"]
        domain_tags = (("table", "TABLE"), ("table_cell", "TABLE_CELL"), ("evidence", "EVIDENCE"), ("choice", "CHOICE"), ("entity", "ENTITY"), ("supporting_fact", "SUPPORTING_FACT"), ("evidence_link", "EVIDENCE_LINK"), ("requirements", "REQUIREMENTS"), ("code", "CODE"), ("test", "TEST"), ("execution", "EXECUTION"), ("error", "ERROR"))
        if any(name in values for name, _ in domain_tags):
            lines.extend(f"<{tag}>{content(name)}</{tag}>" for name, tag in domain_tags if name in values)
        else:
            lines.append(f"<FACTS>{content('facts')}</FACTS>")
        lines.extend([f"<PLAN>{content('plan')}</PLAN>", f"<PLAN_STEPS>{content('plan_steps')}</PLAN_STEPS>", "</GRAPH>"])
        return "\n".join(lines)
    if role == "critic":
        lines = ["<GRAPH>", f"<TASK>{content('task')}</TASK>", f"<QUERY_SPEC>{content('query_spec')}</QUERY_SPEC>"]
        if "choice" in values:
            lines.append(f"<CHOICE>{content('choice')}</CHOICE>")
        if "entity" in values:
            lines.append(f"<ENTITY>{content('entity')}</ENTITY>")
        if "supporting_fact" in values:
            lines.append(f"<SUPPORTING_FACT>{content('supporting_fact')}</SUPPORTING_FACT>")
        if "evidence_link" in values:
            lines.append(f"<EVIDENCE_LINK>{content('evidence_link')}</EVIDENCE_LINK>")
        if "evidence" in values:
            lines.append(f"<EVIDENCE>{content('evidence')}</EVIDENCE>")
        lines.extend([f"<FACTS>{content('facts')}</FACTS>", f"<CALCULATION>{content('calculation')}</CALCULATION>", f"<RESULT>{content('result')}</RESULT>", "</GRAPH>"])
        return "\n".join(lines)
    if role == "final_solver":
        result = values["result"]
        if isinstance(result, str):
            result = json.loads(result)
        result = dict(result)
        result["status"] = "verified"
        return "\n".join(["<GRAPH>", f"<QUERY_SPEC>{content('query_spec')}</QUERY_SPEC>", f"<RESULT status=\"verified\">{json.dumps(result, ensure_ascii=False, separators=(',', ':'))}</RESULT>", '<VERIFY status="verified">verified</VERIFY>', "</GRAPH>"])
    raise ValueError(f"no canonical context renderer for role: {role}")


def require_exact_copy(answer: Any, result_value: Any) -> None:
    if str(answer).strip() != str(result_value):
        raise ValueError("final_solver answer must exactly copy RESULT.value")
