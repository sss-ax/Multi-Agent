"""The sole model-facing Semantic Delta v2 Action protocol.

Models emit small task-specific semantic objects. Graph identifiers, edges,
provenance, verification nodes, routing, and finalization are kernel-owned.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from .semantic_actions import (
    MODEL_TYPED_OPS,
    TYPED_SEMANTIC_OPS,
    model_semantic_ops,
    semantic_action_example,
    validate_typed_semantic_action,
)
from .task_spec import registered_task_types


PROTOCOL_VERSION = "semantic-delta/2.0"
TASK_TYPES = frozenset(registered_task_types())

ROLE_SYSTEM_PROMPTS = {
    "planner": "Read the visible Source Graph and emit only plan/fact semantics plus source references. The kernel owns graph structure.",
    "solver": "Read the visible dependency-aware graph and emit only the domain solution semantics. The kernel owns result nodes and edges.",
    "critic": "Read the visible result and dependency graph and emit only the task-specific verdict or diagnosis. The kernel owns verification and repair routing.",
    "final_solver": "Finalization is deterministic and never calls a model.",
}

ROLE_ACTIONS = {
    "planner": frozenset({"plan", "done"}),
    "solver": frozenset({*TYPED_SEMANTIC_OPS - {"plan"}, "repair_patch", "done"}),
    "critic": frozenset({*MODEL_TYPED_OPS - TYPED_SEMANTIC_OPS, "done"}),
    "final_solver": frozenset(),
}

ACTION_FIELDS = {
    "plan": frozenset({"op", "facts", "strategy"}),
    "solve_numeric": frozenset({"op", "expression", "value"}),
    "choose_option": frozenset({"op", "answer", "slot_reasons", "option_reviews", "confidence"}),
    "submit_code": frozenset({"op", "code"}),
    "answer_with_evidence": frozenset({"op", "answer", "bridge_entity", "citations", "reasoning"}),
    "judge_numeric": frozenset({"op", "model_verdict", "error_type", "error_location", "reason", "repair_instruction"}),
    "audit_choice": frozenset({"op", "candidate_version", "manifest_digest", "selected_option_check", "slot_checks", "option_checks", "summary"}),
    "diagnose_code": frozenset({"op", "diagnosis", "repair_instruction"}),
    "judge_evidence": frozenset({"op", "answer_support", "bridge_status", "citation_status", "repair_target", "reason"}),
    "repair_patch": frozenset({"op", "action", "patch"}),
    "done": frozenset({"op"}),
}


def normalize_action_payload(payload: Any) -> Any:
    """Return the payload unchanged; v2 has no compatibility normalization."""
    return payload


def _validate_repair_patch(role: str, task_type: str, payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if role != "solver" or task_type != "multihop_qa":
        return ["repair_patch is only supported for multihop_qa solver repair"]
    action_name = payload.get("action")
    allowed = {"STOP", "ADD_EVIDENCE", "REPLACE_EVIDENCE", "PATCH_CITATIONS", "PATCH_BRIDGE", "REVISE_ANSWER"}
    if action_name not in allowed:
        errors.append("repair_patch.action is invalid")
    patch = payload.get("patch")
    if not isinstance(patch, dict):
        return [*errors, "repair_patch.patch must be an object"]
    if action_name == "STOP":
        if patch:
            errors.append("STOP repair_patch.patch must be empty")
        return errors
    if action_name in {"ADD_EVIDENCE", "REPLACE_EVIDENCE", "PATCH_CITATIONS"}:
        if set(patch) != {"selected_fact_ids", "preserve_answer", "reasoning_chain"}:
            errors.append("evidence repair patch keys must be exactly preserve_answer, reasoning_chain, selected_fact_ids")
        if patch.get("preserve_answer") is not True:
            errors.append("evidence repair patch must set preserve_answer=true")
        refs = patch.get("selected_fact_ids")
        if not isinstance(refs, list) or len(refs) < 2 or not all(isinstance(item, str) and item.strip() for item in refs):
            errors.append("evidence repair patch requires at least two selected_fact_ids")
    elif action_name == "PATCH_BRIDGE":
        if set(patch) != {"bridge_entity", "preserve_answer", "reasoning_chain"}:
            errors.append("PATCH_BRIDGE patch keys must be exactly bridge_entity, preserve_answer, reasoning_chain")
        if patch.get("preserve_answer") is not True:
            errors.append("PATCH_BRIDGE patch must set preserve_answer=true")
        if not isinstance(patch.get("bridge_entity"), str) or not patch.get("bridge_entity", "").strip():
            errors.append("PATCH_BRIDGE.bridge_entity must be non-empty")
    elif action_name == "REVISE_ANSWER":
        if set(patch) != {"answer", "reasoning_chain"}:
            errors.append("REVISE_ANSWER patch keys must be exactly answer, reasoning_chain")
        if not isinstance(patch.get("answer"), str) or not patch.get("answer", "").strip():
            errors.append("REVISE_ANSWER.answer must be non-empty")
    reasoning = patch.get("reasoning_chain")
    if not isinstance(reasoning, list) or not reasoning or not all(isinstance(item, str) and item.strip() for item in reasoning):
        errors.append("repair_patch.reasoning_chain must be a non-empty array of strings")
    return errors


def validate_action(
    role: str,
    payload: Any,
    *,
    task_type: str,
    protocol_mode: str = "semantic_delta_v2",
) -> list[str]:
    if protocol_mode != "semantic_delta_v2":
        return [f"unsupported action protocol: {protocol_mode}"]
    if role not in ROLE_ACTIONS:
        return [f"unknown action role: {role}"]
    if task_type not in TASK_TYPES:
        return [f"Semantic Delta has no registered TaskSpec for {task_type}"]
    if not isinstance(payload, dict):
        return ["action must be exactly one JSON object"]
    op = payload.get("op")
    if not isinstance(op, str) or not op:
        return ["action op must be a non-empty string"]
    errors: list[str] = []
    if op not in ROLE_ACTIONS[role]:
        errors.append(f"{role} cannot emit action {op} under {PROTOCOL_VERSION}")
    expected = ACTION_FIELDS.get(op)
    if expected is None:
        return [*errors, f"unknown action op: {op}"]
    if set(payload) != expected:
        errors.append(f"{op} keys must be exactly {', '.join(sorted(expected))}")
    if op in MODEL_TYPED_OPS:
        errors.extend(validate_typed_semantic_action(role, task_type, payload))
    elif op == "repair_patch":
        errors.extend(_validate_repair_patch(role, task_type, payload))
    return errors


def parse_action(
    text: str,
    *,
    role: str,
    task_type: str,
    protocol_mode: str = "semantic_delta_v2",
) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"{role} output must be exactly one Action JSON object") from exc
    errors = validate_action(role, value, task_type=task_type, protocol_mode=protocol_mode)
    if errors:
        raise ValueError(f"invalid {role} Action: {'; '.join(errors)}")
    return value


def action_contract(role: str, task_type: str, protocol_mode: str = "semantic_delta_v2") -> str:
    if protocol_mode != "semantic_delta_v2":
        raise ValueError(f"unsupported action protocol: {protocol_mode}")
    if role == "final_solver":
        return "Finalization is a deterministic kernel transition; no model Action is allowed."
    allowed = ", ".join(sorted({*model_semantic_ops(role, task_type), "done"}))
    # A positive Critic example is actively harmful for multiple-choice audit:
    # it anchors a small model to copying "supported" without reading the
    # question.  The workflow supplies the concrete manifest and audit rules
    # in its role-specific prompt instead.
    if role == "critic" and task_type == "multiple_choice":
        example = "<no fixed example; copy the immutable manifest claim IDs and judge each claim independently>"
    else:
        example = json.dumps(semantic_action_example(role, task_type), ensure_ascii=False, separators=(",", ":"))
    return (
        f"Action protocol {PROTOCOL_VERSION}. Task type={task_type}. "
        f"Emit exactly one JSON object per call. Allowed op values: {allowed}. "
        "The object must contain exactly the fields required by its op. "
        "Emit only semantic content and visible source references. Do not emit nodes, edges, IDs, "
        "temporary IDs, provenance, versions, status, visibility, completion flags, routing, send hints, "
        "markdown, explanations outside JSON, or a second object. The deterministic kernel validates and "
        "compiles the action, owns verification, and finalizes verified results. "
        f"Example shape for {role}: {example}"
    )


def render_role_context(role: str, values: Mapping[str, Any]) -> str:
    """Render a compact canonical view for callers outside ContextSlicer."""
    def content(name: str) -> str:
        value = values.get(name, [])
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if role == "planner":
        return f"<GRAPH>\n<TASK>{content('task')}</TASK>\n</GRAPH>"
    if role == "solver":
        lines = ["<GRAPH>", f"<QUERY_SPEC>{content('query_spec')}</QUERY_SPEC>"]
        for name, tag in (("choice", "CHOICE"), ("evidence", "EVIDENCE"), ("requirements", "REQUIREMENTS"), ("test", "TEST")):
            if name in values:
                lines.append(f"<{tag}>{content(name)}</{tag}>")
        lines.extend([f"<FACTS>{content('facts')}</FACTS>", f"<PLAN>{content('plan')}</PLAN>", "</GRAPH>"])
        return "\n".join(lines)
    if role == "critic":
        lines = ["<GRAPH>", f"<TASK>{content('task')}</TASK>", f"<QUERY_SPEC>{content('query_spec')}</QUERY_SPEC>"]
        for name, tag in (("choice", "CHOICE"), ("supporting_fact", "SUPPORTING_FACT"), ("evidence", "EVIDENCE"), ("calculation", "CALCULATION"), ("execution", "EXECUTION"), ("result", "RESULT")):
            if name in values:
                lines.append(f"<{tag}>{content(name)}</{tag}>")
        lines.append("</GRAPH>")
        return "\n".join(lines)
    raise ValueError(f"no model context renderer for role: {role}")
