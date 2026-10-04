"""Task-aware verification signals behind the task-agnostic critic protocol."""

from __future__ import annotations

import ast
import json
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class VerificationResult:
    status: str
    deterministic_checks: tuple[str, ...] = ()
    consistency_checks: tuple[str, ...] = ()
    confidence: float = 0.0
    error_signals: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    verifier_source: tuple[str, ...] = ()
    error_type: str = ""
    error_location: str = ""
    reason: str = ""
    repair_instruction: str = ""
    preserve: tuple[str, ...] = ()
    requested_fragments: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_revision(self) -> bool:
        return self.status in {"need_fix", "uncertain"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "deterministic_checks": list(self.deterministic_checks),
            "consistency_checks": list(self.consistency_checks),
            "confidence": self.confidence,
            "error_signals": list(self.error_signals),
            "missing_evidence": list(self.missing_evidence),
            "verifier_source": list(self.verifier_source),
            "error_type": self.error_type,
            "error_location": self.error_location,
            "reason": self.reason,
            "repair_instruction": self.repair_instruction,
            "preserve": list(self.preserve),
            "requested_fragments": list(self.requested_fragments),
            "metadata": dict(self.metadata),
        }


def verify_task_candidate(task_type: str, graph: Any, *, task_id: str, branch_id: str) -> VerificationResult:
    if task_type == "code_generation":
        return _verify_code(graph, task_id=task_id, branch_id=branch_id)
    if task_type in {"numeric_solve", "numeric_comparison"}:
        return _verify_numeric(graph, task_id=task_id, branch_id=branch_id)
    if task_type == "multiple_choice":
        return _verify_choice(graph, task_id=task_id, branch_id=branch_id)
    if task_type == "multihop_qa":
        return _verify_evidence(graph, task_id=task_id, branch_id=branch_id)
    return VerificationResult(status="verified", confidence=0.0, verifier_source=("TaskVerifier", "no_task_specific_signal"))


def apply_verification_signal(action: dict[str, Any], result: VerificationResult) -> dict[str, Any]:
    """Fuse verifier signal into a critic Action without weakening hard failures."""
    if action.get("op") != "verify":
        return action
    if not result.needs_revision:
        if result.confidence < 1.0:
            return action
        merged = dict(action)
        merged["status"] = "verified"
        return merged
    return {
        **dict(action),
        "status": result.status,
        "error_type": result.error_type or "verification_failure",
        "error_location": result.error_location or "candidate",
        "reason": result.reason or "; ".join((*result.error_signals, *result.missing_evidence)),
        "repair_instruction": result.repair_instruction or "Revise the candidate to address the verifier signal.",
        "preserve": list(result.preserve),
        "requested_fragments": list(result.requested_fragments),
    }


def _latest(graph: Any, task_id: str, branch_id: str, logical_id: str):
    return graph.latest_valid(task_id, branch_id, logical_id)


def _node_content(node: Any) -> Any:
    return getattr(node, "content", None)


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _verify_code(graph: Any, *, task_id: str, branch_id: str) -> VerificationResult:
    execution = _latest(graph, task_id, branch_id, "execution")
    result = _latest(graph, task_id, branch_id, "result")
    execution_valid = bool(
        (execution and execution.validation.get("execution_valid"))
        or (result and result.validation.get("execution_valid"))
    )
    if execution_valid:
        return VerificationResult(
            status="verified",
            deterministic_checks=("tests_passed",),
            confidence=1.0,
            verifier_source=("CodeVerifier", "unit_tests"),
        )
    errors: list[str] = []
    execution_detail: dict[str, Any] = {}
    for content in (_node_content(result), _node_content(execution)):
        if isinstance(content, dict):
            if content.get("kind") or content.get("failed_test"):
                execution_detail = content
            if isinstance(content.get("errors"), list):
                errors.extend(str(item) for item in content["errors"])
            if content.get("stderr"):
                errors.append(str(content["stderr"]))
            if content.get("failed_test"):
                errors.append(str(content["failed_test"]))
    reason = (errors[0] if errors else "Runtime tests did not pass.")[:1000]
    kind = str(execution_detail.get("kind") or "execution_failure")
    error_type = {
        "assertion_failure": "wrong_value",
        "syntax_error": "syntax_error",
        "import_error": "import_error",
        "name_error": "name_error",
        "type_error": "type_error",
        "timeout": "timeout",
        "runtime_exception": "runtime_exception",
    }.get(kind, "execution_failure")
    failed_test = str(execution_detail.get("failed_test") or "")
    expected = execution_detail.get("expected")
    actual = execution_detail.get("actual")
    function_name = str(execution_detail.get("function_name") or "the target function")
    if kind == "assertion_failure" and (expected is not None or actual is not None):
        repair_instruction = (
            f"For {failed_test}, {function_name} returned {actual}; "
            f"it must return {expected}. Fix the implementation for this concrete case."
        )
        location = failed_test or f"{function_name} return value"
    else:
        repair_instruction = (
            "Fix the latest code so all provided tests pass; preserve the function signature "
            "and working logic."
        )
        location = (
            failed_test
            or str(execution_detail.get("exception") or "")
            or _python_error_location(reason)
        )
    return VerificationResult(
        status="need_fix",
        deterministic_checks=("tests_failed",),
        confidence=1.0,
        error_signals=tuple(errors[:4]) or ("tests_failed",),
        verifier_source=("CodeVerifier", "unit_tests"),
        error_type=error_type,
        error_location=location,
        reason=reason,
        repair_instruction=repair_instruction,
        preserve=("function signature", "passing behavior"),
        requested_fragments=("code#function_body", "execution#execution_detail", "test#assertions"),
    )


def _python_error_location(stderr: str) -> str:
    match = re.search(r'File "<string>", line (\d+)', stderr)
    if match:
        return f"line {match.group(1)}"
    if "AssertionError" in stderr:
        return "failed assertion"
    if "NameError" in stderr:
        return "missing name/import"
    if "IndexError" in stderr:
        return "index access"
    return "code execution"


def _verify_numeric(graph: Any, *, task_id: str, branch_id: str) -> VerificationResult:
    calculation = _latest(graph, task_id, branch_id, "calculation")
    result = _latest(graph, task_id, branch_id, "result")
    if calculation is None or result is None:
        return VerificationResult(
            status="uncertain",
            consistency_checks=("missing_calculation_or_result",),
            confidence=0.2,
            verifier_source=("MathVerifier",),
            error_type="missing_numeric_artifact",
            error_location="solver output",
            reason="Numeric verification requires both calculation and result.",
            repair_instruction="Provide a calculation and a final result.",
            requested_fragments=("calculation#final_value", "result#final_value"),
        )
    content = _node_content(calculation)
    expression = content.get("expression") if isinstance(content, dict) else None
    claimed = content.get("value") if isinstance(content, dict) else None
    result_content = _json_value(_node_content(result))
    result_value = result_content.get("value") if isinstance(result_content, dict) else result_content
    expected = _safe_arithmetic(expression)
    if expected is not None and _number(result_value) is not None and abs(expected - _number(result_value)) <= 1e-9:
        checks = ["result_matches_expression"]
        if _number(claimed) is not None and abs(expected - _number(claimed)) > 1e-9:
            checks.append("calculation_trace_stale")
        return VerificationResult(
            status="verified",
            deterministic_checks=tuple(checks),
            confidence=0.9,
            verifier_source=("MathVerifier", "arithmetic"),
        )
    if expected is not None and _number(claimed) is not None and abs(expected - _number(claimed)) > 1e-9:
        return VerificationResult(
            status="need_fix",
            deterministic_checks=("arithmetic_mismatch",),
            confidence=1.0,
            error_signals=(f"{expression} evaluates to {expected}, not {claimed}",),
            verifier_source=("MathVerifier", "arithmetic"),
            error_type="arithmetic",
            error_location="calculation",
            reason=f"{expression} evaluates to {expected}, not {claimed}.",
            repair_instruction="Recompute the calculation and propagate the corrected value to the final result.",
            preserve=("task facts", "plan"),
            requested_fragments=("calculation#expression", "calculation#final_value", "result#final_value"),
        )
    return VerificationResult(
        status="verified",
        deterministic_checks=("arithmetic_consistent",),
        confidence=0.8,
        verifier_source=("MathVerifier", "arithmetic"),
    )


def _safe_arithmetic(expression: Any) -> float | None:
    if not isinstance(expression, str) or not expression.strip():
        return None
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](eval_node(node.left), eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](eval_node(node.operand))
        raise ValueError

    try:
        return eval_node(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def _verify_choice(graph: Any, *, task_id: str, branch_id: str) -> VerificationResult:
    result = _latest(graph, task_id, branch_id, "result")
    if result is None:
        return VerificationResult(status="uncertain", confidence=0.2, verifier_source=("ChoiceVerifier",))
    content = _json_value(_node_content(result))
    value = content.get("value") if isinstance(content, dict) else content
    label = str(value).strip().upper()[:1]
    schema = _latest(graph, task_id, branch_id, "choice_schema")
    schema_content = _json_value(_node_content(schema))
    labels: set[str] = set()
    if isinstance(schema_content, dict):
        labels = {
            str(item).strip().upper()
            for item in schema_content.get("allowed_labels", [])
            if str(item).strip()
        }
    choices = [
        node for node in graph.snapshot().nodes.values()
        if node.task_id == task_id and node.branch_id == branch_id and node.type == "choice" and node.is_operationally_valid()
    ]
    if not labels:
        labels = {
            str((_json_value(node.content).get("label") if isinstance(_json_value(node.content), dict) else "")).strip().upper()
            for node in choices
        }
        labels.discard("")
    if labels and label not in labels:
        return VerificationResult(
            status="need_fix",
            consistency_checks=("invalid_choice_label",),
            confidence=1.0,
            verifier_source=("ChoiceVerifier", "option_validity"),
            error_type="invalid_option",
            error_location="final choice",
            reason=f"Answer {value!r} is not one of the available options.",
            repair_instruction="Choose exactly one available option label.",
            preserve=("question", "choices"),
            requested_fragments=("result#final_value", "choice#label"),
        )
    return VerificationResult(
        status="verified",
        consistency_checks=("option_valid",),
        confidence=0.55,
        verifier_source=("ChoiceVerifier", "option_validity"),
    )


def _verify_evidence(graph: Any, *, task_id: str, branch_id: str) -> VerificationResult:
    facts = [
        node for node in graph.snapshot().nodes.values()
        if node.task_id == task_id and node.branch_id == branch_id and node.type in {"supporting_fact", "evidence", "evidence_link"} and node.is_operationally_valid()
    ]
    if not facts:
        return VerificationResult(
            status="uncertain",
            consistency_checks=("missing_evidence",),
            confidence=0.3,
            missing_evidence=("supporting_fact",),
            verifier_source=("EvidenceVerifier", "coverage"),
            error_type="evidence_insufficiency",
            error_location="supporting evidence",
            reason="No supporting evidence is visible for the answer.",
            repair_instruction="Request or use supporting evidence before finalizing the answer.",
            requested_fragments=("supporting_fact#content", "evidence_link#entity"),
        )
    return VerificationResult(
        status="verified",
        consistency_checks=("evidence_present",),
        confidence=0.5,
        verifier_source=("EvidenceVerifier", "coverage"),
    )
