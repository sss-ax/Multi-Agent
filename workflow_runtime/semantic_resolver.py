"""Deterministic semantic sufficiency resolver for receiver graph views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .agent_graph_view import AgentGraphView
from .models import GraphState, NodeVersion
from .semantic_contract import (
    SemanticCheckResult,
    SemanticContract,
    SemanticRequirement,
    SemanticStatus,
)


@dataclass(frozen=True)
class _Recovery:
    node_ids: tuple[str, ...]
    method: str
    value: Any = None


class SemanticResolver:
    """Resolve structured semantic requirements against visible graph state.

    The resolver is deliberately conservative. ``RECOVERABLE`` is returned only
    when a typed visible artifact already contains a deterministic answer/code
    value that can be copied without asking a model to infer it.
    """

    def __init__(self, state: GraphState, receiver_view: AgentGraphView):
        self.state = state
        self.receiver_view = receiver_view

    def check_contract(self, contract: SemanticContract) -> tuple[SemanticCheckResult, ...]:
        return tuple(self.resolve(requirement) for requirement in contract.requirements)

    def resolve(self, requirement: SemanticRequirement) -> SemanticCheckResult:
        satisfying = self._matching_visible_node_ids(requirement)
        if len(satisfying) >= requirement.min_count:
            return SemanticCheckResult(
                requirement=requirement,
                status=SemanticStatus.SATISFIED,
                satisfying_node_ids=tuple(satisfying),
                reason="visible graph contains required semantic object",
            )

        recovery = self._recover(requirement)
        if len(recovery.node_ids) >= requirement.min_count:
            return SemanticCheckResult(
                requirement=requirement,
                status=SemanticStatus.RECOVERABLE,
                recoverable_from_node_ids=recovery.node_ids,
                recovery_method=recovery.method,
                recovered_value=recovery.value,
                reason="visible deterministic artifact contains recoverable semantic value",
            )

        return SemanticCheckResult(
            requirement=requirement,
            status=SemanticStatus.MISSING,
            reason="visible graph does not satisfy or deterministically recover requirement",
        )

    def _visible_nodes(self) -> list[NodeVersion]:
        nodes = [
            node
            for node_id, node in self.state.nodes.items()
            if node_id in self.receiver_view.visible_node_ids and node.is_operationally_valid()
        ]
        return sorted(nodes, key=lambda node: (node.logical_id, node.version, node.node_id))

    def _matching_visible_node_ids(self, requirement: SemanticRequirement) -> list[str]:
        logical_ids = set(requirement.candidate_logical_ids())
        types = set(requirement.candidate_types())
        matches = []
        for node in self._visible_nodes():
            logical_match = bool(logical_ids) and node.logical_id in logical_ids
            type_match = bool(types) and node.type in types
            if logical_match or type_match:
                matches.append(node.node_id)
        return matches

    def _recover(self, requirement: SemanticRequirement) -> _Recovery:
        if requirement.kind in {"candidate_answer_or_code", "final_candidate"}:
            artifact_nodes = [
                node
                for node in self._visible_nodes()
                if node.type in {"calculation", "execution", "test_result"}
                and _has_recoverable_candidate_value(node.content)
            ]
            if artifact_nodes:
                return _Recovery(
                    node_ids=tuple(node.node_id for node in artifact_nodes),
                    method="deterministic_artifact",
                    value=_recoverable_candidate_value(artifact_nodes[0].content),
                )
            executed = _execute_visible_operation(self._visible_nodes())
            if executed is not None:
                node_ids, value = executed
                return _Recovery(node_ids=node_ids, method="deterministic_executor", value=value)
        if requirement.kind == "candidate_code":
            nodes = [
                node
                for node in self._visible_nodes()
                if node.type == "execution" and _has_recoverable_code_value(node.content)
            ]
            if nodes:
                return _Recovery(
                    node_ids=tuple(node.node_id for node in nodes),
                    method="deterministic_artifact",
                    value=_recoverable_code_value(nodes[0].content),
                )
        return _Recovery(node_ids=(), method="")


def check_semantic_contract(
    state: GraphState,
    receiver_view: AgentGraphView,
    contract: SemanticContract,
) -> tuple[SemanticCheckResult, ...]:
    return SemanticResolver(state, receiver_view).check_contract(contract)


def all_requirements_sufficient(results: Iterable[SemanticCheckResult]) -> bool:
    return all(result.is_sufficient for result in results)


def missing_requirements(results: Iterable[SemanticCheckResult]) -> tuple[SemanticRequirement, ...]:
    return tuple(result.requirement for result in results if result.status == SemanticStatus.MISSING)


def _has_recoverable_candidate_value(content: Any) -> bool:
    if isinstance(content, dict):
        if "value" in content and content["value"] not in (None, ""):
            return True
        if "answer" in content and content["answer"] not in (None, ""):
            return True
        if "code" in content and str(content["code"]).strip():
            return True
        execution = content.get("execution")
        if isinstance(execution, dict):
            return _has_recoverable_candidate_value(execution)
    return False


def _recoverable_candidate_value(content: Any) -> Any:
    if isinstance(content, dict):
        if "value" in content and content["value"] not in (None, ""):
            return content["value"]
        if "answer" in content and content["answer"] not in (None, ""):
            return content["answer"]
        if "code" in content and str(content["code"]).strip():
            return content["code"]
        execution = content.get("execution")
        if isinstance(execution, dict):
            return _recoverable_candidate_value(execution)
    return None


def _has_recoverable_code_value(content: Any) -> bool:
    if isinstance(content, dict):
        if "code" in content and str(content["code"]).strip():
            return True
        value = content.get("value")
        if isinstance(value, dict) and "code" in value and str(value["code"]).strip():
            return True
    return False


def _recoverable_code_value(content: Any) -> Any:
    if isinstance(content, dict):
        if "code" in content and str(content["code"]).strip():
            return content["code"]
        value = content.get("value")
        if isinstance(value, dict) and "code" in value and str(value["code"]).strip():
            return value["code"]
    return None


def _execute_visible_operation(nodes: list[NodeVersion]) -> tuple[tuple[str, ...], Any] | None:
    values = _visible_fact_values(nodes)
    for node in nodes:
        if node.type not in {"plan_steps", "plan"}:
            continue
        for step in _operation_steps(node.content):
            operation = str(step.get("operation", "")).strip().lower()
            inputs = step.get("inputs", ())
            if operation != "add" or not isinstance(inputs, (list, tuple)) or len(inputs) < 2:
                continue
            operands = []
            support_ids = [node.node_id]
            for item in inputs:
                key = str(item)
                if key not in values:
                    operands = []
                    break
                value, source_id = values[key]
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    operands = []
                    break
                operands.append(value)
                support_ids.append(source_id)
            if operands:
                return tuple(dict.fromkeys(support_ids)), sum(operands)
    return None


def _visible_fact_values(nodes: list[NodeVersion]) -> dict[str, tuple[Any, str]]:
    values: dict[str, tuple[Any, str]] = {}
    for node in nodes:
        if node.type == "fact" and isinstance(node.content, dict):
            fact_id = node.content.get("id")
            if fact_id is not None and "value" in node.content:
                values[str(fact_id)] = (node.content["value"], node.node_id)
        elif node.type == "facts" and isinstance(node.content, list):
            for item in node.content:
                if isinstance(item, dict) and item.get("id") is not None and "value" in item:
                    values[str(item["id"])] = (item["value"], node.node_id)
    return values


def _operation_steps(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, dict):
        if "steps" in content and isinstance(content["steps"], list):
            return [item for item in content["steps"] if isinstance(item, dict)]
        if "operation" in content:
            return [content]
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]
    return []
