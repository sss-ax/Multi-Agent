"""Novelty, redundancy, token, and structural scoring for graph deltas."""

from __future__ import annotations

import json
import re
from typing import Any

from .canonical import canonical_json
from .models import DependencyEdge, GraphState, NodeVersion
from .relations import DATA_DEPENDENCY_RELATIONS, VALIDATION_RELATIONS


def estimate_node_tokens(node: NodeVersion) -> int:
    text = canonical_json({
        "logical_id": node.logical_id,
        "type": node.type,
        "content": node.content,
        "status": node.status,
    })
    return max(1, len(text.split()))


def estimate_edge_tokens(edge: DependencyEdge) -> int:
    return max(1, len(f"{edge.source} {edge.relation} {edge.target}".split()))


def normalize_scalar(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return re.sub(r"\s+", " ", text).strip().lower()


def semantic_signature(node: NodeVersion) -> tuple[Any, ...]:
    content = node.content
    if node.type == "fact" and isinstance(content, dict):
        return ("fact", content.get("id", node.logical_id), normalize_scalar(content.get("value")))
    if node.type == "plan_steps" and isinstance(content, dict):
        return (
            "plan_step",
            normalize_scalar(content.get("operation")),
            tuple(normalize_scalar(item) for item in content.get("inputs", [])),
            tuple(sorted(node.dependency_versions)),
        )
    if node.type == "result" and isinstance(content, dict):
        return ("result", normalize_scalar(content.get("id", node.logical_id)), normalize_scalar(content.get("value")))
    if node.type == "verification" and isinstance(content, dict):
        return ("verification", normalize_scalar(content.get("target")), normalize_scalar(content.get("status")))
    if node.type == "final_answer":
        return ("answer", normalize_scalar(content))
    return (node.type, node.logical_id, node.content_digest or normalize_scalar(content))


def semantically_known(node: NodeVersion, receiver_nodes: list[NodeVersion]) -> bool:
    signature = semantic_signature(node)
    return any(semantic_signature(item) == signature for item in receiver_nodes if item.is_operationally_valid())


def structural_features(node: NodeVersion, state: GraphState) -> dict[str, Any]:
    incoming = [
        edge
        for edge in state.edges
        if edge.target == node.node_id and edge.relation in DATA_DEPENDENCY_RELATIONS
    ]
    outgoing = [
        edge
        for edge in state.edges
        if edge.source == node.node_id and edge.relation in DATA_DEPENDENCY_RELATIONS
    ]
    validation_out = [
        edge
        for edge in state.edges
        if edge.source == node.node_id and edge.relation in VALIDATION_RELATIONS
    ]
    descendants = _descendants(node.node_id, state)
    return {
        "in_degree": len(incoming),
        "out_degree": len(outgoing),
        "validation_out_degree": len(validation_out),
        "descendant_count": len(descendants),
        "reachability_to_answer": _reaches_type(node.node_id, state, {"result", "final_answer"}),
        "dependency_missing_count": len(incoming),
    }


def _descendants(node_id: str, state: GraphState) -> set[str]:
    children_by_source: dict[str, list[str]] = {}
    for edge in state.edges:
        if edge.relation in DATA_DEPENDENCY_RELATIONS:
            children_by_source.setdefault(edge.source, []).append(edge.target)
    seen: set[str] = set()
    stack = list(children_by_source.get(node_id, []))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(children_by_source.get(current, []))
    return seen


def _reaches_type(node_id: str, state: GraphState, node_types: set[str]) -> bool:
    for descendant in _descendants(node_id, state):
        node = state.nodes.get(descendant)
        if node is not None and node.type in node_types:
            return True
    node = state.nodes.get(node_id)
    return node is not None and node.type in node_types

