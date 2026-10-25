"""Pure validation for Semantic Delta proposals."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .graph_store import GraphStore
from .models import node_visible_to_role
from .semantic_delta_ir import GroundingRef, ProposedEdge, ProposedNode, SemanticDelta
from .task_spec import TaskSpec, get_task_spec


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "message": self.message}


class SemanticDeltaValidationError(ValueError):
    def __init__(self, issues: Iterable[ValidationIssue]):
        self.issues = tuple(issues)
        super().__init__("; ".join(f"{issue.path}: {issue.message}" for issue in self.issues))


@dataclass(frozen=True)
class ValidatedSemanticDelta:
    delta: SemanticDelta
    existing_nodes: Mapping[str, Any] = field(default_factory=dict)
    grounding_nodes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(content)


def _source_ref(ref: Any) -> str:
    if isinstance(ref, Mapping):
        return str(ref.get("logical_id", ref.get("node_id", ref.get("source", "")))).strip()
    return str(ref).strip()


def _context_node_id(ref: Any, context: Mapping[str, Any] | None) -> str:
    value = _source_ref(ref)
    if not value or not isinstance(context, Mapping):
        return value
    for index_name in ("local_node_index", "fragment_index", "semantic_type_index"):
        index = context.get(index_name, {})
        if not isinstance(index, Mapping) or value not in index:
            continue
        item = index[value]
        if isinstance(item, Mapping):
            candidate = item.get("source_node_id", item.get("node_id", item.get("source_logical_id", "")))
            if candidate:
                return str(candidate)
        if isinstance(item, str):
            return item
        if isinstance(item, list) and len(item) == 1:
            return _context_node_id(item[0], context)
    if "@" in value:
        # Models occasionally combine a semantic label and the displayed local
        # reference (for example ``question_source@n1``).
        prefix, suffix = value.rsplit("@", 1)
        for candidate in (suffix, prefix):
            resolved = _context_node_id(candidate, context)
            if resolved != candidate or candidate in context.get("local_node_index", {}):
                return resolved
    return value


def _resolve_node(
    store: GraphStore,
    task_id: str,
    branch_id: str,
    ref: Any,
    context: Mapping[str, Any] | None = None,
) -> Any:
    value = _context_node_id(ref, context)
    if not value:
        return None
    state = store.snapshot()
    if value in state.nodes:
        node = state.nodes[value]
        return node if node.task_id == task_id and node.branch_id == branch_id else None
    logical = store.latest_valid(task_id, branch_id, value)
    if logical is not None:
        return logical
    # Small models sometimes echo an exact source passage in the reference
    # slot.  Resolve it only when it identifies one unique visible source; this
    # is deterministic reference canonicalization, not semantic matching.
    if len(value) >= 8:
        matches = [
            node
            for node in state.nodes.values()
            if node.task_id == task_id
            and node.branch_id == branch_id
            and node.is_operationally_valid()
            and value in _content_text(node.content)
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def _canonical_endpoint(
    ref: str,
    *,
    endpoint_ids: set[str],
    context: Mapping[str, Any] | None,
) -> str:
    """Canonicalize common model-rendered endpoint spellings."""
    raw = str(ref).strip()
    candidates = [raw]
    resolved = _context_node_id(raw, context)
    if resolved not in candidates:
        candidates.append(resolved)
    if "@" in raw:
        prefix, suffix = raw.rsplit("@", 1)
        candidates.extend((suffix, prefix))
        for item in (suffix, prefix):
            candidate = _context_node_id(item, context)
            if candidate not in candidates:
                candidates.append(candidate)
    return next((item for item in candidates if item in endpoint_ids), resolved or raw)


def _cycle_exists(edges: Iterable[ProposedEdge], known_ids: set[str]) -> bool:
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source, []).append(edge.target)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visiting:
            return True
        if node_id in visited:
            return False
        visiting.add(node_id)
        for target in adjacency.get(node_id, ()):
            if target in known_ids or visit(target):
                if target in visiting or target not in known_ids:
                    return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    return any(visit(node_id) for node_id in adjacency)


def validate_semantic_delta(
    delta: SemanticDelta | Mapping[str, Any],
    *,
    role: str,
    task_id: str,
    branch_id: str,
    store: GraphStore,
    task_spec: TaskSpec | None = None,
    visible_node_ids: Iterable[str] = (),
    known_fragment_ids: Iterable[str] = (),
    allowed_receivers: Iterable[str] = (),
    node_ref_context: Mapping[str, Any] | None = None,
) -> ValidatedSemanticDelta:
    """Validate a delta without mutating ``store``."""
    if not isinstance(delta, SemanticDelta):
        delta = SemanticDelta.from_payload(delta)
    spec = task_spec or get_task_spec("")
    issues: list[ValidationIssue] = []
    visible = set(str(item) for item in visible_node_ids)
    known_fragments = set(str(item) for item in known_fragment_ids)
    allowed_receiver_set = set(str(item) for item in allowed_receivers)
    if not allowed_receiver_set:
        allowed_receiver_set = {
            "planner": {"solver"},
            "solver": {"critic"},
            "critic": {"solver", "final_solver"},
            "tool": {"critic"},
        }.get(role, set())

    by_temp: dict[str, ProposedNode] = {}
    for index, node in enumerate(delta.nodes):
        temp_id = node.temp_id.strip()
        path = f"nodes[{index}]"
        if not temp_id:
            issues.append(ValidationIssue(path, "temp_id must be non-empty"))
        elif temp_id in by_temp:
            issues.append(ValidationIssue(path, f"duplicate temp_id: {temp_id}"))
        else:
            by_temp[temp_id] = node
        if not node.node_type.strip():
            issues.append(ValidationIssue(f"{path}.type", "node type must be non-empty"))
        allowed_types = spec.allowed_node_types(role)
        if allowed_types and node.node_type not in allowed_types:
            issues.append(ValidationIssue(f"{path}.type", f"{role} cannot propose node type {node.node_type}"))
        if node.content is None:
            issues.append(ValidationIssue(f"{path}.content", "content must not be null"))
        if node.node_type in {"candidate_answer", "code"}:
            result_errors = spec.validate_result(node.content)
            issues.extend(ValidationIssue(f"{path}.content", error) for error in result_errors)
        elif node.node_type == "slot_analysis" and not isinstance(node.content, Mapping):
            issues.append(ValidationIssue(f"{path}.content", "slot_analysis content must be an object"))
        if node.node_type in {"fact", "selected_evidence", "sentence_claim"} and not node.grounding:
            issues.append(ValidationIssue(f"{path}.grounding", "this semantic node requires at least one source grounding"))

    if delta.complete:
        proposed_types = {node.node_type for node in delta.nodes}
        for required_type in spec.required_for(role):
            if required_type not in proposed_types:
                issues.append(ValidationIssue(
                    "complete",
                    f"complete Semantic Delta is missing required {role} output: {required_type}",
                ))
    for singleton_type in ("calculation", "candidate_answer", "code", "slot_analysis", "repair_patch"):
        count = sum(node.node_type == singleton_type for node in delta.nodes)
        if count > 1:
            issues.append(ValidationIssue(
                "nodes",
                f"Semantic Delta may contain at most one {singleton_type} node",
            ))

    state = store.snapshot()
    known_ids = set(state.nodes)
    existing_nodes: dict[str, Any] = {}
    grounding_nodes: dict[str, tuple[str, ...]] = {}
    for index, node in enumerate(delta.nodes):
        grounding_ids: list[str] = []
        for grounding_index, grounding in enumerate(node.grounding):
            source = _resolve_node(store, task_id, branch_id, grounding.source, node_ref_context)
            if source is None:
                issues.append(ValidationIssue(
                    f"nodes[{index}].grounding[{grounding_index}]",
                    f"source node does not exist: {_source_ref(grounding.source)}",
                ))
                continue
            if visible and source.node_id not in visible:
                issues.append(ValidationIssue(
                    f"nodes[{index}].grounding[{grounding_index}]",
                    f"source node is not visible to {role}: {source.node_id}",
                ))
            elif not node_visible_to_role(source, role):
                issues.append(ValidationIssue(
                    f"nodes[{index}].grounding[{grounding_index}]",
                    f"source node visibility denies role {role}: {source.node_id}",
                ))
            if source.type not in spec.public_source_types:
                issues.append(ValidationIssue(
                    f"nodes[{index}].grounding[{grounding_index}]",
                    f"grounding must reference a TaskSpec source type, got {source.type}",
                ))
            source_text = _content_text(source.content)
            if grounding.quote and grounding.quote not in source_text:
                issues.append(ValidationIssue(
                    f"nodes[{index}].grounding[{grounding_index}].quote",
                    "quote does not occur in source content",
                ))
            if node.node_type == "fact" and isinstance(node.content, Mapping) and grounding.quote:
                numeric_value = node.content.get("quantity", node.content.get("value"))
                if isinstance(numeric_value, (int, float)) and str(numeric_value) not in grounding.quote.replace(",", ""):
                    issues.append(ValidationIssue(
                        f"nodes[{index}].content",
                        "numeric fact value is not represented in its grounding quote",
                    ))
            grounding_ids.append(source.node_id)
            existing_nodes[source.node_id] = source
        grounding_nodes[node.temp_id] = tuple(grounding_ids)

    proposed_edges: list[ProposedEdge] = []
    known_logical_ids = {
        str(node.logical_id)
        for node in state.nodes.values()
        if node.task_id == task_id and node.branch_id == branch_id
    }
    endpoint_ids = set(by_temp) | known_ids | known_logical_ids
    for index, edge in enumerate(delta.edges):
        path = f"edges[{index}]"
        if not edge.source or not edge.target:
            issues.append(ValidationIssue(path, "source and target must be non-empty"))
        source_ref = _canonical_endpoint(
            edge.source, endpoint_ids=endpoint_ids, context=node_ref_context
        )
        target_ref = _canonical_endpoint(
            edge.target, endpoint_ids=endpoint_ids, context=node_ref_context
        )
        if source_ref not in endpoint_ids:
            issues.append(ValidationIssue(f"{path}.source", f"unknown endpoint: {edge.source}"))
        if target_ref not in endpoint_ids:
            issues.append(ValidationIssue(f"{path}.target", f"unknown endpoint: {edge.target}"))
        relation = {
            "input": "input_to",
            "inputs": "input_to",
            "uses_input": "input_to",
            "output": "produces",
        }.get(edge.relation, edge.relation)
        if relation not in spec.allowed_relations:
            issues.append(ValidationIssue(f"{path}.relation", f"relation is not allowed: {edge.relation}"))
        proposed_edges.append(ProposedEdge(
            source=source_ref,
            relation=relation,
            target=target_ref,
            metadata=edge.metadata,
        ))
    data_edges = [edge for edge in proposed_edges if edge.relation in {"depends_on", "input_to", "derived_from", "produces", "grounded_in"}]
    if _cycle_exists(data_edges, known_ids):
        issues.append(ValidationIssue("edges", "data dependency cycle detected"))

    for index, hint in enumerate(delta.send_hints):
        path = f"send_hints[{index}]"
        if hint.receiver not in allowed_receiver_set:
            issues.append(ValidationIssue(f"{path}.receiver", f"receiver is not allowed: {hint.receiver}"))
        for fragment_index, fragment in enumerate(hint.fragments):
            node_key = fragment.split("#", 1)[0]
            if node_key in by_temp or fragment in known_fragments:
                continue
            referenced = _resolve_node(store, task_id, branch_id, node_key, node_ref_context)
            if referenced is None:
                issues.append(ValidationIssue(
                    f"{path}.fragments[{fragment_index}]",
                    f"fragment is not part of this delta or sender view: {fragment}",
                ))
                continue
            if referenced.node_id not in visible:
                issues.append(ValidationIssue(
                    f"{path}.fragments[{fragment_index}]",
                    f"existing fragment is not visible to sender {role}: {fragment}",
                ))

    if issues:
        raise SemanticDeltaValidationError(issues)
    normalized_delta = SemanticDelta(
        nodes=delta.nodes,
        edges=tuple(proposed_edges),
        send_hints=delta.send_hints,
        complete=delta.complete,
    )
    return ValidatedSemanticDelta(
        delta=normalized_delta,
        existing_nodes=existing_nodes,
        grounding_nodes=grounding_nodes,
    )
