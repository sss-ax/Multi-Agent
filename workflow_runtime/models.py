"""Pure data models for the versioned workflow runtime.

This module intentionally has no model-serving dependency.  Nodes are append-only
with respect to content and dependency metadata; operational status may change
through an explicit invalidation event without changing any digest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Set

DEFAULT_BRANCH_ID = "main"


class NodeOrigin:
    """Trust boundary for graph nodes."""

    SOURCE = "source"
    MODEL = "model"
    RUNTIME = "runtime"
    EVALUATION = "evaluation"


class VisibilityClass:
    """Visibility boundary enforced by AgentGraphViewManager."""

    PUBLIC = "public"
    ROLE_SCOPED = "role_scoped"
    TOOL_ONLY = "tool_only"
    EVALUATOR_ONLY = "evaluator_only"


_VALID_ORIGINS = frozenset({
    NodeOrigin.SOURCE,
    NodeOrigin.MODEL,
    NodeOrigin.RUNTIME,
    NodeOrigin.EVALUATION,
})
_VALID_VISIBILITIES = frozenset({
    VisibilityClass.PUBLIC,
    VisibilityClass.ROLE_SCOPED,
    VisibilityClass.TOOL_ONLY,
    VisibilityClass.EVALUATOR_ONLY,
})


def normalize_provenance(
    provenance: Mapping[str, Any] | None = None,
    *,
    owner: str = "",
    created_by_role: str = "",
    node_type: str = "",
) -> Dict[str, Any]:
    """Return a backward-compatible, explicit provenance record.

    Existing callers can continue passing sparse provenance.  The compatibility
    inference deliberately treats only obvious source/mandatory nodes as public;
    generated semantic nodes default to role-scoped visibility.
    """
    result = dict(provenance or {})
    origin = str(result.get("origin", "")).strip()
    visibility = str(result.get("visibility_class", "")).strip()
    trust = str(result.get("trust", "")).strip()
    role = str(created_by_role or owner or "").strip()

    if origin not in _VALID_ORIGINS:
        if role in {"dataset", "user", "source"} or node_type.endswith("_source"):
            origin = NodeOrigin.SOURCE
        elif role in {"evaluator", "evaluation"}:
            origin = NodeOrigin.EVALUATION
        elif role in {"runtime", "tool", "retriever"}:
            origin = NodeOrigin.RUNTIME
        else:
            origin = NodeOrigin.MODEL
    if visibility not in _VALID_VISIBILITIES:
        if result.get("communication_scope") == "MANDATORY":
            visibility = VisibilityClass.PUBLIC
        elif origin == NodeOrigin.EVALUATION or role in {"evaluator", "evaluation"}:
            visibility = VisibilityClass.EVALUATOR_ONLY
        elif result.get("tool_only") or node_type in {"official_test_source", "private_test_source"}:
            visibility = VisibilityClass.TOOL_ONLY
        elif origin == NodeOrigin.SOURCE and role in {"dataset", "user", "source"}:
            visibility = VisibilityClass.PUBLIC
        else:
            visibility = VisibilityClass.ROLE_SCOPED
    if not trust:
        trust = {
            NodeOrigin.SOURCE: "trusted_input",
            NodeOrigin.MODEL: "model_generated",
            NodeOrigin.RUNTIME: "runtime_derived",
            NodeOrigin.EVALUATION: "evaluator_only",
        }[origin]
    result.update({"origin": origin, "trust": trust, "visibility_class": visibility})
    return result


def node_visibility_class(node: Any) -> str:
    """Read a node's explicit visibility with a safe default."""
    provenance = getattr(node, "provenance", {})
    if isinstance(provenance, Mapping):
        value = str(provenance.get("visibility_class", "")).strip()
        if value in _VALID_VISIBILITIES:
            return value
    normalized = normalize_provenance(
        provenance,
        owner=str(getattr(node, "owner", "")),
        created_by_role=str(getattr(node, "created_by_role", "")),
        node_type=str(getattr(node, "type", "")),
    )
    return str(normalized["visibility_class"])


def node_visible_to_role(node: Any, role: str) -> bool:
    """Return whether a node may enter a model/tool role view."""
    visibility = node_visibility_class(node)
    if visibility == VisibilityClass.PUBLIC:
        return True
    if visibility == VisibilityClass.TOOL_ONLY:
        return role == "tool"
    if visibility == VisibilityClass.EVALUATOR_ONLY:
        return False
    provenance = getattr(node, "provenance", {})
    owner = str(getattr(node, "owner", ""))
    created_by = str(getattr(node, "created_by_role", ""))
    allowed_roles = ()
    if isinstance(provenance, Mapping):
        raw = provenance.get("roles", provenance.get("allowed_roles", ()))
        if isinstance(raw, (list, tuple, set, frozenset)):
            allowed_roles = tuple(str(item) for item in raw)
    return role in allowed_roles or role in {owner, created_by}


@dataclass(frozen=True)
class RuntimeFingerprint:
    model_id: str
    tokenizer_id: str
    prompt_template_digest: str = ""
    graph_schema_version: str = "v1"
    code_version: str = ""
    dtype: str = ""
    model_revision: str = ""
    tokenizer_revision: str = ""
    chat_template_version: str = ""
    attention_backend: str = ""
    position_encoding: str = ""
    quantization: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "prompt_template_digest": self.prompt_template_digest,
            "graph_schema_version": self.graph_schema_version,
            "code_version": self.code_version,
            "dtype": self.dtype,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "chat_template_version": self.chat_template_version,
            "attention_backend": self.attention_backend,
            "position_encoding": self.position_encoding,
            "quantization": self.quantization,
        }


@dataclass
class NodeVersion:
    node_id: str
    task_id: str
    branch_id: str
    logical_id: str
    type: str
    version: int
    content: Any
    owner: str
    status: str = "ready"
    confidence: Any = None
    dependency_versions: Dict[str, str] = field(default_factory=dict)
    source_refs: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, bool] = field(default_factory=dict)
    content_digest: str = ""
    dependency_digest: str = ""
    full_digest: str = ""
    parent_version_id: Optional[str] = None
    created_at: float = 0.0
    created_by_role: str = ""
    run_id: str = ""
    operation_batch_id: str = ""

    @property
    def slot(self) -> tuple[str, str, str]:
        return self.task_id, self.branch_id, self.logical_id

    def is_operationally_valid(self) -> bool:
        return self.status not in {
            "empty",
            "stale",
            "invalid",
            "superseded",
            "conflict",
        }

    def has_validation(self, name: str) -> bool:
        return bool(self.validation.get(name, False))

    def is_semantically_verified(self) -> bool:
        return self.has_validation("evidence_checked") or self.has_validation(
            "model_judged_correct"
        )


@dataclass(frozen=True)
class DependencyEdge:
    edge_id: str
    source: str
    target: str
    relation: str
    task_id: str
    branch_id: str
    created_at: float = 0.0
    created_by_role: str = ""
    run_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphState:
    nodes: Dict[str, NodeVersion] = field(default_factory=dict)
    edges: List[DependencyEdge] = field(default_factory=list)
    status_events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ContextSlice:
    slice_id: str
    task_id: str
    branch_id: str
    target_role: str
    policy: str
    root_node_ids: List[str]
    visible_node_ids: List[str]
    visible_edge_ids: List[str]
    omitted_node_ids: List[str]
    boundary_node_ids: List[str]
    read_versions: Dict[str, str]
    dependency_digest: str
    token_count: int
    render_mode: str
    inclusion_reasons: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    node_order_digest: str = ""
    visible_fragment_ids: List[str] = field(default_factory=list)
    local_node_id_by_node: Dict[str, str] = field(default_factory=dict)
    fragment_provenance: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class CachedResult:
    cache_key: str
    task_id: str
    branch_id: str
    logical_id: str
    role: str
    node: NodeVersion
    runtime_fingerprint: RuntimeFingerprint
    created_at: float
    cache_policy: str


@dataclass
class InvalidationReport:
    changed_node_ids: List[str]
    stale_node_ids: List[str]
    preserved_node_ids: List[str]
    reason: str
    fanout: int


@dataclass
class RecomputePlan:
    task_id: str
    branch_id: str
    dirty_node_ids: List[str]
    reusable_node_ids: List[str]
    recompute_frontier: List[str]
    required_roles: List[str]
    reason_by_node: Dict[str, str] = field(default_factory=dict)


@dataclass
class GraphOperationBatch:
    node_ids: List[str] = field(default_factory=list)
    edge_ids: List[str] = field(default_factory=list)


@dataclass
class CacheLookup:
    hit: bool
    artifact: Optional[CachedResult] = None
    reason: str = ""
