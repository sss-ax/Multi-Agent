"""Losslessly recoverable runtime artifacts for AFlow operator routing.

The graph is deliberately conservative: structured fields are indexed when
they are explicit, while the complete original output remains addressable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping


ARTIFACT_TYPES = (
    "candidate_answer", "reasoning_summary", "claim", "evidence_ref",
    "constraint", "uncertainty", "critique", "decision",
    "verification_result", "code", "test_result", "opaque_text",
)
EVALUATOR_FIELDS = frozenset({"supporting_facts", "gold_answer", "reference_answer", "ground_truth", "label", "score", "evaluation", "hidden_tests", "official_tests"})


class CandidateBatchState(str, Enum):
    OPEN = "open"
    FROZEN = "frozen"
    SELECTED = "selected"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class AgentArtifact:
    artifact_id: str
    artifact_type: str
    content: Any
    producer_operator: str
    invocation_id: str
    source_node_ids: tuple[str, ...] = ()
    dependency_node_ids: tuple[str, ...] = ()
    confidence: float | None = None
    multiplicity: int = 1
    visibility_class: str = "ROLE_SCOPED"
    original_payload_ref: str = ""
    content_digest: str = ""
    schema_version: str = "v1"


@dataclass(frozen=True)
class ArtifactOccurrence:
    occurrence_id: str
    canonical_artifact_id: str
    producer_operator: str
    invocation_id: str
    ordinal: int
    task_id: str = ""
    candidate_batch_id: str | None = None
    multiplicity_contribution: int = 1
    original_payload_ref: str = ""
    source_node_ids: tuple[str, ...] = ()
    raw_value: Any = None

    @property
    def canonical_node_id(self) -> str:
        return self.canonical_artifact_id


@dataclass
class FieldBinding:
    field_name: str
    value_kind: str = "raw"
    occurrence_ids: tuple[str, ...] = ()
    source_node_ids: tuple[str, ...] = ()
    ordered: bool = False
    preserve_multiplicity: bool = False
    resolution: str = "raw_unresolved"
    requirement_class: str = "raw_nonbinding"


@dataclass
class InvocationPlan:
    invocation_id: str
    invocation_index: int
    task_id: str
    operator: str
    candidate_batch_id: str | None
    original_kwargs: Mapping[str, Any]
    rendered_kwargs: Mapping[str, Any]
    field_bindings: Mapping[str, FieldBinding]
    mandatory_root_ids: tuple[str, ...] = ()
    selected_occurrence_ids: tuple[str, ...] = ()
    selected_source_ids: tuple[str, ...] = ()
    communication_mode: str = "original"
    contract_name: str = ""
    committed: bool = False
    fallback: bool = False
    fallback_reason: str = ""


@dataclass
class NativeInvocationPlan:
    invocation_id: str
    task_id: str
    operator: str
    candidate_batch_id: str | None
    original_kwargs: Mapping[str, Any]
    field_bindings: Mapping[str, FieldBinding]
    required_binding_names: tuple[str, ...] = ()
    mandatory_root_ids: tuple[str, ...] = ()
    required_binding_specs: tuple[Any, ...] = ()
    binding_validations: Mapping[str, Any] = None
    mandatory_occurrence_ids: tuple[str, ...] = ()
    mandatory_source_ids: tuple[str, ...] = ()
    original_kwargs_digest: str = ""


@dataclass(frozen=True)
class TestExecutionRecord:
    test_invocation_id: str
    task_id: str
    candidate_batch_id: str
    tested_code_occurrence_ids: tuple[str, ...]
    test_result_occurrence_id: str
    passed: bool
    error_payload: Any = None
    problem_source_ids: tuple[str, ...] = ()
    public_test_source_ids: tuple[str, ...] = ()
    entry_point_source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClosureValidation:
    required_root_count: int
    reached_root_count: int
    required_edge_count: int
    reached_edge_count: int
    unresolved_dependencies: tuple[str, ...] = ()
    cross_task_nodes: tuple[str, ...] = ()
    cross_batch_nodes: tuple[str, ...] = ()
    complete: bool = False


@dataclass(frozen=True)
class ArtifactSlice:
    occurrence_ids: tuple[str, ...] = ()
    canonical_node_ids: tuple[str, ...] = ()
    closure_edge_ids: tuple[str, ...] = ()
    source_node_ids: tuple[str, ...] = ()
    dependency_edge_ids: tuple[str, ...] = ()
    transport_tokens: int = 0
    complete: bool = False
    failure_reason: str = ""


@dataclass(frozen=True)
class CandidateBatch:
    batch_id: str
    task_id: str
    producer_stage: str
    occurrence_ids: tuple[str, ...] = ()
    created_at_invocation: str = ""
    frozen: bool = False
    selected_occurrence_ids: tuple[str, ...] = ()
    state: CandidateBatchState = CandidateBatchState.OPEN
    ensemble_invocation_id: str | None = None
    test_invocation_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MBPPSourceRegistry:
    task_id: str
    problem_source_id: str
    entry_point_source_id: str
    public_test_source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReservedInvocation:
    operator: str
    invocation_index: int
    invocation_id: str
    task_id: str


@dataclass(frozen=True)
class SelectionResolution:
    selected_occurrence_ids: tuple[str, ...] = ()
    resolution: str = "unresolved"
    ambiguous: bool = False
    candidate_batch_id: str | None = None


@dataclass(frozen=True)
class BindingValidation:
    valid: bool
    resolution: str
    field_name: str
    expected_count: int
    delivered_count: int
    content_equivalent: bool
    task_consistent: bool
    batch_consistent: bool
    failure_reason: str = ""


@dataclass(frozen=True)
class RequiredBindingValidation:
    required_field_count: int = 0
    valid_required_field_count: int = 0
    invalid_required_field_count: int = 0
    exact_binding_rate: float = 0.0
    validations_by_field: Mapping[str, BindingValidation] = None
    valid: bool = False


@dataclass(frozen=True)
class ClosureReplayResult:
    replayable: bool
    rendered_kwargs: Mapping[str, Any]
    required_fields_present: bool
    artifact_bindings_valid: bool
    source_bindings_valid: bool
    multiplicity_preserved: bool
    semantic_equivalent: bool
    failure_reason: str = ""


@dataclass(frozen=True)
class InvocationInputBinding:
    binding_id: str
    task_id: str
    consumer_invocation_id: str
    consumer_operator: str
    field_name: str
    artifact_occurrence_ids: tuple[str, ...] = ()
    source_node_ids: tuple[str, ...] = ()
    raw_value_digest: str = ""
    resolution: str = "raw_unresolved"
    candidate_batch_id: str | None = None


@dataclass(frozen=True)
class ArtifactDependencyEdge:
    edge_id: str
    task_id: str
    relation: str
    dependent_node_id: str
    prerequisite_node_id: str
    producer_invocation_id: str
    consumer_invocation_id: str
    input_field: str
    candidate_batch_id: str | None = None
    hard: bool = True
    provenance: str = "runtime_binding"


@dataclass(frozen=True)
class ArtifactRouteResult:
    kwargs: dict[str, Any]
    selected_artifact_ids: tuple[str, ...] = ()
    closure_artifact_ids: tuple[str, ...] = ()
    source_node_ids: tuple[str, ...] = ()
    original_input_tokens: int = 0
    selected_input_tokens: int = 0
    saved_input_tokens: int = 0
    contract_satisfied: bool = False
    fallback: bool = False
    fallback_reason: str = ""
    selection_overhead_ms: float = 0.0
    artifact_binding_hints: Mapping[str, tuple[str, ...]] = None
    candidate_batch_id: str | None = None
    closure_replayable: bool = True
    compression_reason: str = ""
    transport_sendall_tokens: int = 0
    transport_minctx_tokens: int = 0
    route_plan: Mapping[str, Any] | None = None


def stable_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def classify_output(output: Any, *, operator: str = "") -> tuple[str, ...]:
    """Classify only explicit, high-confidence fields; otherwise keep opaque text."""
    if not isinstance(output, Mapping):
        return ("opaque_text",)
    kinds: list[str] = []
    keys = {str(k).casefold() for k in output} - EVALUATOR_FIELDS
    if "answer" in keys or "solution" in keys or "final_answer" in keys:
        kinds.append("candidate_answer")
    if "response" in keys and any(marker in str(operator).casefold() for marker in ("code", "generate")):
        kinds.append("code")
    if "code" in keys or "program" in keys:
        kinds.append("code")
    if {"thought", "reasoning", "analysis", "rationale"} & keys:
        kinds.append("reasoning_summary")
    if "claim" in keys or "claims" in keys:
        kinds.append("claim")
    if "evidence" in keys or "evidence_ref" in keys or "citations" in keys:
        kinds.append("evidence_ref")
    if "constraint" in keys or "constraints" in keys:
        kinds.append("constraint")
    if "confidence" in keys or "uncertainty" in keys:
        kinds.append("uncertainty")
    if "critique" in keys or "feedback" in keys:
        kinds.append("critique")
    if "decision" in keys or "selected" in keys:
        kinds.append("decision")
    if "result" in keys or "passed" in keys or "verified" in keys:
        kinds.append("verification_result")
    if "test" in keys or "tests" in keys:
        kinds.append("test_result")
    if not kinds:
        kinds.append("opaque_text")
    return tuple(dict.fromkeys(kinds))


def artifact_records(output: Any, *, operator: str, invocation_id: str,
                     source_node_ids: Iterable[str] = (),
                     dependency_node_ids: Iterable[str] = (),
                     original_payload_ref: str = "", multiplicity: int = 1) -> tuple[AgentArtifact, ...]:
    digest = stable_digest(output)
    source = tuple(str(x) for x in source_node_ids)
    deps = tuple(str(x) for x in dependency_node_ids)
    kinds = classify_output(output, operator=operator)
    records: list[AgentArtifact] = []
    for kind in kinds:
        content = output
        if isinstance(output, Mapping):
            preferred = {
                "candidate_answer": ("answer", "solution", "final_answer"),
                "code": ("code", "program", "response"),
                "reasoning_summary": ("thought", "reasoning", "analysis", "rationale"),
                "claim": ("claim", "claims"),
                "evidence_ref": ("evidence_ref", "evidence"),
                "constraint": ("constraint", "constraints"),
                "uncertainty": ("uncertainty", "confidence"),
                "critique": ("critique", "feedback"),
                "decision": ("decision", "selected"),
                "verification_result": ("result", "verified", "passed"),
                "code": ("code", "program"),
                "test_result": ("test", "tests"),
            }.get(kind, ())
            content = next((output[k] for k in preferred if k in output), output)
        records.append(AgentArtifact(
            artifact_id=f"artifact:{kind}:{digest[:16]}", artifact_type=kind,
            content=content, producer_operator=str(operator), invocation_id=str(invocation_id),
            source_node_ids=source, dependency_node_ids=deps,
            multiplicity=max(1, int(multiplicity)), original_payload_ref=original_payload_ref,
            content_digest=digest,
        ))
    return tuple(records)
