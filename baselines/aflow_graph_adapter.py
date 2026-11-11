"""Thin GraphStore adapter for AFlow operator communication.

This module deliberately does not implement a second communication algorithm.
It maps AFlow's operator-shaped inputs and outputs onto the repository's
canonical GraphStore, AgentGraphViewManager, communication policies,
dependency closure, context slicer, semantic resolver, and telemetry.
"""

from __future__ import annotations

import inspect
import json
import re
import time
import types
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Mapping

from workflow_runtime.agent_graph_view import AgentGraphView, AgentGraphViewManager
from workflow_runtime.communication import GraphCommunicationPolicy, SendAllPolicy
from workflow_runtime.context_slicer import build_context_slice, render_compact_context_slice
from workflow_runtime.delta_closure import dependency_closure
from workflow_runtime.delta_extractor import extract_delta_candidates
from workflow_runtime.graph_store import GraphStore
from workflow_runtime.models import NodeOrigin, VisibilityClass
from workflow_runtime.semantic_contract import SemanticContract
from workflow_runtime.semantic_resolver import SemanticResolver, all_requirements_sufficient
from workflow_runtime.telemetry import WorkflowTelemetry

from .aflow_reuse import (
    AFlowArtifactRecord,
    AFlowInvocationKey,
    AFlowInvocationReuse,
    normalize_output,
    output_digest,
)
from .aflow_profiles import (
    AFlowDeliveredArtifact,
    AFlowOptimizationProfile,
    AFlowOperatorInput,
    get_aflow_profile,
    RequiredBindingSpec,
)
from .aflow_artifact_graph import (
    ArtifactDependencyEdge,
    ArtifactOccurrence,
    CandidateBatch,
    CandidateBatchState,
    MBPPSourceRegistry,
    BindingValidation,
    ClosureReplayResult,
    FieldBinding,
    InvocationPlan,
    NativeInvocationPlan,
    TestExecutionRecord,
    RequiredBindingValidation,
    ReservedInvocation,
    SelectionResolution,
    InvocationInputBinding,
    artifact_records,
    stable_digest,
)
from .aflow_artifact_dependencies import (
    DATA_DEPENDENCY_RELATIONS,
    capture_invocation_input_bindings,
    materialize_dependency_edges,
)
from .aflow_artifact_router import route_receiver_context, resolve_ensemble_selection
from .hotpot_semantic_graph import (
    SemanticGraphBuild,
    build_semantic_graph,
    select_semantic_paths,
)
from .hotpot_claim_graph import (
    ClaimGraphBuild,
    extract_claims,
    select_claim_proof,
)


@dataclass(frozen=True)
class AFlowCompiledOutput:
    """Deterministically compiled representation of one AFlow output."""

    node_ids: tuple[str, ...]
    primary_node_id: str
    operator: str
    invocation_index: int
    output_kind: str
    content_digest: str = ""
    invocation_node_id: str = ""
    reused: bool = False
    occurrence_count: int = 1
    candidate_role: str = ""


@dataclass(frozen=True)
class AFlowContextResult:
    """Receiver context and accounting for one AFlow communication event."""

    text: str
    receiver: str
    sender: str
    root_node_ids: tuple[str, ...]
    sent_node_ids: tuple[str, ...]
    closure_added_node_ids: tuple[str, ...]
    candidate_tokens: int
    sent_tokens: int
    contract_satisfied: bool
    missing_requirements: tuple[str, ...]


@dataclass(frozen=True)
class AFlowContextPlan:
    """Auditable, receiver-specific context selection for one Operator call."""

    operator: str
    policy: str
    requested_root_ids: tuple[str, ...] = ()
    selected_node_ids: tuple[str, ...] = ()
    closure_node_ids: tuple[str, ...] = ()
    original_input_tokens: int = 0
    effective_input_tokens: int = 0
    contract_satisfied: bool = False
    fallback: bool = False
    fallback_reason: str = ""
    selection_overhead_ms: float = 0.0
    level: str = "L0"
    top_k_documents: int = 0
    document_scores: tuple[Mapping[str, Any], ...] = ()
    sentence_scores: tuple[Mapping[str, Any], ...] = ()
    question_entities: tuple[str, ...] = ()
    covered_entities: tuple[str, ...] = ()
    shared_entities: tuple[str, ...] = ()
    boundary_margin: float = 0.0
    risk_reasons: tuple[str, ...] = ()
    risk_score: float = 0.0
    critical_reasons: tuple[str, ...] = ()
    soft_reasons: tuple[str, ...] = ()
    decision: str = "accept_l0"
    sample_id: str = ""
    level_attempts: tuple[Mapping[str, Any], ...] = ()
    bridge_paths: tuple[Mapping[str, Any], ...] = ()
    cache_hit: bool = False
    attempted_levels: tuple[str, ...] = ()
    effective_level: str = "L0"
    tokens_by_level: Mapping[str, int] = None
    risk_by_level: Mapping[str, float] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "policy": self.policy,
            "requested_root_ids": list(self.requested_root_ids),
            "selected_node_ids": list(self.selected_node_ids),
            "closure_node_ids": list(self.closure_node_ids),
            "original_input_tokens": self.original_input_tokens,
            "effective_input_tokens": self.effective_input_tokens,
            "contract_satisfied": self.contract_satisfied,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
            "selection_overhead_ms": self.selection_overhead_ms,
            "level": self.level,
            "top_k_documents": self.top_k_documents,
            "document_scores": [dict(item) for item in self.document_scores],
            "sentence_scores": [dict(item) for item in self.sentence_scores],
            "question_entities": list(self.question_entities),
            "covered_entities": list(self.covered_entities),
            "shared_entities": list(self.shared_entities),
            "boundary_margin": self.boundary_margin,
            "risk_reasons": list(self.risk_reasons),
            "risk_score": self.risk_score,
            "critical_reasons": list(self.critical_reasons),
            "soft_reasons": list(self.soft_reasons),
            "decision": self.decision,
            "sample_id": self.sample_id,
            "level_attempts": [dict(item) for item in self.level_attempts],
            "bridge_paths": [dict(item) for item in self.bridge_paths],
            "cache_hit": self.cache_hit,
            "attempted_levels": list(self.attempted_levels),
            "effective_level": self.effective_level,
            "tokens_by_level": dict(self.tokens_by_level or {}),
            "risk_by_level": dict(self.risk_by_level or {}),
        }


@dataclass(frozen=True)
class HotpotSelectionResult:
    root_node_ids: tuple[str, ...]
    selected_node_ids: tuple[str, ...]
    closure_node_ids: tuple[str, ...]
    document_scores: tuple[Mapping[str, Any], ...] = ()
    sentence_scores: tuple[Mapping[str, Any], ...] = ()
    question_entities: tuple[str, ...] = ()
    covered_entities: tuple[str, ...] = ()
    shared_entities: tuple[str, ...] = ()
    boundary_margin: float = 0.0
    bridge_paths: tuple[Mapping[str, Any], ...] = ()
    selected_document_ids: tuple[str, ...] = ()
    selected_sentence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextRisk:
    score: float
    reasons: tuple[str, ...] = ()
    critical_reasons: tuple[str, ...] = ()
    soft_reasons: tuple[str, ...] = ()
    decision: str = "accept_l0"


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def extract_weighted_solution_letter(
    response: Any,
    allowed_letters: Iterable[str],
) -> str | None:
    """Extract one validated candidate ID from common formatter outputs."""
    allowed = {str(item).strip().upper() for item in allowed_letters}
    if not allowed:
        return None
    if isinstance(response, Mapping):
        for key in ("solution_letter", "response", "output", "text"):
            if key in response:
                value = extract_weighted_solution_letter(response[key], allowed)
                if value is not None:
                    return value
        return None
    text = str(response or "").strip()
    patterns = (
        r"<solution_letter>\s*([A-Za-z])\s*</solution_letter>",
        r"solution_letter\s*(?:=|:)\s*[\"']?\s*([A-Za-z])",
        r"^[\"']?\s*([A-Za-z])\s*[\"']?$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            letter = match.group(1).upper()
            return letter if letter in allowed else None
    return None


def _operator_kind(operator: str, output: Any) -> str:
    name = operator.casefold()
    mapping = (
        ("test", "execution"),
        ("program", "calculation"),
        ("code", "code"),
        ("ensemble", "result"),
        ("answer", "candidate_answer"),
        ("refine", "final_answer"),
        ("fix", "code"),
    )
    for marker, kind in mapping:
        if marker in name:
            return kind
    if isinstance(output, Mapping) and any(key in output for key in ("code", "solution")):
        return "code"
    return "candidate_answer"


def _is_candidate_output(session: "AFlowGraphSession", operator: str, output_kind: str) -> bool:
    """Return whether an output belongs to the current ensemble candidate batch."""
    if session.profile is not None:
        # Keep the adapter's small standalone/unit-test API backwards
        # compatible for synthetic operator names.  Archived workflows use
        # the registered names and therefore remain strictly profile-bound.
        if session.profile.is_candidate_operator(operator):
            return True
        if str(operator).casefold() not in {
            item.casefold()
            for item in (*session.profile.ensemble_operators, *session.profile.reusable_invocations)
        } and str(operator).casefold() in {"solver", "answergenerate"}:
            return output_kind in {"candidate_answer", "code", "calculation"}
        return False
    if output_kind not in {"candidate_answer", "code", "calculation"}:
        return False
    lowered = str(operator).casefold()
    return not any(marker in lowered for marker in ("ensemble", "program", "test"))


class AFlowOperatorOutputCompiler:
    """Compile AFlow's heterogeneous operator output without an LLM call."""

    def compile(
        self,
        session: "AFlowGraphSession",
        *,
        operator: str,
        output: Any,
        input_node_ids: Iterable[str] = (),
        invocation_index: int | None = None,
        input_kwargs: Mapping[str, Any] | None = None,
    ) -> AFlowCompiledOutput:
        index = int(invocation_index or session.next_invocation(operator))
        kind = _operator_kind(operator, output)
        session.telemetry.record_aflow_metric(aflow_operator_invocations=1)
        if session.content_dedup:
            return self._compile_canonical(
                session,
                operator=operator,
                output=output,
                input_node_ids=input_node_ids,
                invocation_index=index,
                kind=kind,
                input_kwargs=input_kwargs,
            )
        return self._compile_legacy(
            session,
            operator=operator,
            output=output,
            input_node_ids=input_node_ids,
            invocation_index=index,
            kind=kind,
            input_kwargs=input_kwargs,
        )

    def _compile_legacy(
        self,
        session: "AFlowGraphSession",
        *,
        operator: str,
        output: Any,
        input_node_ids: Iterable[str],
        invocation_index: int,
        kind: str,
        input_kwargs: Mapping[str, Any] | None = None,
    ) -> AFlowCompiledOutput:
        operator_key = re.sub(r"[^a-zA-Z0-9_]+", "_", operator).strip("_").lower() or "operator"
        logical_id = f"aflow_{operator_key}_{invocation_index}"
        content: Any = output
        mapping = _as_mapping(output)
        if mapping is not None:
            # Preserve structured fields for downstream contracts, while
            # retaining raw text for auditability and exact replay.
            content = dict(mapping)
            content.setdefault("raw_output", _text_value(output))
        else:
            content = {"value": output, "raw_output": _text_value(output)}

        source_refs = tuple(str(item) for item in input_node_ids)
        candidate_role = ""
        if _is_candidate_output(session, operator, kind) and session.profile is not None:
            candidate_role = session.profile.candidate_signature(
                operator=operator,
                kwargs=input_kwargs or {},
                output=output,
            )
        dependency_versions = {}
        for node_id in source_refs:
            node = session.store.node(node_id)
            dependency_versions[node.logical_id] = node.node_id
        node = session.store.add_node(
            task_id=session.task_id,
            branch_id=session.branch_id,
            logical_id=logical_id,
            node_type=kind,
            content=content,
            owner=operator,
            source_refs=source_refs,
            dependency_versions=dependency_versions,
            provenance={
                "origin": NodeOrigin.MODEL,
                "visibility_class": VisibilityClass.ROLE_SCOPED,
                "communication_scope": "OPTIONAL_POLICY",
                "aflow_operator": operator,
                "aflow_invocation_index": invocation_index,
                "aflow_output_kind": kind,
                "candidate_role": candidate_role,
                "artifact_kind": kind,
                "aflow_profile": session.profile.profile_name if session.profile else "",
            },
            created_by_role=operator,
            run_id=session.run_id,
        )
        for source_id in source_refs:
            session.store.add_edge(
                source=source_id,
                target=node.node_id,
                relation="derived_from",
                created_by_role=operator,
                run_id=session.run_id,
            )
        session._refresh_views()
        session._raw_outputs[node.node_id] = output
        session._last_output_ids = (node.node_id,)
        if _is_candidate_output(session, operator, kind):
            session._candidate_pool_ids.append(node.node_id)
        return AFlowCompiledOutput(
            node_ids=(node.node_id,),
            primary_node_id=node.node_id,
            operator=operator,
            invocation_index=invocation_index,
            output_kind=kind,
        )

    def _compile_canonical(
        self,
        session: "AFlowGraphSession",
        *,
        operator: str,
        output: Any,
        input_node_ids: Iterable[str],
        invocation_index: int,
        kind: str,
        input_kwargs: Mapping[str, Any] | None = None,
    ) -> AFlowCompiledOutput:
        operator_key = re.sub(r"[^a-zA-Z0-9_]+", "_", operator).strip("_").lower() or "operator"
        source_refs = tuple(str(item) for item in input_node_ids)
        normalized = normalize_output(output)
        digest = output_digest(kind, normalized)
        invocation = session.store.add_node(
            task_id=session.task_id,
            branch_id=session.branch_id,
            logical_id=f"aflow_invocation_{operator_key}_{invocation_index}",
            node_type="aflow_invocation",
            content={
                "operator": operator,
                "invocation_index": invocation_index,
                "content_digest": digest,
                "raw_output": normalized,
            },
            owner=operator,
            source_refs=source_refs,
            provenance={
                "origin": NodeOrigin.MODEL,
                "visibility_class": VisibilityClass.ROLE_SCOPED,
                "communication_scope": "INTERNAL_ONLY",
                "aflow_operator": operator,
                "aflow_invocation_index": invocation_index,
            },
            created_by_role=operator,
            run_id=session.run_id,
        )
        for source_id in source_refs:
            session.store.add_edge(
                source=source_id,
                target=invocation.node_id,
                relation="input_to",
                created_by_role=operator,
                run_id=session.run_id,
            )

        is_candidate = _is_candidate_output(session, operator, kind)
        candidate_role = ""
        if is_candidate and session.profile is not None:
            candidate_role = session.profile.candidate_signature(operator=operator, kwargs=input_kwargs or {}, output=output)
        candidate_value = session.candidate_projection(operator=operator, output=output)
        digest_value = candidate_value if is_candidate and candidate_value is not output else normalized
        digest = output_digest(kind, digest_value)
        key = (kind, digest, candidate_role)
        record = session._artifact_by_digest.get(key)
        reused = record is not None
        if record is None:
            artifact = session.store.add_node(
                task_id=session.task_id,
                branch_id=session.branch_id,
                logical_id=f"aflow_artifact_{kind}_{digest[:16]}",
                node_type="aflow_artifact",
                content=normalized,
                owner=operator,
                source_refs=source_refs,
                provenance={
                    "origin": NodeOrigin.MODEL,
                    "visibility_class": VisibilityClass.ROLE_SCOPED,
                    "communication_scope": "OPTIONAL_POLICY",
                    "aflow_operator": operator,
                    "aflow_output_kind": kind,
                    "content_digest": digest,
                    "artifact_kind": kind,
                    "aflow_profile": session.profile.profile_name if session.profile else "",
                    "candidate_role": candidate_role,
                },
                created_by_role=operator,
                run_id=session.run_id,
            )
            record = AFlowArtifactRecord(
                node_id=artifact.node_id,
                output_kind=kind,
                content_digest=digest,
                raw_output=output,
                artifact_kind=kind,
                candidate_role=candidate_role,
                profile=session.profile.profile_name if session.profile else "",
            )
            session._artifact_by_digest[key] = record
            session._raw_outputs[artifact.node_id] = output
            session._artifact_token_cost[artifact.node_id] = session._count_tokens(output)
            session.telemetry.emit("aflow_artifact_created", {
                "task_id": session.task_id,
                "operator": operator,
                "node_id": artifact.node_id,
                "content_digest": digest,
                "output_kind": kind,
            })
            session.telemetry.record_aflow_metric(aflow_canonical_artifacts=1)
        else:
            record.occurrence_count += 1
            session.telemetry.emit("aflow_duplicate_artifact_detected", {
                "task_id": session.task_id,
                "operator": operator,
                "canonical_node_id": record.node_id,
                "content_digest": digest,
                "occurrence_count": record.occurrence_count,
            })
            if is_candidate:
                session.telemetry.record_aflow_metric(aflow_duplicate_candidate_occurrences=1)
        session.store.add_edge(
            source=invocation.node_id,
            target=record.node_id,
            relation="materializes",
            created_by_role=operator,
            run_id=session.run_id,
        )
        session._raw_outputs[record.node_id] = record.raw_output
        session._last_output_ids = (record.node_id,)
        session._occurrence_artifact_ids[invocation.node_id] = record.node_id
        if str(operator).casefold() == "test" and isinstance(output, Mapping):
            tested_solution = output.get("solution", "")
            session.telemetry.record_aflow_diagnostic({
                "task_id": session.task_id,
                "test_result": "passed" if output.get("result") is True else "failed",
                "tested_code_digest": output_digest("tested_code", tested_solution),
            })
        if is_candidate:
            session._candidate_occurrence_ids.append(invocation.node_id)
            is_new_candidate = record.node_id not in session._candidate_artifact_ids
            if record.node_id not in session._candidate_artifact_ids:
                session._candidate_artifact_ids.append(record.node_id)
            session._candidate_tokens_before_dedup += session._count_tokens(output)
            session.telemetry.record_aflow_metric(
                aflow_candidate_occurrences=1,
                aflow_candidate_tokens_before_dedup=session._count_tokens(output),
            )
            session.telemetry.record_aflow_metric(
                aflow_canonical_candidates=1 if is_new_candidate else 0
            )
        session._refresh_views()
        return AFlowCompiledOutput(
            node_ids=(record.node_id,),
            primary_node_id=record.node_id,
            operator=operator,
            invocation_index=invocation_index,
            output_kind=kind,
            content_digest=digest,
            invocation_node_id=invocation.node_id,
            reused=reused,
            occurrence_count=record.occurrence_count,
            candidate_role=candidate_role,
        )


class AFlowGraphSession:
    """Thin AFlow adapter over the existing graph communication runtime."""

    def __init__(
        self,
        *,
        task_id: str,
        task: str,
        dataset: str = "",
        workflow_round: int = 0,
        task_metadata: Mapping[str, Any] | None = None,
        profile: AFlowOptimizationProfile | None = None,
        profile_explicit: bool = False,
        branch_id: str = "main",
        policy: GraphCommunicationPolicy | None = None,
        telemetry: WorkflowTelemetry | None = None,
        run_id: str = "",
        endpoints: Iterable[str] = (),
        content_dedup: bool = False,
        invocation_reuse: bool = False,
        dedup_mode: str = "exact",
        model_fingerprint: str = "",
        generation_config: Mapping[str, Any] | None = None,
        tokenizer: Any = None,
        context_mode: str = "",
        context_policy: str = "full",
        context_budget_tokens: int = 0,
        context_top_k_documents: int = 4,
        context_fallback: str = "original",
        context_recovery: str = "none",
        context_recovery_top_k_documents: int = 6,
        semantic_max_paths: int = 2,
        semantic_max_document_hops: int = 2,
        semantic_budget_tokens: int = 512,
        semantic_entity_policy: str = "deterministic_v1",
        artifact_routing_policy: str = "send_all",
        artifact_fallback: str = "original",
        artifact_preserve_multiplicity: bool = True,
    ) -> None:
        self.task_id = str(task_id)
        self.task = str(task)
        self.dataset = str(dataset)
        self.workflow_round = int(workflow_round or {"GSM8K": 10, "DROP": 3, "HotpotQA": 3, "MATH": 5, "MBPP": 14, "HumanEval": 5}.get(self.dataset, 0))
        self.task_metadata = dict(task_metadata or {})
        self.profile = profile or get_aflow_profile(self.dataset, self.workflow_round)
        self.profile_explicit = bool(profile_explicit)
        self.branch_id = str(branch_id)
        self.run_id = str(run_id or f"aflow:{self.task_id}")
        self.telemetry = telemetry or WorkflowTelemetry()
        if dedup_mode not in {"exact", "structured"}:
            raise ValueError(f"unsupported AFlow dedup mode: {dedup_mode}")
        supports_content_dedup = not self.profile or self.profile.supports_content_dedup
        supports_invocation_reuse = not self.profile or self.profile.supports_invocation_reuse
        self.content_dedup_requested = bool(content_dedup)
        self.invocation_reuse_requested = bool(invocation_reuse)
        self.content_dedup = bool(content_dedup and supports_content_dedup)
        self.invocation_reuse_enabled = bool(invocation_reuse and supports_invocation_reuse)
        self.context_mode = str(context_mode or "")
        self.artifact_enabled = self.context_mode in {"artifact_shadow", "artifact_sendall", "artifact_minctx"}
        self.context_policy = str(context_policy or "full")
        self.context_budget_tokens = max(0, int(context_budget_tokens or 0))
        self.context_top_k_documents = max(1, int(context_top_k_documents or 4))
        self.context_fallback = str(context_fallback or "original")
        self.context_recovery = str(context_recovery or "none")
        self.context_recovery_top_k_documents = max(1, int(context_recovery_top_k_documents or 6))
        self.context_risk_policy = "balanced_v1"
        self.semantic_max_paths = max(1, int(semantic_max_paths or 2))
        self.semantic_max_document_hops = max(1, int(semantic_max_document_hops or 2))
        self.semantic_budget_tokens = max(1, int(semantic_budget_tokens or 512))
        self.semantic_entity_policy = str(semantic_entity_policy or "deterministic_v1")
        self.artifact_routing_policy = str(artifact_routing_policy or "send_all")
        self.artifact_fallback = str(artifact_fallback or "original")
        self.artifact_preserve_multiplicity = bool(artifact_preserve_multiplicity)
        if self.context_policy not in {"full", "topk_documents", "flat_topk_documents", "dependency_minimal", "semantic_minimal", "semantic_shadow"}:
            raise ValueError(f"unsupported AFlow context policy: {self.context_policy}")
        if self.context_recovery not in {"none", "adaptive", "semantic_adaptive"}:
            raise ValueError(f"unsupported AFlow context recovery: {self.context_recovery}")
        if self.semantic_entity_policy != "deterministic_v1":
            raise ValueError(f"unsupported semantic entity policy: {self.semantic_entity_policy}")
        if self.context_fallback not in {"original"}:
            raise ValueError(f"unsupported AFlow context fallback: {self.context_fallback}")
        if self.artifact_routing_policy not in {"send_all", "contract_minimal"}:
            raise ValueError(f"unsupported artifact routing policy: {self.artifact_routing_policy}")
        if self.artifact_fallback not in {"original"}:
            raise ValueError(f"unsupported artifact fallback: {self.artifact_fallback}")
        self.optimization_status = {
            "content_dedup": {
                "requested": self.content_dedup_requested,
                "applied": self.content_dedup,
                "reason": "" if self.content_dedup or not self.content_dedup_requested else "profile_not_supported",
            },
            "invocation_reuse": {
                "requested": self.invocation_reuse_requested,
                "applied": self.invocation_reuse_enabled,
                "reason": "" if self.invocation_reuse_enabled or not self.invocation_reuse_requested else "profile_not_supported",
            },
            "weighted_ensemble": {
                "requested": bool(self.profile and self.profile.ensemble_strategy.startswith("weighted")),
                "applied": bool(
                    self.content_dedup
                    and self.profile
                    and self.profile.ensemble_strategy.startswith("weighted")
                    and self.profile.ensemble_operators
                ),
                "reason": (
                    ""
                    if self.content_dedup and self.profile and self.profile.ensemble_strategy.startswith("weighted")
                    else "profile_strategy"
                ),
            },
            "reasoning_context": {
                "requested": self.context_mode == "graph_minctx",
                "policy": self.context_policy,
                "applied": bool(
                    self.context_mode == "graph_minctx"
                    and self.dataset.casefold() == "hotpotqa"
                    and self.profile is not None
                    and self.profile.supports_context_compression
                ),
                "fallback": self.context_fallback,
                "risk_policy": self.context_risk_policy,
            },
        }
        if self.content_dedup_requested and not self.content_dedup:
            self.telemetry.emit("aflow_optimization_skipped", {
                "task_id": self.task_id, "dataset": self.dataset,
                "optimization": "content_dedup", "reason": "profile_not_supported",
            })
        if self.invocation_reuse_requested and not self.invocation_reuse_enabled:
            self.telemetry.emit("aflow_optimization_skipped", {
                "task_id": self.task_id, "dataset": self.dataset,
                "optimization": "invocation_reuse", "reason": "profile_not_supported",
            })
        if self.invocation_reuse_enabled and not self.content_dedup:
            raise ValueError("AFlow invocation reuse requires content_dedup")
        self.dedup_mode = str(dedup_mode)
        self.model_fingerprint = str(model_fingerprint)
        self.generation_config = dict(generation_config or {})
        self.tokenizer = tokenizer
        self.store = GraphStore()
        self.policy = policy or SendAllPolicy()
        self._invocations: dict[str, int] = {}
        self._last_output_ids: tuple[str, ...] = ()
        self._candidate_pool_ids: list[str] = []
        self._candidate_occurrence_ids: list[str] = []
        self._candidate_artifact_ids: list[str] = []
        self._artifact_by_digest: dict[tuple[str, str, str], AFlowArtifactRecord] = {}
        self._artifact_token_cost: dict[str, int] = {}
        self._occurrence_artifact_ids: dict[str, str] = {}
        self._candidate_tokens_before_dedup = 0
        self._raw_outputs: dict[str, Any] = {}
        self._hotpot_source_node_ids: tuple[str, ...] = ()
        self._hotpot_document_node_ids: tuple[str, ...] = ()
        self._hotpot_sentence_node_ids: tuple[str, ...] = ()
        self._context_plans: list[AFlowContextPlan] = []
        self._context_plan_cache: dict[tuple[Any, ...], tuple[Any, ...]] = {}
        self._context_plan_cache_hits = 0
        self._context_unique_selection_runs = 0
        self._semantic_build: SemanticGraphBuild | None = None
        self._semantic_entity_node_ids: dict[str, str] = {}
        self._semantic_plan_cache: dict[tuple[Any, ...], Any] = {}
        self._claim_build: ClaimGraphBuild | None = None
        self._claim_node_ids: dict[str, str] = {}
        self._claim_plan_cache: dict[tuple[Any, ...], Any] = {}
        self._artifact_records: dict[str, Any] = {}
        self._artifact_occurrences: dict[str, ArtifactOccurrence] = {}
        self._artifact_dependency_edges: dict[str, ArtifactDependencyEdge] = {}
        self._invocation_bindings: dict[str, tuple[InvocationInputBinding, ...]] = {}
        self._candidate_batches: dict[str, CandidateBatch] = {}
        self._active_candidate_batch_id: str | None = None
        self._last_closed_candidate_batch_id: str | None = None
        self._selected_occurrence_ids: tuple[str, ...] = ()
        self._invocation_plans: dict[str, InvocationPlan] = {}
        self._native_invocation_plans: dict[str, NativeInvocationPlan] = {}
        self._test_executions: dict[str, TestExecutionRecord] = {}
        self._latest_failed_test_execution: TestExecutionRecord | None = None
        self._test_execution_by_batch: dict[str, TestExecutionRecord] = {}
        self._mbpp_source_registry: MBPPSourceRegistry | None = None
        self._source_payloads: dict[str, Any] = {}
        self._artifact_route_plans: list[dict[str, Any]] = []
        self._invocation_reuse = AFlowInvocationReuse() if self.invocation_reuse_enabled else None

        task_node = self.store.add_node(
            task_id=self.task_id,
            branch_id=self.branch_id,
            logical_id="task",
            node_type="task",
            content=self.task,
            owner="dataset",
            provenance={
                "origin": NodeOrigin.SOURCE,
                "visibility_class": VisibilityClass.PUBLIC,
                "communication_scope": "MANDATORY",
            },
            created_by_role="dataset",
            run_id=self.run_id,
        )
        query_node = self.store.add_node(
            task_id=self.task_id,
            branch_id=self.branch_id,
            logical_id="query_spec",
            node_type="query_spec",
            content={"dataset": self.dataset, "task": self.task},
            owner="dataset",
            source_refs=(task_node.node_id,),
            provenance={
                "origin": NodeOrigin.SOURCE,
                "visibility_class": VisibilityClass.PUBLIC,
                "communication_scope": "MANDATORY",
            },
            created_by_role="dataset",
            run_id=self.run_id,
        )
        self.store.add_edge(
            source=task_node.node_id,
            target=query_node.node_id,
            relation="derived_from",
            created_by_role="dataset",
            run_id=self.run_id,
        )
        if self.dataset.casefold() == "hotpotqa":
            self._build_hotpot_source_graph(query_node.node_id)
        self.views = AgentGraphViewManager(
            self.store,
            task_id=self.task_id,
            branch_id=self.branch_id,
            agents=tuple(dict.fromkeys(str(item) for item in endpoints)),
        )
        self.compiler = AFlowOperatorOutputCompiler()
        self._refresh_views()

    def _build_hotpot_source_graph(self, query_node_id: str) -> None:
        """Ingest public HotpotQA context without importing evaluator labels."""
        context = self.task_metadata.get("context")
        if not isinstance(context, (list, tuple)):
            return
        question = str(self.task_metadata.get("question") or self.task)
        query_node = self.store.node(query_node_id)
        question_node = self.store.add_node(
            task_id=self.task_id,
            branch_id=self.branch_id,
            logical_id="question_source",
            node_type="question_source",
            content=question,
            owner="dataset",
            source_refs=(query_node_id,),
            provenance={
                "origin": NodeOrigin.SOURCE,
                "visibility_class": VisibilityClass.PUBLIC,
                "communication_scope": "MANDATORY",
                "source_kind": "question",
            },
            created_by_role="dataset",
            run_id=self.run_id,
        )
        self.store.add_edge(
            source=query_node_id,
            target=question_node.node_id,
            relation="derived_from",
            created_by_role="dataset",
            run_id=self.run_id,
        )
        document_ids: list[str] = []
        sentence_ids: list[str] = []
        for document_index, item in enumerate(context):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            title = str(item[0])
            sentences = item[1] if isinstance(item[1], (list, tuple)) else [item[1]]
            document = self.store.add_node(
                task_id=self.task_id,
                branch_id=self.branch_id,
                logical_id=f"document_source_{document_index + 1}",
                node_type="document_source",
                content=title,
                owner="dataset",
                source_refs=(query_node_id,),
                provenance={
                    "origin": NodeOrigin.SOURCE,
                    "visibility_class": VisibilityClass.PUBLIC,
                    "communication_scope": "MANDATORY",
                    "source_kind": "document",
                    "document_title": title,
                    "document_index": document_index,
                },
                created_by_role="dataset",
                run_id=self.run_id,
            )
            self.store.add_edge(
                source=query_node_id,
                target=document.node_id,
                relation="derived_from",
                created_by_role="dataset",
                run_id=self.run_id,
            )
            document_ids.append(document.node_id)
            previous_sentence_id: str | None = None
            for sentence_index, sentence in enumerate(sentences):
                sentence_node = self.store.add_node(
                    task_id=self.task_id,
                    branch_id=self.branch_id,
                    logical_id=f"sentence_source_{document_index + 1}_{sentence_index + 1}",
                    node_type="sentence_source",
                    content=str(sentence),
                    owner="dataset",
                    source_refs=(document.node_id,),
                    provenance={
                        "origin": NodeOrigin.SOURCE,
                        "visibility_class": VisibilityClass.PUBLIC,
                        "communication_scope": "MANDATORY",
                        "source_kind": "sentence",
                        "document_title": title,
                        "document_index": document_index,
                        "sentence_index": sentence_index,
                    },
                    created_by_role="dataset",
                    run_id=self.run_id,
                )
                self.store.add_edge(
                    source=document.node_id,
                    target=sentence_node.node_id,
                    relation="derived_from",
                    created_by_role="dataset",
                    run_id=self.run_id,
                )
                if previous_sentence_id is not None:
                    self.store.add_edge(
                        source=previous_sentence_id,
                        target=sentence_node.node_id,
                        relation="next_sentence",
                        created_by_role="dataset",
                        run_id=self.run_id,
                    )
                previous_sentence_id = sentence_node.node_id
                sentence_ids.append(sentence_node.node_id)
        self._hotpot_source_node_ids = (question_node.node_id, *document_ids, *sentence_ids)
        self._hotpot_document_node_ids = tuple(document_ids)
        self._hotpot_sentence_node_ids = tuple(sentence_ids)
        if self.context_policy in {"semantic_minimal", "semantic_shadow"}:
            semantic_started = time.perf_counter()
            self._build_hotpot_semantic_graph(query_node_id)
            self.telemetry.record_aflow_metric(
                aflow_semantic_graph_build_ms=int((time.perf_counter() - semantic_started) * 1000.0)
            )

    def _build_hotpot_semantic_graph(self, query_node_id: str) -> None:
        state = self.store.snapshot()
        documents = [{"node_id": nid, "title": str(state.nodes[nid].content), "index": int(state.nodes[nid].provenance.get("document_index", 0))} for nid in self._hotpot_document_node_ids]
        sentences = [{
            "node_id": nid, "text": str(state.nodes[nid].content),
            "document_id": state.nodes[nid].source_refs[0],
            "document_index": int(state.nodes[nid].provenance.get("document_index", 0)),
            "sentence_index": int(state.nodes[nid].provenance.get("sentence_index", 0)),
        } for nid in self._hotpot_sentence_node_ids]
        self._semantic_build = build_semantic_graph(str(self.task_metadata.get("question") or self.task), documents, sentences)
        for artifact in self._semantic_build.entities:
            title_refs = tuple(doc_id for doc_id, entity_id in self._semantic_build.title_entities.items() if entity_id == artifact.entity_id)
            source_refs = tuple(dict.fromkeys((*title_refs, *artifact.source_sentence_ids)))
            node = self.store.add_node(
                task_id=self.task_id, branch_id=self.branch_id,
                logical_id=artifact.entity_id, node_type="entity_source",
                content={"canonical_name": artifact.canonical_name, "surface_forms": list(artifact.surface_forms)},
                owner="dataset", source_refs=source_refs,
                provenance={
                    "origin": NodeOrigin.SOURCE, "visibility_class": VisibilityClass.PUBLIC,
                    "communication_scope": "OPTIONAL", "source_kind": "entity",
                    "entity_kind": artifact.entity_kind, "document_frequency": artifact.document_frequency,
                    "generic": artifact.generic, "entity_policy": self.semantic_entity_policy,
                }, created_by_role="dataset", run_id=self.run_id,
            )
            self._semantic_entity_node_ids[artifact.entity_id] = node.node_id
        for entity_id in self._semantic_build.query_entity_ids:
            if entity_id in self._semantic_entity_node_ids:
                self.store.add_edge(source=query_node_id, target=self._semantic_entity_node_ids[entity_id], relation="query_mentions_entity", created_by_role="dataset", run_id=self.run_id)
        for document_id, entity_id in self._semantic_build.title_entities.items():
            if entity_id in self._semantic_entity_node_ids:
                self.store.add_edge(source=document_id, target=self._semantic_entity_node_ids[entity_id], relation="title_identifies_entity", created_by_role="dataset", run_id=self.run_id)
        for sentence_id, entity_ids in self._semantic_build.sentence_mentions.items():
            mapped = [self._semantic_entity_node_ids[eid] for eid in entity_ids if eid in self._semantic_entity_node_ids]
            for entity_node_id in mapped:
                self.store.add_edge(source=sentence_id, target=entity_node_id, relation="sentence_mentions_entity", created_by_role="dataset", run_id=self.run_id)
            for index, left in enumerate(mapped):
                for right in mapped[index + 1:]:
                    self.store.add_edge(source=left, target=right, relation="entity_co_occurs", created_by_role="dataset", run_id=self.run_id)
        claim_started = time.perf_counter()
        self._claim_build = extract_claims(str(self.task_metadata.get("question") or self.task), self._semantic_build, sentences)
        self.telemetry.record_aflow_metric(aflow_claim_extraction_ms=int((time.perf_counter() - claim_started) * 1000.0))
        subgoal_nodes: dict[str, str] = {}
        relation_nodes: dict[str, str] = {}
        for subgoal in self._claim_build.query_spec.subgoals:
            subgoal_node = self.store.add_node(
                task_id=self.task_id, branch_id=self.branch_id, logical_id=subgoal.subgoal_id,
                node_type="query_subgoal", content={"subject_ref": subgoal.subject_ref, "relation": subgoal.relation, "object_ref": subgoal.object_ref},
                owner="dataset", source_refs=(query_node_id,),
                provenance={"origin": NodeOrigin.SOURCE, "visibility_class": VisibilityClass.PUBLIC, "communication_scope": "OPTIONAL", "source_kind": "query_subgoal"},
                created_by_role="dataset", run_id=self.run_id,
            )
            subgoal_nodes[subgoal.subgoal_id] = subgoal_node.node_id
            self.store.add_edge(source=query_node_id, target=subgoal_node.node_id, relation="query_requires_subgoal", created_by_role="dataset", run_id=self.run_id)
            relation_node = relation_nodes.get(subgoal.relation)
            if relation_node is None:
                relation_node = self.store.add_node(
                    task_id=self.task_id, branch_id=self.branch_id, logical_id=f"relation_{subgoal.relation}", node_type="relation_source", content=subgoal.relation,
                    owner="dataset", source_refs=(query_node_id,), provenance={"origin": NodeOrigin.SOURCE, "visibility_class": VisibilityClass.PUBLIC, "communication_scope": "OPTIONAL", "source_kind": "relation"}, created_by_role="dataset", run_id=self.run_id,
                ).node_id
                relation_nodes[subgoal.relation] = relation_node
            self.store.add_edge(source=subgoal_node.node_id, target=relation_node, relation="subgoal_requires_relation", created_by_role="dataset", run_id=self.run_id)
        answer_node = self.store.add_node(
            task_id=self.task_id, branch_id=self.branch_id, logical_id="answer_constraint", node_type="answer_constraint", content={"answer_type": self._claim_build.query_spec.answer_type}, owner="dataset", source_refs=(query_node_id,), provenance={"origin": NodeOrigin.SOURCE, "visibility_class": VisibilityClass.PUBLIC, "communication_scope": "OPTIONAL", "source_kind": "answer_constraint"}, created_by_role="dataset", run_id=self.run_id,
        )
        self.store.add_edge(source=query_node_id, target=answer_node.node_id, relation="query_requires_subgoal", created_by_role="dataset", run_id=self.run_id)
        for claim in self._claim_build.claims:
            node = self.store.add_node(
                task_id=self.task_id, branch_id=self.branch_id,
                logical_id=claim.claim_id, node_type="claim_source",
                content={"relation": claim.relation, "object_value": claim.object_value},
                owner="dataset", source_refs=(claim.evidence_sentence_id,),
                provenance={"origin": NodeOrigin.SOURCE, "visibility_class": VisibilityClass.PUBLIC, "communication_scope": "OPTIONAL", "source_kind": "claim", "confidence": claim.confidence, "extractor_version": claim.extractor_version},
                created_by_role="dataset", run_id=self.run_id,
            )
            self._claim_node_ids[claim.claim_id] = node.node_id
            self.store.add_edge(source=claim.evidence_sentence_id, target=node.node_id, relation="sentence_supports_claim", created_by_role="dataset", run_id=self.run_id)
            for entity_id, relation in ((claim.subject_entity_id, "claim_subject"), (claim.object_entity_id, "claim_object")):
                if entity_id and entity_id in self._semantic_entity_node_ids:
                    self.store.add_edge(source=node.node_id, target=self._semantic_entity_node_ids[entity_id], relation=relation, created_by_role="dataset", run_id=self.run_id)
            for subgoal in self._claim_build.query_spec.subgoals:
                if subgoal.relation == claim.relation and subgoal.subgoal_id in subgoal_nodes:
                    self.store.add_edge(source=node.node_id, target=subgoal_nodes[subgoal.subgoal_id], relation="claim_answers_subgoal", created_by_role="dataset", run_id=self.run_id)

    def hotpot_source_graph(self) -> dict[str, tuple[str, ...]]:
        return {
            "question": tuple(
                node_id for node_id in self._hotpot_source_node_ids
                if self.store.node(node_id).type == "question_source"
            ),
            "documents": self._hotpot_document_node_ids,
            "sentences": self._hotpot_sentence_node_ids,
        }

    def _hotpot_tokens(self, value: Any) -> set[str]:
        return {item.casefold() for item in re.findall(r"[A-Za-z0-9]+", _text_value(value))}

    def _hotpot_entities(self, value: Any) -> set[str]:
        stop_entities = {
            "what", "where", "which", "who", "when", "why", "how",
            "does", "did", "is", "are", "was", "were", "the", "this",
        }
        return {
            item.casefold()
            for item in re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", _text_value(value))
            if item.casefold() not in stop_entities
        }

    def _hotpot_valid_question_entities(self, value: Any) -> set[str]:
        entities = self._hotpot_entities(value)
        tokens = self._hotpot_tokens(value)
        # A capitalized token is useful only when it is not a sentence-initial
        # function word and has more than one occurrence or appears in source.
        return {entity for entity in entities if entity in tokens and len(entity) >= 3}

    def _hotpot_select_source_nodes(self, policy: str, top_k_documents: int | None = None) -> HotpotSelectionResult:
        """Return (roots, selected nodes, closure/document nodes) deterministically."""
        policy = "topk_documents" if policy == "flat_topk_documents" else policy
        top_k_documents = max(1, int(top_k_documents or self.context_top_k_documents))
        state = self.store.snapshot()
        question_tokens = self._hotpot_tokens(self.task_metadata.get("question") or self.task)
        question_entities = self._hotpot_valid_question_entities(self.task_metadata.get("question") or self.task)
        documents = [state.nodes[node_id] for node_id in self._hotpot_document_node_ids]
        sentences = [state.nodes[node_id] for node_id in self._hotpot_sentence_node_ids]
        by_document: dict[int, list[Any]] = defaultdict(list)
        for sentence in sentences:
            by_document[int(sentence.provenance.get("document_index", 0))].append(sentence)

        document_frequency: dict[str, int] = defaultdict(int)
        for sentence in sentences:
            for token in self._hotpot_tokens(sentence.content):
                document_frequency[token] += 1
        sentence_count = max(1, len(sentences))

        def sentence_score(sentence: Any, document: Any) -> float:
            sentence_tokens = self._hotpot_tokens(sentence.content)
            overlap = len(question_tokens & sentence_tokens)
            entity_overlap = len(question_entities & self._hotpot_entities(sentence.content))
            tfidf = sum(
                1.0 + (sentence_count / max(1, document_frequency[token]))
                for token in question_tokens & sentence_tokens
            )
            title_overlap = len(question_tokens & self._hotpot_tokens(document.content))
            token_penalty = 0.001 * self._count_tokens(sentence.content)
            return overlap * 2.0 + entity_overlap * 3.0 + tfidf + title_overlap - token_penalty

        scored: list[tuple[float, int, Any]] = []
        for document in documents:
            index = int(document.provenance.get("document_index", 0))
            doc_sentences = by_document.get(index, [])
            best = max((sentence_score(sentence, document) for sentence in doc_sentences), default=0.0)
            scored.append((best, index, document))
        if policy == "full":
            selected_documents = documents
        else:
            selected_documents = [
                item[2]
                for item in sorted(scored, key=lambda item: (-item[0], item[1]))[: top_k_documents]
            ]
            selected_documents.sort(key=lambda node: int(node.provenance.get("document_index", 0)))

        selected_document_indexes = {
            int(node.provenance.get("document_index", 0)) for node in selected_documents
        }
        selected_sentences: list[Any] = []
        for document in selected_documents:
            index = int(document.provenance.get("document_index", 0))
            candidates = by_document.get(index, [])
            if policy == "full":
                keep = candidates
            else:
                ranked = sorted(
                    candidates,
                    key=lambda sentence: (
                        -sentence_score(sentence, document),
                        int(sentence.provenance.get("sentence_index", 0)),
                    ),
                )
                keep = ranked[: (len(ranked) if policy == "topk_documents" else min(2, len(ranked)))]
                if policy == "dependency_minimal" and keep:
                    keep_indexes = {
                        int(sentence.provenance.get("sentence_index", 0)) for sentence in keep
                    }
                    for sentence in candidates:
                        sentence_index = int(sentence.provenance.get("sentence_index", 0))
                        if any(abs(sentence_index - selected) == 1 for selected in keep_indexes):
                            keep.append(sentence)
                    keep = list({sentence.node_id: sentence for sentence in keep}.values())
            selected_sentences.extend(keep)
        selected_sentences.sort(
            key=lambda node: (
                int(node.provenance.get("document_index", 0)),
                int(node.provenance.get("sentence_index", 0)),
            )
        )
        roots = tuple(node.node_id for node in selected_sentences)
        selected = tuple(node.node_id for node in selected_sentences)
        closure = tuple(
            node.node_id for node in selected_documents
            if node.node_id not in selected
        )
        if not selected_sentences and sentences:
            selected = (sentences[0].node_id,)
            roots = selected
        ranked_docs = sorted(scored, key=lambda item: (-item[0], item[1]))
        boundary_margin = 1.0
        if len(ranked_docs) > top_k_documents and ranked_docs[top_k_documents - 1][0] != 0:
            boundary_margin = abs(ranked_docs[top_k_documents - 1][0] - ranked_docs[top_k_documents][0]) / max(abs(ranked_docs[top_k_documents - 1][0]), 1e-9)
        selected_set = set(selected)
        selected_doc_set = {node.node_id for node in selected_documents}
        covered_entities = set()
        sentence_audit = []
        for rank, sentence in enumerate(sorted(sentences, key=lambda node: (-sentence_score(node, next((d for d in documents if int(d.provenance.get("document_index", 0)) == int(node.provenance.get("document_index", 0))), documents[0] if documents else node)), node.node_id)), 1):
            document = next((d for d in documents if int(d.provenance.get("document_index", 0)) == int(sentence.provenance.get("document_index", 0))), sentence)
            score = sentence_score(sentence, document)
            if sentence.node_id in selected_set:
                covered_entities.update(self._hotpot_entities(sentence.content))
            sentence_audit.append({"node_id": sentence.node_id, "score": round(score, 6), "rank": rank})
        shared_entities = set()
        bridge_paths: list[dict[str, Any]] = []
        selected_docs = [d for d in documents if d.node_id in selected_doc_set]
        selected_sentences_by_doc: dict[int, list[Any]] = defaultdict(list)
        for sentence in selected_sentences:
            selected_sentences_by_doc[int(sentence.provenance.get("document_index", 0))].append(sentence)
        for pos, left in enumerate(selected_docs):
            for right in selected_docs[pos + 1:]:
                left_index = int(left.provenance.get("document_index", 0))
                right_index = int(right.provenance.get("document_index", 0))
                left_text = " ".join(str(s.content) for s in selected_sentences_by_doc.get(left_index, ()))
                right_text = " ".join(str(s.content) for s in selected_sentences_by_doc.get(right_index, ()))
                entities = self._hotpot_entities(left_text) & self._hotpot_entities(right_text)
                title_left = self._hotpot_tokens(left.content)
                title_right = self._hotpot_tokens(right.content)
                if title_right and title_right <= self._hotpot_tokens(left_text):
                    entities.update(title_right)
                if title_left and title_left <= self._hotpot_tokens(right_text):
                    entities.update(title_left)
                shared_entities.update(entities)
                for entity in sorted(entities):
                    bridge_paths.append({
                        "source_document": left.node_id,
                        "entity": entity,
                        "target_document": right.node_id,
                        "evidence_sentence_ids": [
                            s.node_id for s in selected_sentences_by_doc.get(left_index, ())
                            if entity in self._hotpot_entities(s.content) or entity in self._hotpot_tokens(s.content)
                        ],
                    })
        document_audit = tuple({"node_id": node.node_id, "title": str(node.content), "score": round(score, 6), "rank": rank} for rank, (score, _index, node) in enumerate(ranked_docs, 1))
        return HotpotSelectionResult(
            root_node_ids=roots, selected_node_ids=selected, closure_node_ids=closure,
            document_scores=document_audit, sentence_scores=tuple(sentence_audit),
            question_entities=tuple(sorted(question_entities)),
            covered_entities=tuple(sorted(covered_entities & set(question_entities))),
            shared_entities=tuple(sorted(shared_entities)), boundary_margin=float(boundary_margin),
            bridge_paths=tuple(bridge_paths),
            selected_document_ids=tuple(node.node_id for node in selected_documents),
            selected_sentence_ids=selected,
        )

    def _hotpot_context_risk(self, selection: HotpotSelectionResult, *, original_tokens: int = 0, selected_tokens: int = 0, level: str = "L0") -> ContextRisk:
        state = self.store.snapshot()
        selected_docs = {state.nodes[node_id].provenance.get("document_index") for node_id in selection.selected_node_ids if node_id in state.nodes}
        selected_docs.update(state.nodes[node_id].provenance.get("document_index") for node_id in selection.closure_node_ids if node_id in state.nodes)
        critical: list[str] = []
        soft: list[str] = []
        if len(selected_docs) < 2: critical.append("fewer_than_two_documents")
        coverage = len(set(selection.question_entities) & set(selection.covered_entities)) / max(1, len(selection.question_entities))
        if selection.question_entities and coverage == 0: critical.append("question_entities_uncovered")
        elif selection.question_entities and coverage < 0.5: soft.append("low_entity_coverage")
        if len(selection.selected_node_ids) < 2: critical.append("fewer_than_two_sentences")
        if len(selected_docs) >= 2 and not selection.bridge_paths:
            (critical if level == "L1" else soft).append("no_bridge_path")
        if 0 <= selection.boundary_margin < 0.10: soft.append("unstable_topk_boundary")
        if original_tokens and selected_tokens < original_tokens * 0.10: soft.append("context_too_short")
        if len(selected_docs) == 1: soft.append("single_evidence_document")
        selected_text = " ".join(str(state.nodes[node_id].content) for node_id in selection.selected_node_ids if node_id in state.nodes)
        if selection.question_entities and not (set(selection.question_entities) & self._hotpot_entities(selected_text)):
            critical.append("question_entity_missing_from_sentences")
        reasons = tuple(critical + soft)
        soft_limit = 2 if level == "L0" else 3
        decision = "escalate" if critical or len(soft) >= soft_limit else ("accept_l1" if level == "L1" else "accept_l0")
        return ContextRisk(
            score=min(1.0, (len(critical) * 0.5 + len(soft) * 0.2)),
            reasons=reasons,
            critical_reasons=tuple(critical),
            soft_reasons=tuple(soft),
            decision=decision,
        )

    def render_hotpot_context(self, node_ids: Iterable[str]) -> str:
        state = self.store.snapshot()
        selected = {
            str(node_id): state.nodes[str(node_id)]
            for node_id in node_ids
            if str(node_id) in state.nodes
        }
        by_document: dict[int, list[Any]] = defaultdict(list)
        for node in selected.values():
            if node.type == "sentence_source":
                by_document[int(node.provenance.get("document_index", 0))].append(node)
        parts: list[str] = []
        for index in sorted(by_document):
            sentences = sorted(
                by_document[index],
                key=lambda node: int(node.provenance.get("sentence_index", 0)),
            )
            parts.append(" ".join(str(node.content) for node in sentences))
        context = "\n".join(parts)
        question = str(self.task_metadata.get("question") or self.task)
        return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

    def _record_context_plan(self, plan: AFlowContextPlan) -> None:
        self._context_plans.append(plan)
        self.telemetry.record_aflow_metric(
            aflow_context_original_input_tokens=plan.original_input_tokens,
            aflow_context_selected_input_tokens=plan.effective_input_tokens,
            aflow_context_saved_input_tokens=max(0, plan.original_input_tokens - plan.effective_input_tokens),
            aflow_context_selected_node_count=len(plan.selected_node_ids),
            aflow_context_closure_node_count=len(plan.closure_node_ids),
            aflow_context_contract_satisfied=int(plan.contract_satisfied),
            aflow_context_fallback_count=int(plan.fallback),
            aflow_context_selection_overhead_ms=int(plan.selection_overhead_ms),
            aflow_context_level_usage_L0=int(plan.effective_level == "L0"),
            aflow_context_level_usage_L1=int(plan.effective_level == "L1"),
            aflow_context_level_usage_L2=int(plan.effective_level == "L2"),
            aflow_context_risk_score=int(round(plan.risk_score * 1000)),
            aflow_context_L0_sample_count=int(plan.effective_level == "L0"),
            aflow_context_L1_sample_count=int(plan.effective_level == "L1"),
            aflow_context_L2_sample_count=int(plan.effective_level == "L2"),
            aflow_context_adaptive_recovery_count=int(len(plan.attempted_levels) > 1),
            aflow_context_safety_fallback_count=int(plan.fallback),
            aflow_context_plan_cache_hits=int(sum(bool(item.get("cache_hit")) for item in plan.level_attempts)),
            aflow_context_unique_selection_runs=int(len(plan.level_attempts) - sum(bool(item.get("cache_hit")) for item in plan.level_attempts)),
        )
        self.telemetry.record_aflow_context_plan(plan.as_dict())
        self.telemetry.emit("aflow_context_plan", plan.as_dict())

    def _prepare_hotpot_source_input(
        self,
        *,
        operator: str,
        kwargs: Mapping[str, Any],
    ) -> AFlowOperatorInput:
        started = time.perf_counter()
        original = dict(kwargs)
        original_text = str(original.get("input", ""))
        original_tokens = self._count_tokens(original_text)
        levels = [("L0", self.context_policy, self.context_top_k_documents)]
        if self.context_recovery == "adaptive" and self.context_policy in {"dependency_minimal", "topk_documents", "flat_topk_documents"}:
            levels = [("L0", "dependency_minimal", self.context_top_k_documents), ("L1", "dependency_minimal", self.context_recovery_top_k_documents), ("L2", "full", self.context_recovery_top_k_documents)]
        attempts, tokens_by_level, risk_by_level = [], {}, {}
        selector_failed = False
        failure_reason = "selector_exception"
        chosen = None
        last_reason = ""
        level_attempts: list[dict[str, Any]] = []
        for level, policy, top_k in levels:
            cache_key = (tuple(self._hotpot_source_node_ids), original_text, policy, int(top_k), self.context_risk_policy)
            try:
                selection = self._context_plan_cache.get(cache_key)
                cache_hit = selection is not None
                if selection is None:
                    selection = self._hotpot_select_source_nodes(policy=policy, top_k_documents=top_k)
                    self._context_plan_cache[cache_key] = selection
                    self._context_unique_selection_runs += 1
                else:
                    self._context_plan_cache_hits += 1
            except Exception:
                selector_failed = True
                failure_reason = "selector_exception"
                last_reason = failure_reason
                break
            try:
                rendered = original_text if level == "L2" else self.render_hotpot_context(selection.selected_node_ids)
            except Exception:
                selector_failed = True
                failure_reason = "renderer_exception"
                last_reason = failure_reason
                break
            selected_tokens = self._count_tokens(rendered)
            risk = self._hotpot_context_risk(selection, original_tokens=original_tokens, selected_tokens=selected_tokens, level=level)
            attempts.append(level); tokens_by_level[level] = selected_tokens; risk_by_level[level] = risk.score
            contract = bool(self.hotpot_source_graph()["question"] and any(self.store.node(node_id).type == "sentence_source" for node_id in selection.selected_node_ids))
            reason = ""
            if self.context_budget_tokens and selected_tokens > self.context_budget_tokens: reason = "context_budget_exceeded"
            elif not contract: reason = "missing_question_or_evidence"
            elif selected_tokens > original_tokens: reason = "compressed_context_not_shorter"
            elif policy == "full" and rendered != original_text: reason = "full_context_roundtrip_mismatch"
            if reason:
                contract = False
            accepted = contract and (
                self.context_recovery != "adaptive"
                or level == "L2"
                or risk.decision in {"accept_l0", "accept_l1"}
            )
            level_attempts.append({
                "level": level,
                "policy": policy,
                "top_k_documents": top_k,
                "selected_document_ids": list(selection.selected_document_ids),
                "selected_sentence_ids": list(selection.selected_sentence_ids),
                "input_tokens": selected_tokens,
                "critical_reasons": list(risk.critical_reasons),
                "soft_reasons": list(risk.soft_reasons),
                "risk_score": risk.score,
                "accepted": accepted,
                "cache_hit": cache_hit,
            })
            if accepted:
                chosen = (level, policy, selection, rendered, selected_tokens, risk, reason, contract)
                break
            last_reason = reason or ";".join(risk.reasons) or "structural_risk"
        if chosen is None and selector_failed:
            level, policy, selection, rendered, selected_tokens, risk, reason, contract = ("L2", "full", HotpotSelectionResult((), (), ()), original_text, original_tokens, ContextRisk(1.0, (failure_reason,), (failure_reason,), (), "escalate"), failure_reason, False)
        elif chosen is None:
            selection = self._hotpot_select_source_nodes(policy="full")
            level, policy, selection, rendered, selected_tokens, risk, reason, contract = ("L2", "full", selection, original_text, original_tokens, ContextRisk(0.0, (), (), (), "accept_l1"), "adaptive_recovery_exhausted", False)
            last_reason = last_reason or reason
        else:
            level, policy, selection, rendered, selected_tokens, risk, reason, contract = chosen
        fallback = not contract
        effective = original_tokens if fallback else selected_tokens
        if fallback:
            last_reason = last_reason or reason or "context_contract_failed"
        plan = AFlowContextPlan(
            operator=str(operator),
            policy=policy,
            requested_root_ids=selection.root_node_ids,
            selected_node_ids=selection.selected_node_ids,
            closure_node_ids=selection.closure_node_ids,
            original_input_tokens=original_tokens,
            effective_input_tokens=effective,
            contract_satisfied=contract,
            fallback=fallback,
            fallback_reason=last_reason,
            selection_overhead_ms=(time.perf_counter() - started) * 1000.0,
            level=level, top_k_documents=self.context_top_k_documents,
            document_scores=selection.document_scores, sentence_scores=selection.sentence_scores,
            question_entities=selection.question_entities, covered_entities=selection.covered_entities,
            shared_entities=selection.shared_entities, boundary_margin=selection.boundary_margin,
            risk_reasons=risk.reasons, risk_score=risk.score,
            critical_reasons=risk.critical_reasons, soft_reasons=risk.soft_reasons,
            decision=risk.decision, sample_id=self.task_id,
            level_attempts=tuple(level_attempts), bridge_paths=selection.bridge_paths,
            cache_hit=bool(level_attempts and all(bool(item.get("cache_hit")) for item in level_attempts)),
            attempted_levels=tuple(attempts), effective_level=level,
            tokens_by_level=tokens_by_level, risk_by_level=risk_by_level,
        )
        self._record_context_plan(plan)
        if fallback:
            return AFlowOperatorInput(kwargs=original)
        bound = dict(original)
        bound["input"] = rendered
        return AFlowOperatorInput(kwargs=bound, source_node_ids=selection.selected_node_ids)

    def _prepare_hotpot_claim_input(self, *, operator: str, kwargs: Mapping[str, Any]) -> AFlowOperatorInput:
        started = time.perf_counter()
        original = dict(kwargs); original_text = str(original.get("input", "")); original_tokens = self._count_tokens(original_text)
        state = self.store.snapshot()
        sentences = [{"node_id": nid, "text": str(state.nodes[nid].content), "document_id": state.nodes[nid].source_refs[0], "document_index": int(state.nodes[nid].provenance.get("document_index", 0)), "sentence_index": int(state.nodes[nid].provenance.get("sentence_index", 0))} for nid in self._hotpot_sentence_node_ids]
        cache_key = (tuple(self._hotpot_source_node_ids), self.semantic_budget_tokens, "deterministic_claim_v1")
        proof = self._claim_plan_cache.get(cache_key)
        cache_hit = proof is not None
        if proof is None and self._claim_build is not None:
            proof = select_claim_proof(self._claim_build, sentences, token_counter=self._count_tokens, budget_tokens=self.semantic_budget_tokens)
            self._claim_plan_cache[cache_key] = proof
        k4 = self._hotpot_select_source_nodes("dependency_minimal", self.context_top_k_documents)
        k4_text = self.render_hotpot_context(k4.selected_node_ids); k4_tokens = self._count_tokens(k4_text)
        proof_text = self.render_hotpot_context(proof.selected_sentence_ids) if proof else ""
        proof_tokens = self._count_tokens(proof_text) if proof_text else 0
        s0_ok = bool(proof and proof.contract_satisfied and proof_tokens < k4_tokens and proof_tokens <= self.semantic_budget_tokens)
        attempts = [{"level": "S0", "policy": "claim_proof", "input_tokens": proof_tokens, "selected_sentence_ids": list(proof.selected_sentence_ids if proof else ()), "selected_document_ids": list(proof.selected_document_ids if proof else ()), "relation_coverage": proof.relation_coverage if proof else 0.0, "candidate_proof_count": proof.candidate_proof_count if proof else 0, "missing_requirements": list(proof.missing_requirements if proof else ("claim_graph_unavailable",)), "accepted": s0_ok, "cache_hit": cache_hit}]
        level = "S0"; rendered = proof_text; source_ids = proof.selected_sentence_ids if proof else (); reason = ""; contract = s0_ok
        if not s0_ok:
            reason = ";".join(proof.missing_requirements if proof else ("claim_graph_unavailable",)) or "claim_proof_not_shorter_than_k4"
            if self.context_recovery == "semantic_adaptive":
                k4_ok = bool(k4.selected_node_ids and k4_tokens <= original_tokens)
                attempts.append({"level": "S1", "policy": "dependency_minimal", "input_tokens": k4_tokens, "selected_sentence_ids": list(k4.selected_node_ids), "selected_document_ids": list(k4.selected_document_ids), "accepted": k4_ok})
                if k4_ok:
                    level, rendered, source_ids, contract = "S1", k4_text, k4.selected_node_ids, True
                else:
                    level, rendered, source_ids, contract = "S2", original_text, (), True
                    attempts.append({"level": "S2", "policy": "full", "input_tokens": original_tokens, "accepted": True})
            else:
                level, rendered, source_ids, contract = "S2", original_text, (), False
        effective_tokens = self._count_tokens(rendered)
        plan = AFlowContextPlan(
            operator=str(operator), policy="claim_proof", requested_root_ids=tuple(source_ids), selected_node_ids=tuple(source_ids),
            closure_node_ids=tuple(k4.closure_node_ids if level == "S1" else ()), original_input_tokens=original_tokens,
            effective_input_tokens=effective_tokens, contract_satisfied=contract, fallback=level == "S2", fallback_reason=reason,
            selection_overhead_ms=(time.perf_counter() - started) * 1000.0, level=level, effective_level=level,
            attempted_levels=tuple(a["level"] for a in attempts), level_attempts=tuple(attempts), sample_id=self.task_id,
            cache_hit=cache_hit, tokens_by_level={a["level"]: int(a["input_tokens"]) for a in attempts},
        )
        self._record_context_plan(plan)
        self.telemetry.record_aflow_metric(
            aflow_query_subgoal_count=len(self._claim_build.query_spec.subgoals) if self._claim_build else 0,
            aflow_required_relation_count=len(self._claim_build.query_spec.required_relations) if self._claim_build else 0,
            aflow_claim_count=len(self._claim_build.claims) if self._claim_build else 0,
            aflow_grounded_claim_count=len(self._claim_build.claims) if self._claim_build else 0,
            aflow_rejected_claim_count=len(self._claim_build.rejected_claims) if self._claim_build else 0,
            aflow_candidate_proof_count=proof.candidate_proof_count if proof else 0,
            aflow_complete_proof_count=int(bool(proof and proof.contract_satisfied)),
            aflow_S0_eligible_count=int(s0_ok),
            aflow_S0_count=int(level == "S0"), aflow_S1_count=int(level == "S1"), aflow_S2_count=int(level == "S2"),
            aflow_claim_extraction_ms=0, aflow_proof_selection_ms=int((time.perf_counter() - started) * 1000.0),
        )
        if reason: self.telemetry.record_aflow_semantic_fallback(reason)
        if level == "S2": return AFlowOperatorInput(kwargs=original)
        bound = dict(original); bound["input"] = rendered
        return AFlowOperatorInput(kwargs=bound, source_node_ids=tuple(source_ids))

    def _prepare_hotpot_semantic_input(self, *, operator: str, kwargs: Mapping[str, Any]) -> AFlowOperatorInput:
        started = time.perf_counter()
        original = dict(kwargs); original_text = str(original.get("input", "")); original_tokens = self._count_tokens(original_text)
        state = self.store.snapshot()
        documents = [{"node_id": nid, "title": str(state.nodes[nid].content), "index": int(state.nodes[nid].provenance.get("document_index", 0))} for nid in self._hotpot_document_node_ids]
        sentences = [{"node_id": nid, "text": str(state.nodes[nid].content), "document_id": state.nodes[nid].source_refs[0], "document_index": int(state.nodes[nid].provenance.get("document_index", 0)), "sentence_index": int(state.nodes[nid].provenance.get("sentence_index", 0))} for nid in self._hotpot_sentence_node_ids]
        cache_key = (tuple(self._hotpot_source_node_ids), self.semantic_entity_policy, self.semantic_max_paths, self.semantic_max_document_hops, self.semantic_budget_tokens)
        semantic = self._semantic_plan_cache.get(cache_key)
        cache_hit = semantic is not None
        if semantic is None and self._semantic_build is not None:
            semantic = select_semantic_paths(self._semantic_build, documents, sentences, token_counter=self._count_tokens, max_paths=self.semantic_max_paths, max_document_hops=self.semantic_max_document_hops, budget_tokens=self.semantic_budget_tokens)
            self._semantic_plan_cache[cache_key] = semantic
        k4 = self._hotpot_select_source_nodes("dependency_minimal", self.context_top_k_documents)
        k4_rendered = self.render_hotpot_context(k4.selected_node_ids); k4_tokens = self._count_tokens(k4_rendered)
        attempts: list[dict[str, Any]] = []
        semantic_rendered = self.render_hotpot_context(semantic.selected_sentence_ids) if semantic else ""
        semantic_tokens = self._count_tokens(semantic_rendered) if semantic_rendered else 0
        semantic_ok = bool(semantic and semantic.contract_satisfied and semantic_tokens < k4_tokens and semantic_tokens <= self.semantic_budget_tokens)
        attempts.append({
            "level": "S0", "policy": "semantic_minimal", "input_tokens": semantic_tokens,
            "selected_document_ids": list(semantic.selected_document_ids if semantic else ()),
            "selected_sentence_ids": list(semantic.selected_sentence_ids if semantic else ()),
            "selected_entity_ids": list(semantic.selected_entity_ids if semantic else ()),
            "missing_requirements": list(semantic.missing_requirements if semantic else ("semantic_graph_unavailable",)),
            "selected_paths": [path.__dict__ for path in (semantic.selected_paths if semantic else ())],
            "accepted": semantic_ok, "cache_hit": cache_hit,
        })
        level = "S0"; rendered = semantic_rendered; source_ids = semantic.selected_sentence_ids if semantic else (); contract = semantic_ok; reason = ""
        if not semantic_ok:
            reason = ";".join(semantic.missing_requirements if semantic else ("semantic_graph_unavailable",)) or "semantic_not_shorter_than_k4"
            if self.context_recovery == "semantic_adaptive":
                k4_ok = bool(k4.selected_node_ids and k4_tokens <= original_tokens)
                attempts.append({"level": "S1", "policy": "dependency_minimal", "input_tokens": k4_tokens, "selected_document_ids": list(k4.selected_document_ids), "selected_sentence_ids": list(k4.selected_sentence_ids), "accepted": k4_ok, "cache_hit": False})
                if k4_ok:
                    level = "S1"; rendered = k4_rendered; source_ids = k4.selected_node_ids; contract = True
                else:
                    level = "S2"; rendered = original_text; source_ids = (); contract = True; reason = "k4_contract_failed"
                    attempts.append({"level": "S2", "policy": "full", "input_tokens": original_tokens, "accepted": True, "cache_hit": False})
            else:
                level = "S2"; rendered = original_text; source_ids = (); contract = False
        effective_tokens = self._count_tokens(rendered)
        plan = AFlowContextPlan(
            operator=str(operator), policy="semantic_minimal", requested_root_ids=tuple(source_ids), selected_node_ids=tuple(source_ids),
            closure_node_ids=tuple(semantic.closure_node_ids if semantic and level == "S0" else k4.closure_node_ids if level == "S1" else ()),
            original_input_tokens=original_tokens, effective_input_tokens=effective_tokens,
            contract_satisfied=contract, fallback=level == "S2", fallback_reason=reason,
            selection_overhead_ms=(time.perf_counter() - started) * 1000.0,
            level=level, effective_level=level, attempted_levels=tuple(a["level"] for a in attempts),
            level_attempts=tuple(attempts), sample_id=self.task_id, cache_hit=cache_hit,
            tokens_by_level={a["level"]: int(a["input_tokens"]) for a in attempts},
            question_entities=tuple(semantic.query_entity_ids if semantic else ()),
        )
        self._record_context_plan(plan)
        entity_count = len(self._semantic_build.entities) if self._semantic_build else 0
        mention_count = sum(len(x) for x in self._semantic_build.sentence_mentions.values()) if self._semantic_build else 0
        self.telemetry.record_aflow_metric(
            aflow_semantic_entity_count=entity_count,
            aflow_semantic_entity_mention_count=mention_count,
            aflow_semantic_title_entity_count=len(self._semantic_build.title_entities) if self._semantic_build else 0,
            aflow_semantic_bridge_entity_count=len({eid for p in (semantic.selected_paths if semantic else ()) for eid in p.bridge_entity_ids}),
            aflow_semantic_candidate_path_count=len(semantic.candidate_paths if semantic else ()),
            aflow_semantic_selected_path_count=len(semantic.selected_paths if semantic else ()),
            aflow_semantic_complete_path_count=sum(int(p.complete) for p in (semantic.selected_paths if semantic else ())),
            aflow_semantic_selected_document_count=len(semantic.selected_document_ids if semantic and level == "S0" else k4.selected_document_ids if level == "S1" else ()),
            aflow_semantic_selected_sentence_count=len(source_ids),
            aflow_semantic_entity_coverage=int(round((semantic.entity_coverage if semantic else 0.0) * 1000)),
            aflow_semantic_relation_coverage=int(round((semantic.relation_coverage if semantic else 0.0) * 1000)),
            aflow_semantic_path_selection_ms=int((time.perf_counter() - started) * 1000.0),
            aflow_semantic_neighbor_tokens=sum(self._count_tokens(state.nodes[nid].content) for nid in (semantic.neighbor_sentence_ids if semantic and level == "S0" else ()) if nid in state.nodes),
            aflow_semantic_closure_tokens=sum(self._count_tokens(state.nodes[nid].content) for nid in plan.closure_node_ids if nid in state.nodes),
            aflow_semantic_S0_count=int(level == "S0"), aflow_semantic_S1_count=int(level == "S1"), aflow_semantic_S2_count=int(level == "S2"),
        )
        if reason:
            self.telemetry.record_aflow_semantic_fallback(reason)
        if level == "S2": return AFlowOperatorInput(kwargs=original)
        bound = dict(original); bound["input"] = rendered
        return AFlowOperatorInput(kwargs=bound, source_node_ids=tuple(source_ids))

    def _prepare_hotpot_semantic_shadow(self, *, operator: str, kwargs: Mapping[str, Any]) -> AFlowOperatorInput:
        state = self.store.snapshot()
        documents = [{"node_id": nid, "title": str(state.nodes[nid].content), "index": int(state.nodes[nid].provenance.get("document_index", 0))} for nid in self._hotpot_document_node_ids]
        sentences = [{"node_id": nid, "text": str(state.nodes[nid].content), "document_id": state.nodes[nid].source_refs[0], "document_index": int(state.nodes[nid].provenance.get("document_index", 0)), "sentence_index": int(state.nodes[nid].provenance.get("sentence_index", 0))} for nid in self._hotpot_sentence_node_ids]
        if self._semantic_build is not None:
            semantic = select_semantic_paths(self._semantic_build, documents, sentences, token_counter=self._count_tokens, max_paths=self.semantic_max_paths, max_document_hops=self.semantic_max_document_hops, budget_tokens=self.semantic_budget_tokens)
            self.telemetry.emit("aflow_semantic_shadow", {
                "task_id": self.task_id, "candidate_path_count": len(semantic.candidate_paths),
                "selected_path_count": len(semantic.selected_paths), "contract_satisfied": semantic.contract_satisfied,
                "selected_sentence_ids": list(semantic.selected_sentence_ids), "missing_requirements": list(semantic.missing_requirements),
            })
            self.telemetry.record_aflow_metric(
                aflow_semantic_entity_count=len(self._semantic_build.entities),
                aflow_semantic_candidate_path_count=len(semantic.candidate_paths),
                aflow_semantic_selected_path_count=len(semantic.selected_paths),
            )
        original_policy, original_recovery = self.context_policy, self.context_recovery
        try:
            self.context_policy = "dependency_minimal"; self.context_recovery = "none"
            return self._prepare_hotpot_source_input(operator=operator, kwargs=kwargs)
        finally:
            self.context_policy, self.context_recovery = original_policy, original_recovery

    def next_invocation(self, operator: str) -> int:
        key = str(operator)
        self._invocations[key] = self._invocations.get(key, 0) + 1
        return self._invocations[key]

    def reserve_invocation(self, operator: str) -> ReservedInvocation:
        index = self.next_invocation(operator)
        return ReservedInvocation(
            operator=str(operator), invocation_index=index,
            invocation_id=f"{operator}:{index}", task_id=self.task_id,
        )

    def ensure_candidate_batch(self, invocation: ReservedInvocation) -> CandidateBatch:
        if self._active_candidate_batch_id is None:
            batch_id = f"batch:{self.task_id}:{invocation.invocation_id}"
            self._candidate_batches[batch_id] = CandidateBatch(
                batch_id=batch_id, task_id=self.task_id,
                producer_stage=invocation.operator,
                created_at_invocation=invocation.invocation_id,
                state=CandidateBatchState.OPEN,
            )
            self._active_candidate_batch_id = batch_id
        return self._candidate_batches[self._active_candidate_batch_id]

    def freeze_candidate_batch(self) -> CandidateBatch | None:
        if self._active_candidate_batch_id is None:
            return None
        current = self._candidate_batches[self._active_candidate_batch_id]
        frozen = CandidateBatch(
            batch_id=current.batch_id, task_id=current.task_id,
            producer_stage=current.producer_stage,
            occurrence_ids=tuple(current.occurrence_ids),
            created_at_invocation=current.created_at_invocation,
            frozen=True, state=CandidateBatchState.FROZEN,
            selected_occurrence_ids=tuple(self._selected_occurrence_ids),
        )
        self._candidate_batches[frozen.batch_id] = frozen
        self._last_closed_candidate_batch_id = frozen.batch_id
        self._active_candidate_batch_id = None
        return frozen

    def _register_mbpp_sources(self, kwargs: Mapping[str, Any]) -> dict[str, str]:
        """Register public MBPP fields once per task for exact source binding."""
        if str(self.dataset).casefold() != "mbpp":
            return {}
        values = {
            "problem": kwargs.get("problem", kwargs.get("question", self.task)),
            "entry_point": kwargs.get("entry_point", ""),
            "tests": kwargs.get("tests", kwargs.get("public_tests", "")),
        }
        out = {}
        for name, value in values.items():
            if value in (None, ""):
                continue
            existing = self._mbpp_source_registry
            sid = {
                "problem": existing.problem_source_id if existing else None,
                "entry_point": existing.entry_point_source_id if existing else None,
                "tests": existing.public_test_source_ids[0] if existing and existing.public_test_source_ids else None,
            }.get(name)
            if not sid:
                sid = f"source:mbpp:{self.task_id}:{name}:{stable_digest(value)[:16]}"
                node = self.store.add_node(
                    task_id=self.task_id, branch_id=self.branch_id,
                    logical_id=f"{name}_source", node_type=f"{name}_source",
                    content=value, owner="dataset", source_refs=(),
                    provenance={"origin": NodeOrigin.SOURCE, "visibility_class": VisibilityClass.PUBLIC,
                                 "source_type": name, "content_digest": stable_digest(value)},
                    created_by_role="dataset", run_id=self.run_id,
                )
                sid = node.node_id
                self._source_payloads[sid] = value
                if name == "problem":
                    problem_id = sid
                    entry_id = existing.entry_point_source_id if existing else ""
                    test_ids = existing.public_test_source_ids if existing else ()
                    self._mbpp_source_registry = MBPPSourceRegistry(self.task_id, problem_id, entry_id, test_ids)
                elif name == "entry_point":
                    problem_id = existing.problem_source_id if existing else ""
                    test_ids = existing.public_test_source_ids if existing else ()
                    self._mbpp_source_registry = MBPPSourceRegistry(self.task_id, problem_id, sid, test_ids)
                else:
                    problem_id = existing.problem_source_id if existing else ""
                    entry_id = existing.entry_point_source_id if existing else ""
                    self._mbpp_source_registry = MBPPSourceRegistry(self.task_id, problem_id, entry_id, (sid,))
            out[name] = sid
        return out

    def _required_specs(self, operator: str) -> tuple[RequiredBindingSpec, ...]:
        if self.profile is None:
            return ()
        return tuple(self.profile.receiver_contract(operator).required_bindings or ())

    def validate_required_bindings(
        self, native_plan: NativeInvocationPlan, contract: Any | None = None,
    ) -> RequiredBindingValidation:
        contract = contract or (self.profile.receiver_contract(native_plan.operator) if self.profile else None)
        specs = tuple(getattr(contract, "required_bindings", ()) or ())
        validations: dict[str, BindingValidation] = {}
        valid_count = 0
        for spec in specs:
            binding = native_plan.field_bindings.get(spec.field_name)
            if binding is None:
                if spec.cardinality == "zero_or_one":
                    validations[spec.field_name] = BindingValidation(
                        True, "optional_absent", spec.field_name, 0, 0, True, True, True, ""
                    )
                    valid_count += 1
                    continue
                validations[spec.field_name] = BindingValidation(
                    False, "missing", spec.field_name, 1, 0, False, True, True,
                    "missing_binding_hint",
                )
                continue
            refs = tuple(self._artifact_occurrences.get(x) for x in binding.occurrence_ids)
            refs = tuple(x for x in refs if x is not None)
            source_ids = tuple(binding.source_node_ids)
            if spec.requirement_class == "required_source":
                expected = 0 if spec.cardinality == "zero_or_one" else 1
                ok = bool(source_ids) if expected else True
                reason = "" if ok else "missing_required_source"
                validations[spec.field_name] = BindingValidation(
                    ok, binding.resolution, spec.field_name, expected, len(source_ids), ok,
                    True, True, reason,
                )
                valid_count += int(ok)
                continue
            expected = len(refs) if spec.cardinality == "exact_batch" else 1
            if spec.cardinality == "zero_or_one":
                expected = 0
            task_ok = all(x.task_id == self.task_id for x in refs)
            batches = {x.candidate_batch_id for x in refs if x.candidate_batch_id}
            batch_ok = len(batches) <= 1 and (not native_plan.candidate_batch_id or batches <= {native_plan.candidate_batch_id})
            count_ok = bool(refs) if expected else True
            if spec.cardinality == "exact_batch":
                count_ok = bool(refs) and all(x.candidate_batch_id == native_plan.candidate_batch_id for x in refs)
            ok = count_ok and task_ok and batch_ok and binding.resolution not in {"raw_unresolved", "missing"}
            reason = "" if ok else (
                "cross_task_binding" if not task_ok else
                "cross_batch_binding" if not batch_ok else
                "missing_binding_hint" if not refs else
                "multiplicity_mismatch" if spec.cardinality == "exact_batch" else "content_mismatch"
            )
            validations[spec.field_name] = BindingValidation(
                ok, binding.resolution, spec.field_name, expected, len(refs), ok,
                task_ok, batch_ok, reason,
            )
            valid_count += int(ok)
        total = len(specs)
        return RequiredBindingValidation(
            required_field_count=total,
            valid_required_field_count=valid_count,
            invalid_required_field_count=total - valid_count,
            exact_binding_rate=(valid_count / total) if total else 1.0,
            validations_by_field=validations,
            valid=(valid_count == total),
        )

    def validate_exact_binding(
        self, binding: InvocationInputBinding, original_kwargs: Mapping[str, Any] | None = None,
    ) -> BindingValidation:
        occurrences = self._artifact_occurrences
        refs = tuple(occurrences.get(x) for x in binding.artifact_occurrence_ids)
        refs = tuple(x for x in refs if x is not None)
        task_ok = all(x.task_id == self.task_id for x in refs)
        batch_values = {x.candidate_batch_id for x in refs if x.candidate_batch_id}
        batch_ok = len(batch_values) <= 1 and (not binding.candidate_batch_id or batch_values <= {binding.candidate_batch_id})
        expected = len(binding.artifact_occurrence_ids)
        delivered = len(refs)
        valid = bool(refs) and task_ok and batch_ok and expected == delivered
        reason = "" if valid else (
            "missing_binding_hint" if expected == 0 else
            "cross_task_binding" if not task_ok else
            "cross_batch_binding" if not batch_ok else
            "content_mismatch"
        )
        return BindingValidation(
            valid=valid, resolution=binding.resolution, field_name=binding.field_name,
            expected_count=expected, delivered_count=delivered,
            content_equivalent=valid, task_consistent=task_ok,
            batch_consistent=batch_ok, failure_reason=reason,
        )

    def capture_selected_occurrences(self, occurrence_ids: Iterable[str]) -> None:
        self._selected_occurrence_ids = tuple(str(x) for x in occurrence_ids)
        if self._last_closed_candidate_batch_id:
            batch = self._candidate_batches[self._last_closed_candidate_batch_id]
            self._candidate_batches[self._last_closed_candidate_batch_id] = CandidateBatch(
                batch_id=batch.batch_id, task_id=batch.task_id,
                producer_stage=batch.producer_stage,
                occurrence_ids=batch.occurrence_ids,
                created_at_invocation=batch.created_at_invocation,
                frozen=batch.frozen, selected_occurrence_ids=self._selected_occurrence_ids,
                state=CandidateBatchState.SELECTED,
                ensemble_invocation_id=batch.ensemble_invocation_id,
                test_invocation_ids=batch.test_invocation_ids,
            )

    def mark_batch_selected(self, batch_id: str | None, invocation_id: str) -> None:
        if not batch_id or batch_id not in self._candidate_batches:
            return
        batch = self._candidate_batches[batch_id]
        self._candidate_batches[batch_id] = CandidateBatch(
            batch_id=batch.batch_id, task_id=batch.task_id,
            producer_stage=batch.producer_stage, occurrence_ids=batch.occurrence_ids,
            created_at_invocation=batch.created_at_invocation,
            frozen=True, state=CandidateBatchState.SELECTED,
            selected_occurrence_ids=batch.selected_occurrence_ids,
            ensemble_invocation_id=str(invocation_id),
            test_invocation_ids=batch.test_invocation_ids,
        )

    def archive_candidate_batch(self, batch_id: str | None, invocation_id: str | None = None) -> None:
        if not batch_id or batch_id not in self._candidate_batches:
            return
        batch = self._candidate_batches[batch_id]
        tests = tuple(batch.test_invocation_ids) + ((str(invocation_id),) if invocation_id else ())
        self._candidate_batches[batch_id] = CandidateBatch(
            batch_id=batch.batch_id, task_id=batch.task_id,
            producer_stage=batch.producer_stage, occurrence_ids=batch.occurrence_ids,
            created_at_invocation=batch.created_at_invocation, frozen=True,
            state=CandidateBatchState.ARCHIVED,
            selected_occurrence_ids=batch.selected_occurrence_ids,
            ensemble_invocation_id=batch.ensemble_invocation_id,
            test_invocation_ids=tests,
        )

    def record_test_execution(
        self, invocation_id: str, native_plan: NativeInvocationPlan,
        output: Any,
    ) -> TestExecutionRecord | None:
        """Persist the exact Test→code relation for subsequent Fix calls."""
        if str(native_plan.operator).casefold() != "test":
            return None
        result_ids = tuple(
            oid for oid, occurrence in self._artifact_occurrences.items()
            if occurrence.invocation_id == str(invocation_id)
        )
        code_binding = native_plan.field_bindings.get("solution")
        if not result_ids or code_binding is None or not code_binding.occurrence_ids:
            return None
        passed = False
        if isinstance(output, Mapping):
            passed = bool(output.get("passed", output.get("success", output.get("verified", False))))
        else:
            passed = "passed" in str(output).casefold() and "failed" not in str(output).casefold()
        record = TestExecutionRecord(
            test_invocation_id=str(invocation_id), task_id=self.task_id,
            candidate_batch_id=native_plan.candidate_batch_id or "",
            tested_code_occurrence_ids=tuple(code_binding.occurrence_ids),
            test_result_occurrence_id=result_ids[0], passed=passed,
            error_payload=output if not passed else None,
            problem_source_ids=tuple(native_plan.field_bindings.get("problem", FieldBinding("problem")).source_node_ids),
            entry_point_source_ids=tuple(native_plan.field_bindings.get("entry_point", FieldBinding("entry_point")).source_node_ids),
            public_test_source_ids=tuple(native_plan.field_bindings.get("tests", FieldBinding("tests")).source_node_ids),
        )
        self._test_executions[str(invocation_id)] = record
        self._test_execution_by_batch[record.candidate_batch_id] = record
        if not passed:
            self._latest_failed_test_execution = record
        return record

    def build_native_invocation_plan(
        self, reserved: ReservedInvocation, operator: str, original_kwargs: Mapping[str, Any],
    ) -> NativeInvocationPlan:
        source_registry = self._register_mbpp_sources(original_kwargs)
        batch_id = self._active_candidate_batch_id or self._last_closed_candidate_batch_id
        batch = self._candidate_batches.get(batch_id) if batch_id else None
        required: tuple[str, ...] = ()
        bindings: dict[str, FieldBinding] = {}
        op = str(operator).casefold()
        if op == "scensemble":
            required = ("solutions",)
            ids = tuple(batch.occurrence_ids if batch else ())
            bindings["solutions"] = FieldBinding(
                "solutions", "artifact", ids, (), True, True,
                "direct_runtime" if ids else "raw_unresolved", "required_artifact",
            )
        elif op == "customcodegenerate":
            # Source producers are not required Artifact receivers, but their
            # public problem/signature bindings still create hard provenance
            # edges for downstream closure replay.
            for field, source_name in (("problem", "problem"), ("entry_point", "entry_point")):
                sid = source_registry.get(source_name)
                bindings[field] = FieldBinding(
                    field, "source", (), (sid,) if sid else (), False, False,
                    "source_exact" if sid else "missing", "raw_nonbinding",
                )
        elif op == "test":
            required = ("solution", "problem", "entry_point")
            ids = tuple(batch.selected_occurrence_ids if batch else self._selected_occurrence_ids)
            bindings["solution"] = FieldBinding(
                "solution", "artifact", ids[:1], (), False, False,
                "selected_occurrence" if ids else "raw_unresolved", "required_artifact",
            )
            for field, source_name in (("problem", "problem"), ("entry_point", "entry_point"), ("tests", "tests")):
                sid = source_registry.get(source_name)
                if sid:
                    bindings[field] = FieldBinding(field, "source", (), (sid,), False, False, "source_exact", "required_source")
                elif field != "tests":
                    bindings[field] = FieldBinding(field, "source", (), (), False, False, "missing", "required_source")
        elif op == "custom" and "Failed solution:" in str(original_kwargs.get("input", "")):
            required = ("failed_code", "test_result", "problem")
            execution = self._latest_failed_test_execution
            code_ids = execution.tested_code_occurrence_ids if execution else ()
            result_ids = (execution.test_result_occurrence_id,) if execution else ()
            bindings["failed_code"] = FieldBinding("failed_code", "artifact", code_ids, (), False, False, "selected_occurrence" if code_ids else "raw_unresolved", "required_artifact")
            bindings["test_result"] = FieldBinding("test_result", "artifact", result_ids, (), False, False, "selected_occurrence" if result_ids else "raw_unresolved", "required_artifact")
            sid = source_registry.get("problem")
            bindings["problem"] = FieldBinding("problem", "source", (), (sid,) if sid else (), False, False, "source_exact" if sid else "missing", "required_source")
        specs = self._required_specs(operator)
        for spec in specs:
            if spec.field_name not in bindings:
                bindings[spec.field_name] = FieldBinding(
                    spec.field_name, spec.value_kind, (), (), spec.ordered,
                    spec.preserve_multiplicity, "missing", spec.requirement_class,
                )
        mandatory_occurrences = tuple(x for b in bindings.values() for x in b.occurrence_ids)
        mandatory_sources = tuple(x for b in bindings.values() for x in b.source_node_ids)
        return NativeInvocationPlan(
            invocation_id=reserved.invocation_id, task_id=self.task_id,
            operator=str(operator), candidate_batch_id=batch_id,
            original_kwargs=dict(original_kwargs), field_bindings=bindings,
            required_binding_names=tuple(spec.field_name for spec in specs) or required,
            mandatory_root_ids=tuple((*mandatory_occurrences, *mandatory_sources)),
            required_binding_specs=specs,
            mandatory_occurrence_ids=mandatory_occurrences,
            mandatory_source_ids=mandatory_sources,
            original_kwargs_digest=stable_digest(dict(original_kwargs)),
        )

    def ensure_endpoint(self, endpoint: str) -> AgentGraphView:
        endpoint = str(endpoint)
        if endpoint not in self.views.views:
            self.views.views[endpoint] = AgentGraphView(endpoint)
            self._refresh_views()
        return self.views.view(endpoint)

    def compile_output(
        self,
        *,
        operator: str,
        output: Any,
        input_node_ids: Iterable[str] = (),
        invocation_index: int | None = None,
        input_kwargs: Mapping[str, Any] | None = None,
        artifact_binding_hints: Mapping[str, tuple[str, ...]] | None = None,
        candidate_batch_id: str | None = None,
        runtime_invocation_id: str | None = None,
    ) -> AFlowCompiledOutput:
        self.ensure_endpoint(operator)
        input_node_ids = tuple(str(item) for item in input_node_ids)
        compiled = self.compiler.compile(
            self,
            operator=operator,
            output=output,
            input_node_ids=input_node_ids,
            invocation_index=invocation_index,
            input_kwargs=input_kwargs,
        )
        if self.artifact_enabled:
            self.compile_artifacts(
                operator=operator, output=output, input_node_ids=input_node_ids,
                invocation_index=compiled.invocation_index,
                invocation_node_id=compiled.invocation_node_id or compiled.primary_node_id,
                primary_node_id=compiled.primary_node_id,
                multiplicity=compiled.occurrence_count,
                input_kwargs=input_kwargs,
                artifact_binding_hints=artifact_binding_hints,
                candidate_batch_id=candidate_batch_id,
                canonical_node_ids=compiled.node_ids,
                runtime_invocation_id=runtime_invocation_id,
            )
        return compiled

    def compile_artifacts(self, *, operator: str, output: Any,
                          input_node_ids: Iterable[str] = (), invocation_index: int = 0,
                          invocation_node_id: str = "", primary_node_id: str = "",
                          multiplicity: int = 1,
                          input_kwargs: Mapping[str, Any] | None = None,
                          artifact_binding_hints: Mapping[str, tuple[str, ...]] | None = None,
                          candidate_batch_id: str | None = None,
                          canonical_node_ids: Iterable[str] = (),
                          runtime_invocation_id: str | None = None) -> tuple[Any, ...]:
        """Compile explicit output fields into auditable artifact nodes."""
        started = time.perf_counter()
        input_node_ids = tuple(str(x) for x in input_node_ids)
        records = artifact_records(output, operator=operator,
            invocation_id=invocation_node_id or f"{operator}:{invocation_index}",
            source_node_ids=input_node_ids, dependency_node_ids=input_node_ids,
            original_payload_ref=primary_node_id, multiplicity=multiplicity)
        compiled = []
        invocation_id = runtime_invocation_id or invocation_node_id or f"{operator}:{invocation_index}"
        occurrence_records: list[ArtifactOccurrence] = []
        canonical_ids = tuple(str(x) for x in canonical_node_ids)
        for record in records:
            node_id = canonical_ids[0] if canonical_ids else (input_node_ids[0] if input_node_ids else record.artifact_id)
            # AFlow's compiler already owns the canonical GraphStore node.  The
            # artifact compiler only registers a lightweight occurrence view.
            self._artifact_records[node_id] = record
            occurrence_id = f"occurrence:{stable_digest((self.task_id, invocation_id, record.artifact_id, len(occurrence_records)))[:20]}"
            occurrence_records.append(ArtifactOccurrence(
                occurrence_id=occurrence_id, canonical_artifact_id=node_id,
                producer_operator=str(operator), invocation_id=str(invocation_id),
                ordinal=len(occurrence_records), task_id=self.task_id,
                candidate_batch_id=candidate_batch_id,
                multiplicity_contribution=max(1, int(multiplicity)),
                original_payload_ref=record.original_payload_ref,
                source_node_ids=tuple(input_node_ids),
                raw_value=record.content,
            ))
            self._artifact_occurrences[occurrence_id] = occurrence_records[-1]
            if self.profile and self.profile.is_candidate_operator(operator):
                batch_id = candidate_batch_id or self._active_candidate_batch_id
                if batch_id and batch_id in self._candidate_batches:
                    batch = self._candidate_batches[batch_id]
                    self._candidate_batches[batch_id] = CandidateBatch(
                        batch_id=batch.batch_id, task_id=batch.task_id,
                        producer_stage=batch.producer_stage,
                        occurrence_ids=tuple((*batch.occurrence_ids, occurrence_id)),
                        created_at_invocation=batch.created_at_invocation,
                        frozen=batch.frozen, selected_occurrence_ids=batch.selected_occurrence_ids,
                        state=batch.state, ensemble_invocation_id=batch.ensemble_invocation_id,
                        test_invocation_ids=batch.test_invocation_ids,
                    )
            compiled.append(record)
        bindings = self._invocation_bindings.get(str(invocation_id))
        native_plan = self._native_invocation_plans.get(str(runtime_invocation_id or invocation_id))
        if native_plan is not None:
            bindings = tuple(
                InvocationInputBinding(
                    binding_id=f"native:{invocation_id}:{field}",
                    task_id=self.task_id,
                    consumer_invocation_id=str(invocation_id),
                    consumer_operator=str(operator),
                    field_name=field,
                    artifact_occurrence_ids=tuple(binding.occurrence_ids),
                    source_node_ids=tuple(binding.source_node_ids),
                    raw_value_digest=stable_digest(native_plan.original_kwargs.get(field, "")),
                    resolution=binding.resolution,
                    candidate_batch_id=native_plan.candidate_batch_id,
                )
                for field, binding in native_plan.field_bindings.items()
            )
        if bindings is None:
            bindings = capture_invocation_input_bindings(
                task_id=self.task_id, operator=operator, invocation_id=str(invocation_id),
                kwargs=input_kwargs or {}, artifact_binding_hints=artifact_binding_hints,
                source_node_ids=input_node_ids, candidate_batch_id=candidate_batch_id,
            )
            self._invocation_bindings[str(invocation_id)] = bindings
        edges = materialize_dependency_edges(
            dataset=self.dataset, operator=operator, invocation_id=str(invocation_id),
            bindings=bindings, output_occurrences=occurrence_records,
        )
        for edge in edges:
            self._artifact_dependency_edges[edge.edge_id] = edge
        required_validation = self.validate_required_bindings(native_plan) if native_plan is not None else None
        self.telemetry.record_aflow_metric(
            aflow_artifact_binding_count=len(bindings),
            aflow_artifact_exact_binding_count=(required_validation.valid_required_field_count if required_validation else sum(b.resolution == "exact_occurrence" for b in bindings)),
            aflow_artifact_unresolved_binding_count=sum(b.resolution == "raw_unresolved" for b in bindings),
            aflow_artifact_hard_edge_count=len(edges),
            aflow_artifact_required_binding_count=(required_validation.required_field_count if required_validation else 0),
            aflow_artifact_valid_required_binding_count=(required_validation.valid_required_field_count if required_validation else 0),
            aflow_artifact_invalid_required_binding_count=(required_validation.invalid_required_field_count if required_validation else 0),
            aflow_artifact_exact_binding_rate=(required_validation.exact_binding_rate if required_validation else 0.0),
        )
        self.telemetry.update_aflow_mapping_metric(
            "aflow_artifact_edges_by_relation",
            {edge.relation: 1 for edge in edges},
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        type_counts = {}
        for record in records:
            type_counts[record.artifact_type] = type_counts.get(record.artifact_type, 0) + 1
        self.telemetry.record_aflow_metric(
            aflow_artifact_count=len(records),
            aflow_artifact_compile_ms=elapsed,
            aflow_artifact_opaque_count=sum(r.artifact_type == "opaque_text" for r in records),
        )
        self.telemetry.update_aflow_mapping_metric("aflow_artifact_type_counts", type_counts)
        return tuple(compiled)

    def artifact_node_ids(self) -> tuple[str, ...]:
        return tuple(self._artifact_records)

    def latest_output_ids(self) -> tuple[str, ...]:
        return self._last_output_ids

    def candidate_pool_ids(self) -> tuple[str, ...]:
        if self.content_dedup:
            return tuple(self._candidate_artifact_ids)
        return tuple(self._candidate_pool_ids)

    def candidate_occurrence_ids(self) -> tuple[str, ...]:
        return tuple(self._candidate_occurrence_ids)

    def artifact_multiplicity(self, node_id: str) -> int:
        for record in self._artifact_by_digest.values():
            if record.node_id == str(node_id):
                return int(record.occurrence_count)
        return 1

    def _count_tokens(self, value: Any) -> int:
        if self.tokenizer is None:
            return max(1, len(_text_value(value).split()))
        encoded = self.tokenizer(_text_value(value), add_special_tokens=False)
        if hasattr(encoded, "get"):
            encoded = encoded.get("input_ids", [])
        elif hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        return len(encoded)

    def invocation_key(
        self,
        *,
        operator: str,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> AFlowInvocationKey:
        return AFlowInvocationKey.build(
            task_id=self.task_id,
            dataset=self.dataset,
            workflow_round=self.workflow_round,
            operator=operator,
            args=args,
            kwargs=kwargs,
            model_fingerprint=self.model_fingerprint,
            generation_config=self.generation_config,
        )

    def profile_is_ensemble(self, operator: str) -> bool:
        return bool(self.profile and self.profile.is_ensemble_operator(operator))

    def can_reuse_operator(self, operator: str, kwargs: Mapping[str, Any] | None = None) -> bool:
        if self.profile is None:
            allowed = not self.dataset
        else:
            allowed = self.profile.allows_reuse(operator, kwargs)
        if not allowed:
            return False
        do_sample = self.generation_config.get("do_sample")
        temperature = self.generation_config.get("temperature")
        return do_sample is False or temperature in (0, 0.0, None)

    def original_problem(self, kwargs: Mapping[str, Any]) -> str:
        return str(kwargs.get("problem") or self.task)

    def operator_value_for_node(self, node_id: str) -> Any:
        return _operator_value(self.raw_output(node_id))

    def candidate_projection(self, *, operator: str, output: Any) -> Any:
        if self.profile is not None and self.profile.candidate_projection is not None:
            return self.profile.candidate_projection(operator=operator, output=output, session=self)
        return _operator_value(output)

    def is_weighted_ensemble(self, operator: str) -> bool:
        return bool(
            self.profile
            and self.profile.ensemble_strategy.startswith("weighted")
            and self.profile.is_ensemble_operator(operator)
            and self.content_dedup
        )

    def is_native_multiplicity_ensemble(self, operator: str) -> bool:
        """Use the native Ensemble prompt after expanding canonical counts."""
        return bool(
            self.profile
            and self.profile.ensemble_strategy == "native_multiplicity"
            and self.profile.is_ensemble_operator(operator)
            and self.content_dedup
        )

    def weighted_candidates(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (str(self.operator_value_for_node(node_id)), self.artifact_multiplicity(node_id))
            for node_id in self.candidate_pool_ids()
        )

    def expanded_candidate_values(self) -> list[Any]:
        """Reconstruct the native occurrence list from canonical artifacts."""
        expanded: list[Any] = []
        for value, multiplicity in self.weighted_candidates():
            expanded.extend([value] * max(1, int(multiplicity)))
        return expanded

    def weighted_ensemble_prompt(self, original_kwargs: Mapping[str, Any] | None = None) -> tuple[str, dict[str, str]]:
        candidates = self.weighted_candidates()
        if not candidates:
            raise RuntimeError(f"{self.dataset} weighted ensemble received no candidates")
        if self.profile and self.profile.ensemble_prompt_builder is not None:
            prompt, mapping = self.profile.ensemble_prompt_builder(
                session=self,
                candidates=candidates,
                original_kwargs=original_kwargs or {},
            )
        else:
            mapping = {}
            blocks = []
            for index, (value, multiplicity) in enumerate(candidates):
                letter = chr(65 + index)
                mapping[letter] = value
                blocks.append(
                    f"{letter}:\nAnswer: {value}\nOccurrence count: {multiplicity}"
                )
            prompt = (
                "Several candidate answers were generated for the question below. "
                "Identical answers have been combined. Select the candidate with the "
                "highest occurrence count. If counts tie, select the first tied candidate.\n\n"
                f"Question: {self.task}\n\n" + "\n\n".join(blocks) + "\n\n"
                f"Allowed candidate IDs: {', '.join(mapping)}\n"
                "Return exactly one allowed ID in solution_letter."
            )
        self.telemetry.record_aflow_metric(
            aflow_weighted_ensemble_calls=1,
            aflow_weighted_unique_candidates=len(candidates),
            aflow_weighted_candidate_occurrences=sum(item[1] for item in candidates),
        )
        self.telemetry.emit("aflow_weighted_ensemble_started", {
            "task_id": self.task_id, "candidate_count": len(candidates),
            "occurrence_count": sum(item[1] for item in candidates),
        })
        return prompt, mapping

    def record_weighted_prompt_telemetry(
        self,
        *,
        native_prompt: str,
        weighted_prompt: str,
        candidates: tuple[tuple[str, int], ...],
    ) -> None:
        native_tokens = self._count_tokens(native_prompt)
        weighted_tokens = self._count_tokens(weighted_prompt)
        candidate_tokens = sum(self._count_tokens(value) for value, _ in candidates)
        metadata_tokens = max(0, weighted_tokens - candidate_tokens)
        self.telemetry.record_aflow_metric(
            aflow_native_ensemble_prompt_tokens=native_tokens,
            aflow_weighted_ensemble_prompt_tokens=weighted_tokens,
            aflow_weighted_candidate_content_tokens=candidate_tokens,
            aflow_weighted_metadata_tokens=metadata_tokens,
            aflow_weighted_prompt_saved_tokens=max(0, native_tokens - weighted_tokens),
        )
        self.telemetry.emit("aflow_weighted_prompt_accounted", {
            "task_id": self.task_id,
            "native_prompt_tokens": native_tokens,
            "weighted_prompt_tokens": weighted_tokens,
            "candidate_content_tokens": candidate_tokens,
            "metadata_tokens": metadata_tokens,
        })

    def weighted_selection(self, response: Any, mapping: Mapping[str, str]) -> str:
        answer = extract_weighted_solution_letter(response, mapping)
        if answer is not None:
            selected = mapping[answer]
            fallback = False
        else:
            candidates = self.weighted_candidates()
            selected = max(enumerate(candidates), key=lambda item: (item[1][1], -item[0]))[1][0]
            fallback = True
            self.telemetry.record_aflow_metric(aflow_weighted_ensemble_fallbacks=1)
            self.telemetry.emit("aflow_weighted_ensemble_fallback", {
                "task_id": self.task_id,
                "received_output": _text_value(response)[:1000],
                "parse_status": "invalid_or_empty",
                "fallback_reason": "unparseable_or_out_of_range_candidate_id",
            })
        self.telemetry.emit("aflow_weighted_ensemble_selected", {
            "task_id": self.task_id,
            "selected_answer": selected,
            "selected_code_digest": output_digest("selected_candidate", selected),
            "fallback": fallback,
        })
        self.telemetry.record_aflow_diagnostic({
            "task_id": self.task_id,
            "selected_code_digest": output_digest("selected_candidate", selected),
            "selected_candidate_multiplicity": next(
                (multiplicity for value, multiplicity in self.weighted_candidates() if value == selected),
                1,
            ),
            "weighted_fallback": fallback,
        })
        return selected

    def delivered_artifacts(self, node_ids: Iterable[str]) -> tuple[AFlowDeliveredArtifact, ...]:
        artifacts: list[AFlowDeliveredArtifact] = []
        for raw_id in node_ids:
            node_id = str(raw_id)
            node = self.store.node(node_id)
            producer = str(node.provenance.get("aflow_operator", node.owner))
            raw_value = self.raw_output(node_id)
            value = self.operator_value_for_node(node_id)
            if producer.casefold() == "programmer" and isinstance(raw_value, Mapping) and "output" in raw_value:
                value = raw_value["output"]
            artifacts.append(
                AFlowDeliveredArtifact(
                    node_id=node_id,
                    producer_operator=producer,
                    artifact_kind=str(node.provenance.get("artifact_kind", node.provenance.get("aflow_output_kind", node.type))),
                    candidate_role=str(node.provenance.get("candidate_role", "")),
                    multiplicity=self.artifact_multiplicity(node_id),
                    value=value,
                )
            )
        return tuple(artifacts)

    def occurrence_payload(self, occurrence_id: str) -> Any:
        occurrence = self._artifact_occurrences.get(str(occurrence_id))
        if occurrence is None:
            raise KeyError(f"unknown occurrence: {occurrence_id}")
        return occurrence.raw_value

    def canonical_for_occurrence(self, occurrence_id: str) -> str:
        occurrence = self._artifact_occurrences.get(str(occurrence_id))
        if occurrence is None:
            raise KeyError(f"unknown occurrence: {occurrence_id}")
        return occurrence.canonical_artifact_id

    def source_payload(self, source_node_id: str) -> Any:
        node_id = str(source_node_id)
        if node_id not in self.store.snapshot().nodes:
            raise KeyError(f"unknown source node: {node_id}")
        return self.store.node(node_id).content

    def record_bound_candidate_telemetry(
        self,
        *,
        operator: str,
        original_kwargs: Mapping[str, Any],
        bound_kwargs: Mapping[str, Any],
    ) -> None:
        if not self.profile_is_ensemble(operator) or "solutions" not in bound_kwargs:
            return
        original_solutions = original_kwargs.get("solutions", ())
        bound_solutions = bound_kwargs.get("solutions", ())
        before = self._count_tokens(original_solutions)
        after = self._count_tokens(bound_solutions)
        occurrence_count = len(original_solutions) if isinstance(original_solutions, (list, tuple)) else 0
        delivered_count = len(bound_solutions) if isinstance(bound_solutions, (list, tuple)) else 0
        multiplicities = [
            self.artifact_multiplicity(node_id)
            for node_id in self.candidate_pool_ids()
        ]
        self.telemetry.record_aflow_metric(
            aflow_operator_argument_tokens_before_dedup=before,
            aflow_operator_argument_tokens_after_dedup=after,
            aflow_delivered_candidate_count=delivered_count,
            aflow_candidate_occurrence_count=occurrence_count,
            aflow_downstream_tokens_before_dedup=before,
            aflow_downstream_tokens_after_dedup=after,
            aflow_dedup_saved_tokens=max(0, before - after),
        )
        self.telemetry.emit("aflow_context_delivered", {
            "task_id": self.task_id,
            "dataset": self.dataset,
            "operator": operator,
            "candidate_occurrences": occurrence_count,
            "delivered_candidates": delivered_count,
            "multiplicities": multiplicities,
            "operator_argument_tokens_before_dedup": before,
            "operator_argument_tokens_after_dedup": after,
        })

    def commit_context(self, context: AFlowContextResult | None) -> None:
        if context is None:
            return
        receiver_view = self.ensure_endpoint(context.receiver)
        receiver_view.grant_nodes(context.sent_node_ids)
        self._refresh_views()
        self.telemetry.emit("aflow_operator_delivery_committed", {
            "task_id": self.task_id, "receiver": context.receiver,
            "sent_node_ids": list(context.sent_node_ids),
        })

    def abort_context(self, context: AFlowContextResult | None) -> None:
        if context is None:
            return
        self.telemetry.emit("aflow_operator_delivery_aborted", {
            "task_id": self.task_id, "receiver": context.receiver,
            "sent_node_ids": list(context.sent_node_ids),
        })

    async def execute_invocation(
        self,
        key: AFlowInvocationKey,
        compute: Callable[[], Awaitable[Any]],
    ) -> tuple[Any, bool]:
        if self._invocation_reuse is None:
            return await compute(), False
        result = await self._invocation_reuse.get_or_compute(key, compute)
        self.telemetry.record_aflow_metric(
            aflow_invocation_reuse_hits=1 if result.hit else 0,
        )
        self.telemetry.emit(
            "aflow_invocation_reused" if result.hit else "aflow_invocation_started",
            {
                "task_id": self.task_id,
                "operator": key.operator,
                "cache_hit": result.hit,
                "normalized_input_digest": key.normalized_input_digest,
            },
        )
        return result.value, result.hit

    def consume_candidate_pool(self) -> None:
        if self.content_dedup and self._candidate_artifact_ids:
            self.telemetry.record_aflow_metric(
                aflow_candidate_tokens_after_dedup=sum(
                    self._artifact_token_cost.get(node_id, 0)
                    for node_id in self._candidate_artifact_ids
                )
            )
        self._candidate_pool_ids.clear()
        self._candidate_occurrence_ids.clear()
        self._candidate_artifact_ids.clear()

    def raw_output(self, node_id: str) -> Any:
        return self._raw_outputs.get(str(node_id))

    def prepare_operator_input(
        self,
        *,
        operator: str,
        kwargs: Mapping[str, Any],
        communication_mode: str,
    ) -> AFlowOperatorInput:
        """Make the graph slice the authoritative downstream input.

        AFlow still owns the workflow topology and operator order.  Once an
        operator has produced a graph node, however, the next operator in a
        graph mode receives only the receiver-specific slice selected by the
        shared communication runtime.  ``graph_shadow`` intentionally keeps
        the original AFlow arguments as a non-intervening control.
        """
        if communication_mode in {"artifact_shadow", "artifact_sendall", "artifact_minctx"}:
            if communication_mode == "artifact_shadow":
                self.telemetry.record_aflow_metric(aflow_artifact_shadow_parity=1)
                return AFlowOperatorInput(kwargs=dict(kwargs))
            if self.profile is None:
                return AFlowOperatorInput(kwargs=dict(kwargs))
            contract = self.profile.receiver_contract(operator)
            route = route_receiver_context(
                self, operator, kwargs, contract,
                policy=("send_all" if communication_mode == "artifact_sendall" else self.artifact_routing_policy),
            )
            route_plan = {
                "task_id": self.task_id,
                "operator": str(operator),
                "candidate_batch_id": route.candidate_batch_id,
                "selected_occurrences": list(
                    x for ids in (route.artifact_binding_hints or {}).values() for x in ids
                ),
                "selected_artifacts": list(route.selected_artifact_ids),
                "closure_artifacts": list(route.closure_artifact_ids),
                "source_node_ids": list(route.source_node_ids),
                "transport_sendall_tokens": route.transport_sendall_tokens,
                "transport_minctx_tokens": route.transport_minctx_tokens,
                "model_original_tokens": route.original_input_tokens,
                "model_minctx_tokens": route.selected_input_tokens,
                "fallback": route.fallback,
                "fallback_reason": route.fallback_reason,
                "contract_satisfied": route.contract_satisfied,
            }
            self._artifact_route_plans.append(route_plan)
            self.telemetry.record_aflow_artifact_route_plan(route_plan)
            if str(operator).casefold() == "scensemble":
                native_candidates = [
                    plan for plan in self._native_invocation_plans.values()
                    if plan.operator.casefold() == str(operator).casefold()
                ]
                current_native = native_candidates[-1] if native_candidates else None
                expected = len(getattr(current_native, "field_bindings", {}).get("solutions", FieldBinding("solutions")).occurrence_ids) if current_native else 0
                delivered = len(route.kwargs.get("solutions", ())) if isinstance(route.kwargs.get("solutions"), (list, tuple)) else 0
                preserved = int(expected > 0 and expected == delivered)
                self.telemetry.record_aflow_metric(
                    aflow_artifact_ensemble_expected_occurrences=expected,
                    aflow_artifact_ensemble_delivered_occurrences=delivered,
                    aflow_artifact_multiplicity_preservation_rate=float(preserved),
                )
            is_source_producer = self.profile.operator_role(operator, kwargs) == "source_producer" if self.profile else False
            self.telemetry.record_aflow_metric(
                aflow_artifact_route_count=1,
                aflow_artifact_selected_count=len(route.selected_artifact_ids),
                aflow_artifact_closure_count=len(route.closure_artifact_ids),
                aflow_artifact_original_input_tokens=route.original_input_tokens,
                aflow_artifact_selected_input_tokens=route.selected_input_tokens,
                aflow_artifact_saved_input_tokens=route.saved_input_tokens,
                aflow_artifact_fallback_count=int(route.fallback),
                aflow_artifact_contract_failures=int(not route.contract_satisfied),
                aflow_artifact_multiplicity_preserved=int(self.artifact_preserve_multiplicity),
                aflow_artifact_selection_overhead_ms=route.selection_overhead_ms,
                aflow_artifact_closure_replay_success_count=int((not is_source_producer) and route.closure_replayable and route.contract_satisfied),
                aflow_artifact_transport_sendall_tokens=route.transport_sendall_tokens,
                aflow_artifact_transport_minctx_tokens=route.transport_minctx_tokens,
                aflow_artifact_transport_saved_tokens=max(0, route.transport_sendall_tokens - route.transport_minctx_tokens),
                aflow_artifact_model_original_tokens=route.original_input_tokens,
                aflow_artifact_model_minctx_tokens=route.selected_input_tokens,
                aflow_artifact_model_saved_tokens=route.saved_input_tokens,
                aflow_artifact_closure_complete_count=int((not is_source_producer) and route.closure_replayable and route.contract_satisfied),
                aflow_artifact_replay_attempt_count=int(not is_source_producer),
                aflow_artifact_replay_success_count=int((not is_source_producer) and route.closure_replayable and route.contract_satisfied),
                aflow_artifact_replay_failure_count=int((not is_source_producer) and not (route.closure_replayable and route.contract_satisfied)),
            )
            self.telemetry.update_aflow_mapping_metric("aflow_artifact_route_events_by_operator", {str(operator): 1})
            self.telemetry.update_aflow_mapping_metric("aflow_artifact_original_tokens_by_operator", {str(operator): route.original_input_tokens})
            selected_map = (
                "aflow_artifact_sendall_tokens_by_operator"
                if communication_mode == "artifact_sendall"
                else "aflow_artifact_minctx_tokens_by_operator"
            )
            self.telemetry.update_aflow_mapping_metric(selected_map, {str(operator): route.selected_input_tokens})
            self.telemetry.update_aflow_mapping_metric(
                "aflow_artifact_sendall_nodes_by_operator" if communication_mode == "artifact_sendall" else "aflow_artifact_minctx_nodes_by_operator",
                {str(operator): len(route.selected_artifact_ids)},
            )
            self.telemetry.update_aflow_mapping_metric("aflow_artifact_saved_tokens_by_operator", {str(operator): route.saved_input_tokens})
            self.telemetry.update_aflow_mapping_metric(
                "aflow_artifact_compression_opportunity_reasons",
                {route.compression_reason or (route.fallback_reason or "fallback"): 1},
            )
            if self.profile.operator_role(operator, kwargs) == "source_producer":
                self.telemetry.record_aflow_metric(aflow_artifact_source_producer_passthrough_count=1)
            if route.fallback:
                self.telemetry.update_aflow_mapping_metric(
                    "aflow_artifact_fallback_reason_counts", {route.fallback_reason: 1}
                )
                self.telemetry.update_aflow_mapping_metric("aflow_artifact_fallbacks_by_operator", {str(operator): 1})
            return AFlowOperatorInput(
                kwargs=route.kwargs,
                source_node_ids=route.source_node_ids,
                artifact_binding_hints=route.artifact_binding_hints or {},
                candidate_batch_id=route.candidate_batch_id,
            )
        if communication_mode == "graph_shadow":
            return AFlowOperatorInput(kwargs=dict(kwargs))
        if communication_mode == "graph_minctx":
            if self.dataset.casefold() == "hotpotqa" and str(operator).casefold() == "answergenerate":
                if self.profile is None or not self.profile.supports_context_compression:
                    return AFlowOperatorInput(kwargs=dict(kwargs))
                if self.context_policy == "semantic_minimal":
                    return self._prepare_hotpot_claim_input(operator=operator, kwargs=kwargs)
                if self.context_policy == "semantic_shadow":
                    return self._prepare_hotpot_semantic_shadow(operator=operator, kwargs=kwargs)
                return self._prepare_hotpot_source_input(operator=operator, kwargs=kwargs)
            # Downstream AFlow operators retain the existing profile binding;
            # only the initial HotpotQA evidence prompt is compressed in MVP.
        if self.profile is None and self.dataset:
            raise RuntimeError(
                f"AFlow graph input takeover requires a registered profile for "
                f"{self.dataset!r} round {self.workflow_round}; use original or graph_shadow"
            )
        if self.profile is None:
            return AFlowOperatorInput(kwargs=dict(kwargs))
        if not self.profile.validated and not self.profile_explicit:
            raise RuntimeError(
                f"AFlow profile {self.profile.profile_name} is not validated; "
                "use original/graph_shadow or pass an explicit experimental profile"
            )

        operator_name = str(operator)
        if self.profile.source_selector is not None:
            source_ids = tuple(self.profile.source_selector(
                session=self,
                operator=operator_name,
                original_kwargs=kwargs,
            ))
        else:
            candidate_ids = self.candidate_pool_ids() if self.profile_is_ensemble(operator_name) else ()
            source_ids = tuple(candidate_ids or self.latest_output_ids())
        if not source_ids:
            bound = self.profile.input_binding(
                session=self, operator=operator_name, original_kwargs=kwargs, delivered_artifacts=()
            )
            self.record_bound_candidate_telemetry(
                operator=operator_name,
                original_kwargs=kwargs,
                bound_kwargs=bound.kwargs,
            )
            return bound

        source_node = self.store.node(source_ids[0])
        sender = str(source_node.provenance.get("aflow_operator", source_node.owner))
        context = self.receiver_context_for_operator(
            sender=sender,
            receiver=operator_name,
            input_node_ids=source_ids,
            commit=False,
        )
        delivered_ids = tuple(context.sent_node_ids)
        bound = self.profile.input_binding(
            session=self,
            operator=operator_name,
            original_kwargs=kwargs,
            delivered_artifacts=self.delivered_artifacts(delivered_ids),
        )
        self.record_bound_candidate_telemetry(
            operator=operator_name,
            original_kwargs=kwargs,
            bound_kwargs=bound.kwargs,
        )
        return AFlowOperatorInput(kwargs=bound.kwargs, source_node_ids=bound.source_node_ids or delivered_ids, context=context)

    def receiver_context_for_operator(
        self,
        *,
        sender: str,
        receiver: str,
        input_node_ids: Iterable[str],
        budget_tokens: int | None = None,
        commit: bool = True,
    ) -> AFlowContextResult:
        return self.build_receiver_context(
            sender=sender,
            receiver=receiver,
            node_ids=input_node_ids,
            budget_tokens=budget_tokens,
            commit=commit,
        )

    def build_receiver_context(
        self,
        *,
        sender: str,
        receiver: str,
        node_ids: Iterable[str],
        contract: SemanticContract | None = None,
        budget_tokens: int | None = None,
        policy: GraphCommunicationPolicy | None = None,
        commit: bool = True,
    ) -> AFlowContextResult:
        sender = str(sender)
        receiver = str(receiver)
        self.ensure_endpoint(sender)
        receiver_view = self.ensure_endpoint(receiver)
        state = self.store.snapshot()
        source_node_ids = tuple(str(item) for item in node_ids)
        candidates = extract_delta_candidates(
            state,
            node_ids=source_node_ids,
            sender=sender,
            receiver=receiver,
            receiver_view=receiver_view,
        )
        active_policy = policy or self.policy
        roots = active_policy.select_roots(
            state=state,
            sender_view=self.views.view(sender),
            receiver_view=receiver_view,
            candidates=candidates,
            task=self.dataset,
            budget_tokens=budget_tokens,
        )
        if active_policy.include_dependency_closure:
            delta = dependency_closure(
                state,
                root_node_ids=roots,
                sender=sender,
                receiver=receiver,
                receiver_view=receiver_view,
                policy=active_policy.name,
                candidate_token_cost=sum(item.token_cost for item in candidates),
            )
            sent_ids = tuple(delta.node_ids)
            closure_ids = tuple(delta.closure_added_node_ids)
        else:
            sent_ids = tuple(dict.fromkeys(str(item) for item in roots))
            closure_ids = ()

        selected_view = AgentGraphView(receiver)
        selected_view.grant_nodes(receiver_view.visible_node_ids | set(sent_ids))
        contract_results = ()
        if contract is not None:
            contract_results = SemanticResolver(state, selected_view).check_contract(contract)
        missing = tuple(
            result.requirement.description
            for result in contract_results
            if not result.is_sufficient
        )
        contract_satisfied = all_requirements_sufficient(contract_results) if contract is not None else True

        # Use the existing context slicer and compact renderer.  The explicit
        # roots parameter is the only AFlow-specific extension needed for its
        # operator contracts.
        mandatory_ids = tuple(
            node.node_id
            for node in self.store.snapshot().nodes.values()
            if node.task_id == self.task_id
            and node.branch_id == self.branch_id
            and node.logical_id in {"task", "query_spec"}
            and node.is_operationally_valid()
        )
        context_slice = build_context_slice(
            self.store,
            task_id=self.task_id,
            branch_id=self.branch_id,
            role=receiver,
            root_node_ids=tuple(dict.fromkeys((*mandatory_ids, *sent_ids))),
            policy="dependency_closure",
            token_budget=budget_tokens,
            allow_missing=True,
        )
        text = render_compact_context_slice(context_slice, self.store.snapshot())
        candidate_tokens = sum(item.token_cost for item in candidates)
        sent_tokens = int(context_slice.token_count)
        self.telemetry.record_graph_communication({
            "task_id": self.task_id,
            "dataset": self.dataset,
            "sender": sender,
            "receiver": receiver,
            "policy": active_policy.name,
            "candidate_count": len(candidates),
            "sent_count": len(sent_ids),
            "closure_added_count": len(closure_ids),
            "candidate_tokens": candidate_tokens,
            "sent_tokens": sent_tokens,
            "logical_communication_tokens": sent_tokens,
            "contract_satisfied": contract_satisfied,
            "missing_requirements": list(missing),
        })
        if commit:
            receiver_view.grant_nodes(sent_ids)
            self._refresh_views()
        return AFlowContextResult(
            text=text,
            receiver=receiver,
            sender=sender,
            root_node_ids=tuple(roots),
            sent_node_ids=sent_ids,
            closure_added_node_ids=closure_ids,
            candidate_tokens=candidate_tokens,
            sent_tokens=sent_tokens,
            contract_satisfied=contract_satisfied,
            missing_requirements=missing,
        )

    def _refresh_views(self) -> None:
        self.views.refresh_visible_edges()
        state = self.store.snapshot()
        public_ids = {
            node.node_id
            for node in state.nodes.values()
            if node.task_id == self.task_id
            and node.branch_id == self.branch_id
            and node.is_operationally_valid()
            and node.provenance.get("visibility_class") == VisibilityClass.PUBLIC
        }
        for view in self.views.views.values():
            view.grant_nodes(public_ids)


def _operator_value(output: Any) -> Any:
    """Extract the value expected by AFlow ensemble/test operators."""
    if not isinstance(output, Mapping):
        return output
    for key in ("response", "answer", "code", "solution", "value", "output"):
        if key in output and output[key] not in (None, ""):
            return output[key]
    return output


def make_aflow_operator_proxy_module(
    original_module: types.ModuleType,
    *,
    session: AFlowGraphSession,
    communication_mode: str,
) -> types.ModuleType:
    """Proxy AFlow operators while preserving the upstream Workflow graph."""
    if communication_mode not in {"graph_shadow", "graph_sendall", "graph_ra", "graph_minctx", "artifact_shadow", "artifact_sendall", "artifact_minctx"}:
        raise ValueError(f"unsupported AFlow graph mode: {communication_mode}")
    proxy = types.ModuleType(f"_aflow_graph_proxy_{original_module.__name__}")
    proxy.__file__ = getattr(original_module, "__file__", "")

    def wrap_class(original: type[Any]) -> type[Any]:
        class OperatorProxy(original):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._aflow_original_operator = original.__name__

            async def __call__(self, *args: Any, **kwargs: Any) -> Any:
                operator_name = original.__name__
                reserved = session.reserve_invocation(operator_name)
                if session.profile and session.profile.is_candidate_operator(operator_name):
                    session.ensure_candidate_batch(reserved)
                elif str(operator_name).casefold() == "scensemble":
                    session.freeze_candidate_batch()
                native_plan = session.build_native_invocation_plan(reserved, operator_name, kwargs)
                session._native_invocation_plans[reserved.invocation_id] = native_plan
                prepared = session.prepare_operator_input(
                    operator=operator_name,
                    kwargs=kwargs,
                    communication_mode=communication_mode,
                )
                prepared = AFlowOperatorInput(
                    kwargs=prepared.kwargs,
                    source_node_ids=prepared.source_node_ids,
                    context=prepared.context,
                    artifact_binding_hints=prepared.artifact_binding_hints or {},
                    candidate_batch_id=(
                        prepared.candidate_batch_id
                        or (session._active_candidate_batch_id if session.profile and session.profile.operator_role(operator_name, kwargs) == "source_producer" else session._last_closed_candidate_batch_id)
                    ),
                    invocation_id=reserved.invocation_id,
                )
                field_bindings = {
                    field: FieldBinding(
                        field_name=field,
                        value_kind="artifact" if ids else "raw",
                        occurrence_ids=tuple(ids),
                        source_node_ids=tuple(prepared.source_node_ids) if not ids else (),
                        ordered=field in {"solutions", "candidates"},
                        preserve_multiplicity=field in {"solutions", "candidates"},
                        resolution="direct_runtime" if ids else "raw_unresolved",
                    )
                    for field, ids in (prepared.artifact_binding_hints or {}).items()
                }
                plan = InvocationPlan(
                    invocation_id=reserved.invocation_id,
                    invocation_index=reserved.invocation_index,
                    task_id=session.task_id,
                    operator=operator_name,
                    candidate_batch_id=prepared.candidate_batch_id,
                    original_kwargs=dict(kwargs),
                    rendered_kwargs=dict(prepared.kwargs),
                    field_bindings=native_plan.field_bindings,
                    mandatory_root_ids=native_plan.mandatory_root_ids,
                    selected_occurrence_ids=tuple(x for ids in (prepared.artifact_binding_hints or {}).values() for x in ids),
                    selected_source_ids=tuple(prepared.source_node_ids),
                    communication_mode=communication_mode,
                    contract_name=(session.profile.receiver_contract(operator_name).receiver if session.profile else operator_name),
                )
                session._invocation_plans[reserved.invocation_id] = plan
                bindings = capture_invocation_input_bindings(
                    task_id=session.task_id, operator=operator_name,
                    invocation_id=reserved.invocation_id, kwargs=prepared.kwargs,
                    artifact_binding_hints=prepared.artifact_binding_hints,
                    source_node_ids=prepared.source_node_ids,
                    candidate_batch_id=prepared.candidate_batch_id,
                )
                session._invocation_bindings[reserved.invocation_id] = bindings
                required_validation = session.validate_required_bindings(
                    native_plan,
                    session.profile.receiver_contract(operator_name) if session.profile else None,
                )
                native_plan.binding_validations = required_validation.validations_by_field
                validations = tuple(required_validation.validations_by_field.values())
                session.telemetry.record_aflow_metric(
                    aflow_artifact_exact_binding_count=required_validation.valid_required_field_count,
                    aflow_artifact_binding_failure_count=sum(not v.valid for v in validations),
                    aflow_artifact_required_binding_count=required_validation.required_field_count,
                    aflow_artifact_valid_required_binding_count=required_validation.valid_required_field_count,
                    aflow_artifact_invalid_required_binding_count=required_validation.invalid_required_field_count,
                    aflow_artifact_exact_binding_rate=required_validation.exact_binding_rate,
                )
                for validation in validations:
                    if not validation.valid:
                        session.telemetry.update_aflow_mapping_metric(
                            "aflow_artifact_binding_failures_by_reason",
                            {validation.failure_reason or "unknown": 1},
                        )
                async def compute() -> Any:
                    if session.is_weighted_ensemble(operator_name):
                        # Use the call-time kwargs for the native baseline.  The
                        # prepared kwargs may already contain the deduplicated
                        # candidate list and therefore cannot represent A,A,A.
                        prompt, mapping = session.weighted_ensemble_prompt(kwargs)
                        native_template = getattr(original_module, "SC_ENSEMBLE_PROMPT", None)
                        solutions = kwargs.get("solutions", ())
                        solution_text = "".join(
                            f"{chr(65 + index)}: \n{str(solution)}\n\n\n"
                            for index, solution in enumerate(solutions)
                        )
                        if native_template is not None:
                            try:
                                native_prompt = native_template.format(
                                    question=kwargs.get("problem", session.task),
                                    solutions=solution_text,
                                )
                            except (KeyError, IndexError):
                                try:
                                    native_prompt = native_template.format(solutions=solution_text)
                                except (KeyError, IndexError):
                                    native_prompt = solution_text
                        else:
                            native_prompt = solution_text
                        session.record_weighted_prompt_telemetry(
                            native_prompt=native_prompt,
                            weighted_prompt=prompt,
                            candidates=session.weighted_candidates(),
                        )
                        if hasattr(self, "_fill_node"):
                            op_class = getattr(original_module, "ScEnsembleOp", None)
                            mode = session.profile.ensemble_fill_mode if session.profile else "xml_fill"
                            response = await self._fill_node(op_class, prompt, mode=mode or "xml_fill")
                        else:
                            action_node = getattr(original_module, "ActionNode", None)
                            op_class = getattr(original_module, "ScEnsembleOp", None)
                            if action_node is None or op_class is None:
                                raise RuntimeError("weighted ensemble requires ScEnsembleOp and ActionNode")
                            fill_kwargs = {"context": prompt, "llm": self.llm}
                            mode = session.profile.ensemble_fill_mode if session.profile else None
                            if mode:
                                fill_kwargs["mode"] = mode
                            node = await action_node.from_pydantic(op_class).fill(**fill_kwargs)
                            response = node.instruct_content.model_dump()
                        return {"response": session.weighted_selection(response, mapping)}
                    if session.is_native_multiplicity_ensemble(operator_name):
                        native_kwargs = dict(prepared.kwargs)
                        native_kwargs["solutions"] = session.expanded_candidate_values()
                        return await super(OperatorProxy, self).__call__(*args, **native_kwargs)
                    return await super(OperatorProxy, self).__call__(*args, **prepared.kwargs)

                try:
                    reuse_allowed = (
                        session.invocation_reuse_enabled
                        and session.can_reuse_operator(operator_name, prepared.kwargs)
                    )
                    reuse_key = None
                    if reuse_allowed:
                        reuse_key = session.invocation_key(
                            operator=operator_name,
                            args=args,
                            kwargs=prepared.kwargs,
                        )
                        session.telemetry.emit("aflow_invocation_reuse_decision", {
                            "task_id": session.task_id,
                            "operator": operator_name,
                            "candidate_role": (
                                session.profile.candidate_signature(
                                    operator=operator_name,
                                    kwargs=prepared.kwargs,
                                    output=None,
                                )
                                if session.profile and session.profile.is_candidate_operator(operator_name)
                                else ""
                            ),
                            "reuse_allowed": True,
                            "reuse_key_digest": reuse_key.normalized_input_digest,
                        })
                    elif session.invocation_reuse_enabled:
                        session.telemetry.emit("aflow_invocation_reuse_decision", {
                            "task_id": session.task_id,
                            "operator": operator_name,
                            "candidate_role": "",
                            "reuse_allowed": False,
                        })
                    if reuse_key is not None:
                        result, cache_hit = await session.execute_invocation(reuse_key, compute)
                    else:
                        result = await compute()
                        cache_hit = False
                    try:
                        compiled_output = session.compile_output(
                            operator=operator_name,
                            output=result,
                            input_node_ids=prepared.source_node_ids,
                            input_kwargs=prepared.kwargs,
                            artifact_binding_hints=prepared.artifact_binding_hints,
                            candidate_batch_id=prepared.candidate_batch_id,
                            invocation_index=reserved.invocation_index,
                            runtime_invocation_id=reserved.invocation_id,
                        )
                        if str(operator_name).casefold() == "test":
                            session.record_test_execution(reserved.invocation_id, native_plan, result)
                            session.archive_candidate_batch(native_plan.candidate_batch_id, reserved.invocation_id)
                        if str(operator_name).casefold() == "scensemble":
                            batch_id = prepared.candidate_batch_id or session._last_closed_candidate_batch_id
                            batch = session._candidate_batches.get(batch_id) if batch_id else None
                            if batch is not None:
                                resolution = resolve_ensemble_selection(result, batch, session)
                                session.capture_selected_occurrences(resolution.selected_occurrence_ids)
                                if not resolution.ambiguous:
                                    session.mark_batch_selected(batch_id, reserved.invocation_id)
                                session.telemetry.record_aflow_metric(
                                    aflow_artifact_selection_ambiguous_count=int(resolution.ambiguous),
                                )
                        plan.committed = True
                        session.telemetry.record_aflow_metric(aflow_artifact_invocation_plan_count=1)
                    except BaseException as exc:
                        # Shadow is a non-intervening control path.  Graph
                        # bookkeeping must never turn a successful native
                        # AFlow call into a workflow failure.
                        if communication_mode != "graph_shadow":
                            raise
                        session.telemetry.emit("aflow_shadow_compile_error", {
                            "task_id": session.task_id,
                            "operator": operator_name,
                            "error": repr(exc),
                        })
                    session.commit_context(prepared.context)
                    if "ensemble" in operator_name.casefold():
                        session.consume_candidate_pool()
                    return result
                except BaseException:
                    session.abort_context(prepared.context)
                    raise

        OperatorProxy.__name__ = original.__name__
        OperatorProxy.__qualname__ = original.__qualname__
        OperatorProxy.__module__ = proxy.__name__
        return OperatorProxy

    for name, value in vars(original_module).items():
        if inspect.isclass(value) and value.__module__ == original_module.__name__:
            setattr(proxy, name, wrap_class(value))
        else:
            setattr(proxy, name, value)
    return proxy

