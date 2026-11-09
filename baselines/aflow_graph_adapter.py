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
        self.views = AgentGraphViewManager(
            self.store,
            task_id=self.task_id,
            branch_id=self.branch_id,
            agents=tuple(dict.fromkeys(str(item) for item in endpoints)),
        )
        self.compiler = AFlowOperatorOutputCompiler()
        self._refresh_views()

    def next_invocation(self, operator: str) -> int:
        key = str(operator)
        self._invocations[key] = self._invocations.get(key, 0) + 1
        return self._invocations[key]

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
    ) -> AFlowCompiledOutput:
        self.ensure_endpoint(operator)
        return self.compiler.compile(
            self,
            operator=operator,
            output=output,
            input_node_ids=input_node_ids,
            invocation_index=invocation_index,
            input_kwargs=input_kwargs,
        )

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
        if communication_mode == "graph_shadow":
            return AFlowOperatorInput(kwargs=dict(kwargs))
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
    if communication_mode not in {"graph_shadow", "graph_sendall", "graph_ra"}:
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
                prepared = session.prepare_operator_input(
                    operator=operator_name,
                    kwargs=kwargs,
                    communication_mode=communication_mode,
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
                        session.compile_output(
                            operator=operator_name,
                            output=result,
                            input_node_ids=prepared.source_node_ids,
                            input_kwargs=prepared.kwargs,
                        )
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

