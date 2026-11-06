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
class AFlowOperatorInput:
    """Graph-controlled input for one AFlow Operator invocation."""

    kwargs: dict[str, Any]
    source_node_ids: tuple[str, ...] = ()
    context: AFlowContextResult | None = None


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def _is_candidate_output(operator: str, output_kind: str) -> bool:
    """Return whether an output belongs to the current ensemble candidate batch."""
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
            )
        return self._compile_legacy(
            session,
            operator=operator,
            output=output,
            input_node_ids=input_node_ids,
            invocation_index=index,
            kind=kind,
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
        if kind in {"candidate_answer", "code", "calculation"}:
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

        key = (kind, digest)
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
                },
                created_by_role=operator,
                run_id=session.run_id,
            )
            record = AFlowArtifactRecord(
                node_id=artifact.node_id,
                output_kind=kind,
                content_digest=digest,
                raw_output=output,
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
            if _is_candidate_output(operator, kind):
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
        if _is_candidate_output(operator, kind):
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
        )


class AFlowGraphSession:
    """Thin AFlow adapter over the existing graph communication runtime."""

    def __init__(
        self,
        *,
        task_id: str,
        task: str,
        dataset: str = "",
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
        self.branch_id = str(branch_id)
        self.run_id = str(run_id or f"aflow:{self.task_id}")
        if dedup_mode not in {"exact", "structured"}:
            raise ValueError(f"unsupported AFlow dedup mode: {dedup_mode}")
        self.content_dedup = bool(content_dedup)
        self.invocation_reuse_enabled = bool(invocation_reuse)
        self.dedup_mode = str(dedup_mode)
        self.model_fingerprint = str(model_fingerprint)
        self.generation_config = dict(generation_config or {})
        self.tokenizer = tokenizer
        self.store = GraphStore()
        self.telemetry = telemetry or WorkflowTelemetry()
        self.policy = policy or SendAllPolicy()
        self._invocations: dict[str, int] = {}
        self._last_output_ids: tuple[str, ...] = ()
        self._candidate_pool_ids: list[str] = []
        self._candidate_occurrence_ids: list[str] = []
        self._candidate_artifact_ids: list[str] = []
        self._artifact_by_digest: dict[tuple[str, str], AFlowArtifactRecord] = {}
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
    ) -> AFlowCompiledOutput:
        self.ensure_endpoint(operator)
        return self.compiler.compile(
            self,
            operator=operator,
            output=output,
            input_node_ids=input_node_ids,
            invocation_index=invocation_index,
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
            operator=operator,
            args=args,
            kwargs=kwargs,
            model_fingerprint=self.model_fingerprint,
            generation_config=self.generation_config,
        )

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
        rewritten = dict(kwargs)
        if communication_mode == "graph_shadow":
            return AFlowOperatorInput(kwargs=rewritten)

        operator_name = str(operator)
        lowered = operator_name.casefold()
        candidate_ids = self.candidate_pool_ids() if "ensemble" in lowered else ()
        latest_ids = self.latest_output_ids()
        source_ids = tuple(candidate_ids)
        if not source_ids:
            raw_input = str(rewritten.get("input", ""))
            has_embedded_previous = any(
                marker in raw_input
                for marker in (
                    "Best solution:", "Best answer:", "Code output:",
                    "Failed solution:", "Error:",
                )
            )
            consumes_latest = (
                "test" in lowered
                or "program" in lowered
                or "analysis" in rewritten
                or has_embedded_previous
                or (
                    "generate" in lowered
                    and bool(rewritten.get("instruction"))
                    and not (
                        latest_ids
                        and self.store.node(latest_ids[-1]).type
                        in {"candidate_answer", "code"}
                    )
                )
            )
            source_ids = latest_ids if consumes_latest else ()

        if not source_ids:
            return AFlowOperatorInput(kwargs=rewritten)

        source_node = self.store.node(source_ids[0])
        sender = str(source_node.provenance.get("aflow_operator", source_node.owner))
        context = self.receiver_context_for_operator(
            sender=sender,
            receiver=operator_name,
            input_node_ids=source_ids,
        )
        delivered_ids = tuple(context.sent_node_ids)

        # Never fall back to the original full history.  An empty selected
        # candidate set is a real protocol outcome and must remain observable.
        if "solutions" in rewritten and candidate_ids:
            rewritten["solutions"] = [
                _operator_value(self.raw_output(node_id))
                for node_id in delivered_ids
                if node_id in candidate_ids
            ]
        elif "solution" in rewritten:
            delivered_sources = [node_id for node_id in delivered_ids if node_id in source_ids]
            rewritten["solution"] = (
                _operator_value(self.raw_output(delivered_sources[-1]))
                if delivered_sources else ""
            )
        elif "analysis" in rewritten:
            rewritten["analysis"] = context.text
        elif "input" in rewritten:
            rewritten["input"] = context.text
        elif "problem" in rewritten and "generate" in lowered:
            rewritten["problem"] = context.text

        return AFlowOperatorInput(
            kwargs=rewritten,
            source_node_ids=delivered_ids,
            context=context,
        )

    def receiver_context_for_operator(
        self,
        *,
        sender: str,
        receiver: str,
        input_node_ids: Iterable[str],
        budget_tokens: int | None = None,
    ) -> AFlowContextResult:
        return self.build_receiver_context(
            sender=sender,
            receiver=receiver,
            node_ids=input_node_ids,
            budget_tokens=budget_tokens,
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
        if self.content_dedup and "ensemble" in receiver.casefold():
            before_tokens = sum(
                self._count_tokens(self.raw_output(self._occurrence_artifact_ids.get(invocation_id, "")))
                for invocation_id in self._candidate_occurrence_ids
            )
            after_tokens = sum(
                self._artifact_token_cost.get(node_id, 0)
                for node_id in sent_ids
                if node_id in self._candidate_artifact_ids
            )
            self.telemetry.record_aflow_metric(
                aflow_downstream_tokens_before_dedup=before_tokens,
                aflow_downstream_tokens_after_dedup=after_tokens,
                aflow_dedup_saved_tokens=max(0, before_tokens - after_tokens),
            )
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
                    return await super(OperatorProxy, self).__call__(*args, **prepared.kwargs)

                if session.invocation_reuse_enabled:
                    key = session.invocation_key(
                        operator=operator_name,
                        args=args,
                        kwargs=prepared.kwargs,
                    )
                    result, cache_hit = await session.execute_invocation(key, compute)
                else:
                    result = await compute()
                    cache_hit = False
                session.compile_output(
                    operator=operator_name,
                    output=result,
                    input_node_ids=prepared.source_node_ids,
                )
                if "ensemble" in operator_name.casefold():
                    session.consume_candidate_pool()
                return result

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

