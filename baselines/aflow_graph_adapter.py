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
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

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


@dataclass(frozen=True)
class AFlowCompiledOutput:
    """Deterministically compiled representation of one AFlow output."""

    node_ids: tuple[str, ...]
    primary_node_id: str
    operator: str
    invocation_index: int
    output_kind: str


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
        operator_key = re.sub(r"[^a-zA-Z0-9_]+", "_", operator).strip("_").lower() or "operator"
        logical_id = f"aflow_{operator_key}_{index}"
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
                "aflow_invocation_index": index,
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
            invocation_index=index,
            output_kind=kind,
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
    ) -> None:
        self.task_id = str(task_id)
        self.task = str(task)
        self.dataset = str(dataset)
        self.branch_id = str(branch_id)
        self.run_id = str(run_id or f"aflow:{self.task_id}")
        self.store = GraphStore()
        self.telemetry = telemetry or WorkflowTelemetry()
        self.policy = policy or SendAllPolicy()
        self._invocations: dict[str, int] = {}
        self._last_output_ids: tuple[str, ...] = ()
        self._candidate_pool_ids: list[str] = []
        self._raw_outputs: dict[str, Any] = {}

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
        return tuple(self._candidate_pool_ids)

    def consume_candidate_pool(self) -> None:
        self._candidate_pool_ids.clear()

    def raw_output(self, node_id: str) -> Any:
        return self._raw_outputs.get(str(node_id))

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
                lowered = operator_name.casefold()
                candidate_ids = session.candidate_pool_ids() if "ensemble" in lowered else ()
                latest_ids = session.latest_output_ids()
                source_ids = candidate_ids
                if not source_ids:
                    raw_input = str(kwargs.get("input", ""))
                    has_embedded_previous = any(
                        marker in raw_input
                        for marker in ("Best solution:", "Best answer:", "Code output:", "Failed solution:", "Error:")
                    )
                    consumes_latest = (
                        "test" in lowered
                        or "program" in lowered
                        or "analysis" in kwargs
                        or has_embedded_previous
                        or (
                            "generate" in lowered
                            and bool(kwargs.get("instruction"))
                            and not (
                                latest_ids
                                and session.store.node(latest_ids[-1]).type in {"candidate_answer", "code"}
                            )
                        )
                    )
                    source_ids = latest_ids if consumes_latest else ()
                rewritten_kwargs = dict(kwargs)

                if communication_mode != "graph_shadow" and source_ids:
                    source_node = session.store.node(source_ids[0])
                    sender = str(source_node.provenance.get("aflow_operator", source_node.owner))
                    context = session.receiver_context_for_operator(
                        sender=sender,
                        receiver=operator_name,
                        input_node_ids=source_ids,
                    )
                    if "solutions" in rewritten_kwargs and candidate_ids:
                        selected = [
                            _operator_value(session.raw_output(node_id))
                            for node_id in context.sent_node_ids
                            if node_id in candidate_ids
                        ]
                        if selected:
                            rewritten_kwargs["solutions"] = selected
                    elif "solution" in rewritten_kwargs:
                        rewritten_kwargs["solution"] = _operator_value(
                            session.raw_output(source_ids[-1])
                        )
                    elif "analysis" in rewritten_kwargs:
                        rewritten_kwargs["analysis"] = context.text
                    elif "input" in rewritten_kwargs:
                        rewritten_kwargs["input"] = context.text
                    elif "problem" in rewritten_kwargs and "generate" in lowered:
                        rewritten_kwargs["problem"] = context.text

                result = await super().__call__(*args, **rewritten_kwargs)
                session.compile_output(
                    operator=operator_name,
                    output=result,
                    input_node_ids=source_ids,
                )
                if "ensemble" in lowered:
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

