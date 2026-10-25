"""LangGraph orchestration over incremental Actions and GraphStore."""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Mapping, Optional, Sequence, TypedDict

from .action_compiler import ActionCompilationError, ActionCompiler
from .action_constraints import ActionConstraint, build_action_constraint
from .agent_graph_view import AgentGraphViewManager, MANDATORY_LOGICAL_IDS
from .models import node_visible_to_role
from .communication import ClosureAwareHeuristicPolicy, GraphCommunicationPolicy
from .communication_replay import (
    CommunicationDecisionSnapshot,
    CommunicationReplayBuffer,
    candidate_snapshot,
    fragment_snapshot,
    new_snapshot_id,
    normalize_external_state,
)
from .context_slicer import build_context_slice, render_compact_context_slice, _node_sort_key
from .delta_closure import dependency_closure
from .delta_extractor import extract_delta_candidates
from .delta_scoring import estimate_edge_tokens, estimate_node_tokens
from .domain_executors import DomainExecution, execute_domain
from .code_verifier import CodeVerifier
from .feedback_controller import FeedbackAction, decide_feedback_action
from .graph_delta import DeltaCandidate, GraphDelta
from .graph_store import GraphStore
from .hotpot_bridge import bridge_candidates_from_values, deduplicate_bridge_candidates, extract_bridge_candidate
from .semantic_fragments import (
    MANDATORY_FRAGMENT,
    OPTIONAL_FRAGMENT,
    all_fragment_ids,
    fragment_token_cost,
    fragments_for_node,
    node_id_from_fragment,
    render_fragmented_content,
)
from .semantic_contract import default_semantic_contract
from .semantic_feedback import (
    build_semantic_feedback,
    is_feedback_request,
    is_ack,
    is_hard_nack,
    is_nack,
    is_quality_nack,
    is_soft_nack,
    is_verification_nack,
    nack_level,
    nack_missing_semantics,
)
from .semantic_innovation import detect_innovations
from .semantic_packet import build_initial_semantic_packet
from .semantic_resolver import SemanticResolver
from .receiver_need import diagnose_receiver_need
from .refinement_planner import plan_refinement
from .protocol import (
    PROTOCOL_VERSION,
    ROLE_SYSTEM_PROMPTS,
    action_contract,
    parse_action,
)
from .semantic_actions import CRITIC_TYPED_OPS, TYPED_SEMANTIC_OPS, semantic_action_example
from .choice_audit import build_choice_audit_manifest
from .task_spec import get_task_spec
from .tools import ToolRegistry
from .telemetry import WorkflowTelemetry
from .task_verifier import apply_verification_signal, enforce_hotpot_entailment_verdict, verify_task_candidate
from .native_agents import AgentConfig, NativeAgent
from .message_bus import MessageBus, MessageEnvelope
from .native_tooling import (
    NativeToolCall,
    NativeToolResult,
    infer_default_tool_calls,
    parse_tool_calls,
    render_tool_results,
)
from .native_critic import parse_critic_verdict, render_critic_feedback, REPAIR_VERDICTS
from .evaluation_context import EvaluationContext


class WorkflowState(TypedDict, total=False):
    task_id: str
    branch_id: str
    task_type: str
    graph_revision: int
    graph_snapshot_digest: str
    round_id: int
    action_count: int
    session_id: str
    status: str
    error: Optional[str]
    last_role: str
    last_action: dict[str, Any]
    verification_status: str
    logs: list[dict[str, Any]]


@dataclass(frozen=True)
class ModelRequest:
    role: str
    task_type: str
    prompt: str
    mode: str = "normal"
    session_id: str = ""
    session_prompt: str = ""
    session_reset: bool = False
    session_rollback: bool = False
    action_constraint: Optional[ActionConstraint] = None
    # Native agents add identity and policy metadata without changing the
    # optimized Action workflow's positional request contract.
    agent_id: str = ""
    agent_config: Any = None
    tool_permissions: tuple[str, ...] = ()


ModelCallable = Callable[[ModelRequest], Any]


class LangGraphWorkflow:
    """Build a Planner→Solver→Tool→Critic workflow using one Action at a time."""

    def __init__(
        self,
        *,
        store: GraphStore,
        model: ModelCallable,
        task_id: str,
        branch_id: str = "main",
        task_type: str = "numeric_solve",
        tokenizer: Any = None,
        max_rounds: int = 3,
        max_actions_per_role: int = 8,
        tool_registry: Optional[ToolRegistry] = None,
        telemetry: Optional[WorkflowTelemetry] = None,
        enable_action_constraints: bool = True,
        communication_policy: Optional[GraphCommunicationPolicy] = None,
        communication_budget_tokens: Optional[int] = None,
        communication_replay_buffer: Optional[CommunicationReplayBuffer] = None,
        replay_external_state_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
        graph_context_mode: str = "baseline",
        action_protocol: str = "semantic_delta_v2",
        evaluation_context: EvaluationContext | None = None,
        multihop_sentence_limit: int = 48,
    ) -> None:
        self.store = store
        self.model = model
        self.task_id = task_id
        self.branch_id = branch_id
        self.task_type = task_type
        if action_protocol != "semantic_delta_v2":
            raise ValueError(f"unsupported action protocol: {action_protocol}")
        self.tokenizer = tokenizer
        self.session_id = f"workflow:{task_id}:{branch_id}"
        self.tool_registry = tool_registry or ToolRegistry.default()
        self.compiler = ActionCompiler(
            store,
            task_id=task_id,
            branch_id=branch_id,
            task_type=task_type,
            tool_registry=self.tool_registry,
        )
        self.max_rounds = max(1, max_rounds)
        self.max_actions_per_role = max(1, max_actions_per_role)
        self.telemetry = telemetry
        self.enable_action_constraints = bool(enable_action_constraints)
        self.communication_policy = communication_policy or ClosureAwareHeuristicPolicy()
        self.communication_budget_tokens = communication_budget_tokens
        self.communication_replay_buffer = communication_replay_buffer
        self.replay_external_state_provider = replay_external_state_provider
        self._communication_decision_index = 0
        if graph_context_mode not in {"baseline", "deduplicated", "source_state_split", "role_aware"}:
            raise ValueError(f"unsupported graph_context_mode: {graph_context_mode}")
        self.graph_context_mode = graph_context_mode
        self.evaluation_context = evaluation_context
        self.multihop_sentence_limit = max(0, int(multihop_sentence_limit))
        self._graph_source_seen_hashes: set[str] = set()
        self._last_hotpot_repair_packet_audit: dict[str, Any] = {}
        self.agent_views = AgentGraphViewManager(
            store,
            task_id=task_id,
            branch_id=branch_id,
            agents=("planner", "solver", "critic", "final_solver", "tool"),
        )

    def compile(self, *, checkpointer: Any = None) -> Any:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("LangGraph is required. Install it with: pip install langgraph") from exc

        graph = StateGraph(WorkflowState)
        graph.add_node("planner", self._agent_node("planner"))
        graph.add_node("solver", self._agent_node("solver"))
        graph.add_node("tool", self._tool_node)
        graph.add_node("critic", self._agent_node("critic"))
        graph.add_node("repair", self._agent_node("solver", mode="repair"))
        graph.add_node("finalizer", self._deterministic_finalizer_node)
        graph.add_node("unresolved", self._unresolved_verification_node)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "solver")
        graph.add_edge("solver", "tool")
        graph.add_edge("tool", "critic")
        graph.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {"critic": "critic", "repair": "repair", "finalizer": "finalizer", "unresolved": "unresolved"},
        )
        graph.add_edge("repair", "tool")
        graph.add_edge("finalizer", END)
        graph.add_edge("unresolved", END)
        return graph.compile(checkpointer=checkpointer)

    def compile_from_communication_snapshot(
        self,
        snapshot: CommunicationDecisionSnapshot,
        *,
        checkpointer: Any = None,
    ) -> Any:
        """Compile only the suffix after a captured communication decision.

        The communication action has already mutated the restored receiver
        view.  Restarting at Planner would invalidate the counterfactual, so
        each sender/receiver edge maps to its actual downstream workflow node.
        """
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("LangGraph is required. Install it with: pip install langgraph") from exc
        if snapshot.task_id != self.task_id or snapshot.branch_id != self.branch_id:
            raise ValueError("communication snapshot belongs to a different workflow boundary")

        graph = StateGraph(WorkflowState)
        graph.add_node("solver", self._agent_node("solver"))
        graph.add_node("tool", self._tool_node)
        graph.add_node("critic", self._agent_node("critic"))
        graph.add_node("repair", self._agent_node("solver", mode="repair"))
        graph.add_node("finalizer", self._deterministic_finalizer_node)
        graph.add_node("unresolved", self._unresolved_verification_node)
        graph.add_edge("solver", "tool")
        graph.add_edge("tool", "critic")
        graph.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {"critic": "critic", "repair": "repair", "finalizer": "finalizer", "unresolved": "unresolved"},
        )
        graph.add_edge("repair", "tool")
        graph.add_edge("finalizer", END)
        graph.add_edge("unresolved", END)

        start_node = {
            ("planner", "solver"): "solver",
            ("solver", "critic"): "critic",
            ("tool", "critic"): "critic",
            ("critic", "solver"): "repair",
            ("critic", "final_solver"): "finalizer",
        }.get((snapshot.sender, snapshot.receiver))
        if start_node is None:
            raise ValueError(f"unsupported communication continuation stage: {snapshot.stage}")
        graph.add_edge(START, start_node)
        return graph.compile(checkpointer=checkpointer)

    def fork(self, branch_id: str) -> "LangGraphWorkflow":
        """Create an isolated branch workflow from the current branch."""
        self.store.create_branch(
            self.task_id,
            branch_id,
            source_branch_id=self.branch_id,
        )
        return LangGraphWorkflow(
            store=self.store,
            model=self.model,
            task_id=self.task_id,
            branch_id=branch_id,
            task_type=self.task_type,
            tokenizer=self.tokenizer,
            max_rounds=self.max_rounds,
            max_actions_per_role=self.max_actions_per_role,
            tool_registry=self.tool_registry,
            telemetry=self.telemetry,
            enable_action_constraints=self.enable_action_constraints,
            communication_policy=self.communication_policy,
            communication_budget_tokens=self.communication_budget_tokens,
            communication_replay_buffer=self.communication_replay_buffer,
            replay_external_state_provider=self.replay_external_state_provider,
            graph_context_mode=self.graph_context_mode,
            evaluation_context=self.evaluation_context,
            multihop_sentence_limit=self.multihop_sentence_limit,
        )

    def merge_branch(
        self,
        source_branch_id: str,
        *,
        logical_ids: Optional[list[str]] = None,
    ) -> list[str]:
        """Merge selected branch outputs into this workflow's branch."""
        return self.store.merge_branch(
            self.task_id,
            source_branch_id,
            target_branch_id=self.branch_id,
            logical_ids=logical_ids,
        )

    def initial_state(self) -> WorkflowState:
        return {
            "task_id": self.task_id,
            "branch_id": self.branch_id,
            "task_type": self.task_type,
            "graph_revision": len(self.store.snapshot().nodes),
            "round_id": 0,
            "action_count": 0,
            "session_id": self.session_id,
            "status": "started",
            "error": None,
            "logs": [],
        }

    def _agent_node(self, role: str, *, mode: str = "normal") -> Callable[[WorkflowState], dict[str, Any]]:
        def node(state: WorkflowState) -> dict[str, Any]:
            started = time.time()
            logs = list(state.get("logs", []))
            last_action: dict[str, Any] = {}
            completed = False
            session_id = state.get("session_id", self.session_id)
            action_limit = 1 if self._is_hotpot_repair_mode(role, mode) else self.max_actions_per_role

            for action_index in range(action_limit):
                if not self._completion_errors(role, mode=mode):
                    completed = True
                    break
                view_store = self.agent_views.visible_store(role)
                role_view = self.agent_views.view(role)
                context_slice = build_context_slice(
                    view_store,
                    task_id=self.task_id,
                    branch_id=state.get("branch_id", self.branch_id),
                    role=role,
                    policy=(
                        "planner_state" if role == "planner"
                        else "solver_state" if role == "solver"
                        else "dependency_closure"
                    ),
                    allow_missing=False,
                    visible_fragment_ids=role_view.visible_fragment_ids,
                    multihop_sentence_limit=(
                        self.multihop_sentence_limit
                        if self.graph_context_mode == "role_aware"
                        else None
                    ),
                )
                graph_state = view_store.snapshot()
                prompt = self._render_graph_context(
                    context_slice,
                    graph_state,
                    role=role,
                    mode=mode,
                    action_index=action_index,
                )
                if self._is_hotpot_repair_mode(role, mode):
                    prompt = self._render_hotpot_repair_context(graph_state)
                session_prompt = self._session_prompt(role, mode, prompt)
                graph_read_context_tokens = self._token_count(prompt)
                logical_input_tokens = self._token_count(session_prompt)
                context_costs = self._context_cost_breakdown(
                    role=role,
                    mode=mode,
                    context_slice=context_slice,
                    graph_state=graph_state,
                    graph_read_context_tokens=graph_read_context_tokens,
                    logical_input_tokens=logical_input_tokens,
                )
                context_costs.update(self._rendered_graph_cost_breakdown(
                    role=role,
                    mode=mode,
                    action_index=action_index,
                    context_slice=context_slice,
                    graph_state=graph_state,
                ))
                # Semantic validation is part of generation repair, so make the
                # receiver-visible reference context available before asking the
                # model.  build_plan is side-effect free; the live commit still
                # happens exactly once below.
                self.compiler.set_node_ref_context(
                    self._node_ref_context(
                        context_slice,
                        visible_node_ids=role_view.visible_node_ids,
                        known_fragment_ids=role_view.visible_fragment_ids,
                        allowed_receivers=self._communication_receivers(role),
                    )
                )
                action_constraint = (
                    self._action_constraint(
                        role,
                        state.get("branch_id", self.branch_id),
                        mode=mode,
                        context_slice=context_slice,
                    )
                    if self.enable_action_constraints else None
                )
                if self._is_hotpot_repair_mode(role, mode):
                    diagnosis = self._latest_repair_diagnosis()
                    packet = diagnosis.get("repair_packet") if isinstance(diagnosis, dict) else {}
                    packet = packet if isinstance(packet, dict) else {}
                    repair_target = str(packet.get("repair_target") or diagnosis.get("error_location") or "").strip()
                    error_type = str(packet.get("error_type") or diagnosis.get("error_type") or "").strip()
                    if repair_target == "answer" or error_type in {"answer_not_supported", "answer_contradicted"}:
                        allowed_repair_actions = ("REVISE_ANSWER",)
                    elif repair_target == "bridge_entity":
                        allowed_repair_actions = ("PATCH_BRIDGE",)
                    elif repair_target == "cited_fact_ids" or error_type in {"invalid_citation", "missing_hop"}:
                        allowed_repair_actions = ("ADD_EVIDENCE", "REPLACE_EVIDENCE", "PATCH_CITATIONS")
                    else:
                        allowed_repair_actions = ("STOP",)
                    action_constraint = ActionConstraint(
                        role=role,
                        task_type=self.task_type,
                        allowed_ops=("repair_patch",),
                        reason="HotpotQA repair is a bounded local patch",
                        schema_mode="solver_repair",
                        allowed_citation_refs=(
                            action_constraint.allowed_citation_refs if action_constraint is not None else ()
                        ),
                        allowed_repair_actions=allowed_repair_actions,
                    )
                try:
                    action, generation = self._generate_action(
                        role,
                        mode,
                        prompt,
                        graph_state,
                        context_slice,
                        session_id=session_id,
                        session_prompt=session_prompt,
                        logical_input_tokens=logical_input_tokens,
                        graph_read_context_tokens=graph_read_context_tokens,
                        context_slice_tokens=context_slice.token_count,
                        context_costs=context_costs,
                        action_constraint=action_constraint,
                    )
                except ValueError as error:
                    if self._is_hotpot_repair_mode(role, mode):
                        self._raise_stage_failure(
                            f"repair_invalid_action: {error}",
                            state=state,
                            logs=logs,
                            session_id=session_id,
                            role=role,
                            last_action=last_action,
                        )
                    raise
                if self._is_hotpot_repair_mode(role, mode):
                    generation = {
                        **generation,
                        "hotpot_repair_packet_audit": dict(self._last_hotpot_repair_packet_audit),
                    }
                critic_verification_action = None
                typed_critic_verification = role == "critic" and action.get("op") in CRITIC_TYPED_OPS
                if typed_critic_verification:
                    verification_action = self.compiler.critic_verification_action(action)
                else:
                    verification_action = None
                if verification_action is not None:
                    model_verification_action = dict(verification_action)
                    verification_signal = verify_task_candidate(
                        self.task_type,
                        self.store,
                        task_id=self.task_id,
                        branch_id=state.get("branch_id", self.branch_id),
                    )
                    verification_action = apply_verification_signal(verification_action, verification_signal)
                    # The generic multiple-choice verifier can validate shape but
                    # has no gold label.  It must not erase a critic's independent
                    # disagreement with the solver's choice.
                    if (
                        typed_critic_verification
                        and self.task_type == "multiple_choice"
                        and action.get("op") == "audit_choice"
                    ):
                        # ChoiceVerifier only establishes structural validity;
                        # it deliberately returns uncertain without gold. Do
                        # not downgrade the Critic's claim audit with that
                        # structural signal. The kernel audit aggregation above
                        # is the sole source of the verification status.
                        verification_action = model_verification_action
                    if self.task_type == "multihop_qa":
                        verification_action = enforce_hotpot_entailment_verdict(
                            verification_action,
                            self.store,
                            task_id=self.task_id,
                            branch_id=state.get("branch_id", self.branch_id),
                        )
                    if typed_critic_verification:
                        critic_verification_action = verification_action
                    generation = {
                        **generation,
                        "task_verification": verification_signal.as_dict(),
                        "kernel_verification_status": verification_action.get("status", "uncertain"),
                    }
                try:
                    self.compiler.set_node_ref_context(
                        self._node_ref_context(
                            context_slice,
                            visible_node_ids=role_view.visible_node_ids,
                            known_fragment_ids=role_view.visible_fragment_ids,
                            allowed_receivers=self._communication_receivers(role),
                        )
                    )
                    result = (
                        self.compiler.compile_critic_verification(role, action, critic_verification_action)
                        if critic_verification_action is not None
                        else self.compiler.apply(role, action)
                    )
                except ActionCompilationError as error:
                    if self._is_hotpot_repair_mode(role, mode):
                        self._raise_stage_failure(
                            f"repair_compile_failed: {error}",
                            state=state,
                            logs=logs,
                            session_id=session_id,
                            role=role,
                            last_action=action,
                        )
                    if self.telemetry is not None:
                        self.telemetry.record_action(self._telemetry_action(
                            role=role,
                            mode=mode,
                            action_index=action_index,
                            action=action,
                            generation=generation,
                            session_id=session_id,
                            protocol_valid=True,
                            compile_success=False,
                            stage_complete=False,
                            stage_failed=True,
                            compile_error=str(error),
                        ))
                    self._rollback_model_session(session_id)
                    raise
                self.agent_views.grant_global_visibility()
                self.agent_views.grant(role, node_ids=result.node_ids, local=True)
                graph_update_tokens = self._graph_update_tokens(result.node_ids)
                generation = {
                    **generation,
                    "compiled_node_ids": tuple(result.node_ids),
                    "graph_update_tokens": graph_update_tokens,
                }
                communication_events = self._communicate_nodes(
                    sender=role,
                    node_ids=result.node_ids,
                    branch_id=state.get("branch_id", self.branch_id),
                    workflow_state=state,
                    decision_context={"mode": mode, "action_index": action_index},
                    model_send_hints=self._model_send_hints(
                        result.send_hints,
                        sender=role,
                        temp_to_node_id=result.temp_to_node_id,
                    ),
                )
                if result.done:
                    # STOP intentionally preserves the current result and does
                    # not create a newer result version. It still completes the
                    # repair stage; the critic gets the opportunity to re-check
                    # the unchanged state on the next graph edge.
                    if self._is_hotpot_repair_mode(role, mode) and action.get("op") == "repair_patch":
                        completion_errors = []
                    else:
                        completion_errors = self._completion_errors(role, mode=mode)
                    if completion_errors:
                        error = ActionCompilationError(
                            f"{role} cannot finish workflow stage: {'; '.join(completion_errors)}"
                        )
                        if self.telemetry is not None:
                            self.telemetry.record_action(self._telemetry_action(
                                role=role,
                                mode=mode,
                                action_index=action_index,
                                action=action,
                                generation=generation,
                                session_id=session_id,
                                protocol_valid=True,
                                compile_success=True,
                                stage_complete=False,
                                stage_failed=True,
                                compile_error=str(error),
                            ))
                        raise error
                    completed = True
                else:
                    last_action = action
                    if not self._completion_errors(role, mode=mode):
                        completed = True

                if self.telemetry is not None:
                    self.telemetry.record_action(self._telemetry_action(
                        role=role,
                        mode=mode,
                        action_index=action_index,
                        action=action,
                        generation=generation,
                        session_id=session_id,
                        protocol_valid=True,
                        compile_success=True,
                        stage_complete=completed,
                        stage_failed=False,
                    ))
                logs.append({
                    "role": role,
                    "mode": mode,
                    "action_index": action_index,
                    "action": action,
                    "compiled_node_ids": list(result.node_ids),
                    "graph_update_tokens": graph_update_tokens,
                    "communication": communication_events,
                    "context_tokens": generation.get("graph_read_context_tokens", 0),
                    "context_slice_tokens": generation.get("context_slice_tokens", 0),
                    "context_slice_node_ids": list(context_slice.visible_node_ids),
                    "context_slice_edge_ids": list(context_slice.visible_edge_ids),
                    "context_slice_fragment_ids": list(context_slice.visible_fragment_ids),
                    "context_slice_root_node_ids": list(context_slice.root_node_ids),
                    "rendered_context_node_ids": list(context_slice.visible_node_ids),
                    "rendered_context_fragment_ids": list(context_slice.visible_fragment_ids),
                    "logical_context_tokens": generation.get("logical_input_tokens", 0),
                    "generation": {
                        "attempts": generation.get("attempts", 1),
                        "validation_errors": list(generation.get("validation_errors", [])),
                        "hotpot_repair_packet_audit": generation.get("hotpot_repair_packet_audit"),
                    },
                    "telemetry": self._telemetry_action(
                        role=role,
                        mode=mode,
                        action_index=action_index,
                        action=action,
                        generation=generation,
                        session_id=session_id,
                        protocol_valid=True,
                        compile_success=True,
                        stage_complete=completed,
                        stage_failed=False,
                    ),
                    "session": self._session_snapshot(session_id),
                    "latency_sec": time.time() - started,
                })
                if completed:
                    break

            if not completed:
                if self._is_hotpot_repair_mode(role, mode):
                    self._raise_stage_failure(
                        self._hotpot_repair_noop_reason(last_action),
                        state=state,
                        logs=logs,
                        session_id=session_id,
                        role=role,
                        last_action=last_action,
                    )
                raise ActionCompilationError(
                    f"{role} exceeded max_actions_per_role={action_limit} without completing its typed boundary"
                )
            latest = self.store.snapshot()
            branch_id = state.get("branch_id", self.branch_id)
            return {
                "graph_revision": len(latest.nodes),
                "graph_snapshot_digest": self.store.branch_digest(self.task_id, branch_id),
                "round_id": state.get("round_id", 0) + (1 if role in {"solver", "final_solver"} else 0),
                "action_count": state.get("action_count", 0) + len(logs) - len(state.get("logs", [])),
                "status": "running",
                "last_role": role,
                "last_action": last_action,
                "session_id": session_id,
                "logs": logs,
            }

        return node

    def _deterministic_finalizer_node(self, state: WorkflowState) -> dict[str, Any]:
        """Copy the critic-verified value without another model invocation."""
        started = time.time()
        branch_id = state.get("branch_id", self.branch_id)
        verification = self.store.latest_valid(self.task_id, branch_id, "verification")
        if verification is None or verification.status != "verified":
            raise ActionCompilationError("deterministic finalizer requires a verified verification node")
        self.compiler.set_node_ref_context({})
        result = self.compiler.copy_verified_result()
        self.agent_views.grant_global_visibility()
        self.agent_views.grant("final_solver", node_ids=result.node_ids, local=True)
        logs = list(state.get("logs", []))
        logs.append({
            "role": "final_solver",
            "mode": "deterministic_finalization",
            "action_index": 0,
            "action": result.action,
            "compiled_node_ids": list(result.node_ids),
            "graph_update_tokens": self._graph_update_tokens(result.node_ids),
            "communication": [],
            "context_tokens": 0,
            "context_slice_tokens": 0,
            "context_slice_node_ids": [],
            "context_slice_edge_ids": [],
            "context_slice_fragment_ids": [],
            "context_slice_root_node_ids": [],
            "rendered_context_node_ids": [],
            "rendered_context_fragment_ids": [],
            "logical_context_tokens": 0,
            "generation": {"attempts": 0, "validation_errors": [], "model_invoked": False},
            "telemetry": {},
            "session": self._session_snapshot(state.get("session_id", self.session_id)),
            "latency_sec": time.time() - started,
        })
        latest = self.store.snapshot()
        return {
            "graph_revision": len(latest.nodes),
            "graph_snapshot_digest": self.store.branch_digest(self.task_id, branch_id),
            "round_id": state.get("round_id", 0) + 1,
            "action_count": state.get("action_count", 0) + 1,
            "status": "running",
            "last_role": "final_solver",
            "last_action": result.action,
            "session_id": state.get("session_id", self.session_id),
            "logs": logs,
        }

    def _is_hotpot_repair_mode(self, role: str, mode: str) -> bool:
        return role == "solver" and mode == "repair" and self.task_type == "multihop_qa"

    def _raise_stage_failure(
        self,
        message: str,
        *,
        state: WorkflowState,
        logs: list[dict[str, Any]],
        session_id: str,
        role: str,
        last_action: dict[str, Any],
    ) -> None:
        branch_id = state.get("branch_id", self.branch_id)
        latest = self.store.snapshot()
        error = ActionCompilationError(message)
        error.partial_state = {
            "graph_revision": len(latest.nodes),
            "graph_snapshot_digest": self.store.branch_digest(self.task_id, branch_id),
            "round_id": state.get("round_id", 0),
            "action_count": state.get("action_count", 0) + len(logs) - len(state.get("logs", [])),
            "status": "repair_failed",
            "error": message,
            "last_role": role,
            "last_action": last_action,
            "session_id": session_id,
            "logs": logs,
        }
        raise error

    def _generate_action(
        self,
        role: str,
        mode: str,
        prompt: str,
        graph_state: Any,
        context_slice: Any,
        *,
        session_id: str,
        session_prompt: str,
        logical_input_tokens: int,
        graph_read_context_tokens: int,
        context_slice_tokens: int,
        context_costs: dict[str, Any],
        action_constraint: Optional[ActionConstraint],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        attempts = 0
        validation_errors: list[str] = []
        call_metrics: list[dict[str, Any]] = []
        retry_input_tokens = 0
        retry_full_replay_equivalent_tokens = 0
        raw = self.model(ModelRequest(
            role=role,
            task_type=self.task_type,
            prompt=prompt,
            mode=mode,
            session_id=session_id,
            session_prompt=session_prompt,
            session_reset=True,
            session_rollback=False,
            action_constraint=action_constraint,
        ))
        call_metrics.append(self._last_call_metrics(session_id))
        for attempt in range(3):
            attempts = attempt + 1
            raw_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
            try:
                if isinstance(raw, dict):
                    action = self._normalize_generated_action(role, raw)
                    errors = self._validate_action_dict(role, action)
                    errors.extend(self._semantic_action_errors(role, action))
                    if action.get("op") == "repair_patch" and not self._is_hotpot_repair_mode(role, mode):
                        errors.append("repair_patch is only allowed in HotpotQA repair mode")
                    errors.extend(self._hotpot_repair_action_errors(role, mode, action))
                    if errors:
                        raise ValueError(f"invalid {role} Action: {'; '.join(errors)}")
                    return action, {
                        "attempts": attempts,
                        "raw_output": raw_text,
                        "validation_errors": validation_errors,
                        "call_metrics": call_metrics,
                        "logical_input_tokens": logical_input_tokens,
                        "graph_read_context_tokens": graph_read_context_tokens,
                        "context_slice_tokens": context_slice_tokens,
                        "retry_input_tokens": retry_input_tokens,
                        "retry_full_replay_equivalent_tokens": retry_full_replay_equivalent_tokens,
                        "retry_saved_input_tokens": max(0, retry_full_replay_equivalent_tokens - retry_input_tokens),
                        **context_costs,
                        "action_constraint": action_constraint.as_dict() if action_constraint else None,
                    }
                action = self._parse_generated_action(str(raw).strip(), role=role)
                semantic_errors = self._semantic_action_errors(role, action)
                if semantic_errors:
                    raise ValueError(f"invalid {role} Action: {'; '.join(semantic_errors)}")
                if action.get("op") == "repair_patch" and not self._is_hotpot_repair_mode(role, mode):
                    raise ValueError("invalid solver Action: repair_patch is only allowed in HotpotQA repair mode")
                repair_errors = self._hotpot_repair_action_errors(role, mode, action)
                if repair_errors:
                    raise ValueError(f"invalid {role} Action: {'; '.join(repair_errors)}")
                return action, {
                    "attempts": attempts,
                    "raw_output": raw_text,
                    "validation_errors": validation_errors,
                    "call_metrics": call_metrics,
                    "logical_input_tokens": logical_input_tokens,
                    "graph_read_context_tokens": graph_read_context_tokens,
                    "context_slice_tokens": context_slice_tokens,
                    "retry_input_tokens": retry_input_tokens,
                    "retry_full_replay_equivalent_tokens": retry_full_replay_equivalent_tokens,
                    "retry_saved_input_tokens": max(0, retry_full_replay_equivalent_tokens - retry_input_tokens),
                    **context_costs,
                    "action_constraint": action_constraint.as_dict() if action_constraint else None,
                }
            except ValueError as error:
                validation_errors.append(str(error))
                if attempt == 2:
                    if self.telemetry is not None:
                        self.telemetry.record_generation_error({
                            "role": role,
                            "mode": mode,
                            "attempts": attempts,
                            "validation_errors": validation_errors,
                            "call_metrics": call_metrics,
                            "logical_input_tokens": logical_input_tokens,
                            "graph_read_context_tokens": graph_read_context_tokens,
                            "context_slice_tokens": context_slice_tokens,
                            "retry_input_tokens": retry_input_tokens,
                            "retry_full_replay_equivalent_tokens": retry_full_replay_equivalent_tokens,
                            "retry_saved_input_tokens": max(0, retry_full_replay_equivalent_tokens - retry_input_tokens),
                            **context_costs,
                            "raw_output": raw_text,
                        })
                    raise ValueError(
                        f"{role} failed strict Action generation after 3 attempts: {error}"
                    ) from error
                retry_prompt = self._standalone_retry_session_prompt(
                    role,
                    mode,
                    error,
                    raw_text,
                    context_slice=context_slice,
                    graph_state=graph_state,
                )
                retry_tokens = self._token_count(retry_prompt)
                full_replay_tokens = self._token_count(
                    session_prompt + raw_text + self._retry_session_prompt(role, mode, error)
                )
                retry_input_tokens += retry_tokens
                retry_full_replay_equivalent_tokens += full_replay_tokens
                logical_input_tokens += retry_tokens
                raw = self.model(ModelRequest(
                    role=role,
                    task_type=self.task_type,
                    prompt=prompt,
                    mode=mode,
                    session_id=session_id,
                    session_prompt=retry_prompt,
                    session_reset=True,
                    session_rollback=False,
                    action_constraint=action_constraint,
                ))
                call_metrics.append(self._last_call_metrics(session_id))
        raise AssertionError("unreachable")

    def _parse_generated_action(self, text: str, *, role: str) -> dict[str, Any]:
        from .protocol import normalize_action_payload

        candidate = text.strip()
        if candidate.startswith("```") and candidate.endswith("```"):
            lines = candidate.splitlines()
            if len(lines) >= 3 and lines[0].strip().lower() in {"```", "```json"}:
                candidate = "\n".join(lines[1:-1]).strip()
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"{role} output must be exactly one Action JSON object") from exc
        action = self._normalize_generated_action(role, normalize_action_payload(value))
        errors = self._validate_action_dict(role, action)
        if errors:
            raise ValueError(f"invalid {role} Action: {'; '.join(errors)}")
        return action

    def _hotpot_repair_action_errors(self, role: str, mode: str, action: Any) -> list[str]:
        if not self._is_hotpot_repair_mode(role, mode):
            return []
        if not isinstance(action, dict):
            return ["HotpotQA repair must emit exactly one repair_patch Action"]
        if action.get("op") == "repair_patch":
            return self._hotpot_repair_patch_errors(action)
        return ["HotpotQA repair mode allows only repair_patch"]

    def _hotpot_repair_patch_errors(self, action: dict[str, Any]) -> list[str]:
        allowed_actions = {
            "STOP",
            "ADD_EVIDENCE",
            "REPLACE_EVIDENCE",
            "PATCH_CITATIONS",
            "PATCH_BRIDGE",
            "REVISE_ANSWER",
        }
        action_name = str(action.get("action", "")).strip()
        patch = action.get("patch")
        if action_name not in allowed_actions:
            return [f"HotpotQA repair_patch action must be one of {', '.join(sorted(allowed_actions))}"]
        if not isinstance(patch, dict):
            return ["HotpotQA repair_patch.patch must be an object"]
        if action_name == "STOP":
            if patch:
                return ["HotpotQA STOP repair_patch must have an empty patch"]
            return []
        diagnosis = self._latest_repair_diagnosis()
        packet = diagnosis.get("repair_packet") if isinstance(diagnosis, dict) else {}
        packet = packet if isinstance(packet, dict) else {}
        error_type = str(packet.get("error_type") or diagnosis.get("error_type") or "").strip()
        error_location = str(diagnosis.get("error_location") or packet.get("error_location") or "").strip()
        if action_name == "REVISE_ANSWER" and error_type not in {"answer_not_supported", "answer_contradicted"}:
            return [f"HotpotQA {error_type or 'structural'} repair cannot revise answer"]
        if action_name == "PATCH_BRIDGE" and not (
            error_location == "bridge_entity" or packet.get("repair_target") == "bridge_entity"
        ):
            return ["HotpotQA PATCH_BRIDGE does not match the critic repair target"]
        if action_name in {"ADD_EVIDENCE", "REPLACE_EVIDENCE", "PATCH_CITATIONS"} and not (
            error_location == "cited_fact_ids"
            or packet.get("repair_target") == "cited_fact_ids"
            or error_type in {"invalid_citation", "missing_hop"}
        ):
            return ["HotpotQA evidence patch does not match the critic repair target"]
        if action_name in {"ADD_EVIDENCE", "REPLACE_EVIDENCE", "PATCH_CITATIONS"}:
            selected = patch.get("selected_fact_ids")
            if not isinstance(selected, list) or not selected or not all(str(item).strip() for item in selected):
                return [f"HotpotQA {action_name} requires non-empty selected_fact_ids"]
        if action_name == "PATCH_BRIDGE" and not str(patch.get("bridge_entity", "")).strip():
            return ["HotpotQA PATCH_BRIDGE requires bridge_entity"]
        if action_name == "REVISE_ANSWER" and not str(patch.get("answer", "")).strip():
            return ["HotpotQA REVISE_ANSWER requires answer"]
        if action_name != "REVISE_ANSWER" and patch.get("preserve_answer") is False:
            return ["HotpotQA structural repair must preserve answer"]
        return []

    def _hotpot_repair_must_change_errors(self, value: dict[str, Any]) -> list[str]:
        diagnosis = self._latest_repair_diagnosis()
        packet = diagnosis.get("repair_packet") if isinstance(diagnosis, dict) else {}
        if not isinstance(packet, dict):
            return []
        previous = self._latest_result_value()
        if not isinstance(previous, dict):
            return []
        target = str(packet.get("repair_target", "") or "").strip()
        error_type = str(packet.get("error_type") or diagnosis.get("error_type") or "").strip()
        error_location = str(diagnosis.get("error_location") or packet.get("error_location") or "").strip()
        must_change = packet.get("must_change") if isinstance(packet.get("must_change"), dict) else {}
        forbidden = packet.get("forbidden_values") if isinstance(packet.get("forbidden_values"), dict) else {}
        preserve = [
            str(item).strip().lower()
            for item in [
                *self._hotpot_string_list(diagnosis.get("preserve")),
                *self._hotpot_string_list(packet.get("preserve")),
            ]
            if str(item).strip()
        ]
        errors: list[str] = []

        def changed(field: str) -> bool:
            old_value = previous.get(field)
            new_value = value.get(field)
            if isinstance(old_value, list) or isinstance(new_value, list):
                return [str(item) for item in (old_value or [])] != [str(item) for item in (new_value or [])]
            return str(old_value or "").strip() != str(new_value or "").strip()

        if (target == "bridge_entity" or must_change.get("bridge_entity")) and not changed("bridge_entity"):
            errors.append("HotpotQA repair must change bridge_entity")
        if (
            target == "cited_fact_ids"
            or must_change.get("cited_fact_ids")
            or error_type == "invalid_citation"
        ) and not changed("cited_fact_ids"):
            errors.append("HotpotQA repair must add or replace cited_fact_ids")
        if (target == "answer" or must_change.get("answer")) and not changed("answer"):
            errors.append("HotpotQA repair must change answer")

        answer_change_allowed = error_type in {"answer_not_supported", "answer_contradicted"}
        if changed("answer") and not answer_change_allowed:
            errors.append(
                "HotpotQA repair must preserve answer unless error_type is "
                "answer_not_supported or answer_contradicted"
            )

        bridge_only_repair = target == "bridge_entity" or error_location == "bridge_entity"
        citation_only_repair = (
            target == "cited_fact_ids"
            or error_location == "cited_fact_ids"
            or error_type in {"invalid_citation", "missing_hop"}
        )
        if bridge_only_repair and changed("cited_fact_ids"):
            errors.append("HotpotQA bridge_entity repair must preserve cited_fact_ids")
        if citation_only_repair and changed("bridge_entity"):
            errors.append("HotpotQA cited_fact_ids repair must preserve bridge_entity")

        if any("answer" in item for item in preserve) and changed("answer"):
            errors.append("HotpotQA repair violated preserve constraint for answer")
        preserve_bridge = any(
            "bridge_entity" in item or "bridge entity" in item
            for item in preserve
        )
        if preserve_bridge and changed("bridge_entity"):
            errors.append("HotpotQA repair violated preserve constraint for bridge_entity")
        preserve_all_citations = any(
            "cited_fact_ids" in item or item in {"citations", "cited facts"}
            for item in preserve
        )
        if preserve_all_citations and changed("cited_fact_ids"):
            errors.append("HotpotQA repair violated preserve constraint for cited_fact_ids")
        preserve_reasoning = any(
            "reasoning_chain" in item or "reasoning chain" in item
            for item in preserve
        )
        if preserve_reasoning and changed("reasoning_chain"):
            errors.append("HotpotQA repair violated preserve constraint for reasoning_chain")

        known = packet.get("known") if isinstance(packet.get("known"), dict) else {}
        valid_citations = self._hotpot_string_list(known.get("valid_cited_fact_ids"))
        if valid_citations and any("valid citation" in item for item in preserve):
            new_citations = set(self._hotpot_string_list(value.get("cited_fact_ids")))
            dropped = [citation for citation in valid_citations if citation not in new_citations]
            if dropped:
                errors.append(
                    "HotpotQA repair violated preserve constraint for valid citations: "
                    + ", ".join(dropped)
                )

        forbidden_bridge = {str(item).strip().lower() for item in forbidden.get("bridge_entity", [])} if isinstance(forbidden.get("bridge_entity"), list) else set()
        if forbidden_bridge and str(value.get("bridge_entity", "")).strip().lower() in forbidden_bridge:
            errors.append("HotpotQA repair bridge_entity repeats a forbidden value")
        return errors

    def _normalize_generated_action(self, role: str, payload: Any) -> Any:
        del role
        return payload

    def _last_call_metrics(self, session_id: str) -> dict[str, Any]:
        snapshot = self._session_snapshot(session_id)
        metrics = snapshot.get("last_call_metrics", {}) if isinstance(snapshot, dict) else {}
        return dict(metrics) if isinstance(metrics, dict) else {}

    def _token_count(self, text: str) -> int:
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except TypeError:
                return len(self.tokenizer.encode(text))
        return len(text.split())

    def _context_cost_breakdown(
        self,
        *,
        role: str,
        mode: str,
        context_slice: Any,
        graph_state: Any,
        graph_read_context_tokens: int,
        logical_input_tokens: int,
    ) -> dict[str, Any]:
        """Separate reusable task prefix from policy-controlled context.

        The current local Transformers backend is stateless, so physical input
        still includes the full prompt on every call.  These fields expose the
        logical communication cost separately from the prefix/prefill cost that
        a stateful backend or KV reuse layer could avoid in later rounds.
        """
        persistent_node_ids: list[str] = []
        persistent_context_tokens = 0
        for node_id in context_slice.visible_node_ids:
            node = graph_state.nodes.get(node_id)
            if node is None:
                continue
            if (
                node.logical_id in MANDATORY_LOGICAL_IDS
                or node.provenance.get("communication_scope") == "MANDATORY"
                or node.provenance.get("visibility_class") == "public"
            ):
                persistent_node_ids.append(node_id)
                persistent_context_tokens += estimate_node_tokens(node)
        persistent_context_tokens = min(
            max(0, persistent_context_tokens),
            max(0, graph_read_context_tokens),
        )
        incremental_context_tokens = max(0, graph_read_context_tokens - persistent_context_tokens)
        system_prompt_tokens = self._token_count(self._system_contract(role, mode))
        reusable_prefix_tokens = min(
            max(0, logical_input_tokens),
            max(0, system_prompt_tokens + persistent_context_tokens),
        )
        prompt_wrapper_tokens = max(
            0,
            logical_input_tokens - graph_read_context_tokens - system_prompt_tokens,
        )
        return {
            "persistent_context_tokens": persistent_context_tokens,
            "incremental_context_tokens": incremental_context_tokens,
            "logical_communication_tokens": incremental_context_tokens,
            "system_prompt_tokens": system_prompt_tokens,
            "prompt_wrapper_tokens": prompt_wrapper_tokens,
            "reusable_prefix_tokens": reusable_prefix_tokens,
            "persistent_context_node_ids": tuple(persistent_node_ids),
            "reusable_prefix_key": f"{role}:{mode}:{self.task_type}",
        }

    def _render_graph_context(
        self,
        context_slice: Any,
        graph_state: Any,
        *,
        role: str = "",
        mode: str = "",
        action_index: int = 0,
    ) -> str:
        if self.graph_context_mode == "baseline":
            rendered = render_compact_context_slice(context_slice, graph_state)
            return self._append_choice_audit_manifest(rendered, role)
        if self.graph_context_mode in {"source_state_split", "role_aware"}:
            rendered, _ = self._render_source_state_split_graph_context(
                role=role,
                mode=mode,
                action_index=action_index,
                context_slice=context_slice,
                graph_state=graph_state,
                update_seen=False,
            )
            return self._append_choice_audit_manifest(rendered, role)
        rendered, _ = self._render_deduplicated_graph_context(
            role="",
            mode="",
            action_index=0,
            context_slice=context_slice,
            graph_state=graph_state,
            update_seen=False,
        )
        return self._append_choice_audit_manifest(rendered, role)

    def _append_choice_audit_manifest(self, rendered: str, role: str) -> str:
        if role != "critic" or self.task_type != "multiple_choice":
            return rendered
        manifest = build_choice_audit_manifest(self.store, task_id=self.task_id, branch_id=self.branch_id)
        if manifest is None:
            return rendered
        feedback = ""
        verification = self.store.latest_valid(self.task_id, self.branch_id, "verification")
        if (
            verification is not None
            and verification.status == "uncertain"
            and isinstance(verification.content, Mapping)
            and verification.content.get("error_type") == "invalid_choice_audit_reason"
        ):
            raw_reason = verification.content.get("reason", "")
            packet: Any = {}
            if isinstance(raw_reason, str) and raw_reason.strip().startswith("{"):
                try:
                    parsed = json.loads(raw_reason)
                except json.JSONDecodeError:
                    parsed = {}
                if isinstance(parsed, Mapping):
                    packet = parsed.get("repair_packet", {})
            feedback = (
                "\nCRITIC REPAIR PACKET (kernel-generated; fix ONLY these reasons, preserve all statuses and claim IDs):\n"
                + json.dumps(packet, ensure_ascii=False, sort_keys=True)
            )
        return (
            rendered
            + "\nCRITIC INPUT (kernel-assembled; source and Solver candidate are the audit object):\n"
            + json.dumps({
                "source": {
                    "question": manifest.get("original_question", ""),
                    "options": {str(item.get("option", "")): item.get("option_text", "") for item in manifest.get("elimination_claims", ()) if isinstance(item, Mapping)},
                },
                "solver_candidate": {
                    "answer": manifest.get("selected_answer", ""),
                    "slot_reasons": [
                        {"claim_id": item.get("claim_id"), "claim_text": item.get("claim_text", ""), "slot": item.get("slot"), "selected_term": item.get("selected_term", ""), "solver_reason": item.get("solver_reason", "")}
                        for item in manifest.get("slot_claims", ()) if isinstance(item, Mapping)
                    ],
                    "option_reviews": [
                        {"claim_id": item.get("claim_id"), "claim_text": item.get("claim_text", ""), "option": item.get("option", ""), "solver_assessment": item.get("solver_assessment", ""), "solver_reason": item.get("solver_reason", ""), "option_text": item.get("option_text", "")}
                        for item in manifest.get("elimination_claims", ()) if isinstance(item, Mapping)
                    ],
                },
                "audit_manifest": {
                    "candidate_version": manifest.get("candidate_version", ""),
                    "manifest_digest": manifest.get("manifest_digest", ""),
                    "required_claim_ids": [item.get("claim_id") for item in (*manifest.get("slot_claims", ()), *manifest.get("elimination_claims", ())) if isinstance(item, Mapping)],
                },
                "kernel_control": {
                    "required_action": "audit_choice",
                    "must_cover_all_claims": True,
                    "preserve_fields": ["claim_id", "solver_assessment", "solver_reason", "solver_term"],
                },
            }, ensure_ascii=False, sort_keys=True)
            + "\nAUDIT MANIFEST (kernel-owned, immutable; audit claim IDs exactly):\n"
            + json.dumps(manifest, ensure_ascii=False, sort_keys=True)
            + feedback
        )

    @staticmethod
    def _graph_context_category(node_type: str) -> str:
        if node_type in {
            "task",
            "query_spec",
            "facts",
            "fact",
            "requirements",
            "test",
            "choice",
            "entity",
            "supporting_fact",
            "evidence_link",
            "table",
            "table_cell",
            "evidence",
            "task_source",
            "question_source",
            "document_source",
            "sentence_source",
            "entity_source",
            "choice_source",
            "choice_schema",
            "requirements_source",
            "entry_point_source",
            "public_test_source",
        }:
            return "source_payload"
        if node_type in {"error", "tool_request", "tool_result", "execution"}:
            return "history_or_tool"
        return "derived_reasoning"

    def _role_aware_source_ref_only(self, *, role: str, mode: str, node_type: str) -> bool:
        if self.graph_context_mode != "role_aware":
            return False
        # Finalization should ground the answer in result/verification state;
        # source payload is kept addressable but not repeatedly expanded.
        return role == "final_solver"

    def _role_aware_state_ref_only(self, *, role: str, mode: str, node_type: str) -> bool:
        if self.graph_context_mode != "role_aware":
            return False
        if role == "planner":
            return True
        if role == "final_solver":
            return node_type not in {"result", "verification", "final_answer", "code"}
        if role == "critic":
            return node_type in {"plan", "plan_steps"}
        return False

    @staticmethod
    def _rendered_node_content_with_refs(node: Any, fragments: set[str], local_id: str) -> Any:
        rendered = render_fragmented_content(node, fragments)
        if node.type != "supporting_fact":
            return rendered
        if isinstance(rendered, dict):
            return {
                "citation_id": node.logical_id,
                "local_node_id": local_id,
                **rendered,
            }
        return {
            "citation_id": node.logical_id,
            "local_node_id": local_id,
            "text": rendered,
        }

    def _render_deduplicated_graph_context(
        self,
        *,
        role: str,
        mode: str,
        action_index: int,
        context_slice: Any,
        graph_state: Any,
        update_seen: bool,
    ) -> tuple[str, dict[str, Any]]:
        token_by_type = {
            "task": "T",
            "query_spec": "Q",
            "facts": "F",
            "fact": "F",
            "plan": "P",
            "plan_steps": "PS",
            "calculation": "C",
            "result": "R",
            "verification": "V",
            "final_answer": "A",
            "task_source": "TS",
            "question_source": "QS",
            "document_source": "DS",
            "sentence_source": "SS",
            "entity_source": "ES",
            "choice_source": "CS",
            "requirements_source": "RS",
            "entry_point_source": "EP",
            "public_test_source": "PT",
            "plan_step": "P1",
            "candidate_answer": "CA",
            "slot_analysis": "SA",
            "selected_evidence": "SE",
            "bridge_hypothesis": "BH",
            "reasoning_step": "R2",
            "semantic_delta_boundary": "DB",
        }
        nodes = [graph_state.nodes[node_id] for node_id in context_slice.visible_node_ids if node_id in graph_state.nodes]
        nodes.sort(key=_node_sort_key)
        fragments_by_node: dict[str, set[str]] = {}
        for fragment_id in context_slice.visible_fragment_ids:
            node_id = str(fragment_id).split("#", 1)[0]
            fragments_by_node.setdefault(node_id, set()).add(str(fragment_id))
        local_ids = {node.node_id: f"n{index}" for index, node in enumerate(nodes, start=1)}
        lines = ["<G>"]
        content_tokens = 0
        wrapper_tokens = self._token_count("<G>\n</G>")
        edge_tokens = 0
        content_by_type: dict[str, int] = {}
        wrapper_by_type: dict[str, int] = {}
        content_by_category: dict[str, int] = {}
        wrapper_by_category: dict[str, int] = {}
        source_duplicate_in_prompt_tokens = 0
        source_cross_call_reread_tokens = 0
        source_deduplicated_prompt_saved_tokens = 0
        prompt_seen_source_hashes: set[str] = set()
        first_source_local_id_by_hash: dict[str, str] = {}
        prior_seen_source_hashes = set(self._graph_source_seen_hashes)
        source_records: list[dict[str, Any]] = []
        call_id = f"{self.task_id}:{self.branch_id}:{role}:{mode}:{action_index}"

        for node in nodes:
            tag = token_by_type.get(node.type, "N")
            category = self._graph_context_category(node.type)
            rendered_content = self._rendered_node_content_with_refs(
                node,
                fragments_by_node.get(node.node_id, set()),
                local_ids[node.node_id],
            )
            content = json.dumps(rendered_content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            original_content_count = self._token_count(content)
            content_hash = ""
            duplicate_in_prompt = False
            cross_call_reread = False
            deduplicated = False
            if category == "source_payload":
                content_hash = str(node.content_digest or "").strip()
                if not content_hash:
                    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                duplicate_in_prompt = content_hash in prompt_seen_source_hashes
                cross_call_reread = content_hash in prior_seen_source_hashes
                if duplicate_in_prompt:
                    source_duplicate_in_prompt_tokens += original_content_count
                prompt_seen_source_hashes.add(content_hash)

            if (
                self.graph_context_mode == "deduplicated"
                and category == "source_payload"
                and duplicate_in_prompt
            ):
                deduplicated = True
                source_deduplicated_prompt_saved_tokens += original_content_count
                ref = first_source_local_id_by_hash.get(content_hash, "")
                line = f"<{tag} id={local_ids[node.node_id]} v={node.version} s={node.status} ref={ref}/>"
                content_count = 0
                line_count = self._token_count(line)
                wrapper_count = line_count
            else:
                line = f"<{tag} id={local_ids[node.node_id]} v={node.version} s={node.status}>{content}</{tag}>"
                content_count = original_content_count
                line_count = self._token_count(line)
                wrapper_count = max(0, line_count - content_count)
                if category == "source_payload" and cross_call_reread:
                    source_cross_call_reread_tokens += content_count
            if category == "source_payload" and content_hash and content_hash not in first_source_local_id_by_hash:
                first_source_local_id_by_hash[content_hash] = local_ids[node.node_id]

            lines.append(line)
            content_tokens += content_count
            wrapper_tokens += wrapper_count
            content_by_type[node.type] = content_by_type.get(node.type, 0) + content_count
            wrapper_by_type[node.type] = wrapper_by_type.get(node.type, 0) + wrapper_count
            content_by_category[category] = content_by_category.get(category, 0) + content_count
            wrapper_by_category[category] = wrapper_by_category.get(category, 0) + wrapper_count
            if category == "source_payload":
                source_records.append({
                    "task_id": self.task_id,
                    "call_id": call_id,
                    "agent_role": role,
                    "mode": mode,
                    "action_index": action_index,
                    "source_id": node.node_id,
                    "source_logical_id": node.logical_id,
                    "source_type": node.type,
                    "content_hash": content_hash,
                    "content_tokens": content_count,
                    "original_content_tokens": original_content_count,
                    "duplicate_in_prompt": duplicate_in_prompt,
                    "cross_call_reread": cross_call_reread and not deduplicated,
                    "deduplicated_in_prompt": deduplicated,
                    "dedup_ref": first_source_local_id_by_hash.get(content_hash, ""),
                })

        if update_seen:
            self._graph_source_seen_hashes.update(prompt_seen_source_hashes)
        visible_edges = set(context_slice.visible_edge_ids)
        for edge in graph_state.edges:
            if edge.edge_id not in visible_edges:
                continue
            if edge.source not in local_ids or edge.target not in local_ids:
                continue
            line = f"<E a={local_ids[edge.source]} r={edge.relation} b={local_ids[edge.target]}/>"
            lines.append(line)
            edge_tokens += self._token_count(line)
        lines.append("</G>")
        return "\n".join(lines), {
            "graph_context_content_tokens": content_tokens,
            "graph_context_wrapper_tokens": wrapper_tokens,
            "graph_context_edge_tokens": edge_tokens,
            "graph_context_content_tokens_by_type": content_by_type,
            "graph_context_wrapper_tokens_by_type": wrapper_by_type,
            "graph_context_content_tokens_by_category": content_by_category,
            "graph_context_wrapper_tokens_by_category": wrapper_by_category,
            "graph_source_duplicate_in_prompt_tokens": (
                0 if self.graph_context_mode == "deduplicated" else source_duplicate_in_prompt_tokens
            ),
            "graph_source_duplicate_in_prompt_original_tokens": source_duplicate_in_prompt_tokens,
            "graph_source_deduplicated_prompt_saved_tokens": source_deduplicated_prompt_saved_tokens,
            "graph_source_cross_call_reread_tokens": source_cross_call_reread_tokens,
            "graph_role_aware_source_ref_saved_tokens": 0,
            "graph_role_aware_state_ref_saved_tokens": 0,
            "graph_source_context_records": source_records,
        }

    def _render_source_state_split_graph_context(
        self,
        *,
        role: str,
        mode: str,
        action_index: int,
        context_slice: Any,
        graph_state: Any,
        update_seen: bool,
    ) -> tuple[str, dict[str, Any]]:
        token_by_type = {
            "task": "T",
            "query_spec": "Q",
            "facts": "F",
            "fact": "F",
            "plan": "P",
            "plan_steps": "PS",
            "calculation": "C",
            "result": "R",
            "verification": "V",
            "final_answer": "A",
            "task_source": "TS",
            "question_source": "QS",
            "document_source": "DS",
            "sentence_source": "SS",
            "entity_source": "ES",
            "choice_source": "CS",
            "requirements_source": "RS",
            "entry_point_source": "EP",
            "public_test_source": "PT",
            "plan_step": "P1",
            "candidate_answer": "CA",
            "slot_analysis": "SA",
            "selected_evidence": "SE",
            "bridge_hypothesis": "BH",
            "reasoning_step": "R2",
            "semantic_delta_boundary": "DB",
        }
        nodes = [graph_state.nodes[node_id] for node_id in context_slice.visible_node_ids if node_id in graph_state.nodes]
        nodes.sort(key=_node_sort_key)
        fragments_by_node: dict[str, set[str]] = {}
        for fragment_id in context_slice.visible_fragment_ids:
            node_id = str(fragment_id).split("#", 1)[0]
            fragments_by_node.setdefault(node_id, set()).add(str(fragment_id))
        local_ids = {node.node_id: f"n{index}" for index, node in enumerate(nodes, start=1)}
        source_nodes = [node for node in nodes if self._graph_context_category(node.type) == "source_payload"]
        state_nodes = [node for node in nodes if self._graph_context_category(node.type) != "source_payload"]
        lines = ["<G>", "<SRC>"]
        content_tokens = 0
        wrapper_tokens = self._token_count("<G>\n<SRC>\n</SRC>\n<STATE>\n</STATE>\n<REF>\n</REF>\n</G>")
        edge_tokens = 0
        content_by_type: dict[str, int] = {}
        wrapper_by_type: dict[str, int] = {}
        content_by_category: dict[str, int] = {}
        wrapper_by_category: dict[str, int] = {}
        source_duplicate_in_prompt_tokens = 0
        source_cross_call_reread_tokens = 0
        source_deduplicated_prompt_saved_tokens = 0
        role_aware_source_ref_saved_tokens = 0
        role_aware_state_ref_saved_tokens = 0
        prompt_seen_source_hashes: set[str] = set()
        first_source_local_id_by_hash: dict[str, str] = {}
        prior_seen_source_hashes = set(self._graph_source_seen_hashes)
        source_records: list[dict[str, Any]] = []
        call_id = f"{self.task_id}:{self.branch_id}:{role}:{mode}:{action_index}"

        for node in source_nodes:
            rendered_content = self._rendered_node_content_with_refs(
                node,
                fragments_by_node.get(node.node_id, set()),
                local_ids[node.node_id],
            )
            content = json.dumps(rendered_content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            original_content_count = self._token_count(content)
            content_hash = str(node.content_digest or "").strip()
            if not content_hash:
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            duplicate_in_prompt = content_hash in prompt_seen_source_hashes
            cross_call_reread = content_hash in prior_seen_source_hashes
            if duplicate_in_prompt:
                source_duplicate_in_prompt_tokens += original_content_count
            prompt_seen_source_hashes.add(content_hash)
            source_ref_only = self._role_aware_source_ref_only(role=role, mode=mode, node_type=node.type)
            if source_ref_only:
                role_aware_source_ref_saved_tokens += original_content_count
                ref = first_source_local_id_by_hash.get(content_hash, "")
                line = f"{local_ids[node.node_id]} {node.type} {node.logical_id} source_ref hash={content_hash[:12]} ref={ref}"
                content_count = 0
                line_count = self._token_count(line)
                wrapper_count = line_count
                active_reread = False
                deduplicated = False
            elif duplicate_in_prompt:
                source_deduplicated_prompt_saved_tokens += original_content_count
                ref = first_source_local_id_by_hash.get(content_hash, "")
                line = f"{local_ids[node.node_id]} {node.type} {node.logical_id} ref={ref}"
                content_count = 0
                line_count = self._token_count(line)
                wrapper_count = line_count
                active_reread = False
                deduplicated = True
            else:
                line = f"{local_ids[node.node_id]} {node.type} {node.logical_id} {content}"
                content_count = original_content_count
                line_count = self._token_count(line)
                wrapper_count = max(0, line_count - content_count)
                active_reread = cross_call_reread
                if active_reread:
                    source_cross_call_reread_tokens += content_count
                deduplicated = False
            if content_hash not in first_source_local_id_by_hash:
                first_source_local_id_by_hash[content_hash] = local_ids[node.node_id]
            lines.append(line)
            content_tokens += content_count
            wrapper_tokens += wrapper_count
            content_by_type[node.type] = content_by_type.get(node.type, 0) + content_count
            wrapper_by_type[node.type] = wrapper_by_type.get(node.type, 0) + wrapper_count
            content_by_category["source_payload"] = content_by_category.get("source_payload", 0) + content_count
            wrapper_by_category["source_payload"] = wrapper_by_category.get("source_payload", 0) + wrapper_count
            source_records.append({
                "task_id": self.task_id,
                "call_id": call_id,
                "agent_role": role,
                "mode": mode,
                "action_index": action_index,
                "source_id": node.node_id,
                "source_logical_id": node.logical_id,
                "source_type": node.type,
                "content_hash": content_hash,
                "content_tokens": content_count,
                "original_content_tokens": original_content_count,
                "duplicate_in_prompt": duplicate_in_prompt,
                "cross_call_reread": active_reread,
                "deduplicated_in_prompt": deduplicated,
                "role_aware_ref": source_ref_only,
                "dedup_ref": first_source_local_id_by_hash.get(content_hash, ""),
            })

        lines.extend(["</SRC>", "<STATE>"])
        for node in state_nodes:
            tag = token_by_type.get(node.type, "N")
            category = self._graph_context_category(node.type)
            rendered_content = self._rendered_node_content_with_refs(
                node,
                fragments_by_node.get(node.node_id, set()),
                local_ids[node.node_id],
            )
            content = json.dumps(rendered_content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            content_count = self._token_count(content)
            if self._role_aware_state_ref_only(role=role, mode=mode, node_type=node.type):
                role_aware_state_ref_saved_tokens += content_count
                line = f"<{tag} id={local_ids[node.node_id]} v={node.version} s={node.status} ref=state/>"
                line_count = self._token_count(line)
                wrapper_count = line_count
                content_count = 0
            else:
                line = f"<{tag} id={local_ids[node.node_id]} v={node.version} s={node.status}>{content}</{tag}>"
                line_count = self._token_count(line)
                wrapper_count = max(0, line_count - content_count)
            lines.append(line)
            content_tokens += content_count
            wrapper_tokens += wrapper_count
            content_by_type[node.type] = content_by_type.get(node.type, 0) + content_count
            wrapper_by_type[node.type] = wrapper_by_type.get(node.type, 0) + wrapper_count
            content_by_category[category] = content_by_category.get(category, 0) + content_count
            wrapper_by_category[category] = wrapper_by_category.get(category, 0) + wrapper_count

        if update_seen:
            self._graph_source_seen_hashes.update(prompt_seen_source_hashes)
        lines.extend(["</STATE>", "<REF>"])
        visible_edges = set(context_slice.visible_edge_ids)
        for edge in graph_state.edges:
            if edge.edge_id not in visible_edges:
                continue
            if edge.source not in local_ids or edge.target not in local_ids:
                continue
            line = f"{local_ids[edge.source]} {edge.relation} {local_ids[edge.target]}"
            lines.append(line)
            edge_tokens += self._token_count(line)
        lines.extend(["</REF>", "</G>"])
        return "\n".join(lines), {
            "graph_context_content_tokens": content_tokens,
            "graph_context_wrapper_tokens": wrapper_tokens,
            "graph_context_edge_tokens": edge_tokens,
            "graph_context_content_tokens_by_type": content_by_type,
            "graph_context_wrapper_tokens_by_type": wrapper_by_type,
            "graph_context_content_tokens_by_category": content_by_category,
            "graph_context_wrapper_tokens_by_category": wrapper_by_category,
            "graph_source_duplicate_in_prompt_tokens": 0,
            "graph_source_duplicate_in_prompt_original_tokens": source_duplicate_in_prompt_tokens,
            "graph_source_deduplicated_prompt_saved_tokens": source_deduplicated_prompt_saved_tokens,
            "graph_source_cross_call_reread_tokens": source_cross_call_reread_tokens,
            "graph_role_aware_source_ref_saved_tokens": role_aware_source_ref_saved_tokens,
            "graph_role_aware_state_ref_saved_tokens": role_aware_state_ref_saved_tokens,
            "graph_source_context_records": source_records,
        }

    def _rendered_graph_cost_breakdown(
        self,
        *,
        role: str,
        mode: str,
        action_index: int,
        context_slice: Any,
        graph_state: Any,
    ) -> dict[str, Any]:
        """Tokenize compact graph rendering into content, wrapper, and edge costs."""
        if self.graph_context_mode == "source_state_split":
            _, costs = self._render_source_state_split_graph_context(
                role=role,
                mode=mode,
                action_index=action_index,
                context_slice=context_slice,
                graph_state=graph_state,
                update_seen=True,
            )
            return costs
        if self.graph_context_mode == "role_aware":
            _, costs = self._render_source_state_split_graph_context(
                role=role,
                mode=mode,
                action_index=action_index,
                context_slice=context_slice,
                graph_state=graph_state,
                update_seen=True,
            )
            return costs
        if self.graph_context_mode == "deduplicated":
            _, costs = self._render_deduplicated_graph_context(
                role=role,
                mode=mode,
                action_index=action_index,
                context_slice=context_slice,
                graph_state=graph_state,
                update_seen=True,
            )
            return costs
        token_by_type = {
            "task": "T",
            "query_spec": "Q",
            "facts": "F",
            "fact": "F",
            "plan": "P",
            "plan_steps": "PS",
            "calculation": "C",
            "result": "R",
            "verification": "V",
            "final_answer": "A",
            "task_source": "TS",
            "question_source": "QS",
            "document_source": "DS",
            "sentence_source": "SS",
            "entity_source": "ES",
            "choice_source": "CS",
            "requirements_source": "RS",
            "entry_point_source": "EP",
            "public_test_source": "PT",
            "plan_step": "P1",
            "candidate_answer": "CA",
            "slot_analysis": "SA",
            "selected_evidence": "SE",
            "bridge_hypothesis": "BH",
            "reasoning_step": "R2",
            "semantic_delta_boundary": "DB",
        }
        nodes = [graph_state.nodes[node_id] for node_id in context_slice.visible_node_ids if node_id in graph_state.nodes]
        nodes.sort(key=_node_sort_key)
        fragments_by_node: dict[str, set[str]] = {}
        for fragment_id in context_slice.visible_fragment_ids:
            node_id = str(fragment_id).split("#", 1)[0]
            fragments_by_node.setdefault(node_id, set()).add(str(fragment_id))
        local_ids = {node.node_id: f"n{index}" for index, node in enumerate(nodes, start=1)}
        content_tokens = 0
        wrapper_tokens = self._token_count("<G>\n</G>")
        edge_tokens = 0
        content_by_type: dict[str, int] = {}
        wrapper_by_type: dict[str, int] = {}
        category_by_type = {
            "task": "source_payload",
            "query_spec": "source_payload",
            "facts": "source_payload",
            "fact": "source_payload",
            "requirements": "source_payload",
            "test": "source_payload",
            "choice": "source_payload",
            "entity": "source_payload",
            "supporting_fact": "source_payload",
            "evidence_link": "source_payload",
            "table": "source_payload",
            "table_cell": "source_payload",
            "evidence": "source_payload",
            "error": "history_or_tool",
            "tool_request": "history_or_tool",
            "tool_result": "history_or_tool",
            "execution": "history_or_tool",
        }
        content_by_category: dict[str, int] = {}
        wrapper_by_category: dict[str, int] = {}
        source_duplicate_in_prompt_tokens = 0
        source_cross_call_reread_tokens = 0
        prompt_seen_source_hashes: set[str] = set()
        prior_seen_source_hashes = set(self._graph_source_seen_hashes)
        source_records: list[dict[str, Any]] = []
        call_id = f"{self.task_id}:{self.branch_id}:{role}:{mode}:{action_index}"
        for node in nodes:
            tag = token_by_type.get(node.type, "N")
            rendered_content = self._rendered_node_content_with_refs(
                node,
                fragments_by_node.get(node.node_id, set()),
                local_ids[node.node_id],
            )
            content = json.dumps(rendered_content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            content_count = self._token_count(content)
            line = f"<{tag} id={local_ids[node.node_id]} v={node.version} s={node.status}>{content}</{tag}>"
            line_count = self._token_count(line)
            wrapper_count = max(0, line_count - content_count)
            category = category_by_type.get(node.type, "derived_reasoning")
            content_tokens += content_count
            wrapper_tokens += wrapper_count
            content_by_type[node.type] = content_by_type.get(node.type, 0) + content_count
            wrapper_by_type[node.type] = wrapper_by_type.get(node.type, 0) + wrapper_count
            content_by_category[category] = content_by_category.get(category, 0) + content_count
            wrapper_by_category[category] = wrapper_by_category.get(category, 0) + wrapper_count
            if category == "source_payload":
                content_hash = str(node.content_digest or "").strip()
                if not content_hash:
                    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                duplicate_in_prompt = content_hash in prompt_seen_source_hashes
                cross_call_reread = content_hash in prior_seen_source_hashes
                if duplicate_in_prompt:
                    source_duplicate_in_prompt_tokens += content_count
                if cross_call_reread:
                    source_cross_call_reread_tokens += content_count
                prompt_seen_source_hashes.add(content_hash)
                source_records.append({
                    "task_id": self.task_id,
                    "call_id": call_id,
                    "agent_role": role,
                    "mode": mode,
                    "action_index": action_index,
                    "source_id": node.node_id,
                    "source_logical_id": node.logical_id,
                    "source_type": node.type,
                    "content_hash": content_hash,
                    "content_tokens": content_count,
                    "duplicate_in_prompt": duplicate_in_prompt,
                    "cross_call_reread": cross_call_reread,
                })
        self._graph_source_seen_hashes.update(prompt_seen_source_hashes)
        visible_edges = set(context_slice.visible_edge_ids)
        for edge in graph_state.edges:
            if edge.edge_id not in visible_edges:
                continue
            if edge.source not in local_ids or edge.target not in local_ids:
                continue
            edge_tokens += self._token_count(
                f"<E a={local_ids[edge.source]} r={edge.relation} b={local_ids[edge.target]}/>"
            )
        return {
            "graph_context_content_tokens": content_tokens,
            "graph_context_wrapper_tokens": wrapper_tokens,
            "graph_context_edge_tokens": edge_tokens,
            "graph_context_content_tokens_by_type": content_by_type,
            "graph_context_wrapper_tokens_by_type": wrapper_by_type,
            "graph_context_content_tokens_by_category": content_by_category,
            "graph_context_wrapper_tokens_by_category": wrapper_by_category,
            "graph_source_duplicate_in_prompt_tokens": source_duplicate_in_prompt_tokens,
            "graph_source_duplicate_in_prompt_original_tokens": source_duplicate_in_prompt_tokens,
            "graph_source_deduplicated_prompt_saved_tokens": 0,
            "graph_source_cross_call_reread_tokens": source_cross_call_reread_tokens,
            "graph_source_context_records": source_records,
        }

    def _graph_update_tokens(self, node_ids: Sequence[str]) -> int:
        if not node_ids:
            return 0
        state = self.store.snapshot()
        selected = set(node_ids)
        total = 0
        for node_id in selected:
            node = state.nodes.get(node_id)
            if node is not None:
                total += estimate_node_tokens(node)
        for edge in state.edges:
            if edge.source in selected or edge.target in selected:
                total += estimate_edge_tokens(edge)
        return total

    def _action_constraint(
        self,
        role: str,
        branch_id: str,
        *,
        mode: str = "normal",
        context_slice: Any = None,
    ) -> ActionConstraint:
        graph_state = self.store.snapshot()
        source_refs: list[str] = []
        choice_labels: list[str] = []
        citation_refs: list[str] = []
        required_audit_slots: list[int] = []
        required_audit_options: list[str] = []
        required_audit_claim_ids: list[str] = []
        required_audit_elimination_claim_ids: list[str] = []
        required_candidate_version = ""
        required_manifest_digest = ""
        forbidden_choice_labels: list[str] = []
        required_solver_slots: list[int] = []
        if role == "planner" and context_slice is not None:
            visible_ids = set(getattr(context_slice, "visible_node_ids", ()) or ())
            local_ids = dict(getattr(context_slice, "local_node_id_by_node", {}) or {})
            public_source_types = get_task_spec(self.task_type).public_source_types
            for node_id in getattr(context_slice, "visible_node_ids", ()) or ():
                node = graph_state.nodes.get(node_id)
                if (
                    node is None
                    or node_id not in visible_ids
                    or node.type not in public_source_types
                    or not node_visible_to_role(node, role)
                ):
                    continue
                local_id = str(local_ids.get(node_id, "")).strip()
                if local_id:
                    source_refs.append(local_id)
                if node.logical_id:
                    source_refs.append(str(node.logical_id))
        if role in {"solver", "critic"} and context_slice is not None:
            local_ids = dict(getattr(context_slice, "local_node_id_by_node", {}) or {})
            for node_id in getattr(context_slice, "visible_node_ids", ()) or ():
                node = graph_state.nodes.get(node_id)
                if node is None or not node_visible_to_role(node, role):
                    continue
                if node.type in {"choice_schema", "choice_source"}:
                    try:
                        content = json.loads(node.content) if isinstance(node.content, str) else node.content
                    except (TypeError, json.JSONDecodeError):
                        content = {}
                    if isinstance(content, Mapping):
                        labels = content.get("allowed_labels", ())
                        if isinstance(labels, (list, tuple)):
                            choice_labels.extend(str(item).strip().upper() for item in labels if str(item).strip())
                        label = str(content.get("label", "")).strip().upper()
                        if label:
                            choice_labels.append(label)
                if role == "solver" and node.type == "sentence_source":
                    local_id = str(local_ids.get(node_id, "")).strip()
                    if local_id:
                        citation_refs.append(local_id)
                    if node.logical_id:
                        citation_refs.append(str(node.logical_id))
        # The Critic audit scope is a kernel-owned manifest derived from the
        # latest Solver candidate.  This prevents constrained decoding from
        # legally closing arrays before all claims have been audited.
        if role == "critic" and self.task_type == "multiple_choice":
            manifest = build_choice_audit_manifest(self.store, task_id=self.task_id, branch_id=branch_id)
            if manifest is not None:
                required_candidate_version = str(manifest.get("candidate_version", ""))
                required_manifest_digest = str(manifest.get("manifest_digest", ""))
                required_audit_claim_ids = [str(item.get("claim_id")) for item in manifest.get("slot_claims", ()) if isinstance(item, Mapping) and item.get("claim_id")]
                required_audit_elimination_claim_ids = [str(item.get("claim_id")) for item in manifest.get("elimination_claims", ()) if isinstance(item, Mapping) and item.get("claim_id")]
            result = self.store.latest_valid(self.task_id, branch_id, "result")
            value = result.content if result is not None else {}
            if isinstance(value, Mapping) and "value" in value:
                value = value.get("value")
            if isinstance(value, Mapping):
                for item in value.get("slot_analysis", ()):
                    if isinstance(item, Mapping) and isinstance(item.get("slot"), int) and not isinstance(item.get("slot"), bool):
                        required_audit_slots.append(int(item["slot"]))
                eliminated = value.get("eliminated_options", {})
                if isinstance(eliminated, Mapping):
                    required_audit_options.extend(
                        str(label).strip().upper() for label in eliminated if str(label).strip()
                    )
        if role == "solver" and mode == "repair" and self.task_type == "multiple_choice":
            verification = self.store.latest_valid(self.task_id, branch_id, "verification")
            content = verification.content if verification is not None and isinstance(verification.content, Mapping) else {}
            reason = content.get("reason", "") if isinstance(content, Mapping) else ""
            if isinstance(reason, str) and reason.strip().startswith("{"):
                try:
                    packet_wrapper = json.loads(reason)
                except json.JSONDecodeError:
                    packet_wrapper = {}
                packet = packet_wrapper.get("repair_packet", {}) if isinstance(packet_wrapper, Mapping) else {}
                if isinstance(packet, Mapping):
                    forbidden_choice_labels.extend(
                        str(item).strip().upper()
                        for item in packet.get("forbidden_answers", ())
                        if str(item).strip()
                    )
        if role == "solver" and self.task_type == "multiple_choice":
            # The task adapter owns the structural shape.  MMLU-Pro fill-in
            # questions expose one or more underscore spans; ordinary
            # multiple-choice questions still receive one explicit slot.
            task_node = graph_state.nodes.get("task@v1") or graph_state.nodes.get("task")
            task_text = ""
            if task_node is not None:
                task_text = task_node.content if isinstance(task_node.content, str) else json.dumps(task_node.content, ensure_ascii=False)
            slot_count = len(re.findall(r"_{2,}", task_text)) or 1
            schema_node = graph_state.nodes.get("choice_schema@v1") or graph_state.nodes.get("choice_schema")
            if schema_node is not None:
                schema_content = schema_node.content
                if isinstance(schema_content, str):
                    try:
                        schema_content = json.loads(schema_content)
                    except json.JSONDecodeError:
                        schema_content = {}
                slot_values = schema_content.get("slot_values", {}) if isinstance(schema_content, Mapping) else {}
                if isinstance(slot_values, Mapping):
                    manifest_count = max(
                        (len(values) for values in slot_values.values() if isinstance(values, (list, tuple))),
                        default=0,
                    )
                    slot_count = manifest_count or slot_count
            slot_count = min(max(slot_count, 1), 12)
            required_solver_slots = list(range(1, slot_count + 1))
        return build_action_constraint(
            role=role,
            task_type=self.task_type,
            missing=self._completion_errors(role, mode=mode),
            allowed_source_refs=source_refs,
            allowed_choice_labels=choice_labels,
            allowed_citation_refs=citation_refs,
            required_audit_slots=required_audit_slots,
            required_audit_options=required_audit_options,
            required_audit_claim_ids=required_audit_claim_ids,
            required_audit_elimination_claim_ids=required_audit_elimination_claim_ids,
            required_candidate_version=required_candidate_version,
            required_manifest_digest=required_manifest_digest,
            forbidden_choice_labels=forbidden_choice_labels,
            required_solver_slots=required_solver_slots,
        )

    def _telemetry_action(
        self,
        *,
        role: str,
        mode: str,
        action_index: int,
        action: dict[str, Any],
        generation: dict[str, Any],
        session_id: str,
        protocol_valid: bool,
        compile_success: bool,
        stage_complete: bool,
        stage_failed: bool,
        compile_error: Optional[str] = None,
    ) -> dict[str, Any]:
        session = self._session_snapshot(session_id)
        latest_backend = session.get("last_call_metrics", {}) if isinstance(session, dict) else {}
        call_metrics = [item for item in generation.get("call_metrics", []) if item]
        backend: dict[str, Any] = {}
        for key in (
            "physical_input_tokens", "output_tokens", "forward_calls",
        ):
            backend[key] = sum(int(item.get(key, 0) or 0) for item in call_metrics)
        if not call_metrics:
            backend.update(latest_backend)
        logical_input = int(generation.get("logical_input_tokens", 0))
        physical_input = int(backend.get(
            "physical_input_tokens",
            logical_input,
        ))
        output_tokens = int(backend.get("output_tokens", 0))
        if output_tokens <= 0:
            output_tokens = max(1, len(json.dumps(action, ensure_ascii=False, separators=(",", ":")).split()))
        event = {
            "role": role,
            "mode": mode,
            "action_index": action_index,
            "action_type": action.get("type", action.get("op")),
            "action": dict(action),
            "attempts": int(generation.get("attempts", 1)),
            "validation_errors": list(generation.get("validation_errors", [])),
            "action_constraint": generation.get("action_constraint"),
            "logical_input_tokens": logical_input,
            "physical_input_tokens": physical_input,
            "physical_llm_input_tokens": physical_input,
            "output_tokens": output_tokens,
            "prefill_cost_tokens": physical_input,
            "decode_cost_tokens": output_tokens,
            "forward_calls": int(backend.get("forward_calls", 0)),
            "graph_read_context_tokens": int(generation.get("graph_read_context_tokens", 0) or 0),
            "context_slice_tokens": int(generation.get("context_slice_tokens", 0) or 0),
            "retry_input_tokens": int(generation.get("retry_input_tokens", 0) or 0),
            "retry_full_replay_equivalent_tokens": int(
                generation.get("retry_full_replay_equivalent_tokens", 0) or 0
            ),
            "retry_saved_input_tokens": int(generation.get("retry_saved_input_tokens", 0) or 0),
            "persistent_context_tokens": int(generation.get("persistent_context_tokens", 0) or 0),
            "incremental_context_tokens": int(generation.get("incremental_context_tokens", 0) or 0),
            "logical_communication_tokens": int(generation.get("logical_communication_tokens", 0) or 0),
            "system_prompt_tokens": int(generation.get("system_prompt_tokens", 0) or 0),
            "prompt_wrapper_tokens": int(generation.get("prompt_wrapper_tokens", 0) or 0),
            "reusable_prefix_tokens": int(generation.get("reusable_prefix_tokens", 0) or 0),
            "persistent_context_node_ids": list(generation.get("persistent_context_node_ids", ())),
            "graph_context_content_tokens": int(generation.get("graph_context_content_tokens", 0) or 0),
            "graph_context_wrapper_tokens": int(generation.get("graph_context_wrapper_tokens", 0) or 0),
            "graph_context_edge_tokens": int(generation.get("graph_context_edge_tokens", 0) or 0),
            "graph_context_content_tokens_by_type": dict(
                generation.get("graph_context_content_tokens_by_type", {}) or {}
            ),
            "graph_context_wrapper_tokens_by_type": dict(
                generation.get("graph_context_wrapper_tokens_by_type", {}) or {}
            ),
            "graph_context_content_tokens_by_category": dict(
                generation.get("graph_context_content_tokens_by_category", {}) or {}
            ),
            "graph_context_wrapper_tokens_by_category": dict(
                generation.get("graph_context_wrapper_tokens_by_category", {}) or {}
            ),
            "graph_source_duplicate_in_prompt_tokens": int(
                generation.get("graph_source_duplicate_in_prompt_tokens", 0) or 0
            ),
            "graph_source_duplicate_in_prompt_original_tokens": int(
                generation.get("graph_source_duplicate_in_prompt_original_tokens", 0) or 0
            ),
            "graph_source_deduplicated_prompt_saved_tokens": int(
                generation.get("graph_source_deduplicated_prompt_saved_tokens", 0) or 0
            ),
            "graph_source_cross_call_reread_tokens": int(
                generation.get("graph_source_cross_call_reread_tokens", 0) or 0
            ),
            "graph_role_aware_source_ref_saved_tokens": int(
                generation.get("graph_role_aware_source_ref_saved_tokens", 0) or 0
            ),
            "graph_role_aware_state_ref_saved_tokens": int(
                generation.get("graph_role_aware_state_ref_saved_tokens", 0) or 0
            ),
            "graph_source_context_records": list(generation.get("graph_source_context_records", []) or []),
            "reusable_prefix_key": generation.get("reusable_prefix_key", f"{role}:{mode}:{self.task_type}"),
            "graph_update_tokens": int(generation.get("graph_update_tokens", 0) or 0),
            "task_verification": generation.get("task_verification"),
            "kernel_verification_status": generation.get("kernel_verification_status"),
            "hotpot_repair_packet_audit": generation.get("hotpot_repair_packet_audit"),
            "protocol_valid": protocol_valid,
            "compile_success": compile_success,
            "stage_complete": stage_complete,
            "stage_failed": stage_failed,
            "compile_error": compile_error,
            "raw_output": generation.get("raw_output", ""),
        }
        return event

    def _model_send_hints(
        self,
        send_hints: Sequence[Mapping[str, Any]],
        *,
        sender: str,
        temp_to_node_id: Optional[Mapping[str, str]] = None,
    ) -> dict[str, tuple[str, ...]]:
        """Resolve model temp references into receiver-relative fragments."""
        if not send_hints:
            return {}
        state = self.store.snapshot()
        mapping = dict(temp_to_node_id or {})
        result: dict[str, list[str]] = {}
        for hint in send_hints:
            receiver = str(hint.get("receiver", "")).strip()
            if not receiver:
                continue
            for raw_fragment in hint.get("fragments", ()):
                fragment = str(raw_fragment)
                base, separator, suffix = fragment.partition("#")
                node_id = mapping.get(base, base)
                node = state.nodes.get(node_id)
                if node is None:
                    node = self.store.latest_valid(self.task_id, self.branch_id, node_id)
                if node is None:
                    continue
                if separator:
                    result.setdefault(receiver, []).append(f"{node_id}#{suffix}")
                    continue
                candidates = fragments_for_node(
                    node,
                    sender=sender,
                    receiver=receiver,
                    node_scope="MANDATORY_STAGE",
                )
                optional = [item.fragment_id for item in candidates if item.scope == OPTIONAL_FRAGMENT]
                result.setdefault(receiver, []).extend(optional or [node_id])
        return {receiver: tuple(dict.fromkeys(fragments)) for receiver, fragments in result.items()}

    def _communicate_nodes(
        self,
        *,
        sender: str,
        node_ids: Sequence[str],
        branch_id: str,
        workflow_state: Optional[Mapping[str, Any]] = None,
        decision_context: Optional[Mapping[str, Any]] = None,
        model_send_hints: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> list[dict[str, Any]]:
        if not node_ids:
            return []
        state = self.store.snapshot()
        sender_view = self.agent_views.view(sender)
        events: list[dict[str, Any]] = []
        for receiver in self._communication_receivers(sender):
            receiver_view = self.agent_views.view(receiver)
            receiver_visible_before = sorted(receiver_view.visible_node_ids)
            receiver_visible_before_fragments = sorted(receiver_view.visible_fragment_ids)
            candidates = extract_delta_candidates(
                state,
                node_ids=node_ids,
                sender=sender,
                receiver=receiver,
                receiver_view=receiver_view,
            )
            mandatory_candidates = [
                candidate for candidate in candidates
                if candidate.exportable and candidate.scope == "MANDATORY_STAGE" and candidate.novelty_score > 0
            ]
            optional_candidates = [
                candidate for candidate in candidates
                if candidate.exportable and candidate.scope == "OPTIONAL_POLICY"
            ]
            fragment_candidates = self._fragment_candidates(
                state,
                candidates=candidates,
                sender=sender,
                receiver=receiver,
            )
            decision_snapshot_id = self._record_communication_decision(
                state=state,
                sender=sender,
                receiver=receiver,
                branch_id=branch_id,
                candidates=candidates,
                fragments=fragment_candidates,
                workflow_state=workflow_state,
                decision_context=decision_context,
            )
            mandatory_fragment_ids = [
                fragment.fragment_id for fragment in fragment_candidates
                if fragment.scope == MANDATORY_FRAGMENT
            ]
            optional_fragment_ids = [
                fragment.fragment_id for fragment in fragment_candidates
                if fragment.scope == OPTIONAL_FRAGMENT
            ]
            fragment_token_by_id = {
                fragment.fragment_id: fragment.token_cost for fragment in fragment_candidates
            }
            fragment_scope_by_id = {
                fragment.fragment_id: fragment.scope for fragment in fragment_candidates
            }
            candidate_tokens = sum(candidate.token_cost for candidate in candidates)
            mandatory_tokens = sum(candidate.token_cost for candidate in mandatory_candidates)
            optional_tokens = sum(candidate.token_cost for candidate in optional_candidates)
            semantic_event: dict[str, Any] = {}
            model_hint_ids = tuple(str(item) for item in (model_send_hints or {}).get(receiver, ()))
            fallback_delta = None
            targeted_deltas: list[GraphDelta] = []
            targeted_fragment_level_by_id: dict[str, str] = {}
            budget_rejected_fragment_ids: list[str] = []
            semantic_minimal_policy = self.communication_policy.name in {
                "minimal_no_feedback",
                "minimal_sendall_fallback",
                "minimal_targeted_feedback",
                "receiver_aware_targeted_feedback",
            }
            if semantic_minimal_policy:
                contract = default_semantic_contract(sender, receiver, task_type=self.task_type)
                innovation_decisions = detect_innovations(
                    state,
                    sender=sender,
                    receiver=receiver,
                    receiver_view=receiver_view,
                    node_ids=[candidate.node_id for candidate in candidates],
                )
                packet = build_initial_semantic_packet(
                    state,
                    sender=sender,
                    receiver=receiver,
                    contract=contract,
                    innovation_decisions=innovation_decisions,
                )
                packet_roots = list(packet.root_node_ids)
                optional_roots = [
                    candidate.node_id for candidate in optional_candidates
                    if candidate.node_id in set(packet_roots)
                ]
                mandatory_roots = [
                    candidate.node_id for candidate in mandatory_candidates
                    if candidate.node_id not in receiver_view.visible_node_ids
                ]
                selected_optional_fragment_ids: list[str] = []
                roots = list(dict.fromkeys([*mandatory_roots, *packet_roots]))
                packet = packet.with_roots(roots)
                semantic_event = {
                    "semantic_policy": self.communication_policy.name,
                    "semantic_packet_id": packet.packet_id,
                    "semantic_packet_level": packet.level,
                    "semantic_packet_round_index": packet.round_index,
                    "semantic_packet_type": packet.packet_type,
                    "semantic_packet_initial": packet.is_initial,
                    "semantic_packet_fallback": packet.is_fallback,
                    "semantic_packet_root_node_ids": list(packet.root_node_ids),
                    "semantic_packet_closure_node_ids": list(packet.closure_node_ids),
                    "semantic_packet_edge_ids": list(packet.edge_ids),
                    "semantic_packet_payload_node_ids": list(packet.payload_node_ids),
                    "semantic_packet_target_requirements": list(packet.target_requirements),
                    "semantic_packet_tokens": packet.token_cost,
                    "semantic_innovation_decisions": [decision.as_dict() for decision in innovation_decisions],
                    "fallback_send_all": False,
                    "fallback_tokens": 0,
                    "fallback_node_ids": [],
                    "wasted_pre_fallback_tokens": 0,
                    "targeted_refinement": False,
                    "targeted_refinement_tokens": 0,
                    "targeted_refinement_node_ids": [],
                    "targeted_refinement_plan": None,
                    "targeted_refinement_feedback": None,
                }
            else:
                hinted_for_receiver = tuple(
                    str(item) for item in (model_send_hints or {}).get(receiver, ())
                )
                mandatory_roots = [candidate.node_id for candidate in mandatory_candidates]
                if self.communication_policy.name == "model_selected":
                    valid_hinted_fragment_ids = list(dict.fromkeys(
                        fragment_id for fragment_id in hinted_for_receiver
                        if fragment_id in fragment_scope_by_id
                        and fragment_scope_by_id[fragment_id] == OPTIONAL_FRAGMENT
                        and fragment_id not in receiver_view.visible_fragment_ids
                    ))
                    selected_optional_fragment_ids = []
                    if self.communication_budget_tokens is None:
                        selected_optional_fragment_ids = valid_hinted_fragment_ids
                    else:
                        base_delta = dependency_closure(
                            state,
                            root_node_ids=mandatory_roots,
                            sender=sender,
                            receiver=receiver,
                            receiver_view=receiver_view,
                            policy=self.communication_policy.name,
                            candidate_token_cost=candidate_tokens,
                        )
                        base_cost = base_delta.token_cost
                        for fragment_id in valid_hinted_fragment_ids:
                            trial_fragments = [*selected_optional_fragment_ids, fragment_id]
                            trial_roots = list(dict.fromkeys([
                                *mandatory_roots,
                                *(node_id_from_fragment(item) for item in trial_fragments),
                            ]))
                            trial_delta = dependency_closure(
                                state,
                                root_node_ids=trial_roots,
                                sender=sender,
                                receiver=receiver,
                                receiver_view=receiver_view,
                                policy=self.communication_policy.name,
                                candidate_token_cost=candidate_tokens,
                            )
                            optional_marginal_cost = max(0, trial_delta.token_cost - base_cost)
                            if optional_marginal_cost <= self.communication_budget_tokens:
                                selected_optional_fragment_ids.append(fragment_id)
                            else:
                                budget_rejected_fragment_ids.append(fragment_id)
                else:
                    selected_optional_fragment_ids = self.communication_policy.select_roots(
                    state=state,
                    sender_view=sender_view,
                    receiver_view=receiver_view,
                    candidates=[
                        self._fragment_delta_candidate(
                            fragment,
                            sender=sender,
                            receiver=receiver,
                        )
                        for fragment in fragment_candidates
                        if fragment.scope == OPTIONAL_FRAGMENT
                    ],
                    task=self.task_id,
                    budget_tokens=self.communication_budget_tokens,
                )
                optional_roots = [
                    node_id for node_id in dict.fromkeys(
                        node_id_from_fragment(fragment_id)
                        for fragment_id in selected_optional_fragment_ids
                    )
                    if node_id in state.nodes
                ]
                roots = list(dict.fromkeys([*mandatory_roots, *optional_roots]))
            selected_optional_tokens = sum(
                int(fragment_token_by_id.get(fragment_id, 0) or 0)
                for fragment_id in selected_optional_fragment_ids
            )
            if self.communication_policy.include_dependency_closure:
                delta = dependency_closure(
                    state,
                    root_node_ids=roots,
                    sender=sender,
                    receiver=receiver,
                    receiver_view=receiver_view,
                    policy=self.communication_policy.name,
                    candidate_token_cost=candidate_tokens,
                )
            else:
                delta = self._root_only_delta(
                    state=state,
                    root_node_ids=roots,
                    sender=sender,
                    receiver=receiver,
                    receiver_view=receiver_view,
                    candidate_token_cost=candidate_tokens,
                )
            if delta.node_ids or delta.edge_ids:
                if semantic_minimal_policy:
                    packet = packet.with_delta(delta)
                self.agent_views.grant(
                    receiver,
                    node_ids=delta.node_ids,
                    edge_ids=delta.edge_ids,
                    fragment_ids=self._delivery_fragment_ids(
                        state,
                        sender=sender,
                        receiver=receiver,
                        node_ids=delta.node_ids,
                        mandatory_root_node_ids=mandatory_roots,
                        selected_optional_fragment_ids=selected_optional_fragment_ids,
                    ),
                    delta_id=f"{sender}->{receiver}:{len(receiver_view.received_delta_ids) + 1}",
                    packet_id=packet.packet_id if semantic_minimal_policy else "",
                )
            if semantic_minimal_policy:
                semantic_event.update({
                    "semantic_packet_closure_node_ids": list(packet.closure_node_ids),
                    "semantic_packet_edge_ids": list(packet.edge_ids),
                    "semantic_packet_payload_node_ids": list(packet.payload_node_ids),
                    "semantic_packet_tokens": packet.token_cost,
                    "semantic_packet_type": packet.packet_type,
                })
                check_results = SemanticResolver(state, receiver_view).check_contract(contract)
                feedback = build_semantic_feedback(
                    sender=sender,
                    receiver=receiver,
                    round_index=0,
                    results=check_results,
                )
                initial_receiver_need = diagnose_receiver_need(
                    sender=sender,
                    receiver=receiver,
                    stage=f"{sender}->{receiver}",
                    round_index=0,
                    results=check_results,
                )
                initial_feedback = feedback
                fallback_feedback = None
                targeted_feedback = None
                fallback_receiver_need = None
                targeted_receiver_need = None
                targeted_plan = None
                targeted_plans: list[dict[str, Any]] = []
                targeted_feedbacks: list[dict[str, Any]] = []
                targeted_receiver_needs: list[dict[str, Any]] = []
                targeted_requested_fragment_ids: list[str] = []
                feedback_decisions: list[dict[str, Any]] = []
                targeted_refinement_skipped = False
                targeted_refinement_skip_reason = ""
                targeted_refinement_skipped_plan = None
                refinement_rounds = 0
                max_refinement_rounds = self.max_rounds
                feedback_decision = decide_feedback_action(
                    feedback,
                    policy_name=self.communication_policy.name,
                    round_index=refinement_rounds,
                    max_rounds=max_refinement_rounds,
                )
                feedback_decisions.append(feedback_decision.as_dict())
                while (
                    feedback_decision.action == FeedbackAction.TARGETED_REFINEMENT
                    and self.communication_policy.name in {"minimal_targeted_feedback", "receiver_aware_targeted_feedback"}
                ):
                    targeted_plan = plan_refinement(
                        state,
                        sender=sender,
                        receiver=receiver,
                        sender_view=sender_view,
                        receiver_view=receiver_view,
                        missing_semantics=nack_missing_semantics(feedback),
                    )
                    targeted_plans.append(targeted_plan.as_dict())
                    targeted_requested_fragment_ids.extend(targeted_plan.requested_fragment_ids)
                    for fragment_id in targeted_plan.requested_fragment_ids:
                        targeted_fragment_level_by_id[str(fragment_id)] = targeted_plan.request_level
                    if targeted_plan.is_unresolvable or targeted_plan.is_empty:
                        feedback_decision = decide_feedback_action(
                            feedback,
                            policy_name=self.communication_policy.name,
                            round_index=max_refinement_rounds,
                            max_rounds=max_refinement_rounds,
                        )
                        feedback_decisions.append(feedback_decision.as_dict())
                        break
                    if (
                        targeted_plan.request_level == "quality"
                        and self.communication_budget_tokens is not None
                        and targeted_plan.estimated_cost > self.communication_budget_tokens
                    ):
                        targeted_refinement_skipped = True
                        targeted_refinement_skip_reason = "quality_refinement_budget_exceeded"
                        targeted_refinement_skipped_plan = targeted_plan.as_dict()
                        break
                    targeted_delta = dependency_closure(
                        state,
                        root_node_ids=targeted_plan.root_node_ids,
                        sender=sender,
                        receiver=receiver,
                        receiver_view=receiver_view,
                        policy=f"{self.communication_policy.name}:targeted_refinement:r{refinement_rounds + 1}",
                        candidate_token_cost=candidate_tokens,
                    )
                    targeted_deltas.append(targeted_delta)
                    if targeted_delta.node_ids or targeted_delta.edge_ids:
                        targeted_packet_id = (
                            f"packet:{sender}->{receiver}:r{refinement_rounds + 1}:targeted:"
                            f"{'-'.join(targeted_plan.root_node_ids) or 'empty'}"
                        )
                        self.agent_views.grant(
                            receiver,
                            node_ids=targeted_delta.node_ids,
                            edge_ids=targeted_delta.edge_ids,
                            fragment_ids=self._delivery_fragment_ids(
                                state,
                                sender=sender,
                                receiver=receiver,
                                node_ids=targeted_delta.node_ids,
                                mandatory_root_node_ids=targeted_plan.root_node_ids,
                                selected_optional_fragment_ids=targeted_plan.requested_fragment_ids,
                            ),
                            delta_id=f"{sender}->{receiver}:{len(receiver_view.received_delta_ids) + 1}:targeted",
                            packet_id=targeted_packet_id,
                        )
                    targeted_results = SemanticResolver(state, receiver_view).check_contract(contract)
                    targeted_feedback = build_semantic_feedback(
                        sender=sender,
                        receiver=receiver,
                        round_index=refinement_rounds + 1,
                        results=targeted_results,
                    )
                    targeted_receiver_need = diagnose_receiver_need(
                        sender=sender,
                        receiver=receiver,
                        stage=f"{sender}->{receiver}",
                        round_index=refinement_rounds + 1,
                        results=targeted_results,
                    )
                    targeted_feedbacks.append(targeted_feedback.as_dict())
                    targeted_receiver_needs.append(targeted_receiver_need.as_dict())
                    check_results = targeted_results
                    feedback = targeted_feedback
                    refinement_rounds += 1
                    feedback_decision = decide_feedback_action(
                        feedback,
                        policy_name=self.communication_policy.name,
                        round_index=refinement_rounds,
                        max_rounds=max_refinement_rounds,
                    )
                    feedback_decisions.append(feedback_decision.as_dict())
                if feedback_decision.action == FeedbackAction.SEND_ALL_FALLBACK:
                    fallback_roots = [candidate.node_id for candidate in candidates if candidate.exportable]
                    fallback_delta = dependency_closure(
                        state,
                        root_node_ids=fallback_roots,
                        sender=sender,
                        receiver=receiver,
                        receiver_view=receiver_view,
                        policy=f"{self.communication_policy.name}:send_all_fallback",
                        candidate_token_cost=candidate_tokens,
                    )
                    if fallback_delta.node_ids or fallback_delta.edge_ids:
                        self.agent_views.grant(
                            receiver,
                            node_ids=fallback_delta.node_ids,
                            edge_ids=fallback_delta.edge_ids,
                            fragment_ids=self._delivery_fragment_ids(
                                state,
                                sender=sender,
                                receiver=receiver,
                                node_ids=fallback_delta.node_ids,
                                mandatory_root_node_ids=fallback_roots,
                                selected_optional_fragment_ids=optional_fragment_ids,
                                include_all=True,
                            ),
                            delta_id=f"{sender}->{receiver}:{len(receiver_view.received_delta_ids) + 1}:fallback",
                            packet_id=f"packet:{sender}->{receiver}:fallback:{len(receiver_view.received_delta_ids) + 1}",
                        )
                    fallback_results = SemanticResolver(state, receiver_view).check_contract(contract)
                    fallback_feedback = build_semantic_feedback(
                        sender=sender,
                        receiver=receiver,
                        round_index=refinement_rounds + 1,
                        results=fallback_results,
                    )
                    fallback_receiver_need = diagnose_receiver_need(
                        sender=sender,
                        receiver=receiver,
                        stage=f"{sender}->{receiver}",
                        round_index=refinement_rounds + 1,
                        results=fallback_results,
                    )
                    check_results = fallback_results
                    feedback = fallback_feedback
                    feedback_decision = decide_feedback_action(
                        feedback,
                        policy_name=self.communication_policy.name,
                        round_index=refinement_rounds + 1,
                        max_rounds=max_refinement_rounds,
                    )
                    feedback_decisions.append(feedback_decision.as_dict())
                receiver_need = diagnose_receiver_need(
                    sender=sender,
                    receiver=receiver,
                    stage=f"{sender}->{receiver}",
                    round_index=refinement_rounds + (1 if fallback_feedback is not None else 0),
                    results=check_results,
                )
                targeted_token_sum = sum(item.token_cost for item in targeted_deltas)
                targeted_node_ids = tuple(dict.fromkeys(node_id for item in targeted_deltas for node_id in item.node_ids))
                targeted_edge_ids = tuple(dict.fromkeys(edge_id for item in targeted_deltas for edge_id in item.edge_ids))
                semantic_event.update({
                    "semantic_required_count": len(check_results),
                    "semantic_satisfied_count": sum(1 for result in check_results if result.status.value == "SATISFIED"),
                    "semantic_recoverable_count": sum(1 for result in check_results if result.status.value == "RECOVERABLE"),
                    "semantic_missing_count": sum(1 for result in check_results if result.status.value == "MISSING"),
                    "semantic_feedback": feedback.as_dict(),
                    "semantic_feedback_decision": feedback_decision.as_dict(),
                    "semantic_feedback_decisions": feedback_decisions,
                    "semantic_initial_feedback": initial_feedback.as_dict(),
                    "semantic_receiver_need": receiver_need.as_dict(),
                    "semantic_initial_receiver_need": initial_receiver_need.as_dict(),
                    "semantic_feedback_level": nack_level(feedback),
                    "semantic_initial_feedback_level": nack_level(initial_feedback),
                    "semantic_ack": is_ack(feedback),
                    "semantic_nack": is_nack(feedback),
                    "semantic_feedback_request": is_feedback_request(feedback),
                    "semantic_hard_nack": is_hard_nack(feedback),
                    "semantic_soft_nack": is_soft_nack(feedback),
                    "semantic_verification_nack": is_verification_nack(feedback),
                    "semantic_quality_nack": is_quality_nack(feedback),
                    "semantic_nack_unresolved": is_hard_nack(feedback),
                    "semantic_final_contract_satisfied": is_ack(feedback),
                    "semantic_hard_contract_satisfied": not is_hard_nack(feedback),
                    "semantic_initial_nack": is_nack(initial_feedback),
                    "semantic_initial_feedback_request": is_feedback_request(initial_feedback),
                    "semantic_initial_hard_nack": is_hard_nack(initial_feedback),
                    "semantic_initial_soft_nack": is_soft_nack(initial_feedback),
                    "semantic_initial_verification_nack": is_verification_nack(initial_feedback),
                    "semantic_initial_quality_nack": is_quality_nack(initial_feedback),
                    "semantic_initial_missing_count": len(getattr(initial_feedback, "missing_semantics", ())),
                    "semantic_initial_hard_missing_count": len(getattr(initial_feedback, "hard_missing_semantics", ())),
                    "semantic_initial_soft_missing_count": len(getattr(initial_feedback, "soft_missing_semantics", ())),
                    "semantic_initial_verification_missing_count": len(getattr(initial_feedback, "verification_missing_semantics", ())),
                    "semantic_initial_quality_gap_count": len(getattr(initial_feedback, "quality_gaps", ())),
                    "semantic_final_missing_count": len(getattr(feedback, "missing_semantics", ())),
                    "semantic_final_hard_missing_count": len(getattr(feedback, "hard_missing_semantics", ())),
                    "semantic_final_soft_missing_count": len(getattr(feedback, "soft_missing_semantics", ())),
                    "semantic_final_verification_missing_count": len(getattr(feedback, "verification_missing_semantics", ())),
                    "semantic_final_quality_gap_count": len(getattr(feedback, "quality_gaps", ())),
                    "semantic_hard_repaired_count": max(
                        0,
                        len(getattr(initial_feedback, "hard_missing_semantics", ()))
                        - len(getattr(feedback, "hard_missing_semantics", ())),
                    ),
                    "semantic_verification_repaired_count": max(
                        0,
                        len(getattr(initial_feedback, "verification_missing_semantics", ()))
                        - len(getattr(feedback, "verification_missing_semantics", ())),
                    ),
                    "semantic_quality_repaired_count": max(
                        0,
                        len(getattr(initial_feedback, "quality_gaps", ()))
                        - len(getattr(feedback, "quality_gaps", ())),
                    ),
                    "semantic_soft_repaired_count": max(
                        0,
                        len(getattr(initial_feedback, "soft_missing_semantics", ()))
                        - len(getattr(feedback, "soft_missing_semantics", ())),
                    ),
                    "semantic_missing_repaired_count": max(
                        0,
                        len(getattr(initial_feedback, "missing_semantics", ()))
                        - len(getattr(feedback, "missing_semantics", ())),
                    ),
                    "refinement_rounds": refinement_rounds,
                    "max_refinement_rounds": max_refinement_rounds,
                    "targeted_refinement": bool(targeted_plans),
                    "targeted_refinement_skipped": targeted_refinement_skipped,
                    "targeted_refinement_skipped_count": int(targeted_refinement_skipped),
                    "targeted_refinement_skip_reason": targeted_refinement_skip_reason,
                    "quality_refinement_budget_exceeded_count": int(
                        targeted_refinement_skip_reason == "quality_refinement_budget_exceeded"
                    ),
                    "targeted_refinement_skipped_plan": targeted_refinement_skipped_plan,
                    "targeted_refinement_tokens": targeted_token_sum,
                    "targeted_refinement_node_ids": list(targeted_node_ids),
                    "targeted_refinement_edge_ids": list(targeted_edge_ids),
                    "targeted_refinement_plan": targeted_plan.as_dict() if targeted_plan is not None else None,
                    "targeted_refinement_plans": targeted_plans,
                    "targeted_refinement_feedback": targeted_feedback.as_dict() if targeted_feedback is not None else None,
                    "targeted_refinement_feedbacks": targeted_feedbacks,
                    "targeted_refinement_receiver_need": (
                        targeted_receiver_need.as_dict() if targeted_receiver_need is not None else None
                    ),
                    "targeted_refinement_receiver_needs": targeted_receiver_needs,
                    "targeted_refinement_requested_fragment_ids": (
                        list(dict.fromkeys(targeted_requested_fragment_ids))
                    ),
                    "fallback_send_all": fallback_delta is not None,
                    "fallback_tokens": fallback_delta.token_cost if fallback_delta is not None else 0,
                    "fallback_node_ids": list(fallback_delta.node_ids) if fallback_delta is not None else [],
                    "fallback_edge_ids": list(fallback_delta.edge_ids) if fallback_delta is not None else [],
                    "wasted_pre_fallback_tokens": (
                        delta.token_cost + targeted_token_sum
                        if fallback_delta is not None else 0
                    ),
                    "fallback_feedback": fallback_feedback.as_dict() if fallback_feedback is not None else None,
                    "fallback_receiver_need": (
                        fallback_receiver_need.as_dict() if fallback_receiver_need is not None else None
                    ),
                })
            receiver_visible_after = sorted(receiver_view.visible_node_ids)
            rendered_context_tokens = 0
            rendered_context_node_ids: list[str] = []
            rendered_context_fragment_ids: list[str] = []
            try:
                receiver_view_store = self.agent_views.visible_store(receiver)
                receiver_context_slice = build_context_slice(
                    receiver_view_store,
                    task_id=self.task_id,
                    branch_id=branch_id,
                    role=receiver,
                    policy=(
                        "planner_state" if receiver == "planner"
                        else "solver_state" if receiver == "solver"
                        else "dependency_closure"
                    ),
                    allow_missing=True,
                    visible_fragment_ids=receiver_view.visible_fragment_ids,
                    multihop_sentence_limit=(
                        self.multihop_sentence_limit
                        if self.graph_context_mode == "role_aware"
                        else None
                    ),
                )
                rendered_context = render_compact_context_slice(
                    receiver_context_slice,
                    receiver_view_store.snapshot(),
                )
                rendered_context_tokens = self._token_count(rendered_context)
                rendered_context_node_ids = list(receiver_context_slice.visible_node_ids)
                rendered_context_fragment_ids = list(receiver_context_slice.visible_fragment_ids)
            except Exception:
                rendered_context_tokens = 0
                rendered_context_node_ids = []
                rendered_context_fragment_ids = []
            active_targeted_deltas = targeted_deltas if semantic_minimal_policy else []
            total_sent_node_ids = tuple(dict.fromkeys([
                *delta.node_ids,
                *(node_id for item in active_targeted_deltas for node_id in item.node_ids),
                *((fallback_delta.node_ids if fallback_delta is not None else ())),
            ]))
            total_sent_edge_ids = tuple(dict.fromkeys([
                *delta.edge_ids,
                *(edge_id for item in active_targeted_deltas for edge_id in item.edge_ids),
                *((fallback_delta.edge_ids if fallback_delta is not None else ())),
            ]))
            initial_sent_fragment_ids = self._delivery_fragment_ids(
                state,
                sender=sender,
                receiver=receiver,
                node_ids=delta.node_ids,
                mandatory_root_node_ids=mandatory_roots,
                selected_optional_fragment_ids=selected_optional_fragment_ids,
            )
            targeted_sent_fragment_ids = tuple(dict.fromkeys(
                fragment_id
                for item in active_targeted_deltas
                for fragment_id in self._delivery_fragment_ids(
                    state,
                    sender=sender,
                    receiver=receiver,
                    node_ids=item.node_ids,
                    mandatory_root_node_ids=item.root_node_ids,
                    selected_optional_fragment_ids=targeted_requested_fragment_ids,
                )
            ))
            fallback_sent_fragment_ids = (
                self._delivery_fragment_ids(
                    state,
                    sender=sender,
                    receiver=receiver,
                    node_ids=fallback_delta.node_ids,
                    mandatory_root_node_ids=fallback_delta.root_node_ids,
                    selected_optional_fragment_ids=optional_fragment_ids,
                    include_all=True,
                )
                if fallback_delta is not None else ()
            )
            total_sent_fragment_ids = tuple(dict.fromkeys([
                *initial_sent_fragment_ids,
                *targeted_sent_fragment_ids,
                *fallback_sent_fragment_ids,
            ]))
            sent_edge_tokens = sum(estimate_edge_tokens(edge) for edge in state.edges if edge.edge_id in set(total_sent_edge_ids))
            sent_fragment_token_by_id = {
                fragment_id: fragment_token_cost((fragment_id,), state.nodes)
                for fragment_id in total_sent_fragment_ids
            }
            sent_fragment_name_by_id = {
                fragment_id: str(fragment_id).split("#", 1)[1] if "#" in str(fragment_id) else ""
                for fragment_id in total_sent_fragment_ids
            }
            initial_sent_tokens = fragment_token_cost(initial_sent_fragment_ids, state.nodes) + sum(
                estimate_edge_tokens(edge) for edge in state.edges if edge.edge_id in set(delta.edge_ids)
            )
            targeted_sent_tokens = fragment_token_cost(targeted_sent_fragment_ids, state.nodes) + sum(
                estimate_edge_tokens(edge) for edge in state.edges
                if edge.edge_id in {edge_id for item in active_targeted_deltas for edge_id in item.edge_ids}
            )
            fallback_sent_tokens = (
                fragment_token_cost(fallback_sent_fragment_ids, state.nodes) + sum(
                    estimate_edge_tokens(edge) for edge in state.edges if edge.edge_id in set(fallback_delta.edge_ids)
                )
                if fallback_delta is not None else 0
            )
            feedback_sent_fragment_ids = tuple(dict.fromkeys([
                *targeted_sent_fragment_ids,
                *fallback_sent_fragment_ids,
            ]))
            feedback_newly_visible_fragment_ids = tuple(
                fragment_id for fragment_id in feedback_sent_fragment_ids
                if fragment_id in receiver_view.visible_fragment_ids
                and fragment_id not in set(receiver_visible_before_fragments)
            )
            feedback_newly_rendered_fragment_ids = tuple(
                fragment_id for fragment_id in feedback_newly_visible_fragment_ids
                if fragment_id in set(rendered_context_fragment_ids)
            )
            feedback_sent_tokens = targeted_sent_tokens + fallback_sent_tokens
            feedback_newly_visible_tokens = fragment_token_cost(feedback_newly_visible_fragment_ids, state.nodes)
            feedback_newly_rendered_tokens = fragment_token_cost(feedback_newly_rendered_fragment_ids, state.nodes)
            feedback_utilization = (
                feedback_newly_rendered_tokens / feedback_sent_tokens
                if feedback_sent_tokens else 0.0
            )
            total_sent_tokens = fragment_token_cost(total_sent_fragment_ids, state.nodes) + sent_edge_tokens
            core_comm_tokens = 0
            delta_comm_tokens = 0
            verification_comm_tokens = 0
            quality_comm_tokens = 0
            receiver_seen_hit_count = 0
            repeated_comm_tokens = 0
            unique_comm_tokens = sent_edge_tokens
            seen_before_fragments = set(receiver_visible_before_fragments)
            quality_fragment_names = {
                "full_plan", "rationale", "result_metadata", "calculation_trace",
                "full_feedback", "repair_hint", "error_type", "error_location",
                "execution_detail",
            }
            verification_fragment_names = {"dependencies", "key_operation", "task_input"}
            for fragment_id, token_cost in sent_fragment_token_by_id.items():
                if fragment_id in seen_before_fragments:
                    receiver_seen_hit_count += 1
                    repeated_comm_tokens += token_cost
                else:
                    unique_comm_tokens += token_cost
                targeted_level = targeted_fragment_level_by_id.get(fragment_id, "")
                fragment_name = sent_fragment_name_by_id.get(fragment_id, "")
                if targeted_level == "verification":
                    verification_comm_tokens += token_cost
                elif targeted_level == "quality":
                    quality_comm_tokens += token_cost
                elif fragment_scope_by_id.get(fragment_id) == MANDATORY_FRAGMENT:
                    core_comm_tokens += token_cost
                elif fragment_name in verification_fragment_names:
                    verification_comm_tokens += token_cost
                elif fragment_name in quality_fragment_names:
                    quality_comm_tokens += token_cost
                else:
                    delta_comm_tokens += token_cost
            control_comm_tokens = sent_edge_tokens
            total_comm_tokens = total_sent_tokens
            full_state_equivalent_tokens = candidate_tokens + sent_edge_tokens
            state_delta_tokens = total_comm_tokens
            duplicate_ratio = repeated_comm_tokens / total_comm_tokens if total_comm_tokens else 0.0
            incremental_saving = (
                1.0 - (state_delta_tokens / full_state_equivalent_tokens)
                if full_state_equivalent_tokens else 0.0
            )
            communication_token_breakdown_ok = total_comm_tokens == (
                core_comm_tokens
                + delta_comm_tokens
                + verification_comm_tokens
                + quality_comm_tokens
                + control_comm_tokens
            )
            unique_repeated_accounting_ok = total_comm_tokens == unique_comm_tokens + repeated_comm_tokens
            feedback_render_accounting_ok = feedback_sent_tokens >= feedback_newly_rendered_tokens
            revision_success_count = int(
                bool(semantic_minimal_policy)
                and bool(semantic_event.get("semantic_initial_feedback_request", False))
                and bool(semantic_event.get("semantic_ack", False))
            )
            revision_regression_count = int(
                bool(semantic_minimal_policy)
                and bool(semantic_event.get("semantic_initial_feedback_request", False))
                and bool(semantic_event.get("semantic_hard_nack", False))
            )
            early_stop_round = (
                int(semantic_event.get("refinement_rounds", 0) or 0)
                if semantic_minimal_policy and bool(semantic_event.get("semantic_ack", False))
                else None
            )
            total_closure_added = len(delta.closure_added_node_ids) + (
                sum(len(item.closure_added_node_ids) for item in active_targeted_deltas)
            ) + (
                len(fallback_delta.closure_added_node_ids) if fallback_delta is not None else 0
            )
            if semantic_minimal_policy:
                repaired_count = int(semantic_event.get("semantic_missing_repaired_count", 0) or 0)
                hard_repaired_count = int(semantic_event.get("semantic_hard_repaired_count", 0) or 0)
                soft_repaired_count = int(semantic_event.get("semantic_soft_repaired_count", 0) or 0)
                semantic_event.update({
                    "semantic_packet_tokens": initial_sent_tokens,
                    "targeted_refinement_tokens": targeted_sent_tokens,
                    "fallback_tokens": fallback_sent_tokens,
                    "feedback_transport_tokens": feedback_sent_tokens,
                    "feedback_sent_tokens": feedback_sent_tokens,
                    "feedback_newly_visible_tokens": feedback_newly_visible_tokens,
                    "feedback_newly_rendered_tokens": feedback_newly_rendered_tokens,
                    "feedback_utilization": feedback_utilization,
                    "feedback_newly_visible_fragment_ids": list(feedback_newly_visible_fragment_ids),
                    "feedback_newly_rendered_fragment_ids": list(feedback_newly_rendered_fragment_ids),
                    "feedback_nack_repair_efficiency": (
                        repaired_count / feedback_sent_tokens if feedback_sent_tokens else 0.0
                    ),
                    "feedback_hard_repair_efficiency": (
                        hard_repaired_count / feedback_sent_tokens if feedback_sent_tokens else 0.0
                    ),
                    "feedback_soft_repair_efficiency": (
                        soft_repaired_count / feedback_sent_tokens if feedback_sent_tokens else 0.0
                    ),
                    "wasted_pre_fallback_tokens": (
                        initial_sent_tokens + targeted_sent_tokens
                        if fallback_delta is not None else 0
                    ),
                })
            event = {
                "decision_snapshot_id": decision_snapshot_id,
                "sender": sender,
                "receiver": receiver,
                "branch_id": branch_id,
                "policy": self.communication_policy.name,
                "candidate_count": len(candidates),
                "mandatory_root_count": len(mandatory_candidates),
                "optional_root_count": len(optional_candidates),
                "selected_root_count": len(roots),
                "selected_optional_root_count": len(optional_roots),
                "sent_count": len(total_sent_node_ids),
                "edge_count": len(total_sent_edge_ids),
                "candidate_tokens": candidate_tokens,
                "mandatory_root_tokens": mandatory_tokens,
                "optional_root_tokens": optional_tokens,
                "selected_optional_root_tokens": selected_optional_tokens,
                "candidate_fragment_count": len(fragment_candidates),
                "mandatory_fragment_count": len(mandatory_fragment_ids),
                "optional_fragment_count": len(optional_fragment_ids),
                "selected_optional_fragment_count": len(selected_optional_fragment_ids),
                "candidate_fragment_ids": [fragment.fragment_id for fragment in fragment_candidates],
                "mandatory_fragment_ids": mandatory_fragment_ids,
                "optional_fragment_ids": optional_fragment_ids,
                "selected_optional_fragment_ids": list(selected_optional_fragment_ids),
                "model_proposed_fragment_ids": list(model_hint_ids),
                "kernel_rejected_fragment_ids": [
                    fragment_id for fragment_id in model_hint_ids
                    if fragment_id not in fragment_scope_by_id
                    or fragment_scope_by_id.get(fragment_id) != OPTIONAL_FRAGMENT
                ] + budget_rejected_fragment_ids,
                "unused_sent_fragment_ids": [
                    fragment_id for fragment_id in model_hint_ids
                    if fragment_id not in set(total_sent_fragment_ids)
                ],
                "model_proposed_fragments": len(model_hint_ids),
                "kernel_rejected_fragments": sum(
                    1 for fragment_id in model_hint_ids
                    if fragment_id not in fragment_scope_by_id
                    or fragment_scope_by_id.get(fragment_id) != OPTIONAL_FRAGMENT
                ) + len(budget_rejected_fragment_ids),
                "closure_added_fragments": max(0, len(total_sent_fragment_ids) - len(selected_optional_fragment_ids)),
                "receiver_consumed_fragments": len(total_sent_fragment_ids),
                "unused_sent_fragments": sum(
                    1 for fragment_id in model_hint_ids
                    if fragment_id not in set(total_sent_fragment_ids)
                ),
                "fragment_token_by_id": fragment_token_by_id,
                "fragment_scope_by_id": fragment_scope_by_id,
                "sent_tokens": total_sent_tokens,
                "communication_token_accounting_ok": total_sent_tokens == (
                    initial_sent_tokens + targeted_sent_tokens + fallback_sent_tokens
                ),
                "total_comm_tokens": total_comm_tokens,
                "core_comm_tokens": core_comm_tokens,
                "delta_comm_tokens": delta_comm_tokens,
                "verification_comm_tokens": verification_comm_tokens,
                "quality_comm_tokens": quality_comm_tokens,
                "control_comm_tokens": control_comm_tokens,
                "communication_token_breakdown_ok": communication_token_breakdown_ok,
                "unique_comm_tokens": unique_comm_tokens,
                "repeated_comm_tokens": repeated_comm_tokens,
                "duplicate_ratio": duplicate_ratio,
                "unique_repeated_accounting_ok": unique_repeated_accounting_ok,
                "receiver_seen_hit_count": receiver_seen_hit_count,
                "state_delta_tokens": state_delta_tokens,
                "full_state_equivalent_tokens": full_state_equivalent_tokens,
                "incremental_saving": incremental_saving,
                "feedback_render_accounting_ok": feedback_render_accounting_ok,
                "revision_success_count": revision_success_count,
                "revision_regression_count": revision_regression_count,
                "early_stop_round": early_stop_round,
                "initial_packet_tokens": initial_sent_tokens,
                "initial_sent_tokens": initial_sent_tokens,
                "targeted_sent_tokens": targeted_sent_tokens,
                "fallback_sent_tokens": fallback_sent_tokens,
                "initial_sent_node_ids": list(delta.node_ids),
                "initial_sent_edge_ids": list(delta.edge_ids),
                "rendered_context_tokens": rendered_context_tokens,
                "rendered_context_node_ids": rendered_context_node_ids,
                "rendered_context_fragment_ids": sorted(rendered_context_fragment_ids),
                "closure_added_count": total_closure_added,
                "candidate_node_ids": [candidate.node_id for candidate in candidates],
                "candidate_token_by_node": {candidate.node_id: candidate.token_cost for candidate in candidates},
                "candidate_scope_by_node": {candidate.node_id: candidate.scope for candidate in candidates},
                "mandatory_root_node_ids": list(mandatory_roots),
                "mandatory_root_token_by_node": {
                    candidate.node_id: candidate.token_cost for candidate in mandatory_candidates
                },
                "optional_root_node_ids": [candidate.node_id for candidate in optional_candidates],
                "optional_root_token_by_node": {
                    candidate.node_id: candidate.token_cost for candidate in optional_candidates
                },
                "selected_root_node_ids": list(roots),
                "selected_optional_root_node_ids": list(optional_roots),
                "selected_optional_root_token_by_node": {
                    node_id: sum(
                        int(fragment_token_by_id.get(fragment_id, 0) or 0)
                        for fragment_id in selected_optional_fragment_ids
                        if node_id_from_fragment(fragment_id) == node_id
                    )
                    for node_id in optional_roots
                },
                "selected_optional_fragment_token_by_id": {
                    fragment_id: int(fragment_token_by_id.get(fragment_id, 0) or 0)
                    for fragment_id in selected_optional_fragment_ids
                },
                "initial_sent_fragment_ids": list(initial_sent_fragment_ids),
                "targeted_refinement_fragment_ids": list(targeted_sent_fragment_ids),
                "fallback_fragment_ids": list(fallback_sent_fragment_ids),
                "sent_fragment_ids": list(total_sent_fragment_ids),
                "sent_fragment_tokens": fragment_token_cost(total_sent_fragment_ids, state.nodes),
                "sent_node_ids": list(total_sent_node_ids),
                "sent_edge_ids": list(total_sent_edge_ids),
                "receiver_visible_before_node_ids": receiver_visible_before,
                "receiver_visible_after_node_ids": receiver_visible_after,
                "receiver_visible_before_fragment_ids": receiver_visible_before_fragments,
                "receiver_visible_after_fragment_ids": sorted(receiver_view.visible_fragment_ids),
                "receiver_visible_before_count": len(receiver_visible_before),
                "receiver_visible_after_count": len(receiver_visible_after),
            }
            event.update(semantic_event)
            if self.telemetry is not None:
                self.telemetry.record_graph_communication(event)
            events.append(event)
        return events

    def _record_communication_decision(
        self,
        *,
        state: Any,
        sender: str,
        receiver: str,
        branch_id: str,
        candidates: Sequence[Any],
        fragments: Sequence[Any],
        workflow_state: Optional[Mapping[str, Any]],
        decision_context: Optional[Mapping[str, Any]],
    ) -> str:
        """Capture the exact pre-action runtime state when replay is enabled."""
        if self.communication_replay_buffer is None:
            return ""
        decision_index = self._communication_decision_index
        self._communication_decision_index += 1
        receiver_view = self.agent_views.view(receiver)
        checkpoint_state = dict(workflow_state or {})
        checkpoint_state["communication_decision_context"] = dict(decision_context or {})
        checkpoint_state["communication_budget_scope"] = "per_decision"
        external_state = (
            self.replay_external_state_provider()
            if self.replay_external_state_provider is not None else {}
        )
        checkpoint_state["communication_runtime_state"] = {
            "graph_source_seen_hashes": sorted(self._graph_source_seen_hashes),
            "last_hotpot_repair_packet_audit": dict(self._last_hotpot_repair_packet_audit),
            "next_communication_decision_index": self._communication_decision_index,
        }
        snapshot_id = new_snapshot_id(
            task_id=self.task_id,
            branch_id=branch_id,
            stage=f"{sender}->{receiver}",
            decision_index=decision_index,
        )
        snapshot = CommunicationDecisionSnapshot(
            snapshot_id=snapshot_id,
            task_id=self.task_id,
            branch_id=branch_id,
            task_type=self.task_type,
            stage=f"{sender}->{receiver}",
            sender=sender,
            receiver=receiver,
            decision_index=decision_index,
            round_index=len(receiver_view.received_delta_ids),
            remaining_budget=self.communication_budget_tokens,
            policy_name=self.communication_policy.name,
            graph_state=state,
            agent_views=self.agent_views.snapshot(),
            workflow_state=checkpoint_state,
            candidates=tuple(candidate_snapshot(candidate) for candidate in candidates),
            fragments=tuple(fragment_snapshot(fragment) for fragment in fragments),
            external_state=normalize_external_state(external_state),
        )
        self.communication_replay_buffer.record(snapshot)
        return snapshot_id

    def restore_communication_snapshot(
        self,
        snapshot: CommunicationDecisionSnapshot,
        *,
        restore_external: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> WorkflowState:
        """Restore a captured decision point into this workflow instance."""
        if snapshot.task_type != self.task_type:
            raise ValueError(
                f"communication snapshot task type mismatch: {snapshot.task_type}/{self.task_type}"
            )
        restored = snapshot.restore_runtime(
            self.store,
            self.agent_views,
            restore_external=restore_external,
        )
        runtime_state = restored.get("communication_runtime_state", {})
        if isinstance(runtime_state, Mapping):
            self._graph_source_seen_hashes = {
                str(item) for item in runtime_state.get("graph_source_seen_hashes", ())
            }
            audit = runtime_state.get("last_hotpot_repair_packet_audit", {})
            self._last_hotpot_repair_packet_audit = dict(audit) if isinstance(audit, Mapping) else {}
            self._communication_decision_index = int(
                runtime_state.get("next_communication_decision_index", snapshot.decision_index + 1)
            )
        self.compiler.set_node_ref_context({})
        return restored

    def _root_only_delta(
        self,
        *,
        state: Any,
        root_node_ids: Sequence[str],
        sender: str,
        receiver: str,
        receiver_view: Any,
        candidate_token_cost: int,
    ) -> GraphDelta:
        selected: list[str] = []
        baseline_visible = set(receiver_view.visible_node_ids) | {
            node.node_id
            for node in state.nodes.values()
            if node.logical_id in MANDATORY_LOGICAL_IDS
            or node.provenance.get("communication_scope") == "MANDATORY"
            or node_visible_to_role(node, receiver)
        }
        for node_id in root_node_ids:
            if node_id in state.nodes and node_id not in baseline_visible and node_id not in selected:
                selected.append(node_id)
        visible_after = baseline_visible | set(selected)
        edge_ids = tuple(sorted(
            edge.edge_id for edge in state.edges
            if edge.source in visible_after
            and edge.target in visible_after
            and (edge.source in selected or edge.target in selected)
        ))
        token_cost = sum(estimate_node_tokens(state.nodes[node_id]) for node_id in selected)
        token_cost += sum(estimate_edge_tokens(edge) for edge in state.edges if edge.edge_id in edge_ids)
        return GraphDelta(
            sender=sender,
            receiver=receiver,
            root_node_ids=tuple(selected),
            node_ids=tuple(selected),
            edge_ids=edge_ids,
            token_cost=token_cost,
            candidate_token_cost=candidate_token_cost,
            closure_added_node_ids=(),
            redundant_node_ids=(),
            policy=self.communication_policy.name,
        )

    def _fragment_candidates(
        self,
        state: Any,
        *,
        candidates: Sequence[DeltaCandidate],
        sender: str,
        receiver: str,
    ):
        fragments = []
        for candidate in candidates:
            node = state.nodes.get(candidate.node_id)
            if node is None:
                continue
            fragments.extend(
                fragments_for_node(
                    node,
                    sender=sender,
                    receiver=receiver,
                    node_scope=candidate.scope,
                )
            )
        return tuple(dict((fragment.fragment_id, fragment) for fragment in fragments).values())

    def _fragment_delta_candidate(
        self,
        fragment: Any,
        *,
        sender: str,
        receiver: str,
    ) -> DeltaCandidate:
        return DeltaCandidate(
            node_id=fragment.fragment_id,
            parent_node_id=fragment.node_id,
            sender=sender,
            receiver=receiver,
            exportable=True,
            novelty_score=1.0,
            redundancy_score=0.0,
            token_cost=fragment.token_cost,
            structural_features={},
            scope=OPTIONAL_FRAGMENT,
            fragment_id=fragment.fragment_id,
            fragment_name=fragment.name,
            is_fragment=True,
        )

    def _delivery_fragment_ids(
        self,
        state: Any,
        *,
        sender: str,
        receiver: str,
        node_ids: Sequence[str],
        mandatory_root_node_ids: Sequence[str],
        selected_optional_fragment_ids: Sequence[str],
        include_all: bool = False,
    ) -> tuple[str, ...]:
        selected_by_node: dict[str, set[str]] = {}
        for fragment_id in selected_optional_fragment_ids:
            selected_by_node.setdefault(node_id_from_fragment(fragment_id), set()).add(str(fragment_id))
        mandatory_roots = {str(node_id) for node_id in mandatory_root_node_ids}
        fragment_ids: list[str] = []
        for node_id in dict.fromkeys(str(item) for item in node_ids):
            node = state.nodes.get(node_id)
            if node is None:
                continue
            if include_all:
                fragment_ids.extend(all_fragment_ids(
                    node,
                    sender=sender,
                    receiver=receiver,
                    node_scope="MANDATORY_STAGE",
                ))
                continue
            node_selected_optional = sorted(selected_by_node.get(node_id, set()))
            fragment_ids.extend(node_selected_optional)
            if node_id in mandatory_roots or not node_selected_optional:
                fragment_ids.extend(
                    fragment.fragment_id
                    for fragment in fragments_for_node(
                        node,
                        sender=sender,
                        receiver=receiver,
                        node_scope="MANDATORY_STAGE",
                    )
                    if fragment.scope == MANDATORY_FRAGMENT
                )
        return tuple(dict.fromkeys(fragment_ids))

    @staticmethod
    def _node_ref_context(
        context_slice: Any,
        *,
        visible_node_ids: Optional[Sequence[str]] = None,
        known_fragment_ids: Optional[Sequence[str]] = None,
        allowed_receivers: Optional[Sequence[str]] = None,
    ) -> dict[str, Any]:
        fragment_index = dict(getattr(context_slice, "fragment_provenance", {}) or {})
        local_node_index = {
            local_id: {
                "source_node_id": node_id,
                "source_logical_id": "",
                "source_node_type": "",
                "local_node_id": local_id,
            }
            for node_id, local_id in (getattr(context_slice, "local_node_id_by_node", {}) or {}).items()
        }
        semantic_type_index: dict[str, list[dict[str, str]]] = {}
        for record in fragment_index.values():
            if not isinstance(record, dict):
                continue
            for key in ("source_logical_id", "source_node_type"):
                value = str(record.get(key, "")).strip()
                if value:
                    semantic_type_index.setdefault(value, []).append(record)
        collapsed_semantic_index: dict[str, Any] = {}
        for key, values in semantic_type_index.items():
            unique = {
                str(value.get("source_node_id", "")): value
                for value in values
                if str(value.get("source_node_id", "")).strip()
            }
            if len(unique) == 1:
                collapsed_semantic_index[key] = next(iter(unique.values()))
        return {
            "fragment_index": fragment_index,
            "local_node_index": local_node_index,
            "semantic_type_index": collapsed_semantic_index,
            "visible_node_ids": tuple(str(item) for item in (visible_node_ids or ())),
            "known_fragment_ids": tuple(str(item) for item in (known_fragment_ids or ())),
            "allowed_receivers": tuple(str(item) for item in (allowed_receivers or ())),
        }

    @staticmethod
    def _communication_receivers(sender: str) -> tuple[str, ...]:
        return {
            "planner": ("solver",),
            "solver": ("critic",),
            "tool": ("critic",),
            "critic": ("solver", "final_solver"),
            "final_solver": (),
        }.get(sender, ())

    def _rollback_model_session(self, session_id: str) -> None:
        rollback = getattr(self.model, "rollback_session", None)
        if callable(rollback):
            rollback(session_id)

    def _session_snapshot(self, session_id: str) -> dict[str, Any]:
        snapshot = getattr(self.model, "session_snapshot", None)
        if not callable(snapshot):
            return {"session_id": session_id, "backend": "callable-without-persistent-session"}
        return dict(snapshot(session_id))

    def _session_prompt(self, role: str, mode: str, prompt: str) -> str:
        user_content = (
            f"<WORKFLOW_STATE role={role!r} mode={mode!r}>\n"
            f"{prompt}\n"
            "</WORKFLOW_STATE>\n"
            f"{self._action_role_suffix(role, mode)}"
        )
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self._system_contract(role, mode)},
                    {"role": "user", "content": user_content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return (
            f"<SYSTEM>{self._system_contract(role, mode)}</SYSTEM>\n"
            f"{user_content}\n<ASSISTANT>"
        )

    def _retry_session_prompt(self, role: str, mode: str, error: Exception) -> str:
        if self._is_hotpot_repair_mode(role, mode):
            return (
                "\n<|im_end|>\n<|im_start|>user\n"
                "<RETRY role='solver' mode='repair'>\n"
                "HotpotQA repair is a bounded local patch step. Emit ONLY one repair_patch Action for this attempt. "
                "Choose STOP, ADD_EVIDENCE, REPLACE_EVIDENCE, PATCH_CITATIONS, PATCH_BRIDGE, or REVISE_ANSWER. "
                "Do not emit set_result, done, plan, add_fact, add_plan_step, verify, markdown, or explanation. "
                f"Previous Action was rejected: {error}\n"
                + self._hotpot_repair_schema_instruction()
                + "\n</RETRY>\n<|im_end|>\n<|im_start|>assistant\n"
            )
        return (
            "\n<|im_end|>\n<|im_start|>user\n"
            f"<RETRY role={role!r} mode={mode!r}>\n"
            + action_contract(role, self.task_type)
            + "\n"
            + self._action_role_suffix(role, mode)
            + f"\nPrevious Action was rejected: {error}\n"
            + "Emit one corrected Action JSON object only."
            + "\n</RETRY>\n<|im_end|>\n<|im_start|>assistant\n"
        )

    def _standalone_retry_session_prompt(
        self,
        role: str,
        mode: str,
        error: Exception,
        rejected_output: str,
        *,
        context_slice: Any,
        graph_state: Any,
    ) -> str:
        """Build a stateless repair request without replaying the full graph.

        The rejected output already contains the attempted semantic content.
        Syntax/schema repair therefore needs only that output and the validator
        diagnostic.  For semantic reference or grounding failures, include the
        exact graph nodes explicitly referenced by the rejected output.
        """
        reference_context = self._retry_reference_context(
            rejected_output,
            str(error),
            context_slice=context_slice,
            graph_state=graph_state,
        )
        if self._is_hotpot_repair_mode(role, mode):
            contract = (
                "Emit exactly one repair_patch Action. Choose STOP, ADD_EVIDENCE, "
                "REPLACE_EVIDENCE, PATCH_CITATIONS, PATCH_BRIDGE, or REVISE_ANSWER.\n"
                + self._hotpot_repair_schema_instruction()
            )
        else:
            contract = action_contract(role, self.task_type)
        user_content = (
            f"<ACTION_REPAIR role={role!r} mode={mode!r}>\n"
            "Repair the rejected Action. Preserve its intended semantic content; do not solve the task again.\n"
            f"<ERROR>{error}</ERROR>\n"
            f"<REJECTED_ACTION>{rejected_output}</REJECTED_ACTION>\n"
            f"{reference_context}"
            f"<CONTRACT>{contract}</CONTRACT>\n"
            "Emit exactly one corrected Action JSON object and nothing else.\n"
            "</ACTION_REPAIR>"
        )
        system_content = (
            "You are a deterministic structured-output repairer. Return only one JSON object. "
            "Do not add markdown or explanatory text."
        )
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"<SYSTEM>{system_content}</SYSTEM>\n{user_content}\n<ASSISTANT>"

    @staticmethod
    def _retry_reference_context(
        rejected_output: str,
        error: str,
        *,
        context_slice: Any,
        graph_state: Any,
    ) -> str:
        semantic_markers = (
            "grounding",
            "quote",
            "source ref",
            "source reference",
            "unknown source",
            "not visible",
            "endpoint",
        )
        if not any(marker in error.casefold() for marker in semantic_markers):
            return ""
        if context_slice is None or graph_state is None:
            return ""
        local_ids = dict(getattr(context_slice, "local_node_id_by_node", {}) or {})
        records: list[dict[str, Any]] = []
        for node_id in getattr(context_slice, "visible_node_ids", ()):
            node = graph_state.nodes.get(node_id)
            if node is None:
                continue
            local_id = str(local_ids.get(node_id, ""))
            logical_id = str(node.logical_id)
            logical_match = bool(logical_id) and re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(logical_id)}(?![A-Za-z0-9_])",
                rejected_output,
            )
            local_match = bool(local_id) and re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(local_id)}(?![A-Za-z0-9_])",
                rejected_output,
            )
            if not logical_match and not local_match:
                continue
            records.append({
                "local_id": local_id,
                "logical_id": logical_id,
                "type": node.type,
                "content": node.content,
            })
            if len(records) >= 8:
                break
        if not records:
            return ""
        return (
            "<REFERENCED_GRAPH_NODES>"
            + json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "</REFERENCED_GRAPH_NODES>\n"
        )

    def _validate_action_dict(self, role: str, payload: dict[str, Any]) -> list[str]:
        from .protocol import normalize_action_payload, validate_action

        normalized = normalize_action_payload(payload)
        return validate_action(
            role,
            normalized,
            task_type=self.task_type,
        )

    def _semantic_action_errors(self, role: str, action: dict[str, Any]) -> list[str]:
        """Dry-run typed-action lowering and Delta validation for model repair."""
        if action.get("op") not in TYPED_SEMANTIC_OPS:
            return []
        try:
            self.compiler.validate_semantic(role, action)
        except ValueError as error:
            return [str(error)]
        return []

    def _system_contract(self, role: str, mode: str) -> str:
        if self._is_hotpot_repair_mode(role, mode):
            return (
                "You are a HotpotQA local repair operator. Task type=multihop_qa. Mode=repair. "
                "This is one bounded local patch inside an iterative repair loop. "
                "The only allowed action for this attempt is repair_patch. Forbidden actions: set_result, plan, "
                "add_plan_step, add_fact, explain, verify, done, and any other action. "
                + self._hotpot_repair_schema_instruction()
            )
        return (
            ROLE_SYSTEM_PROMPTS[role]
            + f" Task type={self.task_type}. Mode={mode}."
            + " The existing task is read-only input. Emit one Action only."
            + " Protocol version="
            + PROTOCOL_VERSION
            + ". "
            + action_contract(role, self.task_type)
        )

    @staticmethod
    def _hotpot_repair_schema_instruction() -> str:
        return (
            'Output exactly one JSON object with schema {"op":"repair_patch",'
            '"action":"PATCH_CITATIONS","patch":{"selected_fact_ids":'
            '["sentence_source_1","sentence_source_2"],"preserve_answer":true,'
            '"reasoning_chain":["first hop","second hop"]}}. '
            "Allowed actions are STOP, ADD_EVIDENCE, REPLACE_EVIDENCE, PATCH_CITATIONS, "
            "PATCH_BRIDGE, and REVISE_ANSWER. Use only citation_ids visible in allowed_evidence. "
            "Only REVISE_ANSWER may change answer."
        )

    def _latest_repair_diagnosis(self) -> dict[str, Any]:
        verification = self.store.latest_valid(self.task_id, self.branch_id, "verification")
        if verification is None or verification.status not in {"need_fix", "uncertain"}:
            return {}
        content = verification.content if isinstance(verification.content, dict) else {}
        reason = content.get("reason", "")
        repair_packet: dict[str, Any] = {}
        if isinstance(reason, str) and reason.strip().startswith("{"):
            try:
                parsed_reason = json.loads(reason)
            except json.JSONDecodeError:
                parsed_reason = {}
            if isinstance(parsed_reason, dict) and isinstance(parsed_reason.get("repair_packet"), dict):
                repair_packet = parsed_reason["repair_packet"]
                reason = str(parsed_reason.get("reason") or reason)
        return {
            "target": content.get("target", "result"),
            "error_type": content.get("error_type", ""),
            "error_location": content.get("error_location", ""),
            "reason": reason,
            "repair_instruction": content.get("repair_instruction", ""),
            "preserve": list(content.get("preserve", [])) if isinstance(content.get("preserve", []), list) else [],
            "requested_fragments": (
                list(content.get("requested_fragments", []))
                if isinstance(content.get("requested_fragments", []), list)
                else []
            ),
            "repair_packet": repair_packet,
        }

    def _render_hotpot_repair_context(self, graph_state: Any) -> str:
        previous = self._latest_result_value()
        diagnosis = self._latest_repair_diagnosis()
        evidence = self._hotpot_allowed_evidence(graph_state)
        entities = self._hotpot_available_entities(graph_state)
        bridge_stage_candidates = self._hotpot_bridge_candidate_stages(graph_state, previous, evidence)
        bridge_candidates = list(bridge_stage_candidates["repair_target_filtered"])
        repair_packet = diagnosis.get("repair_packet") or {
            "status": "need_fix",
            "error_type": diagnosis.get("error_type", ""),
            "known": {
                "reason": diagnosis.get("reason", ""),
                "error_location": diagnosis.get("error_location", ""),
            },
            "required": [diagnosis.get("repair_instruction", "")],
            "preserve": diagnosis.get("preserve", []),
        }
        if isinstance(repair_packet, dict) and not repair_packet.get("bridge_candidates"):
            repair_packet = {**repair_packet, "bridge_candidates": bridge_candidates}
        serialized_candidates = (
            self._hotpot_string_list(repair_packet.get("bridge_candidates"))
            if isinstance(repair_packet, dict)
            else bridge_candidates
        )
        bridge_stage_candidates = {
            **bridge_stage_candidates,
            "serialized_packet": serialized_candidates,
            "solver_visible": self._hotpot_dedup_strings([*serialized_candidates, *bridge_candidates]),
        }
        self._last_hotpot_repair_packet_audit = {
            "bridge_candidate_stages": bridge_stage_candidates,
            "notes": {
                "raw_pool": "candidate strings before case-insensitive deduplication",
                "ranked_top8": "first eight raw candidates in runtime construction order",
                "deduplicated_top8": "case-insensitive deduplicated candidates before packet insertion",
                "visibility_filtered": "same as deduplicated_top8; source graph was already receiver-visible",
                "budget_filtered": "same as visibility_filtered; no extra bridge candidate budget filter is applied",
                "repair_target_filtered": "same as budget_filtered; repair target currently constrains validation, not candidate delivery",
                "serialized_packet": "bridge_candidates field serialized inside repair_packet",
                "solver_visible": "union of repair_packet.bridge_candidates and top-level bridge_candidates in the prompt",
            },
        }
        packet = {
            "previous_structured_answer": previous,
            "repair_packet": repair_packet,
            "bridge_candidates": bridge_candidates,
            "allowed_entities": entities,
            "allowed_evidence": evidence,
        }
        return (
            "<HOTPOT_REPAIR_PACKET>\n"
            + json.dumps(packet, ensure_ascii=False, indent=2, default=str)
            + "\n</HOTPOT_REPAIR_PACKET>"
        )

    def _latest_result_value(self) -> Any:
        result = self.store.latest_valid(self.task_id, self.branch_id, "result")
        if result is None:
            return None
        content = result.content
        if isinstance(content, dict) and "value" in content:
            return content.get("value")
        return content

    def _hotpot_allowed_evidence(self, graph_state: Any) -> list[dict[str, Any]]:
        nodes = [
            node for node in graph_state.nodes.values()
            if node.task_id == self.task_id
            and node.branch_id == self.branch_id
            and node.type == "supporting_fact"
            and node.is_operationally_valid()
        ]
        nodes.sort(key=_node_sort_key)
        evidence: list[dict[str, Any]] = []
        for node in nodes:
            content = node.content if isinstance(node.content, dict) else {"text": node.content}
            evidence.append({
                "citation_id": node.logical_id,
                "title": content.get("title", content.get("entity", "")),
                "sent_id": content.get("sent_id"),
                "text": content.get("text", content.get("sentence", content)),
            })
        return evidence

    def _hotpot_available_entities(self, graph_state: Any) -> list[Any]:
        values: list[Any] = []
        for node in sorted(graph_state.nodes.values(), key=_node_sort_key):
            if (
                node.task_id == self.task_id
                and node.branch_id == self.branch_id
                and node.type == "entity"
                and node.is_operationally_valid()
            ):
                values.append(node.content)
        return values

    def _hotpot_bridge_candidates(self, graph_state: Any, previous: Any, evidence: list[dict[str, Any]]) -> list[str]:
        values: list[Any] = []
        if isinstance(previous, dict):
            values.append(previous.get("bridge_entity"))
            values.append(previous.get("answer"))
        for entity in self._hotpot_available_entities(graph_state):
            if isinstance(entity, dict):
                for key in ("name", "entity", "title", "value"):
                    values.append(entity.get(key))
            else:
                values.append(entity)
        for fact in evidence:
            values.append(fact.get("title"))
            values.append(fact.get("entity"))
        return bridge_candidates_from_values(values, limit=24)

    def _hotpot_bridge_candidate_stages(self, graph_state: Any, previous: Any, evidence: list[dict[str, Any]]) -> dict[str, list[str]]:
        records: list[dict[str, Any]] = []
        order = 0

        def add_candidate(value: Any, source: str, priority: float) -> None:
            nonlocal order
            text = extract_bridge_candidate(value)
            if text:
                records.append({
                    "candidate": text,
                    "source": source,
                    "priority": priority,
                    "order": order,
                })
                order += 1

        for entity in self._hotpot_available_entities(graph_state):
            if isinstance(entity, dict):
                for key in ("name", "entity", "title", "value"):
                    add_candidate(entity.get(key), "dataset_entity_field", 1.0)
            else:
                add_candidate(entity, "dataset_entity", 1.0)
        for fact in evidence:
            add_candidate(fact.get("title"), "supporting_fact_title", 0.95)
            add_candidate(fact.get("entity"), "supporting_fact_entity", 0.95)
        if isinstance(previous, dict):
            add_candidate(previous.get("bridge_entity"), "previous_bridge", 0.5)
            add_candidate(previous.get("answer"), "previous_answer", 0.2)

        raw_pool = [str(record["candidate"]) for record in records]
        ranked_records = sorted(
            records,
            key=lambda record: (-float(record["priority"]), int(record["order"])),
        )
        ranked = deduplicate_bridge_candidates(
            (record["candidate"] for record in ranked_records),
            limit=24,
        )
        ranked_top8 = ranked[:8]
        deduplicated = deduplicate_bridge_candidates(raw_pool, limit=24)
        deduplicated_top8 = deduplicated[:8]
        return {
            "raw_pool": raw_pool,
            "ranked_top8": ranked_top8,
            "deduplicated_top8": deduplicated_top8,
            "visibility_filtered": list(ranked),
            "budget_filtered": list(ranked),
            "repair_target_filtered": list(ranked),
        }

    @staticmethod
    def _hotpot_dedup_strings(values: list[Any]) -> list[str]:
        return deduplicate_bridge_candidates(values, limit=24)

    @staticmethod
    def _hotpot_string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    def _hotpot_repair_noop_reason(self, action: dict[str, Any]) -> str:
        if action.get("op") != "repair_patch":
            return "repair_invalid_patch: solver did not produce a repair_patch"
        if action.get("action") == "STOP":
            return "repair_stop: solver explicitly preserved the current answer and evidence"
        return "repair_noop: solver patch produced no operationally new result"

    def _action_role_suffix(self, role: str, mode: str) -> str:
        """Give the small local model an explicit, state-aware next-action target."""
        if not self.enable_action_constraints:
            return "Emit exactly one Action JSON object now."
        missing = self._completion_errors(role, mode=mode)
        if missing:
            boundary = "; ".join(missing)
            instruction = f"Current stage is incomplete. Missing boundaries: {boundary}."
        else:
            instruction = "All required boundaries for this stage exist. Emit {\"op\":\"done\"}."
        # Multiple-choice Critic is an independent auditor, not a generic semantic
        # action generator.  Keep this branch before the generic role suffix below:
        # otherwise the generic example is returned first and the model never sees
        # the audit-specific rules (in particular, it tends to copy an
        # all-supported template without checking the question).
        if role == "critic" and self.task_type == "multiple_choice" and missing:
            return (
                "You are an INDEPENDENT multiple-choice auditor. The Solver answer and every "
                "Solver reason are untrusted claims; never accept them by default. Read the "
                "original question, the exact visible choice text, and the immutable kernel "
                "AUDIT MANIFEST before judging. For EVERY manifest slot_claim and EVERY "
                "elimination_claim, audit the complete Solver claim against the original question. "
                "Emit only assessment=fits when the term or option satisfies the source, "
                "assessment=violates when it does not, or assessment=unknown only when "
                "the required source text is missing or genuinely undecidable. The source in this "
                "prompt is complete, so generic Solver explanations are never a reason for "
                "insufficient. Do not classify isolated words. "
                "Audit the selected label itself as well. "
                "Do not propose a replacement answer and do not omit, merge, or invent claims. "
                "Use each exact claim_id once and copy candidate_version and manifest_digest "
                "exactly. The claim_id already identifies the option label or slot. Every reason "
                "MUST cite the concrete option text (or the exact selected term) and the relevant "
                "constraint from the original question; generic reasons such as 'fits the slot', "
                "'correctly eliminated', or 'matches the question' are invalid. Emit exactly one "
                "Before emitting each item, check reason polarity: if the reason says 'contradicts', "
                "'does not fit', 'wrong', or 'fails', assessment MUST be violates; if it says "
                "'fits', 'satisfies', or 'correctly completes', assessment MUST be fits. For a "
                "rejected option that does not satisfy the source, emit violates. Do not emit "
                "fits for every option. "
                "audit_choice Action JSON object and nothing else. The kernel, not the Solver or "
                "your prose, determines verified, need_fix, or uncertain from coverage and field consistency."
            )
        if role in {"planner", "solver", "critic"} and mode != "repair" and not self._is_hotpot_repair_mode(role, mode):
            return (
                instruction
                + " Emit exactly this domain-specific semantic Action shape: "
                + json.dumps(semantic_action_example(role, self.task_type), ensure_ascii=False, separators=(",", ":"))
                + ". Supply only semantic values and source citations. The kernel creates all nodes, edges, "
                "IDs, versions, provenance, visibility, completion state, and routing. Do not emit graph "
                "structure, propose_delta, legacy add_* operations, or set_result."
            )
        if role == "planner":
            if "missing query_spec" in missing:
                task = self.store.latest_valid(self.task_id, self.branch_id, "task")
                task_text = str(task.content) if task is not None else ""
                query_text = task_text if len(task_text) <= 512 else "task"
                return (
                    "The next required boundary is query_spec. Emit exactly this one JSON object "
                    "and nothing else: "
                    + json.dumps(
                        {"op": "declare_query", "content": {"question": query_text, "task_type": self.task_type}},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + ". Do not include a second object or duplicate keys."
                )
            if "missing plan" in missing or "missing plan_steps" in missing:
                if self.task_type in {"numeric_solve", "numeric_comparison"}:
                    inputs = self._numeric_fact_ids() or ["A", "B"]
                    return (
                        "The next required boundary is the numeric plan. Emit exactly this one JSON "
                        "object and nothing else: "
                        + json.dumps(
                            {"op": "add_plan_step", "id": "R1", "operation": "solve_numeric", "inputs": inputs},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + ". "
                        "Do not emit add_fact, do not include a derived total, do not include duplicate keys."
                    )
                return (
                    "The next required boundary is the workflow plan. Emit exactly this one JSON "
                    "object and nothing else: "
                    '{"op":"add_plan_step","id":"M1","operation":"synthesize","inputs":["task"]}. '
                    "Do not emit a result before the solver stage."
                )
            instruction += (
                " Emit only information explicitly present in the task as facts. "
                "Do not add a derived total as a fact. For numeric_solve, create the missing "
                "add_plan_step with operation and fact ids, then emit done; do not repeat an "
                "existing fact or plan step."
            )
        if role == "solver" and "missing result" in missing:
            if self.task_type == "multiple_choice":
                return (
                    "The next required boundary is result. Read the complete original question and "
                    "all choice texts before selecting. Choose one option, give one source-grounded "
                    "reason for every SLOT, and review every visible option exactly once. Each slot "
                    "reason MUST quote or name the selected term and the exact question constraint "
                    "it satisfies. Each option review MUST name the decisive term or slot from that "
                    "option; never emit generic phrases such as 'fits the question' or 'contradicts "
                    "the question'. "
                    "The kernel fixes the option labels; emit option_reviews with assessment selected "
                    "for the answer and rejected for every other label. Emit exactly one JSON object "
                    "and nothing else. Do not put pseudo-JSON or additional option labels inside a reason."
                )
            if self.task_type == "multihop_qa":
                return (
                    "The next required boundary is result. Answer the question using cited "
                    "SUPPORTING_FACT evidence. Emit exactly one JSON object and nothing else, "
                    'for example: {"op":"set_result","id":"answer","value":{"answer":"short answer",'
                    '"bridge_entity":"bridge","cited_fact_ids":["supporting_fact_1",'
                    '"supporting_fact_2"],"reasoning_chain":["supporting_fact_1 -> bridge",'
                    '"bridge + supporting_fact_2 -> answer"]}}. Bare text answers are invalid.'
                )
        if role == "solver" and mode == "repair":
            diagnosis = self._latest_repair_diagnosis()
            diagnosis_text = json.dumps(diagnosis, ensure_ascii=False, separators=(",", ":"))
            repair_directive = (
                f"Use this critic diagnosis exactly: {diagnosis_text}. "
                "Modify only the faulty part identified by error_location. "
                "Preserve every item listed in preserve. Do not repeat the same failed repair; "
                "the new artifact must address repair_instruction."
            )
            if self.task_type == "code_generation":
                return (
                    "The critic requested repair. "
                    + repair_directive
                    + " Emit revised Python code as exactly one JSON "
                    'object, for example: {"op":"emit_code","code":"def solution(...):\\n    ..."} . '
                    "Do not emit done until revised code exists after the latest need_fix verification."
                )
            if self.task_type in {"multiple_choice", "multihop_qa"}:
                if self.task_type == "multihop_qa":
                    return (
                        "The critic requested typed HotpotQA repair. "
                        + repair_directive
                        + " Use repair_packet.known as fixed context and satisfy repair_packet.required. "
                        "Emit exactly one repair_patch Action. For citation repairs use PATCH_CITATIONS "
                        "or ADD_EVIDENCE; for bridge repairs use PATCH_BRIDGE; for answer entailment "
                        "repairs use REVISE_ANSWER; use STOP only when the current result should be preserved. "
                        "Only REVISE_ANSWER may change answer."
                    )
                return (
                    "The critic requested repair. "
                    + repair_directive
                    + " The repair_packet fields contradicted_slots and contradicted_eliminations "
                    "are mandatory feedback: address every listed item. If forbidden_answers is "
                    "non-empty, the new answer MUST NOT use any of those labels. Emit exactly one "
                    "choose_option Action with the complete revised answer, slot_reasons, "
                    "option_reviews, and confidence. Prefer a label in repair_packet.viable_challengers "
                    "when present; these are options the Critic independently judged to fit "
                    "with the original question. Do not select an arbitrary unreviewed label merely "
                    "to satisfy the new-answer constraint. If required_change is audited_claims, "
                    "you may retain the answer only if you materially rewrite every listed reason "
                    "using the exact question constraint and option text; never copy the prior action. "
                    "The answer must be a new candidate, "
                    "not an audit or set_result Action. Do not emit done."
                )
            return (
                "The critic requested repair. "
                + repair_directive
                + " Emit exactly one revised result with set_result. Do not emit calculate, "
                "do not rebuild the full solution, and do not emit done until the revised result "
                "exists after the latest need_fix verification."
            )
        if role == "critic" and missing:
            if self.task_type == "code_generation":
                return (
                    "Verify the latest execution/result for the latest code. If tests pass, emit "
                    '{"op":"verify","target":"result","status":"verified","error_type":"",'
                    '"error_location":"","reason":"","repair_instruction":"","preserve":[],'
                    '"requested_fragments":[]}. If tests fail, emit need_fix with non-empty '
                    "error_type, error_location, and repair_instruction. The instruction must say "
                    "what to change and what to preserve; do not emit a bare need_fix flag."
                )
            if self.task_type == "multihop_qa":
                return (
                    "Verify the latest structured HotpotQA result as the semantic entailment "
                    "critic. Make three explicit judgments: citation_valid, chain_complete, and "
                    "answer_entailed and answers_question. For verified, reason MUST be a JSON string with exactly these "
                    "semantic fields: hop1_conclusion, hop2_conclusion, derived_answer, "
                    "candidate_answer, match, contradiction, answers_question. Only verify when the two hop "
                    "conclusions derive candidate_answer from the cited facts, match is true, and "
                    "contradiction is false, and derived_answer directly answers the question. For need_fix, use a typed error_type from "
                    "invalid_citation, missing_hop, incomplete_evidence_chain, wrong_bridge_entity, "
                    "answer_not_supported, or answer_contradicted, and include actionable "
                    "error_location, repair_instruction, preserve, and requested_fragments."
                )
            if self.task_type == "multiple_choice":
                return (
                    "Audit the Solver candidate against the complete original question and every visible choice. "
                    "The CRITIC INPUT block is the audit object: source is authoritative, solver_candidate is untrusted. "
                    "Emit exactly one audit_choice with exact candidate_version/manifest_digest and every required claim_id. "
                "For selected_option_check, slot_checks, and option_checks, output only assessment=fits, violates, or unknown plus a concrete reason. "
                "Do not output claim verdicts: the kernel derives supported or contradicted from the Solver assessment in the immutable manifest. "
                "Audit the complete claim_text bound to each claim_id, not whether an isolated option word looks compatible. "
                "For every option, fits means the option text satisfies the original question, and violates means it does not. "
                "For every slot, fits means the Solver term fits that slot and violates means it does not. "
                "The source question and all option texts are complete in this input. Do not use unknown merely because the Solver's explanation is generic; judge the term or option directly against the source. Use unknown only when the required source text is missing or genuinely undecidable. "
                "Before emitting each item, check the polarity of your own reason: if the reason says 'contradicts', 'does not fit', 'wrong', or 'fails', the assessment MUST be violates; if it says 'fits', 'satisfies', or 'correctly completes', the assessment MUST be fits. For a rejected Solver option that does not satisfy the source, emit violates (the kernel will derive that the Solver's rejection was supported). Do not emit fits for every option. "
                "The kernel combines this independent assessment with selected/rejected to derive claim status and routing. "
                    "Reasons must cite concrete option/term text. Do not invent IDs, alter Solver fields, or generate a replacement answer."
                )
            return (
                "Verify the latest result. If correct, emit verify with status=verified and empty "
                "diagnostic fields. If incorrect, emit need_fix with non-empty error_type, "
                "error_location, and repair_instruction; do not emit a bare need_fix flag."
            )
        return f"{instruction} Emit exactly one new Action JSON object now."

    def _numeric_fact_ids(self) -> list[str]:
        ids = [
            node.logical_id.removeprefix("fact_")
            for node in self.store.snapshot().nodes.values()
            if node.task_id == self.task_id
            and node.branch_id == self.branch_id
            and node.logical_id.startswith("fact_")
            and node.is_operationally_valid()
        ]

        def key(value: str) -> tuple[int, str]:
            if value.startswith("N") and value[1:].isdigit():
                return int(value[1:]), value
            return 10**9, value

        return sorted(set(ids), key=key)

    def _completion_errors(self, role: str, *, mode: str = "normal") -> list[str]:
        if role in {"planner", "solver"}:
            spec = get_task_spec(self.task_type)
            boundary = self.store.latest_valid(
                self.task_id,
                self.branch_id,
                f"semantic_delta_{role}_boundary",
            )
            errors: list[str] = []
            if boundary is None:
                errors.append(f"missing semantic_delta boundary for {role}")
            state_nodes = [
                node
                for node in self.store.snapshot().nodes.values()
                if node.task_id == self.task_id
                and node.branch_id == self.branch_id
                and node.is_operationally_valid()
            ]
            for required_type in spec.required_for(role):
                if not any(node.type == required_type and node.owner == role for node in state_nodes):
                    errors.append(f"missing required semantic output {required_type}")
            if mode == "repair":
                anchor = self.store.latest_valid(self.task_id, self.branch_id, "verification")
                if anchor is None or anchor.status not in {"need_fix", "uncertain"}:
                    anchor = self.store.latest_valid(self.task_id, self.branch_id, "error")

                def is_newer(node: Any) -> bool:
                    if node is None or anchor is None:
                        return False
                    return (
                        float(node.created_at or 0.0), node.version, node.node_id
                    ) > (
                        float(anchor.created_at or 0.0), anchor.version, anchor.node_id
                    )

                if self.task_type == "multihop_qa":
                    repaired_result = self.store.latest_valid(
                        self.task_id,
                        self.branch_id,
                        "result",
                    )
                    return (
                        []
                        if is_newer(repaired_result)
                        else ["missing post-verification Hotpot repair result"]
                    )

                if not is_newer(boundary):
                    errors.append("missing post-verification semantic_delta boundary for solver repair")
                for required_type in spec.required_for(role):
                    if not any(
                        node.type == required_type
                        and node.owner == role
                        and is_newer(node)
                        for node in state_nodes
                    ):
                        errors.append(f"missing revised semantic output {required_type}")
            return list(dict.fromkeys(errors))

        def exists(logical_id: str) -> bool:
            return self.store.latest_valid(self.task_id, self.branch_id, logical_id) is not None

        def latest(logical_id: str):
            return self.store.latest_valid(self.task_id, self.branch_id, logical_id)

        def newer_than(node: Any, reference: Any) -> bool:
            if node is None or reference is None:
                return False
            return (
                float(getattr(node, "created_at", 0.0) or 0.0),
                int(getattr(node, "version", 0) or 0),
                str(getattr(node, "node_id", "")),
            ) > (
                float(getattr(reference, "created_at", 0.0) or 0.0),
                int(getattr(reference, "version", 0) or 0),
                str(getattr(reference, "node_id", "")),
            )

        def repair_anchor():
            verification = latest("verification")
            if verification is not None and verification.status in {"need_fix", "uncertain"}:
                state = self.store.snapshot()
                target_edge = next(
                    (
                        edge for edge in state.edges
                        if edge.source == verification.node_id
                        and edge.relation == "contradicts"
                    ),
                    None,
                )
                if target_edge is not None:
                    target = state.nodes.get(target_edge.target)
                    if target is not None:
                        return target
                return verification
            return latest("error")

        def current_critic_target():
            if self.task_type == "code_generation":
                return latest("result") or latest("execution") or latest("code")
            return latest("result")

        def result_has_value() -> bool:
            result = latest("result")
            if result is None:
                return False
            content = result.content
            if isinstance(content, dict) and "value" in content:
                return content.get("value") is not None
            return content is not None

        errors: list[str] = []
        if role == "planner":
            required = {
                "numeric_solve": {"query_spec", "facts", "plan", "plan_steps"},
                "numeric_comparison": {"query_spec", "facts", "plan", "plan_steps"},
                "table_qa": {"query_spec", "table", "plan", "plan_steps"},
                "multihop_qa": {"query_spec", "plan", "plan_steps"},
                "multiple_choice": {"query_spec", "plan", "plan_steps"},
                "code_generation": {"query_spec", "requirements", "plan", "plan_steps"},
                "marble_research": {"query_spec", "plan", "plan_steps"},
                "marble_bargaining": {"query_spec", "plan", "plan_steps"},
                "marble_database": {"query_spec", "requirements", "plan", "plan_steps"},
            }[self.task_type]
            errors.extend(f"missing {item}" for item in sorted(required) if not exists(item))
            nodes = self.store.snapshot().nodes.values()
            if self.task_type == "table_qa" and not any(node.logical_id.startswith("table_cell_") for node in nodes):
                errors.append("missing table_cell")
            if self.task_type == "multihop_qa" and not any(node.type in {"entity", "supporting_fact", "evidence_link"} for node in nodes):
                errors.append("missing multihop evidence")
            if self.task_type == "multiple_choice" and not any(node.type == "choice" for node in nodes):
                errors.append("missing choice")
        elif role == "solver":
            anchor = repair_anchor() if mode == "repair" else None
            if self.task_type == "code_generation":
                code = latest("code")
                if mode == "repair":
                    errors.append("missing revised code") if not newer_than(code, anchor) else None
                else:
                    errors.append("missing code") if code is None else None
            elif self.task_type in {"multiple_choice", "multihop_qa"}:
                result = latest("result")
                if mode == "repair":
                    errors.append("missing revised result") if not newer_than(result, anchor) else None
                else:
                    errors.append("missing result") if result is None else None
            elif self.task_type not in {"marble_research", "marble_bargaining", "marble_database"}:
                calculation = latest("calculation")
                result = latest("result")
                if mode == "repair":
                    errors.append("missing revised result") if not newer_than(result, anchor) else None
                else:
                    if self.task_type in {"numeric_solve", "numeric_comparison"} and calculation is not None and result_has_value():
                        pass
                    elif not result_has_value():
                        errors.append("missing calculation") if calculation is None else None
                        errors.append("missing result") if result is None else None
            else:
                result = latest("result")
                if mode == "repair":
                    errors.append("missing revised result") if not newer_than(result, anchor) else None
                else:
                    errors.append("missing result") if result is None else None
        elif role == "critic":
            verification = latest("verification")
            target = current_critic_target()
            if verification is None:
                errors.append("missing verification")
            elif (
                self.task_type == "multiple_choice"
                and verification.status == "uncertain"
                and isinstance(verification.content, Mapping)
                and verification.content.get("error_type") in {
                    "invalid_choice_audit_scope", "invalid_choice_audit_manifest",
                    "invalid_choice_audit_consistency", "invalid_choice_audit_reason",
                }
            ):
                # An incomplete audit is a Critic-owned protocol failure.  It
                # must re-enter Critic, rather than being misrouted as a Solver
                # semantic repair.
                errors.append("critic must complete the kernel audit manifest")
            elif target is not None and not newer_than(verification, target):
                errors.append("missing updated verification")
        elif role == "final_solver":
            final_answer = latest("final_answer")
            verification = latest("verification")
            if final_answer is None:
                errors.append("missing final_answer")
            elif verification is not None and verification.status == "verified" and not newer_than(final_answer, verification):
                errors.append("missing updated final_answer")
        return errors

    def _tool_node(self, state: WorkflowState) -> dict[str, Any]:
        if self.task_type != "code_generation":
            return {"status": "running"}
        branch_id = state.get("branch_id", self.branch_id)
        code_node = self.store.latest_valid(self.task_id, branch_id, "code")
        if code_node is None:
            return {"error": "solver did not produce code", "status": "failed"}
        code_value = (
            code_node.content.get("code", code_node.content)
            if isinstance(code_node.content, Mapping)
            else code_node.content
        )
        if self.evaluation_context is not None:
            verification = CodeVerifier().verify(
                str(code_value),
                public_tests=(),
                private_tests=self.evaluation_context.official_tests or self.evaluation_context.private_tests,
                evaluation=self.evaluation_context,
            )
            execution = DomainExecution(
                success=verification.passed,
                value={"code": code_value, "tests_passed": verification.passed},
                execution=verification.as_dict(),
                errors=[verification.failure_summary] if not verification.passed else [],
            )
        else:
            execution = execute_domain(self.task_type, self.store, {"code": code_value})
        safe_context = self.evaluation_context or EvaluationContext(
            task_id=self.task_id,
            task_type=self.task_type,
        )
        execution = DomainExecution(
            success=execution.success,
            value=execution.value,
            execution=safe_context.safe_verification_summary(
                execution.execution,
                private=self.evaluation_context is not None,
            ),
            errors=[] if execution.success else [str(execution.execution.get("kind", "verification_failure"))],
        )
        node = self.store.add_node(
            task_id=self.task_id,
            branch_id=branch_id,
            logical_id="execution",
            node_type="execution",
            content=execution.execution,
            owner="tool",
            status="verified" if execution.success else "need_fix",
            validation={"schema_valid": True, "execution_valid": execution.success},
            created_by_role="tool",
        )
        self.store.add_edge(source=code_node.node_id, target=node.node_id, relation="produces", created_by_role="tool")
        result_value = execution.value if execution.success else {
            "code": code_value,
            "tests_passed": False,
            "errors": list(execution.errors),
        }
        result = self.store.add_node(
            task_id=self.task_id,
            branch_id=branch_id,
            logical_id="result",
            node_type="result",
            content=result_value,
            owner="tool",
            status="ready",
            validation={"schema_valid": True, "execution_valid": execution.success},
            created_by_role="tool",
        )
        self.store.add_edge(source=node.node_id, target=result.node_id, relation="produces", created_by_role="tool")
        self.agent_views.grant("tool", node_ids=[node.node_id, result.node_id], local=True)
        self._communicate_nodes(
            sender="tool",
            node_ids=[node.node_id, result.node_id],
            branch_id=branch_id,
            workflow_state=state,
            decision_context={"mode": "tool_execution", "action_index": 0},
        )
        return {"status": "running", "error": None if execution.success else "; ".join(execution.errors)}

    def _unresolved_verification_node(self, state: WorkflowState) -> dict[str, Any]:
        branch_id = state.get("branch_id", self.branch_id)
        verification = self.store.latest_valid(self.task_id, branch_id, "verification")
        content = verification.content if verification is not None and isinstance(verification.content, dict) else {}
        error = content.get("reason") or content.get("repair_instruction") or "verification unresolved"
        logs = list(state.get("logs", []))
        logs.append({
            "role": "runtime",
            "mode": "verification_unresolved",
            "action_index": 0,
            "action": {
                "op": "verification_unresolved",
                "status": getattr(verification, "status", "missing") if verification is not None else "missing",
                "error_type": content.get("error_type", "verification_unresolved"),
                "error_location": content.get("error_location", "verification"),
            },
            "compiled_node_ids": [],
            "graph_update_tokens": 0,
            "communication": [],
            "context_tokens": 0,
            "context_slice_tokens": 0,
            "context_slice_node_ids": [],
            "context_slice_edge_ids": [],
            "context_slice_fragment_ids": [],
            "context_slice_root_node_ids": [],
            "rendered_context_node_ids": [],
            "rendered_context_fragment_ids": [],
            "logical_context_tokens": 0,
            "generation": {"attempts": 0, "validation_errors": [], "verification_unresolved": True},
            "telemetry": {},
            "session": self._session_snapshot(state.get("session_id", self.session_id)),
            "latency_sec": 0.0,
        })
        return {
            "status": "verification_unresolved",
            "error": str(error),
            "last_role": "runtime",
            "session_id": state.get("session_id", self.session_id),
            "logs": logs,
        }

    def _route_after_critic(self, state: WorkflowState) -> str:
        verification = self.store.latest_valid(self.task_id, state.get("branch_id", self.branch_id), "verification")
        if verification is not None and verification.status == "verified" and not self._completion_errors("critic"):
            return "finalizer"
        if (
            verification is not None
            and verification.status == "uncertain"
            and isinstance(verification.content, Mapping)
            and verification.content.get("error_type") in {
                "invalid_choice_audit_scope", "invalid_choice_audit_manifest",
                "invalid_choice_audit_consistency", "invalid_choice_audit_reason",
            }
        ):
            critic_attempts = sum(
                1 for item in state.get("logs", [])
                if isinstance(item, Mapping) and item.get("role") == "critic"
            )
            if critic_attempts < 2:
                return "critic"
            return "unresolved"
        if int(state.get("round_id", 0)) >= self.max_rounds:
            return "unresolved"
        return "repair"


class NativeWorkflowState(TypedDict, total=False):
    """State for the unoptimized, natural-language LangGraph baseline.

    This state deliberately contains message-bus envelopes instead of graph
    nodes or Action objects. Each node receives messages from its mailbox and
    returns free-form text plus resulting transport/tool state.
    """

    task_id: str
    task_type: str
    task: str
    round_id: int
    status: str
    planner_output: str
    solver_output: str
    repair_output: str
    tool_output: str
    tool_attempts: int
    tool_retry_count: int
    tool_failures: int
    tool_requests: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    environment_state: dict[str, Any]
    tool_failed: bool
    critic_output: str
    critic_verdict: dict[str, Any]
    final_answer: str
    # Serialized MessageEnvelope values from the native message bus.
    messages: list[dict[str, Any]]
    agent_snapshots: dict[str, dict[str, Any]]


class NativeLangGraphWorkflow:
    """Natural-language LangGraph baseline without the optimized protocol.

    Native mode intentionally does not use GraphStore, ActionCompiler,
    Action parsing, or constrained decoding. It retains only the runtime
    topology so the comparison is between a normal LangGraph workflow and the
    optimized workflow. Each role is nevertheless
    represented by a real ``NativeAgent`` with its own identity, memory,
    policy and lifecycle.  The default agents may share one model provider to
    avoid loading duplicate weights; ``agent_models`` or ``agents`` can inject
    separate backends.
    """

    _DEFAULT_SYSTEM_PROMPTS = {
        "planner": (
            "You are the planner in a normal multi-agent workflow. Analyze the task and "
            "give the next agent a concise, useful natural-language plan. Do not emit "
            "Action JSON, Graph-Delta, or protocol envelopes."
        ),
        "solver": (
            "You are the solver in a normal multi-agent workflow. Solve the task using the "
            "provided context. Return a natural-language candidate solution or code. If a "
            "deterministic runtime tool is needed, request it with exactly "
            "<tool_call>{\"name\":\"calculator\",\"arguments\":{\"expression\":\"2+3\"}}</tool_call>. "
            "Do not emit Action JSON, Graph-Delta, or protocol envelopes."
        ),
        "critic": (
            "You are the critic in a normal multi-agent workflow. Check the candidate for "
            "correctness and completeness. Explain any concrete correction in natural language. "
            "Then emit exactly one control envelope using this schema: "
            "<critic_verdict>{\"verdict\":\"pass|needs_repair|reject|uncertain\","
            "\"confidence\":0.0,\"target\":\"candidate\",\"issues\":[],"
            "\"repair_instructions\":\"\"}</critic_verdict>. "
            "For needs_repair or reject, repair_instructions must be actionable. "
            "Do not emit Action JSON or Graph-Delta."
        ),
        "final_solver": (
            "You are the finalizer in a normal multi-agent workflow. Produce the answer the "
            "user should receive from the supplied evidence. Do not emit Action JSON, "
            "Graph-Delta, or protocol envelopes."
        ),
    }

    def __init__(
        self,
        *,
        model: ModelCallable,
        task_id: str,
        task: str,
        task_type: str,
        max_rounds: int = 3,
        max_tool_retries: int = 2,
        critic_confidence_threshold: float = 0.7,
        telemetry: Optional[WorkflowTelemetry] = None,
        agents: Mapping[str, NativeAgent] | None = None,
        agent_models: Mapping[str, ModelCallable] | None = None,
        agent_factory: Callable[[AgentConfig], NativeAgent] | None = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.model = model
        self.task_id = task_id
        self.task = task
        self.task_type = task_type
        self.max_rounds = max(1, int(max_rounds))
        self.max_tool_retries = max(0, int(max_tool_retries))
        self.critic_confidence_threshold = min(1.0, max(0.0, float(critic_confidence_threshold)))
        self.telemetry = telemetry
        self._native_message_ids: set[str] = set()
        self.tool_registry = tool_registry or ToolRegistry.default()
        self.agents = self._build_agents(
            model,
            agents=agents,
            agent_models=agent_models,
            agent_factory=agent_factory,
        )
        self.message_bus = MessageBus(conversation_id=f"native:{self.task_id}")
        for actor_id in ("system", "user", "tool"):
            self.message_bus.register(actor_id)
        for agent in self.agents.values():
            self.message_bus.register(agent.agent_id)
        self.message_bus.broadcast(
            "system",
            self.task,
            recipients=[agent.agent_id for agent in self.agents.values()],
            message_type="task",
            correlation_id=self.task_id,
            metadata={"task_type": self.task_type},
        )

    def _build_agents(
        self,
        model: ModelCallable,
        *,
        agents: Mapping[str, NativeAgent] | None,
        agent_models: Mapping[str, ModelCallable] | None,
        agent_factory: Callable[[AgentConfig], NativeAgent] | None,
    ) -> dict[str, NativeAgent]:
        roles = tuple(self._DEFAULT_SYSTEM_PROMPTS)
        if agents is not None:
            result: dict[str, NativeAgent] = {}
            for key, agent in agents.items():
                if not isinstance(agent, NativeAgent):
                    raise TypeError(f"native agent {key!r} must be a NativeAgent")
                if key != agent.role:
                    raise ValueError(f"native agent key {key!r} does not match role {agent.role!r}")
                result[key] = agent
            missing = [role for role in roles if role not in result]
            if missing:
                raise ValueError(f"native agents missing roles: {', '.join(missing)}")
            return result

        result = {}
        agent_models = agent_models or {}
        tool_permissions = {
            "planner": frozenset(),
            "solver": frozenset({"calculator", "python_syntax_check", "json_validate"}),
            "critic": frozenset(),
            "final_solver": frozenset(),
        }
        for role in roles:
            config = AgentConfig(
                agent_id=f"native:{self.task_id}:{role}",
                role=role,
                system_prompt=self._DEFAULT_SYSTEM_PROMPTS[role],
                tool_permissions=tool_permissions[role],
                metadata={"task_id": self.task_id, "task_type": self.task_type},
            )
            if agent_factory is not None:
                agent = agent_factory(config)
                if not isinstance(agent, NativeAgent):
                    raise TypeError(f"agent_factory returned {type(agent).__name__}, expected NativeAgent")
            else:
                agent = NativeAgent(
                    config=config,
                    model=agent_models.get(role, model),
                    tools=self.tool_registry,
                )
            if agent.role != role:
                raise ValueError(f"agent for role {role!r} has role {agent.role!r}")
            result[role] = agent
        return result

    def agent_snapshot(self) -> dict[str, dict[str, Any]]:
        """Return observable identity, policy and lifecycle state for all agents."""
        return {role: agent.snapshot() for role, agent in self.agents.items()}

    def _agent(self, role: str) -> NativeAgent:
        try:
            return self.agents[role]
        except KeyError as exc:
            raise KeyError(f"native workflow has no agent for role {role!r}") from exc

    def _complete_agents(self) -> None:
        for agent in self.agents.values():
            if agent.lifecycle.state in {"created", "active", "suspended"}:
                agent.complete()

    def compile(self, *, checkpointer: Any = None) -> Any:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("LangGraph is required. Install it with: pip install langgraph") from exc

        graph = StateGraph(NativeWorkflowState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("solver", self._solver_node)
        graph.add_node("tool", self._tool_node)
        graph.add_node("critic", self._critic_node)
        graph.add_node("repair", self._repair_node)
        graph.add_node("finalizer", self._finalizer_node)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "solver")
        graph.add_edge("solver", "tool")
        graph.add_conditional_edges(
            "tool",
            self._route_after_tool,
            {"repair": "repair", "critic": "critic"},
        )
        graph.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {"repair": "repair", "finalizer": "finalizer"},
        )
        graph.add_edge("repair", "tool")
        graph.add_edge("finalizer", END)
        return graph.compile(checkpointer=checkpointer)

    def initial_state(self) -> NativeWorkflowState:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "task": self.task,
            "round_id": 0,
            "status": "started",
            "tool_attempts": 0,
            "tool_retry_count": 0,
            "tool_failures": 0,
            "tool_requests": [],
            "tool_results": [],
            "environment_state": {
                "task_id": self.task_id,
                "task_type": self.task_type,
                "tool_call_count": 0,
                "last_tool_call": None,
                "last_tool_result": None,
            },
            "tool_failed": False,
            "critic_verdict": {},
            **self._message_state(),
        }

    def _restore_message_bus(self, state: NativeWorkflowState) -> None:
        """Restore the transport from LangGraph state when resuming a run."""
        checkpoint_messages = state.get("messages")
        if checkpoint_messages and checkpoint_messages != self.message_bus.transcript():
            self.message_bus.restore(checkpoint_messages)

    def _message_state(self) -> dict[str, Any]:
        transcript = self.message_bus.transcript()
        if self.telemetry is not None:
            for message in transcript:
                if message["message_id"] in self._native_message_ids:
                    continue
                self._native_message_ids.add(message["message_id"])
                self.telemetry.record_native_message(message)
        return {"messages": transcript}

    def _receive_for(
        self,
        role: str,
        state: NativeWorkflowState,
    ) -> list[MessageEnvelope]:
        """Read the role's pending mailbox messages for one node turn."""
        self._restore_message_bus(state)
        agent = self._agent(role)
        # Leave messages pending until the node finishes successfully.  A
        # LangGraph retry can therefore redeliver the same envelope instead of
        # losing it after a model or downstream failure.
        incoming = agent.receive_messages(self.message_bus, mark_delivered=False)
        if not incoming:
            raise RuntimeError(f"agent {agent.agent_id} mailbox is empty")
        return incoming

    def _acknowledge(self, actor_id: str, messages: Sequence[MessageEnvelope]) -> None:
        if messages:
            self.message_bus.acknowledge(actor_id, [message.message_id for message in messages])

    @staticmethod
    def _render_inbox(messages: Sequence[MessageEnvelope]) -> str:
        """Render only messages delivered to one agent's mailbox."""
        sections: list[str] = []
        for message in messages:
            recipients = ",".join(message.recipient_ids)
            headers = [
                f"MESSAGE_ID: {message.message_id}",
                f"FROM: {message.sender_id}",
                f"TO: {recipients}",
                f"TYPE: {message.message_type}",
            ]
            if message.reply_to:
                headers.append(f"REPLY_TO: {message.reply_to}")
            if message.correlation_id:
                headers.append(f"CORRELATION_ID: {message.correlation_id}")
            sections.append("\n".join(headers) + f"\nCONTENT:\n{message.content}")
        return "\n\n".join(sections)

    def _call_role(self, role: str, prompt: str, state: NativeWorkflowState) -> str:
        agent = self._agent(role)
        request = ModelRequest(
            role=role,
            task_type=self.task_type,
            prompt=prompt,
            mode="native",
            session_id=agent.session_id,
            session_prompt=prompt,
            session_reset=True,
            session_rollback=False,
            action_constraint=None,
        )
        raw = agent.invoke(request)
        text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
        metrics = self._last_call_metrics(agent.session_id, agent.model)
        if self.telemetry is not None:
            self.telemetry.record_native_call({
                "role": role,
                "agent_id": agent.agent_id,
                "mode": "native",
                "round_id": int(state.get("round_id", 0)),
                "prompt": prompt,
                "raw_output": text,
                "call_metrics": metrics,
            })
        return text.strip()

    def _render_prompt(self, role: str, user_content: str) -> str:
        """Render one independent standard chat request for the native baseline."""
        agent = self._agent(role)
        system = agent.config.system_prompt
        memory = agent.render_memory()
        if memory:
            user_content = f"{user_content}\n\n{memory}"
        tokenizer = getattr(agent.model, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        # Test doubles and generic callables may not expose a tokenizer. Keep
        # the same two-message boundary in the plain representation.
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

    def _last_call_metrics(self, session_id: str, model: Any = None) -> dict[str, Any]:
        snapshot = getattr(model or self.model, "session_snapshot", None)
        if not callable(snapshot):
            return {}
        value = snapshot(session_id)
        metrics = value.get("last_call_metrics", {}) if isinstance(value, dict) else {}
        return dict(metrics) if isinstance(metrics, dict) else {}

    def _send_agent_message(
        self,
        sender_role: str,
        recipient_role: str,
        content: str,
        *,
        message_type: str,
        reply_to: Sequence[MessageEnvelope] = (),
        correlation_id: str | None = None,
    ) -> MessageEnvelope:
        sender = self._agent(sender_role)
        recipient_id = (
            self._agent(recipient_role).agent_id
            if recipient_role in self.agents
            else recipient_role
        )
        return sender.send_message(
            self.message_bus,
            recipient_id,
            content,
            message_type=message_type,
            reply_to=reply_to[-1].message_id if reply_to else None,
            correlation_id=correlation_id or self.task_id,
            metadata={"sender_role": sender_role, "recipient_role": recipient_role},
        )

    def _planner_node(self, state: NativeWorkflowState) -> dict[str, Any]:
        incoming = self._receive_for("planner", state)
        prompt = self._render_prompt(
            "planner",
            self._render_inbox(incoming),
        )
        output = self._call_role("planner", prompt, state)
        self._acknowledge(self._agent("planner").agent_id, incoming)
        self._send_agent_message(
            "planner",
            "solver",
            output,
            message_type="plan",
            reply_to=incoming,
        )
        return {
            "planner_output": output,
            "status": "planned",
            **self._message_state(),
        }

    def _solver_node(self, state: NativeWorkflowState) -> dict[str, Any]:
        incoming = self._receive_for("solver", state)
        prompt = self._render_prompt(
            "solver",
            self._render_inbox(incoming),
        )
        output = self._call_role("solver", prompt, state)
        self._acknowledge(self._agent("solver").agent_id, incoming)
        self._send_agent_message(
            "solver",
            "tool",
            output,
            message_type="candidate",
            reply_to=incoming,
        )
        return {
            "solver_output": output,
            "status": "solved",
            **self._message_state(),
        }

    def _execute_native_tool_call(
        self,
        solver_agent: NativeAgent,
        call: NativeToolCall,
    ) -> NativeToolResult:
        if not solver_agent.can_use_tool(call.name):
            return NativeToolResult(
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
                ok=False,
                error=f"agent {solver_agent.agent_id} is not allowed to use tool {call.name}",
                source=call.source,
            )
        try:
            output = solver_agent.execute_tool(call.name, call.arguments)
            return NativeToolResult(
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
                ok=True,
                output=output,
                source=call.source,
            )
        except Exception as error:
            return NativeToolResult(
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
                ok=False,
                error=str(error),
                source=call.source,
            )

    def _tool_node(self, state: NativeWorkflowState) -> dict[str, Any]:
        self._restore_message_bus(state)
        incoming = self.message_bus.receive("tool", mark_delivered=False)
        if not incoming:
            raise RuntimeError("tool mailbox is empty")
        candidate = incoming[-1].content
        solver_agent = self._agent("solver")
        attempt = int(state.get("tool_attempts", 0)) + 1
        call_prefix = f"{self.task_id}-tool-{attempt}"
        parse_error: str | None = None
        try:
            calls = parse_tool_calls(candidate, call_prefix=call_prefix)
        except ValueError as error:
            calls = []
            parse_error = str(error)
        if not calls and parse_error is None:
            calls = infer_default_tool_calls(
                self.task_type,
                self.task,
                candidate,
                call_prefix=call_prefix,
            )

        if parse_error is not None:
            results = [NativeToolResult(
                call_id=f"{call_prefix}-parse",
                name="tool_call_parser",
                arguments={},
                ok=False,
                error=parse_error,
                source="parser",
            )]
        elif calls:
            results = [self._execute_native_tool_call(solver_agent, call) for call in calls]
        else:
            # A tool stage still produces a structured observation when no
            # tool is requested; it no longer fabricates a generic prose
            # message or silently discards the candidate.
            results = [NativeToolResult(
                call_id=f"{call_prefix}-not-requested",
                name="none",
                arguments={},
                ok=True,
                output={"status": "not_requested", "reason": "candidate did not request a registered tool"},
                source="none",
            )]

        failed = any(not result.ok for result in results)
        failure_count = sum(not result.ok for result in results)
        tool_output = render_tool_results(results, candidate)
        serialized_requests = [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": dict(call.arguments),
                "source": call.source,
            }
            for call in calls
        ]
        serialized_results = [result.as_dict() for result in results]
        previous_requests = list(state.get("tool_requests", []))
        previous_results = list(state.get("tool_results", []))
        environment = dict(state.get("environment_state", {}))
        history = list(environment.get("tool_history", []))
        history.extend(serialized_results)
        environment.update({
            "tool_call_count": int(environment.get("tool_call_count", 0)) + len(calls),
            "last_tool_call": calls[-1].__dict__ if calls else None,
            "last_tool_result": serialized_results[-1],
            "tool_history": history[-32:],
        })
        retry_count = int(state.get("tool_retry_count", 0)) + 1 if failed else 0
        self.message_bus.acknowledge("tool", [message.message_id for message in incoming])
        self.message_bus.broadcast(
            "tool",
            tool_output,
            recipients=[self._agent("critic").agent_id, self._agent("final_solver").agent_id],
            message_type="tool_result",
            reply_to=incoming[-1].message_id,
            correlation_id=self.task_id,
            metadata={"task_type": self.task_type},
        )
        if failed and retry_count <= self.max_tool_retries:
            self.message_bus.send(
                "tool",
                solver_agent.agent_id,
                tool_output,
                message_type="tool_error",
                correlation_id=self.task_id,
                metadata={"retry_count": retry_count},
            )
        return {
            "tool_output": tool_output,
            "status": "tooled",
            "tool_attempts": attempt,
            "tool_retry_count": retry_count,
            "tool_failures": int(state.get("tool_failures", 0)) + failure_count,
            "tool_requests": previous_requests + serialized_requests,
            "tool_results": previous_results + serialized_results,
            "environment_state": environment,
            "tool_failed": failed,
            **self._message_state(),
        }

    def _route_after_tool(self, state: NativeWorkflowState) -> str:
        if state.get("tool_failed") and int(state.get("tool_retry_count", 0)) <= self.max_tool_retries:
            return "repair"
        return "critic"

    def _critic_node(self, state: NativeWorkflowState) -> dict[str, Any]:
        incoming = self._receive_for("critic", state)
        prompt = self._render_prompt("critic", self._render_inbox(incoming))
        output = self._call_role("critic", prompt, state)
        verdict = parse_critic_verdict(output)
        verdict_payload = verdict.as_dict()
        feedback = render_critic_feedback(verdict, output)
        self._acknowledge(self._agent("critic").agent_id, incoming)
        route = self._route_after_critic({**state, "critic_output": output, "critic_verdict": verdict_payload})
        if route == "repair":
            self._send_agent_message(
                "critic",
                "solver",
                feedback,
                message_type="critique",
                reply_to=incoming,
            )
        self._send_agent_message(
            "critic",
            "final_solver",
            feedback,
            message_type="critique",
            reply_to=incoming,
        )
        return {
            "critic_output": output,
            "critic_verdict": verdict_payload,
            "status": "criticized",
            **self._message_state(),
        }

    def _repair_node(self, state: NativeWorkflowState) -> dict[str, Any]:
        incoming = self._receive_for("solver", state)
        prompt = self._render_prompt(
            "solver",
            "Repair the candidate using the critic's feedback or tool error and return a corrected solution.\n\n"
            + self._render_inbox(incoming),
        )
        output = self._call_role("solver", prompt, state)
        self._acknowledge(self._agent("solver").agent_id, incoming)
        self._send_agent_message(
            "solver",
            "tool",
            output,
            message_type="repair_candidate",
            reply_to=incoming,
        )
        return {
            "repair_output": output,
            "round_id": int(state.get("round_id", 0)) + 1,
            "status": "repaired",
            **self._message_state(),
        }

    def _finalizer_node(self, state: NativeWorkflowState) -> dict[str, Any]:
        incoming = self._receive_for("final_solver", state)
        prompt = self._render_prompt(
            "final_solver",
            "Return only the final answer or solution using this workflow context.\n\n"
            + self._render_inbox(incoming),
        )
        output = self._call_role("final_solver", prompt, state)
        self._acknowledge(self._agent("final_solver").agent_id, incoming)
        self._agent("final_solver").send_message(
            self.message_bus,
            "user",
            output,
            message_type="final_answer",
            reply_to=incoming[-1].message_id,
            correlation_id=self.task_id,
        )
        self._complete_agents()
        return {
            "final_answer": output,
            "status": "completed",
            **self._message_state(),
            "agent_snapshots": self.agent_snapshot(),
        }

    def _route_after_critic(self, state: NativeWorkflowState) -> str:
        if int(state.get("round_id", 0)) >= self.max_rounds:
            return "finalizer"
        verdict = state.get("critic_verdict", {})
        if not isinstance(verdict, Mapping):
            return "finalizer"
        if not verdict.get("valid", False):
            return "finalizer"
        return (
            "repair"
            if verdict.get("verdict") in REPAIR_VERDICTS
            and float(verdict.get("confidence", 0.0)) >= self.critic_confidence_threshold
            else "finalizer"
        )


def build_langgraph_workflow(**kwargs: Any) -> LangGraphWorkflow:
    return LangGraphWorkflow(**kwargs)
