"""Model-independent versioned workflow runtime primitives."""

from .canonical import (
    canonical_json,
    dependency_digest,
    full_digest,
    node_content_digest,
    node_order_digest,
    runtime_fingerprint_digest,
)
from .context_slicer import (
    ROLE_REQUIREMENTS,
    SliceError,
    build_context_slice,
    render_compact_context_slice,
    render_context_slice,
    render_full_graph_context,
    upstream_closure,
)
from .graph_store import BranchError, GraphStore, GraphTransaction
from .invalidation import (
    downstream_dependents,
    invalidate_downstream,
    plan_local_recompute,
)
from .metrics import CallRecord, RuntimeMetrics
from .models import (
    CachedResult,
    ContextSlice,
    DEFAULT_BRANCH_ID,
    DependencyEdge,
    GraphState,
    InvalidationReport,
    NodeVersion,
    RecomputePlan,
    RuntimeFingerprint,
)
from .reuse_cache import InMemoryResultCache, ResultCache
from .domain_executors import DomainExecution, execute_domain
from .action_compiler import ActionCompilationError, ActionCompiler, CompilationResult
from .agent_graph_view import AgentGraphView, AgentGraphViewManager
from .graph_delta import DeltaCandidate, GraphDelta
from .communication import (
    ClosureAwareHeuristicPolicy,
    GraphCommunicationPolicy,
    MinimalNoFeedbackPolicy,
    MinimalSendAllFallbackPolicy,
    MinimalTargetedFeedbackPolicy,
    RandomKeepPolicy,
    SendAllPolicy,
)
from .langgraph_workflow import LangGraphWorkflow, WorkflowState, build_langgraph_workflow
from .native_agents import AgentConfig, AgentLifecycle, AgentMemory, NativeAgent
from .message_bus import MessageBus, MessageBusError, MessageEnvelope
from .native_tooling import NativeToolCall, NativeToolResult, parse_tool_calls
from .native_critic import CriticVerdict, parse_critic_verdict
from .model_backend import DirectTransformersModel, TransformersModel
from .tools import ToolError, ToolRegistry, ToolSpec
from .telemetry import WorkflowTelemetry
from .evaluator import EvaluationResult, TaskEvaluator, parse_reference_answer
from .action_constraints import ActionConstraint, build_action_constraint, explicit_fact_ids
from .protocol import PROTOCOL_VERSION, action_contract, parse_action, validate_action
from .semantic_contract import (
    SemanticCheckResult,
    SemanticContract,
    SemanticRequirement,
    SemanticStatus,
    build_role_semantic_contract,
    default_semantic_contract,
    global_semantic_requirements,
    semantic_requirements_for,
)
from .semantic_resolver import (
    SemanticResolver,
    all_requirements_sufficient,
    check_semantic_contract,
    missing_requirements,
)
from .semantic_feedback import (
    SemanticACK,
    SemanticFeedback,
    SemanticFeedbackType,
    SemanticNACK,
    build_semantic_feedback,
    is_ack,
    is_nack,
)
from .semantic_innovation import (
    InnovationDecision,
    InnovationDetector,
    detect_innovations,
    innovative_node_ids,
    semantic_requirement_for_node,
)
from .semantic_packet import (
    DECISION_SUFFICIENT,
    FULL_SUPPORT,
    RECONSTRUCTION_SUFFICIENT,
    VERIFICATION_SUFFICIENT,
    SemanticPacket,
    SemanticPacketBuilder,
    build_initial_semantic_packet,
)
from .refinement_planner import RefinementPlan, RefinementPlanner, plan_refinement

__all__ = [
    "CachedResult",
    "CallRecord",
    "ContextSlice",
    "DEFAULT_BRANCH_ID",
    "DependencyEdge",
    "GraphState",
    "GraphStore",
    "GraphTransaction",
    "BranchError",
    "InvalidationReport",
    "InMemoryResultCache",
    "NodeVersion",
    "ROLE_REQUIREMENTS",
    "RecomputePlan",
    "ResultCache",
    "RuntimeFingerprint",
    "RuntimeMetrics",
    "SliceError",
    "build_context_slice",
    "canonical_json",
    "dependency_digest",
    "downstream_dependents",
    "DomainExecution",
    "execute_domain",
    "ActionCompilationError",
    "ActionCompiler",
    "CompilationResult",
    "AgentGraphView",
    "AgentGraphViewManager",
    "DeltaCandidate",
    "GraphDelta",
    "GraphCommunicationPolicy",
    "SendAllPolicy",
    "MinimalNoFeedbackPolicy",
    "MinimalSendAllFallbackPolicy",
    "MinimalTargetedFeedbackPolicy",
    "RandomKeepPolicy",
    "ClosureAwareHeuristicPolicy",
    "LangGraphWorkflow",
    "WorkflowState",
    "build_langgraph_workflow",
    "AgentConfig",
    "AgentLifecycle",
    "AgentMemory",
    "NativeAgent",
    "MessageBus",
    "MessageBusError",
    "MessageEnvelope",
    "NativeToolCall",
    "NativeToolResult",
    "parse_tool_calls",
    "CriticVerdict",
    "parse_critic_verdict",
    "TransformersModel",
    "DirectTransformersModel",
    "ToolError",
    "ToolRegistry",
    "ToolSpec",
    "WorkflowTelemetry",
    "EvaluationResult",
    "TaskEvaluator",
    "parse_reference_answer",
    "ActionConstraint",
    "build_action_constraint",
    "explicit_fact_ids",
    "PROTOCOL_VERSION",
    "action_contract",
    "parse_action",
    "validate_action",
    "SemanticCheckResult",
    "SemanticContract",
    "SemanticRequirement",
    "SemanticStatus",
    "SemanticResolver",
    "SemanticACK",
    "SemanticFeedback",
    "SemanticFeedbackType",
    "SemanticNACK",
    "InnovationDecision",
    "InnovationDetector",
    "SemanticPacket",
    "SemanticPacketBuilder",
    "RefinementPlan",
    "RefinementPlanner",
    "DECISION_SUFFICIENT",
    "VERIFICATION_SUFFICIENT",
    "RECONSTRUCTION_SUFFICIENT",
    "FULL_SUPPORT",
    "all_requirements_sufficient",
    "build_initial_semantic_packet",
    "build_semantic_feedback",
    "build_role_semantic_contract",
    "check_semantic_contract",
    "default_semantic_contract",
    "detect_innovations",
    "innovative_node_ids",
    "is_ack",
    "is_nack",
    "missing_requirements",
    "global_semantic_requirements",
    "semantic_requirement_for_node",
    "semantic_requirements_for",
    "full_digest",
    "invalidate_downstream",
    "node_content_digest",
    "node_order_digest",
    "plan_local_recompute",
    "plan_refinement",
    "render_compact_context_slice",
    "render_context_slice",
    "render_full_graph_context",
    "runtime_fingerprint_digest",
    "upstream_closure",
]
