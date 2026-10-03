"""Extract exportable graph-delta candidates from compiled graph mutations."""

from __future__ import annotations

from typing import Iterable

from .agent_graph_view import AgentGraphView
from .delta_scoring import estimate_node_tokens, semantically_known, structural_features
from .graph_delta import DeltaCandidate
from .models import GraphState, NodeVersion

LOCAL_TYPES = frozenset({
    "tool_request",
    "error",
    "runtime_error",
    "scratch",
    "parser_trace",
    "parser_state",
    "internal_diagnostic",
})
MANDATORY_TYPES = frozenset({"task", "query_spec"})
SHAREABLE_TYPES = frozenset({
    "facts", "fact", "plan", "plan_steps", "calculation", "result",
    "verification", "requirements", "test", "code", "execution",
    "test_result", "table", "table_cell", "evidence", "entity",
    "supporting_fact", "evidence_link", "choice", "tool_result",
})

MANDATORY_STAGE_TYPES_BY_PAIR = {
    ("planner", "solver"): frozenset({"facts", "plan", "plan_steps"}),
    ("solver", "critic"): frozenset({"calculation", "result", "code"}),
    ("tool", "critic"): frozenset({"execution", "test_result", "result"}),
    ("critic", "final_solver"): frozenset({"verification"}),
}


def communication_scope(node: NodeVersion, *, sender: str = "", receiver: str = "") -> str:
    raw = node.provenance.get("communication_scope") if isinstance(node.provenance, dict) else None
    if raw in {"LOCAL", "OPTIONAL_POLICY", "MANDATORY_STAGE", "MANDATORY", "OPTIONAL", "SHAREABLE", "REQUIRED"}:
        if raw == "SHAREABLE":
            return "OPTIONAL_POLICY"
        if raw == "REQUIRED":
            return "MANDATORY_STAGE"
        if raw == "OPTIONAL":
            return "OPTIONAL_POLICY"
        return raw
    if node.type in LOCAL_TYPES:
        return "LOCAL"
    if node.type in MANDATORY_TYPES:
        return "MANDATORY"
    if node.type in MANDATORY_STAGE_TYPES_BY_PAIR.get((sender, receiver), frozenset()):
        return "MANDATORY_STAGE"
    if node.type in SHAREABLE_TYPES:
        return "OPTIONAL_POLICY"
    return "OPTIONAL_POLICY"


def extract_delta_candidates(
    state: GraphState,
    *,
    node_ids: Iterable[str],
    sender: str,
    receiver: str,
    receiver_view: AgentGraphView,
) -> list[DeltaCandidate]:
    receiver_nodes = [
        state.nodes[node_id]
        for node_id in receiver_view.visible_node_ids
        if node_id in state.nodes
    ]
    candidates: list[DeltaCandidate] = []
    for node_id in dict.fromkeys(str(item) for item in node_ids):
        node = state.nodes.get(node_id)
        if node is None or not node.is_operationally_valid():
            continue
        scope = communication_scope(node, sender=sender, receiver=receiver)
        if scope in {"LOCAL", "MANDATORY"}:
            continue
        known = semantically_known(node, receiver_nodes)
        candidates.append(DeltaCandidate(
            node_id=node_id,
            parent_node_id=node_id,
            sender=sender,
            receiver=receiver,
            exportable=True,
            novelty_score=0.0 if known else 1.0,
            redundancy_score=1.0 if known else 0.0,
            token_cost=estimate_node_tokens(node),
            structural_features=structural_features(node, state),
            scope=scope,
        ))
    return candidates
