"""Receiver-relative dependency closure for selected graph-delta roots."""

from __future__ import annotations

from .agent_graph_view import AgentGraphView, MANDATORY_LOGICAL_IDS
from .delta_scoring import estimate_edge_tokens, estimate_node_tokens, semantically_known
from .graph_delta import GraphDelta
from .models import GraphState
from .relations import DATA_DEPENDENCY_RELATIONS, VALIDATION_RELATIONS


def dependency_closure(
    state: GraphState,
    *,
    root_node_ids: list[str] | tuple[str, ...],
    sender: str,
    receiver: str,
    receiver_view: AgentGraphView,
    policy: str,
    candidate_token_cost: int = 0,
) -> GraphDelta:
    """Close selected roots over dependencies missing from the receiver view."""
    selected = [node_id for node_id in dict.fromkeys(root_node_ids) if node_id in state.nodes]
    closure: set[str] = set()
    edge_ids: set[str] = set()
    stack = list(selected)

    baseline_visible = set(receiver_view.visible_node_ids) | {
        node.node_id
        for node in state.nodes.values()
        if node.logical_id in MANDATORY_LOGICAL_IDS
        or node.provenance.get("communication_scope") == "MANDATORY"
    }
    receiver_nodes = [
        state.nodes[node_id]
        for node_id in baseline_visible
        if node_id in state.nodes
    ]
    while stack:
        node_id = stack.pop()
        if node_id in baseline_visible or node_id in closure:
            continue
        node = state.nodes.get(node_id)
        if node is None or semantically_known(node, receiver_nodes):
            continue
        closure.add(node_id)
        for edge in _required_incoming_edges(state, node_id):
            edge_ids.add(edge.edge_id)
            dependency_id = edge.source if edge.relation in DATA_DEPENDENCY_RELATIONS else edge.target
            if dependency_id not in baseline_visible and dependency_id not in closure:
                stack.append(dependency_id)

    visible_after = baseline_visible | closure
    for edge in state.edges:
        if edge.source in visible_after and edge.target in visible_after:
            if edge.source in closure or edge.target in closure:
                edge_ids.add(edge.edge_id)

    closure_added = tuple(sorted(node_id for node_id in closure if node_id not in selected))
    token_cost = sum(estimate_node_tokens(state.nodes[node_id]) for node_id in closure)
    token_cost += sum(estimate_edge_tokens(edge) for edge in state.edges if edge.edge_id in edge_ids)
    return GraphDelta(
        sender=sender,
        receiver=receiver,
        root_node_ids=tuple(selected),
        node_ids=tuple(sorted(closure)),
        edge_ids=tuple(sorted(edge_ids)),
        token_cost=token_cost,
        candidate_token_cost=candidate_token_cost,
        closure_added_node_ids=closure_added,
        redundant_node_ids=(),
        policy=policy,
    )


def _required_incoming_edges(state: GraphState, node_id: str):
    for edge in state.edges:
        if edge.relation in DATA_DEPENDENCY_RELATIONS and edge.target == node_id:
            yield edge
        elif edge.relation in VALIDATION_RELATIONS and edge.source == node_id:
            yield edge
