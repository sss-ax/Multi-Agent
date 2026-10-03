"""Agent visibility views over one canonical GraphStore."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable

from .graph_store import GraphStore
from .models import GraphState

MANDATORY_LOGICAL_IDS = frozenset({"task", "query_spec"})


@dataclass
class AgentGraphView:
    """A role-scoped subset of the canonical graph.

    Communication grants visibility to canonical node/edge IDs instead of
    copying nodes into independent stores.
    """

    agent_id: str
    visible_node_ids: set[str] = field(default_factory=set)
    visible_edge_ids: set[str] = field(default_factory=set)
    received_delta_ids: list[str] = field(default_factory=list)
    locally_created_node_ids: set[str] = field(default_factory=set)
    visibility_packet_ids: dict[str, str] = field(default_factory=dict)
    visible_fragment_ids: set[str] = field(default_factory=set)

    def grant_nodes(self, node_ids: Iterable[str], *, local: bool = False, packet_id: str = "") -> set[str]:
        ids = {str(node_id) for node_id in node_ids}
        newly_visible = ids - self.visible_node_ids
        self.visible_node_ids.update(ids)
        if local:
            self.locally_created_node_ids.update(ids)
        if packet_id:
            for node_id in newly_visible:
                self.visibility_packet_ids[node_id] = packet_id
        return newly_visible

    def grant_edges(self, edge_ids: Iterable[str]) -> None:
        self.visible_edge_ids.update(str(edge_id) for edge_id in edge_ids)

    def grant_fragments(self, fragment_ids: Iterable[str]) -> None:
        self.visible_fragment_ids.update(str(fragment_id) for fragment_id in fragment_ids)


class AgentGraphViewManager:
    """Manage canonical graph visibility for workflow roles."""

    def __init__(
        self,
        store: GraphStore,
        *,
        task_id: str,
        branch_id: str,
        agents: Iterable[str],
    ) -> None:
        self.store = store
        self.task_id = task_id
        self.branch_id = branch_id
        self.views = {agent: AgentGraphView(agent) for agent in agents}
        self.grant_initial_visibility()

    def view(self, agent_id: str) -> AgentGraphView:
        return self.views[agent_id]

    def grant_initial_visibility(self) -> None:
        """Expose immutable input seed nodes to all agents.

        Global task context and dataset seed nodes are baseline context, not
        inter-agent communication. They are visible to every role before
        policy-controlled graph deltas begin.
        """
        self.grant_global_visibility()

    def grant_global_visibility(self) -> None:
        """Expose mandatory baseline nodes to all agents."""
        state = self.store.snapshot()
        initial = [
            node.node_id
            for node in state.nodes.values()
            if node.task_id == self.task_id
            and node.branch_id == self.branch_id
            and node.is_operationally_valid()
            and (
                node.logical_id in MANDATORY_LOGICAL_IDS
                or node.created_by_role in {"user", "dataset", ""}
                or node.provenance.get("communication_scope") == "MANDATORY"
            )
        ]
        for view in self.views.values():
            view.grant_nodes(initial)
        self.refresh_visible_edges()

    def refresh_visible_edges(self) -> None:
        state = self.store.snapshot()
        for view in self.views.values():
            for edge in state.edges:
                if edge.source in view.visible_node_ids and edge.target in view.visible_node_ids:
                    view.visible_edge_ids.add(edge.edge_id)

    def grant(
        self,
        agent_id: str,
        *,
        node_ids: Iterable[str],
        edge_ids: Iterable[str] = (),
        fragment_ids: Iterable[str] = (),
        local: bool = False,
        delta_id: str = "",
        packet_id: str = "",
    ) -> None:
        view = self.view(agent_id)
        view.grant_nodes(node_ids, local=local, packet_id=packet_id)
        view.grant_edges(edge_ids)
        view.grant_fragments(fragment_ids)
        if delta_id:
            view.received_delta_ids.append(delta_id)
        self.refresh_visible_edges()

    def visible_store(self, agent_id: str) -> GraphStore:
        """Return a filtered GraphStore snapshot for one agent view."""
        view = self.view(agent_id)
        state = self.store.snapshot()
        nodes = {
            node_id: copy.deepcopy(node)
            for node_id, node in state.nodes.items()
            if node_id in view.visible_node_ids
        }
        edges = [
            copy.deepcopy(edge)
            for edge in state.edges
            if edge.edge_id in view.visible_edge_ids
            and edge.source in nodes
            and edge.target in nodes
        ]
        events = [
            copy.deepcopy(event)
            for event in state.status_events
            if event.get("node_id") in nodes
        ]
        return GraphStore(GraphState(nodes=nodes, edges=edges, status_events=events))
