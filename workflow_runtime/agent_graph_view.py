"""Agent visibility views over one canonical GraphStore."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .graph_store import GraphStore
from .models import GraphState, VisibilityClass, node_visible_to_role, node_visibility_class

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

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable, deterministic visibility snapshot."""
        return {
            "agent_id": self.agent_id,
            "visible_node_ids": sorted(self.visible_node_ids),
            "visible_edge_ids": sorted(self.visible_edge_ids),
            "received_delta_ids": list(self.received_delta_ids),
            "locally_created_node_ids": sorted(self.locally_created_node_ids),
            "visibility_packet_ids": dict(sorted(self.visibility_packet_ids.items())),
            "visible_fragment_ids": sorted(self.visible_fragment_ids),
        }

    @classmethod
    def from_snapshot(cls, payload: Mapping[str, Any]) -> "AgentGraphView":
        agent_id = str(payload.get("agent_id", "")).strip()
        if not agent_id:
            raise ValueError("agent view snapshot is missing agent_id")
        view = cls(agent_id=agent_id)
        view.visible_node_ids = {str(item) for item in payload.get("visible_node_ids", ())}
        view.visible_edge_ids = {str(item) for item in payload.get("visible_edge_ids", ())}
        view.received_delta_ids = [str(item) for item in payload.get("received_delta_ids", ())]
        view.locally_created_node_ids = {
            str(item) for item in payload.get("locally_created_node_ids", ())
        }
        view.visibility_packet_ids = {
            str(key): str(value)
            for key, value in dict(payload.get("visibility_packet_ids", {})).items()
        }
        view.visible_fragment_ids = {
            str(item) for item in payload.get("visible_fragment_ids", ())
        }
        return view


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
        """Expose public source and mandatory nodes before communication.

        Evaluator-only and tool-only nodes remain outside model views even when
        they were created by a dataset adapter or runtime helper.
        """
        self.grant_global_visibility()

    def grant_global_visibility(self) -> None:
        """Expose mandatory baseline nodes to all agents."""
        state = self.store.snapshot()
        for view in self.views.values():
            initial = [
                node.node_id
                for node in state.nodes.values()
                if node.task_id == self.task_id
                and node.branch_id == self.branch_id
                and node.is_operationally_valid()
                # Restricted visibility is deny-first.  A malformed node must
                # not become public merely because it reused a mandatory
                # logical ID or communication scope.
                and node_visible_to_role(node, view.agent_id)
                and (
                    node.logical_id in MANDATORY_LOGICAL_IDS
                    or node.provenance.get("communication_scope") == "MANDATORY"
                    or node_visibility_class(node) == VisibilityClass.PUBLIC
                )
            ]
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
        state = self.store.snapshot()
        requested_node_ids = {str(node_id) for node_id in node_ids}
        unknown = requested_node_ids - set(state.nodes)
        if unknown:
            raise ValueError(f"cannot grant unknown nodes to {agent_id}: {sorted(unknown)}")
        forbidden = {
            node_id
            for node_id in requested_node_ids
            if (
                node_visibility_class(state.nodes[node_id]) == VisibilityClass.EVALUATOR_ONLY
                or (
                    node_visibility_class(state.nodes[node_id]) == VisibilityClass.TOOL_ONLY
                    and agent_id != "tool"
                )
                or (
                    node_visibility_class(state.nodes[node_id]) == VisibilityClass.ROLE_SCOPED
                    and isinstance(state.nodes[node_id].provenance.get("allowed_roles"), (list, tuple, set, frozenset))
                    and bool(state.nodes[node_id].provenance.get("allowed_roles"))
                    and agent_id not in {
                        str(role) for role in state.nodes[node_id].provenance.get("allowed_roles", ())
                    }
                    and agent_id not in {state.nodes[node_id].owner, state.nodes[node_id].created_by_role}
                )
            )
        }
        if forbidden:
            raise ValueError(
                f"cannot grant restricted nodes to {agent_id}: {sorted(forbidden)}"
            )
        granted_edges = {
            str(edge_id)
            for edge_id in edge_ids
            if any(
                edge.edge_id == str(edge_id)
                and edge.source in (view.visible_node_ids | requested_node_ids)
                and edge.target in (view.visible_node_ids | requested_node_ids)
                for edge in state.edges
            )
        }
        requested_fragment_ids = {str(fragment_id) for fragment_id in fragment_ids}
        available_after = view.visible_node_ids | requested_node_ids
        invalid_fragments = {
            fragment_id
            for fragment_id in requested_fragment_ids
            if fragment_id.split("#", 1)[0] not in available_after
        }
        if invalid_fragments:
            raise ValueError(
                f"cannot grant fragments without visible parent nodes to {agent_id}: "
                f"{sorted(invalid_fragments)}"
            )
        view.grant_nodes(requested_node_ids, local=local, packet_id=packet_id)
        view.grant_edges(granted_edges)
        view.grant_fragments(requested_fragment_ids)
        if delta_id:
            view.received_delta_ids.append(delta_id)
        self.refresh_visible_edges()

    def snapshot(self) -> dict[str, Any]:
        """Capture all receiver-relative visibility needed for exact replay."""
        return {
            "task_id": self.task_id,
            "branch_id": self.branch_id,
            "views": {
                agent_id: self.views[agent_id].snapshot()
                for agent_id in sorted(self.views)
            },
        }

    def restore(self, payload: Mapping[str, Any], *, strict: bool = True) -> None:
        """Restore a visibility snapshot against the manager's current graph."""
        task_id = str(payload.get("task_id", ""))
        branch_id = str(payload.get("branch_id", ""))
        if strict and (task_id != self.task_id or branch_id != self.branch_id):
            raise ValueError(
                "agent view snapshot boundary mismatch: "
                f"expected {self.task_id}/{self.branch_id}, got {task_id}/{branch_id}"
            )
        raw_views = payload.get("views", {})
        if not isinstance(raw_views, Mapping):
            raise ValueError("agent view snapshot views must be an object")
        restored = {
            str(agent_id): AgentGraphView.from_snapshot(view_payload)
            for agent_id, view_payload in raw_views.items()
            if isinstance(view_payload, Mapping)
        }
        if strict and set(restored) != set(self.views):
            raise ValueError(
                "agent view snapshot agents mismatch: "
                f"expected {sorted(self.views)}, got {sorted(restored)}"
            )
        graph = self.store.snapshot()
        node_ids = set(graph.nodes)
        edge_ids = {edge.edge_id for edge in graph.edges}
        for agent_id, view in restored.items():
            unknown_nodes = view.visible_node_ids - node_ids
            unknown_edges = view.visible_edge_ids - edge_ids
            if unknown_nodes or unknown_edges:
                raise ValueError(
                    f"agent view {agent_id} references unknown graph objects: "
                    f"nodes={sorted(unknown_nodes)}, edges={sorted(unknown_edges)}"
                )
            forbidden_nodes = {
                node_id for node_id in view.visible_node_ids
                if node_id in graph.nodes
                and (
                    node_visibility_class(graph.nodes[node_id]) == VisibilityClass.EVALUATOR_ONLY
                    or (
                        node_visibility_class(graph.nodes[node_id]) == VisibilityClass.TOOL_ONLY
                        and agent_id != "tool"
                    )
                )
            }
            if forbidden_nodes:
                raise ValueError(
                    f"agent view {agent_id} contains restricted nodes: {sorted(forbidden_nodes)}"
                )
            if not view.locally_created_node_ids.issubset(view.visible_node_ids):
                raise ValueError(f"agent view {agent_id} has invisible locally-created nodes")
            if not set(view.visibility_packet_ids).issubset(view.visible_node_ids):
                raise ValueError(f"agent view {agent_id} has packet metadata for invisible nodes")
        self.views = restored

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
