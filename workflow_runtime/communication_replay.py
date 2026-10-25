"""Replayable snapshots captured immediately before communication decisions."""

from __future__ import annotations

import copy
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .agent_graph_view import AgentGraphViewManager
from .graph_store import GraphStore
from .models import DependencyEdge, GraphState, NodeVersion


SNAPSHOT_SCHEMA_VERSION = "communication-decision-v1"


def new_snapshot_id(*, task_id: str, branch_id: str, stage: str, decision_index: int) -> str:
    """Create a process-independent ID; counters alone collide after restarts."""
    return f"{task_id}:{branch_id}:{stage}:{decision_index}:{uuid.uuid4().hex}"


def graph_state_to_dict(state: GraphState) -> dict[str, Any]:
    """Serialize GraphState without losing node or edge provenance."""
    return {
        "nodes": {
            node_id: asdict(state.nodes[node_id])
            for node_id in sorted(state.nodes)
        },
        "edges": [asdict(edge) for edge in state.edges],
        "status_events": copy.deepcopy(state.status_events),
    }


def graph_state_from_dict(payload: Mapping[str, Any]) -> GraphState:
    raw_nodes = payload.get("nodes", {})
    raw_edges = payload.get("edges", ())
    if not isinstance(raw_nodes, Mapping) or not isinstance(raw_edges, (list, tuple)):
        raise ValueError("invalid graph state snapshot")
    nodes: dict[str, NodeVersion] = {}
    for node_id, node_payload in raw_nodes.items():
        if not isinstance(node_payload, Mapping):
            raise ValueError(f"invalid node snapshot: {node_id}")
        node = NodeVersion(**copy.deepcopy(dict(node_payload)))
        if node.node_id != str(node_id):
            raise ValueError(f"node snapshot key/id mismatch: {node_id}/{node.node_id}")
        nodes[node.node_id] = node
    edges = []
    edge_ids: set[str] = set()
    for edge_payload in raw_edges:
        if not isinstance(edge_payload, Mapping):
            raise ValueError("invalid edge snapshot")
        edge = DependencyEdge(**copy.deepcopy(dict(edge_payload)))
        if edge.edge_id in edge_ids:
            raise ValueError(f"duplicate edge in snapshot: {edge.edge_id}")
        if edge.source not in nodes or edge.target not in nodes:
            raise ValueError(f"edge references missing node: {edge.edge_id}")
        edge_ids.add(edge.edge_id)
        edges.append(edge)
    status_events = copy.deepcopy(list(payload.get("status_events", ())))
    return GraphState(nodes=nodes, edges=edges, status_events=status_events)


@dataclass
class CommunicationDecisionSnapshot:
    """All runtime-owned state required to replay one communication choice.

    ``external_state`` is deliberately opaque. Model RNG, remote services, and
    environment state are only replayable when their owner supplies matching
    capture and restore hooks.
    """

    snapshot_id: str
    task_id: str
    branch_id: str
    task_type: str
    stage: str
    sender: str
    receiver: str
    decision_index: int
    round_index: int
    remaining_budget: int | None
    policy_name: str
    graph_state: GraphState
    agent_views: dict[str, Any]
    workflow_state: dict[str, Any] = field(default_factory=dict)
    candidates: tuple[dict[str, Any], ...] = ()
    fragments: tuple[dict[str, Any], ...] = ()
    external_state: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    schema_version: str = SNAPSHOT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "task_id": self.task_id,
            "branch_id": self.branch_id,
            "task_type": self.task_type,
            "stage": self.stage,
            "sender": self.sender,
            "receiver": self.receiver,
            "decision_index": self.decision_index,
            "round_index": self.round_index,
            "remaining_budget": self.remaining_budget,
            "policy_name": self.policy_name,
            "workflow_state": copy.deepcopy(self.workflow_state),
            "graph_state": graph_state_to_dict(self.graph_state),
            "agent_views": copy.deepcopy(self.agent_views),
            "candidates": copy.deepcopy(list(self.candidates)),
            "fragments": copy.deepcopy(list(self.fragments)),
            "external_state": copy.deepcopy(self.external_state),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CommunicationDecisionSnapshot":
        schema_version = str(payload.get("schema_version", ""))
        if schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(f"unsupported communication snapshot schema: {schema_version}")
        graph_payload = payload.get("graph_state", {})
        view_payload = payload.get("agent_views", {})
        if not isinstance(graph_payload, Mapping) or not isinstance(view_payload, Mapping):
            raise ValueError("communication snapshot is missing graph or view state")
        remaining_budget = payload.get("remaining_budget")
        return cls(
            snapshot_id=str(payload.get("snapshot_id", "")),
            created_at=float(payload.get("created_at", 0.0)),
            task_id=str(payload.get("task_id", "")),
            branch_id=str(payload.get("branch_id", "")),
            task_type=str(payload.get("task_type", "")),
            stage=str(payload.get("stage", "")),
            sender=str(payload.get("sender", "")),
            receiver=str(payload.get("receiver", "")),
            decision_index=int(payload.get("decision_index", 0)),
            round_index=int(payload.get("round_index", 0)),
            remaining_budget=None if remaining_budget is None else int(remaining_budget),
            policy_name=str(payload.get("policy_name", "")),
            workflow_state=copy.deepcopy(dict(payload.get("workflow_state", {}))),
            graph_state=graph_state_from_dict(graph_payload),
            agent_views=copy.deepcopy(dict(view_payload)),
            candidates=tuple(copy.deepcopy(list(payload.get("candidates", ())))),
            fragments=tuple(copy.deepcopy(list(payload.get("fragments", ())))),
            external_state=copy.deepcopy(dict(payload.get("external_state", {}))),
            schema_version=schema_version,
        )

    def restore_runtime(
        self,
        store: GraphStore,
        views: AgentGraphViewManager,
        *,
        restore_external: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Restore graph/views and return the checkpointed workflow state."""
        if views.task_id != self.task_id or views.branch_id != self.branch_id:
            raise ValueError("cannot restore communication snapshot into a different task boundary")
        # Validate graph-relative visibility before mutating the live runtime.
        validation_store = GraphStore(self.graph_state)
        validation_views = AgentGraphViewManager(
            validation_store,
            task_id=self.task_id,
            branch_id=self.branch_id,
            agents=tuple(views.views),
        )
        validation_views.restore(self.agent_views)
        store.replace_state(self.graph_state)
        views.restore(self.agent_views)
        if restore_external is not None:
            restore_external(copy.deepcopy(self.external_state))
        return copy.deepcopy(self.workflow_state)


class CommunicationReplayBuffer:
    """In-memory decision snapshots with optional append-only JSONL storage."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._snapshots: list[CommunicationDecisionSnapshot] = []

    def record(self, snapshot: CommunicationDecisionSnapshot) -> None:
        self._snapshots.append(copy.deepcopy(snapshot))
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(snapshot.to_dict(), ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")

    def snapshots(self) -> tuple[CommunicationDecisionSnapshot, ...]:
        return tuple(copy.deepcopy(self._snapshots))

    def get(self, snapshot_id: str) -> CommunicationDecisionSnapshot:
        for snapshot in reversed(self._snapshots):
            if snapshot.snapshot_id == snapshot_id:
                return copy.deepcopy(snapshot)
        raise KeyError(snapshot_id)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "CommunicationReplayBuffer":
        source = Path(path)
        buffer = cls()
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                buffer._snapshots.append(CommunicationDecisionSnapshot.from_dict(json.loads(line)))
        return buffer

    def __len__(self) -> int:
        return len(self._snapshots)


def candidate_snapshot(candidate: Any) -> dict[str, Any]:
    """Capture policy-visible candidate features without depending on its type."""
    return {
        "node_id": str(getattr(candidate, "node_id", "")),
        "fragment_id": str(getattr(candidate, "fragment_id", "")),
        "fragment_name": str(getattr(candidate, "fragment_name", "")),
        "parent_node_id": str(getattr(candidate, "parent_node_id", "")),
        "sender": str(getattr(candidate, "sender", "")),
        "receiver": str(getattr(candidate, "receiver", "")),
        "exportable": bool(getattr(candidate, "exportable", False)),
        "novelty_score": float(getattr(candidate, "novelty_score", 0.0)),
        "redundancy_score": float(getattr(candidate, "redundancy_score", 0.0)),
        "token_cost": int(getattr(candidate, "token_cost", 0)),
        "scope": str(getattr(candidate, "scope", "")),
        "structural_features": copy.deepcopy(getattr(candidate, "structural_features", {})),
    }


def fragment_snapshot(fragment: Any) -> dict[str, Any]:
    return {
        "fragment_id": str(getattr(fragment, "fragment_id", "")),
        "node_id": str(getattr(fragment, "node_id", "")),
        "name": str(getattr(fragment, "name", "")),
        "scope": str(getattr(fragment, "scope", "")),
        "content": copy.deepcopy(getattr(fragment, "content", None)),
        "token_cost": int(getattr(fragment, "token_cost", 0)),
    }


def normalize_external_state(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return copy.deepcopy(dict(value or {}))
