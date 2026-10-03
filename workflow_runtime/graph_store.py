"""Append-only graph storage with explicit task/branch/version semantics."""

from __future__ import annotations

import copy
import hashlib
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .canonical import (
    canonical_json,
    dependency_digest,
    full_digest,
    node_content_digest,
)
from .models import DependencyEdge, GraphState, NodeVersion, RuntimeFingerprint
from .relations import DATA_DEPENDENCY_RELATIONS, RELATION_SPECS


class GraphConflictError(RuntimeError):
    """Raised when a transaction read an outdated logical version."""


class GraphValidationError(ValueError):
    """Raised when a graph invariant is violated."""


class BranchError(GraphValidationError):
    """Raised when a branch fork or merge cannot be completed."""


class GraphStore:
    def __init__(self, state: Optional[GraphState] = None):
        self._state = copy.deepcopy(state) if state is not None else GraphState()
        self._counters: Dict[tuple[str, str, str], int] = {}
        self._branches: Dict[tuple[str, str], Dict[str, Any]] = {}
        for node in self._state.nodes.values():
            key = node.slot
            self._counters[key] = max(self._counters.get(key, 0), node.version)
            self._branches.setdefault(
                (node.task_id, node.branch_id),
                {"parent_branch_id": None, "created_at": node.created_at},
            )

    def snapshot(self) -> GraphState:
        return copy.deepcopy(self._state)

    def replace_state(self, state: GraphState) -> None:
        """Atomically replace the store contents after a validated batch."""
        self._state = copy.deepcopy(state)
        self._counters = {}
        self._branches = {}
        for node in self._state.nodes.values():
            key = node.slot
            self._counters[key] = max(self._counters.get(key, 0), node.version)
            self._branches.setdefault(
                (node.task_id, node.branch_id),
                {"parent_branch_id": None, "created_at": node.created_at},
            )

    def node(self, node_id: str) -> NodeVersion:
        try:
            return self._state.nodes[node_id]
        except KeyError as exc:
            raise GraphValidationError(f"unknown node: {node_id}") from exc

    def latest_valid(
        self,
        task_id: str,
        branch_id: str,
        logical_id: str,
    ) -> Optional[NodeVersion]:
        candidates = [
            node
            for node in self._state.nodes.values()
            if node.task_id == task_id
            and node.branch_id == branch_id
            and node.logical_id == logical_id
            and node.is_operationally_valid()
        ]
        if not candidates:
            return None
        return copy.deepcopy(max(candidates, key=lambda node: (node.version, node.created_at, node.node_id)))

    def add_node(
        self,
        *,
        task_id: str,
        branch_id: str,
        logical_id: str,
        node_type: str,
        content: Any,
        owner: str,
        status: str = "ready",
        confidence: Any = None,
        dependency_versions: Optional[Mapping[str, str]] = None,
        source_refs: Optional[Iterable[str]] = None,
        evidence_refs: Optional[Iterable[str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        validation: Optional[Mapping[str, bool]] = None,
        runtime: Optional[RuntimeFingerprint] = None,
        created_by_role: str = "",
        run_id: str = "",
        operation_batch_id: str = "",
    ) -> NodeVersion:
        key = (task_id, branch_id, logical_id)
        version = self._counters.get(key, 0) + 1
        parent = self.latest_valid(task_id, branch_id, logical_id)
        node_id = (
            f"{logical_id}@v{version}"
            if branch_id == "main"
            else f"{branch_id}:{logical_id}@v{version}"
        )
        node = NodeVersion(
            node_id=node_id,
            task_id=task_id,
            branch_id=branch_id,
            logical_id=logical_id,
            type=node_type,
            version=version,
            content=copy.deepcopy(content),
            owner=owner,
            status=status,
            confidence=confidence,
            dependency_versions=dict(dependency_versions or {}),
            source_refs=list(source_refs or []),
            evidence_refs=list(evidence_refs or []),
            provenance=dict(provenance or {}),
            validation=dict(validation or {}),
            parent_version_id=parent.node_id if parent is not None else None,
            created_at=time.time(),
            created_by_role=created_by_role or owner,
            run_id=run_id,
            operation_batch_id=operation_batch_id,
        )
        self._populate_digests(node, runtime)
        self.append_node(node)
        if parent is not None:
            self.replace_version(parent.node_id, node.node_id)
        return copy.deepcopy(node)

    def append_node(self, node: NodeVersion) -> None:
        if node.node_id in self._state.nodes:
            raise GraphValidationError(f"duplicate node id: {node.node_id}")
        if node.version < 1:
            raise GraphValidationError("node version must be positive")
        key = node.slot
        current_counter = self._counters.get(key, 0)
        if node.version <= current_counter:
            raise GraphValidationError(
                f"version must increase for {key}: {node.version} <= {current_counter}"
            )
        self._counters[key] = node.version
        self._state.nodes[node.node_id] = copy.deepcopy(node)

    def add_edge(
        self,
        *,
        source: str,
        target: str,
        relation: str,
        created_by_role: str = "",
        run_id: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> DependencyEdge:
        if relation not in RELATION_SPECS:
            raise GraphValidationError(f"unknown runtime relation: {relation}")
        source_node = self.node(source)
        target_node = self.node(target)
        if source_node.task_id != target_node.task_id or source_node.branch_id != target_node.branch_id:
            raise GraphValidationError("cross-task or cross-branch edges are not allowed")
        edge_id = f"edge_{len(self._state.edges) + 1}_{source}_{relation}_{target}"
        if any(edge.source == source and edge.target == target and edge.relation == relation for edge in self._state.edges):
            return next(
                edge
                for edge in self._state.edges
                if edge.source == source and edge.target == target and edge.relation == relation
            )
        if relation in DATA_DEPENDENCY_RELATIONS and self._would_cycle(source, target):
            raise GraphValidationError(f"data dependency would create a cycle: {source} -> {target}")
        edge = DependencyEdge(
            edge_id=edge_id,
            source=source,
            target=target,
            relation=relation,
            task_id=source_node.task_id,
            branch_id=source_node.branch_id,
            created_at=time.time(),
            created_by_role=created_by_role,
            run_id=run_id,
            metadata=dict(metadata or {}),
        )
        self._state.edges.append(edge)
        if relation in DATA_DEPENDENCY_RELATIONS:
            target_node.dependency_versions[source_node.logical_id] = (
                f"{source_node.node_id}:{source_node.content_digest}:{source_node.status}"
            )
            self._populate_digests(target_node, None)
        elif relation == "verifies" and source_node.status == "verified":
            target_node.validation["model_judged_correct"] = True
            self._populate_digests(target_node, None)
        return copy.deepcopy(edge)

    def replace_version(self, old_node_id: str, new_node_id: str) -> None:
        old = self.node(old_node_id)
        new = self.node(new_node_id)
        if old.task_id != new.task_id or old.branch_id != new.branch_id or old.logical_id != new.logical_id:
            raise GraphValidationError("superseded versions must share task, branch, and logical id")
        self.add_edge(source=new_node_id, target=old_node_id, relation="supersedes")
        old.status = "superseded"
        self._state.status_events.append(
            {
                "node_id": old_node_id,
                "status": "superseded",
                "reason": "replaced_by_new_version",
                "at": time.time(),
            }
        )

    def mark_status(self, node_id: str, status: str, *, reason: str = "") -> None:
        node = self.node(node_id)
        if status == "empty":
            raise GraphValidationError("empty is only valid before a node is created")
        node.status = status
        self._state.status_events.append(
            {
                "node_id": node_id,
                "status": status,
                "reason": reason,
                "at": time.time(),
            }
        )

    def branch_exists(self, task_id: str, branch_id: str) -> bool:
        return (task_id, branch_id) in self._branches or any(
            node.task_id == task_id and node.branch_id == branch_id
            for node in self._state.nodes.values()
        )

    def list_branches(self, task_id: str) -> list[str]:
        branches = {
            branch_id
            for current_task, branch_id in self._branches
            if current_task == task_id
        }
        branches.update(
            node.branch_id
            for node in self._state.nodes.values()
            if node.task_id == task_id
        )
        return sorted(branches)

    def branch_info(self, task_id: str, branch_id: str) -> Dict[str, Any]:
        if not self.branch_exists(task_id, branch_id):
            raise BranchError(f"unknown branch: {task_id}/{branch_id}")
        return copy.deepcopy(self._branches.get((task_id, branch_id), {
            "parent_branch_id": None,
            "created_at": 0.0,
        }))

    def create_branch(
        self,
        task_id: str,
        branch_id: str,
        *,
        source_branch_id: str = "main",
    ) -> str:
        """Fork one task branch into an isolated version namespace."""
        if not branch_id or any(char.isspace() for char in branch_id):
            raise BranchError("branch_id must be non-empty and contain no whitespace")
        if self.branch_exists(task_id, branch_id):
            raise BranchError(f"branch already exists: {task_id}/{branch_id}")
        source_nodes = [
            node for node in self._state.nodes.values()
            if node.task_id == task_id and node.branch_id == source_branch_id
        ]
        if not source_nodes:
            raise BranchError(f"source branch does not exist: {task_id}/{source_branch_id}")
        source_nodes.sort(key=lambda node: (node.created_at, node.node_id))
        node_map: Dict[str, str] = {}
        for source in source_nodes:
            copied = copy.deepcopy(source)
            copied.branch_id = branch_id
            copied.node_id = f"{branch_id}:{source.node_id}"
            copied.parent_version_id = node_map.get(source.parent_version_id or "")
            copied.provenance = {
                **copied.provenance,
                "forked_from_branch": source_branch_id,
                "forked_from_node": source.node_id,
            }
            self.append_node(copied)
            node_map[source.node_id] = copied.node_id
        source_edges = [
            edge for edge in self._state.edges
            if edge.task_id == task_id and edge.branch_id == source_branch_id
        ]
        for edge in source_edges:
            source_id = node_map.get(edge.source)
            target_id = node_map.get(edge.target)
            if source_id is None or target_id is None:
                continue
            self.add_edge(
                source=source_id,
                target=target_id,
                relation=edge.relation,
                created_by_role=edge.created_by_role,
                run_id=edge.run_id,
                metadata={**edge.metadata, "forked_from_edge": edge.edge_id},
            )
        self._branches[(task_id, branch_id)] = {
            "parent_branch_id": source_branch_id,
            "created_at": time.time(),
            "fork_node_count": len(source_nodes),
        }
        return branch_id

    def branch_digest(self, task_id: str, branch_id: str) -> str:
        if not self.branch_exists(task_id, branch_id):
            raise BranchError(f"unknown branch: {task_id}/{branch_id}")
        nodes = sorted(
            (node for node in self._state.nodes.values()
             if node.task_id == task_id and node.branch_id == branch_id),
            key=lambda node: node.node_id,
        )
        edges = sorted(
            (edge for edge in self._state.edges
             if edge.task_id == task_id and edge.branch_id == branch_id),
            key=lambda edge: edge.edge_id,
        )
        payload = {"nodes": nodes, "edges": edges}
        return "branch_" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def merge_branch(
        self,
        task_id: str,
        source_branch_id: str,
        *,
        target_branch_id: str = "main",
        logical_ids: Optional[Iterable[str]] = None,
    ) -> list[str]:
        """Overlay selected current source values into a target branch.

        The target receives new versions; source node IDs and source edges are
        never reused.  This makes merge provenance explicit and preserves the
        append-only graph invariant.
        """
        if not self.branch_exists(task_id, source_branch_id):
            raise BranchError(f"unknown source branch: {task_id}/{source_branch_id}")
        if not self.branch_exists(task_id, target_branch_id):
            raise BranchError(f"unknown target branch: {task_id}/{target_branch_id}")
        allowed = set(logical_ids) if logical_ids is not None else None
        source_latest: dict[str, NodeVersion] = {}
        for node in self._state.nodes.values():
            if node.task_id != task_id or node.branch_id != source_branch_id:
                continue
            if not node.is_operationally_valid() or node.logical_id == "task":
                continue
            if allowed is not None and node.logical_id not in allowed:
                continue
            previous = source_latest.get(node.logical_id)
            if previous is None or (node.version, node.created_at) > (previous.version, previous.created_at):
                source_latest[node.logical_id] = node
        source_to_target: dict[str, str] = {}
        merged: list[str] = []
        for logical_id, source in sorted(source_latest.items()):
            merged_node = self.add_node(
                task_id=task_id,
                branch_id=target_branch_id,
                logical_id=logical_id,
                node_type=source.type,
                content=source.content,
                owner=source.owner,
                status=source.status,
                confidence=source.confidence,
                dependency_versions=source.dependency_versions,
                source_refs=source.source_refs,
                evidence_refs=source.evidence_refs,
                provenance={**source.provenance, "merged_from_branch": source_branch_id, "merged_from_node": source.node_id},
                validation=source.validation,
                created_by_role="merge",
                run_id=source.run_id,
                operation_batch_id=source.operation_batch_id,
            )
            source_to_target[source.node_id] = merged_node.node_id
            merged.append(merged_node.node_id)
        for edge in self._state.edges:
            if edge.task_id != task_id or edge.branch_id != source_branch_id:
                continue
            source_id = source_to_target.get(edge.source)
            target_source = self._state.nodes.get(edge.target)
            target_id = source_to_target.get(edge.target)
            if target_id is None and target_source is not None:
                target_current = self.latest_valid(task_id, target_branch_id, target_source.logical_id)
                target_id = target_current.node_id if target_current is not None else None
            if source_id is None or target_id is None:
                continue
            self.add_edge(
                source=source_id,
                target=target_id,
                relation=edge.relation,
                created_by_role="merge",
                metadata={"merged_from_edge": edge.edge_id, "source_branch_id": source_branch_id},
            )
        return merged

    def current_versions(self, task_id: str, branch_id: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for node in self._state.nodes.values():
            if node.task_id != task_id or node.branch_id != branch_id:
                continue
            current = self.latest_valid(task_id, branch_id, node.logical_id)
            if current is not None:
                result[node.logical_id] = current.node_id
        return result

    def begin_transaction(self, read_versions: Mapping[str, str]) -> "GraphTransaction":
        return GraphTransaction(self, dict(read_versions))

    def _populate_digests(
        self,
        node: NodeVersion,
        runtime: Optional[RuntimeFingerprint],
    ) -> None:
        node.content_digest = node_content_digest(node.type, node.content)
        node.dependency_digest = dependency_digest(node.dependency_versions)
        if runtime is not None:
            node.full_digest = full_digest(
                node_type=node.type,
                content_digest=node.content_digest,
                dependency_digest_value=node.dependency_digest,
                runtime=runtime,
            )
        else:
            node.full_digest = ""

    def _would_cycle(self, source: str, target: str) -> bool:
        if source == target:
            return True
        adjacency: Dict[str, List[str]] = {}
        for edge in self._state.edges:
            if edge.relation in DATA_DEPENDENCY_RELATIONS:
                adjacency.setdefault(edge.source, []).append(edge.target)
        adjacency.setdefault(source, []).append(target)
        stack = [target]
        visited = set()
        while stack:
            current = stack.pop()
            if current == source:
                return True
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adjacency.get(current, []))
        return False


class GraphTransaction:
    def __init__(self, store: GraphStore, read_versions: Dict[str, str]):
        self.store = store
        self.read_versions = read_versions
        self._nodes: List[NodeVersion] = []
        self._edges: List[Dict[str, Any]] = []
        self._committed = False

    def add_node(self, node: NodeVersion) -> None:
        if self._committed:
            raise GraphConflictError("transaction already committed")
        self._nodes.append(copy.deepcopy(node))

    def add_edge(self, **kwargs: Any) -> None:
        if self._committed:
            raise GraphConflictError("transaction already committed")
        self._edges.append(dict(kwargs))

    def commit(self) -> List[str]:
        if self._committed:
            raise GraphConflictError("transaction already committed")
        for logical_id, expected_node_id in self.read_versions.items():
            expected = self.store.node(expected_node_id)
            current_node = self.store.latest_valid(
                expected.task_id,
                expected.branch_id,
                logical_id,
            )
            if current_node is None or current_node.node_id != expected_node_id:
                raise GraphConflictError(
                    f"read version changed for {logical_id}: expected {expected_node_id}, "
                    f"got {current_node.node_id if current_node else None}"
                )
        for node in self._nodes:
            self.store.append_node(node)
        edge_ids: List[str] = []
        for edge_kwargs in self._edges:
            edge_ids.append(self.store.add_edge(**edge_kwargs).edge_id)
        self._committed = True
        return [node.node_id for node in self._nodes] + edge_ids
