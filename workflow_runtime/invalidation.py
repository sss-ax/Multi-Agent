"""Version-aware stale propagation and local recompute planning."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Set, Tuple

from .graph_store import GraphStore
from .models import InvalidationReport, NodeVersion, RecomputePlan
from .relations import DATA_DEPENDENCY_RELATIONS, VALIDATION_RELATIONS


def downstream_dependents(
    store: GraphStore,
    changed_node_ids: Iterable[str],
) -> Set[str]:
    state = store.snapshot()
    outgoing: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for edge in state.edges:
        if edge.relation in DATA_DEPENDENCY_RELATIONS:
            outgoing[edge.source].append((edge.target, edge.relation))
        elif edge.relation in VALIDATION_RELATIONS:
            # validator -> validated; a changed validated node invalidates the
            # validator, so this relation is traversed in reverse.
            outgoing[edge.target].append((edge.source, edge.relation))

    changed = set(changed_node_ids)
    result: Set[str] = set()
    queue = deque(changed)
    while queue:
        current = queue.popleft()
        for dependent, _relation in outgoing.get(current, []):
            if dependent in result or dependent in changed:
                continue
            result.add(dependent)
            queue.append(dependent)
    return result


def invalidate_downstream(
    store: GraphStore,
    changed_node_ids: Iterable[str],
    *,
    reason: str,
) -> InvalidationReport:
    changed = list(dict.fromkeys(changed_node_ids))
    if not changed:
        return InvalidationReport(
            changed_node_ids=[],
            stale_node_ids=[],
            preserved_node_ids=[],
            reason=reason,
            fanout=0,
        )
    affected = downstream_dependents(store, changed)
    state = store.snapshot()
    stale: List[str] = []
    preserved: List[str] = []
    changed_set = set(changed)
    for node in state.nodes.values():
        if node.node_id in changed_set:
            continue
        if node.node_id in affected:
            if node.status not in {"stale", "superseded", "invalid"}:
                store.mark_status(node.node_id, "stale", reason=reason)
            stale.append(node.node_id)
        elif node.task_id == state.nodes[changed[0]].task_id and node.branch_id == state.nodes[changed[0]].branch_id:
            preserved.append(node.node_id)
    return InvalidationReport(
        changed_node_ids=changed,
        stale_node_ids=sorted(stale),
        preserved_node_ids=sorted(preserved),
        reason=reason,
        fanout=len(stale),
    )


def _topological_order(store: GraphStore, node_ids: Set[str]) -> List[str]:
    state = store.snapshot()
    adjacency: Dict[str, List[str]] = defaultdict(list)
    indegree: Dict[str, int] = {node_id: 0 for node_id in node_ids}
    for edge in state.edges:
        if edge.relation not in DATA_DEPENDENCY_RELATIONS:
            continue
        if edge.source not in node_ids or edge.target not in node_ids:
            continue
        adjacency[edge.source].append(edge.target)
        indegree[edge.target] += 1
    queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    result: List[str] = []
    while queue:
        node_id = queue.popleft()
        result.append(node_id)
        for target in sorted(adjacency.get(node_id, [])):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(result) != len(node_ids):
        raise ValueError("recompute dependency graph contains a cycle")
    return result


def plan_local_recompute(
    store: GraphStore,
    *,
    task_id: str,
    branch_id: str,
    target_logical_ids: Iterable[str],
) -> RecomputePlan:
    state = store.snapshot()
    target_ids: Set[str] = set()
    for logical_id in target_logical_ids:
        candidates = [
            node
            for node in state.nodes.values()
            if node.task_id == task_id
            and node.branch_id == branch_id
            and node.logical_id == logical_id
        ]
        if candidates:
            target_ids.add(max(candidates, key=lambda node: (node.version, node.created_at)).node_id)
    dirty = {
        node.node_id
        for node in state.nodes.values()
        if node.task_id == task_id
        and node.branch_id == branch_id
        and node.status in {"stale", "invalid", "conflict"}
    }
    if target_ids:
        dirty &= downstream_dependents(store, target_ids) | target_ids
    else:
        dirty = set()

    reusable = {
        node.node_id
        for node in state.nodes.values()
        if node.task_id == task_id
        and node.branch_id == branch_id
        and node.is_operationally_valid()
        and node.node_id not in dirty
    }
    ordered = _topological_order(store, dirty) if dirty else []
    frontier = [
        node_id
        for node_id in ordered
        if not any(
            edge.target == node_id
            and edge.source in dirty
            and edge.relation in DATA_DEPENDENCY_RELATIONS
            for edge in state.edges
        )
    ]
    roles = sorted(
        {
            state.nodes[node_id].owner
            for node_id in dirty
            if state.nodes[node_id].owner
        }
    )
    reasons = {
        node_id: "stale dependency requires local recompute"
        for node_id in sorted(dirty)
    }
    return RecomputePlan(
        task_id=task_id,
        branch_id=branch_id,
        dirty_node_ids=ordered,
        reusable_node_ids=sorted(reusable),
        recompute_frontier=frontier,
        required_roles=roles,
        reason_by_node=reasons,
    )
