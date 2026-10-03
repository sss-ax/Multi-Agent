"""Small scheduling helpers for the data-dependency DAG."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Set

from .graph_store import GraphStore
from .models import NodeVersion
from .relations import DATA_DEPENDENCY_RELATIONS


def topological_node_order(store: GraphStore, node_ids: Iterable[str]) -> List[str]:
    selected = set(node_ids)
    state = store.snapshot()
    indegree: Dict[str, int] = {node_id: 0 for node_id in selected}
    adjacency: Dict[str, List[str]] = defaultdict(list)
    for edge in state.edges:
        if edge.relation not in DATA_DEPENDENCY_RELATIONS:
            continue
        if edge.source in selected and edge.target in selected:
            adjacency[edge.source].append(edge.target)
            indegree[edge.target] += 1
    ready = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    result: List[str] = []
    while ready:
        current = ready.popleft()
        result.append(current)
        for target in sorted(adjacency.get(current, [])):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if len(result) != len(selected):
        raise ValueError("selected nodes do not form a DAG")
    return result


def roles_in_order(store: GraphStore, node_ids: Iterable[str]) -> List[str]:
    state = store.snapshot()
    seen: Set[str] = set()
    result: List[str] = []
    for node_id in topological_node_order(store, node_ids):
        role = state.nodes[node_id].owner
        if role and role not in seen:
            result.append(role)
            seen.add(role)
    return result
