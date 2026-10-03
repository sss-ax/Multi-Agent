"""Graph-delta data structures for inter-agent semantic communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DeltaCandidate:
    """One exportable canonical graph node considered for transmission."""

    node_id: str
    sender: str
    receiver: str
    exportable: bool = True
    novelty_score: float = 1.0
    redundancy_score: float = 0.0
    token_cost: int = 0
    structural_features: dict[str, Any] = field(default_factory=dict)
    scope: str = "OPTIONAL_POLICY"
    fragment_id: str = ""
    fragment_name: str = ""
    parent_node_id: str = ""
    is_fragment: bool = False


@dataclass(frozen=True)
class GraphDelta:
    """A receiver-relative dependency-closed graph delta."""

    sender: str
    receiver: str
    root_node_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    token_cost: int = 0
    candidate_token_cost: int = 0
    closure_added_node_ids: tuple[str, ...] = ()
    redundant_node_ids: tuple[str, ...] = ()
    policy: str = ""

    @property
    def sent_count(self) -> int:
        return len(self.node_ids)
