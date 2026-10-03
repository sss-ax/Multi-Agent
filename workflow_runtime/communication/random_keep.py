"""Deterministic keep-ratio root selection policies."""

from __future__ import annotations

import hashlib

from .base import GraphCommunicationPolicy
from ..agent_graph_view import AgentGraphView
from ..graph_delta import DeltaCandidate
from ..models import GraphState


class RandomKeepPolicy(GraphCommunicationPolicy):
    """Keep a deterministic pseudo-random subset of novel optional roots."""

    def __init__(self, *, ratio: float, seed: int = 0) -> None:
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError("random keep ratio must be in [0, 1]")
        self.ratio = float(ratio)
        self.seed = int(seed)
        self.name = f"random_keep_{int(round(self.ratio * 100))}"

    def select_roots(
        self,
        *,
        state: GraphState,
        sender_view: AgentGraphView,
        receiver_view: AgentGraphView,
        candidates: list[DeltaCandidate],
        task: str = "",
        budget_tokens: int | None = None,
    ) -> list[str]:
        selected: list[str] = []
        for candidate in candidates:
            if not candidate.exportable or candidate.novelty_score <= 0:
                continue
            if candidate.scope != "OPTIONAL_POLICY":
                continue
            if self._keep(candidate):
                selected.append(candidate.node_id)
        return selected

    def _keep(self, candidate: DeltaCandidate) -> bool:
        key = "|".join((
            str(self.seed),
            self.name,
            candidate.sender,
            candidate.receiver,
            candidate.node_id,
        ))
        value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
        threshold = int(self.ratio * ((1 << 64) - 1))
        return value <= threshold
