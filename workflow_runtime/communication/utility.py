"""Utility-table-driven fragment selection policies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .base import GraphCommunicationPolicy
from ..agent_graph_view import AgentGraphView
from ..graph_delta import DeltaCandidate
from ..models import GraphState


SUPPORT_DEPENDENCY_FRAGMENTS = {"dependencies", "key_operation", "task_input"}


class UtilityTable:
    """Lookup utility estimates produced by fragment utility ablation."""

    def __init__(self, payload: dict[str, Any] | None = None, *, task_family: str = "") -> None:
        data = payload or {}
        tables = data.get("utility_tables", data)
        self.global_table = dict(tables.get("global", {})) if isinstance(tables, dict) else {}
        self.conditioned_table = dict(tables.get("conditioned", {})) if isinstance(tables, dict) else {}
        self.task_family = str(task_family or data.get("domain", "")).strip()

    @classmethod
    def from_path(cls, path: str | Path | None, *, task_family: str = "") -> "UtilityTable":
        if not path:
            return cls(task_family=task_family)
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"fragment utility table does not exist: {source}")
        return cls(json.loads(source.read_text(encoding="utf-8")), task_family=task_family)

    def global_utility(self, fragment_name: str) -> float:
        return self._mean(self.global_table.get(fragment_arm(fragment_name)))

    def receiver_aware_utility(self, fragment_name: str, *, receiver: str, task_family: str = "") -> float:
        arm = fragment_arm(fragment_name)
        family = str(task_family or self.task_family).strip()
        key = f"{arm}|{receiver}|{family}"
        if key in self.conditioned_table:
            return self._mean(self.conditioned_table.get(key))
        return self.global_utility(fragment_name)

    @staticmethod
    def _mean(value: Any) -> float:
        if isinstance(value, dict):
            return float(value.get("mean_utility", 0.0) or 0.0)
        return 0.0


class StaticUtilityPolicy(GraphCommunicationPolicy):
    name = "static_utility"

    def __init__(self, utility_table: UtilityTable | None = None) -> None:
        self.utility_table = utility_table or UtilityTable()

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
        return _select_by_score(
            candidates,
            budget_tokens=budget_tokens,
            score=lambda candidate: self.utility_table.global_utility(candidate.fragment_name),
        )


class ReceiverAwareHeuristicPolicy(GraphCommunicationPolicy):
    name = "receiver_aware_heuristic"

    def __init__(self, utility_table: UtilityTable | None = None, *, task_family: str = "") -> None:
        self.utility_table = utility_table or UtilityTable(task_family=task_family)
        self.task_family = task_family

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
        return _select_by_score(
            candidates,
            budget_tokens=budget_tokens,
            score=lambda candidate: self.utility_table.receiver_aware_utility(
                candidate.fragment_name,
                receiver=candidate.receiver,
                task_family=self.task_family or task,
            ),
        )


class RandomSameBudgetPolicy(GraphCommunicationPolicy):
    name = "random_same_budget"

    def __init__(self, utility_table: UtilityTable | None = None, *, task_family: str = "", seed: int = 0) -> None:
        self.utility_table = utility_table or UtilityTable(task_family=task_family)
        self.task_family = task_family
        self.seed = int(seed)

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
        heuristic_selected = _select_by_score(
            candidates,
            budget_tokens=budget_tokens,
            score=lambda candidate: self.utility_table.receiver_aware_utility(
                candidate.fragment_name,
                receiver=candidate.receiver,
                task_family=self.task_family or task,
            ),
        )
        heuristic_budget = sum(
            candidate.token_cost for candidate in candidates
            if candidate.node_id in set(heuristic_selected)
        )
        if heuristic_budget <= 0:
            return []
        selected: list[str] = []
        used = 0
        for candidate in sorted(
            _eligible(candidates),
            key=lambda item: _stable_random_key(item, seed=self.seed),
        ):
            if used + candidate.token_cost > heuristic_budget:
                continue
            selected.append(candidate.node_id)
            used += candidate.token_cost
        return selected


def fragment_arm(fragment_name: str) -> str:
    name = str(fragment_name).strip()
    if name in SUPPORT_DEPENDENCY_FRAGMENTS:
        return "support_dependencies"
    return name


def _eligible(candidates: list[DeltaCandidate]) -> list[DeltaCandidate]:
    return [
        candidate for candidate in candidates
        if candidate.exportable
        and candidate.scope == "OPTIONAL_POLICY"
        and candidate.novelty_score > 0
        and candidate.fragment_id
    ]


def _select_by_score(
    candidates: list[DeltaCandidate],
    *,
    budget_tokens: int | None,
    score,
) -> list[str]:
    ranked = []
    for candidate in _eligible(candidates):
        value = float(score(candidate))
        if value <= 0:
            continue
        ranked.append((value / max(1, candidate.token_cost), value, -candidate.token_cost, candidate.node_id))
    selected: list[str] = []
    used = 0
    candidate_by_id = {candidate.node_id: candidate for candidate in candidates}
    for _ratio, _value, _neg_cost, node_id in sorted(ranked, reverse=True):
        candidate = candidate_by_id[node_id]
        if budget_tokens is not None and used + candidate.token_cost > budget_tokens:
            continue
        selected.append(node_id)
        used += candidate.token_cost
    return selected


def _stable_random_key(candidate: DeltaCandidate, *, seed: int) -> str:
    raw = "|".join((
        str(seed),
        candidate.sender,
        candidate.receiver,
        candidate.node_id,
        candidate.fragment_name,
    ))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
