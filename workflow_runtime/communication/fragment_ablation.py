"""Fragment-specific ablation policies for utility experiments."""

from __future__ import annotations

from .base import GraphCommunicationPolicy
from ..agent_graph_view import AgentGraphView
from ..graph_delta import DeltaCandidate
from ..models import GraphState


FRAGMENT_ABLATION_POLICY_PREFIX = "fragment_ablation_"

FRAGMENT_ABLATION_TARGETS = {
    "core_only": (),
    "support_dependencies": ("dependencies", "key_operation", "task_input"),
    "full_plan": ("full_plan",),
    "result_metadata": ("result_metadata",),
    "calculation_trace": ("calculation_trace",),
    "full_feedback": ("full_feedback",),
}


class FragmentAblationPolicy(GraphCommunicationPolicy):
    """Select only optional fragments named by one ablation arm."""

    def __init__(self, arm: str) -> None:
        key = arm.strip().lower()
        if key not in FRAGMENT_ABLATION_TARGETS:
            raise ValueError(f"unknown fragment ablation arm: {arm}")
        self.arm = key
        self.name = f"{FRAGMENT_ABLATION_POLICY_PREFIX}{key}"
        self.fragment_names = tuple(FRAGMENT_ABLATION_TARGETS[key])

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
        used = 0
        for candidate in candidates:
            if not candidate.exportable or candidate.scope != "OPTIONAL_POLICY":
                continue
            if candidate.novelty_score <= 0:
                continue
            if candidate.fragment_name not in self.fragment_names:
                continue
            if budget_tokens is not None and used + candidate.token_cost > budget_tokens:
                continue
            selected.append(candidate.node_id)
            used += candidate.token_cost
        return selected
