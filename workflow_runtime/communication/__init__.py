"""Graph-delta communication policies."""

from .base import GraphCommunicationPolicy
from .fragment_ablation import FragmentAblationPolicy, FRAGMENT_ABLATION_POLICY_PREFIX, FRAGMENT_ABLATION_TARGETS
from .heuristic import ClosureAwareHeuristicPolicy
from .minimal import MinimalNoFeedbackPolicy, MinimalSendAllFallbackPolicy, MinimalTargetedFeedbackPolicy
from .random_keep import RandomKeepPolicy
from .send_all import SendAllPolicy
from .utility import RandomSameBudgetPolicy, ReceiverAwareHeuristicPolicy, StaticUtilityPolicy, UtilityTable

__all__ = [
    "GraphCommunicationPolicy",
    "SendAllPolicy",
    "MinimalNoFeedbackPolicy",
    "MinimalSendAllFallbackPolicy",
    "MinimalTargetedFeedbackPolicy",
    "RandomKeepPolicy",
    "ClosureAwareHeuristicPolicy",
    "FragmentAblationPolicy",
    "FRAGMENT_ABLATION_POLICY_PREFIX",
    "FRAGMENT_ABLATION_TARGETS",
    "RandomSameBudgetPolicy",
    "ReceiverAwareHeuristicPolicy",
    "StaticUtilityPolicy",
    "UtilityTable",
    "make_communication_policy",
]


def make_communication_policy(
    name: str | None,
    *,
    seed: int = 0,
    utility_table_path: str | None = None,
    task_family: str = "",
) -> GraphCommunicationPolicy:
    key = (name or "closure_aware_heuristic").strip().lower()
    utility_table = None
    if key in {"static_utility", "receiver_aware_heuristic", "random_same_budget"}:
        utility_table = UtilityTable.from_path(utility_table_path, task_family=task_family)
    if key == "send_all":
        return SendAllPolicy()
    if key == "minimal_no_feedback":
        return MinimalNoFeedbackPolicy()
    if key == "minimal_sendall_fallback":
        return MinimalSendAllFallbackPolicy()
    if key == "minimal_targeted_feedback":
        return MinimalTargetedFeedbackPolicy()
    if key == "random_keep_75":
        return RandomKeepPolicy(ratio=0.75, seed=seed)
    if key == "random_keep_50":
        return RandomKeepPolicy(ratio=0.50, seed=seed)
    if key == "random_keep_25":
        return RandomKeepPolicy(ratio=0.25, seed=seed)
    if key == "closure_aware_heuristic":
        return ClosureAwareHeuristicPolicy()
    if key == "static_utility":
        return StaticUtilityPolicy(utility_table)
    if key == "receiver_aware_heuristic":
        return ReceiverAwareHeuristicPolicy(utility_table, task_family=task_family)
    if key == "random_same_budget":
        return RandomSameBudgetPolicy(utility_table, task_family=task_family, seed=seed)
    if key.startswith(FRAGMENT_ABLATION_POLICY_PREFIX):
        return FragmentAblationPolicy(key.removeprefix(FRAGMENT_ABLATION_POLICY_PREFIX))
    raise ValueError(f"unknown communication policy: {name}")
