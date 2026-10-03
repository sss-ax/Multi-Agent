"""Graph-delta communication policies."""

from .base import GraphCommunicationPolicy
from .heuristic import ClosureAwareHeuristicPolicy
from .minimal import MinimalNoFeedbackPolicy, MinimalSendAllFallbackPolicy, MinimalTargetedFeedbackPolicy
from .random_keep import RandomKeepPolicy
from .send_all import SendAllPolicy

__all__ = [
    "GraphCommunicationPolicy",
    "SendAllPolicy",
    "MinimalNoFeedbackPolicy",
    "MinimalSendAllFallbackPolicy",
    "MinimalTargetedFeedbackPolicy",
    "RandomKeepPolicy",
    "ClosureAwareHeuristicPolicy",
    "make_communication_policy",
]


def make_communication_policy(name: str | None, *, seed: int = 0) -> GraphCommunicationPolicy:
    key = (name or "closure_aware_heuristic").strip().lower()
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
    raise ValueError(f"unknown communication policy: {name}")
