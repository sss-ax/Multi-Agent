"""Deployable state features and strict oracle-leakage checks."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from ..candidate_selector import ORACLE_ONLY_FEATURE_KEYS
from ..communication_replay import CommunicationDecisionSnapshot
from .schemas import CurrentAnswerLabel, DualRiskExample, FinalOutcomeLabel


FORBIDDEN_TRAINING_ONLY_KEYS = frozenset({
    *ORACLE_ONLY_FEATURE_KEYS,
    "current_answer_correct",
    "final_task_success",
    "final_correct",
    "teacher_action",
    "oracle_error_type",
    "reference_answer",
    "reward",
})


def assert_deployable_features(features: Mapping[str, Any]) -> None:
    forbidden: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                name = str(key)
                child = f"{path}.{name}" if path else name
                if name.casefold() in FORBIDDEN_TRAINING_ONLY_KEYS:
                    forbidden.append(child)
                visit(item, child)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(features, "")
    if forbidden:
        raise ValueError(f"deployable features contain oracle/training-only fields: {sorted(forbidden)}")


def extract_deployable_error_features(
    snapshot: CommunicationDecisionSnapshot,
) -> dict[str, Any]:
    graph = snapshot.graph_state
    receiver_payload = snapshot.agent_views.get("views", {}).get(snapshot.receiver, {})
    visible_nodes = set(receiver_payload.get("visible_node_ids", ()))
    visible_fragments = set(receiver_payload.get("visible_fragment_ids", ()))
    type_counts = Counter(node.type for node in graph.nodes.values())
    visible_type_counts = Counter(
        graph.nodes[node_id].type
        for node_id in visible_nodes
        if node_id in graph.nodes
    )
    validation_counts = Counter()
    confidence_values: list[float] = []
    for node in graph.nodes.values():
        for key, value in node.validation.items():
            if value:
                validation_counts[str(key)] += 1
        if isinstance(node.confidence, (int, float)) and not isinstance(node.confidence, bool):
            confidence_values.append(float(node.confidence))
    fragment_scope_counts = Counter(str(item.get("scope", "")) for item in snapshot.fragments)
    features: dict[str, Any] = {
        "task_type": snapshot.task_type,
        "sender_role": snapshot.sender,
        "receiver_role": snapshot.receiver,
        "stage": snapshot.stage,
        "round_index": snapshot.round_index,
        "remaining_budget": snapshot.remaining_budget,
        "graph_node_count": len(graph.nodes),
        "graph_edge_count": len(graph.edges),
        "visible_node_count": len(visible_nodes),
        "visible_fragment_count": len(visible_fragments),
        "candidate_count": len(snapshot.candidates),
        "fragment_count": len(snapshot.fragments),
        "mandatory_fragment_count": fragment_scope_counts.get("MANDATORY_STAGE", 0),
        "optional_fragment_count": fragment_scope_counts.get("OPTIONAL_POLICY", 0),
        "node_type_counts": dict(sorted(type_counts.items())),
        "visible_node_type_counts": dict(sorted(visible_type_counts.items())),
        "validation_signal_counts": dict(sorted(validation_counts.items())),
        "mean_declared_confidence": (
            sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        ),
    }
    assert_deployable_features(features)
    return features


def build_dual_risk_example(
    snapshot: CommunicationDecisionSnapshot,
    *,
    current_answer: CurrentAnswerLabel,
    final_outcome: FinalOutcomeLabel,
    policy_features: Mapping[str, Any] | None = None,
) -> DualRiskExample:
    """Build inputs for Verify(s) and Value(s, policy) without label leakage."""
    state_features = extract_deployable_error_features(snapshot)
    deployable_policy_features = {
        "policy_name": final_outcome.policy_name,
        **dict(policy_features or {}),
    }
    assert_deployable_features(deployable_policy_features)
    return DualRiskExample(
        snapshot_id=snapshot.snapshot_id,
        task_id=snapshot.task_id,
        state_features=state_features,
        policy_features=deployable_policy_features,
        current_answer=current_answer,
        final_outcome=final_outcome,
    )
