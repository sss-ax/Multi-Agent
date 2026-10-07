"""Feature extraction and deployable ranking for candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .candidate_graph import (
    CandidateGraphResult,
    assert_deployable_verifier_payload,
)
from .graph_store import GraphStore


ORACLE_ONLY_FEATURE_KEYS = frozenset(
    {
        "correct",
        "status",
        "gold_answer",
        "gold_label",
        "hidden_test_result",
        "hidden_tests",
        "oracle_correct",
        "final_value",
        "artifact",
    }
)


@dataclass(frozen=True)
class CandidateSelectorExample:
    group_node_id: str
    candidate_id: str
    candidate_node_id: str
    artifact_node_id: str
    verification_node_id: str
    domain: str
    features: dict[str, Any]
    label: int | None = None
    label_source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_node_id": self.group_node_id,
            "candidate_id": self.candidate_id,
            "candidate_node_id": self.candidate_node_id,
            "artifact_node_id": self.artifact_node_id,
            "verification_node_id": self.verification_node_id,
            "domain": self.domain,
            "features": dict(self.features),
            "label": self.label,
            "label_source": self.label_source,
        }


@dataclass(frozen=True)
class DeployableSelectionResult:
    selected_candidate_id: str
    selection_score: float
    selection_reason: tuple[str, ...]
    confidence: float
    trace: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "selection_score": self.selection_score,
            "selection_reason": list(self.selection_reason),
            "confidence": self.confidence,
            "trace": [dict(item) for item in self.trace],
        }


def build_oracle_selector_examples(
    store: GraphStore,
    pool: CandidateGraphResult,
    *,
    label_source: str = "oracle_evaluator",
) -> list[CandidateSelectorExample]:
    """Build supervised selector examples with oracle correctness only as label."""

    examples: list[CandidateSelectorExample] = []
    for candidate_id in _ordered_candidate_ids(store, pool):
        verification = store.node(pool.verification_node_ids[candidate_id])
        content = verification.content if isinstance(verification.content, Mapping) else {}
        label = 1 if bool(content.get("correct")) else 0
        examples.append(
            _selector_example(
                store,
                pool,
                candidate_id,
                label=label,
                label_source=label_source,
                enforce_deployable=False,
            )
        )
    return examples


def build_deployable_selector_examples(
    store: GraphStore,
    pool: CandidateGraphResult,
) -> list[CandidateSelectorExample]:
    """Build unlabeled selector examples from non-oracle verifier payloads only."""

    return [
        _selector_example(
            store,
            pool,
            candidate_id,
            label=None,
            label_source="",
            enforce_deployable=True,
        )
        for candidate_id in _ordered_candidate_ids(store, pool)
    ]


def select_candidate_by_feature_score(
    store: GraphStore,
    pool: CandidateGraphResult,
) -> tuple[str, list[dict[str, Any]]]:
    """Deployable deterministic selector over sanitized verifier features."""

    result = select_candidate_with_deployable_verifier(store, pool)
    return result.selected_candidate_id, list(result.trace)


def select_candidate_with_deployable_verifier(
    store: GraphStore,
    pool: CandidateGraphResult,
) -> DeployableSelectionResult:
    """Deployable deterministic selector with score, reasons, confidence, trace."""

    examples = build_deployable_selector_examples(store, pool)
    if not examples:
        raise ValueError("cannot select from an empty candidate pool")
    scored: list[tuple[float, int, str, CandidateSelectorExample, tuple[str, ...]]] = []
    for example in examples:
        generation_index = int(example.features.get("generation_index", 0) or 0)
        score, reasons = score_candidate_features_with_reasons(example.features)
        scored.append((
            score,
            -generation_index,
            example.candidate_id,
            example,
            reasons,
        ))
    ranked = sorted(scored, key=lambda item: (item[0], item[1], item[2]), reverse=True)
    best = ranked[0]
    trace = tuple(
        {
            **example.as_dict(),
            "feature_score": score,
            "rank_tiebreak_generation": -neg_generation,
            "selection_reason": list(reasons),
        }
        for score, neg_generation, _candidate_id, example, reasons in ranked
    )
    confidence = _selection_confidence(ranked)
    return DeployableSelectionResult(
        selected_candidate_id=best[3].candidate_id,
        selection_score=best[0],
        selection_reason=best[4],
        confidence=confidence,
        trace=trace,
    )


def score_candidate_features(features: Mapping[str, Any]) -> float:
    return score_candidate_features_with_reasons(features)[0]


def score_candidate_features_with_reasons(features: Mapping[str, Any]) -> tuple[float, tuple[str, ...]]:
    assert_no_oracle_features(features)
    score = 0.0
    reasons: list[str] = []
    confidence = _float(features.get("verifier_confidence"))
    score += confidence
    if confidence > 0:
        reasons.append(f"verifier_confidence={confidence:.3f}")

    for key, weight in (
        ("tests_passed", 12.0),
        ("verifier_pass", 12.0),
        ("public_tests_passed", 10.0),
        ("execution_valid", 8.0),
        ("runtime_ok", 3.0),
        ("no_exception", 2.0),
        ("syntax_valid", 1.0),
        ("compile_success", 1.0),
        ("arithmetic_consistency", 3.0),
        ("constraint_consistency", 5.0),
        ("trace_result_agreement", 3.0),
        ("independent_recompute_consistency", 5.0),
        ("final_value_valid", 0.8),
        ("expression_parseable", 0.6),
        ("option_valid", 5.0),
        ("evidence_covered", 4.0),
    ):
        if bool(features.get(key)):
            score += weight
            reasons.append(f"+{key}")

    score += 0.5 * _float(features.get("cross_candidate_answer_consensus"))
    if _float(features.get("cross_candidate_answer_consensus")) > 0:
        reasons.append("+cross_candidate_answer_consensus")

    pass_rate = _optional_float(features.get("test_pass_rate"))
    if pass_rate is not None:
        score += 10.0 * pass_rate
        reasons.append(f"test_pass_rate={pass_rate:.3f}")

    execution_diversity = _optional_float(features.get("execution_diversity"))
    if execution_diversity is not None:
        score += min(2.0, execution_diversity)
        reasons.append(f"execution_diversity={execution_diversity:.3f}")

    for key, weight in (
        ("tests_failed", 8.0),
        ("execution_failed", 8.0),
        ("assertion_failure", 4.0),
        ("runtime_exception", 8.0),
        ("timeout", 12.0),
        ("wrong_return_type", 6.0),
        ("syntax_error", 12.0),
        ("compile_error", 10.0),
        ("invalid_option", 10.0),
        ("missing_evidence", 3.0),
        ("trace_result_disagreement", 5.0),
        ("constraint_inconsistency", 7.0),
        ("hardcoded_output", 4.0),
    ):
        if bool(features.get(key)):
            score -= weight
            reasons.append(f"-{key}")

    failed_test_count = _optional_float(features.get("failed_test_count"))
    if failed_test_count is not None:
        penalty = min(10.0, failed_test_count)
        score -= penalty
        reasons.append(f"failed_test_count={int(failed_test_count)}")

    severity = _optional_float(features.get("exception_severity"))
    if severity is not None:
        score -= severity
        reasons.append(f"exception_severity={severity:.3f}")

    score -= 0.001 * _float(features.get("generation_index"))
    return score, tuple(reasons or ("deterministic_tiebreak",))


def assert_no_oracle_features(features: Mapping[str, Any]) -> None:
    forbidden = sorted(ORACLE_ONLY_FEATURE_KEYS.intersection(features))
    if forbidden:
        raise ValueError(f"selector features contain oracle/value fields: {forbidden}")
    assert_deployable_verifier_payload(features)


def _selector_example(
    store: GraphStore,
    pool: CandidateGraphResult,
    candidate_id: str,
    *,
    label: int | None,
    label_source: str,
    enforce_deployable: bool,
) -> CandidateSelectorExample:
    candidate = store.node(pool.candidate_node_ids[candidate_id])
    artifact = store.node(pool.artifact_node_ids[candidate_id])
    verification = store.node(pool.verification_node_ids[candidate_id])
    candidate_content = candidate.content if isinstance(candidate.content, Mapping) else {}
    artifact_content = artifact.content if isinstance(artifact.content, Mapping) else {}
    verification_content = verification.content if isinstance(verification.content, Mapping) else {}
    source = str(verification_content.get("verification_source") or "")
    if enforce_deployable:
        if "oracle" in source or "gold" in source or "hidden" in source:
            raise ValueError(f"deployable selector cannot use verifier source: {source}")
        assert_deployable_verifier_payload(verification_content)
        score = verification_content.get("score") if isinstance(verification_content.get("score"), Mapping) else {}
        signals = verification_content.get("signals") if isinstance(verification_content.get("signals"), Mapping) else {}
        assert_deployable_verifier_payload(score)
        assert_deployable_verifier_payload(signals)

    features = _sanitized_features(candidate_content, artifact_content, verification_content)
    assert_no_oracle_features(features)
    return CandidateSelectorExample(
        group_node_id=pool.group_node_id,
        candidate_id=candidate_id,
        candidate_node_id=pool.candidate_node_ids[candidate_id],
        artifact_node_id=pool.artifact_node_ids[candidate_id],
        verification_node_id=pool.verification_node_ids[candidate_id],
        domain=str(verification_content.get("domain") or artifact_content.get("domain") or ""),
        features=features,
        label=label,
        label_source=label_source,
    )


def _sanitized_features(
    candidate_content: Mapping[str, Any],
    artifact_content: Mapping[str, Any],
    verification_content: Mapping[str, Any],
) -> dict[str, Any]:
    score = verification_content.get("score") if isinstance(verification_content.get("score"), Mapping) else {}
    signals = verification_content.get("signals") if isinstance(verification_content.get("signals"), Mapping) else {}
    features: dict[str, Any] = {
        "generation_index": int(candidate_content.get("generation_index") or 0),
        "repair_round": int(candidate_content.get("repair_round") or 0),
        "input_tokens": int(candidate_content.get("input_tokens") or 0),
        "output_tokens": int(candidate_content.get("output_tokens") or 0),
        "artifact_type": str(artifact_content.get("artifact_type") or ""),
        "verifier_confidence": _float(score.get("confidence") or signals.get("confidence")),
    }
    for key in (
        "tests_passed",
        "verifier_pass",
        "execution_valid",
        "runtime_ok",
        "no_exception",
        "syntax_valid",
        "compile_success",
        "public_tests_passed",
        "arithmetic_consistent",
        "final_value_valid",
        "expression_parseable",
        "arithmetic_consistency",
        "constraint_consistency",
        "trace_result_agreement",
        "independent_recompute_consistency",
        "option_valid",
        "evidence_covered",
        "tests_failed",
        "execution_failed",
        "assertion_failure",
        "runtime_exception",
        "timeout",
        "wrong_return_type",
        "syntax_error",
        "compile_error",
        "invalid_option",
        "missing_evidence",
        "trace_result_disagreement",
        "constraint_inconsistency",
        "hardcoded_output",
    ):
        if key in score:
            features[key] = bool(score[key])
        if key in signals:
            features[key] = bool(signals[key])
    for key in (
        "test_pass_rate",
        "failed_test_count",
        "exception_severity",
        "execution_diversity",
        "cross_candidate_answer_consensus",
    ):
        if key in score:
            features[key] = _float(score[key])
        if key in signals:
            features[key] = _float(signals[key])
    return features


def _ordered_candidate_ids(store: GraphStore, pool: CandidateGraphResult) -> list[str]:
    return sorted(
        pool.candidate_node_ids,
        key=lambda candidate_id: _candidate_generation_index(store, pool.candidate_node_ids[candidate_id]),
    )


def _candidate_generation_index(store: GraphStore, candidate_node_id: str) -> int:
    node = store.node(candidate_node_id)
    content = node.content if isinstance(node.content, Mapping) else {}
    try:
        return int(content.get("generation_index", 0))
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _selection_confidence(ranked: list[tuple[float, int, str, CandidateSelectorExample, tuple[str, ...]]]) -> float:
    if not ranked:
        return 0.0
    if len(ranked) == 1:
        return 1.0
    margin = ranked[0][0] - ranked[1][0]
    return max(0.0, min(1.0, margin / 10.0))
