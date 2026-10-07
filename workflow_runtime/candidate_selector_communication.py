"""Receiver-aware fragment communication for candidate selectors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .candidate_selector import score_candidate_features_with_reasons


CORE_FRAGMENTS = {"final_value", "status"}
VERIFIER_FRAGMENTS = {
    "verifier_summary",
    "answer_cluster",
    "disagreement",
    "unique_claim",
    "conflict_signal",
}
REQUESTED_EVIDENCE_FRAGMENTS = {"failed_test", "expression", "trace"}
FULL_FRAGMENTS = {"full_artifact"}


@dataclass(frozen=True)
class CandidateFragment:
    fragment_id: str
    candidate_id: str
    fragment_name: str
    content: Any
    token_cost: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "candidate_id": self.candidate_id,
            "fragment_name": self.fragment_name,
            "content": self.content,
            "token_cost": self.token_cost,
        }


@dataclass(frozen=True)
class SelectorCommunicationResult:
    policy: str
    selected_candidate_id: str
    selection_score: float
    selection_reason: tuple[str, ...]
    visible_fragment_ids: tuple[str, ...]
    rendered_tokens: int
    trace: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "selected_candidate_id": self.selected_candidate_id,
            "selection_score": self.selection_score,
            "selection_reason": list(self.selection_reason),
            "visible_fragment_ids": list(self.visible_fragment_ids),
            "rendered_tokens": self.rendered_tokens,
            "trace": [dict(item) for item in self.trace],
        }


def build_candidate_fragments(candidates: Sequence[Mapping[str, Any]]) -> list[CandidateFragment]:
    answer_counts: dict[str, int] = {}
    for candidate in candidates:
        value = str(candidate.get("final_value") or candidate.get("artifact") or "").strip()
        if value:
            answer_counts[value] = answer_counts.get(value, 0) + 1

    fragments: list[CandidateFragment] = []
    for index, candidate in enumerate(candidates):
        candidate_id = str(candidate.get("candidate_id") or f"c{index}")
        score = candidate.get("score") if isinstance(candidate.get("score"), Mapping) else {}
        signals = candidate.get("verification_signals") if isinstance(candidate.get("verification_signals"), Mapping) else {}
        value = str(candidate.get("final_value") or candidate.get("artifact") or "").strip()
        summary = _sanitized_summary({**dict(score), **dict(signals)})
        conflict = _conflict_signal(summary)
        cluster_size = answer_counts.get(value, 0)
        for name, content in (
            ("final_value", value),
            ("status", {"artifact_type": candidate.get("artifact_type", ""), "generation_index": index}),
            ("verifier_summary", summary),
            ("answer_cluster", {"value": value, "cluster_size": cluster_size}),
            ("disagreement", {"disagrees_with_majority": cluster_size < max(answer_counts.values() or [0])}),
            ("unique_claim", {"unique": cluster_size == 1}),
            ("conflict_signal", conflict),
            ("failed_test", summary.get("failed_test") or summary.get("failed_test_count")),
            ("expression", summary.get("expression") or summary.get("expression_parseable")),
            ("trace", candidate.get("raw_tail", "")),
            ("full_artifact", candidate.get("artifact", "")),
        ):
            fragments.append(
                CandidateFragment(
                    fragment_id=f"{candidate_id}#{name}",
                    candidate_id=candidate_id,
                    fragment_name=name,
                    content=content,
                    token_cost=_token_cost(content),
                )
            )
    return fragments


def select_fragment_ids(
    fragments: Sequence[CandidateFragment],
    *,
    policy: str,
    seed: int = 0,
    budget_tokens: int | None = None,
    oracle_candidate_id: str | None = None,
) -> tuple[str, ...]:
    if policy == "selector_send_all":
        selected = [fragment.fragment_id for fragment in fragments]
    elif policy == "selector_core_only":
        selected = [
            fragment.fragment_id for fragment in fragments
            if fragment.fragment_name in CORE_FRAGMENTS
        ]
    elif policy == "selector_verifier_only":
        selected = [
            fragment.fragment_id for fragment in fragments
            if fragment.fragment_name in CORE_FRAGMENTS | VERIFIER_FRAGMENTS
        ]
    elif policy == "selector_receiver_aware":
        selected = [
            fragment.fragment_id for fragment in fragments
            if fragment.fragment_name in CORE_FRAGMENTS | {"verifier_summary", "conflict_signal"}
        ]
        selected_set = set(selected)
        unresolved = {
            fragment.candidate_id for fragment in fragments
            if fragment.fragment_name == "conflict_signal" and bool(fragment.content)
        }
        for fragment in fragments:
            if fragment.candidate_id in unresolved and fragment.fragment_name in {"failed_test", "expression"}:
                selected_set.add(fragment.fragment_id)
        selected = list(selected_set)
    elif policy == "selector_random_same_budget":
        if budget_tokens is None:
            budget_tokens = sum(
                fragment.token_cost for fragment in fragments
                if fragment.fragment_name in CORE_FRAGMENTS | {"verifier_summary", "conflict_signal"}
            )
        used = 0
        selected = []
        for fragment in sorted(fragments, key=lambda item: _stable_random_key(item, seed=seed)):
            if used + fragment.token_cost > budget_tokens:
                continue
            selected.append(fragment.fragment_id)
            used += fragment.token_cost
    elif policy == "selector_oracle_minimal":
        if not oracle_candidate_id:
            raise ValueError("selector_oracle_minimal requires oracle_candidate_id")
        selected = [
            fragment.fragment_id for fragment in fragments
            if fragment.candidate_id == oracle_candidate_id
            and fragment.fragment_name in CORE_FRAGMENTS | {"verifier_summary"}
        ]
    else:
        raise ValueError(f"unknown selector communication policy: {policy}")
    return tuple(sorted(dict.fromkeys(selected)))


def run_selector_with_fragments(
    candidates: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    seed: int = 0,
    budget_tokens: int | None = None,
    oracle_candidate_id: str | None = None,
) -> SelectorCommunicationResult:
    fragments = build_candidate_fragments(candidates)
    selected_ids = select_fragment_ids(
        fragments,
        policy=policy,
        seed=seed,
        budget_tokens=budget_tokens,
        oracle_candidate_id=oracle_candidate_id,
    )
    visible = render_visible_candidate_fragments(fragments, selected_ids)
    return select_from_visible_fragments(
        visible,
        policy=policy,
        visible_fragment_ids=selected_ids,
    )


def render_visible_candidate_fragments(
    fragments: Sequence[CandidateFragment],
    visible_fragment_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    visible = set(visible_fragment_ids)
    rendered: dict[str, dict[str, Any]] = {}
    for fragment in fragments:
        if fragment.fragment_id not in visible:
            continue
        candidate = rendered.setdefault(fragment.candidate_id, {})
        candidate[fragment.fragment_name] = fragment.content
    return rendered


def select_from_visible_fragments(
    visible: Mapping[str, Mapping[str, Any]],
    *,
    policy: str,
    visible_fragment_ids: Sequence[str],
) -> SelectorCommunicationResult:
    if not visible:
        raise ValueError("selector cannot run without visible candidate fragments")
    trace = []
    for candidate_id, fragments in visible.items():
        features = _features_from_visible(fragments)
        score, reasons = score_candidate_features_with_reasons(features)
        trace.append({
            "candidate_id": candidate_id,
            "features": features,
            "feature_score": score,
            "selection_reason": list(reasons),
        })
    trace = sorted(trace, key=lambda item: (item["feature_score"], item["candidate_id"]), reverse=True)
    best = trace[0]
    rendered_tokens = rendered_fragment_tokens(
        build_candidate_fragments(_visible_to_candidate_records(visible)),
        visible_fragment_ids,
    )
    return SelectorCommunicationResult(
        policy=policy,
        selected_candidate_id=str(best["candidate_id"]),
        selection_score=float(best["feature_score"]),
        selection_reason=tuple(best["selection_reason"]),
        visible_fragment_ids=tuple(sorted(visible_fragment_ids)),
        rendered_tokens=rendered_tokens,
        trace=tuple(trace),
    )


def rendered_fragment_tokens(
    fragments: Sequence[CandidateFragment],
    visible_fragment_ids: Sequence[str],
) -> int:
    visible = set(visible_fragment_ids)
    return sum(fragment.token_cost for fragment in fragments if fragment.fragment_id in visible)


def _features_from_visible(fragments: Mapping[str, Any]) -> dict[str, Any]:
    summary = fragments.get("verifier_summary") if isinstance(fragments.get("verifier_summary"), Mapping) else {}
    features = dict(summary)
    if "final_value" in fragments:
        features["final_value_valid"] = bool(str(fragments.get("final_value", "")).strip())
    cluster = fragments.get("answer_cluster") if isinstance(fragments.get("answer_cluster"), Mapping) else {}
    if "cluster_size" in cluster:
        features["cross_candidate_answer_consensus"] = float(cluster["cluster_size"])
    if fragments.get("conflict_signal"):
        features["constraint_inconsistency"] = True
    if fragments.get("failed_test"):
        features["failed_test_count"] = features.get("failed_test_count", 1)
    if fragments.get("expression"):
        features["expression_parseable"] = True
    if fragments.get("trace"):
        features["trace_result_agreement"] = bool(features.get("trace_result_agreement", False))
    return features


def _sanitized_summary(values: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = {"correct", "gold_answer", "gold_label", "hidden_test_result", "hidden_tests", "oracle_correct"}
    return {str(key): value for key, value in values.items() if str(key) not in forbidden}


def _conflict_signal(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: bool(summary.get(key))
        for key in (
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
        )
        if bool(summary.get(key))
    }


def _visible_to_candidate_records(visible: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for candidate_id, fragments in visible.items():
        records.append({
            "candidate_id": candidate_id,
            "artifact": fragments.get("full_artifact", ""),
            "final_value": fragments.get("final_value", ""),
            "score": fragments.get("verifier_summary", {}),
            "verification_signals": fragments.get("verifier_summary", {}),
            "raw_tail": fragments.get("trace", ""),
        })
    return records


def _token_cost(content: Any) -> int:
    text = json.dumps(content, ensure_ascii=False, sort_keys=True)
    return max(1, (len(text) + 3) // 4)


def _stable_random_key(fragment: CandidateFragment, *, seed: int) -> str:
    raw = f"{seed}|{fragment.fragment_id}|{fragment.fragment_name}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
