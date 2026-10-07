"""Candidate-pool materialization for verifier-guided selection experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .graph_store import GraphStore


_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_]+")
FORBIDDEN_DEPLOYABLE_VERIFIER_FIELDS = frozenset(
    {"gold_answer", "gold_label", "hidden_test_result", "hidden_tests", "oracle_correct"}
)


def safe_id(value: Any) -> str:
    text = _SAFE_ID_RE.sub("_", str(value).strip())
    text = text.strip("_").lower()
    return text or "item"


@dataclass(frozen=True)
class CandidateGraphResult:
    group_node_id: str
    candidate_node_ids: dict[str, str]
    artifact_node_ids: dict[str, str]
    verification_node_ids: dict[str, str]
    selection_node_id: str | None = None
    selected_candidate_id: str | None = None
    selected_artifact_node_id: str | None = None
    selected_verification_node_id: str | None = None
    invariant_errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_node_id": self.group_node_id,
            "candidate_node_ids": dict(self.candidate_node_ids),
            "artifact_node_ids": dict(self.artifact_node_ids),
            "verification_node_ids": dict(self.verification_node_ids),
            "selection_node_id": self.selection_node_id,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_artifact_node_id": self.selected_artifact_node_id,
            "selected_verification_node_id": self.selected_verification_node_id,
            "candidate_count": len(self.candidate_node_ids),
            "artifact_count": len(self.artifact_node_ids),
            "verification_count": len(self.verification_node_ids),
            "invariant_errors": list(self.invariant_errors),
        }


@dataclass(frozen=True)
class CandidateSelectionDecision:
    selected_candidate_id: str
    selected_candidate_node_id: str
    selected_artifact_node_id: str
    selected_verification_node_id: str
    selection_policy: str
    reason: str
    verified_candidate_ids: tuple[str, ...] = ()
    rejected_candidate_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "selected_candidate_node_id": self.selected_candidate_node_id,
            "selected_artifact_node_id": self.selected_artifact_node_id,
            "selected_verification_node_id": self.selected_verification_node_id,
            "selection_policy": self.selection_policy,
            "reason": self.reason,
            "verified_candidate_ids": list(self.verified_candidate_ids),
            "rejected_candidate_ids": list(self.rejected_candidate_ids),
        }


def coverage_at_k(candidates: Sequence[Mapping[str, Any]], ks: Sequence[int]) -> dict[str, float]:
    """Return pass/coverage@k for an already generated candidate list."""

    result: dict[str, float] = {}
    for k in ks:
        prefix = list(candidates[: max(0, int(k))])
        result[f"coverage@{int(k)}"] = 1.0 if any(bool(c.get("correct")) for c in prefix) else 0.0
    return result


def oracle_select_candidate_id(candidates: Sequence[Mapping[str, Any]]) -> str | None:
    """Oracle selector: first correct candidate, deterministic first-candidate fallback."""

    if not candidates:
        return None
    for index, candidate in enumerate(candidates):
        if bool(candidate.get("correct")):
            return str(candidate.get("candidate_id") or f"c{index}")
    return str(candidates[0].get("candidate_id") or "c0")


def assert_deployable_verifier_payload(payload: Mapping[str, Any]) -> None:
    forbidden = sorted(FORBIDDEN_DEPLOYABLE_VERIFIER_FIELDS.intersection(payload))
    if forbidden:
        raise ValueError(f"deployable verifier payload contains oracle-only fields: {forbidden}")


def materialize_candidate_pool(
    store: GraphStore,
    *,
    task_id: str,
    branch_id: str,
    group_id: str,
    domain: str,
    candidates: Sequence[Mapping[str, Any]],
    selector_policy: str | None = None,
    selected_candidate_id: str | None = None,
    source_node_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CandidateGraphResult:
    """Create graph nodes for a candidate pool and an optional selection.

    Candidate identity is local to the pool, while artifacts and verification
    signals are first-class graph nodes.  This keeps "candidate #3 was correct"
    auditable without letting candidate ids replace canonical graph ids.
    """

    safe_group = safe_id(group_id)
    group = store.add_node(
        task_id=task_id,
        branch_id=branch_id,
        logical_id=f"candidate_group_{safe_group}",
        node_type="candidate_group",
        content={
            "group_id": group_id,
            "domain": domain,
            "candidate_count": len(candidates),
            "metadata": dict(metadata or {}),
        },
        owner="candidate_graph",
        created_by_role="runtime",
    )
    if source_node_id:
        store.add_edge(
            source=source_node_id,
            target=group.node_id,
            relation="input_to",
            created_by_role="runtime",
            metadata={"purpose": "candidate_pool_source"},
        )

    candidate_node_ids: dict[str, str] = {}
    artifact_node_ids: dict[str, str] = {}
    verification_node_ids: dict[str, str] = {}
    selected_artifact_node_id: str | None = None
    selected_verification_node_id: str | None = None

    for index, candidate in enumerate(candidates):
        candidate_id = str(candidate.get("candidate_id") or f"c{index}")
        candidate_key = _unique_key(candidate_id, candidate_node_ids)
        safe_candidate = safe_id(candidate_key)
        correct = bool(candidate.get("correct"))
        score = candidate.get("score") or {}
        artifact_type = str(candidate.get("artifact_type") or "text")

        candidate_node = store.add_node(
            task_id=task_id,
            branch_id=branch_id,
            logical_id=f"candidate_{safe_group}_{safe_candidate}",
            node_type="candidate",
            content={
                "candidate_id": candidate_key,
                "original_candidate_id": candidate_id,
                "group_id": group_id,
                "domain": domain,
                "generation_index": index,
                "seed": candidate.get("seed"),
                "repair_round": candidate.get("repair_round", 0),
                "input_tokens": int(candidate.get("input_tokens") or 0),
                "output_tokens": int(candidate.get("output_tokens") or 0),
            },
            owner="candidate_graph",
            created_by_role="runtime",
        )
        store.add_edge(
            source=group.node_id,
            target=candidate_node.node_id,
            relation="produces",
            created_by_role="runtime",
        )

        artifact_node = store.add_node(
            task_id=task_id,
            branch_id=branch_id,
            logical_id=f"candidate_artifact_{safe_group}_{safe_candidate}",
            node_type="candidate_artifact",
            content={
                "candidate_id": candidate_key,
                "group_id": group_id,
                "domain": domain,
                "artifact_type": artifact_type,
                "artifact": candidate.get("artifact"),
                "final_value": candidate.get("final_value"),
                "raw_tail": candidate.get("raw_tail"),
            },
            owner="candidate_graph",
            created_by_role="runtime",
        )
        store.add_edge(
            source=candidate_node.node_id,
            target=artifact_node.node_id,
            relation="produces",
            created_by_role="runtime",
        )

        verification_status = "verified" if correct else "ready"
        verification_node = store.add_node(
            task_id=task_id,
            branch_id=branch_id,
            logical_id=f"candidate_verification_{safe_group}_{safe_candidate}",
            node_type="candidate_verification",
            content={
                "candidate_id": candidate_key,
                "group_id": group_id,
                "domain": domain,
                "status": "pass" if correct else "fail",
                "correct": correct,
                "verification_source": candidate.get("verification_source", "oracle_evaluator"),
                "score": score,
                "signals": candidate.get("verification_signals") or {},
            },
            owner="candidate_graph",
            status=verification_status,
            validation={"evidence_checked": correct},
            created_by_role="verifier",
        )
        store.add_edge(
            source=artifact_node.node_id,
            target=verification_node.node_id,
            relation="input_to",
            created_by_role="runtime",
        )
        store.add_edge(
            source=verification_node.node_id,
            target=artifact_node.node_id,
            relation="verifies" if correct else "invalidates",
            created_by_role="verifier",
        )

        candidate_node_ids[candidate_key] = candidate_node.node_id
        artifact_node_ids[candidate_key] = artifact_node.node_id
        verification_node_ids[candidate_key] = verification_node.node_id

    selection_node_id: str | None = None
    if selected_candidate_id is not None:
        decision = CandidateSelectionDecision(
            selected_candidate_id=_resolve_selected_candidate_id(str(selected_candidate_id), candidate_node_ids),
            selected_candidate_node_id="",
            selected_artifact_node_id="",
            selected_verification_node_id="",
            selection_policy=selector_policy or "manual",
            reason="manual_selection",
        )
        selected_key = decision.selected_candidate_id
        selected_artifact_node_id = artifact_node_ids[selected_key]
        selected_verification_node_id = verification_node_ids[selected_key]
        result_with_selection = _add_selection_decision(
            store,
            task_id=task_id,
            branch_id=branch_id,
            group_node_id=group.node_id,
            group_id=group_id,
            domain=domain,
            selector_policy=selector_policy or "manual",
            selected_candidate_id=selected_key,
            candidate_node_ids=candidate_node_ids,
            artifact_node_ids=artifact_node_ids,
            verification_node_ids=verification_node_ids,
            reason="manual_selection",
            metadata={"candidate_count": len(candidates)},
        )
        selection_node_id = result_with_selection.selection_node_id
        selected_candidate_id = selected_key

    result = CandidateGraphResult(
        group_node_id=group.node_id,
        candidate_node_ids=candidate_node_ids,
        artifact_node_ids=artifact_node_ids,
        verification_node_ids=verification_node_ids,
        selection_node_id=selection_node_id,
        selected_candidate_id=selected_candidate_id,
        selected_artifact_node_id=selected_artifact_node_id,
        selected_verification_node_id=selected_verification_node_id,
        invariant_errors=_candidate_pool_invariant_errors(
            candidates=candidates,
            candidate_node_ids=candidate_node_ids,
            artifact_node_ids=artifact_node_ids,
            verification_node_ids=verification_node_ids,
            selected_candidate_id=selected_candidate_id,
        ),
    )
    return result


def select_candidate_by_verification(
    store: GraphStore,
    pool: CandidateGraphResult,
    *,
    selection_policy: str = "verifier_guided_first_verified",
) -> CandidateSelectionDecision:
    """Select a candidate by reading only candidate_verification graph nodes."""

    if not pool.candidate_node_ids:
        raise ValueError("cannot select from an empty candidate pool")

    verified: list[str] = []
    rejected: list[str] = []
    ordered = sorted(
        pool.candidate_node_ids,
        key=lambda candidate_id: _candidate_generation_index(store, pool.candidate_node_ids[candidate_id]),
    )
    for candidate_id in ordered:
        verification = store.node(pool.verification_node_ids[candidate_id])
        content = verification.content if isinstance(verification.content, Mapping) else {}
        if _verification_passes(verification, content):
            verified.append(candidate_id)
        else:
            rejected.append(candidate_id)

    selected = verified[0] if verified else ordered[0]
    return CandidateSelectionDecision(
        selected_candidate_id=selected,
        selected_candidate_node_id=pool.candidate_node_ids[selected],
        selected_artifact_node_id=pool.artifact_node_ids[selected],
        selected_verification_node_id=pool.verification_node_ids[selected],
        selection_policy=selection_policy,
        reason="verified_candidate" if verified else "no_verified_candidate_fallback_first",
        verified_candidate_ids=tuple(verified),
        rejected_candidate_ids=tuple(rejected),
    )


def select_candidate_by_deployable_verification(
    store: GraphStore,
    pool: CandidateGraphResult,
    *,
    selection_policy: str = "deployable_verifier_guided_first_verified",
) -> CandidateSelectionDecision:
    """Production selector that rejects oracle/gold-only verifier signals."""

    if not pool.candidate_node_ids:
        raise ValueError("cannot select from an empty candidate pool")

    verified: list[str] = []
    rejected: list[str] = []
    ordered = sorted(
        pool.candidate_node_ids,
        key=lambda candidate_id: _candidate_generation_index(store, pool.candidate_node_ids[candidate_id]),
    )
    for candidate_id in ordered:
        verification = store.node(pool.verification_node_ids[candidate_id])
        content = verification.content if isinstance(verification.content, Mapping) else {}
        assert_deployable_verifier_payload(content)
        score = content.get("score") if isinstance(content.get("score"), Mapping) else {}
        signals = content.get("signals") if isinstance(content.get("signals"), Mapping) else {}
        assert_deployable_verifier_payload(score)
        assert_deployable_verifier_payload(signals)
        source = str(content.get("verification_source") or "")
        if "oracle" in source or "gold" in source or "hidden" in source:
            raise ValueError(f"deployable selector cannot use verifier source: {source}")
        if _verification_passes(verification, content):
            verified.append(candidate_id)
        else:
            rejected.append(candidate_id)

    selected = verified[0] if verified else ordered[0]
    return CandidateSelectionDecision(
        selected_candidate_id=selected,
        selected_candidate_node_id=pool.candidate_node_ids[selected],
        selected_artifact_node_id=pool.artifact_node_ids[selected],
        selected_verification_node_id=pool.verification_node_ids[selected],
        selection_policy=selection_policy,
        reason="deployable_verified_candidate" if verified else "no_deployable_verified_candidate_fallback_first",
        verified_candidate_ids=tuple(verified),
        rejected_candidate_ids=tuple(rejected),
    )


def add_verifier_guided_selection(
    store: GraphStore,
    pool: CandidateGraphResult,
    *,
    selection_policy: str = "verifier_guided_first_verified",
    metadata: Mapping[str, Any] | None = None,
) -> CandidateGraphResult:
    group = store.node(pool.group_node_id)
    content = group.content if isinstance(group.content, Mapping) else {}
    decision = select_candidate_by_verification(
        store,
        pool,
        selection_policy=selection_policy,
    )
    return _add_selection_decision(
        store,
        task_id=group.task_id,
        branch_id=group.branch_id,
        group_node_id=group.node_id,
        group_id=str(content.get("group_id") or group.logical_id),
        domain=str(content.get("domain") or ""),
        selector_policy=decision.selection_policy,
        selected_candidate_id=decision.selected_candidate_id,
        candidate_node_ids=pool.candidate_node_ids,
        artifact_node_ids=pool.artifact_node_ids,
        verification_node_ids=pool.verification_node_ids,
        reason=decision.reason,
        metadata={
            "source": "candidate_verification_nodes",
            **dict(metadata or {}),
            "verified_candidate_ids": list(decision.verified_candidate_ids),
            "rejected_candidate_ids": list(decision.rejected_candidate_ids),
        },
        pool=pool,
    )


def add_candidate_selection_decision(
    store: GraphStore,
    pool: CandidateGraphResult,
    *,
    selected_candidate_id: str,
    selection_policy: str,
    reason: str,
    metadata: Mapping[str, Any] | None = None,
) -> CandidateGraphResult:
    group = store.node(pool.group_node_id)
    content = group.content if isinstance(group.content, Mapping) else {}
    return _add_selection_decision(
        store,
        task_id=group.task_id,
        branch_id=group.branch_id,
        group_node_id=group.node_id,
        group_id=str(content.get("group_id") or group.logical_id),
        domain=str(content.get("domain") or ""),
        selector_policy=selection_policy,
        selected_candidate_id=selected_candidate_id,
        candidate_node_ids=pool.candidate_node_ids,
        artifact_node_ids=pool.artifact_node_ids,
        verification_node_ids=pool.verification_node_ids,
        reason=reason,
        metadata=metadata,
        pool=pool,
    )


def dereference_selected_candidate_value(store: GraphStore, selection_node_id: str) -> Any:
    """Return the selected artifact value through graph identity only.

    The selector chooses a canonical artifact node; this helper dereferences that
    node and copies its canonical value.  It intentionally does not accept or
    regenerate a model-facing answer string.
    """

    selection = store.node(selection_node_id)
    if selection.type != "selection_decision":
        raise ValueError(f"not a selection_decision node: {selection_node_id}")
    artifact_node_id = selection.content.get("selected_artifact_node_id")
    if not artifact_node_id:
        raise ValueError(f"selection has no selected_artifact_node_id: {selection_node_id}")
    artifact = store.node(str(artifact_node_id))
    if artifact.type != "candidate_artifact":
        raise ValueError(f"selected node is not a candidate_artifact: {artifact_node_id}")
    content = artifact.content if isinstance(artifact.content, Mapping) else {}
    if content.get("final_value") is not None:
        return content["final_value"]
    return content.get("artifact")


def _add_selection_decision(
    store: GraphStore,
    *,
    task_id: str,
    branch_id: str,
    group_node_id: str,
    group_id: str,
    domain: str,
    selector_policy: str,
    selected_candidate_id: str,
    candidate_node_ids: Mapping[str, str],
    artifact_node_ids: Mapping[str, str],
    verification_node_ids: Mapping[str, str],
    reason: str,
    metadata: Mapping[str, Any] | None = None,
    pool: CandidateGraphResult | None = None,
) -> CandidateGraphResult:
    selected_key = _resolve_selected_candidate_id(str(selected_candidate_id), candidate_node_ids)
    selected_artifact_node_id = artifact_node_ids[selected_key]
    selected_verification_node_id = verification_node_ids[selected_key]
    selection = store.add_node(
        task_id=task_id,
        branch_id=branch_id,
        logical_id=f"selection_decision_{safe_id(group_id)}_{safe_id(selector_policy)}",
        node_type="selection_decision",
        content={
            "group_id": group_id,
            "domain": domain,
            "selection_policy": selector_policy,
            "selection_reason": reason,
            "selected_candidate_id": selected_key,
            "selected_candidate_node_id": candidate_node_ids[selected_key],
            "selected_artifact_node_id": selected_artifact_node_id,
            "selected_verification_node_id": selected_verification_node_id,
            "candidate_count": len(candidate_node_ids),
            "metadata": dict(metadata or {}),
        },
        owner="candidate_graph",
        created_by_role="selector",
    )
    store.add_edge(
        source=group_node_id,
        target=selection.node_id,
        relation="produces",
        created_by_role="selector",
    )
    for source in (
        candidate_node_ids[selected_key],
        selected_artifact_node_id,
        selected_verification_node_id,
    ):
        store.add_edge(
            source=source,
            target=selection.node_id,
            relation="input_to",
            created_by_role="selector",
        )

    if pool is not None:
        invariant_errors = _candidate_pool_invariant_errors(
            candidates=[{} for _ in pool.candidate_node_ids],
            candidate_node_ids=pool.candidate_node_ids,
            artifact_node_ids=pool.artifact_node_ids,
            verification_node_ids=pool.verification_node_ids,
            selected_candidate_id=selected_key,
        )
        return replace(
            pool,
            selection_node_id=selection.node_id,
            selected_candidate_id=selected_key,
            selected_artifact_node_id=selected_artifact_node_id,
            selected_verification_node_id=selected_verification_node_id,
            invariant_errors=invariant_errors,
        )
    invariant_errors = _candidate_pool_invariant_errors(
        candidates=[{} for _ in candidate_node_ids],
        candidate_node_ids=candidate_node_ids,
        artifact_node_ids=artifact_node_ids,
        verification_node_ids=verification_node_ids,
        selected_candidate_id=selected_key,
    )
    return CandidateGraphResult(
        group_node_id=group_node_id,
        candidate_node_ids=dict(candidate_node_ids),
        artifact_node_ids=dict(artifact_node_ids),
        verification_node_ids=dict(verification_node_ids),
        selection_node_id=selection.node_id,
        selected_candidate_id=selected_key,
        selected_artifact_node_id=selected_artifact_node_id,
        selected_verification_node_id=selected_verification_node_id,
        invariant_errors=invariant_errors,
    )


def _candidate_generation_index(store: GraphStore, candidate_node_id: str) -> int:
    node = store.node(candidate_node_id)
    content = node.content if isinstance(node.content, Mapping) else {}
    try:
        return int(content.get("generation_index", 0))
    except (TypeError, ValueError):
        return 0


def _verification_passes(verification: Any, content: Mapping[str, Any]) -> bool:
    if verification.status == "verified":
        return True
    if verification.validation.get("evidence_checked"):
        return True
    if content.get("status") == "pass":
        return True
    return bool(content.get("correct"))


def _unique_key(candidate_id: str, existing: Mapping[str, str]) -> str:
    if candidate_id not in existing:
        return candidate_id
    suffix = 2
    while f"{candidate_id}_{suffix}" in existing:
        suffix += 1
    return f"{candidate_id}_{suffix}"


def _resolve_selected_candidate_id(selected: str, existing: Mapping[str, str]) -> str:
    if selected in existing:
        return selected
    normalized = safe_id(selected)
    matches = [candidate_id for candidate_id in existing if safe_id(candidate_id) == normalized]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"selected candidate does not exist: {selected}")


def _candidate_pool_invariant_errors(
    *,
    candidates: Sequence[Mapping[str, Any]],
    candidate_node_ids: Mapping[str, str],
    artifact_node_ids: Mapping[str, str],
    verification_node_ids: Mapping[str, str],
    selected_candidate_id: str | None,
) -> list[str]:
    errors: list[str] = []
    if len(candidate_node_ids) != len(candidates):
        errors.append("candidate_count_mismatch")
    if len(artifact_node_ids) != len(candidate_node_ids):
        errors.append("artifact_count_mismatch")
    if len(verification_node_ids) != len(candidate_node_ids):
        errors.append("verification_count_mismatch")
    if len(set(candidate_node_ids.values())) != len(candidate_node_ids):
        errors.append("duplicate_candidate_node_id")
    if selected_candidate_id is not None and selected_candidate_id not in candidate_node_ids:
        errors.append("selection_missing_candidate")
    missing_artifacts = set(candidate_node_ids) - set(artifact_node_ids)
    if missing_artifacts:
        errors.append("candidate_without_artifact")
    missing_verifications = set(candidate_node_ids) - set(verification_node_ids)
    if missing_verifications:
        errors.append("candidate_without_verification")
    return errors
