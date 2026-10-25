"""Kernel-owned immutable audit manifests for multiple-choice candidates."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def build_choice_audit_manifest(store: Any, *, task_id: str, branch_id: str = "main") -> dict[str, Any] | None:
    """Build a deterministic manifest for the latest candidate result.

    The manifest contains no evaluator/gold data. It binds every audit claim
    to the exact candidate version and immutable source values visible to the
    Critic.
    """
    result = store.latest_valid(task_id, branch_id, "result")
    if result is None:
        return None
    value = _json_value(result.content)
    if isinstance(value, Mapping) and "value" in value:
        value = _json_value(value.get("value"))
    if not isinstance(value, Mapping):
        return None
    answer = str(value.get("answer", value.get("solver_choice", ""))).strip().upper()
    question_text = ""
    for node in store.snapshot().nodes.values():
        if node.task_id != task_id or node.branch_id != branch_id:
            continue
        if node.type not in {"task", "question_source", "task_source"}:
            continue
        raw = _json_value(node.content)
        if isinstance(raw, Mapping):
            question_text = str(raw.get("question", raw.get("text", ""))).strip()
        else:
            question_text = str(raw).strip()
        if question_text:
            break
    slots = value.get("slot_analysis", ())
    slot_claims: list[dict[str, Any]] = []
    for item in slots if isinstance(slots, (list, tuple)) else ():
        if not isinstance(item, Mapping) or not isinstance(item.get("slot"), int):
            continue
        slot = int(item["slot"])
        slot_claims.append({
            "claim_id": f"{result.node_id}:slot:{slot}",
            "slot": slot,
            "selected_term": str(item.get("selected_term", "")),
            "claim_text": f"Solver claims {str(item.get('selected_term', '')).strip()!r} correctly fills slot {slot}.",
            "context": f"slot {slot} in the original question",
            "solver_reason": str(item.get("reason", "")),
        })
    reviews = value.get("option_reviews", ())
    eliminated = value.get("eliminated_options", {})
    elimination_claims: list[dict[str, Any]] = []
    choice_text: dict[str, str] = {}
    for node in store.snapshot().nodes.values():
        if node.task_id != task_id or node.branch_id != branch_id or node.type != "choice_source":
            continue
        raw = _json_value(node.content)
        if isinstance(raw, Mapping):
            label = str(raw.get("label", "")).strip().upper()
            if label:
                choice_text[label] = str(raw.get("text", ""))
    review_by_label = {
        str(item.get("label", "")).strip().upper(): item
        for item in reviews if isinstance(item, Mapping) and str(item.get("label", "")).strip()
    }
    source_labels = sorted(choice_text)
    for option in source_labels:
        item = review_by_label.get(option, {})
        label = option
        if not label:
            continue
        elimination_claims.append({
            "claim_id": f"{result.node_id}:elimination:{label}",
            "option": label,
            "option_text": choice_text.get(label, ""),
            "solver_assessment": str(item.get("assessment", "")),
            "solver_reason": str(item.get("reason", eliminated.get(label, "") if isinstance(eliminated, Mapping) else "")),
            "claim_text": f"Solver claims option {label} should be {'selected' if str(item.get('assessment', '')).strip().lower() == 'selected' else 'rejected'}.",
        })
    manifest = {
        "candidate_version": result.node_id,
        "selected_answer": answer,
        "original_question": question_text,
        "selected_option_text": choice_text.get(answer, ""),
        "slot_claims": slot_claims,
        "elimination_claims": elimination_claims,
    }
    digest_payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest["manifest_digest"] = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()
    return manifest


def manifest_claim_ids(manifest: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(manifest, Mapping):
        return ()
    ids: list[str] = []
    for key in ("slot_claims", "elimination_claims"):
        values = manifest.get(key, ())
        if isinstance(values, (list, tuple)):
            ids.extend(str(item.get("claim_id")) for item in values if isinstance(item, Mapping) and item.get("claim_id"))
    return tuple(ids)
