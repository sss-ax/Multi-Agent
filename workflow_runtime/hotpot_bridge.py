"""HotpotQA bridge candidate canonicalization helpers."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable


ENTITY_KEYS = ("name", "title", "entity", "label", "value")
TEXT_KEYS = ("text", "sentence")


def parse_content(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def extract_bridge_candidate(value: Any, *, allow_text: bool = False) -> str:
    parsed = parse_content(value)
    if isinstance(parsed, dict):
        for key in ENTITY_KEYS:
            candidate = _clean_candidate(parsed.get(key))
            if candidate:
                return candidate
        if allow_text:
            for key in TEXT_KEYS:
                candidate = _clean_candidate(parsed.get(key))
                if candidate:
                    return candidate
        return ""
    return _clean_candidate(parsed)


def bridge_candidates_from_values(values: Iterable[Any], *, limit: int = 24, allow_text: bool = False) -> list[str]:
    return deduplicate_bridge_candidates(
        (extract_bridge_candidate(value, allow_text=allow_text) for value in values),
        limit=limit,
    )


def deduplicate_bridge_candidates(values: Iterable[Any], *, limit: int = 24) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for value in values:
        candidate = _clean_candidate(value)
        if not candidate:
            continue
        key = normalize_bridge(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return candidates


def normalize_bridge(value: Any) -> str:
    candidate = extract_bridge_candidate(value, allow_text=False)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", candidate.lower())).strip()


def bridge_matches_any(value: Any, gold: Iterable[Any]) -> bool:
    normalized = normalize_bridge(value)
    return bool(normalized and any(normalized == normalize_bridge(item) for item in gold))


def _clean_candidate(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    if not text or len(text) > 120:
        return ""
    return text
