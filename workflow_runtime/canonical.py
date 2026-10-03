"""Stable serialization and digest helpers.

Lists remain ordered deliberately: arithmetic operands, evidence order, and
workflow slots can be semantically ordered.  Only mappings and explicit sets are
normalized for deterministic hashing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Dict, Iterable, Mapping, Sequence

from .models import NodeVersion, RuntimeFingerprint


def _canonical_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _canonical_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_value(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def node_content_digest(node_type: str, content: Any) -> str:
    return _sha256(canonical_json({"type": node_type, "content": content}))


def dependency_digest(
    dependencies: Mapping[str, str] | Iterable[Mapping[str, Any]],
) -> str:
    """Hash dependencies by explicit keys without sorting dictionaries directly."""
    if isinstance(dependencies, Mapping):
        records = [
            {"slot": str(slot), "version_id": str(version_id)}
            for slot, version_id in dependencies.items()
        ]
    else:
        records = [dict(record) for record in dependencies]
    records.sort(
        key=lambda record: (
            str(record.get("slot", "")),
            str(record.get("relation", "")),
            str(record.get("version_id", "")),
        )
    )
    return _sha256(canonical_json(records))


def node_order_digest(node_ids: Iterable[str]) -> str:
    """Return a stable digest for the ordered nodes in a graph slice.

    The digest intentionally captures order, not node content.  Content and
    validation state are already captured by ``ContextSlice.dependency_digest``;
    keeping the two dimensions separate makes tensorization cache misses
    diagnosable.
    """
    return _sha256(canonical_json(list(node_ids)))


def runtime_fingerprint_digest(runtime: RuntimeFingerprint | Mapping[str, Any]) -> str:
    payload = runtime.as_dict() if isinstance(runtime, RuntimeFingerprint) else dict(runtime)
    return _sha256(canonical_json(payload))


def full_digest(
    *,
    node_type: str,
    content_digest: str,
    dependency_digest_value: str,
    runtime: RuntimeFingerprint | Mapping[str, Any],
) -> str:
    return _sha256(
        canonical_json(
            {
                "type": node_type,
                "content_digest": content_digest,
                "dependency_digest": dependency_digest_value,
                "runtime": runtime.as_dict() if isinstance(runtime, RuntimeFingerprint) else dict(runtime),
            }
        )
    )


def populate_node_digests(
    node: NodeVersion,
    runtime: RuntimeFingerprint | Mapping[str, Any] | None = None,
) -> NodeVersion:
    node.content_digest = node_content_digest(node.type, node.content)
    node.dependency_digest = dependency_digest(node.dependency_versions)
    if runtime is not None:
        node.full_digest = full_digest(
            node_type=node.type,
            content_digest=node.content_digest,
            dependency_digest_value=node.dependency_digest,
            runtime=runtime,
        )
    return node
