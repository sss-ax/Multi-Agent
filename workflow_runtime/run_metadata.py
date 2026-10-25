"""Reproducibility metadata for workflow and benchmark reports."""

from __future__ import annotations

import hashlib
import os
import subprocess
from typing import Any, Iterable, Mapping


def _git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=os.getcwd(),
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (result.stdout or "").strip()


def sample_ids_sha256(sample_ids: Iterable[Any]) -> str:
    normalized = "\n".join(sorted(str(item) for item in sample_ids))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_run_metadata(
    *,
    protocol_version: str,
    model_id: str = "",
    tokenizer_id: str = "",
    dataset_revision: str = "",
    sample_ids: Iterable[Any] = (),
    seed: int = 0,
    max_rounds: int | None = None,
    max_new_tokens: int | None = None,
    policy: str = "",
    graph_context_mode: str = "",
    action_protocol: str = "semantic_delta_v2",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata without reading or serializing evaluator labels."""
    metadata = {
        "protocol_version": protocol_version,
        "action_protocol": action_protocol,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "dirty_worktree": bool(_git_value("status", "--porcelain")),
        "model_id": str(model_id or ""),
        "tokenizer_id": str(tokenizer_id or ""),
        "dataset_revision": str(dataset_revision or ""),
        "sample_ids_sha256": sample_ids_sha256(sample_ids),
        "seed": int(seed),
        "max_rounds": None if max_rounds is None else int(max_rounds),
        "max_new_tokens": None if max_new_tokens is None else int(max_new_tokens),
        "policy": str(policy or ""),
        "graph_context_mode": str(graph_context_mode or ""),
    }
    if extra:
        for key, value in extra.items():
            if str(key).casefold() not in {
                "gold_answer", "gold_label", "answer", "supporting_facts", "hidden_tests", "official_tests",
            }:
                metadata[str(key)] = value
    return metadata
