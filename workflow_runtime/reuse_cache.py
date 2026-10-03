"""Exact, validated result reuse.  Compilation/render caches are intentionally absent."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Dict, Optional

from .canonical import canonical_json, runtime_fingerprint_digest
from .models import CacheLookup, CachedResult, NodeVersion, RuntimeFingerprint


class ResultCache(ABC):
    @abstractmethod
    def put(
        self,
        node: NodeVersion,
        *,
        role: str,
        runtime: RuntimeFingerprint,
        cache_policy: str = "exact_validated",
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def get(
        self,
        *,
        task_id: str,
        branch_id: str,
        logical_id: str,
        dependency_digest: str,
        role: str,
        runtime: RuntimeFingerprint,
        cache_policy: str = "exact_validated",
    ) -> CacheLookup:
        raise NotImplementedError

    @abstractmethod
    def invalidate_nodes(self, node_ids: list[str]) -> None:
        raise NotImplementedError


class InMemoryResultCache(ResultCache):
    def __init__(self) -> None:
        self._items: Dict[str, CachedResult] = {}
        self._invalidated_node_ids: set[str] = set()

    @staticmethod
    def make_key(
        *,
        task_id: str,
        branch_id: str,
        logical_id: str,
        dependency_digest: str,
        role: str,
        runtime: RuntimeFingerprint,
        cache_policy: str,
    ) -> str:
        payload = canonical_json(
            {
                "task_id": task_id,
                "branch_id": branch_id,
                "logical_id": logical_id,
                "dependency_digest": dependency_digest,
                "role": role,
                "runtime": runtime.as_dict(),
                "cache_policy": cache_policy,
            }
        )
        import hashlib

        return "result_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _cacheable(node: NodeVersion, cache_policy: str) -> tuple[bool, str]:
        if not node.is_operationally_valid():
            return False, f"node status is not reusable: {node.status}"
        if not node.has_validation("schema_valid"):
            return False, "schema_valid is false"
        if cache_policy == "verified_only" and not node.is_semantically_verified():
            return False, "semantic verification is required"
        if cache_policy == "exact_validated" and not (
            node.has_validation("execution_valid")
            or node.is_semantically_verified()
        ):
            return False, "execution_valid or semantic verification is required"
        return True, "cacheable"

    def put(
        self,
        node: NodeVersion,
        *,
        role: str,
        runtime: RuntimeFingerprint,
        cache_policy: str = "exact_validated",
    ) -> str:
        allowed, reason = self._cacheable(node, cache_policy)
        if not allowed:
            raise ValueError(f"cannot cache node {node.node_id}: {reason}")
        key = self.make_key(
            task_id=node.task_id,
            branch_id=node.branch_id,
            logical_id=node.logical_id,
            dependency_digest=node.dependency_digest,
            role=role,
            runtime=runtime,
            cache_policy=cache_policy,
        )
        self._items[key] = CachedResult(
            cache_key=key,
            task_id=node.task_id,
            branch_id=node.branch_id,
            logical_id=node.logical_id,
            role=role,
            node=copy.deepcopy(node),
            runtime_fingerprint=runtime,
            created_at=node.created_at,
            cache_policy=cache_policy,
        )
        return key

    def get(
        self,
        *,
        task_id: str,
        branch_id: str,
        logical_id: str,
        dependency_digest: str,
        role: str,
        runtime: RuntimeFingerprint,
        cache_policy: str = "exact_validated",
    ) -> CacheLookup:
        key = self.make_key(
            task_id=task_id,
            branch_id=branch_id,
            logical_id=logical_id,
            dependency_digest=dependency_digest,
            role=role,
            runtime=runtime,
            cache_policy=cache_policy,
        )
        artifact = self._items.get(key)
        if artifact is None:
            return CacheLookup(hit=False, reason="cache miss")
        if artifact.node.node_id in self._invalidated_node_ids:
            return CacheLookup(hit=False, reason="cached node was explicitly invalidated")
        allowed, reason = self._cacheable(artifact.node, cache_policy)
        if not allowed:
            return CacheLookup(hit=False, reason=reason)
        if artifact.node.dependency_digest != dependency_digest:
            return CacheLookup(hit=False, reason="dependency digest mismatch")
        return CacheLookup(hit=True, artifact=copy.deepcopy(artifact), reason="exact validated hit")

    def invalidate_nodes(self, node_ids: list[str]) -> None:
        self._invalidated_node_ids.update(node_ids)

    def __len__(self) -> int:
        return len(self._items)
