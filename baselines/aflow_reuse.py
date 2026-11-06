"""Exact, task-scoped reuse primitives for the AFlow graph adapter."""

from __future__ import annotations

import asyncio
import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from workflow_runtime.canonical import canonical_json


@dataclass(frozen=True)
class AFlowInvocationKey:
    """Stable key for a deterministic AFlow operator invocation."""

    task_id: str
    operator: str
    normalized_input_digest: str
    model_fingerprint: str
    generation_digest: str

    @classmethod
    def build(
        cls,
        *,
        task_id: str,
        operator: str,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        model_fingerprint: str,
        generation_config: Mapping[str, Any],
    ) -> "AFlowInvocationKey":
        input_digest = _digest({"args": args, "kwargs": dict(kwargs)})
        generation_digest = _digest(dict(generation_config))
        return cls(
            task_id=str(task_id),
            operator=str(operator),
            normalized_input_digest=input_digest,
            model_fingerprint=str(model_fingerprint),
            generation_digest=generation_digest,
        )


@dataclass
class AFlowArtifactRecord:
    """Canonical output plus occurrence accounting."""

    node_id: str
    output_kind: str
    content_digest: str
    raw_output: Any
    occurrence_count: int = 1


@dataclass(frozen=True)
class AFlowReuseResult:
    value: Any
    hit: bool


class AFlowInvocationReuse:
    """Task-local exact cache with async single-flight behavior."""

    def __init__(self) -> None:
        self._cache: dict[AFlowInvocationKey, Any] = {}
        self._inflight: dict[AFlowInvocationKey, asyncio.Future[Any]] = {}
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    async def get_or_compute(
        self,
        key: AFlowInvocationKey,
        compute: Callable[[], Awaitable[Any]],
    ) -> AFlowReuseResult:
        async with self._lock:
            if key in self._cache:
                self.hits += 1
                return AFlowReuseResult(copy.deepcopy(self._cache[key]), True)
            future = self._inflight.get(key)
            leader = future is None
            if leader:
                future = asyncio.get_running_loop().create_future()
                self._inflight[key] = future

        if not leader:
            self.hits += 1
            return AFlowReuseResult(copy.deepcopy(await asyncio.shield(future)), True)

        self.misses += 1
        try:
            value = await compute()
            async with self._lock:
                self._cache[key] = copy.deepcopy(value)
                current = self._inflight.pop(key, None)
                if current is not None and not current.done():
                    current.set_result(copy.deepcopy(value))
            return AFlowReuseResult(value, False)
        except BaseException as exc:
            async with self._lock:
                current = self._inflight.pop(key, None)
                if current is not None and not current.done():
                    current.set_exception(exc)
            raise


def normalize_output(output: Any) -> Any:
    """Conservative normalization used by exact content deduplication."""
    if isinstance(output, str):
        return output.replace("\r\n", "\n").replace("\r", "\n").strip()
    if isinstance(output, Mapping):
        return {
            str(key): normalize_output(value)
            for key, value in sorted(output.items(), key=lambda item: str(item[0]))
            if key != "raw_output"
        }
    if isinstance(output, (list, tuple)):
        return [normalize_output(item) for item in output]
    return output


def output_digest(output_kind: str, output: Any) -> str:
    return _digest({"output_kind": str(output_kind), "content": normalize_output(output)})


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
