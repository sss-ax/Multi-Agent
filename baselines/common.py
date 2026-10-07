"""Shared schema and accounting helpers for external baseline runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Protocol


class ExternalBaselineRunner(Protocol):
    """Minimal interface implemented by black-box baseline adapters."""

    name: str

    def run(self, row: dict[str, Any]) -> "BaselineResult":
        """Run one normalized benchmark row and return a common result."""


@dataclass(frozen=True)
class BaselineMessage:
    """One rendered message in an external baseline trace."""

    sender: str
    receiver: str
    content: str
    role: str = "assistant"
    message_type: str = "agent"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BaselineTokenUsage:
    """Token usage under the repository tokenizer, not framework-native counts."""

    input_tokens: int = 0
    output_tokens: int = 0
    communication_tokens: int = 0
    communication_messages: int = 0
    forward_calls: int = 0
    system_prompt_tokens: int = 0
    task_prompt_tokens: int = 0

    @property
    def total_model_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def phase8_total_cost_tokens(self) -> int:
        return self.communication_tokens + self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["total_model_tokens"] = self.total_model_tokens
        payload["phase8_total_cost_tokens"] = self.phase8_total_cost_tokens
        return payload


@dataclass(frozen=True)
class BaselineResult:
    """Common single-sample output for all external baselines."""

    sample_id: str
    method: str
    domain: str
    final_answer: Any = None
    trace: list[BaselineMessage] = field(default_factory=list)
    token_usage: BaselineTokenUsage = field(default_factory=BaselineTokenUsage)
    raw_output: Any = None
    failure: str = ""
    latency_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_record(self, *, scores: dict[str, Any] | None = None, gold_answer: Any = None) -> dict[str, Any]:
        token_payload = self.token_usage.as_dict()
        return {
            "sample_id": self.sample_id,
            "method": self.method,
            "domain": self.domain,
            "correct": bool(scores.get("correct")) if scores else False,
            "scores": scores or {},
            "gold_answer": gold_answer,
            "failure": self.failure,
            "latency_sec": self.latency_sec,
            "final_answer": self.final_answer,
            "raw_output": self.raw_output,
            "trace": [message.as_dict() for message in self.trace],
            "runtime_summary": {
                "records": [],
                "external_baseline": {
                    "method": self.method,
                    "trace_messages": len(self.trace),
                    **token_payload,
                },
            },
            **token_payload,
            **self.metadata,
        }


def count_tokens(tokenizer: Any, text: Any) -> int:
    """Count tokens with the experiment tokenizer while tolerating light fakes in tests."""

    if tokenizer is None:
        return len(str(text).split())
    encoded = tokenizer(str(text), add_special_tokens=False)
    if isinstance(encoded, dict):
        ids = encoded.get("input_ids", [])
    else:
        ids = getattr(encoded, "input_ids", encoded)
    if ids and hasattr(ids, "shape"):
        return int(ids.shape[-1])
    if ids and isinstance(ids[0], list):
        return len(ids[0])
    return len(ids)


def communication_token_usage(tokenizer: Any, messages: Iterable[BaselineMessage]) -> tuple[int, int]:
    """Return total tokens and count for messages exchanged between agents."""

    total = 0
    count = 0
    for message in messages:
        if message.message_type != "agent":
            continue
        count += 1
        rendered = f"{message.sender} -> {message.receiver}\n{message.content}"
        total += count_tokens(tokenizer, rendered)
    return total, count


def aggregate_external_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the report-level fields used by external baseline evaluations."""

    count = len(records)
    correct = sum(int(bool(item.get("correct"))) for item in records)
    totals = {
        "total_input_tokens": sum(int(item.get("input_tokens", 0) or 0) for item in records),
        "total_output_tokens": sum(int(item.get("output_tokens", 0) or 0) for item in records),
        "total_model_tokens": sum(int(item.get("total_model_tokens", 0) or 0) for item in records),
        "communication_cost_tokens": sum(int(item.get("communication_tokens", 0) or 0) for item in records),
        "communication_round_count": sum(int(item.get("communication_messages", 0) or 0) for item in records),
        "forward_calls": sum(int(item.get("forward_calls", 0) or 0) for item in records),
        "system_prompt_tokens": sum(int(item.get("system_prompt_tokens", 0) or 0) for item in records),
        "task_prompt_tokens": sum(int(item.get("task_prompt_tokens", 0) or 0) for item in records),
    }
    totals["inference_prefill_cost_tokens"] = totals["total_input_tokens"]
    totals["inference_decode_cost_tokens"] = totals["total_output_tokens"]
    totals["phase8_total_cost_tokens"] = (
        totals["communication_cost_tokens"]
        + totals["inference_prefill_cost_tokens"]
        + totals["inference_decode_cost_tokens"]
    )
    metric_values: dict[str, float] = {}
    for key in ("exact_match", "f1", "pass_at_1"):
        values = [float(item["scores"][key]) for item in records if key in item.get("scores", {})]
        if values:
            metric_values[key] = sum(values) / len(values)
    if count:
        totals["tokens_per_sample"] = totals["phase8_total_cost_tokens"] / count
        totals["communication_tokens_per_sample"] = totals["communication_cost_tokens"] / count
    else:
        totals["tokens_per_sample"] = 0.0
        totals["communication_tokens_per_sample"] = 0.0
    return {
        "count": count,
        "accuracy_or_pass_at_1": correct / count if count else 0.0,
        "failure_count": sum(bool(item.get("failure")) for item in records),
        **metric_values,
        **totals,
    }

