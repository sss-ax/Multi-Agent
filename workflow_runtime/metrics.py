"""Runtime cost metrics independent of a particular model backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CallRecord:
    call_id: int
    caller: str
    stage: str
    input_tokens: int = 0
    output_tokens: int = 0
    context_tokens: int = 0
    full_context_tokens: int = 0
    slice_tokens: int = 0
    latency_sec: float = 0.0
    graph_overhead_ms: float = 0.0
    cache_mode: Optional[str] = None
    cache_hit: bool = False
    cache_key: Optional[str] = None
    avoided_llm_calls: int = 0
    estimated_reused_input_tokens: int = 0
    # Communication accounting is intentionally separate from model I/O.
    # Graph workflows should report a2a_text_tokens=0 and count typed writes
    # and graph reads independently.
    a2a_text_tokens: int = 0
    graph_update_tokens: int = 0
    graph_read_tokens: int = 0
    recompute_reason: Optional[str] = None
    logical_context_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def slice_saving_tokens(self) -> int:
        return max(self.full_context_tokens - self.slice_tokens, 0)


@dataclass
class RuntimeMetrics:
    records: List[CallRecord] = field(default_factory=list)
    graph_nodes_created: int = 0
    graph_edges_created: int = 0
    invalidation_events: int = 0
    stale_nodes: int = 0
    graph_overhead_ms: float = 0.0

    def add_call(self, record: CallRecord) -> None:
        self.records.append(record)

    def summary(self) -> Dict[str, Any]:
        input_tokens = sum(record.input_tokens for record in self.records)
        output_tokens = sum(record.output_tokens for record in self.records)
        return {
            "total_input_tokens": input_tokens,
            "total_output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "total_latency_sec": sum(record.latency_sec for record in self.records),
            "context_tokens": sum(record.context_tokens for record in self.records),
            "full_context_tokens": sum(record.full_context_tokens for record in self.records),
            "slice_saving_tokens": sum(record.slice_saving_tokens for record in self.records),
            "cache_hits": sum(int(record.cache_hit) for record in self.records),
            "avoided_llm_calls": sum(record.avoided_llm_calls for record in self.records),
            "estimated_reused_input_tokens": sum(
                record.estimated_reused_input_tokens for record in self.records
            ),
            "a2a_text_tokens": sum(record.a2a_text_tokens for record in self.records),
            "graph_update_tokens": sum(record.graph_update_tokens for record in self.records),
            "graph_read_tokens": sum(record.graph_read_tokens for record in self.records),
            "logical_context_tokens": sum(record.logical_context_tokens for record in self.records),
            "graph_nodes_created": self.graph_nodes_created,
            "graph_edges_created": self.graph_edges_created,
            "invalidation_events": self.invalidation_events,
            "stale_nodes": self.stale_nodes,
            "graph_overhead_ms": self.graph_overhead_ms
            + sum(record.graph_overhead_ms for record in self.records),
            "records": [asdict(record) | {"total_tokens": record.total_tokens} for record in self.records],
        }
