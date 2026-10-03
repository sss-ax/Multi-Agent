"""Benchmark adapters and aggregation helpers."""

from .multiagentbench import (
    MultiAgentBenchTask,
    load_multiagentbench_tasks,
    summarize_results,
)

__all__ = [
    "MultiAgentBenchTask",
    "load_multiagentbench_tasks",
    "summarize_results",
]
