"""Adapter for the official MultiAgentBench/MARBLE data.

The upstream repository contains multiple environments. This adapter turns
task records into typed LangGraph workflow inputs and preserves domain metadata
for environment-specific judges.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional


SUPPORTED_TASK_TYPES = frozenset(
    {"numeric_solve", "numeric_comparison", "table_qa", "multihop_qa", "code_generation"}
)
MARBLE_TASK_TYPES = frozenset({"marble_research", "marble_bargaining", "marble_database"})


@dataclass(frozen=True)
class MultiAgentBenchTask:
    """Normalized task consumed by the local LangGraph benchmark runner."""

    task_id: str
    prompt: str
    task_type: str
    domain: str = "unknown"
    coordination_category: str = "unknown"
    reference_answer: Any = None
    has_reference: bool = False
    answer_tolerance: float = 1e-6
    requirements: tuple[str, ...] = ()
    milestones: tuple[str, ...] = ()
    topology: str = "graph"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: str = ""

    @property
    def supported(self) -> bool:
        return self.task_type in SUPPORTED_TASK_TYPES or self.task_type in MARBLE_TASK_TYPES

    @property
    def unsupported_reason(self) -> Optional[str]:
        if self.supported:
            return None
        return (
            f"task domain {self.domain!r} is not implemented by the current runtime; "
            "an environment-specific adapter/evaluator is required"
        )

    def runtime_prompt(self) -> str:
        """Render the official task content without losing requirements."""
        if not self.requirements:
            return self.prompt
        requirements = "\n".join(f"{index}. {item}" for index, item in enumerate(self.requirements, 1))
        return f"{self.prompt}\n\nAcceptance requirements:\n{requirements}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "runtime_prompt": self.runtime_prompt(),
            "task_type": self.task_type,
            "domain": self.domain,
            "coordination_category": self.coordination_category,
            "reference_answer": self.reference_answer,
            "has_reference": self.has_reference,
            "answer_tolerance": self.answer_tolerance,
            "requirements": list(self.requirements),
            "milestones": list(self.milestones),
            "topology": self.topology,
            "metadata": dict(self.metadata),
            "source": self.source,
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
        }


def _records(path: Path) -> Iterator[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on {path}:{line_number}: {exc}") from exc
                if not isinstance(item, Mapping):
                    raise ValueError(f"expected an object on {path}:{line_number}")
                yield item
        return
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("YAML input requires PyYAML; install requirements-train.txt") from exc
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            items = value
        elif isinstance(value, Mapping):
            items = value.get("tasks", value.get("data", value.get("items")))
            if items is None:
                items = [value]
        else:
            raise ValueError(f"{path} must contain a YAML object or array")
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError(f"record {index} in {path} is not an object")
            yield item
        return
    if suffix != ".json":
        raise ValueError(
            f"unsupported benchmark file {path}; use the official JSONL asset or a JSON manifest"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        items = value
    elif isinstance(value, Mapping):
        items = value.get("tasks", value.get("data", value.get("items")))
        if items is None:
            items = [value]
    else:
        raise ValueError(f"{path} must contain an object or array")
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"record {index} in {path} is not an object")
        yield item


def _first(record: Mapping[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return default


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _task_type(record: Mapping[str, Any]) -> tuple[str, str]:
    explicit = str(_first(record, ("task_type", "runtime_task_type", "type"), "")).strip()
    if explicit in SUPPORTED_TASK_TYPES or explicit in MARBLE_TASK_TYPES:
        return explicit, str(_first(record, ("domain", "topic_category", "scenario"), explicit))
    domain = str(_first(record, ("domain", "topic_category", "scenario", "category"), "unknown")).strip()
    lowered = domain.casefold()
    if any(marker in lowered for marker in ("coding", "code", "software", "program")):
        return "code_generation", domain
    if any(marker in lowered for marker in ("research", "scientific", "paper")):
        return "marble_research", domain
    if any(marker in lowered for marker in ("bargain", "negotiat", "auction", "trade")):
        return "marble_bargaining", domain
    if any(marker in lowered for marker in ("database", "sql", "postgres", "data management")):
        return "marble_database", domain
    return "unsupported", domain


def _normalize(record: Mapping[str, Any], *, source: str, index: int) -> MultiAgentBenchTask:
    task_id = str(_first(record, ("task_id", "id", "uid", "name"), f"task_{index}"))
    raw_task = record.get("task")
    if isinstance(raw_task, Mapping):
        prompt = _as_text(_first(raw_task, ("content", "prompt", "description", "task"), ""))
        nested_requirements = raw_task.get("requirements", [])
    else:
        prompt = _as_text(_first(record, ("task_prompt", "prompt", "content", "description", "task"), ""))
        nested_requirements = []
    if not prompt:
        raise ValueError(f"task {task_id!r} has no task/prompt/content field")
    task_type, domain = _task_type(record)
    raw_requirements = _first(
        record,
        ("requirements", "milestones", "acceptance_criteria"),
        nested_requirements,
    )
    if isinstance(raw_requirements, str):
        requirements = (raw_requirements,)
    elif isinstance(raw_requirements, list):
        requirements = tuple(_as_text(item) for item in raw_requirements if _as_text(item))
    else:
        requirements = ()
    raw_reference = _first(record, ("reference_answer", "gold_answer", "answer"), None)
    has_reference = any(name in record for name in ("reference_answer", "gold_answer", "answer"))
    tolerance = float(_first(record, ("answer_tolerance", "tolerance"), 1e-6))
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(f"task {task_id!r} has invalid answer tolerance")
    metadata = dict(record)
    return MultiAgentBenchTask(
        task_id=task_id,
        prompt=prompt,
        task_type=task_type,
        domain=domain,
        coordination_category=str(_first(record, ("coordination_category", "coordination", "category", "coordinate_mode"), "unknown")),
        reference_answer=raw_reference,
        has_reference=has_reference,
        answer_tolerance=tolerance,
        requirements=requirements,
        milestones=requirements,
        topology=str(_first(record, ("topology", "coordinate_mode", "communication_topology"), "graph")),
        metadata=metadata,
        source=source,
    )


def load_multiagentbench_tasks(
    path: str | Path,
    *,
    limit: Optional[int] = None,
    task_ids: Optional[set[str]] = None,
) -> list[MultiAgentBenchTask]:
    """Load official MARBLE JSONL or a normalized JSON/JSONL manifest."""
    source = str(path)
    result: list[MultiAgentBenchTask] = []
    for index, record in enumerate(_records(Path(path))):
        task = _normalize(record, source=source, index=index)
        if task_ids and task.task_id not in task_ids:
            continue
        result.append(task)
        if limit is not None and len(result) >= max(0, int(limit)):
            break
    return result


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _mean(values: list[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_results(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-task records without treating unevaluated tasks as correct."""
    records = list(results)
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in sorted({str(item.get("execution_mode", "unknown")) for item in records}):
        selected = [item for item in records if item.get("execution_mode") == mode]
        successful = [item for item in selected if item.get("runtime_success") is True]
        evaluated = [item for item in selected if item.get("answer_status") == "evaluated"]
        correct = [item for item in evaluated if item.get("answer_correct") is True]
        physical = [float(item.get("physical_input_tokens", 0) or 0) for item in selected]
        output = [float(item.get("output_tokens", 0) or 0) for item in selected]
        total = [float(item.get("total_model_tokens", 0) or 0) for item in selected]
        delta_candidate = [float(item.get("graph_delta_candidate_tokens", 0) or 0) for item in selected]
        delta_sent = [float(item.get("graph_delta_sent_tokens", 0) or 0) for item in selected]
        latency = [float(item.get("duration_sec", 0) or 0) for item in selected]
        successful_total = [float(item.get("total_model_tokens", 0) or 0) for item in successful]
        milestone_proxy = [float(item.get("milestone_proxy_score", 0) or 0) for item in selected]
        official = [
            item.get("official_marble")
            for item in selected
            if item.get("official_marble_status") == "scored"
            and isinstance(item.get("official_marble"), Mapping)
        ]

        def _official_mean(metric: str) -> Optional[float]:
            values: list[float] = []
            for item in official:
                raw = item.get(metric, [])
                if isinstance(raw, list):
                    values.extend(float(value) for value in raw if _number(value) is not None)
            return _mean(values)

        by_mode[mode] = {
            "runs": len(selected),
            "runtime_successes": len(successful),
            "runtime_completion_rate": len(successful) / len(selected) if selected else None,
            "evaluated_runs": len(evaluated),
            "correct_runs": len(correct),
            "answer_accuracy": len(correct) / len(evaluated) if evaluated else None,
            "avg_physical_input_tokens": _mean(physical),
            "avg_output_tokens": _mean(output),
            "avg_total_model_tokens": _mean(total),
            "avg_graph_delta_candidate_tokens": _mean(delta_candidate),
            "avg_graph_delta_sent_tokens": _mean(delta_sent),
            "graph_delta_sent_nodes": sum(int(item.get("graph_delta_sent_nodes", 0) or 0) for item in selected),
            "graph_communication_events": sum(int(item.get("graph_communication_events", 0) or 0) for item in selected),
            "avg_duration_sec": _mean(latency),
            "p50_duration_sec": _percentile(latency, 0.50),
            "p95_duration_sec": _percentile(latency, 0.95),
            "tokens_per_runtime_success": (
                sum(successful_total) / len(successful_total) if successful_total else None
            ),
            "avg_milestone_proxy_score": _mean(milestone_proxy),
            "official_marble_scored_runs": len(official),
            "official_marble_communication_score": _official_mean("communication_score"),
            "official_marble_planning_score": _official_mean("planning_score"),
            "retry_count": sum(int(item.get("retry_count", 0) or 0) for item in selected),
        }
    return {
        "runs": len(records),
        "modes": by_mode,
        "quality_note": (
            "runtime_completion_rate is not benchmark correctness. For official MARBLE domains "
            "without a reference/evaluator, answer_accuracy remains null."
        ),
    }
