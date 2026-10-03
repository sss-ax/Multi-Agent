"""Structured quality and token-cost telemetry for workflow runs."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional


class WorkflowTelemetry:
    """Append JSONL events while keeping an in-memory run summary."""

    def __init__(self, path: str | Path | None = None, *, max_output_chars: int = 4000) -> None:
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_output_chars = max(256, int(max_output_chars))
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self._native_stage_roles: set[str] = set()
        self._native_message_ids: set[str] = set()
        self._summary: dict[str, Any] = {
            "action_events": 0,
            "native_model_calls": 0,
            "native_messages": 0,
            "native_agent_tokens": {},
            "graph_communication_events": 0,
            "graph_delta_candidate_tokens": 0,
            "graph_delta_sent_tokens": 0,
            "graph_delta_candidates": 0,
            "graph_delta_sent_nodes": 0,
            "graph_delta_closure_added_nodes": 0,
            "graph_delta_mandatory_roots": 0,
            "graph_delta_optional_roots": 0,
            "graph_delta_selected_optional_roots": 0,
            "graph_delta_mandatory_root_tokens": 0,
            "graph_delta_optional_root_tokens": 0,
            "graph_delta_selected_optional_root_tokens": 0,
            "invalid_action_events": 0,
            "compile_failures": 0,
            "generation_errors": 0,
            "evaluated_answers": 0,
            "correct_answers": 0,
            "incorrect_answers": 0,
            "retry_count": 0,
            "logical_input_tokens": 0,
            "physical_input_tokens": 0,
            "output_tokens": 0,
            "forward_calls": 0,
            "graph_read_context_tokens": 0,
            "graph_update_tokens": 0,
            "peak_context_tokens": 0,
            "quality": {
                "protocol_valid_actions": 0,
                "compiled_actions": 0,
                "completed_stages": 0,
                "failed_stages": 0,
            },
        }

    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "event": event,
            "timestamp": time.time(),
            **payload,
        }
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":")))
                stream.write("\n")

    def record_action(self, payload: dict[str, Any]) -> None:
        event = dict(payload)
        raw_output = str(event.pop("raw_output", ""))
        event["raw_output_chars"] = len(raw_output)
        event["raw_output_sha256"] = self._digest(raw_output)
        event["raw_output"] = raw_output[: self.max_output_chars]
        self.emit("model_action", event)

        self._summary["action_events"] += 1
        self._summary["retry_count"] += max(0, int(event.get("attempts", 1)) - 1)
        if not event.get("protocol_valid", False):
            self._summary["invalid_action_events"] += 1
        if not event.get("compile_success", False):
            self._summary["compile_failures"] += 1
        quality = self._summary["quality"]
        quality["protocol_valid_actions"] += int(bool(event.get("protocol_valid", False)))
        quality["compiled_actions"] += int(bool(event.get("compile_success", False)))
        quality["completed_stages"] += int(bool(event.get("stage_complete", False)))
        quality["failed_stages"] += int(bool(event.get("stage_failed", False)))
        for key in (
            "logical_input_tokens", "physical_input_tokens", "output_tokens",
            "forward_calls", "graph_read_context_tokens", "graph_update_tokens",
        ):
            self._summary[key] += int(event.get(key, 0) or 0)
        self._summary["peak_context_tokens"] = max(
            int(self._summary.get("peak_context_tokens", 0) or 0),
            int(event.get("graph_read_context_tokens", 0) or 0),
        )

    def record_generation_error(self, payload: dict[str, Any]) -> None:
        """Record a generation that exhausted protocol-repair attempts."""
        event = dict(payload)
        raw_output = str(event.pop("raw_output", ""))
        event["raw_output_chars"] = len(raw_output)
        event["raw_output_sha256"] = self._digest(raw_output)
        event["raw_output"] = raw_output[: self.max_output_chars]
        self.emit("model_generation_error", event)

        self._summary["generation_errors"] += 1
        attempts = int(event.get("attempts", 1) or 1)
        self._summary["retry_count"] += max(0, attempts - 1)
        self._summary["invalid_action_events"] += 1
        self._summary["quality"]["failed_stages"] += 1
        self._summary["logical_input_tokens"] += int(event.get("logical_input_tokens", 0) or 0)
        self._summary["graph_read_context_tokens"] += int(event.get("graph_read_context_tokens", 0) or 0)
        self._summary["graph_update_tokens"] += int(event.get("graph_update_tokens", 0) or 0)
        self._summary["peak_context_tokens"] = max(
            int(self._summary.get("peak_context_tokens", 0) or 0),
            int(event.get("graph_read_context_tokens", 0) or 0),
        )
        metrics = event.get("call_metrics", [])
        for metric in metrics if isinstance(metrics, list) else []:
            for key in (
                "physical_input_tokens", "output_tokens", "forward_calls",
            ):
                self._summary[key] += int(metric.get(key, 0) or 0)

    def record_graph_communication(self, payload: dict[str, Any]) -> None:
        """Record one graph-delta communication decision."""
        event = dict(payload)
        self.emit("graph_delta_communication", event)
        self._summary["graph_communication_events"] += 1
        self._summary["graph_delta_candidate_tokens"] += int(event.get("candidate_tokens", 0) or 0)
        self._summary["graph_delta_sent_tokens"] += int(event.get("sent_tokens", 0) or 0)
        self._summary["graph_delta_candidates"] += int(event.get("candidate_count", 0) or 0)
        self._summary["graph_delta_sent_nodes"] += int(event.get("sent_count", 0) or 0)
        self._summary["graph_delta_closure_added_nodes"] += int(event.get("closure_added_count", 0) or 0)
        self._summary["graph_delta_mandatory_roots"] += int(event.get("mandatory_root_count", 0) or 0)
        self._summary["graph_delta_optional_roots"] += int(event.get("optional_root_count", 0) or 0)
        self._summary["graph_delta_selected_optional_roots"] += int(event.get("selected_optional_root_count", 0) or 0)
        self._summary["graph_delta_mandatory_root_tokens"] += int(event.get("mandatory_root_tokens", 0) or 0)
        self._summary["graph_delta_optional_root_tokens"] += int(event.get("optional_root_tokens", 0) or 0)
        self._summary["graph_delta_selected_optional_root_tokens"] += int(event.get("selected_optional_root_tokens", 0) or 0)

    def record_native_call(self, payload: dict[str, Any]) -> None:
        """Record one unconstrained natural-language baseline model call."""
        event = dict(payload)
        raw_output = str(event.pop("raw_output", ""))
        event["raw_output_chars"] = len(raw_output)
        event["raw_output_sha256"] = self._digest(raw_output)
        event["raw_output"] = raw_output[: self.max_output_chars]
        metrics = event.pop("call_metrics", {})
        event["call_metrics"] = dict(metrics) if isinstance(metrics, dict) else {}
        self.emit("native_model_call", event)

        self._summary["native_model_calls"] += 1
        role = str(event.get("role", ""))
        if role in {"planner", "solver", "critic", "final_solver"} and role not in self._native_stage_roles:
            self._native_stage_roles.add(role)
            self._summary["quality"]["completed_stages"] += 1
        for key in (
            "physical_input_tokens", "output_tokens", "forward_calls",
        ):
            self._summary[key] += int(event["call_metrics"].get(key, 0) or 0)
        agent_id = str(event.get("agent_id", ""))
        if agent_id:
            agent_tokens = self._summary["native_agent_tokens"]
            agent_tokens[agent_id] = int(agent_tokens.get(agent_id, 0)) + sum(
                int(event["call_metrics"].get(key, 0) or 0)
                for key in ("physical_input_tokens", "output_tokens")
            )
        # Native mode has no prefix planner, so logical input equals the full
        # prompt fed to the direct backend.
        self._summary["logical_input_tokens"] += int(
            event["call_metrics"].get("physical_input_tokens", 0) or 0
        )

    def record_native_message(self, payload: dict[str, Any]) -> None:
        """Record one message-bus envelope without conflating it with a model call."""
        self.emit("native_message", dict(payload))
        self._summary["native_messages"] += 1

    def finish(self, *, status: str, error: Optional[str] = None, **extra: Any) -> dict[str, Any]:
        self.finished_at = time.time()
        summary = {
            **self._summary,
            "status": status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": self.finished_at - self.started_at,
            **extra,
        }
        summary["total_model_tokens"] = int(summary.get("physical_input_tokens", 0) or 0) + int(summary.get("output_tokens", 0) or 0)
        evaluated_answers = int(self._summary["evaluated_answers"])
        summary["answer_accuracy"] = (
            self._summary["correct_answers"] / evaluated_answers
            if evaluated_answers else None
        )
        if error:
            summary["error"] = error
        self.emit("workflow_summary", summary)
        return summary

    def record_evaluation(self, evaluation: dict[str, Any]) -> None:
        """Record a task-level answer evaluation and update accuracy counters."""
        self.emit("answer_evaluation", dict(evaluation))
        if evaluation.get("status") != "evaluated":
            return
        self._summary["evaluated_answers"] += 1
        if evaluation.get("correct") is True:
            self._summary["correct_answers"] += 1
        elif evaluation.get("correct") is False:
            self._summary["incorrect_answers"] += 1

    def summary(self) -> dict[str, Any]:
        summary = dict(self._summary)
        summary["total_model_tokens"] = int(summary.get("physical_input_tokens", 0) or 0) + int(summary.get("output_tokens", 0) or 0)
        return summary
