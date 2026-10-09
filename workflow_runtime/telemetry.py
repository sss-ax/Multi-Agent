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
        self._seen_reusable_prefix_keys: set[str] = set()
        self._summary: dict[str, Any] = {
            "action_events": 0,
            "native_model_calls": 0,
            "native_messages": 0,
            "native_agent_tokens": {},
            "graph_communication_events": 0,
            "reasoning_round_count": 0,
            "solver_revision_round_count": 0,
            "critic_need_fix_count": 0,
            "critic_verified_count": 0,
            "communication_round_count": 0,
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
            "semantic_nack_count": 0,
            "semantic_hard_nack_count": 0,
            "semantic_soft_nack_count": 0,
            "semantic_verification_nack_count": 0,
            "semantic_quality_nack_count": 0,
            "semantic_initial_nack_count": 0,
            "semantic_initial_hard_nack_count": 0,
            "semantic_initial_soft_nack_count": 0,
            "semantic_initial_verification_nack_count": 0,
            "semantic_initial_quality_nack_count": 0,
            "semantic_missing_repaired_count": 0,
            "semantic_hard_repaired_count": 0,
            "semantic_soft_repaired_count": 0,
            "semantic_verification_repaired_count": 0,
            "semantic_quality_repaired_count": 0,
            "feedback_sent_tokens": 0,
            "feedback_transport_tokens": 0,
            "feedback_newly_visible_tokens": 0,
            "feedback_newly_rendered_tokens": 0,
            "total_comm_tokens": 0,
            "core_comm_tokens": 0,
            "delta_comm_tokens": 0,
            "verification_comm_tokens": 0,
            "quality_comm_tokens": 0,
            "control_comm_tokens": 0,
            "unique_comm_tokens": 0,
            "repeated_comm_tokens": 0,
            "receiver_seen_hit_count": 0,
            "state_delta_tokens": 0,
            "full_state_equivalent_tokens": 0,
            "revision_success_count": 0,
            "revision_regression_count": 0,
            "early_stop_rounds": [],
            "communication_token_accounting_errors": 0,
            "communication_token_breakdown_errors": 0,
            "feedback_render_accounting_errors": 0,
            "unique_repeated_accounting_errors": 0,
            "targeted_refinement_skipped_count": 0,
            "quality_refinement_budget_exceeded_count": 0,
            "invalid_action_events": 0,
            "compile_failures": 0,
            "generation_errors": 0,
            "evaluated_answers": 0,
            "correct_answers": 0,
            "incorrect_answers": 0,
            "retry_count": 0,
            "logical_input_tokens": 0,
            "physical_input_tokens": 0,
            "physical_llm_input_tokens": 0,
            "output_tokens": 0,
            "prefill_cost_tokens": 0,
            "decode_cost_tokens": 0,
            "forward_calls": 0,
            "graph_read_context_tokens": 0,
            "persistent_context_tokens": 0,
            "incremental_context_tokens": 0,
            "logical_communication_tokens": 0,
            "system_prompt_tokens": 0,
            "prompt_wrapper_tokens": 0,
            "reusable_prefix_tokens": 0,
            "unique_prefix_tokens": 0,
            "repeated_prefix_tokens": 0,
            "simulated_context_reuse_input_tokens": 0,
            "graph_update_tokens": 0,
            "graph_context_content_tokens": 0,
            "graph_context_wrapper_tokens": 0,
            "graph_context_edge_tokens": 0,
            "graph_source_duplicate_in_prompt_tokens": 0,
            "graph_source_duplicate_in_prompt_original_tokens": 0,
            "graph_source_deduplicated_prompt_saved_tokens": 0,
            "graph_source_cross_call_reread_tokens": 0,
            "graph_role_aware_source_ref_saved_tokens": 0,
            "graph_role_aware_state_ref_saved_tokens": 0,
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
        self._summary["reasoning_round_count"] += 1
        if event.get("role") == "solver" and event.get("mode") == "repair":
            self._summary["solver_revision_round_count"] += 1
        action_payload = event.get("action") if isinstance(event.get("action"), dict) else {}
        if event.get("role") == "critic" and isinstance(action_payload, dict):
            if action_payload.get("op") == "verify" and action_payload.get("status") == "need_fix":
                self._summary["critic_need_fix_count"] += 1
            if action_payload.get("op") == "verify" and action_payload.get("status") == "verified":
                self._summary["critic_verified_count"] += 1
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
            "persistent_context_tokens", "incremental_context_tokens",
            "logical_communication_tokens", "system_prompt_tokens",
            "prompt_wrapper_tokens", "reusable_prefix_tokens",
            "graph_context_content_tokens", "graph_context_wrapper_tokens",
            "graph_context_edge_tokens", "graph_source_duplicate_in_prompt_tokens",
            "graph_source_duplicate_in_prompt_original_tokens",
            "graph_source_deduplicated_prompt_saved_tokens",
            "graph_source_cross_call_reread_tokens",
            "graph_role_aware_source_ref_saved_tokens",
            "graph_role_aware_state_ref_saved_tokens",
        ):
            self._summary[key] += int(event.get(key, 0) or 0)
        physical_input = int(event.get("physical_input_tokens", 0) or 0)
        output_tokens = int(event.get("output_tokens", 0) or 0)
        self._summary["physical_llm_input_tokens"] += int(
            event.get("physical_llm_input_tokens", physical_input) or 0
        )
        self._summary["prefill_cost_tokens"] += int(
            event.get("prefill_cost_tokens", physical_input) or 0
        )
        self._summary["decode_cost_tokens"] += int(
            event.get("decode_cost_tokens", output_tokens) or 0
        )
        reusable_prefix_tokens = int(event.get("reusable_prefix_tokens", 0) or 0)
        reusable_prefix_key = str(event.get("reusable_prefix_key", ""))
        repeated_prefix_tokens = 0
        if reusable_prefix_key:
            if reusable_prefix_key in self._seen_reusable_prefix_keys:
                repeated_prefix_tokens = reusable_prefix_tokens
                self._summary["repeated_prefix_tokens"] += repeated_prefix_tokens
            else:
                self._seen_reusable_prefix_keys.add(reusable_prefix_key)
                self._summary["unique_prefix_tokens"] += reusable_prefix_tokens
        self._summary["simulated_context_reuse_input_tokens"] += max(
            0,
            physical_input - min(physical_input, repeated_prefix_tokens),
        )
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
        self._summary["persistent_context_tokens"] += int(event.get("persistent_context_tokens", 0) or 0)
        self._summary["incremental_context_tokens"] += int(event.get("incremental_context_tokens", 0) or 0)
        self._summary["logical_communication_tokens"] += int(event.get("logical_communication_tokens", 0) or 0)
        self._summary["system_prompt_tokens"] += int(event.get("system_prompt_tokens", 0) or 0)
        self._summary["prompt_wrapper_tokens"] += int(event.get("prompt_wrapper_tokens", 0) or 0)
        self._summary["reusable_prefix_tokens"] += int(event.get("reusable_prefix_tokens", 0) or 0)
        self._summary["graph_update_tokens"] += int(event.get("graph_update_tokens", 0) or 0)
        self._summary["graph_context_content_tokens"] += int(event.get("graph_context_content_tokens", 0) or 0)
        self._summary["graph_context_wrapper_tokens"] += int(event.get("graph_context_wrapper_tokens", 0) or 0)
        self._summary["graph_context_edge_tokens"] += int(event.get("graph_context_edge_tokens", 0) or 0)
        self._summary["graph_source_duplicate_in_prompt_tokens"] += int(
            event.get("graph_source_duplicate_in_prompt_tokens", 0) or 0
        )
        self._summary["graph_source_duplicate_in_prompt_original_tokens"] += int(
            event.get("graph_source_duplicate_in_prompt_original_tokens", 0) or 0
        )
        self._summary["graph_source_deduplicated_prompt_saved_tokens"] += int(
            event.get("graph_source_deduplicated_prompt_saved_tokens", 0) or 0
        )
        self._summary["graph_source_cross_call_reread_tokens"] += int(
            event.get("graph_source_cross_call_reread_tokens", 0) or 0
        )
        self._summary["graph_role_aware_source_ref_saved_tokens"] += int(
            event.get("graph_role_aware_source_ref_saved_tokens", 0) or 0
        )
        self._summary["graph_role_aware_state_ref_saved_tokens"] += int(
            event.get("graph_role_aware_state_ref_saved_tokens", 0) or 0
        )
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
            physical_input = int(metric.get("physical_input_tokens", 0) or 0)
            self._summary["physical_llm_input_tokens"] += physical_input
            self._summary["prefill_cost_tokens"] += physical_input
            self._summary["decode_cost_tokens"] += int(metric.get("output_tokens", 0) or 0)

    def record_graph_communication(self, payload: dict[str, Any]) -> None:
        """Record one graph-delta communication decision."""
        event = dict(payload)
        self.emit("graph_delta_communication", event)
        self._summary["graph_communication_events"] += 1
        self._summary["communication_round_count"] += 1
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
        self._summary["semantic_nack_count"] += int(bool(event.get("semantic_nack", False)))
        self._summary["semantic_hard_nack_count"] += int(bool(event.get("semantic_hard_nack", False)))
        self._summary["semantic_soft_nack_count"] += int(bool(event.get("semantic_soft_nack", False)))
        self._summary["semantic_verification_nack_count"] += int(bool(event.get("semantic_verification_nack", False)))
        self._summary["semantic_quality_nack_count"] += int(bool(event.get("semantic_quality_nack", False)))
        self._summary["semantic_initial_nack_count"] += int(bool(event.get("semantic_initial_nack", False)))
        self._summary["semantic_initial_hard_nack_count"] += int(bool(event.get("semantic_initial_hard_nack", False)))
        self._summary["semantic_initial_soft_nack_count"] += int(bool(event.get("semantic_initial_soft_nack", False)))
        self._summary["semantic_initial_verification_nack_count"] += int(bool(event.get("semantic_initial_verification_nack", False)))
        self._summary["semantic_initial_quality_nack_count"] += int(bool(event.get("semantic_initial_quality_nack", False)))
        for key in (
            "semantic_missing_repaired_count",
            "semantic_hard_repaired_count",
            "semantic_soft_repaired_count",
            "semantic_verification_repaired_count",
            "semantic_quality_repaired_count",
            "feedback_sent_tokens",
            "feedback_transport_tokens",
            "feedback_newly_visible_tokens",
            "feedback_newly_rendered_tokens",
            "total_comm_tokens",
            "core_comm_tokens",
            "delta_comm_tokens",
            "verification_comm_tokens",
            "quality_comm_tokens",
            "control_comm_tokens",
            "unique_comm_tokens",
            "repeated_comm_tokens",
            "receiver_seen_hit_count",
            "state_delta_tokens",
            "full_state_equivalent_tokens",
            "revision_success_count",
            "revision_regression_count",
        ):
            self._summary[key] += int(event.get(key, 0) or 0)
        early_stop_round = event.get("early_stop_round")
        if early_stop_round is not None:
            self._summary["early_stop_rounds"].append(int(early_stop_round))
        if not bool(event.get("communication_token_accounting_ok", True)):
            self._summary["communication_token_accounting_errors"] += 1
        if not bool(event.get("communication_token_breakdown_ok", True)):
            self._summary["communication_token_breakdown_errors"] += 1
        if not bool(event.get("feedback_render_accounting_ok", True)):
            self._summary["feedback_render_accounting_errors"] += 1
        if not bool(event.get("unique_repeated_accounting_ok", True)):
            self._summary["unique_repeated_accounting_errors"] += 1
        if event.get("targeted_refinement_skipped", False):
            self._summary["targeted_refinement_skipped_count"] += 1
            if event.get("targeted_refinement_skip_reason") == "quality_refinement_budget_exceeded":
                self._summary["quality_refinement_budget_exceeded_count"] += 1

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
        physical_input = int(event["call_metrics"].get("physical_input_tokens", 0) or 0)
        output_tokens = int(event["call_metrics"].get("output_tokens", 0) or 0)
        self._summary["physical_llm_input_tokens"] += physical_input
        self._summary["prefill_cost_tokens"] += physical_input
        self._summary["decode_cost_tokens"] += output_tokens
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
        summary = self._with_derived_summary(summary)
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
        return self._with_derived_summary(dict(self._summary))

    def _with_derived_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        summary["total_model_tokens"] = int(summary.get("physical_input_tokens", 0) or 0) + int(summary.get("output_tokens", 0) or 0)
        summary["communication_cost_tokens"] = int(summary.get("total_comm_tokens", 0) or 0)
        summary["semantic_communication_cost_tokens"] = int(summary.get("logical_communication_tokens", 0) or 0)
        summary["inference_prefill_cost_tokens"] = int(summary.get("prefill_cost_tokens", 0) or 0)
        summary["inference_decode_cost_tokens"] = int(summary.get("decode_cost_tokens", 0) or 0)
        summary["phase8_total_cost_tokens"] = (
            int(summary.get("communication_cost_tokens", 0) or 0)
            + int(summary.get("inference_prefill_cost_tokens", 0) or 0)
            + int(summary.get("inference_decode_cost_tokens", 0) or 0)
        )
        physical_input = int(summary.get("physical_input_tokens", 0) or 0)
        repeated_prefix = int(summary.get("repeated_prefix_tokens", 0) or 0)
        summary["context_reuse_saving_ratio"] = (
            repeated_prefix / physical_input if physical_input else 0.0
        )
        summary["logical_vs_physical_input_gap_tokens"] = (
            physical_input - int(summary.get("logical_communication_tokens", 0) or 0)
        )
        feedback_sent = int(summary.get("feedback_sent_tokens", 0) or 0)
        summary["feedback_utilization"] = (
            int(summary.get("feedback_newly_rendered_tokens", 0) or 0) / feedback_sent
            if feedback_sent else 0.0
        )
        summary["feedback_nack_repair_efficiency"] = (
            int(summary.get("semantic_missing_repaired_count", 0) or 0) / feedback_sent
            if feedback_sent else 0.0
        )
        summary["feedback_hard_repair_efficiency"] = (
            int(summary.get("semantic_hard_repaired_count", 0) or 0) / feedback_sent
            if feedback_sent else 0.0
        )
        summary["feedback_soft_repair_efficiency"] = (
            int(summary.get("semantic_soft_repaired_count", 0) or 0) / feedback_sent
            if feedback_sent else 0.0
        )
        summary["feedback_verification_repair_efficiency"] = (
            int(summary.get("semantic_verification_repaired_count", 0) or 0) / feedback_sent
            if feedback_sent else 0.0
        )
        summary["feedback_quality_repair_efficiency"] = (
            int(summary.get("semantic_quality_repaired_count", 0) or 0) / feedback_sent
            if feedback_sent else 0.0
        )
        total_comm = int(summary.get("total_comm_tokens", 0) or 0)
        repeated_comm = int(summary.get("repeated_comm_tokens", 0) or 0)
        state_delta = int(summary.get("state_delta_tokens", 0) or 0)
        full_state = int(summary.get("full_state_equivalent_tokens", 0) or 0)
        summary["duplicate_ratio"] = repeated_comm / total_comm if total_comm else 0.0
        summary["incremental_saving"] = (
            1.0 - (state_delta / full_state)
            if full_state else 0.0
        )
        summary["communication_token_breakdown_ok"] = total_comm == sum(
            int(summary.get(key, 0) or 0)
            for key in (
                "core_comm_tokens",
                "delta_comm_tokens",
                "verification_comm_tokens",
                "quality_comm_tokens",
                "control_comm_tokens",
            )
        )
        summary["feedback_render_accounting_ok"] = (
            int(summary.get("feedback_sent_tokens", 0) or 0)
            >= int(summary.get("feedback_newly_rendered_tokens", 0) or 0)
        )
        summary["unique_repeated_accounting_ok"] = total_comm == (
            int(summary.get("unique_comm_tokens", 0) or 0)
            + int(summary.get("repeated_comm_tokens", 0) or 0)
        )
        early_stop_rounds = summary.get("early_stop_rounds", [])
        summary["early_stop_round"] = min(early_stop_rounds) if early_stop_rounds else None
        return summary
