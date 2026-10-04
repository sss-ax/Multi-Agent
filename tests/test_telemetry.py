import json

from workflow_runtime.telemetry import WorkflowTelemetry


def test_telemetry_writes_quality_and_cost_events(tmp_path):
    path = tmp_path / "workflow.jsonl"
    telemetry = WorkflowTelemetry(path, max_output_chars=12)
    telemetry.record_action({
        "role": "planner",
        "mode": "normal",
        "action_index": 0,
        "raw_output": '{"type":"done"}' * 30,
        "attempts": 2,
        "protocol_valid": True,
        "compile_success": True,
        "stage_complete": True,
        "stage_failed": False,
        "logical_input_tokens": 100,
        "physical_input_tokens": 40,
        "output_tokens": 9,
        "forward_calls": 3,
    })
    summary = telemetry.finish(status="error", error="test failure")

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == ["model_action", "workflow_summary"]
    action = records[0]
    assert action["raw_output_chars"] > len(action["raw_output"])
    assert action["raw_output_sha256"]
    assert summary["retry_count"] == 1
    assert summary["logical_input_tokens"] == 100
    assert summary["physical_input_tokens"] == 40
    assert summary["quality"]["completed_stages"] == 1


def test_telemetry_records_answer_accuracy(tmp_path):
    telemetry = WorkflowTelemetry(tmp_path / "evaluation.jsonl")
    telemetry.record_evaluation({
        "task_type": "numeric_solve",
        "status": "evaluated",
        "correct": True,
    })
    summary = telemetry.finish(status="success")
    assert summary["evaluated_answers"] == 1
    assert summary["correct_answers"] == 1
    assert summary["incorrect_answers"] == 0
    assert summary["answer_accuracy"] == 1.0


def test_telemetry_records_native_calls_as_unconstrained_stages(tmp_path):
    telemetry = WorkflowTelemetry(tmp_path / "native.jsonl")
    telemetry.record_native_call({
        "role": "planner",
        "raw_output": "ordinary planner text",
        "call_metrics": {"physical_input_tokens": 10, "output_tokens": 3},
    })
    telemetry.record_native_call({
        "role": "final_solver",
        "raw_output": "FINAL ANSWER: 5",
        "call_metrics": {"physical_input_tokens": 20, "output_tokens": 4},
    })
    summary = telemetry.finish(status="success")
    assert summary["native_model_calls"] == 2
    assert summary["physical_input_tokens"] == 30
    assert summary["output_tokens"] == 7
    assert summary["quality"]["completed_stages"] == 2


def test_telemetry_records_phase7_multiround_communication_metrics(tmp_path):
    telemetry = WorkflowTelemetry(tmp_path / "phase7.jsonl")
    telemetry.record_action({
        "role": "solver",
        "raw_output": "{}",
        "protocol_valid": True,
        "compile_success": True,
    })
    telemetry.record_graph_communication({
        "candidate_tokens": 20,
        "sent_tokens": 10,
        "total_comm_tokens": 10,
        "core_comm_tokens": 3,
        "delta_comm_tokens": 2,
        "verification_comm_tokens": 1,
        "quality_comm_tokens": 1,
        "control_comm_tokens": 3,
        "unique_comm_tokens": 7,
        "repeated_comm_tokens": 3,
        "receiver_seen_hit_count": 2,
        "state_delta_tokens": 10,
        "full_state_equivalent_tokens": 20,
        "revision_success_count": 1,
        "revision_regression_count": 0,
        "early_stop_round": 1,
        "feedback_sent_tokens": 5,
        "feedback_newly_rendered_tokens": 4,
        "communication_token_accounting_ok": True,
        "communication_token_breakdown_ok": True,
        "feedback_render_accounting_ok": True,
        "unique_repeated_accounting_ok": True,
    })

    summary = telemetry.finish(status="success")

    assert summary["reasoning_round_count"] == 1
    assert summary["communication_round_count"] == 1
    assert summary["total_comm_tokens"] == 10
    assert summary["duplicate_ratio"] == 0.3
    assert summary["incremental_saving"] == 0.5
    assert summary["receiver_seen_hit_count"] == 2
    assert summary["revision_success_count"] == 1
    assert summary["revision_regression_count"] == 0
    assert summary["early_stop_round"] == 1
    assert summary["communication_token_breakdown_ok"] is True
    assert summary["feedback_render_accounting_ok"] is True
    assert summary["unique_repeated_accounting_ok"] is True
    assert summary["communication_token_accounting_errors"] == 0
    assert summary["communication_token_breakdown_errors"] == 0
    assert summary["feedback_render_accounting_errors"] == 0
    assert summary["unique_repeated_accounting_errors"] == 0


def test_telemetry_records_phase8_context_reuse_cost_split(tmp_path):
    telemetry = WorkflowTelemetry(tmp_path / "phase8.jsonl")
    for _ in range(2):
        telemetry.record_action({
            "role": "solver",
            "mode": "normal",
            "raw_output": "{}",
            "protocol_valid": True,
            "compile_success": True,
            "logical_input_tokens": 30,
            "physical_input_tokens": 30,
            "output_tokens": 4,
            "persistent_context_tokens": 8,
            "incremental_context_tokens": 7,
            "logical_communication_tokens": 7,
            "system_prompt_tokens": 5,
            "prompt_wrapper_tokens": 10,
            "reusable_prefix_tokens": 13,
            "reusable_prefix_key": "solver:normal:numeric_solve",
        })

    summary = telemetry.summary()

    assert summary["physical_llm_input_tokens"] == 60
    assert summary["prefill_cost_tokens"] == 60
    assert summary["decode_cost_tokens"] == 8
    assert summary["persistent_context_tokens"] == 16
    assert summary["incremental_context_tokens"] == 14
    assert summary["logical_communication_tokens"] == 14
    assert summary["unique_prefix_tokens"] == 13
    assert summary["repeated_prefix_tokens"] == 13
    assert summary["simulated_context_reuse_input_tokens"] == 47
    assert summary["context_reuse_saving_ratio"] == 13 / 60
    assert summary["logical_vs_physical_input_gap_tokens"] == 46
    assert summary["phase8_total_cost_tokens"] == 68


def test_telemetry_phase7_accounting_errors_are_counted(tmp_path):
    telemetry = WorkflowTelemetry(tmp_path / "phase7_bad.jsonl")
    telemetry.record_graph_communication({
        "total_comm_tokens": 10,
        "core_comm_tokens": 1,
        "delta_comm_tokens": 1,
        "verification_comm_tokens": 1,
        "quality_comm_tokens": 1,
        "control_comm_tokens": 1,
        "unique_comm_tokens": 4,
        "repeated_comm_tokens": 4,
        "feedback_sent_tokens": 1,
        "feedback_newly_rendered_tokens": 2,
        "communication_token_accounting_ok": False,
        "communication_token_breakdown_ok": False,
        "feedback_render_accounting_ok": False,
        "unique_repeated_accounting_ok": False,
    })

    summary = telemetry.summary()

    assert summary["communication_token_accounting_errors"] == 1
    assert summary["communication_token_breakdown_errors"] == 1
    assert summary["feedback_render_accounting_errors"] == 1
    assert summary["unique_repeated_accounting_errors"] == 1
    assert summary["communication_token_breakdown_ok"] is False
    assert summary["feedback_render_accounting_ok"] is False
    assert summary["unique_repeated_accounting_ok"] is False
