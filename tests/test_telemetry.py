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
