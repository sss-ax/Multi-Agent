import asyncio

import pytest

from baselines.aflow_runner import _AFlowLLMBridge
from scripts.reproduce_aflow_paper import (
    _initial_run_result,
    _row_identifier,
)
from workflow_runtime.model_backend import TransformersModel


class SpaceTokenizer:
    def __call__(self, text, *, add_special_tokens=False):
        return {"input_ids": str(text).split()}


def test_transformers_backend_renders_user_chat_template():
    class ChatTokenizer:
        chat_template = "available"

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "solve this"}]
            assert tokenize is False
            assert add_generation_prompt is True
            return "<user>solve this</user><assistant>"

    backend = object.__new__(TransformersModel)
    backend.tokenizer = ChatTokenizer()
    backend.use_chat_template = True

    assert backend.render_prompt("solve this") == "<user>solve this</user><assistant>"


def test_aflow_bridge_reports_call_timing_and_timeout():
    events = []

    class TimedOutModel:
        last_call_metrics = {}

        def __call__(self, request):
            assert request.timeout_seconds == 12.5
            self.last_call_metrics = {"timed_out": True}
            return "partial answer"

    bridge = _AFlowLLMBridge(
        TimedOutModel(),
        SpaceTokenizer(),
        call_timeout_seconds=12.5,
        event_callback=events.append,
    )
    bridge.set_call_context(problem_index=3, problem_id="id:sample")

    output = asyncio.run(bridge("one two three"))
    usage = bridge.get_usage_summary()

    assert output == "partial answer"
    assert [event["event"] for event in events] == ["start", "timeout"]
    assert usage["call_count"] == 1
    assert usage["timeout_count"] == 1
    assert usage["history"][0]["problem_id"] == "id:sample"
    assert usage["history"][0]["elapsed_seconds"] >= 0


def test_aflow_bridge_logs_failed_call_before_reraising():
    events = []

    class BrokenModel:
        def __call__(self, request):
            raise RuntimeError("generation failed")

    bridge = _AFlowLLMBridge(BrokenModel(), SpaceTokenizer(), event_callback=events.append)

    with pytest.raises(RuntimeError, match="generation failed"):
        asyncio.run(bridge("prompt"))

    assert [event["event"] for event in events] == ["start", "error"]
    assert bridge.get_usage_summary()["call_count"] == 1
    assert "generation failed" in bridge.calls[0]["error"]


def test_partial_run_checkpoint_retains_row_resume_position(tmp_path):
    first_row = {"id": 7, "question": "first"}
    checkpoint = {
        "status": "running",
        "records": [["first", "answer", "gold", 1.0, 0.0]],
        "record_ids": [_row_identifier(first_row, 0)],
        "completed_rows": 1,
        "model_calls": 5,
    }

    result = _initial_run_result(
        dataset="DROP",
        split="test",
        workflow_round=3,
        graph_path="graph.py",
        data_path=tmp_path / "drop_test.jsonl",
        resume_result=checkpoint,
    )

    assert result["status"] == "running"
    assert result["completed_rows"] == 1
    assert result["record_ids"] == ["id:7"]
    assert result["model_calls"] == 5


def test_row_identifier_is_stable_without_native_id():
    row = {"problem": "2 + 2", "answer": "4"}
    assert _row_identifier(row, 0) == _row_identifier(dict(reversed(list(row.items()))), 99)
