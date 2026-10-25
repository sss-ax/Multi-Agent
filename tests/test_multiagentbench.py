import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.multiagentbench import load_multiagentbench_tasks, summarize_results
from workflow_runtime.graph_store import GraphStore
from workflow_runtime.langgraph_workflow import LangGraphWorkflow


def test_official_coding_record_is_normalized(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    path.write_text(json.dumps({
        "id": 7,
        "topic_category": "Coding",
        "coordination_category": "dependency_task",
        "content": "Build a service",
        "requirements": ["It must pass tests", "It must expose an API"],
    }) + "\n", encoding="utf-8")

    task = load_multiagentbench_tasks(path)[0]
    assert task.task_id == "7"
    assert task.task_type == "code_generation"
    assert task.supported
    assert "Acceptance requirements" in task.runtime_prompt()
    assert task.milestones == ("It must pass tests", "It must expose an API")


def test_official_nested_marble_coding_record_is_normalized(tmp_path):
    path = tmp_path / "coding_main.jsonl"
    path.write_text(json.dumps({
        "scenario": "coding",
        "task_id": 1,
        "coordinate_mode": "graph",
        "task": {"content": "Software Development Task: build a service"},
        "agents": [],
    }) + "\n", encoding="utf-8")

    task = load_multiagentbench_tasks(path)[0]
    assert task.task_id == "1"
    assert task.task_type == "code_generation"
    assert task.domain == "coding"
    assert task.prompt == "Software Development Task: build a service"
    assert task.supported


def test_marble_domains_are_first_class_task_types(tmp_path):
    path = tmp_path / "tasks.jsonl"
    rows = [
        {"id": "r", "topic_category": "Research", "content": "Write a proposal"},
        {"id": "b", "topic_category": "Bargaining", "content": "Reach a deal"},
        {"id": "d", "topic_category": "Database", "content": "Diagnose a query"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    tasks = load_multiagentbench_tasks(path)
    assert [task.task_type for task in tasks] == [
        "marble_research", "marble_bargaining", "marble_database"
    ]
    assert all(task.supported for task in tasks)


def test_summary_does_not_treat_unevaluated_runs_as_correct():
    summary = summarize_results([
        {
            "execution_mode": "optimized",
            "runtime_success": True,
            "answer_status": "not_evaluated",
            "answer_correct": None,
            "physical_input_tokens": 10,
            "output_tokens": 2,
            "total_model_tokens": 12,
            "duration_sec": 1,
        },
    ])
    mode = summary["modes"]["optimized"]
    assert mode["runtime_completion_rate"] == 1.0
    assert mode["answer_accuracy"] is None
    assert mode["tokens_per_runtime_success"] == 12


def test_unregistered_marble_workflow_is_fail_closed():
    store = GraphStore()
    store.add_node(
        task_id="m",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="[domain=marble_research]\nWrite a proposal",
        owner="user",
    )
    import pytest
    with pytest.raises(ValueError, match="does not support"):
        LangGraphWorkflow(store=store, model=lambda request: {}, task_id="m", task_type="marble_research")
