from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.marble_adapter import MARBLETrajectoryAdapter, official_trajectory_record


def _task():
    return SimpleNamespace(
        prompt="Develop a shared research proposal.",
        task_type="marble_research",
        metadata={
            "agents": [
                {"agent_id": "benchmark_agent_1", "profile": "researcher"},
            ],
        },
        runtime_prompt=lambda: "Develop a shared research proposal.",
    )


def _outcome():
    planner = "native:task:planner"
    solver = "native:task:solver"
    critic = "native:task:critic"
    finalizer = "native:task:final_solver"
    messages = [
        {
            "message_id": "m1",
            "sender_id": "system",
            "recipient_ids": [planner],
            "message_type": "task",
            "content": "Develop a shared research proposal.",
        },
        {
            "message_id": "m2",
            "sender_id": planner,
            "recipient_ids": [solver],
            "message_type": "plan",
            "content": "Split literature review and proposal design.",
        },
        {
            "message_id": "m3",
            "sender_id": solver,
            "recipient_ids": [critic],
            "message_type": "candidate",
            "content": "Candidate proposal.",
        },
        {
            "message_id": "m4",
            "sender_id": "tool",
            "recipient_ids": [critic, finalizer],
            "message_type": "tool_result",
            "content": "Tool result.",
        },
    ]
    snapshots = {
        planner: {"agent_id": planner, "role": "planner", "system_prompt": "planner"},
        solver: {"agent_id": solver, "role": "solver", "system_prompt": "solver"},
        critic: {"agent_id": critic, "role": "critic", "system_prompt": "critic"},
        finalizer: {"agent_id": finalizer, "role": "final_solver", "system_prompt": "finalizer"},
    }
    return {
        "state": {
            "messages": messages,
            "agent_snapshots": snapshots,
            "planner_output": "Split literature review and proposal design.",
            "solver_output": "Candidate proposal.",
            "critic_output": "Looks good.",
            "final_answer": "Final proposal.",
            "environment_state": {"tool_call_count": 1},
        },
        "final_answer": "Final proposal.",
        "telemetry": {"native_agent_tokens": {planner: 10, solver: 20}},
    }


def test_trajectory_adapter_keeps_agent_messages_and_filters_tool_messages():
    trajectory = MARBLETrajectoryAdapter(_task(), _outcome()).adapt()

    assert "native:task:planner -> native:task:solver" in trajectory.communications
    assert "native:task:solver -> native:task:critic" in trajectory.communications
    assert "tool ->" not in trajectory.communications
    assert "Split literature review" in trajectory.results
    assert trajectory.agents[0].token_usage == 10
    assert trajectory.final_result == "Final proposal."


def test_official_trajectory_record_is_json_ready():
    record = official_trajectory_record(_task(), _outcome())

    assert record["messages"]
    assert record["environment_state"]["tool_call_count"] == 1
    assert {agent["role"] for agent in record["agents"]} == {
        "planner", "solver", "critic", "final_solver"
    }

