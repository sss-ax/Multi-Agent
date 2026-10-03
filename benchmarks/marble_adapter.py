"""Adapters between the native workflow and the official MARBLE evaluator.

The native workflow deliberately does not depend on MARBLE's Engine.  This
module is the compatibility boundary used when an official MARBLE score is
requested:

* :class:`MARBLETrajectoryAdapter` turns message-bus envelopes into the
  communication/planning/results contract consumed by ``marble.evaluator``.
* :class:`MARBLEEnvironmentAdapter` exposes the small environment interface
  used by the official evaluator without replaying actions or starting a
  database container.
* :class:`MARBLEEvaluatorBridge` imports and invokes the upstream
  ``Evaluator`` rather than reimplementing its scoring prompts.

The bridge is intentionally opt-in.  MARBLE's evaluator uses a judge LLM and
some official environments have external side effects; a normal runtime run
must not activate either accidentally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


class MARBLEAdapterError(RuntimeError):
    """Raised when an official MARBLE evaluation cannot be constructed."""


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _compact(value: Any, limit: int = 12_000) -> str:
    text = _text(value)
    return text if len(text) <= limit else text[:limit] + "..."


@dataclass
class MARBLEAgentAdapter:
    """Minimal Agent-compatible view required by MARBLE ``Evaluator``."""

    agent_id: str
    profile: str
    role: str = "agent"
    token_usage: int = 0
    task_history: list[str] = field(default_factory=list)

    def get_token_usage(self) -> int:
        return int(self.token_usage)

    def get_profile(self) -> str:
        return self.profile


@dataclass
class MARBLEEnvironmentAdapter:
    """Environment-compatible snapshot backed by a native workflow result.

    The official evaluator only requires ``is_task_completed`` and the state
    accessors for the graph metrics.  Keeping this adapter side-effect free is
    important: communication/planning scoring must not rerun tools, web
    requests, or Docker database setup after the agent trajectory finished.
    """

    task: str
    task_type: str
    state: dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    name: str = "Native MARBLE Environment Adapter"
    current_iteration: int = 0
    max_iterations: int = 0
    done: bool = False

    def is_done(self) -> bool:
        return bool(self.done)

    def is_task_completed(self) -> bool:
        return bool(self.completed)

    def get_state(self) -> dict[str, Any]:
        return dict(self.state)

    def get_description(self) -> str:
        return self.task

    def apply_action(
        self,
        agent_id: str | None,
        action_name: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Record, but never execute, a replayed action.

        Native tool execution has already happened inside the workflow.  This
        method exists so code using the standard MARBLE environment contract
        can inspect the resulting action history without duplicating effects.
        """

        action = {
            "agent_id": agent_id,
            "action_name": action_name,
            "arguments": dict(arguments),
        }
        history = list(self.state.get("replayed_actions", []))
        history.append(action)
        self.state["replayed_actions"] = history
        self.state["last_action_result"] = action
        self.current_iteration += 1
        return action


@dataclass(frozen=True)
class MARBLETrajectory:
    """Normalized trajectory fields expected by the official graph evaluator."""

    task: str
    task_type: str
    communications: str
    summary: str
    agent_profiles: str
    agent_tasks: str
    results: str
    final_result: str
    messages: tuple[dict[str, Any], ...]
    agents: tuple[MARBLEAgentAdapter, ...]
    environment_state: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "task_type": self.task_type,
            "communications": self.communications,
            "summary": self.summary,
            "agent_profiles": self.agent_profiles,
            "agent_tasks": self.agent_tasks,
            "results": self.results,
            "final_result": self.final_result,
            "messages": [dict(message) for message in self.messages],
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "profile": agent.profile,
                    "role": agent.role,
                    "token_usage": agent.token_usage,
                }
                for agent in self.agents
            ],
            "environment_state": dict(self.environment_state),
        }


class MARBLETrajectoryAdapter:
    """Convert a native ``run_workflow`` outcome into MARBLE input fields."""

    _NON_AGENT_ACTORS = {"system", "user", "tool"}

    def __init__(self, task: Any, outcome: Mapping[str, Any]) -> None:
        self.task = task
        self.outcome = outcome

    def adapt(self) -> MARBLETrajectory:
        state = self.outcome.get("state") or {}
        raw_messages = state.get("messages") or []
        messages = tuple(
            dict(message) for message in raw_messages if isinstance(message, Mapping)
        )
        snapshots = state.get("agent_snapshots") or self.outcome.get("agents") or {}
        if not isinstance(snapshots, Mapping):
            snapshots = {}

        actor_ids = self._actor_ids(messages, snapshots)
        metadata = dict(getattr(self.task, "metadata", {}) or {})
        official_agents = metadata.get("agents", [])
        official_profiles = {
            str(item.get("agent_id")): _text(item.get("profile", ""))
            for item in official_agents
            if isinstance(item, Mapping) and item.get("agent_id") is not None
        }

        agents: list[MARBLEAgentAdapter] = []
        for actor_id in actor_ids:
            snapshot = snapshots.get(actor_id, {})
            if not isinstance(snapshot, Mapping):
                snapshot = {}
            role = str(snapshot.get("role", actor_id.rsplit(":", 1)[-1]))
            profile = _text(snapshot.get("system_prompt", ""))
            if not profile:
                profile = official_profiles.get(actor_id, f"Native workflow {role} agent")
            agents.append(
                MARBLEAgentAdapter(
                    agent_id=actor_id,
                    profile=profile,
                    role=role,
                    token_usage=self._agent_tokens(actor_id),
                )
            )

        return MARBLETrajectory(
            task=_text(getattr(self.task, "prompt", "")),
            task_type=str(getattr(self.task, "task_type", "unknown")),
            communications=self._communications(messages, actor_ids),
            summary=self._summary(state),
            agent_profiles=self._profiles(agents, official_profiles),
            agent_tasks=self._tasks(messages, agents, official_profiles),
            results=self._results(messages, actor_ids),
            final_result=_compact(self.outcome.get("final_answer")),
            messages=messages,
            agents=tuple(agents),
            environment_state=dict(state.get("environment_state") or {}),
        )

    def _actor_ids(
        self,
        messages: Sequence[Mapping[str, Any]],
        snapshots: Mapping[str, Any],
    ) -> list[str]:
        found: list[str] = []
        for message in messages:
            sender = str(message.get("sender_id", ""))
            if sender and sender not in self._NON_AGENT_ACTORS and sender not in found:
                found.append(sender)
            for recipient in message.get("recipient_ids", ()):  # type: ignore[union-attr]
                recipient = str(recipient)
                if recipient and recipient not in self._NON_AGENT_ACTORS and recipient not in found:
                    found.append(recipient)
        for actor_id in snapshots:
            actor_id = str(actor_id)
            if actor_id not in self._NON_AGENT_ACTORS and actor_id not in found:
                found.append(actor_id)
        return found

    def _agent_tokens(self, actor_id: str) -> int:
        telemetry = self.outcome.get("telemetry") or {}
        by_agent = telemetry.get("native_agent_tokens", {}) if isinstance(telemetry, Mapping) else {}
        if isinstance(by_agent, Mapping) and actor_id in by_agent:
            return int(by_agent.get(actor_id, 0) or 0)
        calls = telemetry.get("native_calls", []) if isinstance(telemetry, Mapping) else []
        total = 0
        for call in calls:
            if not isinstance(call, Mapping) or call.get("agent_id") != actor_id:
                continue
            metrics = call.get("call_metrics") or {}
            if isinstance(metrics, Mapping):
                total += int(metrics.get("output_tokens", 0) or 0)
                total += int(metrics.get("physical_input_tokens", 0) or 0)
        return total

    @staticmethod
    def _is_agent_to_agent(message: Mapping[str, Any], actor_ids: Sequence[str]) -> bool:
        sender = str(message.get("sender_id", ""))
        recipients = [str(value) for value in message.get("recipient_ids", ())]
        return sender in actor_ids and any(recipient in actor_ids for recipient in recipients)

    def _communications(
        self,
        messages: Sequence[Mapping[str, Any]],
        actor_ids: Sequence[str],
    ) -> str:
        lines: list[str] = []
        for index, message in enumerate(messages, 1):
            if not self._is_agent_to_agent(message, actor_ids):
                continue
            recipients = ", ".join(str(value) for value in message.get("recipient_ids", ()))
            lines.append(
                f"[{index}] {message.get('sender_id')} -> {recipients} "
                f"(type={message.get('message_type', 'text')})\n"
                f"{_compact(message.get('content', ''), 4000)}"
            )
        return "\n\n".join(lines) or "(No direct agent-to-agent messages were recorded.)"

    def _profiles(
        self,
        agents: Sequence[MARBLEAgentAdapter],
        official_profiles: Mapping[str, str],
    ) -> str:
        lines = [f"{agent.agent_id} ({agent.role}): {agent.profile}" for agent in agents]
        for agent_id, profile in official_profiles.items():
            if agent_id not in {agent.agent_id for agent in agents}:
                lines.append(f"{agent_id} (benchmark profile): {profile}")
        return "\n".join(lines) or "(No agent profiles were recorded.)"

    def _tasks(
        self,
        messages: Sequence[Mapping[str, Any]],
        agents: Sequence[MARBLEAgentAdapter],
        official_profiles: Mapping[str, str],
    ) -> str:
        task = _text(getattr(self.task, "runtime_prompt", lambda: getattr(self.task, "prompt", ""))())
        return "\n".join(
            f"{agent.agent_id}: assigned workflow task\n{_compact(task, 3000)}"
            for agent in agents
        ) or "(No agent assignments were recorded.)"

    def _results(self, messages: Sequence[Mapping[str, Any]], actor_ids: Sequence[str]) -> str:
        lines: list[str] = []
        for actor_id in actor_ids:
            outputs = [
                _compact(message.get("content", ""), 5000)
                for message in messages
                if message.get("sender_id") == actor_id
                and message.get("message_type") not in {"task"}
            ]
            if outputs:
                lines.append(f"{actor_id}:\n" + "\n\n".join(outputs))
        return "\n\n".join(lines) or "(No agent results were recorded.)"

    def _summary(self, state: Mapping[str, Any]) -> str:
        pieces = []
        for key in ("planner_output", "solver_output", "repair_output", "critic_output", "final_answer"):
            if state.get(key):
                pieces.append(f"{key}:\n{_compact(state[key], 5000)}")
        return "\n\n".join(pieces) or _compact(self.outcome.get("final_answer"))


def _load_marble_modules(marble_root: str | Path) -> tuple[Any, Any, Any]:
    root = Path(marble_root).resolve()
    package_root = root / "marble"
    if not package_root.is_dir():
        raise MARBLEAdapterError(f"MARBLE package not found under {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        evaluator_module = importlib.import_module("marble.evaluator.evaluator")
        environments_module = importlib.import_module("marble.environments")
        base_module = importlib.import_module("marble.environments.base_env")
    except Exception as error:  # pragma: no cover - depends on optional MARBLE deps
        raise MARBLEAdapterError(
            "official MARBLE dependencies are unavailable; install MARBLE/pyproject.toml"
        ) from error
    return evaluator_module, environments_module, base_module


def _official_environment(
    marble_root: str | Path,
    trajectory: MARBLETrajectory,
    task: Any,
) -> Any:
    _, _, base_module = _load_marble_modules(marble_root)
    base_environment = base_module.BaseEnvironment
    metadata = dict(getattr(task, "metadata", {}) or {})
    environment_config = dict(metadata.get("environment") or {})
    environment_config.update(
        {
            "task_description": trajectory.task,
            "max_iterations": environment_config.get("max_iterations") or 1,
        }
    )
    state = dict(trajectory.environment_state)
    state["task_description"] = trajectory.task
    state["final_result"] = trajectory.final_result

    class NativeEnvironment(base_environment, MARBLEEnvironmentAdapter):
        def __init__(self) -> None:
            base_environment.__init__(
                self,
                name="Native MARBLE Environment Adapter",
                config=environment_config,
            )
            self.task = trajectory.task
            self.task_type = trajectory.task_type
            self.state.update(state)
            self.completed = bool((task_result := trajectory.final_result.strip()))
            self.done = True
            self.current_iteration = 1
            self.max_iterations = int(environment_config["max_iterations"])

        def is_task_completed(self) -> bool:
            return bool(self.completed)

        def get_state(self) -> dict[str, Any]:
            return dict(self.state)

    return NativeEnvironment()


class MARBLEEvaluatorBridge:
    """Invoke the upstream MARBLE Evaluator on an adapted trajectory."""

    def __init__(
        self,
        *,
        marble_root: str | Path = "MARBLE",
        evaluator_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        self.marble_root = Path(marble_root)
        self.evaluator_factory = evaluator_factory

    def evaluate(
        self,
        task: Any,
        outcome: Mapping[str, Any],
        *,
        evaluator_model: str | None = None,
    ) -> dict[str, Any]:
        trajectory = MARBLETrajectoryAdapter(task, outcome).adapt()
        metadata = dict(getattr(task, "metadata", {}) or {})
        metrics_config = dict(metadata.get("metrics") or {})
        if evaluator_model:
            metrics_config["evaluate_llm"] = evaluator_model
        elif not metrics_config.get("evaluate_llm"):
            metrics_config["evaluate_llm"] = "gpt-4o"

        if self.evaluator_factory is not None:
            evaluator = self.evaluator_factory(metrics_config)
        else:
            evaluator_module, _, _ = _load_marble_modules(self.marble_root)
            evaluator = evaluator_module.Evaluator(metrics_config=metrics_config)

        environment = _official_environment(self.marble_root, trajectory, task)
        agents = list(trajectory.agents)
        evaluator.update(environment, agents)
        evaluator.evaluate_communication(trajectory.task, trajectory.communications)
        evaluator.evaluate_planning(
            trajectory.summary,
            trajectory.agent_profiles,
            trajectory.agent_tasks,
            trajectory.results,
        )
        evaluator.evaluate_kpi(trajectory.task, trajectory.results)
        self._evaluate_domain(evaluator, task, trajectory)
        evaluator.finalize()

        metrics = dict(getattr(evaluator, "metrics", {}) or {})
        return {
            "evaluator": "official_marble",
            "metrics": metrics,
            "communication_score": list(metrics.get("communication_score", [])),
            "planning_score": list(metrics.get("planning_score", [])),
            "total_milestones": int(metrics.get("total_milestones", 0) or 0),
            "agent_kpis": dict(metrics.get("agent_kpis", {}) or {}),
            "task_evaluation": metrics.get("task_evaluation", {}),
            "trajectory": trajectory.as_dict(),
        }

    @staticmethod
    def _evaluate_domain(evaluator: Any, task: Any, trajectory: MARBLETrajectory) -> None:
        task_type = str(getattr(task, "task_type", ""))
        if task_type == "marble_research" and hasattr(evaluator, "evaluate_task_research"):
            evaluator.evaluate_task_research(trajectory.task, trajectory.final_result)
        elif task_type == "marble_bargaining" and hasattr(evaluator, "evaluate_task_world"):
            evaluator.evaluate_task_world(trajectory.task, trajectory.final_result)
        elif task_type == "marble_database" and hasattr(evaluator, "evaluate_task_db"):
            metadata = dict(getattr(task, "metadata", {}) or {})
            evaluator.evaluate_task_db(
                trajectory.task,
                trajectory.final_result,
                list(metadata.get("labels", [])),
                int(metadata.get("number_of_labels_pred", 0) or 0),
                list(metadata.get("root_causes", [])),
            )
        elif task_type == "code_generation" and hasattr(evaluator, "evaluate_code_quality"):
            evaluator.evaluate_code_quality(trajectory.task, trajectory.final_result)


def official_trajectory_record(task: Any, outcome: Mapping[str, Any]) -> dict[str, Any]:
    """Return the serializable adapter output for trajectory persistence."""

    return MARBLETrajectoryAdapter(task, outcome).adapt().as_dict()
