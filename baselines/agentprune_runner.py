"""AgentPrune-style baselines over the AutoGen round-robin workflow."""

from __future__ import annotations

import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .autogen_runner import _ModelRequest, _system_message, _task_prompt
from .common import (
    BaselineMessage,
    BaselineResult,
    BaselineTokenUsage,
    communication_token_usage,
    count_tokens,
)


@dataclass(frozen=True)
class TemporalMessageNode:
    """One node in the temporal message-passing graph consumed by pruning."""

    node_id: str
    turn: int
    sender: str
    receiver: str
    content: str
    token_cost: int
    kept: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "turn": self.turn,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "token_cost": self.token_cost,
            "kept": self.kept,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PrunedContext:
    """Rendered context and audit graph for one receiver turn."""

    receiver: str
    messages: list[BaselineMessage]
    temporal_graph: list[TemporalMessageNode]
    pruned_tokens: int
    kept_tokens: int


class AgentPruneLocalPolicy:
    """Deterministic local approximation of spatial-temporal message pruning.

    The policy keeps the task anchor, the latest solver output, and critic
    guidance for finalization while dropping older peer messages once a newer
    equivalent stage is available.  It is intentionally conservative and
    auditable; the faithful upstream runner can replace this policy later.
    """

    name = "agentprune_local_latest_stage"

    def select(self, *, receiver: str, history: list[BaselineMessage], tokenizer: Any) -> PrunedContext:
        latest_by_sender: dict[str, int] = {}
        for index, message in enumerate(history):
            latest_by_sender[message.sender] = index

        kept_indices: set[int] = set()
        reasons: dict[int, str] = {}
        for index, message in enumerate(history):
            if message.sender == "user":
                kept_indices.add(index)
                reasons[index] = "task_anchor"

        if receiver == "solver_2":
            self._keep_latest("solver_1", latest_by_sender, kept_indices, reasons, "latest_solver_input")
        elif receiver == "critic":
            self._keep_latest("solver_2", latest_by_sender, kept_indices, reasons, "latest_solver_input")
            if "solver_2" not in latest_by_sender:
                self._keep_latest("solver_1", latest_by_sender, kept_indices, reasons, "fallback_solver_input")
        elif receiver == "final_solver":
            self._keep_latest("solver_2", latest_by_sender, kept_indices, reasons, "latest_solver_input")
            self._keep_latest("critic", latest_by_sender, kept_indices, reasons, "critic_guidance")
            if "solver_2" not in latest_by_sender:
                self._keep_latest("solver_1", latest_by_sender, kept_indices, reasons, "fallback_solver_input")
        else:
            if history:
                kept_indices.add(len(history) - 1)
                reasons[len(history) - 1] = "latest_context"

        temporal_graph: list[TemporalMessageNode] = []
        kept_messages: list[BaselineMessage] = []
        kept_tokens = 0
        pruned_tokens = 0
        for index, message in enumerate(history):
            rendered = _render_message(message)
            token_cost = count_tokens(tokenizer, rendered)
            kept = index in kept_indices
            if kept:
                kept_messages.append(message)
                kept_tokens += token_cost
            else:
                pruned_tokens += token_cost
            temporal_graph.append(
                TemporalMessageNode(
                    node_id=f"m{index}",
                    turn=index,
                    sender=message.sender,
                    receiver=receiver,
                    content=message.content,
                    token_cost=token_cost,
                    kept=kept,
                    reason=reasons.get(index, "masked_by_latest_stage"),
                )
            )
        return PrunedContext(
            receiver=receiver,
            messages=kept_messages,
            temporal_graph=temporal_graph,
            pruned_tokens=pruned_tokens,
            kept_tokens=kept_tokens,
        )

    @staticmethod
    def _keep_latest(
        sender: str,
        latest_by_sender: dict[str, int],
        kept_indices: set[int],
        reasons: dict[int, str],
        reason: str,
    ) -> None:
        index = latest_by_sender.get(sender)
        if index is not None:
            kept_indices.add(index)
            reasons[index] = reason


class AgentPruneAutoGenLocalRunner:
    """AutoGen-style round-robin with a local AgentPrune message mask."""

    name = "agentprune_autogen_local"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_messages: int = 4,
        roles: tuple[str, ...] = ("solver_1", "solver_2", "critic", "final_solver"),
        policy: AgentPruneLocalPolicy | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_messages = int(max_messages)
        self.roles = roles
        self.policy = policy or AgentPruneLocalPolicy()

    def run(self, row: dict[str, Any]) -> BaselineResult:
        sample_id = str(row.get("sample_id", "sample"))
        started = time.time()
        history = [BaselineMessage(sender="user", receiver="group", content=_task_prompt(row), role="user")]
        temporal_graph_by_turn: list[dict[str, Any]] = []
        input_tokens = 0
        full_context_equivalent_input_tokens = 0
        output_tokens = 0
        for role in self.roles[: self.max_messages]:
            pruned = self.policy.select(receiver=role, history=history, tokenizer=self.tokenizer)
            prompt = self._render_prompt(role, row, pruned.messages)
            full_context_prompt = self._render_prompt(role, row, history)
            full_context_equivalent_input_tokens += count_tokens(self.tokenizer, full_context_prompt)
            output = str(
                self.model(
                    _ModelRequest(
                        prompt=prompt,
                        session_prompt=prompt,
                        session_id=f"external:agentprune:{sample_id}:{role}",
                    )
                )
            )
            input_tokens += count_tokens(self.tokenizer, prompt)
            output_tokens += count_tokens(self.tokenizer, output)
            history.append(BaselineMessage(sender=role, receiver="group", content=output))
            temporal_graph_by_turn.append({
                "receiver": role,
                "policy": self.policy.name,
                "kept_tokens": pruned.kept_tokens,
                "pruned_tokens": pruned.pruned_tokens,
                "nodes": [node.as_dict() for node in pruned.temporal_graph],
            })

        communication_tokens, communication_messages = communication_token_usage(self.tokenizer, history)
        final_answer = history[-1].content if history else ""
        system_tokens = sum(count_tokens(self.tokenizer, _system_message(role, row)) for role in self.roles)
        usage = BaselineTokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            communication_tokens=communication_tokens,
            communication_messages=communication_messages,
            forward_calls=min(self.max_messages, len(self.roles)),
            system_prompt_tokens=system_tokens,
            task_prompt_tokens=count_tokens(self.tokenizer, _task_prompt(row)),
        )
        pruned_total = sum(int(turn["pruned_tokens"]) for turn in temporal_graph_by_turn)
        kept_total = sum(int(turn["kept_tokens"]) for turn in temporal_graph_by_turn)
        input_saving_tokens = max(0, full_context_equivalent_input_tokens - input_tokens)
        return BaselineResult(
            sample_id=sample_id,
            method=self.name,
            domain=str(row.get("domain", "")),
            final_answer=final_answer,
            trace=history,
            token_usage=usage,
            raw_output=final_answer,
            latency_sec=time.time() - started,
            metadata={
                "agentprune_mode": "local",
                "agentprune_policy": self.policy.name,
                "agentprune_temporal_graph": temporal_graph_by_turn,
                "agentprune_pruned_tokens": pruned_total,
                "agentprune_kept_tokens": kept_total,
                "agentprune_prune_ratio": pruned_total / (pruned_total + kept_total) if (pruned_total + kept_total) else 0.0,
                "full_context_equivalent_input_tokens": full_context_equivalent_input_tokens,
                "agentprune_input_saving_tokens": input_saving_tokens,
                "agentprune_input_saving_ratio": (
                    input_saving_tokens / full_context_equivalent_input_tokens
                    if full_context_equivalent_input_tokens else 0.0
                ),
                "autogen_roles": list(self.roles),
                "autogen_max_messages": self.max_messages,
            },
        )

    def _render_prompt(self, role: str, row: dict[str, Any], messages: list[BaselineMessage]) -> str:
        parts = [_system_message(role, row), "", "Visible conversation:"]
        parts.extend(_render_message(message) for message in messages)
        parts.append("")
        if role == "final_solver":
            parts.append("Return only the final answer. For code tasks, return only executable Python code.")
        else:
            parts.append("Respond with your concise contribution for the next agent.")
        return "\n".join(parts)


class AgentPruneAutoGenFaithfulRunner:
    """Boundary for the vendored upstream AgentPrune implementation."""

    name = "agentprune_autogen_faithful"

    def __init__(self, *_: Any, **__: Any) -> None:
        root = _ensure_agentprune_importable()
        try:
            importlib.import_module("AgentPrune.graph.autogen_graph")
        except ImportError as exc:
            raise RuntimeError(
                "Vendored AgentPrune was found but its dependencies are missing. "
                "Install the upstream requirements into .venv, then retry "
                "--method agentprune_autogen_faithful."
            ) from exc
        raise NotImplementedError(
            "agentprune_autogen_faithful found vendored AgentPrune at "
            f"{root}, but the upstream GraphAutoGen API is not yet bound to "
            "the repository BaselineResult schema. Use agentprune_autogen_local "
            "for current experiments."
        )


def _render_message(message: BaselineMessage) -> str:
    return f"{message.sender}: {message.content}"


def _ensure_agentprune_importable() -> Path:
    """Add the script-style upstream repository to sys.path when vendored."""

    try:
        module = importlib.import_module("AgentPrune")
        origin = getattr(module, "__file__", "")
        return Path(origin).resolve().parents[1] if origin else Path("")
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "third_party" / "AgentPrune",
        repo_root.parent / "AgentPrune",
    ]
    for candidate in candidates:
        package_init = candidate / "AgentPrune" / "__init__.py"
        if package_init.exists():
            sys.path.insert(0, str(candidate))
            importlib.import_module("AgentPrune")
            return candidate
    raise RuntimeError(
        "Upstream AgentPrune is not importable. Clone it to "
        "third_party/AgentPrune or set PYTHONPATH to the cloned repository root. "
        "This repository is script-style and does not support pip install -e."
    )
