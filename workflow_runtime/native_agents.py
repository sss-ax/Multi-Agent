"""Agent runtime primitives for the natural-language LangGraph workflow.

The native workflow deliberately keeps its messages unstructured, but that does
not mean that roles have to be anonymous prompt invocations.  This module
provides the runtime boundary for an agent: identity, configuration, private
memory, tool permissions and lifecycle state are all owned by the agent.

Agents may share a model provider (which is useful when one model is loaded in
GPU memory), while still having isolated runtime state and independently
replaceable model callables.  Passing a different ``model`` to each agent gives
fully separate backends when that is desired.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

from .message_bus import MessageBus, MessageEnvelope
from .tools import ToolError, ToolRegistry


@dataclass(frozen=True)
class AgentConfig:
    """Immutable policy and identity configuration for one agent."""

    agent_id: str
    role: str
    system_prompt: str
    tool_permissions: frozenset[str] = frozenset()
    memory_window: int = 12
    max_turns: int = 32
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.agent_id.strip():
            raise ValueError("agent_id must be non-empty")
        if not self.role.strip():
            raise ValueError("role must be non-empty")
        if not self.system_prompt.strip():
            raise ValueError("system_prompt must be non-empty")
        if self.memory_window < 0:
            raise ValueError("memory_window must be non-negative")
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive")


@dataclass
class AgentMemory:
    """Private append-only short-term memory for one agent."""

    messages: list[dict[str, str]] = field(default_factory=list)
    turn_count: int = 0

    def remember(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def recent(self, limit: int) -> list[dict[str, str]]:
        if limit <= 0:
            return []
        return list(self.messages[-limit:])

    def clear(self) -> None:
        self.messages.clear()
        self.turn_count = 0


@dataclass
class AgentLifecycle:
    """Small explicit lifecycle state machine for an agent instance."""

    state: str = "created"
    started_at: float | None = None
    last_active_at: float | None = None
    completed_at: float | None = None
    failed_at: float | None = None
    error: str | None = None

    def activate(self) -> None:
        now = time.time()
        if self.state == "created":
            self.started_at = now
        elif self.state == "failed":
            raise RuntimeError("failed agent cannot be activated")
        self.state = "active"
        self.last_active_at = now
        self.completed_at = None

    def suspend(self) -> None:
        if self.state != "active":
            raise RuntimeError(f"cannot suspend agent in state {self.state}")
        self.state = "suspended"

    def complete(self) -> None:
        if self.state not in {"created", "active", "suspended"}:
            raise RuntimeError(f"cannot complete agent in state {self.state}")
        self.state = "completed"
        self.completed_at = time.time()

    def fail(self, error: BaseException | str) -> None:
        self.state = "failed"
        self.failed_at = time.time()
        self.error = str(error)


@dataclass
class NativeAgent:
    """One independently addressable agent in the native workflow.

    ``model`` is intentionally per-agent.  The default workflow wires the
    same provider into several agents to avoid loading duplicate model weights,
    but callers can inject a distinct callable for any role.
    """

    config: AgentConfig
    model: Callable[[Any], Any]
    tools: ToolRegistry = field(default_factory=ToolRegistry.default)
    memory: AgentMemory = field(default_factory=AgentMemory)
    lifecycle: AgentLifecycle = field(default_factory=AgentLifecycle)

    @property
    def agent_id(self) -> str:
        return self.config.agent_id

    @property
    def role(self) -> str:
        return self.config.role

    @property
    def session_id(self) -> str:
        return f"agent:{self.agent_id}"

    @property
    def tool_permissions(self) -> frozenset[str]:
        return self.config.tool_permissions

    def can_use_tool(self, name: str) -> bool:
        return name in self.tool_permissions

    def execute_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        if not self.can_use_tool(name):
            raise ToolError(f"agent {self.agent_id} is not allowed to use tool {name}")
        return self.tools.execute(name, arguments)

    def send_message(
        self,
        bus: MessageBus,
        recipient_id: str,
        content: str,
        *,
        message_type: str = "text",
        reply_to: str | None = None,
        correlation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MessageEnvelope:
        """Send a direct message using this agent's identity."""
        return bus.send(
            self.agent_id,
            recipient_id,
            content,
            message_type=message_type,
            reply_to=reply_to,
            correlation_id=correlation_id,
            metadata=metadata,
        )

    def broadcast(
        self,
        bus: MessageBus,
        content: str,
        *,
        recipients: list[str] | tuple[str, ...] | None = None,
        message_type: str = "broadcast",
        reply_to: str | None = None,
        correlation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MessageEnvelope:
        """Broadcast a message to a selected group or every other actor."""
        return bus.broadcast(
            self.agent_id,
            content,
            recipients=recipients,
            message_type=message_type,
            reply_to=reply_to,
            correlation_id=correlation_id,
            metadata=metadata,
        )

    def receive_messages(
        self,
        bus: MessageBus,
        *,
        message_types: list[str] | tuple[str, ...] | None = None,
        limit: int | None = None,
        mark_delivered: bool = True,
    ) -> list[MessageEnvelope]:
        """Receive pending messages from this agent's mailbox."""
        return bus.receive(
            self.agent_id,
            message_types=message_types,
            limit=limit,
            mark_delivered=mark_delivered,
        )

    def render_memory(self) -> str:
        messages = self.memory.recent(self.config.memory_window)
        if not messages:
            return ""
        lines = ["AGENT MEMORY:"]
        for message in messages:
            lines.append(f"{message['role'].upper()}:\n{message['content']}")
        return "\n\n".join(lines)

    def invoke(self, request: Any) -> Any:
        if self.memory.turn_count >= self.config.max_turns:
            raise RuntimeError(f"agent {self.agent_id} exceeded max_turns={self.config.max_turns}")
        if self.lifecycle.state in {"created", "suspended", "completed"}:
            self.lifecycle.activate()
        elif self.lifecycle.state != "active":
            raise RuntimeError(f"agent {self.agent_id} is not runnable: {self.lifecycle.state}")

        prompt = str(getattr(request, "session_prompt", "") or getattr(request, "prompt", ""))
        self.memory.remember("user", prompt)
        try:
            enriched_request = replace(
                request,
                agent_id=self.agent_id,
                agent_config=self.config,
                tool_permissions=tuple(sorted(self.tool_permissions)),
            )
            raw = self.model(enriched_request)
            text = raw if isinstance(raw, str) else str(raw)
            self.memory.remember("assistant", text)
            self.memory.turn_count += 1
            self.lifecycle.last_active_at = time.time()
            return raw
        except Exception as error:
            self.lifecycle.fail(error)
            raise

    def complete(self) -> None:
        if self.lifecycle.state != "completed":
            self.lifecycle.complete()

    def snapshot(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "lifecycle": self.lifecycle.state,
            "turn_count": self.memory.turn_count,
            "memory_messages": len(self.memory.messages),
            "tool_permissions": sorted(self.tool_permissions),
            "config": {
                "memory_window": self.config.memory_window,
                "max_turns": self.config.max_turns,
                "metadata": dict(self.config.metadata),
            },
        }
