"""AutoGen RoundRobin external baseline adapter."""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Any

from .common import (
    BaselineMessage,
    BaselineResult,
    BaselineTokenUsage,
    communication_token_usage,
    count_tokens,
)


@dataclass(frozen=True)
class AutoGenCall:
    prompt: str
    output: str
    input_tokens: int
    output_tokens: int


class HFBackedAutoGenModelClient:
    """Small AutoGen model-client facade over the repository model callable."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.calls: list[AutoGenCall] = []
        self._create_result_cls: Any = None
        self._request_usage_cls: Any = None
        try:
            from autogen_core.models import CreateResult, RequestUsage

            self._create_result_cls = CreateResult
            self._request_usage_cls = RequestUsage
        except ImportError:
            self._create_result_cls = None
            self._request_usage_cls = None

    async def create(self, messages: list[Any], **_: Any) -> Any:
        prompt = self._render_messages(messages)
        output = str(self.model(_ModelRequest(prompt=prompt, session_prompt=prompt, session_id="external:autogen")))
        input_tokens = count_tokens(self.tokenizer, prompt)
        output_tokens = count_tokens(self.tokenizer, output)
        self.calls.append(AutoGenCall(prompt, output, input_tokens, output_tokens))
        if self._create_result_cls is None or self._request_usage_cls is None:
            return output
        usage = self._request_usage_cls(prompt_tokens=input_tokens, completion_tokens=output_tokens)
        return self._create_result_cls(
            finish_reason="stop",
            content=output,
            usage=usage,
            cached=False,
        )

    async def close(self) -> None:
        return None

    def actual_usage(self) -> Any:
        return self._usage()

    def total_usage(self) -> Any:
        return self._usage()

    def count_tokens(self, messages: list[Any], **_: Any) -> int:
        return count_tokens(self.tokenizer, self._render_messages(messages))

    def remaining_tokens(self, messages: list[Any], **_: Any) -> int:
        return 10_000_000 - self.count_tokens(messages)

    @property
    def capabilities(self) -> dict[str, Any]:
        return {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "structured_output": False,
        }

    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "structured_output": False,
            "family": "unknown",
        }

    @staticmethod
    def _message_content(message: Any) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)

    def _render_messages(self, messages: list[Any]) -> str:
        rendered: list[str] = []
        for message in messages:
            source = getattr(message, "source", getattr(message, "role", "message"))
            rendered.append(f"{source}: {self._message_content(message)}")
        return "\n\n".join(rendered)

    def _usage(self) -> Any:
        prompt_tokens = sum(call.input_tokens for call in self.calls)
        completion_tokens = sum(call.output_tokens for call in self.calls)
        if self._request_usage_cls is None:
            return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        return self._request_usage_cls(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


@dataclass(frozen=True)
class _ModelRequest:
    prompt: str
    session_prompt: str
    session_id: str
    session_reset: bool = True
    session_rollback: bool = False


class AutoGenRoundRobinRunner:
    """Four-agent AutoGen RoundRobin workflow used as the standard MAS baseline."""

    name = "autogen_roundrobin"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_messages: int = 4,
        roles: tuple[str, ...] = ("solver_1", "solver_2", "critic", "final_solver"),
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_messages = int(max_messages)
        self.roles = roles

    def run(self, row: dict[str, Any]) -> BaselineResult:
        return asyncio.run(self._run_async(row))

    async def _run_async(self, row: dict[str, Any]) -> BaselineResult:
        sample_id = str(row.get("sample_id", "sample"))
        started = time.time()
        client = HFBackedAutoGenModelClient(self.model, self.tokenizer)
        try:
            AssistantAgent, RoundRobinGroupChat, MaxMessageTermination = _import_autogen()
            agents = [
                AssistantAgent(
                    name=role,
                    model_client=client,
                    system_message=_system_message(role, row),
                )
                for role in self.roles
            ]
            team = RoundRobinGroupChat(
                agents,
                # AutoGen counts the initial task message, so allow one extra
                # message to obtain the requested number of agent turns.
                termination_condition=MaxMessageTermination(max_messages=self.max_messages + 1),
            )
            result = await team.run(task=_task_prompt(row))
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                maybe = close()
                if inspect.isawaitable(maybe):
                    await maybe
        trace = _extract_trace(result)
        final_answer = _final_answer(result, trace)
        communication_tokens, communication_messages = communication_token_usage(self.tokenizer, trace)
        input_tokens = sum(call.input_tokens for call in client.calls)
        output_tokens = sum(call.output_tokens for call in client.calls)
        usage = BaselineTokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            communication_tokens=communication_tokens,
            communication_messages=communication_messages,
            forward_calls=len(client.calls),
            system_prompt_tokens=sum(count_tokens(self.tokenizer, _system_message(role, row)) for role in self.roles),
            task_prompt_tokens=count_tokens(self.tokenizer, _task_prompt(row)),
        )
        return BaselineResult(
            sample_id=sample_id,
            method=self.name,
            domain=str(row.get("domain", "")),
            final_answer=final_answer,
            trace=trace,
            token_usage=usage,
            raw_output=final_answer,
            latency_sec=time.time() - started,
            metadata={"autogen_max_messages": self.max_messages, "autogen_roles": list(self.roles)},
        )


def _import_autogen() -> tuple[Any, Any, Any]:
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import MaxMessageTermination
        from autogen_agentchat.teams import RoundRobinGroupChat
    except ImportError as exc:
        raise RuntimeError(
            "AutoGen is not installed. Install the optional AutoGen packages "
            "(for example autogen-agentchat and autogen-core) to run --method autogen_roundrobin."
        ) from exc
    return AssistantAgent, RoundRobinGroupChat, MaxMessageTermination


def _task_prompt(row: dict[str, Any]) -> str:
    from scripts.evaluate_domain_workflow import render_native_task

    return render_native_task(row)


def _system_message(role: str, row: dict[str, Any]) -> str:
    task_type = str(row.get("task_type", ""))
    if role == "critic":
        return (
            "You are a critic in a multi-agent benchmark workflow. Check previous reasoning, "
            "identify mistakes, and give concise repair guidance."
        )
    if role == "final_solver":
        return (
            "You are the final solver. Use the prior agents' messages and return only the final answer. "
            "For code tasks, return only executable Python code."
        )
    if task_type == "code_generation":
        return "You are a coding solver. Propose executable Python code that satisfies the tests."
    return "You are a solver in a multi-agent benchmark workflow. Solve the task carefully and concisely."


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def _extract_trace(result: Any) -> list[BaselineMessage]:
    raw_messages = list(getattr(result, "messages", []) or [])
    trace: list[BaselineMessage] = []
    previous = "user"
    for message in raw_messages:
        sender = str(getattr(message, "source", getattr(message, "role", previous)))
        content = _message_text(message)
        if not content.strip():
            continue
        trace.append(BaselineMessage(sender=sender, receiver="group", content=content))
        previous = sender
    return trace


def _final_answer(result: Any, trace: list[BaselineMessage]) -> str:
    direct = getattr(result, "final_answer", None)
    if direct is not None:
        return str(direct)
    if trace:
        return trace[-1].content
    messages = getattr(result, "messages", []) or []
    if messages:
        return _message_text(messages[-1])
    return str(result)
