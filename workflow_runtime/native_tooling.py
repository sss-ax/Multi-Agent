"""Natural-language tool-call parsing and result models for native agents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class NativeToolCall:
    """A tool request extracted from a free-form Solver response."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    source: str = "explicit"


@dataclass(frozen=True)
class NativeToolResult:
    """Serializable result of one native tool invocation."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    ok: bool
    output: Any = None
    error: str | None = None
    attempts: int = 1
    source: str = "explicit"

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "call_id": self.call_id,
            "name": self.name,
            "arguments": dict(self.arguments),
            "ok": self.ok,
            "attempts": self.attempts,
            "source": self.source,
        }
        if self.ok:
            payload["output"] = self.output
        else:
            payload["error"] = self.error or "tool call failed"
        return payload


_TOOL_CALL_RE = re.compile(
    r"<tool_call(?:\s+name=(?:\"([^\"]+)\"|'([^']+)'|([^\s>]+)))?\s*>"
    r"\s*(.*?)\s*</tool_call>",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_tool_calls(text: str, *, call_prefix: str = "native-call") -> list[NativeToolCall]:
    """Parse explicit native tool envelopes without interpreting Action JSON."""
    calls: list[NativeToolCall] = []
    for index, match in enumerate(_TOOL_CALL_RE.finditer(text), start=1):
        named_name = next((value for value in match.groups()[:3] if value), None)
        body = match.group(4).strip()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid tool_call JSON at position {match.start()}: {exc.msg}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("tool_call payload must be a JSON object")
        name = named_name or payload.get("name")
        arguments = payload.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tool_call.name must be a non-empty string")
        if not isinstance(arguments, Mapping):
            raise ValueError("tool_call.arguments must be an object")
        calls.append(NativeToolCall(
            call_id=str(payload.get("call_id") or f"{call_prefix}-{index}"),
            name=name.strip(),
            arguments=dict(arguments),
        ))
    if not calls and "<tool_call" in text.lower():
        raise ValueError("malformed tool_call envelope or missing closing tag")
    return calls


def extract_fenced_code(text: str) -> str:
    """Extract the first fenced Python block, or return a conservative plain candidate."""
    match = re.search(r"```(?:python|py)?\s*\n?(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def infer_default_tool_calls(task_type: str, task: str, candidate: str, *, call_prefix: str) -> list[NativeToolCall]:
    """Create safe domain defaults when a Solver did not emit an explicit call."""
    if task_type == "code_generation":
        return [NativeToolCall(
            call_id=f"{call_prefix}-syntax",
            name="python_syntax_check",
            arguments={"code": extract_fenced_code(candidate)},
            source="domain_default",
        )]

    if task_type in {"numeric_solve", "numeric_comparison"}:
        expression = _find_arithmetic_expression(candidate) or _find_arithmetic_expression(task)
        if expression:
            return [NativeToolCall(
                call_id=f"{call_prefix}-calculator",
                name="calculator",
                arguments={"expression": expression},
                source="domain_default",
            )]
    return []


def _find_arithmetic_expression(text: str) -> str | None:
    match = re.search(r"(?<![\w.])[-+]?\d+(?:\s*[+\-*/×÷]\s*[-+]?\d+)+(?![\w.])", text)
    if not match:
        return None
    return match.group(0).replace("×", "*").replace("÷", "/").replace(" ", "")


def render_tool_results(results: list[NativeToolResult], candidate: str) -> str:
    """Create the Tool→Critic/Finalizer message body."""
    payload = {
        "tool_results": [result.as_dict() for result in results],
        "candidate": candidate,
    }
    return "TOOL_EXECUTION_RESULT\n" + json.dumps(payload, ensure_ascii=False, default=str, indent=2)
