"""Allowlisted runtime tools for Action-based agents."""

from __future__ import annotations

import ast
import json
import operator
from dataclasses import dataclass
from typing import Any, Callable, Mapping


class ToolError(ValueError):
    """Raised when a tool call is unknown or violates its input contract."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    handler: Callable[[Mapping[str, Any]], Any]
    description: str = ""


class ToolRegistry:
    """Explicit allowlist of tools callable from a model Action."""

    def __init__(self, specs: list[ToolSpec] | None = None) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if not spec.name or not spec.name.strip():
            raise ValueError("tool name must be non-empty")
        if spec.name in self._specs:
            raise ValueError(f"tool already registered: {spec.name}")
        self._specs[spec.name] = spec

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def execute(self, name: str, arguments: Mapping[str, Any]) -> Any:
        spec = self._specs.get(name)
        if spec is None:
            raise ToolError(f"unknown tool: {name}")
        if not isinstance(arguments, Mapping):
            raise ToolError("tool arguments must be an object")
        try:
            return spec.handler(arguments)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"tool {name} failed: {exc}") from exc

    @classmethod
    def default(cls) -> "ToolRegistry":
        registry = cls()
        registry.register(ToolSpec(
            name="calculator",
            handler=_calculator,
            description="Evaluate a numeric arithmetic expression.",
        ))
        registry.register(ToolSpec(
            name="echo",
            handler=lambda arguments: dict(arguments),
            description="Return the supplied JSON object unchanged.",
        ))
        registry.register(ToolSpec(
            name="python_syntax_check",
            handler=_python_syntax_check,
            description="Validate Python syntax without executing the submitted code.",
        ))
        registry.register(ToolSpec(
            name="json_validate",
            handler=_json_validate,
            description="Parse and normalize a JSON document.",
        ))
        return registry


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate_numeric(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_numeric(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_numeric(node.left)
        right = _evaluate_numeric(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ToolError("calculator exponent is too large")
        result = _BINARY_OPERATORS[type(node.op)](left, right)
        if abs(result) > 10**12:
            raise ToolError("calculator result is too large")
        return result
    raise ToolError("calculator accepts only numeric arithmetic")


def _calculator(arguments: Mapping[str, Any]) -> int | float:
    expression = arguments.get("expression")
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 256:
        raise ToolError("calculator.expression must be a non-empty string up to 256 characters")
    try:
        tree = ast.parse(expression, mode="eval")
        return _evaluate_numeric(tree.body)
    except (SyntaxError, ZeroDivisionError, OverflowError) as exc:
        raise ToolError(f"invalid arithmetic expression: {exc}") from exc


def _python_syntax_check(arguments: Mapping[str, Any]) -> dict[str, Any]:
    code = arguments.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ToolError("python_syntax_check.code must be a non-empty string")
    if len(code) > 200_000:
        raise ToolError("python_syntax_check.code is too large")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ToolError(f"python syntax error at line {exc.lineno}: {exc.msg}") from exc
    return {"ok": True, "lines": len(code.splitlines()), "top_level_nodes": len(tree.body)}


def _json_validate(arguments: Mapping[str, Any]) -> Any:
    value = arguments.get("value", arguments.get("text"))
    if not isinstance(value, (str, bytes, bytearray)):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ToolError(f"invalid JSON at line {exc.lineno}, column {exc.colno}") from exc
