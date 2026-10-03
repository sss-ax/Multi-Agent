"""State-aware, token-level constraints for incremental Action decoding."""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .protocol import ROLE_ACTIONS


_FACT_PATTERNS = (
    re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s+has\s+[-+]?\d+(?:\.\d+)?", re.IGNORECASE),
    re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*(?:=|is)\s*[-+]?\d+(?:\.\d+)?", re.IGNORECASE),
)


def explicit_fact_ids(task_text: str) -> tuple[str, ...]:
    found: list[str] = []
    for pattern in _FACT_PATTERNS:
        for match in pattern.findall(task_text):
            if match not in found:
                found.append(match)
    return tuple(found)


def explicit_fact_values(task_text: str) -> dict[str, int | float]:
    values: dict[str, int | float] = {}
    patterns = (
        re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s+has\s+([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
        re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*(?:=|is)\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    )
    for pattern in patterns:
        for fact_id, raw_value in pattern.findall(task_text):
            values.setdefault(fact_id, float(raw_value) if "." in raw_value else int(raw_value))
    return values


@dataclass
class ActionConstraint:
    """Constraint that hard-masks the operation prefix during token decoding.

    The operation prefix is generated from tokenizer token IDs rather than
    decoded characters. This makes the constraint compatible with BPE/Sentence
    Piece tokenizers and prevents the model from entering a forbidden Action
    branch. ActionCompiler remains responsible for validating values and exact
    fields after decoding.
    """

    role: str
    task_type: str
    allowed_ops: tuple[str, ...]
    reason: str
    required_fact_ids: tuple[str, ...] = ()
    fixed_actions: tuple[str, ...] = ()
    _operation_paths: tuple[tuple[int, ...], ...] = field(default=(), init=False, repr=False)
    _done_path: tuple[int, ...] = field(default=(), init=False, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "task_type": self.task_type,
            "allowed_ops": list(self.allowed_ops),
            "reason": self.reason,
            "required_fact_ids": list(self.required_fact_ids),
            "fixed_action_count": len(self.fixed_actions),
            "mode": "fixed-action" if self.fixed_actions else "operation-prefix-constrained",
        }

    def bind(self, tokenizer: Any) -> None:
        paths: list[tuple[int, ...]] = []
        if self.fixed_actions:
            paths.extend(tuple(self._encode(tokenizer, action)) for action in self.fixed_actions)
        else:
            for op in self.allowed_ops:
                literal = '{"op":"done"}' if op == "done" else f'{{"op":"{op}"'
                paths.append(tuple(self._encode(tokenizer, literal)))
        self._operation_paths = tuple(paths)
        self._done_path = next(
            (path for op, path in zip(self.allowed_ops, paths) if op == "done"),
            (),
        )
        if self.fixed_actions and "done" in self.allowed_ops and paths:
            self._done_path = paths[0]

    @staticmethod
    def _encode(tokenizer: Any, text: str) -> list[int]:
        encoded = tokenizer(text, add_special_tokens=False)
        ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(item) for item in ids]

    def is_complete(self, generated_token_ids: Iterable[int]) -> bool:
        generated = tuple(int(item) for item in generated_token_ids)
        if self.fixed_actions:
            return generated in self._operation_paths
        return bool(self._done_path) and generated == self._done_path

    def mask_logits(self, logits: Any, generated_token_ids: Iterable[int]) -> Any:
        """Mask the next token to a compatible operation-prefix token.

        If the tokenizer was not bound or generation has already passed the
        operation prefix, logits are left unchanged so value generation remains
        model-controlled. The guard leaves logits unchanged rather than
        producing an all ``-inf`` distribution when no compatible prefix exists.
        """
        if not self._operation_paths:
            return logits
        generated = tuple(int(item) for item in generated_token_ids)
        compatible = [path for path in self._operation_paths if path[: len(generated)] == generated]
        if not compatible or any(len(path) == len(generated) for path in compatible):
            return logits
        next_ids = {path[len(generated)] for path in compatible if len(path) > len(generated)}
        if not next_ids:
            return logits
        masked = logits.clone()
        masked[...] = -self._negative_infinity(masked)
        for token_id in next_ids:
            if 0 <= token_id < masked.shape[-1]:
                masked[..., token_id] = logits[..., token_id]
        return masked

    @staticmethod
    def _negative_infinity(logits: Any) -> Any:
        # torch tensors expose finfo through their dtype; keeping this generic
        # allows graph-only tests to use a lightweight tensor substitute.
        try:
            import torch

            return torch.inf
        except ImportError:  # pragma: no cover
            return float("inf")


def build_action_constraint(
    *,
    role: str,
    task_type: str,
    missing: Iterable[str],
    task_text: str = "",
    existing_logical_ids: Iterable[str] = (),
) -> ActionConstraint:
    """Build the legal next-operation set from the current graph boundary."""
    missing_set = set(missing)
    existing = set(existing_logical_ids)
    required_facts = explicit_fact_ids(task_text)
    fact_values = explicit_fact_values(task_text)
    existing_fact_ids = {
        logical_id.removeprefix("fact_")
        for logical_id in existing
        if logical_id.startswith("fact_")
    }
    fixed_actions: tuple[str, ...] = ()

    if role == "planner":
        if "missing query_spec" in missing_set:
            allowed, reason = ("declare_query",), "query_spec is the first missing planner boundary"
            query_text = task_text if len(task_text) <= 512 else "task"
            fixed_actions = (
                json.dumps(
                    {"op": "declare_query", "content": {"question": query_text, "task_type": task_type}},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        elif task_type in {"numeric_solve", "numeric_comparison"} and (
            "missing facts" in missing_set
            or required_facts and not set(required_facts).issubset(existing_fact_ids)
        ):
            allowed, reason = ("add_fact",), "explicit task facts are still missing"
            missing_fact_ids = [fact_id for fact_id in required_facts if fact_id not in existing_fact_ids]
            fixed_actions = tuple(
                json.dumps(
                    {"op": "add_fact", "id": fact_id, "value": fact_values[fact_id]},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                for fact_id in missing_fact_ids
                if fact_id in fact_values
            )
        elif "missing plan" in missing_set or "missing plan_steps" in missing_set:
            allowed, reason = ("add_plan_step",), "the workflow plan is missing"
            if task_type in {"numeric_solve", "numeric_comparison"}:
                existing_numeric_facts = sorted(
                    fact_id for fact_id in existing_fact_ids
                    if re.fullmatch(r"N\d+", fact_id)
                )
                inputs = list(required_facts) or existing_numeric_facts or ["A", "B"]
                step_id, operation = "R1", "solve_numeric"
            else:
                inputs = ["task"]
                step_id, operation = "M1", "synthesize"
            fixed_actions = (
                json.dumps(
                    {"op": "add_plan_step", "id": step_id, "operation": operation, "inputs": inputs},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        elif not missing_set:
            allowed, reason = ("done",), "all planner boundaries are complete"
            fixed_actions = ('{"op":"done"}',)
        else:
            # Other domains have multiple possible evidence/artifact actions;
            # retain the role allowlist while still preventing done too early.
            allowed = tuple(sorted(ROLE_ACTIONS[role] - {"done"}))
            reason = "domain-specific planner boundary is incomplete"
            fixed_actions = ()
    elif role == "solver" and task_type in {"numeric_solve", "numeric_comparison"}:
        if "missing calculation" in missing_set:
            allowed, reason = ("calculate",), "calculation is missing"
        elif "missing result" in missing_set:
            allowed, reason = ("set_result",), "result is missing after calculation"
        else:
            allowed, reason = ("done",), "solver boundaries are complete"
    elif role == "solver" and task_type == "code_generation":
        if "missing code" in missing_set:
            allowed, reason = ("emit_code",), "the coding artifact is missing"
        else:
            allowed, reason = ("done",), "the coding artifact is complete"
    elif role == "solver" and task_type in {"multiple_choice", "multihop_qa", "marble_research", "marble_bargaining", "marble_database"}:
        if "missing result" in missing_set:
            allowed, reason = ("set_result",), "the domain result artifact is missing"
        else:
            allowed, reason = ("done",), "solver boundaries are complete"
    elif role == "critic":
        allowed, reason = (("verify",), "verification is missing") if missing_set else (("done",), "verification is complete")
    elif role == "final_solver":
        allowed, reason = (("answer",), "final answer is missing") if missing_set else (("done",), "final answer is complete")
    else:
        allowed = tuple(sorted(ROLE_ACTIONS.get(role, ())))
        reason = "role default Action allowlist"

    return ActionConstraint(
        role=role,
        task_type=task_type,
        allowed_ops=tuple(allowed),
        reason=reason,
        required_fact_ids=required_facts,
        fixed_actions=fixed_actions,
    )
