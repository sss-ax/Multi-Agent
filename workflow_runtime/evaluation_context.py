"""Evaluator-only state kept outside model-visible graph views."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Mapping


_FORBIDDEN_KEYS = frozenset({
    "gold_answer",
    "gold_label",
    "answer",
    "supporting_facts",
    "official_tests",
    "private_tests",
    "hidden_tests",
    "reference_answer",
    "failed_test",
    "test_expression",
    "test_source",
    "traceback",
    "stdout",
    "stderr",
})


@dataclass(frozen=True)
class EvaluationContext:
    """Gold labels and private execution material owned by the evaluator.

    This object is intentionally not a GraphStore node and has no method that
    renders its values into a model prompt.  Runtime verifiers should return
    structured summaries instead of copying private test/source text.
    """

    task_id: str = ""
    task_type: str = ""
    gold_answer: Any = None
    gold_label: Any = None
    gold_supporting_facts: tuple[Mapping[str, Any], ...] = ()
    official_tests: tuple[Any, ...] = ()
    private_tests: tuple[Any, ...] = ()
    dataset_revision: str = ""
    sample_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def scorer_payload(self) -> dict[str, Any]:
        """Return a copy for evaluator/scorer code, never for model prompts."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "gold_answer": copy.deepcopy(self.gold_answer),
            "gold_label": copy.deepcopy(self.gold_label),
            "gold_supporting_facts": copy.deepcopy(list(self.gold_supporting_facts)),
            "official_tests": copy.deepcopy(list(self.official_tests)),
            "private_tests": copy.deepcopy(list(self.private_tests)),
            "dataset_revision": self.dataset_revision,
            "sample_id": self.sample_id,
            "metadata": copy.deepcopy(dict(self.metadata)),
        }

    def model_safe_metadata(self) -> dict[str, Any]:
        """Return only non-oracle identifiers safe for run telemetry/context."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "dataset_revision": self.dataset_revision,
            "sample_id": self.sample_id,
        }

    def assert_not_in_payload(self, payload: Any) -> None:
        """Reject accidental oracle fields in a model-visible payload."""
        found: list[str] = []

        def visit(value: Any, path: str = "") -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    name = str(key).casefold()
                    child = f"{path}.{name}" if path else name
                    if name in _FORBIDDEN_KEYS:
                        found.append(child)
                    visit(item, child)
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    visit(item, f"{path}[{index}]")
            elif isinstance(value, str):
                # A value-level check catches common accidental prompt/template
                # interpolation without attempting to infer arbitrary answers.
                for candidate in (self.gold_answer, self.gold_label):
                    if candidate is not None and str(candidate) and str(candidate) in value:
                        found.append(path or "<value>")

        visit(payload)
        if found:
            raise ValueError(f"model-visible payload contains evaluator-only data: {sorted(set(found))}")

    def safe_verification_summary(
        self,
        result: Mapping[str, Any],
        *,
        private: bool = True,
    ) -> dict[str, Any]:
        """Return a model-safe verification result.

        Private-test diagnostics are deliberately lossy: they expose only
        control information and never the assertion, inputs, expected value,
        actual value, traceback, or captured output.  Public-test callers may
        opt into compact value diagnostics with ``private=False``.
        """
        summary: dict[str, Any] = {}
        for key, value in result.items():
            normalized = str(key).casefold()
            if (
                normalized in _FORBIDDEN_KEYS
                or normalized.endswith("_test_source")
                or (private and normalized in {"expected", "actual", "input", "input_repr", "error", "errors"})
            ):
                continue
            if normalized in {"expected", "actual", "input", "input_repr"}:
                text = str(value)
                summary[key] = text[:240]
                continue
            if isinstance(value, Mapping):
                summary[key] = self.safe_verification_summary(value, private=private)
            elif isinstance(value, list):
                summary[key] = [
                    self.safe_verification_summary(item, private=private)
                    if isinstance(item, Mapping)
                    else copy.deepcopy(item)
                    for item in value[:32]
                    if not private or isinstance(item, Mapping)
                ]
            else:
                summary[key] = copy.deepcopy(value)
        self.assert_not_in_payload(summary)
        return summary

    def as_safe_json(self) -> str:
        """Serialize identifiers only; useful for telemetry metadata."""
        return json.dumps(self.model_safe_metadata(), ensure_ascii=False, sort_keys=True)
