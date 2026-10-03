"""Reference-answer evaluation for completed workflow tasks."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Optional


NUMERIC_TASK_TYPES = frozenset({"numeric_solve", "numeric_comparison"})


@dataclass(frozen=True)
class EvaluationResult:
    task_type: str
    status: str
    correct: Optional[bool]
    exact_match: Optional[bool]
    normalized_match: Optional[bool]
    numeric_match: Optional[bool]
    comparison: str
    reference_available: bool
    reference_answer: Any = None
    predicted_answer: Any = None
    tolerance: float = 0.0
    reason: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_reference_answer(value: Any) -> Any:
    """Parse a CLI reference value while preserving ordinary text answers."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return re.sub(r"\s+", " ", text)


def _numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = _normalized_text(value).replace(",", "")
    text = text.rstrip(".。!?！？;；:：")
    match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", text)
    if match:
        number = float(text)
        return number if math.isfinite(number) else None

    # Native LangGraph intentionally returns ordinary prose.  For numeric
    # tasks, accept a number attached to an explicit answer/result label while
    # avoiding accidental matches from arbitrary explanatory text.
    labelled = re.findall(
        r"(?:final\s+answer|answer|result|total|答案|结果|总数)"
        r"[^0-9+\-]{0,80}"
        r"([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if not labelled:
        return None
    number = float(labelled[-1])
    return number if math.isfinite(number) else None


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, str):
        return _normalized_text(value)
    return value


class TaskEvaluator:
    """Compare a workflow's final answer with an explicit reference answer."""

    def __init__(self, task_type: str, reference_answer: Any = None, *, tolerance: float = 1e-6) -> None:
        self.task_type = task_type
        self.reference_answer = parse_reference_answer(reference_answer)
        self.has_reference = reference_answer is not None
        self.tolerance = max(0.0, float(tolerance))

    def evaluate(self, predicted_answer: Any) -> EvaluationResult:
        if not self.has_reference:
            return EvaluationResult(
                task_type=self.task_type,
                status="not_evaluated",
                correct=None,
                exact_match=None,
                normalized_match=None,
                numeric_match=None,
                comparison="none",
                reference_available=False,
                predicted_answer=predicted_answer,
                tolerance=self.tolerance,
                reason="reference answer was not provided",
            )
        if predicted_answer is None:
            return EvaluationResult(
                task_type=self.task_type,
                status="not_available",
                correct=False,
                exact_match=False,
                normalized_match=False,
                numeric_match=False if self.task_type in NUMERIC_TASK_TYPES else None,
                comparison="none",
                reference_available=True,
                reference_answer=self.reference_answer,
                tolerance=self.tolerance,
                reason="workflow did not produce a final answer",
            )

        exact_match = predicted_answer == self.reference_answer
        normalized_match = _canonical(predicted_answer) == _canonical(self.reference_answer)
        predicted_number = _numeric(predicted_answer)
        reference_number = _numeric(self.reference_answer)
        numeric_match: Optional[bool] = None
        comparison = "canonical_json" if isinstance(self.reference_answer, (dict, list, tuple)) else "normalized_text"
        if predicted_number is not None and reference_number is not None:
            numeric_match = math.isclose(
                predicted_number,
                reference_number,
                rel_tol=self.tolerance,
                abs_tol=self.tolerance,
            )
            comparison = "numeric_tolerance"
        correct = numeric_match if self.task_type in NUMERIC_TASK_TYPES and numeric_match is not None else normalized_match
        return EvaluationResult(
            task_type=self.task_type,
            status="evaluated",
            correct=correct,
            exact_match=exact_match,
            normalized_match=normalized_match,
            numeric_match=numeric_match,
            comparison=comparison,
            reference_available=True,
            reference_answer=self.reference_answer,
            predicted_answer=predicted_answer,
            tolerance=self.tolerance,
            reason=None,
        )
