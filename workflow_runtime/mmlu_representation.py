"""Runtime views for MMLU-Pro option hypotheses.

These views are structured renderings over candidate-local option evaluations.
They are not canonical graph nodes; they exist to give selectors a stable
option-hypothesis graph without polluting GraphStore.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class MMLUOptionHypothesisView:
    option_id: str
    option_text: str
    support_fragments: list[str]
    contradiction_fragments: list[str]
    score: float
    uncertainty: float
    provenance: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MMLUOptionHypothesisGraphView:
    candidate_id: str
    selected_option_id: str
    score_margin: float
    options: list[MMLUOptionHypothesisView]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "selected_option_id": self.selected_option_id,
            "score_margin": self.score_margin,
            "options": [option.as_dict() for option in self.options],
        }


def build_mmlu_option_hypothesis_graph_view(
    *,
    candidate_id: str,
    hypotheses: Iterable[dict[str, Any]],
    selected_option_id: str,
) -> MMLUOptionHypothesisGraphView:
    options = []
    for index, hypothesis in enumerate(hypotheses):
        option_id = str(hypothesis.get("option_id") or hypothesis.get("label") or hypothesis.get("option") or "").strip().upper()
        if not option_id:
            continue
        support = _string_list(hypothesis.get("support", hypothesis.get("support_fragments")))
        contradictions = _string_list(hypothesis.get("contradictions", hypothesis.get("contradiction_fragments")))
        support_strength = _float(hypothesis.get("support_strength"), default=min(1.0, 0.35 * len(support)))
        contradiction_strength = _float(
            hypothesis.get("contradiction_strength"),
            default=min(1.0, 0.35 * len(contradictions)),
        )
        uncertainty = _float(hypothesis.get("uncertainty"), default=0.0)
        score = support_strength - contradiction_strength - uncertainty
        options.append(
            MMLUOptionHypothesisView(
                option_id=option_id,
                option_text=str(hypothesis.get("option_text") or "").strip(),
                support_fragments=support,
                contradiction_fragments=contradictions,
                score=score,
                uncertainty=uncertainty,
                provenance={
                    "source": "candidate_option_independent_evaluation",
                    "option_index": hypothesis.get("option_index", index),
                    "support_strength": support_strength,
                    "contradiction_strength": contradiction_strength,
                },
            )
        )
    return MMLUOptionHypothesisGraphView(
        candidate_id=candidate_id,
        selected_option_id=selected_option_id,
        score_margin=_score_margin([option.score for option in options]),
        options=options,
    )


def render_mmlu_option_hypothesis_graph_view(view: MMLUOptionHypothesisGraphView | dict[str, Any]) -> str:
    payload = view.as_dict() if isinstance(view, MMLUOptionHypothesisGraphView) else dict(view)
    lines = [
        f"[CANDIDATE {payload.get('candidate_id', '')}]",
        f"selected_option: {payload.get('selected_option_id', '')}",
        f"score_margin: {payload.get('score_margin', 0.0):.3f}",
    ]
    for option in payload.get("options", []):
        if not isinstance(option, dict):
            continue
        lines.extend([
            "",
            f"Option {option.get('option_id', '')}",
            f"text: {option.get('option_text', '')}",
            f"score: {_float(option.get('score'), default=0.0):.3f}",
            f"uncertainty: {_float(option.get('uncertainty'), default=0.0):.3f}",
            "support: " + _summary(option.get("support_fragments")),
            "contradiction: " + _summary(option.get("contradiction_fragments")),
        ])
    return "\n".join(lines)


def _summary(value: Any) -> str:
    items = _string_list(value)
    return "; ".join(items[:2]) if items else "none"


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_margin(scores: list[float]) -> float:
    if len(scores) < 2:
        return 0.0
    ranked = sorted(scores, reverse=True)
    return ranked[0] - ranked[1]
