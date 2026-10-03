"""Structured Critic verdicts for the natural-language workflow."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping


VERDICTS = frozenset({"pass", "needs_repair", "reject", "uncertain"})
REPAIR_VERDICTS = frozenset({"needs_repair", "reject"})
REPAIR_TARGETS = frozenset({"candidate", "tool_result", "calculation", "evidence", "final_answer"})


@dataclass(frozen=True)
class CriticVerdict:
    """Validated control signal emitted by Critic."""

    verdict: str
    confidence: float
    target: str
    issues: tuple[str, ...]
    repair_instructions: str
    valid: bool = True
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "target": self.target,
            "issues": list(self.issues),
            "repair_instructions": self.repair_instructions,
            "valid": self.valid,
            "error": self.error,
        }


_VERDICT_RE = re.compile(
    r"<critic_verdict>\s*(.*?)\s*</critic_verdict>",
    flags=re.IGNORECASE | re.DOTALL,
)


def invalid_verdict(error: str) -> CriticVerdict:
    return CriticVerdict(
        verdict="uncertain",
        confidence=0.0,
        target="candidate",
        issues=(),
        repair_instructions="",
        valid=False,
        error=error,
    )


def parse_critic_verdict(text: str) -> CriticVerdict:
    """Parse and validate the Critic control envelope.

    Natural-language Critic text is intentionally ignored for routing.  Only
    the explicit ``critic_verdict`` JSON object can request a repair.
    """
    match = _VERDICT_RE.search(text)
    if match is None:
        return invalid_verdict("missing <critic_verdict> envelope")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        return invalid_verdict(f"invalid critic verdict JSON: {exc.msg}")
    if not isinstance(payload, Mapping):
        return invalid_verdict("critic verdict must be a JSON object")

    verdict = payload.get("verdict")
    confidence = payload.get("confidence")
    target = payload.get("target", "candidate")
    issues = payload.get("issues", [])
    instructions = payload.get("repair_instructions", "")
    if verdict not in VERDICTS:
        return invalid_verdict(f"verdict must be one of {sorted(VERDICTS)}")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return invalid_verdict("confidence must be a number")
    if not 0.0 <= float(confidence) <= 1.0:
        return invalid_verdict("confidence must be between 0 and 1")
    if not isinstance(target, str) or target not in REPAIR_TARGETS:
        return invalid_verdict(f"target must be one of {sorted(REPAIR_TARGETS)}")
    if not isinstance(issues, list) or not all(isinstance(item, str) for item in issues):
        return invalid_verdict("issues must be an array of strings")
    if not isinstance(instructions, str):
        return invalid_verdict("repair_instructions must be a string")
    if verdict in REPAIR_VERDICTS and not instructions.strip():
        return invalid_verdict("repair verdict requires repair_instructions")
    return CriticVerdict(
        verdict=verdict,
        confidence=float(confidence),
        target=target,
        issues=tuple(issues),
        repair_instructions=instructions,
    )


def render_critic_feedback(verdict: CriticVerdict, raw_text: str) -> str:
    """Render a structured Critic→Solver/Finalizer message."""
    return (
        "CRITIC_VERDICT\n"
        + json.dumps(verdict.as_dict(), ensure_ascii=False, indent=2)
        + "\nCRITIC_NARRATIVE\n"
        + raw_text
    )
