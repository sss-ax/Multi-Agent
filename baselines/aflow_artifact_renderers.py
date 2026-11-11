"""Native-argument renderers for receiver-specific Artifact slices."""
from __future__ import annotations

from typing import Any, Iterable, Mapping


def _value(item: Any) -> Any:
    return getattr(item, "value", item)


def render_mbpp_scensemble(occurrences: Iterable[Any]) -> dict[str, Any]:
    """Restore native ``solutions`` while preserving occurrence order/duplicates."""
    return {"solutions": [_value(item) for item in occurrences]}


def render_mbpp_test(
    problem: str,
    entry_point: str,
    selected_code: str,
    public_tests: Any | None = None,
) -> dict[str, Any]:
    result = {"problem": problem, "entry_point": entry_point, "solution": selected_code}
    if public_tests is not None:
        result["tests"] = public_tests
    return result


def render_mbpp_fix(
    problem: str,
    failed_code: str,
    test_result: Any,
    critique: str | None = None,
) -> dict[str, Any]:
    text = f"Problem: {problem}\nFailed solution: {failed_code}\nError: {test_result}"
    if critique:
        text += f"\nCritique: {critique}"
    return {"input": text}


def render_receiver(receiver: str, values: Mapping[str, Any]) -> dict[str, Any]:
    """Small dispatcher used by tests and future profile renderers."""
    renderer = str(values.get("renderer", "native"))
    if renderer == "mbpp_ensemble":
        return render_mbpp_scensemble(values.get("occurrences", ()))
    if renderer == "mbpp_test":
        return render_mbpp_test(
            str(values.get("problem", "")), str(values.get("entry_point", "")),
            str(values.get("selected_code", "")), values.get("public_tests"),
        )
    if renderer == "mbpp_fix":
        return render_mbpp_fix(
            str(values.get("problem", "")), str(values.get("failed_code", "")),
            values.get("test_result", ""), values.get("critique"),
        )
    return dict(values.get("kwargs", {}))
