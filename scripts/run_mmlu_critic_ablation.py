"""Run an isolated MMLU-Pro Critic prompt ablation.

The experiment holds the source question, choices, and intentionally wrong
candidate fixed.  Only the Critic prompt changes.  All arms use the same
token-level audit schema, so the comparison measures semantic auditing rather
than JSON formatting reliability.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_domain_workflow import read  # noqa: E402
from workflow_runtime.action_constraints import build_action_constraint  # noqa: E402
from workflow_runtime.model_backend import TransformersModel  # noqa: E402
from workflow_runtime.semantic_actions import validate_typed_semantic_action  # noqa: E402


ARMS = ("template", "manifest", "manifest_checklist")


def _slot_values(choice: dict[str, Any]) -> list[str]:
    text = str(choice.get("text", "")).strip()
    return [part.strip() for part in re.split(r"\s*,\s*", text) if part.strip()] or [text]


def _choice_lines(choices: list[dict[str, Any]]) -> str:
    return "\n".join(f"{str(c.get('label', '')).upper()}. {c.get('text', '')}" for c in choices)


def _manifest(question: str, selected: str, choices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_choice = next(c for c in choices if str(c.get("label", "")).upper() == selected)
    values = _slot_values(selected_choice)
    blanks = re.findall(r"_{2,}", question)
    contexts: list[str] = []
    for index in range(len(values)):
        if index < len(blanks):
            contexts.append(f"blank {index + 1} in the original sentence")
        else:
            contexts.append(f"slot {index + 1} in the selected option")
    return [
        {"slot": index + 1, "context": contexts[index], "selected_term": value}
        for index, value in enumerate(values)
    ]


def _candidate(question: str, choices: list[dict[str, Any]]) -> dict[str, Any]:
    selected = "A"
    values = _manifest(question, selected, choices)
    return {
        "answer": selected,
        "slot_analysis": [
            {"slot": item["slot"], "selected_term": item["selected_term"], "reason": "the selected term fits the question"}
            for item in values
        ],
        "eliminated_options": {
            str(choice.get("label", "")).upper(): "the option conflicts with the question"
            for choice in choices
            if str(choice.get("label", "")).upper() not in {"A", ""}
        },
    }


def _prompt(arm: str, question: str, choices: list[dict[str, Any]], candidate: dict[str, Any]) -> str:
    selected = str(candidate["answer"]).upper()
    slots = candidate["slot_analysis"]
    selected_lines = "\n".join(
        f"  slot {item['slot']}: selected_term={item['selected_term']!r}; Solver reason={item['reason']!r}"
        for item in slots
    )
    options = _choice_lines(choices)
    common = (
        "You are the Critic in a multiple-choice audit.\n"
        "Audit the Solver's selected option against the original question and every visible choice.\n"
        "Do not trust the Solver's verdict. Do not produce a replacement answer.\n\n"
        f"ORIGINAL QUESTION:\n{question}\n\nCHOICES:\n{options}\n\n"
        f"SOLVER CANDIDATE: answer={selected}\n{selected_lines}\n\n"
        "Return exactly one audit_choice JSON object. Use claim_supported when the claim is correct, "
        "claim_contradicted when it is false, and insufficient only when the source is genuinely inadequate."
    )
    if arm == "template":
        return common + (
            "\nUse this generic shape as a guide: slot reasons should explain whether each term matches the question."
        )
    if arm == "manifest":
        return common + (
            "\nThe following source-derived audit manifest is authoritative; cite the exact selected_term in "
            "each slot reason:\n" + json.dumps(slots, ensure_ascii=False, indent=2) +
            "\nFor the selected option, explicitly assess every slot before deciding its overall status."
        )
    return common + (
        "\nAUDIT CHECKLIST:\n"
        "1. For each slot, compare selected_term with the exact sentence context.\n"
        "2. Write the slot number and exact selected_term in the reason.\n"
        "3. Independently decide whether the selected option is supported; a well-formed JSON candidate is not evidence of correctness.\n"
        "4. Check each eliminated option against its option text and cite a distinctive term.\n"
        "5. If any selected slot is false, mark that slot claim_contradicted and mark the selected option claim_contradicted.\n"
        "SOURCE-DERIVED AUDIT MANIFEST:\n" + json.dumps(slots, ensure_ascii=False, indent=2)
    )


def _parse_action(raw: str) -> dict[str, Any] | None:
    text = str(raw).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _score(action: dict[str, Any] | None, *, candidate: dict[str, Any], gold: str) -> dict[str, Any]:
    if action is None:
        return {"parse_valid": False, "typed_valid": False}
    errors = validate_typed_semantic_action("critic", "multiple_choice", action)
    slots = action.get("slot_checks", [])
    slot_map = {int(item.get("slot")): item for item in slots if isinstance(item, dict) and isinstance(item.get("slot"), int)}
    exact_terms = {
        int(item["slot"]): str(item["selected_term"]).lower()
        for item in candidate["slot_analysis"]
    }
    cited = sum(
        1 for slot, term in exact_terms.items()
        if term and term in str(slot_map.get(slot, {}).get("reason", "")).lower()
    )
    selected = action.get("selected_option_check", {})
    return {
        "parse_valid": True,
        "typed_valid": not errors,
        "validation_errors": errors,
        "selected_status": selected.get("status"),
        "slot_statuses": {str(slot): item.get("status") for slot, item in sorted(slot_map.items())},
        "slot_count": len(slot_map),
        "slot_exact_term_citations": cited,
        "slot_exact_term_citation_rate": cited / len(exact_terms) if exact_terms else 0.0,
        "selected_is_gold": str(candidate["answer"]).upper() == str(gold).upper(),
        "selected_should_be_contradicted": str(candidate["answer"]).upper() != str(gold).upper(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated MMLU-Pro Critic prompt ablation")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", default=".runtime/mmlu_critic_ablation.json")
    args = parser.parse_args()

    rows = read(Path(args.data_path), "mmlu_pro", args.sample_index + 1)
    if args.sample_index >= len(rows):
        raise SystemExit(f"sample index {args.sample_index} is outside loaded rows ({len(rows)})")
    row = rows[args.sample_index]
    choices = list(row.get("choices", []))
    question = str(row.get("question_text", row.get("question", "")))
    candidate = _candidate(question, choices)
    slot_count = len(candidate["slot_analysis"])
    labels = [str(c.get("label", "")).upper() for c in choices if str(c.get("label", "")).strip()]
    eliminated = [label for label in labels if label != candidate["answer"]][:4]
    model = TransformersModel(args.model_path, max_new_tokens=args.max_new_tokens)
    arms: dict[str, Any] = {}
    for arm in ARMS:
        prompt = _prompt(arm, question, choices, candidate)
        constraint = build_action_constraint(
            role="critic", task_type="multiple_choice", missing=["audit result"],
            allowed_choice_labels=labels, required_audit_slots=range(1, slot_count + 1),
            required_audit_options=eliminated, protocol_mode="semantic_delta_v2",
        )
        request = SimpleNamespace(prompt=prompt, session_prompt=prompt, action_constraint=constraint)
        started = time.perf_counter()
        raw = model(request)
        elapsed = time.perf_counter() - started
        action = _parse_action(raw)
        arms[arm] = {
            "elapsed_sec": elapsed,
            "raw_output": raw,
            "action": action,
            "score": _score(action, candidate=candidate, gold=str(row.get("gold_answer", ""))),
            "model_metrics": dict(getattr(model, "last_call_metrics", {})),
        }
    result = {
        "experiment": "mmlu_critic_prompt_ablation",
        "sample_id": row.get("sample_id", args.sample_index),
        "gold_answer": row.get("gold_answer"),
        "candidate": candidate,
        "arms": arms,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "arms": {name: arm["score"] for name, arm in arms.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
