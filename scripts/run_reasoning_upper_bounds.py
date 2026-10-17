"""Run reasoning upper-bound experiments for repairable benchmark domains.

This script intentionally bypasses the graph communication workflow.  It asks:

* can the backbone produce a correct candidate if sampled N times?
* can a deterministic/oracle selector pick a correct candidate?
* can structured verifier feedback repair an initially wrong candidate?
* how far can repeated full-context repair go before communication compression?
* can MMLU-Pro/HotpotQA join the same deployable selector-communication
  schema without gold answers in the selector path?

Supported domains are gsm8k, humaneval, mbpp, mmlu_pro, and hotpotqa.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.evaluate_domain_workflow import (  # noqa: E402
    extract_code,
    extract_choice,
    extract_numeric_answer,
    hotpot_normalize,
    read,
    render_native_task,
    score_prediction,
)
from workflow_runtime.domain_executors import execute_python_tests_detailed  # noqa: E402
from workflow_runtime.candidate_graph import (  # noqa: E402
    add_candidate_selection_decision,
    add_verifier_guided_selection,
    coverage_at_k,
    materialize_candidate_pool,
)
from workflow_runtime.candidate_selector import select_candidate_with_deployable_verifier  # noqa: E402
from workflow_runtime.candidate_selector_communication import (  # noqa: E402
    build_candidate_fragments,
    rendered_fragment_tokens,
    run_selector_with_fragments,
    select_fragment_ids,
)
from workflow_runtime.graph_store import GraphStore  # noqa: E402
from workflow_runtime.hotpot_bridge import (  # noqa: E402
    bridge_candidates_from_values,
    normalize_bridge,
)
from workflow_runtime.hotpot_representation import (  # noqa: E402
    HotpotEndpointBinding,
    HotpotFactEndpoint,
    build_hotpot_chain_view_from_facts,
    build_hotpot_reasoning_candidate_object,
    render_hotpot_structured_candidate_view,
)
from workflow_runtime.hotpot_topology import ReasoningCandidate, ReasoningIR, VerificationResult, clamp_score  # noqa: E402
from workflow_runtime.mmlu_representation import build_mmlu_option_hypothesis_graph_view  # noqa: E402
from scripts.evaluate_deployable_candidate_selector import code_features, numeric_features  # noqa: E402


DOMAINS = ("gsm8k", "humaneval", "mbpp", "mmlu_pro", "hotpotqa")
CODE_DOMAINS = {"humaneval", "mbpp"}
SELECTION_ONLY_DOMAINS = {"mmlu_pro", "hotpotqa"}
MMLU_OPTION_PAIRWISE_MARGIN_THRESHOLD = 0.15
MMLU_OPTIONWISE_MODE = "independent"
HOTPOT_CANDIDATE_MODE = "default"
SELECTOR_COMMUNICATION_POLICIES = (
    "deployable_full",
    "selector_send_all",
    "selector_core_only",
    "selector_verifier_only",
    "selector_receiver_aware",
    "selector_random_same_budget",
)


@dataclass
class Candidate:
    text: str
    score: dict[str, Any]
    input_tokens: int
    output_tokens: int
    latency_sec: float
    seed: int
    repair_round: int = 0
    feedback: dict[str, Any] | None = None
    generation_role: str = ""
    redundant_candidate: bool = False
    redundancy_reason: str = ""
    rationale_similarity_to_previous: float = 0.0


@dataclass(frozen=True)
class AdaptiveDecision:
    action: str
    reason: str
    should_stop: bool = False


@dataclass(frozen=True)
class BridgeCandidate:
    name: str
    source_type: str
    source_node_id: str
    question_relevance: float
    connectivity_score: float
    evidence_support: float
    model_score: float = 0.0

    @property
    def score(self) -> float:
        return self.question_relevance + self.connectivity_score + self.evidence_support + self.model_score

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_type": self.source_type,
            "source_node_id": self.source_node_id,
            "question_relevance": self.question_relevance,
            "connectivity_score": self.connectivity_score,
            "evidence_support": self.evidence_support,
            "model_score": self.model_score,
            "score": self.score,
        }


@dataclass
class HotpotEvidenceIndex:
    by_title: dict[str, list[dict[str, Any]]]
    by_entity: dict[str, list[dict[str, Any]]]
    by_relation: dict[str, list[dict[str, Any]]]
    by_entity_relation: dict[tuple[str, str], list[dict[str, Any]]]
    all_facts: list[dict[str, Any]]


@dataclass(frozen=True)
class RelationSemantics:
    predicate: str
    subject: str
    object: str
    direction: str = "anchor_to_target"
    object_role: str = "UNKNOWN"
    confidence: float = 0.0


@dataclass(frozen=True)
class HotpotConstraint:
    anchor: str
    relation: str
    direction: str = "anchor_to_target"
    target_role: str = "ANY_RESOLVED"
    source_var: str = ""
    target_var: str = "X"
    depends_on: str = ""
    context_title: str = ""
    resolution_type: str = "unresolved"
    confidence: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "anchor": self.anchor,
            "relation": self.relation,
            "direction": self.direction,
            "target_role": self.target_role,
            "source_var": self.source_var,
            "target_var": self.target_var,
            "depends_on": self.depends_on,
            "context_title": self.context_title,
            "resolution_type": self.resolution_type,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ResolvedAnchor:
    surface: str
    canonical: str
    context_title: str = ""
    matched_entity_id: str = ""
    aliases: tuple[str, ...] = ()
    resolution_type: str = "unresolved"
    confidence: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "canonical": self.canonical,
            "context_title": self.context_title,
            "matched_entity_id": self.matched_entity_id,
            "aliases": list(self.aliases),
            "resolution_type": self.resolution_type,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class HotpotQuestionSemantics:
    reasoning_family: str
    reasoning_type: str
    confidence: float = 1.0
    compatible_topologies: tuple[str, ...] = ()
    entities_in_question: tuple[str, ...] = ()
    requested_relation: str = "unresolved"
    answer_mode: str = "short_answer"
    constraint_graph_type: str = ""
    answer_var: str = ""
    source_entity: str = ""
    source_relation_hint: str = ""
    target_relation: str = "unresolved"
    expected_answer_role: str = "ANY_RESOLVED"
    bridge_relation_hint: str = ""
    entity_a: str = ""
    entity_b: str = ""
    compared_relation: str = "unresolved"
    comparison_operator: str = "compare"
    constraints: tuple[HotpotConstraint, ...] = ()
    valid: bool = True
    validation_errors: tuple[str, ...] = ()


class SamplingModel:
    def __init__(self, model_path: str, *, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
        signature = inspect.signature(AutoModelForCausalLM.from_pretrained)
        kwargs["dtype" if "dtype" in signature.parameters else "torch_dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **kwargs,
        )
        self.model.eval()
        self.max_new_tokens = int(max_new_tokens)

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> tuple[str, dict[str, int | float]]:
        self.torch.manual_seed(int(seed))
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(int(seed))
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        device = next(self.model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": bool(do_sample),
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            kwargs["temperature"] = max(1e-5, float(temperature))
            kwargs["top_p"] = float(top_p)
        started = time.time()
        with self.torch.inference_mode():
            out = self.model.generate(**kwargs)
        elapsed = time.time() - started
        new_ids = out[0, input_ids.shape[-1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return text, {
            "input_tokens": int(input_ids.shape[-1]),
            "output_tokens": int(new_ids.shape[-1]),
            "latency_sec": elapsed,
        }


def task_prompt(row: dict[str, Any], domain: str) -> str:
    if domain == "gsm8k":
        return (
            "Solve the math word problem. Show concise reasoning, then write the final answer "
            "as 'Final answer: <number>'.\n\n"
            f"Problem:\n{row['question']}"
        )
    if domain == "mmlu_pro":
        return (
            f"{render_native_task(row)}\n\n"
            "Think briefly if needed, then end with exactly 'Final answer: <option letter>'."
        )
    if domain == "hotpotqa":
        return (
            f"{render_native_task(row)}\n\n"
            "Answer concisely. End with exactly 'Final answer: <short answer>'."
        )
    return render_native_task(row)


def generation_prompt(
    row: dict[str, Any],
    domain: str,
    *,
    attempt: int,
    previous_candidates: list[Candidate] | None = None,
) -> tuple[str, str]:
    if domain != "mmlu_pro":
        return task_prompt(row, domain), ""
    base = render_native_task(row)
    previous_candidates = previous_candidates or []
    previous_answers = [
        str(candidate_artifact(domain, candidate.text)[2] or "").strip().upper()
        for candidate in previous_candidates
    ]
    previous_answers = [answer for answer in previous_answers if answer]
    previous_block = (
        "\nPrevious candidate answers: " + ", ".join(previous_answers)
        if previous_answers else
        "\nPrevious candidate answers: none"
    )
    labels = [
        str(choice.get("label", "")).strip().upper()
        for choice in row.get("choices", [])
        if isinstance(choice, dict) and str(choice.get("label", "")).strip()
    ]
    labels_text = ", ".join(labels) if labels else "the visible option labels"
    prompt = (
        f"{base}\n"
        f"{previous_block}\n\n"
        "Task: evaluate every option independently as a hypothesis. Do not start by choosing "
        "an answer label. For each option, list concise support and contradiction evidence, "
        "then assign support_strength, contradiction_strength, and uncertainty from 0.0 to 1.0.\n\n"
        "Return only JSON with this schema:\n"
        "{\n"
        '  "option_hypotheses": [\n'
        "    {\n"
        '      "option_id": "A",\n'
        '      "support": ["..."],\n'
        '      "contradictions": ["..."],\n'
        '      "support_strength": 0.0,\n'
        '      "contradiction_strength": 0.0,\n'
        '      "uncertainty": 0.0\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Include exactly these option ids if present: {labels_text}.\n"
        "Do not include a final answer field; the runtime will select argmax."
    )
    return prompt, "option_hypothesis_builder"


def score_candidate(row: dict[str, Any], domain: str, text: str) -> dict[str, Any]:
    if domain in CODE_DOMAINS:
        return score_prediction(row, extract_code(text))
    if domain in {"mmlu_pro", "hotpotqa"}:
        _artifact_type, artifact, _final_value = candidate_artifact(domain, text)
        return score_prediction(row, artifact)
    return score_prediction(row, text)


def candidate_artifact(domain: str, text: str) -> tuple[str, Any, Any]:
    if domain in CODE_DOMAINS:
        code = extract_code(text)
        return "code", code, None
    if domain == "mmlu_pro":
        choice = select_mmlu_option_from_hypotheses(text) or extract_mmlu_final_choice(text)
        return "choice", choice, choice
    if domain == "hotpotqa":
        answer = extract_hotpot_short_answer(text)
        return "answer", answer, answer
    final_value = extract_numeric_answer(text)
    return "result", final_value, final_value


def select_mmlu_option_from_hypotheses(value: Any) -> str:
    selected = _mmlu_runtime_selected_option(value)
    if selected:
        return selected
    hypotheses = extract_mmlu_option_hypotheses(value)
    if not hypotheses:
        return ""
    scored = []
    for index, hypothesis in enumerate(hypotheses):
        option_id = str(hypothesis.get("option_id") or hypothesis.get("label") or hypothesis.get("option") or "").strip().upper()
        if not re.fullmatch(r"[A-J]", option_id):
            continue
        support = _option_strength(hypothesis, "support_strength", "support")
        contradiction = _option_strength(hypothesis, "contradiction_strength", "contradictions", "contradiction")
        uncertainty = _float_value(hypothesis.get("uncertainty"), default=0.0)
        score = support - contradiction - uncertainty
        scored.append((score, -index, option_id))
    if not scored:
        return ""
    ranked = sorted(scored, reverse=True)
    if len(ranked) > 1 and abs(ranked[0][0] - ranked[1][0]) <= 1e-9:
        return ""
    return ranked[0][2]


def _mmlu_runtime_selected_option(value: Any) -> str:
    parsed = _parse_json_payload(value)
    if not isinstance(parsed, dict):
        return ""
    selection = parsed.get("runtime_selection")
    if isinstance(selection, dict):
        option_id = str(selection.get("selected_option_id") or "").strip().upper()
        if re.fullmatch(r"[A-J]", option_id):
            return option_id
    pairwise = parsed.get("pairwise_critic")
    if isinstance(pairwise, dict):
        option_id = str(pairwise.get("selected_option_id") or pairwise.get("better_option_id") or "").strip().upper()
        if re.fullmatch(r"[A-J]", option_id):
            return option_id
    return ""


def extract_mmlu_option_hypotheses(value: Any) -> list[dict[str, Any]]:
    parsed = _parse_json_payload(value)
    if parsed is None:
        return _parse_mmlu_option_blocks(str(value or ""))
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if not isinstance(parsed, dict):
        return []
    for key in ("option_hypotheses", "hypotheses", "options"):
        items = parsed.get(key)
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    analysis = parsed.get("option_analysis")
    if isinstance(analysis, dict):
        result = []
        for option_id, payload in analysis.items():
            if isinstance(payload, dict):
                result.append({"option_id": option_id, **payload})
        return result
    return []


def extract_mmlu_final_choice(value: Any) -> str:
    text = str(value or "").strip()
    parsed = _parse_json_payload(text)
    if isinstance(parsed, dict):
        for key in ("answer", "final_answer", "choice", "selected_option", "solver_choice", "independent_choice"):
            if key in parsed:
                choice = extract_mmlu_final_choice(parsed[key])
                if choice:
                    return choice
    labelled = re.findall(
        r"(?:final\s+answer|answer|selected\s+option|choice)\s*[:：]\s*([A-J])\b",
        text,
        flags=re.IGNORECASE,
    )
    if labelled:
        return labelled[-1].upper()
    last_lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(last_lines[-3:]):
        match = re.fullmatch(r"(?:\(?([A-Ja-j])\)?\.?)", line)
        if match:
            return match.group(1).upper()
    return ""


def _parse_json_payload(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.append(text)
    brace = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if brace:
        candidates.append(brace.group(1))
    bracket = re.search(r"(\[.*\])", text, flags=re.DOTALL)
    if bracket:
        candidates.append(bracket.group(1))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _parse_mmlu_option_blocks(text: str) -> list[dict[str, Any]]:
    result = []
    pattern = re.compile(
        r"(?:^|\n)\s*(?:Option\s+)?([A-J])\s*[:.)-]\s*(.*?)(?=(?:\n\s*(?:Option\s+)?[A-J]\s*[:.)-])|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        option_id = match.group(1).upper()
        body = match.group(2)
        result.append({
            "option_id": option_id,
            "support": _extract_lines_after_label(body, "support"),
            "contradictions": _extract_lines_after_label(body, "contradiction"),
            "support_strength": _extract_float_after_label(body, "support_strength"),
            "contradiction_strength": _extract_float_after_label(body, "contradiction_strength"),
            "uncertainty": _extract_float_after_label(body, "uncertainty"),
        })
    return result


def _extract_lines_after_label(text: str, label: str) -> list[str]:
    match = re.search(rf"{label}s?\s*[:：]\s*(.*)", text, flags=re.IGNORECASE)
    if not match:
        return []
    value = match.group(1).strip()
    return [value] if value else []


def _extract_float_after_label(text: str, label: str) -> float:
    match = re.search(rf"{label}\s*[:：]\s*([01](?:\.\d+)?)", text, flags=re.IGNORECASE)
    return _float_value(match.group(1), default=0.0) if match else 0.0


def _option_strength(hypothesis: dict[str, Any], strength_key: str, *evidence_keys: str) -> float:
    if strength_key in hypothesis:
        return _float_value(hypothesis.get(strength_key), default=0.0)
    evidence_count = 0
    for key in evidence_keys:
        value = hypothesis.get(key)
        if isinstance(value, list):
            evidence_count += sum(1 for item in value if str(item).strip())
        elif str(value or "").strip():
            evidence_count += 1
    return min(1.0, 0.35 * evidence_count)


def _float_value(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number < 0:
        return 0.0
    if number > 1:
        return 1.0
    return number


def extract_hotpot_short_answer(value: Any) -> str:
    """Extract a deployable short answer from model text without gold labels."""

    if isinstance(value, dict):
        for key in ("answer", "final_answer", "derived_answer", "value"):
            if key in value:
                return extract_hotpot_short_answer(value[key])
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        extracted = extract_hotpot_short_answer(parsed)
        if extracted:
            return extracted
    labelled = re.findall(
        r"(?:final\s+answer|answer|short\s+answer)\s*[:：]\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )
    if labelled:
        text = labelled[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[-1]
    text = re.sub(r"^(?:final\s+answer|answer|short\s+answer)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("\"'`")
    text = re.split(r"\s+(?:because|since|as)\s+", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    if len(text.split()) > 12 and "." in text:
        text = text.split(".", 1)[0].strip()
    return text


def tests_for_row(row: dict[str, Any]) -> tuple[list[str], str]:
    tests = row.get("tests", [])
    setup = "\n".join(str(item.get("setup", "")) for item in tests if isinstance(item, dict))
    test_text = [str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in tests]
    return test_text, setup


def code_failure_detail(row: dict[str, Any], text: str) -> dict[str, Any]:
    tests, setup = tests_for_row(row)
    code = extract_code(text)
    result = execute_python_tests_detailed(code=code, tests=tests, setup=setup, timeout=5)
    return dict(result.execution)


def build_repair_feedback(row: dict[str, Any], domain: str, candidate: Candidate) -> dict[str, Any]:
    if domain in CODE_DOMAINS:
        detail = code_failure_detail(row, candidate.text)
        kind = str(detail.get("kind") or "execution_failure")
        failed_test = str(detail.get("failed_test") or "")
        expected = detail.get("expected")
        actual = detail.get("actual")
        function_name = str(detail.get("function_name") or "the target function")
        if kind == "assertion_failure" and (expected is not None or actual is not None):
            instruction = (
                f"For this failing test, {function_name} returned {actual}; it must return "
                f"{expected}. Fix the implementation for this concrete case while preserving "
                "the function signature."
            )
        else:
            instruction = (
                "Fix the implementation so all provided tests pass. Preserve the required "
                "function signature and return only executable Python code."
            )
        return {
            "feedback_type": "oracle_repair_signal",
            "kind": kind,
            "failed_test": failed_test,
            "test_expression": detail.get("test_expression"),
            "function_name": function_name,
            "input": detail.get("input_repr"),
            "expected": expected,
            "actual": actual,
            "exception_type": detail.get("exception_type"),
            "exception": detail.get("exception"),
            "repair_instruction": instruction,
        }
    if domain == "mmlu_pro":
        _artifact_type, choice, _final_value = candidate_artifact(domain, candidate.text)
        labels = [str(choice_row.get("label")) for choice_row in row.get("choices", []) if isinstance(choice_row, dict)]
        return {
            "feedback_type": "deployable_repair_signal",
            "kind": "multiple_choice_uncertain",
            "predicted_choice": choice,
            "allowed_choices": labels,
            "repair_instruction": (
                "Re-evaluate the question and every option. Return exactly one option letter "
                "using the schema 'Final answer: <option letter>'."
            ),
        }
    if domain == "hotpotqa":
        _artifact_type, answer, _final_value = candidate_artifact(domain, candidate.text)
        return {
            "feedback_type": "deployable_repair_signal",
            "kind": "multihop_answer_uncertain",
            "predicted_answer": answer,
            "repair_instruction": (
                "Re-check the evidence chain and return only the concise short answer using "
                "the schema 'Final answer: <short answer>'."
            ),
        }
    predicted = extract_numeric_answer(candidate.text)
    return {
        "feedback_type": "oracle_repair_signal",
        "kind": "numeric_answer_incorrect",
        "predicted_answer": predicted,
        "repair_instruction": (
            "The previous final numeric answer is incorrect. Recompute the word problem from "
            "the quantities and relationships, check units, and provide a new final answer. "
            "Do not simply repeat the previous answer."
        ),
    }


def repair_prompt(row: dict[str, Any], domain: str, previous: Candidate, feedback: dict[str, Any], *, round_id: int) -> str:
    if domain in CODE_DOMAINS:
        tests = "\n".join(tests_for_row(row)[0])
        return (
            "You are repairing a Python solution. Return only executable Python code, no Markdown.\n\n"
            f"Task:\n{row['question']}\n\n"
            f"Provided tests:\n{tests}\n\n"
            f"Previous candidate:\n{extract_code(previous.text)}\n\n"
            f"Verifier feedback:\n{json.dumps(feedback, ensure_ascii=False, indent=2)}\n\n"
            "Repair only the faulty behavior and preserve correct behavior."
        )
    if domain == "mmlu_pro":
        return (
            "You are repairing a multiple-choice answer. Use only the visible question and options. "
            "End with exactly 'Final answer: <option letter>'.\n\n"
            f"Task:\n{render_native_task(row)}\n\n"
            f"Previous answer:\n{previous.text}\n\n"
            f"Verifier feedback:\n{json.dumps(feedback, ensure_ascii=False, indent=2)}\n"
        )
    if domain == "hotpotqa":
        return (
            "You are repairing a multihop QA answer. Use the evidence below and return only a concise "
            "short answer at the end as 'Final answer: <short answer>'.\n\n"
            f"Task:\n{render_native_task(row)}\n\n"
            f"Previous answer:\n{previous.text}\n\n"
            f"Verifier feedback:\n{json.dumps(feedback, ensure_ascii=False, indent=2)}\n"
        )
    return (
        "You are repairing a math solution. Give concise corrected reasoning and end with "
        "'Final answer: <number>'.\n\n"
        f"Problem:\n{row['question']}\n\n"
        f"Previous solution:\n{previous.text}\n\n"
        f"Verifier feedback:\n{json.dumps(feedback, ensure_ascii=False, indent=2)}\n"
    )


def generate_candidate(
    model: SamplingModel,
    row: dict[str, Any],
    domain: str,
    prompt: str,
    *,
    seed: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    repair_round: int = 0,
    feedback: dict[str, Any] | None = None,
    generation_role: str = "",
) -> Candidate:
    text, metrics = model.generate(
        prompt,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )
    return Candidate(
        text=text,
        score=score_candidate(row, domain, text),
        input_tokens=int(metrics["input_tokens"]),
        output_tokens=int(metrics["output_tokens"]),
        latency_sec=float(metrics["latency_sec"]),
        seed=seed,
        repair_round=repair_round,
        feedback=feedback,
        generation_role=generation_role,
    )


def generate_attempt_candidate(
    model: SamplingModel,
    row: dict[str, Any],
    domain: str,
    *,
    attempt: int,
    previous_candidates: list[Candidate],
    seed: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> Candidate:
    if domain == "mmlu_pro":
        candidate = generate_mmlu_option_independent_candidate(
            model,
            row,
            attempt=attempt,
            previous_candidates=previous_candidates,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        return mark_candidate_redundancy(candidate, previous_candidates, domain=domain)
    if domain == "hotpotqa" and HOTPOT_CANDIDATE_MODE == "preserve_first":
        candidate = generate_hotpot_preserve_first_candidate(
            model,
            row,
            attempt=attempt,
            previous_candidates=previous_candidates,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        return mark_candidate_redundancy(candidate, previous_candidates, domain=domain)
    prompt, role = generation_prompt(
        row,
        domain,
        attempt=attempt,
        previous_candidates=previous_candidates,
    )
    candidate = generate_candidate(
        model,
        row,
        domain,
        prompt,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        generation_role=role,
    )
    return mark_candidate_redundancy(candidate, previous_candidates, domain=domain)


def generate_mmlu_option_independent_candidate(
    model: SamplingModel,
    row: dict[str, Any],
    *,
    attempt: int,
    previous_candidates: list[Candidate],
    seed: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> Candidate:
    hypotheses: list[dict[str, Any]] = []
    raw_outputs: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    latency_sec = 0.0
    choices = [
        choice for choice in row.get("choices", [])
        if isinstance(choice, dict) and str(choice.get("text", "")).strip()
    ]
    mode = MMLU_OPTIONWISE_MODE
    if mode == "permutation" or (mode == "second_pass" and attempt % 2 == 1):
        choices = _deterministic_permutation(choices, seed=seed + attempt * 3571)
    for option_index, choice in enumerate(choices):
        option_text = str(choice.get("text", "")).strip()
        prompt = mmlu_option_independent_prompt(
            row,
            option_text,
            previous_candidates=previous_candidates,
            mode=mode,
            pass_index=attempt,
        )
        text, metrics = model.generate(
            prompt,
            seed=seed + option_index * 7919,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        input_tokens += int(metrics["input_tokens"])
        output_tokens += int(metrics["output_tokens"])
        latency_sec += float(metrics["latency_sec"])
        hypothesis = parse_single_option_hypothesis(text)
        hypothesis["option_id"] = str(choice.get("label", "")).strip().upper()
        hypothesis["option_text"] = option_text
        hypotheses.append(hypothesis)
        raw_outputs.append({
            "option_index": option_index,
            "option_id": hypothesis["option_id"],
            "option_text": option_text,
            "raw_output": text,
        })
    calibration: dict[str, Any] = {}
    cheap_scores = mmlu_option_hypothesis_scores(hypotheses)
    top2 = _top2_mmlu_options(cheap_scores)
    cheap_margin = _top_option_margin(cheap_scores)
    pairwise_critic: dict[str, Any] = {}
    selected_option_id = top2[0][0] if top2 else ""
    selection_source = "cheap_argmax"
    if len(top2) >= 2 and cheap_margin < MMLU_OPTION_PAIRWISE_MARGIN_THRESHOLD:
        pairwise_prompt = mmlu_pairwise_option_critic_prompt(row, hypotheses, top2)
        pairwise_text, pairwise_metrics = model.generate(
            pairwise_prompt,
            seed=seed + 104729,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        input_tokens += int(pairwise_metrics["input_tokens"])
        output_tokens += int(pairwise_metrics["output_tokens"])
        latency_sec += float(pairwise_metrics["latency_sec"])
        pairwise_critic = parse_mmlu_pairwise_critic(pairwise_text, allowed_option_ids=[top2[0][0], top2[1][0]])
        pairwise_critic["raw_output"] = pairwise_text
        if pairwise_critic.get("selected_option_id"):
            selected_option_id = str(pairwise_critic["selected_option_id"])
            selection_source = "pairwise_top2_critic"
    payload = {
        "generation_mode": "option_independent",
        "optionwise_mode": mode,
        "attempt": attempt,
        "option_hypotheses": hypotheses,
        "cheap_option_scores": cheap_scores,
        "cheap_top_options": [
            {"option_id": option_id, "score": score}
            for option_id, score in top2
        ],
        "cheap_score_margin": cheap_margin,
        "pairwise_margin_threshold": MMLU_OPTION_PAIRWISE_MARGIN_THRESHOLD,
        "pairwise_critic": pairwise_critic,
        "score_calibration": calibration,
        "runtime_selection": {
            "selected_option_id": selected_option_id,
            "selection_source": selection_source,
            "score_margin": cheap_margin,
        },
        "raw_option_outputs": raw_outputs,
    }
    text = json.dumps(payload, ensure_ascii=False)
    return Candidate(
        text=text,
        score=score_candidate(row, "mmlu_pro", text),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_sec=latency_sec,
        seed=seed,
        generation_role=f"option_independent_hypothesis_builder:{mode}",
    )


def generate_hotpot_preserve_first_candidate(
    model: SamplingModel,
    row: dict[str, Any],
    *,
    attempt: int,
    previous_candidates: list[Candidate],
    seed: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> Candidate:
    facts = _hotpot_facts(row)
    bridge_records = _hotpot_ranked_bridge_pool(row, facts=facts, previous_candidates=previous_candidates, limit=8)
    previous_signatures = {_hotpot_chain_signature(candidate.text) for candidate in previous_candidates}
    chain_options = _hotpot_hierarchical_chain_options(
        row=row,
        facts=facts,
        bridge_records=bridge_records,
        previous_signatures=previous_signatures,
    )
    option = chain_options[attempt % len(chain_options)] if chain_options else {}
    bridge = str(option.get("bridge", ""))
    view = option.get("view")
    binding = option.get("binding", {}) if isinstance(option.get("binding"), dict) else {}
    if view is None:
        view, binding = _hotpot_preserve_first_chain_binding(
            row=row,
            candidate_id=f"c{attempt}",
            bridge=bridge,
            facts=facts,
            pair_offset=attempt,
            previous_signatures=previous_signatures,
        )
    endpoint_binding, endpoint_metrics = _hotpot_bind_endpoints_with_model(
        model=model,
        row=row,
        view=view,
        facts=facts,
        reasoning_type=str(option.get("candidate_topology") or _hotpot_reasoning_type(row)),
        seed=seed + 271828,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )
    endpoint_payload = endpoint_binding.as_dict()
    endpoint_bridge = str(endpoint_binding.inferred_bridge or "").strip()
    if endpoint_bridge:
        bridge = endpoint_bridge
        view = build_hotpot_chain_view_from_facts(
            candidate_id=f"c{attempt}",
            bridge_entity=endpoint_bridge,
            question=str(row.get("question", "")),
            facts=facts,
            hop1_fact_ids=view.hop1_fact_ids,
            hop2_fact_ids=view.hop2_fact_ids,
            derived_answer=endpoint_binding.derived_answer or view.derived_answer,
        )
    prompt = hotpot_preserve_first_answer_prompt(row, view)
    text, metrics = model.generate(
        prompt,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )
    answer = extract_hotpot_short_answer(text)
    if not answer:
        answer = str(endpoint_binding.derived_answer or "").strip()
    if answer:
        view = build_hotpot_chain_view_from_facts(
            candidate_id=f"c{attempt}",
            bridge_entity=bridge,
            question=str(row.get("question", "")),
            facts=facts,
            hop1_fact_ids=view.hop1_fact_ids,
            hop2_fact_ids=view.hop2_fact_ids,
            derived_answer=answer,
        )
    reasoning_type = str(option.get("candidate_topology") or _hotpot_reasoning_type(row))
    topology_classifier = option.get("topology_classifier", {}) if isinstance(option.get("topology_classifier"), dict) else {}
    reasoning_family = str(option.get("reasoning_family") or _hotpot_reasoning_family_for_type(reasoning_type))
    candidate_object = build_hotpot_reasoning_candidate_object(
        view=view,
        reasoning_type=reasoning_type,
        evidence_bindings={
            "hop1_pool": binding.get("hop1_pool", []),
            "hop2_pool": binding.get("hop2_pool", []),
            "chain_pair_pool": binding.get("chain_pair_pool", []),
            "endpoint_binding": endpoint_payload,
        },
        derived_decision={"answer": answer or view.derived_answer},
        provenance={
            "allocation_policy": option.get("allocation_policy", ""),
            "constructor_family": option.get("constructor_family", ""),
            "candidate_topology": reasoning_type,
            "reasoning_family": reasoning_family,
            "topology_classifier": topology_classifier,
        },
    )
    payload = {
        "generation_mode": "preserve_first_evidence_chain_binding",
        "attempt": attempt,
        "answer": answer or view.derived_answer,
        "bridge_entity": view.bridge_entity,
        "cited_fact_ids": list(view.cited_fact_ids),
        "hop1": {"fact_ids": list(view.hop1_fact_ids), "evidence": view.hop1_key_sentence},
        "hop2": {"fact_ids": list(view.hop2_fact_ids), "evidence": view.hop2_key_sentence},
        "hop1_pool": binding.get("hop1_pool", []),
        "hop2_pool": binding.get("hop2_pool", []),
        "chain_pair_pool": binding.get("chain_pair_pool", []),
        "endpoint_binding": endpoint_payload,
        "answer_alignment": view.answer_alignment,
        "contradiction_signal": view.contradiction_signal,
        "chain_signature": _hotpot_view_signature(view),
        "bridge_candidate_rank": option.get("bridge_rank", attempt),
        "global_chain_rank": option.get("hierarchical_chain_rank", attempt + 1),
        "hierarchical_chain_rank": option.get("hierarchical_chain_rank", attempt + 1),
        "bridge_chain_pair_rank": option.get("pair_rank"),
        "reasoning_type": reasoning_type,
        "reasoning_family": reasoning_family,
        "topology_classifier": topology_classifier,
        "allocation_policy": option.get("allocation_policy", ""),
        "constructor_family": option.get("constructor_family", ""),
        "bridge_entity_telemetry": option.get("bridge_entity_telemetry", {}),
        "bridge_attribute_telemetry": option.get("bridge_attribute_telemetry", {}),
        "multi_constraint_telemetry": option.get("multi_constraint_telemetry", {}),
        "parallel_retrieval_telemetry": option.get("parallel_retrieval_telemetry", {}),
        "global_chain_score": option.get("global_chain_score"),
        "retrieval_score": option.get("retrieval_score"),
        "semantic_chain_score": option.get("semantic_chain_score"),
        "bridge_candidate_pool": [record.as_dict() for record in bridge_records],
        "raw_bridge_pool_size": len(_hotpot_raw_bridge_pool(row, facts=facts)),
        "structured_candidate_view": json.loads(render_hotpot_structured_candidate_view(view)),
        "hotpot_reasoning_candidate": candidate_object.as_dict(),
        "raw_answer_output": text,
    }
    candidate_text = json.dumps(payload, ensure_ascii=False)
    return Candidate(
        text=candidate_text,
        score=score_candidate(row, "hotpotqa", candidate_text),
        input_tokens=int(metrics["input_tokens"]) + int(endpoint_metrics["input_tokens"]),
        output_tokens=int(metrics["output_tokens"]) + int(endpoint_metrics["output_tokens"]),
        latency_sec=float(metrics["latency_sec"]) + float(endpoint_metrics["latency_sec"]),
        seed=seed,
        generation_role="preserve_first_evidence_chain_binding",
    )


def _hotpot_hierarchical_chain_options(
    *,
    row: dict[str, Any],
    facts: list[dict[str, Any]],
    bridge_records: list[BridgeCandidate],
    previous_signatures: set[str],
) -> list[dict[str, Any]]:
    classification = _hotpot_topology_classification(row)
    reasoning_type = str(classification["reasoning_type"])
    reasoning_family = str(classification["reasoning_family"])
    topology_confidence = float(classification["confidence"])
    compatible_topologies = list(classification["compatible_topologies"])
    evidence_index = _hotpot_evidence_index(row, facts)
    topology_options: list[dict[str, Any]] = []
    for candidate_topology in compatible_topologies:
        if candidate_topology == "bridge_entity":
            options = _hotpot_bridge_entity_chain_options(
                row=row,
                facts=facts,
                evidence_index=evidence_index,
                previous_signatures=previous_signatures,
            )
        elif candidate_topology == "bridge_attribute":
            options = _hotpot_bridge_family_chain_options(
                row=row,
                facts=facts,
                reasoning_type=candidate_topology,
                evidence_index=evidence_index,
                previous_signatures=previous_signatures,
            )
        elif candidate_topology == "multi_constraint_lookup":
            options = _hotpot_constraint_chain_options(
                row=row,
                facts=facts,
                reasoning_type=candidate_topology,
                evidence_index=evidence_index,
                previous_signatures=previous_signatures,
            )
        else:
            options = _hotpot_topology_chain_options(
                row=row,
                facts=facts,
                evidence_index=evidence_index,
                reasoning_type=candidate_topology,
                previous_signatures=previous_signatures,
            )
        for option in options[:16]:
            option["candidate_topology"] = candidate_topology
            option["reasoning_family"] = _hotpot_reasoning_family_for_type(candidate_topology)
            option["topology_classifier"] = {
                "reasoning_family": reasoning_family,
                "reasoning_type": reasoning_type,
                "confidence": topology_confidence,
                "compatible_topologies": list(compatible_topologies),
                "entities_in_question": list(classification["entities_in_question"]),
                "requested_relation": classification["requested_relation"],
                "answer_mode": classification["answer_mode"],
            }
            if candidate_topology == reasoning_type:
                option["topology_prior"] = 1.0
            elif candidate_topology in compatible_topologies and _hotpot_reasoning_family_for_type(candidate_topology) == reasoning_family:
                option["topology_prior"] = 0.7
            else:
                option["topology_prior"] = 0.25
            topology_options.append(option)
    if topology_options:
        topology_options.sort(key=lambda item: (
            -float(item.get("topology_prior", 0.0)),
            -float(item.get("semantic_chain_score", item.get("global_chain_score", 0.0)) or 0.0),
            -float(item.get("retrieval_score", 0.0) or 0.0),
            int(item.get("hierarchical_chain_rank", item.get("pair_rank", 999))),
        ))
        for index, item in enumerate(topology_options, start=1):
            item["hierarchical_chain_rank"] = index
            item["delayed_topology_commitment"] = True
        return topology_options[:64]
    # Fallback for sparse/odd inputs: keep the old bridge-diverse path as a
    # backup, but only after topology-native construction has failed.
    first_pass: list[dict[str, Any]] = []
    second_pass: list[dict[str, Any]] = []
    for bridge_rank, bridge_record in enumerate(bridge_records, start=1):
        view, binding = _hotpot_preserve_first_chain_binding(
            row=row,
            candidate_id=f"candidate_bridge_{bridge_rank}",
            bridge=bridge_record.name,
            facts=facts,
            pair_offset=0,
            previous_signatures=previous_signatures,
        )
        pair_pool = binding.get("chain_pair_pool", [])
        for pair_rank, pair in enumerate(pair_pool[:2], start=1):
            if not isinstance(pair, dict):
                continue
            hop1_id = str(pair.get("hop1_fact_id", ""))
            hop2_id = str(pair.get("hop2_fact_id", ""))
            pair_view = build_hotpot_chain_view_from_facts(
                candidate_id=f"candidate_bridge_{bridge_rank}_pair_{pair_rank}",
                bridge_entity=bridge_record.name,
                question=str(row.get("question", "")),
                facts=facts,
                hop1_fact_ids=[hop1_id],
                hop2_fact_ids=[hop2_id],
            )
            signature = _hotpot_view_signature(pair_view)
            if signature in previous_signatures:
                continue
            option = {
                "bridge": bridge_record.name,
                "bridge_rank": bridge_rank,
                "pair_rank": pair_rank,
                "view": pair_view,
                "binding": binding,
                "global_chain_score": float(pair.get("semantic_chain_score", pair.get("chain_score", 0.0)) or 0.0),
                "retrieval_score": float(pair.get("retrieval_score", 0.0) or 0.0),
                "semantic_chain_score": float(pair.get("semantic_chain_score", pair.get("chain_score", 0.0)) or 0.0),
                "allocation_policy": "hierarchical_topk",
            }
            if pair_rank == 1:
                first_pass.append(option)
            else:
                second_pass.append(option)
    first_pass.sort(key=lambda item: (int(item["bridge_rank"]), int(item["pair_rank"])))
    second_pass.sort(key=lambda item: (int(item["bridge_rank"]), int(item["pair_rank"])))
    options = [*first_pass, *second_pass]
    for index, item in enumerate(options, start=1):
        item["hierarchical_chain_rank"] = index
    return options


def _hotpot_candidate_topologies(primary: str) -> list[str]:
    family = _hotpot_reasoning_family_for_type(primary)
    return _hotpot_candidate_topologies_for_family(family, primary=primary, confidence=1.0)


def _hotpot_candidate_topologies_for_family(family: str, *, primary: str, confidence: float) -> list[str]:
    family_topologies = {
        "compositional": ["bridge_entity", "bridge_attribute"],
        "parallel": ["comparison_attribute", "shared_property_yesno", "set_intersection"],
        "constraint": ["multi_constraint_lookup"],
    }
    all_topologies = [
        *family_topologies["compositional"],
        *family_topologies["parallel"],
        *family_topologies["constraint"],
    ]
    base = family_topologies.get(family, family_topologies["compositional"])
    extras: list[str] = []
    if confidence < 0.80:
        if family == "compositional":
            extras = ["multi_constraint_lookup", "comparison_attribute"]
        elif family == "parallel":
            extras = ["bridge_attribute"]
        elif family == "constraint":
            extras = ["bridge_attribute", "bridge_entity"]
    ordered = [primary, *all_topologies]
    ordered = [primary, *base, *extras, *all_topologies]
    result: list[str] = []
    for item in ordered:
        if item in all_topologies and item not in result:
            result.append(item)
    return result


def _hotpot_bridge_entity_chain_options(
    *,
    row: dict[str, Any],
    facts: list[dict[str, Any]],
    evidence_index: HotpotEvidenceIndex | None = None,
    previous_signatures: set[str],
) -> list[dict[str, Any]]:
    evidence_index = evidence_index or _hotpot_evidence_index(row, facts)
    qsem = _hotpot_question_semantics(row)
    source_entity = qsem.source_entity
    source_pool = _hotpot_merge_fact_pool([
        *_hotpot_rank_bridge_entity_source_facts(row, facts, source_entity=source_entity, limit=20),
        *_hotpot_retrieve_by_anchor(evidence_index, anchor=source_entity, query_type="bridge_discovery", topk=40),
    ])[:20]
    options: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for source_rank, hop1 in enumerate(source_pool, start=1):
        inferred_bridges = _hotpot_inferred_intermediate_hypotheses_from_fact(row, hop1, limit=3)
        if not inferred_bridges:
            continue
        for bridge_hypothesis_rank, inferred_bridge in enumerate(inferred_bridges, start=1):
            hop2_pool = _hotpot_merge_fact_pool([
                *_hotpot_rank_relation_specific_hop2(row, inferred_bridge, facts, reasoning_type="bridge_entity", limit=30),
                *_hotpot_retrieve_by_anchor_relation(
                    evidence_index,
                    anchor=inferred_bridge,
                    relation=qsem.target_relation,
                    query_type="answer_lookup",
                    topk=40,
                ),
            ])[:10]
            pair_pool = _hotpot_rank_chain_pairs(row, inferred_bridge, [hop1], hop2_pool)
            binding = {
                "hop1_pool": [_hotpot_scored_fact_as_dict(item) for item in source_pool],
                "hop2_pool": [_hotpot_scored_fact_as_dict(item) for item in hop2_pool],
                "chain_pair_pool": [_hotpot_pair_as_dict(item) for item in pair_pool],
                "constructor_family": "evidence_derived_bridge_entity",
                "inferred_bridge_hypotheses": inferred_bridges,
            }
            for pair_rank, pair in enumerate(pair_pool, start=1):
                hop1_id = str(pair.get("hop1", {}).get("fact_id", ""))
                hop2_id = str(pair.get("hop2", {}).get("fact_id", ""))
                pair_key = (hop1_id, hop2_id)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                pair_verdict = pair.get("semantic_chain_verification", {})
                pair_view = build_hotpot_chain_view_from_facts(
                    candidate_id=f"candidate_bridge_entity_source_{source_rank}_bridge_{bridge_hypothesis_rank}_pair_{pair_rank}",
                    bridge_entity=str(pair_verdict.get("inferred_bridge") or inferred_bridge),
                    question=str(row.get("question", "")),
                    facts=facts,
                    hop1_fact_ids=[hop1_id],
                    hop2_fact_ids=[hop2_id],
                )
                signature = _hotpot_view_signature(pair_view)
                if signature in previous_signatures:
                    continue
                telemetry = {
                    "source_entity": source_entity,
                    "source_fact_id": hop1_id,
                    "inferred_bridge": str(pair_verdict.get("inferred_bridge") or inferred_bridge),
                    "inferred_bridge_hypotheses": inferred_bridges,
                    "requested_relation": qsem.target_relation,
                    "expected_answer_role": qsem.expected_answer_role,
                    "hop2_pool_fact_ids": [str(item.get("fact_id", "")) for item in hop2_pool],
                    "constructor_family": "evidence_derived_bridge_entity",
                }
                options.append({
                    "bridge": pair_view.bridge_entity,
                    "bridge_rank": source_rank,
                    "pair_rank": pair_rank,
                    "view": pair_view,
                    "binding": binding,
                    "global_chain_score": float(pair.get("semantic_chain_score", pair.get("chain_score", 0.0)) or 0.0),
                    "retrieval_score": float(pair.get("retrieval_score", 0.0) or 0.0),
                    "semantic_chain_score": float(pair.get("semantic_chain_score", pair.get("chain_score", 0.0)) or 0.0),
                    "allocation_policy": "topology_native_bridge_entity_evidence_derived",
                    "constructor_family": "evidence_derived_bridge_entity",
                    "bridge_entity_telemetry": telemetry,
                })
    # Preserve source-side diversity first; semantic score orders local pairs
    # for each discovered intermediate.
    options.sort(key=lambda item: (int(item["bridge_rank"]), -float(item["semantic_chain_score"]), -float(item["retrieval_score"]), int(item["pair_rank"])))
    for index, item in enumerate(options, start=1):
        item["hierarchical_chain_rank"] = index
    return options


def _hotpot_bridge_family_chain_options(
    *,
    row: dict[str, Any],
    facts: list[dict[str, Any]],
    reasoning_type: str,
    evidence_index: HotpotEvidenceIndex | None = None,
    previous_signatures: set[str],
) -> list[dict[str, Any]]:
    evidence_index = evidence_index or _hotpot_evidence_index(row, facts)
    qsem = _hotpot_question_semantics(row)
    source_entity = qsem.source_entity
    source_pool = _hotpot_merge_fact_pool([
        *_hotpot_rank_source_side_facts(row, facts, limit=20),
        *_hotpot_retrieve_by_anchor(evidence_index, anchor=source_entity, query_type="bridge_discovery", topk=40),
    ])[:16]
    options: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for source_rank, hop1 in enumerate(source_pool, start=1):
        inferred_bridges = _hotpot_inferred_intermediate_hypotheses_from_fact(row, hop1, limit=3)
        if not inferred_bridges:
            continue
        for bridge_hypothesis_rank, inferred_bridge in enumerate(inferred_bridges, start=1):
            hop2_pool = _hotpot_merge_fact_pool([
                *_hotpot_rank_relation_specific_hop2(row, inferred_bridge, facts, reasoning_type=reasoning_type, limit=30),
                *_hotpot_retrieve_by_anchor_relation(
                    evidence_index,
                    anchor=inferred_bridge,
                    relation=qsem.requested_relation or qsem.target_relation,
                    query_type="answer_lookup",
                    topk=40,
                ),
            ])[:10]
            pair_pool = _hotpot_rank_chain_pairs(row, inferred_bridge, [hop1], hop2_pool)
            binding = {
                "hop1_pool": [_hotpot_scored_fact_as_dict(item) for item in source_pool],
                "hop2_pool": [_hotpot_scored_fact_as_dict(item) for item in hop2_pool],
                "chain_pair_pool": [_hotpot_pair_as_dict(item) for item in pair_pool],
                "constructor_family": "evidence_derived_bridge",
                "inferred_bridge_hypotheses": inferred_bridges,
            }
            for pair_rank, pair in enumerate(pair_pool, start=1):
                hop1_id = str(pair.get("hop1", {}).get("fact_id", ""))
                hop2_id = str(pair.get("hop2", {}).get("fact_id", ""))
                pair_key = (hop1_id, hop2_id)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                pair_view = build_hotpot_chain_view_from_facts(
                    candidate_id=f"candidate_{reasoning_type}_source_{source_rank}_bridge_{bridge_hypothesis_rank}_pair_{pair_rank}",
                    bridge_entity=str(pair.get("semantic_chain_verification", {}).get("inferred_bridge", inferred_bridge)),
                    question=str(row.get("question", "")),
                    facts=facts,
                    hop1_fact_ids=[hop1_id],
                    hop2_fact_ids=[hop2_id],
                )
                signature = _hotpot_view_signature(pair_view)
                if signature in previous_signatures:
                    continue
                bridge_attribute_telemetry = {}
                if reasoning_type == "bridge_attribute":
                    bridge_attribute_telemetry = {
                        "source_fact_id": hop1_id,
                        "inferred_bridge": str(pair.get("semantic_chain_verification", {}).get("inferred_bridge", inferred_bridge)),
                        "inferred_bridge_hypotheses": inferred_bridges,
                        "requested_relation": qsem.requested_relation or qsem.target_relation,
                        "expected_answer_role": qsem.expected_answer_role,
                        "hop2_retrieved_any_about_bridge": any(bool(item.get("about_bridge")) for item in hop2_pool),
                        "hop2_retrieved_requested_relation": any(bool(item.get("requested_relation_match")) for item in hop2_pool),
                        "hop2_pool_fact_ids": [str(item.get("fact_id", "")) for item in hop2_pool],
                        "hop2_relation_matches": [
                            {
                                "fact_id": str(item.get("fact_id", "")),
                                "about_bridge": bool(item.get("about_bridge")),
                                "requested_relation_match": bool(item.get("requested_relation_match")),
                                "relation_match_type": str(item.get("relation_match_type", "")),
                                "target_role_match": bool(item.get("target_role_match")),
                                "fact_relation": str(item.get("fact_relation", "")),
                            }
                            for item in hop2_pool
                        ],
                        "constructor_family": "evidence_derived_bridge_attribute",
                    }
                options.append({
                    "bridge": pair_view.bridge_entity,
                    "bridge_rank": source_rank,
                    "pair_rank": pair_rank,
                    "view": pair_view,
                    "binding": binding,
                    "global_chain_score": float(pair.get("semantic_chain_score", pair.get("chain_score", 0.0)) or 0.0),
                    "retrieval_score": float(pair.get("retrieval_score", 0.0) or 0.0),
                    "semantic_chain_score": float(pair.get("semantic_chain_score", pair.get("chain_score", 0.0)) or 0.0),
                    "allocation_policy": f"topology_native_{reasoning_type}",
                    "constructor_family": "evidence_derived_bridge",
                    "bridge_attribute_telemetry": bridge_attribute_telemetry,
                })
    options.sort(key=lambda item: (-float(item["semantic_chain_score"]), -float(item["retrieval_score"]), int(item["bridge_rank"]), int(item["pair_rank"])))
    for index, item in enumerate(options, start=1):
        item["hierarchical_chain_rank"] = index
    return options


def _hotpot_constraint_chain_options(
    *,
    row: dict[str, Any],
    facts: list[dict[str, Any]],
    reasoning_type: str,
    evidence_index: HotpotEvidenceIndex | None = None,
    previous_signatures: set[str],
) -> list[dict[str, Any]]:
    evidence_index = evidence_index or _hotpot_evidence_index(row, facts)
    qsem = _hotpot_question_semantics(row)
    if not qsem.valid:
        return []
    constraints = [constraint.as_dict() for constraint in qsem.constraints]
    if len(constraints) < 2:
        return _hotpot_topology_chain_options(
            row=row,
            facts=facts,
            reasoning_type=reasoning_type,
            previous_signatures=previous_signatures,
        )
    targets_1 = _hotpot_retrieve_relation_targets(row, constraints[0], facts, role="constraint_1", limit=20, evidence_index=evidence_index)
    if qsem.constraint_graph_type == "serial":
        targets_2: list[dict[str, Any]] = []
        shared_target_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for left in targets_1:
            if not bool(left.get("satisfied")):
                continue
            bound = str(left.get("target", "")).strip()
            if not bound:
                continue
            second_constraint = dict(constraints[1])
            second_constraint["anchor"] = bound
            second_constraint["bound_from_var"] = str(constraints[1].get("source_var") or constraints[1].get("depends_on") or "X")
            second_constraint["bound_from_fact_id"] = str(left.get("fact_id", ""))
            right_targets = _hotpot_retrieve_relation_targets(
                row,
                second_constraint,
                facts,
                role="constraint_2",
                limit=10,
                evidence_index=evidence_index,
            )
            targets_2.extend(right_targets)
            for right in right_targets:
                if bool(right.get("satisfied")) and str(left.get("fact_id", "")) != str(right.get("fact_id", "")):
                    shared_target_pairs.append((left, right))
        shared_target_pairs.sort(key=lambda pair: (
            -(float(pair[0].get("retrieval_score", 0.0)) + float(pair[1].get("retrieval_score", 0.0))),
            str(pair[0].get("fact_id", "")),
            str(pair[1].get("fact_id", "")),
        ))
    else:
        targets_2 = _hotpot_retrieve_relation_targets(row, constraints[1], facts, role="constraint_2", limit=20, evidence_index=evidence_index)
        shared_target_pairs = _hotpot_intersect_constraint_targets(targets_1, targets_2)
    hop1_pool = [_hotpot_constraint_target_as_fact(item, role="constraint_1") for item in targets_1[:5]]
    hop2_pool = [_hotpot_constraint_target_as_fact(item, role="constraint_2") for item in targets_2[:5]]
    if shared_target_pairs:
        pair_pool = [
            _hotpot_constraint_target_pair_as_chain_pair(row, left, right)
            for left, right in shared_target_pairs
        ]
    else:
        pair_pool = _hotpot_rank_chain_pairs(row, "", hop1_pool, hop2_pool)
    binding = {
        "hop1_pool": [_hotpot_scored_fact_as_dict(item) for item in hop1_pool],
        "hop2_pool": [_hotpot_scored_fact_as_dict(item) for item in hop2_pool],
        "chain_pair_pool": [_hotpot_pair_as_dict(item) for item in pair_pool],
        "constructor_family": "constraint_target_intersection",
        "constraints": constraints,
            "constraint_graph_type": qsem.constraint_graph_type,
            "answer_var": qsem.answer_var,
        "constraint_target_sets": [
            [_hotpot_constraint_target_as_dict(item) for item in targets_1],
            [_hotpot_constraint_target_as_dict(item) for item in targets_2],
        ],
        "shared_constraint_targets": [
            {
                    "target": str(right.get("target", "") if qsem.constraint_graph_type == "serial" else left.get("target", "")),
                    "intermediate": str(left.get("target", "")) if qsem.constraint_graph_type == "serial" else "",
                "constraint_1_fact_id": str(left.get("fact", {}).get("fact_id", "")),
                "constraint_2_fact_id": str(right.get("fact", {}).get("fact_id", "")),
            }
            for left, right in shared_target_pairs
        ],
    }
    options = []
    for pair_rank, pair in enumerate(pair_pool, start=1):
        verifier = pair.get("semantic_chain_verification", {})
        details = verifier.get("type_specific_details", {}) if isinstance(verifier, dict) else {}
        targets = details.get("constraint_targets", []) if isinstance(details, dict) else []
        hop1_id = str(pair.get("hop1", {}).get("fact_id", ""))
        hop2_id = str(pair.get("hop2", {}).get("fact_id", ""))
        pair_view = build_hotpot_chain_view_from_facts(
            candidate_id=f"candidate_{reasoning_type}_constraint_pair_{pair_rank}",
            bridge_entity=str(targets[0] if targets else ""),
            question=str(row.get("question", "")),
            facts=facts,
            hop1_fact_ids=[hop1_id],
            hop2_fact_ids=[hop2_id],
        )
        signature = _hotpot_view_signature(pair_view)
        if signature in previous_signatures:
            continue
        telemetry = {
            "constraints": constraints,
            "target_set_sizes": [len(targets_1), len(targets_2)],
            "shared_target_count": len(shared_target_pairs),
            "shared_targets": [
                str(right.get("target", "") if qsem.constraint_graph_type == "serial" else left.get("target", ""))
                for left, right in shared_target_pairs
            ],
            "constraint_graph_type": qsem.constraint_graph_type,
            "answer_var": qsem.answer_var,
            "constraint_1_targets": [_hotpot_constraint_target_as_dict(item) for item in targets_1],
            "constraint_2_targets": [_hotpot_constraint_target_as_dict(item) for item in targets_2],
            "constructor_family": "serial_constraint_binding" if qsem.constraint_graph_type == "serial" else "constraint_target_intersection",
        }
        options.append({
            "bridge": pair_view.bridge_entity,
            "bridge_rank": 0,
            "pair_rank": pair_rank,
            "view": pair_view,
            "binding": binding,
            "global_chain_score": float(pair.get("semantic_chain_score", pair.get("chain_score", 0.0)) or 0.0),
            "retrieval_score": float(pair.get("retrieval_score", 0.0) or 0.0),
            "semantic_chain_score": float(pair.get("semantic_chain_score", pair.get("chain_score", 0.0)) or 0.0),
            "allocation_policy": f"topology_native_{reasoning_type}",
            "constructor_family": "serial_constraint_binding" if qsem.constraint_graph_type == "serial" else "constraint_target_intersection",
            "multi_constraint_telemetry": telemetry,
        })
    options.sort(key=lambda item: (-float(item["semantic_chain_score"]), -float(item["retrieval_score"]), int(item["pair_rank"])))
    for index, item in enumerate(options, start=1):
        item["hierarchical_chain_rank"] = index
    return options


def _hotpot_topology_chain_options(
    *,
    row: dict[str, Any],
    facts: list[dict[str, Any]],
    evidence_index: HotpotEvidenceIndex | None = None,
    reasoning_type: str,
    previous_signatures: set[str],
) -> list[dict[str, Any]]:
    evidence_index = evidence_index or _hotpot_evidence_index(row, facts)
    hop1_pool, hop2_pool = _hotpot_topology_fact_pools(row, facts=facts, bridge="", reasoning_type=reasoning_type, evidence_index=evidence_index)
    pair_pool = _hotpot_rank_chain_pairs(row, "", hop1_pool, hop2_pool)
    options = []
    binding = {
        "hop1_pool": [_hotpot_scored_fact_as_dict(item) for item in hop1_pool],
        "hop2_pool": [_hotpot_scored_fact_as_dict(item) for item in hop2_pool],
        "chain_pair_pool": [_hotpot_pair_as_dict(item) for item in pair_pool],
        "constructor_family": "entity_side_parallel_retrieval",
    }
    parallel_telemetry = _hotpot_parallel_retrieval_telemetry(row, reasoning_type, hop1_pool, hop2_pool)
    for pair_rank, pair in enumerate(pair_pool, start=1):
        hop1_id = str(pair.get("hop1", {}).get("fact_id", ""))
        hop2_id = str(pair.get("hop2", {}).get("fact_id", ""))
        pair_view = build_hotpot_chain_view_from_facts(
            candidate_id=f"candidate_{reasoning_type}_pair_{pair_rank}",
            bridge_entity=str(pair.get("semantic_chain_verification", {}).get("inferred_bridge", "")),
            question=str(row.get("question", "")),
            facts=facts,
            hop1_fact_ids=[hop1_id],
            hop2_fact_ids=[hop2_id],
        )
        signature = _hotpot_view_signature(pair_view)
        if signature in previous_signatures:
            continue
        options.append({
            "bridge": str(pair.get("semantic_chain_verification", {}).get("inferred_bridge", "")),
            "bridge_rank": 0,
            "pair_rank": pair_rank,
            "view": pair_view,
            "binding": binding,
            "global_chain_score": float(pair.get("semantic_chain_score", pair.get("chain_score", 0.0)) or 0.0),
            "retrieval_score": float(pair.get("retrieval_score", 0.0) or 0.0),
            "semantic_chain_score": float(pair.get("semantic_chain_score", pair.get("chain_score", 0.0)) or 0.0),
            "allocation_policy": f"topology_specific_{reasoning_type}",
            "constructor_family": "entity_side_parallel_retrieval",
            "parallel_retrieval_telemetry": parallel_telemetry,
        })
    options.sort(key=lambda item: (-float(item["semantic_chain_score"]), int(item["pair_rank"])))
    for index, item in enumerate(options, start=1):
        item["hierarchical_chain_rank"] = index
    return options


def _hotpot_facts(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "fact_id": str(fact.get("fact_id") or f"supporting_fact_{index}"),
            "title": str(fact.get("title", fact.get("entity", ""))),
            "sent_id": fact.get("sent_id"),
            "text": str(fact.get("text", fact.get("sentence", ""))),
        }
        for index, fact in enumerate(row.get("supporting_facts", row.get("evidence", [])), start=1)
        if isinstance(fact, dict)
    ]


def _hotpot_raw_bridge_pool(row: dict[str, Any], *, facts: list[dict[str, Any]]) -> list[BridgeCandidate]:
    records: list[BridgeCandidate] = []
    for index, entity in enumerate(row.get("entities", []), start=1):
        records.append(_hotpot_bridge_record(str(entity), "question_entity", f"entity_{index}", row, facts))
    for index, fact in enumerate(facts, start=1):
        title = str(fact.get("title", "")).strip()
        if title:
            records.append(_hotpot_bridge_record(title, "supporting_fact_title", str(fact.get("fact_id") or f"supporting_fact_{index}"), row, facts))
        for entity in _hotpot_entities_in_text(str(fact.get("text", ""))):
            if _hotpot_bridge_role_admissible(_hotpot_ontology_role(entity)):
                records.append(_hotpot_bridge_record(entity, "evidence_entity", str(fact.get("fact_id") or f"supporting_fact_{index}"), row, facts))
    return records


def _hotpot_evidence_index(row: dict[str, Any], facts: list[dict[str, Any]]) -> HotpotEvidenceIndex:
    index = HotpotEvidenceIndex(by_title={}, by_entity={}, by_relation={}, by_entity_relation={}, all_facts=[])
    for fact in facts:
        semantics = _hotpot_fact_semantics(row, fact)
        normalized_relation = _hotpot_canonical_relation(str(semantics.get("relation", "")))
        indexed_fact = {
            **fact,
            "explicit_relation": str(semantics.get("relation", "")),
            "normalized_relation": normalized_relation,
            "relation_aliases": sorted(RELATION_ALIASES.get(normalized_relation, {normalized_relation})),
            "fact_semantics": semantics,
        }
        index.all_facts.append(indexed_fact)
        title = str(semantics.get("title_entity", fact.get("title", "")))
        entities = {
            title,
            str(semantics.get("subject", "")),
            str(semantics.get("object", "")),
            *[str(item) for item in semantics.get("referenced_entities", [])],
            *[str(item) for item in _hotpot_entities_in_text(f"{fact.get('title', '')} {fact.get('text', '')}")],
        }
        _hotpot_add_index_fact(index.by_title, normalize_bridge(title), indexed_fact)
        if normalized_relation and normalized_relation != "unresolved":
            _hotpot_add_index_fact(index.by_relation, normalized_relation, indexed_fact)
        for entity in entities:
            entity_key = normalize_bridge(entity)
            if not entity_key or entity_key == "unresolved":
                continue
            _hotpot_add_index_fact(index.by_entity, entity_key, indexed_fact)
            if normalized_relation and normalized_relation != "unresolved":
                _hotpot_add_index_fact(index.by_entity_relation, (entity_key, normalized_relation), indexed_fact)
    return index


def _hotpot_add_index_fact(mapping: dict[Any, list[dict[str, Any]]], key: Any, fact: dict[str, Any]) -> None:
    if not key:
        return
    bucket = mapping.setdefault(key, [])
    if all(str(item.get("fact_id", "")) != str(fact.get("fact_id", "")) for item in bucket):
        bucket.append(fact)


def _hotpot_retrieve_by_anchor(
    index: HotpotEvidenceIndex,
    *,
    anchor: str,
    query_type: str,
    topk: int,
    relation_hint: str = "",
) -> list[dict[str, Any]]:
    key = normalize_bridge(anchor)
    pool = []
    pool.extend(_hotpot_tag_retrieved_facts(index.by_title.get(key, []), source="document_title_match", query_type=query_type, anchor=anchor, relation_hint=relation_hint, base_score=6.0))
    pool.extend(_hotpot_tag_retrieved_facts(index.by_entity.get(key, []), source="question_entity_anchor", query_type=query_type, anchor=anchor, relation_hint=relation_hint, base_score=5.0))
    pool.extend(_hotpot_tag_retrieved_facts(
        _hotpot_raw_anchor_context_facts(index, anchor),
        source="anchor_preserving_raw",
        query_type=query_type,
        anchor=anchor,
        relation_hint=relation_hint,
        base_score=4.0,
    ))
    return _hotpot_merge_fact_pool(pool)[:topk]


def _hotpot_retrieve_by_anchor_relation(
    index: HotpotEvidenceIndex,
    *,
    anchor: str,
    relation: str,
    query_type: str,
    topk: int,
) -> list[dict[str, Any]]:
    anchor_key = normalize_bridge(anchor)
    relation_key = _hotpot_canonical_relation(relation)
    pool = []
    pool.extend(_hotpot_tag_retrieved_facts(index.by_entity_relation.get((anchor_key, relation_key), []), source="relation_conditioned", query_type=query_type, anchor=anchor, relation_hint=relation_key, base_score=8.0))
    # Recall-first backoff: relation extraction helps ordering, but failure to
    # parse a predicate must not prevent a raw context sentence from entering
    # the candidate evidence pool.
    pool.extend(_hotpot_tag_retrieved_facts(index.by_title.get(anchor_key, []), source="document_title_match", query_type=query_type, anchor=anchor, relation_hint=relation_key, base_score=6.0))
    pool.extend(_hotpot_tag_retrieved_facts(index.by_entity.get(anchor_key, []), source="dataset_context_scan", query_type=query_type, anchor=anchor, relation_hint=relation_key, base_score=5.0))
    pool.extend(_hotpot_tag_retrieved_facts(
        _hotpot_raw_anchor_context_facts(index, anchor),
        source="anchor_preserving_raw",
        query_type=query_type,
        anchor=anchor,
        relation_hint=relation_key,
        base_score=4.0,
    ))
    return _hotpot_merge_fact_pool(pool)[:topk]


def _hotpot_raw_anchor_context_facts(index: HotpotEvidenceIndex, anchor: str) -> list[dict[str, Any]]:
    anchor_key = normalize_bridge(anchor)
    if not anchor_key:
        return []
    matches = []
    for fact in index.all_facts:
        title_key = normalize_bridge(fact.get("title", ""))
        text_key = normalize_bridge(fact.get("text", ""))
        if title_key == anchor_key or anchor_key in text_key:
            matches.append(fact)
    return matches


def _hotpot_tag_retrieved_facts(
    facts: list[dict[str, Any]],
    *,
    source: str,
    query_type: str,
    anchor: str,
    relation_hint: str,
    base_score: float,
) -> list[dict[str, Any]]:
    tagged = []
    relation_key = _hotpot_canonical_relation(relation_hint)
    relation_terms = set(RELATION_ALIASES.get(relation_key, {relation_key})) if relation_key and relation_key != "unresolved" else set()
    for fact in facts:
        fact_relation = _hotpot_canonical_relation(str(fact.get("normalized_relation", fact.get("explicit_relation", ""))))
        relation_bonus = 0.0
        if relation_key and relation_key != "unresolved":
            if fact_relation == relation_key:
                relation_bonus += 1.5
            fact_text = normalize_bridge(f"{fact.get('title', '')} {fact.get('text', '')}")
            if any(normalize_bridge(term) and normalize_bridge(term) in fact_text for term in relation_terms):
                relation_bonus += 0.8
        score = base_score + relation_bonus + float(fact.get("retrieval_score", fact.get("hop_score", 0.0)) or 0.0)
        tagged.append({
            **fact,
            "retrieval_score": score,
            "hop_score": score,
            "retrieval_source": source,
            "retrieval_query_type": query_type,
            "anchor": anchor,
            "relation_hint": relation_hint,
        })
    return tagged


def _hotpot_merge_fact_pool(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for fact in facts:
        fact_id = str(fact.get("fact_id", ""))
        if not fact_id:
            continue
        existing = by_id.get(fact_id)
        sources = set()
        if existing:
            sources.update(existing.get("retrieval_sources", []))
            if existing.get("retrieval_source"):
                sources.add(str(existing["retrieval_source"]))
        if fact.get("retrieval_source"):
            sources.add(str(fact["retrieval_source"]))
        fact_tier = int(fact.get("anchor_tier", 9))
        existing_tier = int(existing.get("anchor_tier", 9)) if existing else 9
        if existing is None or (
            fact_tier < existing_tier
            or (
                fact_tier == existing_tier
                and float(fact.get("retrieval_score", 0.0) or 0.0) > float(existing.get("retrieval_score", 0.0) or 0.0)
            )
        ):
            by_id[fact_id] = {**fact, "retrieval_sources": sorted(sources)}
        elif existing is not None:
            existing["retrieval_sources"] = sorted(sources)
            if fact_tier < int(existing.get("anchor_tier", 9)):
                existing["anchor_tier"] = fact_tier
                existing["anchor_tier_name"] = str(fact.get("anchor_tier_name", existing.get("anchor_tier_name", "")))
    return sorted(by_id.values(), key=lambda item: (int(item.get("anchor_tier", 9)), -float(item.get("retrieval_score", 0.0) or 0.0), str(item.get("fact_id", ""))))


def _hotpot_ranked_bridge_pool(
    row: dict[str, Any],
    *,
    facts: list[dict[str, Any]],
    previous_candidates: list[Candidate],
    limit: int,
) -> list[BridgeCandidate]:
    by_key: dict[str, BridgeCandidate] = {}
    source_priority = {
        "question_entity": 4,
        "supporting_fact_title": 3,
        "evidence_entity": 2,
        "model_generated": 1,
    }
    for record in _hotpot_raw_bridge_pool(row, facts=facts):
        key = normalize_bridge(record.name)
        if not key:
            continue
        existing = by_key.get(key)
        if existing is None or (
            record.score,
            source_priority.get(record.source_type, 0),
            -len(record.name),
        ) > (
            existing.score,
            source_priority.get(existing.source_type, 0),
            -len(existing.name),
        ):
            by_key[key] = record
    previous_bridges = {
        normalize_bridge(_hotpot_candidate_bridge(candidate.text))
        for candidate in previous_candidates
    }
    ranked = sorted(
        by_key.values(),
        key=lambda record: (
            normalize_bridge(record.name) in previous_bridges,
            -record.score,
            -source_priority.get(record.source_type, 0),
            record.name.lower(),
        ),
    )
    return ranked[:limit]


def _hotpot_bridge_record(
    name: str,
    source_type: str,
    source_node_id: str,
    row: dict[str, Any],
    facts: list[dict[str, Any]],
) -> BridgeCandidate:
    clean = bridge_candidates_from_values([name], limit=1)
    candidate = clean[0] if clean else ""
    question_relevance = _hotpot_question_relevance(candidate, str(row.get("question", "")))
    related_facts = _hotpot_facts_for_bridge_like(candidate, facts)
    evidence_support = min(1.0, len(related_facts) / 3.0)
    connected_titles = {
        normalize_bridge(fact.get("title", ""))
        for fact in _rank_hotpot_facts_for_terms(facts, f"{row.get('question', '')} {candidate}", limit=6)
        if str(fact.get("title", "")).strip()
    }
    connectivity = min(1.0, len([title for title in connected_titles if title]) / 4.0)
    return BridgeCandidate(
        name=candidate,
        source_type=source_type,
        source_node_id=source_node_id,
        question_relevance=question_relevance,
        connectivity_score=connectivity,
        evidence_support=evidence_support,
    )


def _hotpot_question_relevance(candidate: str, question: str) -> float:
    cand_terms = _hotpot_terms(candidate)
    if not cand_terms:
        return 0.0
    question_terms = _hotpot_terms(question)
    return min(1.0, len(cand_terms & question_terms) / max(1, len(cand_terms)))


def _hotpot_facts_for_bridge_like(candidate: str, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key = normalize_bridge(candidate)
    if not key:
        return []
    return [
        fact for fact in facts
        if normalize_bridge(fact.get("title", fact.get("entity", ""))) == key
        or key in normalize_bridge(fact.get("text", ""))
    ]


def _hotpot_entities_in_text(text: str) -> list[str]:
    # Conservative entity-ish spans.  This is recall-oriented but filters short
    # one-token relation words by requiring capitalization and length.
    spans = re.findall(r"\b(?:[A-Z][A-Za-z0-9'’.-]+(?:\s+|$)){1,5}", text)
    result = []
    for span in spans:
        value = re.sub(r"\s+", " ", span).strip(" ,.;:()[]{}\"'")
        if len(value) >= 4 and not re.fullmatch(r"The|This|That|He|She|It|In|On|At", value):
            result.append(value)
    return result[:8]


def _hotpot_ontology_role(value: Any) -> str:
    text = str(value or "").strip()
    key = normalize_bridge(text)
    if not key:
        return "UNKNOWN"
    nationality_values = {
        "american", "british", "canadian", "french", "german", "italian", "dutch", "greek",
        "south korean", "korean", "turkish", "ottoman", "eastern", "southern", "western", "northern",
    }
    type_values = {"genus", "species", "family", "album", "film", "book", "city", "country", "neighborhood"}
    if key in nationality_values or key in type_values:
        return "ATTRIBUTE_VALUE"
    if re.fullmatch(r"\d[\d,]*(?:\.\d+)?", text):
        return "LITERAL"
    if re.search(r"\b(inc|corp|corporation|company|university|entertainment|records|fc|club|group|band|mosque|mansion)\b", key):
        return "ORGANIZATION"
    if re.search(r"\b(city|country|island|islands|ocean|river|lake|neighborhood|district|village|town|istanbul|turkey|england)\b", key):
        return "LOCATION"
    if re.search(r"\b(album|film|novel|book|song|series|episode)\b", key):
        return "WORK"
    if len(key.split()) >= 2:
        return "NAMED_ENTITY"
    if text[:1].isupper() and len(text) >= 4:
        return "NAMED_ENTITY"
    return "UNKNOWN"


def _hotpot_bridge_role_admissible(role: str) -> bool:
    return role in {"PERSON", "ORGANIZATION", "LOCATION", "WORK", "EVENT", "NAMED_ENTITY"}


def _hotpot_terms(value: str) -> set[str]:
    return {
        term for term in re.findall(r"[a-z0-9]+", str(value).lower())
        if len(term) > 2 and term not in {"the", "and", "for", "with", "was", "were", "that", "this"}
    }


def _hotpot_target_terms(question: str) -> set[str]:
    q = question.lower()
    terms = _hotpot_terms(question)
    if "nationality" in q:
        terms |= {"nationality", "american", "british", "canadian", "country"}
    if "which country" in q or "what country" in q:
        terms |= {"country", "nation", "located", "born"}
    if "which city" in q or "what city" in q:
        terms |= {"city", "located", "born", "based"}
    if "born" in q:
        terms |= {"born", "birth", "city", "country"}
    if "government position" in q or "position" in q:
        terms |= {"position", "office", "served", "minister", "chief", "protocol"}
    if "how many" in q or "seat" in q:
        terms |= {"capacity", "seats", "seated", "attendance"}
    if "war" in q or "casualties" in q:
        terms |= {"war", "casualties", "world"}
    return terms


def _hotpot_answer_type_match(question: str, text: str) -> float:
    q = question.lower()
    t = text.lower()
    if "how many" in q:
        return 1.0 if re.search(r"\b\d[\d,]*\b", text) else 0.0
    if "when" in q or "what year" in q:
        return 1.0 if re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", text) else 0.0
    if "which country" in q or "nationality" in q:
        return 1.0 if re.search(r"\b(country|nationality|american|british|canadian|french|german|italian)\b", t) else 0.0
    if "which city" in q or "where" in q:
        return 1.0 if re.search(r"\b(city|located|based|born|village|town)\b", t) else 0.0
    if "who" in q:
        return 0.5 if re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+", text) else 0.0
    return 0.0


RELATION_ALIASES = {
    "formed_by": {"formed_by", "formed by", "founded_by", "founded by", "established_by", "established by", "founded", "formed", "established"},
    "created_by": {"created_by", "created by", "creator", "created"},
    "written_by": {"written_by", "written by", "wrote", "authored_by", "authored by", "author", "author of", "writer"},
    "directed_by": {"directed_by", "directed by", "director", "director of", "directed"},
    "located_in": {"located_in", "located in", "situated in", "based_in", "based in", "location"},
    "born_in": {"born_in", "born in", "birthplace"},
    "nationality": {"nationality", "country", "citizen"},
    "home_venue": {"home_venue", "home venue", "home games at", "played at", "arena"},
    "capacity": {"capacity", "seat", "seats", "seated", "can seat"},
    "position": {"position", "government position", "office", "served as", "chief of protocol"},
    "acted_in": {"acted_in", "portrayed", "played", "starred", "starring", "actor"},
}


def _hotpot_canonical_relation(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    if not text or text == "unresolved":
        return "unresolved"
    for canonical, aliases in RELATION_ALIASES.items():
        if text == canonical.replace("_", " ") or text in aliases:
            return canonical
    if "formed" in text or "established" in text:
        return "formed_by"
    if "founded" in text:
        return "formed_by"
    if "author" in text or "wrote" in text or "written" in text or "authored" in text:
        return "written_by"
    if "created" in text:
        return "created_by"
    if "directed" in text or "director" in text:
        return "directed_by"
    if "located" in text or "based" in text:
        return "located_in"
    if "born" in text:
        return "born_in"
    if "home games" in text or "home venue" in text or "played at" in text or "arena" in text:
        return "home_venue"
    if "capacity" in text or "seat" in text:
        return "capacity"
    if "position" in text or "served" in text or "office" in text:
        return "position"
    if "portrayed" in text or "played" in text or "starred" in text or "starring" in text or "actor" in text:
        return "acted_in"
    return text.replace(" ", "_")


def _hotpot_relation_match_type(fact_relation: str, requested_relation: str) -> str:
    fact = _hotpot_canonical_relation(fact_relation)
    requested = _hotpot_canonical_relation(requested_relation)
    if not fact or not requested or fact == "unresolved" or requested == "unresolved":
        return "mismatch"
    if fact == requested:
        return "exact"
    compatible = {
        ("formed_by", "created_by"),
        ("created_by", "formed_by"),
        ("written_by", "created_by"),
        ("created_by", "written_by"),
        ("home_venue", "located_in"),
        ("located_in", "home_venue"),
    }
    return "compatible" if (fact, requested) in compatible else "mismatch"


def _hotpot_question_relation_label(row: dict[str, Any]) -> str:
    q = str(row.get("question", "")).lower()
    # Prefer the answer-side relation.  Bridge questions often mention the
    # source-side relation ("director of X", "album created by Y") before the
    # actual requested answer relation ("born where", "based in what city").
    if "how many" in q and re.search(r"\b(seat|seats|seated|capacity)\b", q):
        return "capacity"
    if "government position" in q or re.search(r"\bwhat position\b|\bwhich position\b", q):
        return "position"
    if "born in" in q or "birthplace" in q or re.search(r"\bwhere\b.*\bborn\b|\bborn\b.*\bwhere\b", q):
        return "born_in"
    if "based in" in q or re.search(r"\bwhat .*city\b|\bwhich .*city\b|\bwhere\b", q):
        return "located_in"
    if "formed" in q or "founded" in q or "established" in q:
        return "formed_by"
    if "written by" in q or "wrote" in q or "authored" in q or "author" in q:
        return "written_by"
    if "created by" in q or "created" in q:
        return "created_by"
    if "directed by" in q or "director" in q:
        return "directed_by"
    if "located in" in q:
        return "located_in"
    if "nationality" in q:
        return "nationality"
    return "unresolved"


def _hotpot_expected_answer_role(row: dict[str, Any]) -> str:
    q = str(row.get("question", "")).lower()
    relation = _hotpot_question_relation_label(row)
    if relation == "capacity" or "how many" in q:
        return "LITERAL"
    if relation == "position":
        return "NAMED_ENTITY"
    if "company" in q or "organization" in q or relation in {"formed_by", "created_by", "directed_by"}:
        return "ORGANIZATION_OR_NAMED_ENTITY"
    if "country" in q or "city" in q or "where" in q or relation in {"located_in", "born_in"}:
        return "LOCATION"
    if "nationality" in q:
        return "ATTRIBUTE_VALUE"
    if "who" in q:
        return "NAMED_ENTITY"
    return "ANY_RESOLVED"


def _hotpot_target_role_matches(object_role: str, expected_role: str) -> bool:
    role = str(object_role or "").upper()
    expected = str(expected_role or "ANY_RESOLVED").upper()
    if role in {"", "UNRESOLVED", "UNKNOWN"}:
        return False
    if expected == "ANY_RESOLVED":
        return True
    if expected == "ORGANIZATION_OR_NAMED_ENTITY":
        return role in {"ORGANIZATION", "NAMED_ENTITY", "PERSON"}
    if expected == "NAMED_ENTITY":
        return role in {"NAMED_ENTITY", "PERSON", "ORGANIZATION", "ATTRIBUTE_VALUE"}
    if expected == "LITERAL":
        return role in {"LITERAL", "NUMBER"}
    return role == expected


def _hotpot_preserve_first_chain_binding(
    *,
    row: dict[str, Any],
    candidate_id: str,
    bridge: str,
    facts: list[dict[str, Any]],
    pair_offset: int,
    previous_signatures: set[str],
) -> tuple[Any, dict[str, Any]]:
    reasoning_type = _hotpot_reasoning_type(row)
    hop1_pool, hop2_pool = _hotpot_topology_fact_pools(row, facts=facts, bridge=bridge, reasoning_type=reasoning_type)
    pair_pool = _hotpot_rank_chain_pairs(row, bridge, hop1_pool, hop2_pool)
    pair = pair_pool[pair_offset % len(pair_pool)] if pair_pool else {}
    hop1 = pair.get("hop1") if isinstance(pair, dict) else None
    hop2 = pair.get("hop2") if isinstance(pair, dict) else None
    if hop1 and hop2:
        view = build_hotpot_chain_view_from_facts(
            candidate_id=candidate_id,
            bridge_entity=bridge,
            question=str(row.get("question", "")),
            facts=facts,
            hop1_fact_ids=[str(hop1.get("fact_id"))],
            hop2_fact_ids=[str(hop2.get("fact_id"))],
        )
        if _hotpot_view_signature(view) not in previous_signatures:
            return view, {
                "hop1_pool": [_hotpot_scored_fact_as_dict(item) for item in hop1_pool],
                "hop2_pool": [_hotpot_scored_fact_as_dict(item) for item in hop2_pool],
                "chain_pair_pool": [_hotpot_pair_as_dict(item) for item in pair_pool],
            }
    view = build_hotpot_chain_view_from_facts(
        candidate_id=candidate_id,
        bridge_entity=bridge,
        question=str(row.get("question", "")),
        facts=facts,
    )
    return view, {
        "hop1_pool": [_hotpot_scored_fact_as_dict(item) for item in hop1_pool],
        "hop2_pool": [_hotpot_scored_fact_as_dict(item) for item in hop2_pool],
        "chain_pair_pool": [_hotpot_pair_as_dict(item) for item in pair_pool],
    }


def _hotpot_topology_fact_pools(
    row: dict[str, Any],
    *,
    facts: list[dict[str, Any]],
    bridge: str,
    reasoning_type: str,
    evidence_index: HotpotEvidenceIndex | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if reasoning_type in {"bridge_entity", "bridge_attribute"}:
        return (
            _hotpot_rank_hop1_candidates(row, bridge, facts, limit=3),
            _hotpot_rank_hop2_candidates(row, bridge, facts, limit=3),
        )
    if reasoning_type in {"comparison_attribute", "shared_property_yesno", "set_intersection"}:
        qsem = _hotpot_question_semantics(row)
        entity_a, entity_b = qsem.entity_a, qsem.entity_b
        relation = qsem.compared_relation
        if evidence_index is not None:
            return (
                _hotpot_merge_fact_pool([
                    *_hotpot_rank_parallel_fact_candidates(row, entity_a, facts, role="hop1", limit=20),
                    *_hotpot_retrieve_by_anchor_relation(evidence_index, anchor=entity_a, relation=relation, query_type="entity_a_lookup", topk=20),
                    *_hotpot_retrieve_by_anchor(evidence_index, anchor=entity_a, query_type="entity_a_lookup", relation_hint=relation, topk=20),
                ])[:6],
                _hotpot_merge_fact_pool([
                    *_hotpot_rank_parallel_fact_candidates(row, entity_b, facts, role="hop2", limit=20),
                    *_hotpot_retrieve_by_anchor_relation(evidence_index, anchor=entity_b, relation=relation, query_type="entity_b_lookup", topk=20),
                    *_hotpot_retrieve_by_anchor(evidence_index, anchor=entity_b, query_type="entity_b_lookup", relation_hint=relation, topk=20),
                ])[:6],
            )
        return (
            _hotpot_rank_parallel_fact_candidates(row, entity_a, facts, role="hop1", limit=3),
            _hotpot_rank_parallel_fact_candidates(row, entity_b, facts, role="hop2", limit=3),
        )
    return (
        _hotpot_rank_constraint_fact_candidates(row, facts, role="hop1", limit=3),
        _hotpot_rank_constraint_fact_candidates(row, facts, role="hop2", limit=3, reverse=True),
    )


def _hotpot_rank_parallel_fact_candidates(row: dict[str, Any], entity: str, facts: list[dict[str, Any]], *, role: str, limit: int) -> list[dict[str, Any]]:
    relation_terms = _hotpot_relation_target_terms(row)
    requested_relation = _hotpot_question_relation_label(row)
    scored = []
    for fact in facts:
        semantics = _hotpot_fact_semantics(row, fact)
        fact_terms = _hotpot_terms(f"{fact.get('title', '')} {fact.get('text', '')}")
        extracted_value = _hotpot_parallel_attribute_value(row, fact, entity)
        entity_match = _hotpot_fact_supports_entity(fact, entity) or _hotpot_entities_equivalent(semantics.get("subject", ""), entity)
        lexical_relation_match = len(relation_terms & fact_terms) / max(1, len(relation_terms)) if relation_terms else 0.0
        relation_match_type = _hotpot_relation_match_type(str(semantics.get("relation", "")), requested_relation)
        relation_compatible = relation_match_type in {"exact", "compatible"} or lexical_relation_match > 0.0 or bool(extracted_value)
        value_present = bool(extracted_value or semantics.get("object") not in {"", "unresolved"})
        if str(requested_relation) == "unresolved":
            relation_compatible = bool(lexical_relation_match > 0.0 or value_present)
        # Entity-side retrieval: each side must first be anchored on its own
        # entity, then prefer facts with the requested relation/property and a
        # comparable value.
        score = (
            3.0 * float(entity_match)
            + 2.0 * float(relation_compatible)
            + 1.2 * float(value_present)
            + 0.5 * lexical_relation_match
            + 0.2 * float(semantics.get("confidence", 0.0) or 0.0)
        )
        scored.append({
            **fact,
            "hop_role": role,
            "retrieval_score": score,
            "hop_score": score,
            "entity_side": entity,
            "entity_match": bool(entity_match),
            "relation_compatible": bool(relation_compatible),
            "relation_match_type": relation_match_type,
            "lexical_relation_match": lexical_relation_match,
            "value_present": bool(value_present),
            "extracted_value": extracted_value or str(semantics.get("object", "")),
            "fact_relation": str(semantics.get("relation", "")),
        })
    scored.sort(key=lambda item: (-float(item["retrieval_score"]), str(item.get("fact_id", ""))))
    return scored[:limit]


def _hotpot_parallel_retrieval_telemetry(
    row: dict[str, Any],
    reasoning_type: str,
    hop1_pool: list[dict[str, Any]],
    hop2_pool: list[dict[str, Any]],
) -> dict[str, Any]:
    entity_a, entity_b = _hotpot_question_entities(row)
    return {
        "reasoning_type": reasoning_type,
        "entity_a": entity_a,
        "entity_b": entity_b,
        "requested_relation": _hotpot_question_relation_label(row),
        "entity_a_fact_count": len(hop1_pool),
        "entity_b_fact_count": len(hop2_pool),
        "entity_a_relation_compatible_count": sum(int(bool(item.get("relation_compatible"))) for item in hop1_pool),
        "entity_b_relation_compatible_count": sum(int(bool(item.get("relation_compatible"))) for item in hop2_pool),
        "entity_a_value_count": sum(int(bool(item.get("value_present"))) for item in hop1_pool),
        "entity_b_value_count": sum(int(bool(item.get("value_present"))) for item in hop2_pool),
        "entity_a_facts": [_hotpot_parallel_fact_as_dict(item) for item in hop1_pool],
        "entity_b_facts": [_hotpot_parallel_fact_as_dict(item) for item in hop2_pool],
        "constructor_family": "entity_side_parallel_retrieval",
    }


def _hotpot_parallel_fact_as_dict(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "fact_id": str(item.get("fact_id", "")),
        "entity_side": str(item.get("entity_side", "")),
        "entity_match": bool(item.get("entity_match")),
        "relation_compatible": bool(item.get("relation_compatible")),
        "relation_match_type": str(item.get("relation_match_type", "")),
        "value_present": bool(item.get("value_present")),
        "extracted_value": str(item.get("extracted_value", "")),
        "fact_relation": str(item.get("fact_relation", "")),
        "retrieval_score": float(item.get("retrieval_score", 0.0) or 0.0),
    }


def _hotpot_rank_constraint_fact_candidates(
    row: dict[str, Any],
    facts: list[dict[str, Any]],
    *,
    role: str,
    limit: int,
    reverse: bool = False,
) -> list[dict[str, Any]]:
    question_terms = _hotpot_terms(str(row.get("question", "")))
    scored = []
    for index, fact in enumerate(facts):
        fact_terms = _hotpot_terms(f"{fact.get('title', '')} {fact.get('text', '')}")
        overlap = len(question_terms & fact_terms) / max(1, len(question_terms))
        semantics = _hotpot_fact_semantics(row, fact)
        value_present = 1.0 if semantics.get("object") not in {"", "unresolved"} else 0.0
        score = overlap + 0.5 * value_present + 0.1 * float(semantics.get("confidence", 0.0) or 0.0)
        scored.append({**fact, "hop_role": role, "retrieval_score": score, "hop_score": score, "_constraint_index": index})
    scored.sort(key=lambda item: (-float(item["retrieval_score"]), -int(item["_constraint_index"]) if reverse else int(item["_constraint_index"])))
    return scored[:limit]


def _hotpot_rank_source_side_facts(row: dict[str, Any], facts: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    question_entities = [str(item) for item in row.get("entities", []) if str(item).strip()]
    question_terms = _hotpot_terms(str(row.get("question", "")))
    scored = []
    for fact in facts:
        relation = _hotpot_fact_relation_view(row, "", fact, role="source")
        source_match = any(_hotpot_fact_supports_entity(fact, entity) for entity in question_entities)
        target = str(relation.get("target_entity", ""))
        target_valid = bool(target and target != "unresolved" and _hotpot_bridge_role_admissible(_hotpot_ontology_role(target)))
        text_terms = _hotpot_terms(f"{fact.get('title', '')} {fact.get('text', '')}")
        question_overlap = len(question_terms & text_terms) / max(1, len(question_terms))
        score = 2.0 * float(source_match) + 1.5 * float(target_valid) + 0.5 * question_overlap + 0.2 * float(relation.get("confidence", 0.0) or 0.0)
        scored.append({**fact, "hop_role": "source_side", "retrieval_score": score, "hop_score": score})
    scored.sort(key=lambda item: (-float(item["retrieval_score"]), str(item.get("fact_id", ""))))
    return scored[:limit]


def _hotpot_rank_bridge_entity_source_facts(
    row: dict[str, Any],
    facts: list[dict[str, Any]],
    *,
    source_entity: str,
    limit: int,
) -> list[dict[str, Any]]:
    question_terms = _hotpot_terms(str(row.get("question", "")))
    scored = []
    for fact in facts:
        relation = _hotpot_fact_relation_view(row, "", fact, role="bridge_discovery")
        anchor_match = _hotpot_fact_supports_entity(fact, source_entity) or _hotpot_entities_equivalent(
            relation.get("source_entity", ""),
            source_entity,
        )
        inferred_bridge = _hotpot_inferred_intermediate_from_fact(row, fact)
        bridge_role = _hotpot_ontology_role(inferred_bridge)
        bridge_valid = bool(inferred_bridge and _hotpot_bridge_role_admissible(bridge_role))
        fact_terms = _hotpot_terms(f"{fact.get('title', '')} {fact.get('text', '')}")
        question_overlap = len(question_terms & fact_terms) / max(1, len(question_terms))
        # This score is recall-oriented: fact1 should be anchored on A and
        # expose a named intermediate B. Relation precision is deferred to
        # the second-hop lookup and verifier.
        score = (
            3.0 * float(anchor_match)
            + 2.0 * float(bridge_valid)
            + 0.4 * question_overlap
            + 0.2 * float(relation.get("confidence", 0.0) or 0.0)
        )
        scored.append({
            **fact,
            "hop_role": "bridge_discovery",
            "retrieval_score": score,
            "hop_score": score,
            "inferred_bridge": inferred_bridge,
            "source_entity": source_entity,
        })
    scored.sort(key=lambda item: (-float(item["retrieval_score"]), str(item.get("fact_id", ""))))
    return scored[:limit]


def _hotpot_inferred_intermediate_from_fact(row: dict[str, Any], fact: dict[str, Any]) -> str:
    hypotheses = _hotpot_inferred_intermediate_hypotheses_from_fact(row, fact, limit=1)
    return hypotheses[0] if hypotheses else ""


def _hotpot_inferred_intermediate_hypotheses_from_fact(
    row: dict[str, Any],
    fact: dict[str, Any],
    *,
    limit: int,
) -> list[str]:
    relation = _hotpot_fact_relation_view(row, "", fact, role="source")
    qsem = _hotpot_question_semantics(row)
    source_entity = qsem.source_entity or _hotpot_question_entities(row)[0]
    source_relation = qsem.source_relation_hint or qsem.bridge_relation_hint
    answer_relation = qsem.target_relation or qsem.requested_relation
    expected_answer_role = qsem.expected_answer_role
    candidates: list[str] = []

    def add_candidate(value: Any) -> None:
        text = str(value or "").strip()
        if not text or text == "unresolved":
            return
        if source_entity and _hotpot_entities_equivalent(text, source_entity):
            return
        if answer_relation and answer_relation != "unresolved" and str(relation.get("relation", "")) == answer_relation:
            return
        role = _hotpot_ontology_role(text)
        if _hotpot_target_role_matches(role, expected_answer_role) and str(relation.get("relation", "")) == answer_relation:
            return
        if _hotpot_answer_side_value_like(row, text):
            return
        if not _hotpot_bridge_role_admissible(_hotpot_ontology_role(text)):
            return
        key = normalize_bridge(text)
        if key and all(normalize_bridge(existing) != key for existing in candidates):
            candidates.append(text)

    if source_entity:
        subject = str(relation.get("source_entity", ""))
        obj = str(relation.get("target_entity", ""))
        relation_match = (
            not source_relation
            or source_relation == "unresolved"
            or _hotpot_relation_match_type(str(relation.get("relation", "")), source_relation) in {"exact", "compatible"}
        )
        if obj and obj != "unresolved" and relation_match and _hotpot_entities_equivalent(subject, source_entity):
            add_candidate(obj)
        if subject and subject != "unresolved" and relation_match and _hotpot_entities_equivalent(obj, source_entity):
            add_candidate(subject)
        if candidates:
            return candidates[:limit]
    source = str(relation.get("source_entity", ""))
    values = [
        str(relation.get("target_entity", "")),
        *[str(item) for item in relation.get("referenced_entities", [])],
        str(fact.get("title", "")),
        *_hotpot_entities_in_text(f"{fact.get('title', '')} {fact.get('text', '')}"),
    ]
    for value in values:
        if not value or value == "unresolved":
            continue
        if source and _hotpot_entities_equivalent(value, source):
            continue
        add_candidate(value)
        if len(candidates) >= limit:
            break
    return candidates[:limit]


def _hotpot_answer_side_value_like(row: dict[str, Any], value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    q = str(row.get("question", "")).lower()
    key = normalize_bridge(text)
    if re.fullmatch(r"\d[\d,]*(?:\.\d+)?", text):
        return True
    if key in {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    }:
        return True
    if ("what city" in q or "which city" in q or "where" in q) and _hotpot_ontology_role(text) == "LOCATION":
        return True
    return False


def _hotpot_rank_relation_specific_hop2(
    row: dict[str, Any],
    inferred_bridge: str,
    facts: list[dict[str, Any]],
    *,
    reasoning_type: str,
    limit: int,
) -> list[dict[str, Any]]:
    requested_relation = _hotpot_question_relation_label(row)
    expected_role = _hotpot_expected_answer_role(row)
    question_terms = _hotpot_terms(str(row.get("question", "")))
    scored = []
    for fact in facts:
        relation = _hotpot_fact_relation_view(row, inferred_bridge, fact, role="hop2")
        subject_match = _hotpot_relation_starts_from_entity(relation, inferred_bridge)
        relation_match_type = _hotpot_relation_match_type(str(relation.get("relation", "")), requested_relation)
        relation_match = relation_match_type in {"exact", "compatible"}
        target_match = _hotpot_target_role_matches(str(relation.get("object_role", "")), expected_role)
        fact_terms = _hotpot_terms(f"{fact.get('title', '')} {fact.get('text', '')}")
        overlap = len(question_terms & fact_terms) / max(1, len(question_terms))
        if reasoning_type == "bridge_entity" and requested_relation == "unresolved":
            relation_match = str(relation.get("relation", "")) != "unresolved"
            target_match = str(relation.get("target_entity", "")) not in {"", "unresolved"}
        if reasoning_type == "bridge_attribute":
            score = (
                2.0 * float(subject_match)
                + 3.0 * float(relation_match)
                + 1.5 * float(target_match)
                + 0.3 * overlap
                + 0.2 * float(relation.get("confidence", 0.0) or 0.0)
            )
        else:
            score = (
                2.5 * float(subject_match)
                + 2.0 * float(relation_match)
                + 1.0 * float(target_match)
                + 0.4 * overlap
                + 0.2 * float(relation.get("confidence", 0.0) or 0.0)
            )
        scored.append({
            **fact,
            "hop_role": "relation_specific_hop2",
            "retrieval_score": score,
            "hop_score": score,
            "about_bridge": bool(subject_match),
            "requested_relation_match": bool(relation_match),
            "relation_match_type": relation_match_type,
            "target_role_match": bool(target_match),
            "fact_relation": str(relation.get("relation", "")),
            "expected_answer_role": expected_role,
        })
    scored.sort(key=lambda item: (-float(item["retrieval_score"]), str(item.get("fact_id", ""))))
    return scored[:limit]


def _hotpot_retrieve_relation_targets(
    row: dict[str, Any],
    constraint: dict[str, str],
    facts: list[dict[str, Any]],
    *,
    role: str,
    limit: int,
    evidence_index: HotpotEvidenceIndex | None = None,
) -> list[dict[str, Any]]:
    anchor = str(constraint.get("anchor", ""))
    relation_hint = str(constraint.get("relation", ""))
    context_title = str(constraint.get("context_title", ""))
    candidate_facts = facts
    if evidence_index is not None:
        candidate_facts = _hotpot_title_bound_constraint_facts(
            evidence_index,
            anchor=anchor,
            context_title=context_title,
            relation=relation_hint,
            query_type=role,
            topk=80,
        )
    targets: list[dict[str, Any]] = []
    for fact in candidate_facts:
        binding = _hotpot_best_constraint_binding(row, constraint, [fact], used_fact_ids=set())
        relation = _hotpot_fact_relation_view(row, "", fact, role="constraint")
        fallback_target = str(binding.get("target", "")) or str(relation.get("target_entity", ""))
        target_valid = bool(fallback_target and fallback_target != "unresolved")
        anchor_match = bool(binding.get("anchor_match"))
        relation_match = bool(binding.get("relation_match"))
        satisfied = bool(binding.get("satisfied"))
        score = (
            4.0 * float(satisfied)
            + 2.0 * float(anchor_match)
            + 1.5 * float(relation_match)
            + 0.5 * float(target_valid)
            + 1.5 * float(fact.get("anchor_tier", 9) == 0)
            + 1.0 * float(fact.get("anchor_tier", 9) == 1)
            + float(fact.get("retrieval_score", fact.get("hop_score", 0.0)) or 0.0)
        )
        targets.append({
            "constraint": constraint,
            "fact": {**fact, "hop_role": role, "retrieval_score": score, "hop_score": score},
            "fact_id": str(fact.get("fact_id", "")),
            "target": fallback_target,
            "anchor_match": anchor_match,
            "relation_match": relation_match,
            "relation_match_type": str(binding.get("relation_match_type", "")),
            "observed_anchor": str(binding.get("observed_anchor", "")),
            "observed_relation": str(binding.get("observed_relation", "")),
            "target_valid": target_valid,
            "satisfied": satisfied,
            "retrieval_score": score,
            "anchor_tier": int(fact.get("anchor_tier", 9)),
            "anchor_resolution_type": str(constraint.get("resolution_type", "")),
            "context_title": context_title,
        })
    targets.sort(key=lambda item: (int(item.get("anchor_tier", 9)), -float(item["retrieval_score"]), normalize_bridge(item.get("target", "")), str(item.get("fact_id", ""))))
    return targets[:limit]


def _hotpot_title_bound_constraint_facts(
    index: HotpotEvidenceIndex,
    *,
    anchor: str,
    context_title: str,
    relation: str,
    query_type: str,
    topk: int,
) -> list[dict[str, Any]]:
    relation_key = _hotpot_canonical_relation(relation)
    parts: list[dict[str, Any]] = []
    if context_title:
        parts.extend(_hotpot_with_anchor_tier(
            _hotpot_retrieve_by_anchor_relation(index, anchor=context_title, relation=relation_key, query_type=query_type, topk=topk),
            tier=0,
            tier_name="bound_title_relation",
        ))
        parts.extend(_hotpot_with_anchor_tier(
            _hotpot_retrieve_by_anchor(index, anchor=context_title, query_type=query_type, relation_hint=relation_key, topk=topk),
            tier=1,
            tier_name="bound_title_anchor",
        ))
    parts.extend(_hotpot_with_anchor_tier(
        _hotpot_retrieve_by_anchor_relation(index, anchor=anchor, relation=relation_key, query_type=query_type, topk=topk),
        tier=2,
        tier_name="global_anchor_relation",
    ))
    parts.extend(_hotpot_with_anchor_tier(
        _hotpot_retrieve_by_anchor(index, anchor=anchor, query_type=query_type, relation_hint=relation_key, topk=topk),
        tier=3,
        tier_name="global_anchor_only",
    ))
    return _hotpot_merge_fact_pool(parts)[:topk]


def _hotpot_with_anchor_tier(facts: list[dict[str, Any]], *, tier: int, tier_name: str) -> list[dict[str, Any]]:
    return [
        {
            **fact,
            "anchor_tier": min(int(fact.get("anchor_tier", tier)), tier),
            "anchor_tier_name": str(fact.get("anchor_tier_name", tier_name)),
        }
        for fact in facts
    ]


def _hotpot_intersect_constraint_targets(
    targets_1: list[dict[str, Any]],
    targets_2: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for left in targets_1:
        if not bool(left.get("satisfied")):
            continue
        for right in targets_2:
            if not bool(right.get("satisfied")):
                continue
            if str(left.get("fact_id", "")) == str(right.get("fact_id", "")):
                continue
            if _hotpot_entities_equivalent(left.get("target", ""), right.get("target", "")):
                pairs.append((left, right))
    pairs.sort(key=lambda pair: (
        -(float(pair[0].get("retrieval_score", 0.0)) + float(pair[1].get("retrieval_score", 0.0))),
        normalize_bridge(pair[0].get("target", "")),
        str(pair[0].get("fact_id", "")),
        str(pair[1].get("fact_id", "")),
    ))
    return pairs


def _hotpot_constraint_target_as_fact(item: dict[str, Any], *, role: str) -> dict[str, Any]:
    fact = dict(item.get("fact", {}))
    fact["hop_role"] = role
    fact["retrieval_score"] = float(item.get("retrieval_score", 0.0) or 0.0)
    fact["hop_score"] = float(item.get("retrieval_score", 0.0) or 0.0)
    fact["constraint_target"] = str(item.get("target", ""))
    fact["constraint_satisfied"] = bool(item.get("satisfied"))
    return fact


def _hotpot_constraint_target_as_dict(item: dict[str, Any]) -> dict[str, Any]:
    constraint = item.get("constraint", {}) if isinstance(item.get("constraint"), dict) else {}
    return {
        "fact_id": str(item.get("fact_id", "")),
        "target": str(item.get("target", "")),
        "source_var": str(constraint.get("source_var", "")),
        "target_var": str(constraint.get("target_var", "")),
        "depends_on": str(constraint.get("depends_on", "")),
        "anchor_match": bool(item.get("anchor_match")),
        "relation_match": bool(item.get("relation_match")),
        "relation_match_type": str(item.get("relation_match_type", "")),
        "observed_anchor": str(item.get("observed_anchor", "")),
        "observed_relation": str(item.get("observed_relation", "")),
        "satisfied": bool(item.get("satisfied")),
        "retrieval_score": float(item.get("retrieval_score", 0.0) or 0.0),
    }


def _hotpot_constraint_target_pair_as_chain_pair(
    row: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    hop1 = _hotpot_constraint_target_as_fact(left, role="constraint_1")
    hop2 = _hotpot_constraint_target_as_fact(right, role="constraint_2")
    semantic_verification = _hotpot_verify_semantic_chain(row, "", hop1, hop2)
    semantic_score = _hotpot_semantic_chain_score(semantic_verification)
    retrieval_score = float(left.get("retrieval_score", 0.0) or 0.0) + float(right.get("retrieval_score", 0.0) or 0.0)
    return {
        "hop1": hop1,
        "hop2": hop2,
        "retrieval_score": retrieval_score,
        "semantic_chain_verification": semantic_verification,
        "semantic_chain_score": semantic_score,
        "compatibility_score": semantic_score,
        "chain_score": semantic_score,
    }


def _hotpot_rank_facts_for_constraint(
    row: dict[str, Any],
    constraint: dict[str, str],
    facts: list[dict[str, Any]],
    *,
    role: str,
    limit: int,
) -> list[dict[str, Any]]:
    scored = []
    for fact in facts:
        binding = _hotpot_best_constraint_binding(row, constraint, [fact], used_fact_ids=set())
        score = float(binding.get("score", 0.0) or 0.0)
        scored.append({**fact, "hop_role": role, "retrieval_score": score, "hop_score": score})
    scored.sort(key=lambda item: (-float(item["retrieval_score"]), str(item.get("fact_id", ""))))
    return scored[:limit]


def _hotpot_rank_hop1_candidates(row: dict[str, Any], bridge: str, facts: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    question_terms = _hotpot_terms(str(row.get("question", "")))
    bridge_key = normalize_bridge(bridge)
    scored = []
    for fact in facts:
        title_key = normalize_bridge(fact.get("title", ""))
        text_key = normalize_bridge(fact.get("text", ""))
        text_terms = _hotpot_terms(f"{fact.get('title', '')} {fact.get('text', '')}")
        entity_overlap = len(question_terms & text_terms) / max(1, len(question_terms))
        bridge_mention = 1.0 if bridge_key and (bridge_key == title_key or bridge_key in text_key) else 0.0
        title_endpoint = 1.0 if title_key and title_key in {normalize_bridge(item) for item in row.get("entities", [])} else 0.0
        retrieval_score = 1.5 * bridge_mention + 1.0 * entity_overlap + 0.5 * title_endpoint
        scored.append({**fact, "hop_role": "hop1", "retrieval_score": retrieval_score, "hop_score": retrieval_score})
    scored.sort(key=lambda item: (-float(item["retrieval_score"]), str(item.get("fact_id", ""))))
    return scored[:limit]


def _hotpot_rank_hop2_candidates(row: dict[str, Any], bridge: str, facts: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    question = str(row.get("question", ""))
    question_terms = _hotpot_terms(question)
    target_terms = _hotpot_relation_target_terms(row)
    bridge_key = normalize_bridge(bridge)
    scored = []
    for fact in facts:
        title_key = normalize_bridge(fact.get("title", ""))
        text_key = normalize_bridge(fact.get("text", ""))
        text_terms = _hotpot_terms(f"{fact.get('title', '')} {fact.get('text', '')}")
        bridge_mention = 1.0 if bridge_key and (bridge_key == title_key or bridge_key in text_key) else 0.0
        target_match = len(target_terms & text_terms) / max(1, len(target_terms)) if target_terms else 0.0
        question_relevance = len(question_terms & text_terms) / max(1, len(question_terms))
        answer_type_match = _hotpot_answer_type_match(question, str(fact.get("text", "")))
        retrieval_score = 1.2 * bridge_mention + 1.6 * target_match + 0.8 * answer_type_match + 0.4 * question_relevance
        scored.append({**fact, "hop_role": "hop2", "retrieval_score": retrieval_score, "hop_score": retrieval_score})
    scored.sort(key=lambda item: (-float(item["retrieval_score"]), str(item.get("fact_id", ""))))
    return scored[:limit]


def _hotpot_rank_chain_pairs(
    row: dict[str, Any],
    bridge: str,
    hop1_pool: list[dict[str, Any]],
    hop2_pool: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs = []
    for hop1 in hop1_pool:
        for hop2 in hop2_pool:
            hop1_id = str(hop1.get("fact_id") or "").strip()
            hop2_id = str(hop2.get("fact_id") or "").strip()
            if not hop1_id or not hop2_id or hop1_id == hop2_id:
                continue
            semantic_verification = _hotpot_verify_semantic_chain(row, bridge, hop1, hop2)
            semantic_score = _hotpot_semantic_chain_score(semantic_verification)
            retrieval_score = _hotpot_pair_retrieval_score(row, bridge, hop1, hop2)
            pairs.append({
                "hop1": hop1,
                "hop2": hop2,
                "retrieval_score": retrieval_score,
                "semantic_chain_verification": semantic_verification,
                "semantic_chain_score": semantic_score,
                "compatibility_score": semantic_score,
                "chain_score": semantic_score,
            })
    pairs.sort(key=lambda item: (
        -float(item["semantic_chain_score"]),
        -float(item["retrieval_score"]),
        str(item["hop1"].get("fact_id")),
        str(item["hop2"].get("fact_id")),
    ))
    return pairs


def _hotpot_pair_retrieval_score(row: dict[str, Any], bridge: str, hop1: dict[str, Any], hop2: dict[str, Any]) -> float:
    bridge_key = normalize_bridge(bridge)
    hop1_text = normalize_bridge(f"{hop1.get('title', '')} {hop1.get('text', '')}")
    hop2_text = normalize_bridge(f"{hop2.get('title', '')} {hop2.get('text', '')}")
    bridge_overlap = 0.0
    if bridge_key and bridge_key in hop1_text:
        bridge_overlap += 0.5
    if bridge_key and bridge_key in hop2_text:
        bridge_overlap += 0.5
    target_terms = _hotpot_target_terms(str(row.get("question", "")))
    target_overlap = 0.0
    if target_terms:
        target_overlap = len(target_terms & _hotpot_terms(f"{hop1.get('text', '')} {hop2.get('text', '')}")) / max(1, len(target_terms))
    return (
        float(hop1.get("retrieval_score", hop1.get("hop_score", 0.0)) or 0.0)
        + float(hop2.get("retrieval_score", hop2.get("hop_score", 0.0)) or 0.0)
        + bridge_overlap
        + target_overlap
    )


def _hotpot_reasoning_ir(
    *,
    row: dict[str, Any],
    reasoning_type: str,
    candidate_bridge: str,
    inferred_bridge: str,
    hop1: dict[str, Any],
    hop2: dict[str, Any],
    hop1_relation: dict[str, Any],
    hop2_relation: dict[str, Any],
    type_details: dict[str, Any],
) -> ReasoningIR:
    qsem = _hotpot_question_semantics(row)
    constraint_targets = type_details.get("constraint_targets", [])
    if isinstance(constraint_targets, list) and constraint_targets:
        constraint_answer = str(constraint_targets[0])
    else:
        constraint_answer = ""
    answer_candidate = str(type_details.get("derived_answer") or constraint_answer or "")
    return ReasoningIR(
        topology={
            "family": _hotpot_reasoning_family_for_type(reasoning_type),
            "type": reasoning_type,
            "confidence": qsem.confidence,
        },
        question={
            "entities": list(qsem.entities_in_question),
            "requested_relation": qsem.requested_relation or qsem.target_relation or qsem.compared_relation,
            "comparison_operator": qsem.comparison_operator,
            "answer_type": qsem.expected_answer_role,
            "answer_mode": qsem.answer_mode,
        },
        constraints=[constraint.as_dict() for constraint in qsem.constraints],
        facts={
            "fact1": _hotpot_fact_binding(hop1, hop1_relation),
            "fact2": _hotpot_fact_binding(hop2, hop2_relation),
        },
        bindings={
            "candidate_bridge": candidate_bridge,
            "inferred_bridge": inferred_bridge,
            "shared_target": type_details.get("shared_entity_aliases", []),
            "attribute_values": {
                "a": type_details.get("attribute_a", ""),
                "b": type_details.get("attribute_b", ""),
            },
            "variable_bindings": type_details.get("variable_bindings", {}),
        },
        derived={
            "answer_candidate": answer_candidate,
            "answer_supported_proposal": bool(type_details.get("answer_supported", False)),
        },
    )


def _hotpot_programmatic_gates(ir: ReasoningIR, *, hop1: dict[str, Any], hop2: dict[str, Any], type_details: dict[str, Any]) -> dict[str, Any]:
    payload = ir.as_dict()
    reasoning_type = str(payload["topology"].get("type", ""))
    gate_a = _hotpot_gate_endpoint_grounding(payload, hop1=hop1, hop2=hop2)
    gate_b = _hotpot_gate_relation_compatibility(payload)
    gate_c = _hotpot_gate_topology_consistency(payload, type_details=type_details)
    gate_d = _hotpot_gate_answer_grounding(payload, type_details=type_details)
    gates = {
        "endpoint_grounding": gate_a,
        "relation_compatibility": gate_b,
        "topology_consistency": gate_c,
        "answer_grounding": gate_d,
    }
    failure_code = ""
    for name, result in gates.items():
        if not bool(result.get("pass")):
            failure_code = str(result.get("failure_code") or name)
            break
    return {
        "reasoning_type": reasoning_type,
        "pass": all(bool(result.get("pass")) for result in gates.values()),
        "failure_code": failure_code,
        "gates": gates,
    }


def _hotpot_gate_endpoint_grounding(ir: dict[str, Any], *, hop1: dict[str, Any], hop2: dict[str, Any]) -> dict[str, Any]:
    facts = ir.get("facts", {})
    bindings = ir.get("bindings", {})
    fact1_semantics = facts.get("fact1", {}).get("semantics", {}) if isinstance(facts.get("fact1"), dict) else {}
    fact2_semantics = facts.get("fact2", {}).get("semantics", {}) if isinstance(facts.get("fact2"), dict) else {}
    checks = {
        "fact1_subject": _hotpot_value_grounded(fact1_semantics.get("subject") or fact1_semantics.get("source_entity"), hop1),
        "fact1_object": _hotpot_value_grounded(fact1_semantics.get("object") or fact1_semantics.get("target_entity"), hop1),
        "fact2_subject": _hotpot_value_grounded(fact2_semantics.get("subject") or fact2_semantics.get("source_entity"), hop2),
        "fact2_object": _hotpot_value_grounded(fact2_semantics.get("object") or fact2_semantics.get("target_entity"), hop2),
    }
    bridge = str(bindings.get("inferred_bridge", ""))
    if bridge:
        checks["inferred_bridge"] = _hotpot_value_grounded(bridge, hop1) or _hotpot_value_grounded(bridge, hop2)
    answer = str(ir.get("derived", {}).get("answer_candidate", ""))
    if answer and str(ir.get("topology", {}).get("type")) not in {"comparison_attribute", "shared_property_yesno", "set_intersection"}:
        checks["answer_candidate"] = _hotpot_value_grounded(answer, hop1) or _hotpot_value_grounded(answer, hop2)
    passed = all(bool(value) for value in checks.values())
    return {"pass": passed, "failure_code": "" if passed else "ungrounded_endpoint", "checks": checks}


def _hotpot_gate_relation_compatibility(ir: dict[str, Any]) -> dict[str, Any]:
    reasoning_type = str(ir.get("topology", {}).get("type", ""))
    question = ir.get("question", {})
    requested = str(question.get("requested_relation") or "unresolved")
    fact1 = ir.get("facts", {}).get("fact1", {}).get("semantics", {})
    fact2 = ir.get("facts", {}).get("fact2", {}).get("semantics", {})
    if reasoning_type in {"bridge_entity", "bridge_attribute"}:
        match_type = _hotpot_relation_match_type(str(fact2.get("relation", "")), requested)
        passed = requested == "unresolved" or match_type in {"exact", "compatible"}
        return {"pass": passed, "failure_code": "" if passed else "requested_relation_mismatch", "requested_relation": requested, "fact_relation": fact2.get("relation", ""), "relation_match_type": match_type}
    if reasoning_type in {"comparison_attribute", "shared_property_yesno", "set_intersection"}:
        relation_a = _hotpot_relation_match_type(str(fact1.get("relation", "")), requested)
        relation_b = _hotpot_relation_match_type(str(fact2.get("relation", "")), requested)
        passed = requested == "unresolved" or relation_a in {"exact", "compatible"} or relation_b in {"exact", "compatible"}
        return {"pass": passed, "failure_code": "" if passed else "requested_relation_mismatch", "requested_relation": requested, "fact1_match": relation_a, "fact2_match": relation_b}
    bindings = ir.get("bindings", {}).get("variable_bindings", {})
    passed = bool(bindings) or bool(ir.get("constraints"))
    return {"pass": passed, "failure_code": "" if passed else "constraint_relation_unbound"}


def _hotpot_gate_topology_consistency(ir: dict[str, Any], *, type_details: dict[str, Any]) -> dict[str, Any]:
    reasoning_type = str(ir.get("topology", {}).get("type", ""))
    if reasoning_type in {"bridge_entity", "bridge_attribute"}:
        passed = bool(type_details.get("hop1_relation_valid")) and bool(type_details.get("hop2_relation_valid")) and bool(type_details.get("bridge_alignment", type_details.get("chain_complete")))
        return {"pass": passed, "failure_code": "" if passed else "topology_inconsistent", "expected": "source_to_bridge_then_bridge_to_answer"}
    if reasoning_type == "comparison_attribute":
        passed = bool(type_details.get("entity_a_supported")) and bool(type_details.get("entity_b_supported")) and bool(type_details.get("comparison_valid"))
        return {"pass": passed, "failure_code": "" if passed else "topology_inconsistent", "expected": "parallel_attribute_comparison"}
    if reasoning_type == "shared_property_yesno":
        passed = bool(type_details.get("entity_a_supported")) and bool(type_details.get("entity_b_supported"))
        return {"pass": passed, "failure_code": "" if passed else "topology_inconsistent", "expected": "parallel_shared_property"}
    if reasoning_type == "set_intersection":
        passed = bool(type_details.get("entity_a_supported")) and bool(type_details.get("entity_b_supported"))
        return {"pass": passed, "failure_code": "" if passed else "topology_inconsistent", "expected": "parallel_set_intersection"}
    bindings = type_details.get("constraint_bindings", [])
    passed = bool(bindings) and all(bool(item.get("satisfied")) for item in bindings)
    return {"pass": passed, "failure_code": "" if passed else "constraint_not_satisfied", "expected": "all_constraints_satisfied"}


def _hotpot_gate_answer_grounding(ir: dict[str, Any], *, type_details: dict[str, Any]) -> dict[str, Any]:
    reasoning_type = str(ir.get("topology", {}).get("type", ""))
    if reasoning_type in {"bridge_entity", "bridge_attribute"}:
        hop2 = ir.get("facts", {}).get("fact2", {}).get("semantics", {})
        answer = str(ir.get("derived", {}).get("answer_candidate", "") or type_details.get("derived_answer", ""))
        obj = str(hop2.get("object") or hop2.get("target_entity") or "")
        passed = bool(obj) and (not answer or _hotpot_entities_equivalent(answer, obj) or normalize_bridge(answer) in normalize_bridge(obj) or normalize_bridge(obj) in normalize_bridge(answer))
        return {"pass": passed, "failure_code": "" if passed else "answer_not_derivable", "answer_candidate": answer, "expected_object": obj}
    if reasoning_type == "comparison_attribute":
        answer = _hotpot_comparison_decision(type_details)
        passed = bool(answer) and bool(type_details.get("comparison_valid"))
        return {"pass": passed, "failure_code": "" if passed else "answer_not_derivable", "deterministic_answer": answer}
    if reasoning_type == "shared_property_yesno":
        passed = "answer_supported" in type_details
        return {"pass": passed, "failure_code": "" if passed else "answer_not_derivable", "deterministic_answer": "yes" if type_details.get("answer_supported") else "no"}
    if reasoning_type == "set_intersection":
        a = set(_hotpot_terms(str(type_details.get("attribute_a", ""))))
        b = set(_hotpot_terms(str(type_details.get("attribute_b", ""))))
        passed = bool(a & b) if type_details.get("answer_supported") else True
        return {"pass": passed, "failure_code": "" if passed else "answer_not_derivable", "intersection": sorted(a & b)}
    answer_var = str(type_details.get("answer_var", ""))
    variable_bindings = type_details.get("variable_bindings", {})
    passed = bool(answer_var and isinstance(variable_bindings, dict) and variable_bindings.get(answer_var))
    if not answer_var:
        targets = type_details.get("constraint_targets", [])
        passed = bool(targets)
    return {"pass": passed, "failure_code": "" if passed else "answer_not_derivable", "answer_var": answer_var, "variable_bindings": variable_bindings}


def _hotpot_value_grounded(value: Any, fact: dict[str, Any]) -> bool:
    text = str(value or "").strip()
    if not text or text == "unresolved":
        return True
    value_key = normalize_bridge(text)
    if not value_key:
        return True
    haystacks = [
        normalize_bridge(fact.get("title", "")),
        normalize_bridge(fact.get("text", "")),
    ]
    aliases = _hotpot_entity_aliases(text)
    for haystack in haystacks:
        if value_key and value_key in haystack:
            return True
        if any(alias and alias in haystack for alias in aliases):
            return True
    return False


def _hotpot_verify_semantic_chain(row: dict[str, Any], bridge: str, hop1: dict[str, Any], hop2: dict[str, Any]) -> dict[str, Any]:
    question = str(row.get("question", ""))
    hop1_terms = _hotpot_terms(f"{hop1.get('title', '')} {hop1.get('text', '')}")
    hop2_terms = _hotpot_terms(f"{hop2.get('title', '')} {hop2.get('text', '')}")
    reasoning_type = _hotpot_reasoning_type(row)
    hop1_bridge_match = _hotpot_fact_bridge_match(bridge, hop1)
    hop2_bridge_match = _hotpot_fact_bridge_match(bridge, hop2)
    hop1_relation = _hotpot_fact_relation_view(row, bridge, hop1, role="hop1")
    hop2_relation = _hotpot_fact_relation_view(row, bridge, hop2, role="hop2")
    different_evidence = str(hop1.get("fact_id", "")) != str(hop2.get("fact_id", ""))
    different_titles = normalize_bridge(hop1.get("title", "")) != normalize_bridge(hop2.get("title", ""))
    inferred_bridge = _hotpot_infer_bridge_from_relations(hop1_relation, hop2_relation)
    bridge_entity_aligned = bool(inferred_bridge)
    bridge_hypothesis_match = _hotpot_entities_equivalent(bridge, inferred_bridge)
    bridge_match_type = _hotpot_bridge_hypothesis_match_type(bridge, inferred_bridge, hop1_bridge_match, hop2_bridge_match)
    type_details = _hotpot_type_specific_verification(
        row=row,
        reasoning_type=reasoning_type,
        candidate_bridge=bridge,
        inferred_bridge=inferred_bridge,
        hop1=hop1,
        hop2=hop2,
        hop1_relation=hop1_relation,
        hop2_relation=hop2_relation,
        hop1_terms=hop1_terms,
        hop2_terms=hop2_terms,
        bridge_entity_aligned=bridge_entity_aligned,
    )
    reasoning_ir = _hotpot_reasoning_ir(
        row=row,
        reasoning_type=reasoning_type,
        candidate_bridge=bridge,
        inferred_bridge=inferred_bridge,
        hop1=hop1,
        hop2=hop2,
        hop1_relation=hop1_relation,
        hop2_relation=hop2_relation,
        type_details=type_details,
    )
    programmatic_gates = _hotpot_programmatic_gates(
        reasoning_ir,
        hop1=hop1,
        hop2=hop2,
        type_details=type_details,
    )
    reasoning_candidate = _hotpot_construct_reasoning_candidate(
        row=row,
        reasoning_type=reasoning_type,
        candidate_bridge=bridge,
        inferred_bridge=inferred_bridge,
        hop1=hop1,
        hop2=hop2,
        hop1_relation=hop1_relation,
        hop2_relation=hop2_relation,
        type_details=type_details,
    )
    hop1_relation_valid = bool(type_details.get("hop1_relation_valid", type_details.get("entity_a_supported", False)))
    hop2_relation_valid = bool(type_details.get("hop2_relation_valid", type_details.get("entity_b_supported", False)))
    chain_complete = bool(type_details.get("chain_complete", type_details.get("comparison_valid", False))) and different_evidence
    contradiction = False
    derived_answer = _hotpot_derive_answer_hint(question, hop1, hop2)
    question_relation_supported = bool(type_details.get("answer_supported", False)) if chain_complete else False
    gate_pass = bool(programmatic_gates.get("pass"))
    evidence_valid = bool(type_details.get("evidence_valid", chain_complete)) and gate_pass
    answer_supported = question_relation_supported and gate_pass
    confidence_parts = [
        float(evidence_valid),
        float(answer_supported),
        float(hop1_relation_valid),
        float(hop2_relation_valid),
        float(bridge_entity_aligned if reasoning_type in {"bridge_entity", "bridge_attribute"} else True),
        float(chain_complete),
        float(question_relation_supported),
        0.25 if different_titles else 0.0,
    ]
    confidence = sum(confidence_parts) / len(confidence_parts)
    score = _hotpot_verification_score(
        evidence_valid=evidence_valid,
        reasoning_complete=chain_complete,
        answer_supported=answer_supported,
        confidence=confidence,
        contradiction=contradiction,
    )
    failure_code = "" if evidence_valid and chain_complete and answer_supported else str(
        programmatic_gates.get("failure_code")
        or type_details.get("failure_code")
        or type_details.get("failure_reason")
        or "not_supported"
    )
    verification_result = VerificationResult(
        reasoning_type=reasoning_type,
        evidence_valid=evidence_valid,
        reasoning_complete=chain_complete,
        answer_supported=answer_supported,
        derived_answer=derived_answer,
        confidence=round(confidence, 4),
        score=score,
        failure_code=failure_code,
        details={
            "candidate_bridge": bridge,
            "inferred_bridge": inferred_bridge,
            "bridge_hypothesis_match": bridge_hypothesis_match,
            "bridge_match_type": bridge_match_type,
            "type_specific_details": type_details,
            "reasoning_ir": reasoning_ir.as_dict(),
            "programmatic_gates": programmatic_gates,
        },
    )
    return {
        "reasoning_type": reasoning_type,
        "reasoning_candidate": reasoning_candidate.as_dict(),
        "verification_result": verification_result.as_dict(),
        "candidate_bridge": bridge,
        "inferred_bridge": inferred_bridge,
        "bridge_hypothesis_match": bridge_hypothesis_match,
        "evidence_chain_valid": evidence_valid,
        "evidence_valid": evidence_valid,
        "answer_supported": answer_supported,
        "failure_reason": failure_code,
        "failure_code": failure_code,
        "type_specific_details": type_details,
        "reasoning_ir": reasoning_ir.as_dict(),
        "programmatic_gates": programmatic_gates,
        "gate_results": programmatic_gates.get("gates", {}),
        "hop1_relation_valid": hop1_relation_valid,
        "hop2_relation_valid": hop2_relation_valid,
        "bridge_entity_aligned": bridge_entity_aligned,
        "bridge_match_type": bridge_match_type,
        "chain_complete": chain_complete,
        "question_relation_supported": question_relation_supported,
        "derived_answer": derived_answer,
        "contradiction": contradiction,
        "confidence": round(confidence, 4),
        "score": score,
        "entity_resolution": {
            "bridge_aliases": sorted(_hotpot_entity_aliases(bridge)),
            "hop1_bridge_match": hop1_bridge_match,
            "hop2_bridge_match": hop2_bridge_match,
        },
        "relation_endpoints": {
            "hop1": hop1_relation,
            "hop2": hop2_relation,
            "bridge_endpoint_equivalent": bridge_entity_aligned,
        },
    }


def _hotpot_construct_reasoning_candidate(
    *,
    row: dict[str, Any],
    reasoning_type: str,
    candidate_bridge: str,
    inferred_bridge: str,
    hop1: dict[str, Any],
    hop2: dict[str, Any],
    hop1_relation: dict[str, Any],
    hop2_relation: dict[str, Any],
    type_details: dict[str, Any],
) -> ReasoningCandidate:
    candidate_id = "|".join([
        reasoning_type,
        str(hop1.get("fact_id", "hop1")),
        str(hop2.get("fact_id", "hop2")),
        normalize_bridge(candidate_bridge) or "no_bridge",
    ])
    fact1 = _hotpot_fact_binding(hop1, hop1_relation)
    fact2 = _hotpot_fact_binding(hop2, hop2_relation)
    derived_answer = _hotpot_derive_answer_hint(str(row.get("question", "")), hop1, hop2)
    if reasoning_type == "bridge_entity":
        evidence_bindings = {
            "source_entity": hop1_relation.get("source_entity", ""),
            "fact1": fact1,
            "inferred_bridge": inferred_bridge,
            "fact2": fact2,
        }
        derived_decision = {"derived_answer": derived_answer or hop2_relation.get("target_entity", "")}
    elif reasoning_type == "bridge_attribute":
        evidence_bindings = {
            "source_entity": hop1_relation.get("source_entity", ""),
            "fact1": fact1,
            "inferred_bridge": inferred_bridge,
            "fact2": fact2,
            "target_relation": hop2_relation.get("relation", ""),
        }
        derived_decision = {"derived_answer": derived_answer or hop2_relation.get("target_entity", "")}
    elif reasoning_type == "comparison_attribute":
        evidence_bindings = {
            "entity_a": type_details.get("entity_a", ""),
            "fact_a": fact1,
            "value_a": type_details.get("attribute_a", ""),
            "entity_b": type_details.get("entity_b", ""),
            "fact_b": fact2,
            "value_b": type_details.get("attribute_b", ""),
        }
        derived_decision = {
            "comparison_operator": type_details.get("comparison_operator", ""),
            "derived_answer": _hotpot_comparison_decision(type_details),
        }
    elif reasoning_type == "shared_property_yesno":
        evidence_bindings = {
            "entity_a": type_details.get("entity_a", ""),
            "property_support_a": fact1,
            "entity_b": type_details.get("entity_b", ""),
            "property_support_b": fact2,
            "shared_property": type_details.get("attribute_a") or type_details.get("attribute_b") or "",
        }
        derived_decision = {"derived_boolean": bool(type_details.get("answer_supported"))}
    elif reasoning_type == "set_intersection":
        values_a = sorted(_hotpot_terms(str(type_details.get("attribute_a", ""))))
        values_b = sorted(_hotpot_terms(str(type_details.get("attribute_b", ""))))
        intersection = sorted(set(values_a) & set(values_b))
        evidence_bindings = {
            "entity_a": type_details.get("entity_a", ""),
            "value_set_a": values_a,
            "entity_b": type_details.get("entity_b", ""),
            "value_set_b": values_b,
        }
        derived_decision = {"intersection": intersection, "derived_answer": " ".join(intersection)}
    else:
        shared = list(type_details.get("shared_entity_aliases", []))
        evidence_bindings = {
            "candidate_target": shared[0] if shared else "",
            "constraint_1": str(row.get("question", "")),
            "support_1": fact1,
            "constraint_2": str(row.get("question", "")),
            "support_2": fact2,
        }
        derived_decision = {
            "all_constraints_satisfied": bool(type_details.get("evidence_valid")),
            "derived_answer": shared[0] if shared else "",
        }
    return ReasoningCandidate(
        candidate_id=candidate_id,
        reasoning_type=reasoning_type,
        evidence_bindings=evidence_bindings,
        derived_decision=derived_decision,
        construction_score=_hotpot_construction_score(type_details),
        metadata={
            "candidate_bridge_hypothesis": candidate_bridge,
            "candidate_bridge_is_hypothesis_only": True,
            "hop1_fact_id": str(hop1.get("fact_id", "")),
            "hop2_fact_id": str(hop2.get("fact_id", "")),
        },
    )


def _hotpot_fact_binding(fact: dict[str, Any], relation: dict[str, Any]) -> dict[str, Any]:
    return {
        "fact_id": str(fact.get("fact_id", "")),
        "title": str(fact.get("title", "")),
        "text": str(fact.get("text", "")),
        "semantics": dict(relation),
    }


def _hotpot_construction_score(details: dict[str, Any]) -> float:
    signals = [
        bool(details.get("evidence_valid")),
        bool(details.get("chain_complete")),
        bool(details.get("answer_supported")),
        bool(details.get("hop1_relation_valid", details.get("entity_a_supported", False))),
        bool(details.get("hop2_relation_valid", details.get("entity_b_supported", False))),
    ]
    return round(sum(float(item) for item in signals) / len(signals), 4)


def _hotpot_comparison_decision(details: dict[str, Any]) -> str:
    op = str(details.get("comparison_operator", "compare"))
    a = str(details.get("attribute_a", ""))
    b = str(details.get("attribute_b", ""))
    if op == "same":
        return "yes" if a and b and (_hotpot_terms(a) & _hotpot_terms(b) or normalize_bridge(a) == normalize_bridge(b)) else "no"
    return "supported" if bool(details.get("answer_supported")) else "unsupported"


def _hotpot_verification_score(
    *,
    evidence_valid: bool,
    reasoning_complete: bool,
    answer_supported: bool,
    confidence: float,
    contradiction: bool,
) -> float:
    raw = (
        0.25 * float(evidence_valid)
        + 0.25 * float(reasoning_complete)
        + 0.35 * float(answer_supported)
        + 0.15 * float(confidence)
        - 0.35 * float(contradiction)
    )
    return round(clamp_score(raw), 4)


def _hotpot_semantic_chain_score(verdict: dict[str, Any]) -> float:
    reasoning_type = str(verdict.get("reasoning_type", ""))
    details = verdict.get("type_specific_details", {})
    if not isinstance(details, dict):
        details = {}
    if reasoning_type in {"bridge_entity", "bridge_attribute"}:
        return _hotpot_bridge_family_score(verdict, details)
    if reasoning_type == "comparison_attribute":
        return _hotpot_comparison_score(details)
    if reasoning_type == "shared_property_yesno":
        return _hotpot_shared_property_score(details)
    if reasoning_type == "set_intersection":
        return _hotpot_intersection_score(details)
    if reasoning_type == "multi_constraint_lookup":
        return _hotpot_constraint_score(details)
    return (
        1.5 * float(bool(verdict.get("evidence_valid")))
        + 2.0 * float(bool(verdict.get("answer_supported")))
        + 1.0 * float(bool(verdict.get("hop1_relation_valid")))
        + 1.0 * float(bool(verdict.get("hop2_relation_valid")))
        + 1.0 * float(bool(verdict.get("bridge_entity_aligned")) if verdict.get("reasoning_type") in {"bridge_entity", "bridge_attribute"} else True)
        + 1.5 * float(bool(verdict.get("chain_complete")))
        + 2.0 * float(bool(verdict.get("question_relation_supported")))
        - 2.0 * float(bool(verdict.get("contradiction")))
        + 0.25 * float(verdict.get("confidence", 0.0) or 0.0)
    )


def _hotpot_bridge_family_score(verdict: dict[str, Any], details: dict[str, Any]) -> float:
    # S = fact1 + fact2 + intermediate + joint-entailment.
    fact1 = float(bool(details.get("hop1_relation_valid", verdict.get("hop1_relation_valid"))))
    fact2 = float(bool(details.get("hop2_relation_valid", verdict.get("hop2_relation_valid"))))
    intermediate = float(bool(details.get("bridge_alignment", verdict.get("bridge_entity_aligned"))))
    joint = float(bool(details.get("answer_supported", verdict.get("question_relation_supported"))))
    relation_bonus = 0.5 * float(details.get("relation_match_type") in {"exact", "compatible"})
    target_bonus = 0.5 * float(bool(details.get("hop2_target_type_matches", False)))
    return fact1 + fact2 + intermediate + joint + relation_bonus + target_bonus


def _hotpot_comparison_score(details: dict[str, Any]) -> float:
    # S = factA + factB + relation-match + comparison-validity.
    fact_a = float(bool(details.get("entity_a_supported")))
    fact_b = float(bool(details.get("entity_b_supported")))
    relation = float(bool(details.get("comparison_valid")))
    comparison = float(bool(details.get("answer_supported")))
    return fact_a + fact_b + relation + comparison


def _hotpot_shared_property_score(details: dict[str, Any]) -> float:
    # S = P(A) + P(B) + joint-property.
    prop_a = float(bool(details.get("entity_a_supported")))
    prop_b = float(bool(details.get("entity_b_supported")))
    joint = float(bool(details.get("answer_supported")))
    return prop_a + prop_b + joint


def _hotpot_intersection_score(details: dict[str, Any]) -> float:
    # S = setA + setB + intersection-support.
    set_a = float(bool(details.get("entity_a_supported")))
    set_b = float(bool(details.get("entity_b_supported")))
    intersection = float(bool(details.get("answer_supported")))
    return set_a + set_b + intersection


def _hotpot_constraint_score(details: dict[str, Any]) -> float:
    # S = sum_i S_constraint_i + S_same-target.
    bindings = details.get("constraint_bindings", [])
    if not isinstance(bindings, list):
        bindings = []
    constraint_score = 0.0
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        constraint_score += (
            1.0 * float(bool(binding.get("anchor_match")))
            + 1.0 * float(bool(binding.get("relation_match")))
            + 0.5 * float(bool(binding.get("target_valid")))
        )
    same_target = 1.0 * float(bool(details.get("same_target")))
    return constraint_score + same_target


def _hotpot_chain_validity_score(row: dict[str, Any], bridge: str, hop1: dict[str, Any], hop2: dict[str, Any]) -> float:
    return _hotpot_semantic_chain_score(_hotpot_verify_semantic_chain(row, bridge, hop1, hop2))


def _hotpot_reasoning_type(row: dict[str, Any]) -> str:
    return _hotpot_topology_classification(row)["reasoning_type"]


def _hotpot_reasoning_family_for_type(reasoning_type: str) -> str:
    if reasoning_type in {"bridge_entity", "bridge_attribute"}:
        return "compositional"
    if reasoning_type in {"comparison_attribute", "shared_property_yesno", "set_intersection"}:
        return "parallel"
    if reasoning_type == "multi_constraint_lookup":
        return "constraint"
    return "compositional"


def _hotpot_answer_mode(row: dict[str, Any], reasoning_type: str) -> str:
    q = str(row.get("question", "")).lower()
    if reasoning_type == "shared_property_yesno" or q.startswith(("are ", "is ", "was ", "were ", "do ", "does ", "did ")):
        return "yes_no"
    if reasoning_type in {"comparison_attribute", "set_intersection"}:
        return "comparison"
    if reasoning_type == "multi_constraint_lookup":
        return "entity_lookup"
    return "short_answer"


def _hotpot_topology_classification(row: dict[str, Any]) -> dict[str, Any]:
    q = str(row.get("question", "")).lower()
    entity_a, entity_b = _hotpot_question_entities(row)
    entities = tuple(entity for entity in (entity_a, entity_b) if entity)
    family = "compositional"
    reasoning_type = "bridge_entity"
    confidence = 0.72
    if re.search(r"^(are|were|was|is|do|does|did)\b", q) and re.search(r"\bboth\b|\band\b|\bsame\b", q):
        family = "parallel"
        if re.search(r"\bboth\b", q):
            reasoning_type = "shared_property_yesno"
            confidence = 0.88
        else:
            reasoning_type = "comparison_attribute"
            confidence = 0.82
    elif re.search(r"\b(common|share|shared|both.*what|what.*both|intersection)\b", q):
        family = "parallel"
        reasoning_type = "set_intersection"
        confidence = 0.84
    elif re.search(r"\b(are both|were both|do both|does both|both .* described as|both .* classified as|both .* genus|both .* located)\b", q):
        family = "parallel"
        reasoning_type = "shared_property_yesno"
        confidence = 0.86
    elif re.search(r"\b(same|larger|smaller|older|younger|more|less|higher|lower|which of|are .* and .* of the same)\b", q):
        family = "parallel"
        reasoning_type = "comparison_attribute"
        confidence = 0.78
    elif re.search(r"\b(woman|man|person|actor|actress|individual|one)\s+who\s+(?:portrayed|played|starred)", q) and re.search(r"\b(government position|position|office|served as|held)\b", q):
        family = "constraint"
        reasoning_type = "multi_constraint_lookup"
        confidence = 0.84
    elif q.count(" and ") >= 1 and re.search(r"\b(which|what|who)\b", q):
        family = "constraint"
        reasoning_type = "multi_constraint_lookup"
        confidence = 0.76
    elif re.search(r"\b(that was|who was|which was|formed by|founded by|created by|created|debut album|born in|located in|portrayed|played)\b", q):
        family = "compositional"
        reasoning_type = "bridge_attribute"
        confidence = 0.74
    compatible = tuple(_hotpot_candidate_topologies_for_family(family, primary=reasoning_type, confidence=confidence))
    return {
        "reasoning_family": family,
        "reasoning_type": reasoning_type,
        "confidence": confidence,
        "compatible_topologies": compatible,
        "entities_in_question": entities,
        "requested_relation": _hotpot_question_relation_label(row),
        "answer_mode": _hotpot_answer_mode(row, reasoning_type),
    }


def _hotpot_question_semantics(row: dict[str, Any]) -> HotpotQuestionSemantics:
    classification = _hotpot_topology_classification(row)
    reasoning_type = str(classification["reasoning_type"])
    reasoning_family = str(classification["reasoning_family"])
    confidence = float(classification["confidence"])
    compatible_topologies = tuple(str(item) for item in classification["compatible_topologies"])
    entities_in_question = tuple(str(item) for item in classification["entities_in_question"])
    requested_relation = str(classification["requested_relation"])
    answer_mode = str(classification["answer_mode"])
    entity_a, entity_b = _hotpot_question_entities(row)
    target_relation = requested_relation
    expected_role = _hotpot_expected_answer_role(row)
    common = {
        "reasoning_family": reasoning_family,
        "reasoning_type": reasoning_type,
        "confidence": confidence,
        "compatible_topologies": compatible_topologies,
        "entities_in_question": entities_in_question,
        "requested_relation": requested_relation,
        "answer_mode": answer_mode,
    }
    if reasoning_type == "bridge_entity":
        qs = HotpotQuestionSemantics(
            **common,
            source_entity=entity_a,
            source_relation_hint=_hotpot_bridge_source_relation_hint(row),
            target_relation=target_relation,
            expected_answer_role=expected_role,
        )
    elif reasoning_type == "bridge_attribute":
        qs = HotpotQuestionSemantics(
            **common,
            source_entity=entity_a,
            bridge_relation_hint=_hotpot_bridge_source_relation_hint(row),
            target_relation=target_relation,
            expected_answer_role=expected_role,
        )
    elif reasoning_type in {"comparison_attribute", "shared_property_yesno", "set_intersection"}:
        qs = HotpotQuestionSemantics(
            **common,
            entity_a=entity_a,
            entity_b=entity_b,
            compared_relation=target_relation,
            comparison_operator=_hotpot_comparison_operator(row),
            expected_answer_role=expected_role,
        )
    elif reasoning_type == "multi_constraint_lookup":
        graph = _hotpot_multi_constraint_graph(row)
        constraints = tuple(
            HotpotConstraint(
                anchor=str(item.get("anchor", "")),
                relation=str(item.get("relation", "")),
                direction=str(item.get("direction", "anchor_to_target")),
                target_role=str(item.get("target_role", item.get("target_var", "ANY_RESOLVED"))),
                source_var=str(item.get("source_var", "")),
                target_var=str(item.get("target_var", "X")),
                depends_on=str(item.get("depends_on", "")),
                context_title=str(item.get("context_title", "")),
                resolution_type=str(item.get("resolution_type", "unresolved")),
                confidence=float(item.get("confidence", 0.0) or 0.0),
            )
            for item in graph.get("constraints", [])
        )
        qs = HotpotQuestionSemantics(
            **common,
            constraint_graph_type=str(graph.get("constraint_graph_type", "")),
            answer_var=str(graph.get("answer_var", "")),
            constraints=constraints,
            expected_answer_role=expected_role,
        )
    else:
        qs = HotpotQuestionSemantics(**common)
    return _hotpot_validate_question_semantics(qs)


def _hotpot_validate_question_semantics(qs: HotpotQuestionSemantics) -> HotpotQuestionSemantics:
    errors: list[str] = []
    if qs.reasoning_type == "multi_constraint_lookup":
        if len(qs.constraints) < 2:
            errors.append("missing_two_constraints")
        if qs.constraint_graph_type not in {"serial", "convergent_shared_target"}:
            errors.append("missing_constraint_graph_type")
        if not qs.answer_var:
            errors.append("missing_answer_var")
        for index, constraint in enumerate(qs.constraints):
            if not constraint.anchor and not constraint.source_var:
                errors.append(f"constraint_{index}_missing_anchor")
            if not constraint.relation or constraint.relation == "unresolved":
                errors.append(f"constraint_{index}_missing_relation")
    elif qs.reasoning_type in {"bridge_entity", "bridge_attribute"}:
        if not qs.source_entity:
            errors.append("missing_source_entity")
        relation = qs.target_relation if qs.reasoning_type == "bridge_entity" else qs.requested_relation
        if not relation:
            errors.append("missing_target_relation")
    elif qs.reasoning_type in {"comparison_attribute", "shared_property_yesno", "set_intersection"}:
        if not qs.entity_a or not qs.entity_b:
            errors.append("missing_comparison_entities")
    if not errors:
        return qs
    return HotpotQuestionSemantics(
        **{**qs.__dict__, "valid": False, "validation_errors": tuple(errors)}
    )


def _hotpot_bridge_source_relation_hint(row: dict[str, Any]) -> str:
    q = str(row.get("question", "")).lower()
    if "home games" in q or "home arena" in q or "arena" in q:
        return "home_venue"
    if "director of" in q or "director" in q or "directed" in q:
        return "directed_by"
    if "created" in q or "debut album" in q or "work by" in q:
        return "created_by"
    if "author of" in q or "written" in q or "wrote" in q:
        return "written_by"
    if "portrayed" in q or "played" in q:
        return "acted_in"
    return "unresolved"


def _hotpot_type_specific_verification(
    *,
    row: dict[str, Any],
    reasoning_type: str,
    candidate_bridge: str,
    inferred_bridge: str,
    hop1: dict[str, Any],
    hop2: dict[str, Any],
    hop1_relation: dict[str, Any],
    hop2_relation: dict[str, Any],
    hop1_terms: set[str],
    hop2_terms: set[str],
    bridge_entity_aligned: bool,
) -> dict[str, Any]:
    if reasoning_type in {"bridge_entity", "bridge_attribute"}:
        return _hotpot_verify_bridge_topology(
            row=row,
            reasoning_type=reasoning_type,
            candidate_bridge=candidate_bridge,
            inferred_bridge=inferred_bridge,
            hop1_relation=hop1_relation,
            hop2_relation=hop2_relation,
            hop1_terms=hop1_terms,
            hop2_terms=hop2_terms,
            bridge_entity_aligned=bridge_entity_aligned,
        )
    if reasoning_type in {"comparison_attribute", "shared_property_yesno", "set_intersection"}:
        return _hotpot_verify_parallel_topology(
            row=row,
            reasoning_type=reasoning_type,
            hop1=hop1,
            hop2=hop2,
            hop1_relation=hop1_relation,
            hop2_relation=hop2_relation,
            hop1_terms=hop1_terms,
            hop2_terms=hop2_terms,
        )
    return _hotpot_verify_multi_constraint_topology(
        row=row,
        hop1=hop1,
        hop2=hop2,
        hop1_terms=hop1_terms,
        hop2_terms=hop2_terms,
    )


def _hotpot_verify_bridge_topology(
    *,
    row: dict[str, Any],
    reasoning_type: str,
    candidate_bridge: str,
    inferred_bridge: str,
    hop1_relation: dict[str, Any],
    hop2_relation: dict[str, Any],
    hop1_terms: set[str],
    hop2_terms: set[str],
    bridge_entity_aligned: bool,
) -> dict[str, Any]:
    if reasoning_type == "bridge_attribute":
        return _hotpot_verify_bridge_attribute_topology(
            row=row,
            candidate_bridge=candidate_bridge,
            inferred_bridge=inferred_bridge,
            hop1_relation=hop1_relation,
            hop2_relation=hop2_relation,
            bridge_entity_aligned=bridge_entity_aligned,
        )
    question = str(row.get("question", ""))
    target_terms = _hotpot_relation_target_terms(row)
    requested_relation = _hotpot_question_relation_label(row)
    expected_answer_type = _hotpot_expected_answer_role(row)
    relation_match = _hotpot_relation_match_type(str(hop2_relation.get("relation", "")), requested_relation)
    hop2_relation_matches_request = relation_match in {"exact", "compatible"}
    hop2_target_type_matches = _hotpot_target_role_matches(str(hop2_relation.get("object_role", "")), expected_answer_type)
    hop1_valid = _hotpot_endpoint_matches_question_side(row, hop1_relation["source_entity"]) and _hotpot_relation_reaches_entity(hop1_relation, inferred_bridge)
    hop2_answer_side = bool(target_terms & hop2_terms) or _hotpot_answer_type_match(question, str(hop2_relation.get("target_entity", ""))) > 0.0
    if requested_relation == "unresolved":
        hop2_relation_matches_request = str(hop2_relation.get("relation", "")) != "unresolved"
    hop2_valid = _hotpot_relation_starts_from_entity(hop2_relation, inferred_bridge) and (
        (hop2_relation_matches_request and hop2_target_type_matches)
        or (requested_relation == "unresolved" and hop2_answer_side)
    )
    chain_complete = hop1_valid and hop2_valid and bridge_entity_aligned
    answer_supported = _hotpot_chain_entails_question_relation(
        row=row,
        hop1_relation=hop1_relation,
        hop2_relation=hop2_relation,
        hop1_terms=hop1_terms,
        hop2_terms=hop2_terms,
        derived_answer="",
    ) if chain_complete else False
    if (
        reasoning_type == "bridge_entity"
        and chain_complete
        and hop2_relation.get("target_entity")
        and str(hop2_relation.get("relation", "")) != "unresolved"
    ):
        answer_supported = True
    return {
        "hop1_relation_valid": hop1_valid,
        "hop2_relation_valid": hop2_valid,
        "requested_relation": requested_relation,
        "fact_relation": str(hop2_relation.get("relation", "")),
        "relation_match_type": relation_match,
        "expected_answer_type": expected_answer_type,
        "hop2_relation_matches_request": hop2_relation_matches_request,
        "hop2_target_type_matches": hop2_target_type_matches,
        "candidate_bridge": candidate_bridge,
        "inferred_bridge": inferred_bridge,
        "bridge_hypothesis_match": _hotpot_entities_equivalent(candidate_bridge, inferred_bridge),
        "bridge_alignment": bridge_entity_aligned,
        "chain_complete": chain_complete,
        "evidence_valid": chain_complete,
        "answer_supported": answer_supported,
        "failure_reason": "" if chain_complete and answer_supported else "bridge_chain_not_entailed",
    }


def _hotpot_verify_bridge_attribute_topology(
    *,
    row: dict[str, Any],
    candidate_bridge: str,
    inferred_bridge: str,
    hop1_relation: dict[str, Any],
    hop2_relation: dict[str, Any],
    bridge_entity_aligned: bool,
) -> dict[str, Any]:
    requested_relation = _hotpot_question_relation_label(row)
    expected_answer_type = _hotpot_expected_answer_role(row)
    relation_match = _hotpot_relation_match_type(str(hop2_relation.get("relation", "")), requested_relation)
    source_anchor_valid = _hotpot_endpoint_matches_question_side(row, str(hop1_relation.get("source_entity", "")))
    hop1_relation_valid = source_anchor_valid and _hotpot_relation_reaches_entity(hop1_relation, inferred_bridge)
    hop2_subject_matches_bridge = _hotpot_relation_starts_from_entity(hop2_relation, inferred_bridge)
    hop2_relation_matches_request = relation_match in {"exact", "compatible"}
    hop2_target_type_matches = _hotpot_target_role_matches(str(hop2_relation.get("object_role", "")), expected_answer_type)
    hop2_valid = hop2_subject_matches_bridge and hop2_relation_matches_request and hop2_target_type_matches
    chain_complete = hop1_relation_valid and hop2_valid and bool(inferred_bridge) and bridge_entity_aligned
    failure_code = ""
    if not source_anchor_valid:
        failure_code = "source_anchor_not_satisfied"
    elif not hop1_relation_valid:
        failure_code = "hop1_does_not_reach_bridge"
    elif not hop2_subject_matches_bridge:
        failure_code = "hop2_subject_not_bridge"
    elif not hop2_relation_matches_request:
        failure_code = "requested_relation_not_satisfied"
    elif not hop2_target_type_matches:
        failure_code = "answer_type_not_satisfied"
    elif not bridge_entity_aligned:
        failure_code = "bridge_not_aligned"
    return {
        "source_anchor_valid": source_anchor_valid,
        "hop1_relation_valid": hop1_relation_valid,
        "candidate_bridge": candidate_bridge,
        "inferred_bridge": inferred_bridge,
        "bridge_hypothesis_match": _hotpot_entities_equivalent(candidate_bridge, inferred_bridge),
        "bridge_alignment": bridge_entity_aligned,
        "requested_relation": requested_relation,
        "fact_relation": str(hop2_relation.get("relation", "")),
        "relation_match_type": relation_match,
        "expected_answer_type": expected_answer_type,
        "hop2_subject_matches_bridge": hop2_subject_matches_bridge,
        "hop2_relation_matches_request": hop2_relation_matches_request,
        "hop2_target_type_matches": hop2_target_type_matches,
        "hop2_relation_valid": hop2_valid,
        "chain_complete": chain_complete,
        "evidence_valid": chain_complete,
        "answer_supported": chain_complete,
        "failure_code": failure_code,
        "failure_reason": failure_code,
    }


def _hotpot_verify_parallel_topology(
    *,
    row: dict[str, Any],
    reasoning_type: str,
    hop1: dict[str, Any],
    hop2: dict[str, Any],
    hop1_relation: dict[str, Any],
    hop2_relation: dict[str, Any],
    hop1_terms: set[str],
    hop2_terms: set[str],
) -> dict[str, Any]:
    entity_a, entity_b = _hotpot_question_entities(row)
    entity_a_supported = _hotpot_fact_supports_entity(hop1, entity_a) or _hotpot_fact_supports_entity(hop2, entity_a)
    entity_b_supported = _hotpot_fact_supports_entity(hop1, entity_b) or _hotpot_fact_supports_entity(hop2, entity_b)
    attribute_a = _hotpot_parallel_attribute_value(row, hop1, entity_a)
    attribute_b = _hotpot_parallel_attribute_value(row, hop2, entity_b)
    if not attribute_a or not attribute_b:
        attribute_a = attribute_a or _hotpot_parallel_attribute_value(row, hop2, entity_a)
        attribute_b = attribute_b or _hotpot_parallel_attribute_value(row, hop1, entity_b)
    relation_terms = _hotpot_relation_target_terms(row)
    relation_supported = bool(relation_terms & (hop1_terms | hop2_terms)) or bool(attribute_a and attribute_b)
    comparison_valid = entity_a_supported and entity_b_supported and relation_supported
    if reasoning_type == "shared_property_yesno":
        answer_supported = comparison_valid and _hotpot_shared_property_holds(row, hop1_terms, hop2_terms, attribute_a, attribute_b)
    elif reasoning_type == "set_intersection":
        answer_supported = comparison_valid and bool(_hotpot_terms(attribute_a) & _hotpot_terms(attribute_b))
    else:
        answer_supported = comparison_valid and _hotpot_comparison_supported(row, attribute_a, attribute_b, hop1_terms, hop2_terms)
    return {
        "entity_a": entity_a,
        "entity_b": entity_b,
        "entity_a_supported": entity_a_supported,
        "entity_b_supported": entity_b_supported,
        "attribute_a": attribute_a,
        "attribute_b": attribute_b,
        "comparison_operator": _hotpot_comparison_operator(row),
        "comparison_valid": comparison_valid,
        "chain_complete": comparison_valid,
        "evidence_valid": comparison_valid,
        "answer_supported": answer_supported,
        "failure_reason": "" if comparison_valid and answer_supported else "parallel_comparison_not_supported",
    }


def _hotpot_verify_multi_constraint_topology(
    *,
    row: dict[str, Any],
    hop1: dict[str, Any],
    hop2: dict[str, Any],
    hop1_terms: set[str],
    hop2_terms: set[str],
) -> dict[str, Any]:
    del hop1_terms, hop2_terms
    qsem = _hotpot_question_semantics(row)
    constraints = [constraint.as_dict() for constraint in qsem.constraints]
    facts = [hop1, hop2]
    bindings = []
    used_fact_ids: set[str] = set()
    variables: dict[str, str] = {}
    for index, constraint in enumerate(constraints):
        effective_constraint = dict(constraint)
        anchor = str(effective_constraint.get("anchor", ""))
        source_var = str(effective_constraint.get("source_var") or effective_constraint.get("depends_on") or "")
        if source_var and source_var in variables:
            effective_constraint["anchor"] = variables[source_var]
        elif anchor in variables:
            effective_constraint["anchor"] = variables[anchor]
        best_binding = _hotpot_best_constraint_binding(row, effective_constraint, facts, used_fact_ids=used_fact_ids)
        if best_binding.get("fact_id"):
            used_fact_ids.add(str(best_binding["fact_id"]))
        if best_binding.get("satisfied"):
            target_var = str(constraint.get("target_var") or "")
            if target_var:
                variables[target_var] = str(best_binding.get("target", ""))
        bindings.append({
            "constraint_index": index,
            "constraint": constraint,
            "effective_constraint": effective_constraint,
            **best_binding,
        })
    satisfied_bindings = [item for item in bindings if bool(item.get("satisfied"))]
    target_values = [str(item.get("target", "")) for item in satisfied_bindings if str(item.get("target", "")).strip()]
    if qsem.constraint_graph_type == "serial":
        same_target = bool(qsem.answer_var and variables.get(qsem.answer_var))
    else:
        same_target = bool(target_values) and all(_hotpot_entities_equivalent(target_values[0], value) for value in target_values[1:])
    all_constraints_satisfied = len(satisfied_bindings) == len(constraints) and bool(constraints)
    evidence_valid = all_constraints_satisfied and same_target
    answer_value = str(variables.get(qsem.answer_var, "") if qsem.answer_var else (target_values[0] if target_values else ""))
    shared_entities = sorted({normalize_bridge(answer_value if qsem.constraint_graph_type == "serial" else value) for value in target_values if normalize_bridge(value)})
    failure_code = ""
    if not constraints:
        failure_code = "constraints_not_parsed"
    elif not all_constraints_satisfied:
        failure_code = "constraint_not_satisfied"
    elif not same_target:
        failure_code = "constraint_targets_disagree"
    return {
        "constraints": constraints,
        "constraint_graph_type": qsem.constraint_graph_type,
        "answer_var": qsem.answer_var,
        "variable_bindings": dict(variables),
        "constraint_bindings": bindings,
        "constraint_1_supported": bool(bindings[0].get("satisfied")) if len(bindings) > 0 else False,
        "constraint_2_supported": bool(bindings[1].get("satisfied")) if len(bindings) > 1 else False,
        "constraint_targets": [answer_value] if qsem.constraint_graph_type == "serial" and answer_value else target_values,
        "same_target": same_target,
        "shared_entity_aliases": shared_entities,
        "chain_complete": evidence_valid,
        "evidence_valid": evidence_valid,
        "answer_supported": evidence_valid,
        "failure_code": failure_code,
        "failure_reason": failure_code,
    }


def _hotpot_chain_entails_question_relation(
    *,
    row: dict[str, Any],
    hop1_relation: dict[str, Any],
    hop2_relation: dict[str, Any],
    hop1_terms: set[str],
    hop2_terms: set[str],
    derived_answer: str,
) -> bool:
    question = str(row.get("question", ""))
    target_terms = _hotpot_relation_target_terms(row)
    relation_terms = {
        str(hop1_relation.get("relation", "")),
        str(hop2_relation.get("relation", "")),
    }
    answer_entity = str(hop2_relation.get("target_entity", ""))
    if not answer_entity and not derived_answer:
        return False
    if "how many" in question.lower():
        return bool(re.search(r"\b\d[\d,]*\b", answer_entity or derived_answer))
    if "when" in question.lower() or "what year" in question.lower():
        return bool(re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", answer_entity or derived_answer))
    if question.lower().startswith(("are ", "is ", "was ", "were ", "did ", "do ", "does ")):
        return bool((hop1_terms & hop2_terms) or derived_answer)
    if target_terms & (hop1_terms | hop2_terms):
        return True
    return any(term and term != "unresolved" and term in target_terms for term in relation_terms)


def _hotpot_bridge_match_type(hop1_match: dict[str, Any], hop2_match: dict[str, Any], aligned: bool) -> str:
    types = [str(hop1_match.get("match_type") or "unresolved"), str(hop2_match.get("match_type") or "unresolved")]
    if not aligned:
        return "unresolved"
    if "exact" in types:
        return "exact"
    if "title_entity" in types:
        return "title_entity"
    if "alias" in types:
        return "alias"
    return "relation_inferred" if aligned else "unresolved"


def _hotpot_infer_bridge_from_relations(hop1_relation: dict[str, Any], hop2_relation: dict[str, Any]) -> str:
    hop1_target = str(hop1_relation.get("target_entity", "")).strip()
    hop2_source = str(hop2_relation.get("source_entity", "")).strip()
    if hop1_target and hop2_source and _hotpot_entities_equivalent(hop1_target, hop2_source):
        return hop2_source if len(hop2_source) >= len(hop1_target) else hop1_target
    hop1_entities = [
        str(hop1_relation.get("target_entity", "")),
        *[str(item) for item in hop1_relation.get("referenced_entities", [])],
        str(hop1_relation.get("title_entity", "")),
    ]
    hop2_anchor_entities = [
        str(hop2_relation.get("source_entity", "")),
        str(hop2_relation.get("title_entity", "")),
    ]
    for left in hop1_entities:
        for right in hop2_anchor_entities:
            if left and right and _hotpot_entities_equivalent(left, right):
                return right if len(right) >= len(left) else left
    hop2_entities = [
        *hop2_anchor_entities,
        *[str(item) for item in hop2_relation.get("referenced_entities", [])],
    ]
    for left in hop1_entities:
        for right in hop2_entities:
            if left and right and _hotpot_entities_equivalent(left, right):
                return right if len(right) >= len(left) else left
    return ""


def _hotpot_relation_reaches_entity(relation: dict[str, Any], entity: str) -> bool:
    if not entity:
        return False
    values = [
        str(relation.get("target_entity", "")),
        *[str(item) for item in relation.get("referenced_entities", [])],
    ]
    return any(_hotpot_entities_equivalent(value, entity) for value in values)


def _hotpot_relation_starts_from_entity(relation: dict[str, Any], entity: str) -> bool:
    if not entity:
        return False
    values = [str(relation.get("source_entity", "")), str(relation.get("title_entity", ""))]
    return any(_hotpot_entities_equivalent(value, entity) for value in values)


def _hotpot_bridge_hypothesis_match_type(
    candidate_bridge: str,
    inferred_bridge: str,
    hop1_match: dict[str, Any],
    hop2_match: dict[str, Any],
) -> str:
    if not inferred_bridge:
        return "unresolved"
    candidate_key = normalize_bridge(candidate_bridge)
    inferred_key = normalize_bridge(inferred_bridge)
    if candidate_key and candidate_key == inferred_key:
        return "exact"
    if _hotpot_entities_equivalent(candidate_bridge, inferred_bridge):
        return "alias"
    if any(match.get("match_type") == "title_entity" for match in (hop1_match, hop2_match)):
        return "title_entity"
    return "relation_inferred"


def _hotpot_question_entities(row: dict[str, Any]) -> tuple[str, str]:
    entities = [str(item).strip() for item in row.get("entities", []) if str(item).strip()]
    if len(entities) >= 2:
        return entities[0], entities[1]
    if len(entities) == 1:
        return entities[0], ""
    extracted = _hotpot_entities_in_text(str(row.get("question", "")))
    if len(extracted) >= 2:
        return extracted[0], extracted[1]
    if len(extracted) == 1:
        return extracted[0], ""
    return "", ""


def _hotpot_fact_supports_entity(fact: dict[str, Any], entity: str) -> bool:
    if not entity:
        return False
    profile = _hotpot_fact_entity_profile(fact)
    return bool(_hotpot_entity_aliases(entity) & profile["aliases"])


def _hotpot_parallel_attribute_value(row: dict[str, Any], fact: dict[str, Any], entity: str) -> str:
    text = str(fact.get("text", ""))
    q = str(row.get("question", "")).lower()
    if "genus" in q and "genus" in text.lower():
        return "genus"
    if "nationality" in q:
        match = re.search(r"\b(American|British|Canadian|French|German|Italian|Dutch|Greek|South Korean|Korean|Turkish|Ottoman)\b", text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    if "neighborhood" in q or "located" in q or "where" in q:
        match = re.search(r"\bin\s+([A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){0,3})", text)
        if match:
            return match.group(1).strip()
    if "formed by" in q or "founded by" in q:
        match = re.search(r"\b(?:formed|founded|created)\s+(?:in\s+\d{4}\s+)?by\s+([A-Z][A-Za-z0-9&'’.-]+(?:\s+[A-Z][A-Za-z0-9&'’.-]+){0,4})", text)
        if match:
            return match.group(1).strip()
    entities = [item for item in _hotpot_entities_in_text(text) if not _hotpot_entities_equivalent(item, entity)]
    return entities[-1] if entities else ""


def _hotpot_comparison_operator(row: dict[str, Any]) -> str:
    q = str(row.get("question", "")).lower()
    if "same" in q:
        return "same"
    if "both" in q:
        return "both"
    if "larger" in q or "more" in q or "higher" in q:
        return "greater_than"
    if "smaller" in q or "less" in q or "lower" in q:
        return "less_than"
    if "older" in q:
        return "older"
    if "younger" in q:
        return "younger"
    if "common" in q or "share" in q or "shared" in q:
        return "intersection"
    return "compare"


def _hotpot_shared_property_holds(row: dict[str, Any], hop1_terms: set[str], hop2_terms: set[str], attribute_a: str, attribute_b: str) -> bool:
    q_terms = _hotpot_relation_target_terms(row)
    if q_terms & hop1_terms and q_terms & hop2_terms:
        return True
    return bool(attribute_a and attribute_b and _hotpot_terms(attribute_a) & _hotpot_terms(attribute_b))


def _hotpot_comparison_supported(row: dict[str, Any], attribute_a: str, attribute_b: str, hop1_terms: set[str], hop2_terms: set[str]) -> bool:
    op = _hotpot_comparison_operator(row)
    if op == "same":
        if attribute_a and attribute_b:
            return bool(_hotpot_terms(attribute_a) & _hotpot_terms(attribute_b)) or normalize_bridge(attribute_a) == normalize_bridge(attribute_b)
        return bool(_hotpot_relation_target_terms(row) & hop1_terms and _hotpot_relation_target_terms(row) & hop2_terms)
    if op == "intersection":
        return bool(attribute_a and attribute_b and _hotpot_terms(attribute_a) & _hotpot_terms(attribute_b))
    return bool(attribute_a or attribute_b or (_hotpot_relation_target_terms(row) & (hop1_terms | hop2_terms)))


def _hotpot_question_constraints(row: dict[str, Any]) -> list[dict[str, str]]:
    return _hotpot_parse_multi_constraints(row)


def _hotpot_multi_constraint_graph(row: dict[str, Any]) -> dict[str, Any]:
    constraints = _hotpot_parse_multi_constraints(row)
    graph_type = str(constraints[0].get("constraint_graph_type", "convergent_shared_target")) if constraints else ""
    answer_var = str(constraints[0].get("answer_var", "X")) if constraints else ""
    clean_constraints = []
    for constraint in constraints:
        item = dict(constraint)
        item.pop("constraint_graph_type", None)
        item.pop("answer_var", None)
        clean_constraints.append(item)
    return {
        "constraint_graph_type": graph_type,
        "answer_var": answer_var,
        "constraints": clean_constraints,
    }


def _hotpot_parse_multi_constraints(row: dict[str, Any]) -> list[dict[str, str]]:
    question = str(row.get("question", ""))
    entities = [str(item).strip() for item in row.get("entities", []) if str(item).strip()]
    serial = _hotpot_parse_serial_multi_constraints(row, question, entities)
    if serial:
        return serial
    if len(entities) < 2:
        return []
    constraints: list[dict[str, str]] = []
    for entity in entities[:2]:
        relation = _hotpot_relation_for_entity_in_question(question, entity)
        if relation == "unresolved":
            relation = _hotpot_relation_by_order(question, len(constraints))
        if relation == "unresolved":
            continue
        resolved = _hotpot_resolve_anchor(row, entity)
        constraints.append(HotpotConstraint(
            anchor=entity,
            relation=relation,
            direction="anchor_to_target",
            target_role=_hotpot_expected_answer_role(row),
            target_var="X",
            context_title=resolved.context_title,
            resolution_type=resolved.resolution_type,
            confidence=resolved.confidence,
        ).as_dict() | {
            "constraint_graph_type": "convergent_shared_target",
            "answer_var": "X",
        })
    return constraints[:2] if len(constraints) >= 2 else []


def _hotpot_parse_serial_multi_constraints(row: dict[str, Any], question: str, entities: list[str]) -> list[dict[str, Any]]:
    q = str(question or "")
    q_lower = q.lower()
    if not re.search(r"\b(woman|man|person|actor|actress|individual|one)\s+who\s+(?:portrayed|played|starred)", q_lower):
        return []
    if not re.search(r"\b(government position|position|office|served as|held)\b", q_lower):
        return []
    anchor = ""
    portrayed_match = re.search(
        r"\b(?:portrayed|played|starred(?:\s+as)?)\s+([A-Z][A-Za-z0-9&'’.-]+(?:\s+[A-Z][A-Za-z0-9&'’.-]+){0,4})",
        q,
    )
    if portrayed_match:
        anchor = _hotpot_clean_extracted_value(portrayed_match.group(1))
    if not anchor and entities:
        anchor = entities[0]
    if not anchor:
        return []
    resolved = _hotpot_resolve_anchor(row, anchor)
    first = HotpotConstraint(
        anchor=anchor,
        relation="acted_in",
        direction="anchor_to_target",
        target_role="PERSON",
        target_var="X",
        context_title=resolved.context_title,
        resolution_type=resolved.resolution_type,
        confidence=max(0.75, resolved.confidence),
    ).as_dict()
    second = HotpotConstraint(
        anchor="X",
        relation="position",
        direction="anchor_to_target",
        target_role=_hotpot_expected_answer_role(row),
        source_var="X",
        target_var="Y",
        depends_on="X",
        confidence=0.75,
    ).as_dict()
    return [
        first | {"constraint_graph_type": "serial", "answer_var": "Y"},
        second | {"constraint_graph_type": "serial", "answer_var": "Y"},
    ]


def _hotpot_context_titles(row: dict[str, Any]) -> list[str]:
    titles: list[str] = []
    for fact in row.get("supporting_facts", row.get("evidence", [])):
        if isinstance(fact, dict):
            title = str(fact.get("title", fact.get("entity", ""))).strip()
            if title and all(normalize_bridge(existing) != normalize_bridge(title) for existing in titles):
                titles.append(title)
    return titles


def _hotpot_resolve_anchor(row: dict[str, Any], surface: str) -> ResolvedAnchor:
    surface_text = str(surface or "").strip()
    surface_key = normalize_bridge(surface_text)
    titles = _hotpot_context_titles(row)
    for title in titles:
        if normalize_bridge(title) == surface_key:
            return ResolvedAnchor(
                surface=surface_text,
                canonical=title,
                context_title=title,
                matched_entity_id=title,
                aliases=(surface_text,),
                resolution_type="exact_title",
                confidence=1.0,
            )
    for title in titles:
        title_key = normalize_bridge(title)
        if title_key and (title_key in surface_key or surface_key in title_key):
            return ResolvedAnchor(
                surface=surface_text,
                canonical=title,
                context_title=title,
                matched_entity_id=title,
                aliases=(surface_text,),
                resolution_type="normalized_title",
                confidence=0.9,
            )
    for index, entity in enumerate(row.get("entities", []), start=1):
        entity_text = str(entity).strip()
        if normalize_bridge(entity_text) == surface_key:
            return ResolvedAnchor(
                surface=surface_text,
                canonical=entity_text,
                matched_entity_id=f"entity_{index}",
                aliases=(surface_text,),
                resolution_type="entity_exact",
                confidence=0.7,
            )
    return ResolvedAnchor(
        surface=surface_text,
        canonical=surface_text,
        aliases=(surface_text,),
        resolution_type="unresolved",
        confidence=0.0,
    )


def _hotpot_relation_by_order(question: str, index: int) -> str:
    relation_sequence = []
    for pattern, relation in (
        (r"\bportrayed\b|\bplayed\b|\bstarred\b|\bstarring\b", "acted_in"),
        (r"\bwrote\b|\bwritten\b|\bauthored\b|\bauthor(?:ed)?\b", "written_by"),
        (r"\bcreated\b|\bcreator\b", "created_by"),
        (r"\bdirected\b|\bdirector\b", "directed_by"),
        (r"\bfounded\b|\bformed\b|\bestablished\b", "formed_by"),
        (r"\blocated\b|\bsituated\b|\bbased\b", "located_in"),
    ):
        if re.search(pattern, question, flags=re.IGNORECASE):
            relation_sequence.append(relation)
    return relation_sequence[index] if index < len(relation_sequence) else "unresolved"


def _hotpot_relation_for_entity_in_question(question: str, entity: str) -> str:
    q = str(question or "")
    entity_text = str(entity or "").strip()
    entity_key = re.escape(entity_text)
    if not entity_key:
        return "unresolved"
    match = re.search(rf"\b{entity_key}\b", q, flags=re.IGNORECASE)
    if not match:
        return "unresolved"
    relation_pattern = r"\b(wrote|written|authored|author|created|directed|director|founded|formed|established|located|situated|based|portrayed|played|starred|starring)\b"
    before_window = q[max(0, match.start() - 100):match.start()]
    before_matches = list(re.finditer(relation_pattern, before_window, flags=re.IGNORECASE))
    if before_matches:
        return _hotpot_canonical_relation(before_matches[-1].group(1))
    after_window = q[match.end():match.end() + 100]
    after_match = re.search(relation_pattern, after_window, flags=re.IGNORECASE)
    if after_match:
        return _hotpot_canonical_relation(after_match.group(1))
    return "unresolved"


def _hotpot_best_constraint_binding(
    row: dict[str, Any],
    constraint: dict[str, str],
    facts: list[dict[str, Any]],
    *,
    used_fact_ids: set[str],
) -> dict[str, Any]:
    candidates = []
    for fact in facts:
        relation = _hotpot_fact_relation_view(row, "", fact, role="constraint")
        fact_id = str(fact.get("fact_id", ""))
        observed_relation = str(relation.get("relation", ""))
        requested_relation = str(constraint.get("relation", ""))
        source_entity = str(relation.get("source_entity", ""))
        target_entity = str(relation.get("target_entity", ""))
        anchor = str(constraint.get("anchor", ""))
        anchor_match = _hotpot_entities_equivalent(source_entity, anchor)
        relation_match_type = _hotpot_relation_match_type(str(relation.get("relation", "")), str(constraint.get("relation", "")))
        relation_match = relation_match_type in {"exact", "compatible"}
        target = target_entity
        if (
            not anchor_match
            and _hotpot_canonical_relation(observed_relation) in {"acted_in"}
            and _hotpot_canonical_relation(requested_relation) in {"acted_in"}
            and _hotpot_entities_equivalent(target_entity, anchor)
            and source_entity
            and source_entity != "unresolved"
        ):
            anchor_match = True
            target = source_entity
        target_valid = bool(target and target != "unresolved")
        satisfied = anchor_match and relation_match and target_valid
        score = (
            3.0 * float(satisfied)
            + 1.0 * float(anchor_match)
            + 1.0 * float(relation_match)
            + 0.25 * float(target_valid)
            - 0.1 * float(fact_id in used_fact_ids)
        )
        candidates.append({
            "fact_id": fact_id,
            "anchor_match": anchor_match,
            "relation_match": relation_match,
            "relation_match_type": relation_match_type,
            "observed_anchor": source_entity,
            "observed_relation": observed_relation,
            "target": target,
            "target_valid": target_valid,
            "satisfied": satisfied,
            "score": score,
        })
    candidates.sort(key=lambda item: (-float(item["score"]), str(item["fact_id"])))
    return candidates[0] if candidates else {
        "fact_id": "",
        "anchor_match": False,
        "relation_match": False,
        "relation_match_type": "mismatch",
        "observed_anchor": "",
        "observed_relation": "unresolved",
        "target": "",
        "target_valid": False,
        "satisfied": False,
        "score": 0.0,
    }


def _hotpot_relation_target_terms(row: dict[str, Any]) -> set[str]:
    terms = set(_hotpot_target_terms(str(row.get("question", ""))))
    for entity in row.get("entities", []):
        terms -= _hotpot_terms(str(entity))
    return terms


def _hotpot_fact_bridge_match(bridge: str, fact: dict[str, Any]) -> dict[str, Any]:
    bridge_aliases = _hotpot_entity_aliases(bridge)
    bridge_key = normalize_bridge(bridge)
    profile = _hotpot_fact_entity_profile(fact)
    title_key = normalize_bridge(profile["title_entity"])
    text_key = normalize_bridge(str(fact.get("text", "")))
    matched_aliases = sorted(bridge_aliases & profile["aliases"])
    match_type = "unresolved"
    if bridge_key and bridge_key == title_key:
        match_type = "title_entity"
    elif bridge_key and bridge_key in text_key:
        match_type = "exact"
    elif matched_aliases:
        match_type = "alias"
    return {
        "matched": bool(matched_aliases),
        "matched_aliases": matched_aliases,
        "match_type": match_type,
        "title_entity": profile["title_entity"],
        "referenced_entities": profile["referenced_entities"],
    }


def _hotpot_fact_relation_view(row: dict[str, Any], bridge: str, fact: dict[str, Any], *, role: str) -> dict[str, Any]:
    semantics = _hotpot_fact_semantics(row, fact)
    return {
        "source_entity": semantics["subject"],
        "subject": semantics["subject"],
        "subject_role": semantics["subject_role"],
        "relation": semantics["relation"],
        "predicate": semantics.get("predicate", semantics["relation"]),
        "direction": semantics.get("direction", "anchor_to_target"),
        "target_entity": semantics["object"],
        "object": semantics["object"],
        "object_role": semantics["object_role"],
        "title_entity": semantics["title_entity"],
        "document_title": semantics["title_entity"],
        "referenced_entities": semantics["referenced_entities"],
        "confidence": semantics["confidence"],
    }


def _hotpot_fact_semantics(row: dict[str, Any], fact: dict[str, Any]) -> dict[str, Any]:
    profile = _hotpot_fact_entity_profile(fact)
    title = profile["title_entity"]
    text = str(fact.get("text", ""))
    structured = _hotpot_extract_relation_semantics(row, fact)
    relation = structured.predicate
    subject = structured.subject
    obj = structured.object
    object_role = structured.object_role
    confidence = 0.0
    if subject != "unresolved":
        confidence += 0.4
    if relation != "unresolved":
        confidence += 0.3
    if obj != "unresolved":
        confidence += 0.3
    aliases = set(profile["aliases"])
    if subject != "unresolved":
        aliases |= _hotpot_entity_aliases(subject)
    if obj != "unresolved":
        aliases |= _hotpot_entity_aliases(obj)
    return {
        "title_entity": title,
        "subject": subject,
        "subject_role": _hotpot_ontology_role(subject),
        "relation": relation,
        "predicate": relation,
        "direction": structured.direction,
        "object": obj,
        "object_role": object_role if object_role != "entity" else _hotpot_ontology_role(obj),
        "aliases": aliases,
        "referenced_entities": profile["referenced_entities"],
        "confidence": round(confidence, 4),
    }


def _hotpot_extract_relation_semantics(row: dict[str, Any], fact: dict[str, Any]) -> RelationSemantics:
    title = str(fact.get("title", "")).strip()
    text = str(fact.get("text", ""))
    relation = _hotpot_relation_label(str(row.get("question", "")), text)
    subject = _hotpot_extract_subject(title, text)

    active = re.search(r"^([^.;,]+?)\s+(wrote|authored|created|directed)\s+([^.;,]+)", text, flags=re.IGNORECASE)
    if active:
        left = _hotpot_clean_extracted_value(active.group(1))
        verb = active.group(2).lower()
        right = _hotpot_clean_extracted_value(active.group(3))
        predicate = _hotpot_canonical_relation(verb)
        if title and _hotpot_entities_equivalent(title, right):
            return RelationSemantics(
                predicate=predicate,
                subject=title,
                object=left,
                direction="anchor_to_target",
                object_role=_hotpot_ontology_role(left),
                confidence=0.9,
            )
        if title and _hotpot_entities_equivalent(title, left):
            return RelationSemantics(
                predicate=predicate,
                subject=title,
                object=right,
                direction="anchor_to_target",
                object_role=_hotpot_ontology_role(right),
                confidence=0.9,
            )

    patterns = [
        ("written_by", rf"\b{re.escape(title)}\b.*?\b(?:was\s+)?(?:written|authored)\s+by\s+([^.;,]+)", title, 1),
        ("directed_by", rf"\b{re.escape(title)}\b.*?\b(?:was\s+)?directed\s+by\s+([^.;,]+)", title, 1),
        ("formed_by", rf"\b{re.escape(title)}\b.*?\b(?:was\s+)?(?:formed|founded|established)\s+(?:in\s+\d{{4}}\s+)?by\s+([^.;,]+)", title, 1),
        ("created_by", rf"\b{re.escape(title)}\b.*?\b(?:was\s+)?created\s+by\s+([^.;,]+)", title, 1),
        ("home_venue", rf"\b{re.escape(title)}\b.*?\b(?:played|plays|play)\s+(?:their\s+)?home\s+games\s+at\s+([^.;,]+)", title, 1),
        ("home_venue", rf"\b{re.escape(title)}\b.*?\bhome\s+(?:arena|venue|stadium)\s+(?:is|was)\s+([^.;,]+)", title, 1),
        ("located_in", rf"\b{re.escape(title)}\b.*?\bbased\s+in\s+([^.;,]+)", title, 1),
        ("born_in", rf"\b{re.escape(title)}\b.*?\bborn\s+in\s+([^.;,]+)", title, 1),
        ("capacity", rf"\b{re.escape(title)}\b.*?\b(?:can\s+seat|seats|capacity(?:\s+of)?)\s+(\d[\d,]*)", title, 1),
    ]
    for predicate, pattern, fixed_subject, object_group in patterns:
        if not title and fixed_subject:
            continue
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        if object_group == "title_object":
            actor = _hotpot_clean_extracted_value(match.group(1))
            if actor and title:
                return RelationSemantics(
                    predicate=predicate,
                    subject=title,
                    object=actor,
                    direction="anchor_to_target",
                    object_role=_hotpot_ontology_role(actor),
                    confidence=0.9,
                )
        else:
            obj = _hotpot_clean_extracted_value(match.group(int(object_group)))
            if obj:
                return RelationSemantics(
                    predicate=predicate,
                    subject=str(fixed_subject or subject),
                    object=obj,
                    direction="anchor_to_target",
                    object_role=_hotpot_ontology_role(obj),
                    confidence=0.9,
                )

    subject = _hotpot_extract_subject(title, text)
    obj, object_role = _hotpot_extract_object_value(row, fact, subject=subject, relation=relation)
    confidence = 0.0
    if subject != "unresolved":
        confidence += 0.4
    if relation != "unresolved":
        confidence += 0.3
    if obj != "unresolved":
        confidence += 0.3
    return RelationSemantics(
        predicate=relation,
        subject=subject,
        object=obj,
        direction="anchor_to_target",
        object_role=object_role if object_role != "entity" else _hotpot_ontology_role(obj),
        confidence=confidence,
    )


def _hotpot_extract_subject(title: str, text: str) -> str:
    if title and normalize_bridge(title) in normalize_bridge(text):
        return title
    entities = _hotpot_entities_in_text(text)
    if title and any(_hotpot_entities_equivalent(title, entity) for entity in entities):
        return title
    return entities[0] if entities else "unresolved"


def _hotpot_extract_object_value(row: dict[str, Any], fact: dict[str, Any], *, subject: str, relation: str) -> tuple[str, str]:
    text = str(fact.get("text", ""))
    q = str(row.get("question", "")).lower()
    if "genus" in q and "genus" in text.lower():
        return "genus", "attribute"
    if "nationality" in q:
        match = re.search(r"\b(American|British|Canadian|French|German|Italian|Dutch|Greek|South Korean|Korean|Turkish|Ottoman)\b", text, flags=re.IGNORECASE)
        if match:
            return match.group(1), "attribute"
    if relation == "unresolved":
        return "unresolved", "unresolved"
    if relation in {"located_in", "location", "born_in"}:
        match = re.search(r"\b(?:located|born|based)\s+(?:at|in|on)\s+([A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){0,3})", text)
        if match:
            return _hotpot_clean_extracted_value(match.group(1)), "entity"
        match = re.search(r"\bin\s+([A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){0,3})", text)
        if match:
            return _hotpot_clean_extracted_value(match.group(1)), "entity"
    if relation in {"formed_by", "founded_by", "created_by", "written_by"} or "formed by" in q or "founded by" in q:
        match = re.search(r"\b(?:formed|founded|created|written|authored)\s+(?:in\s+\d{4}\s+)?by\s+([A-Z][A-Za-z0-9&'’.-]+(?:\s+[A-Z][A-Za-z0-9&'’.-]+){0,4})", text)
        if match:
            return _hotpot_clean_extracted_value(match.group(1)), "entity"
        match = re.search(r"\bwrote\s+([A-Z][A-Za-z0-9&'’.-]+(?:\s+[A-Z][A-Za-z0-9&'’.-]+){0,4})", text)
        if match:
            return _hotpot_clean_extracted_value(match.group(1)), "entity"
    if relation == "directed_by":
        match = re.search(r"\bdirected\s+(?:by\s+)?([A-Z][A-Za-z0-9&'’.-]+(?:\s+[A-Z][A-Za-z0-9&'’.-]+){0,4})", text)
        if match:
            return _hotpot_clean_extracted_value(match.group(1)), "entity"
    if relation == "acted_in":
        match = re.search(r"\b(?:played|portrayed|starring|starred)\s+(?:by\s+)?([A-Z][A-Za-z0-9&'’.-]+(?:\s+[A-Z][A-Za-z0-9&'’.-]+){0,4})", text)
        if match:
            return _hotpot_clean_extracted_value(match.group(1)), "entity"
    if relation == "home_venue":
        match = re.search(r"\bhome\s+games\s+at\s+([^.;,]+)", text, flags=re.IGNORECASE)
        if match:
            return _hotpot_clean_extracted_value(match.group(1)), "entity"
    if relation == "capacity":
        match = re.search(r"\b(?:can\s+seat|seats|capacity(?:\s+of)?)\s+(\d[\d,]*)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1), "LITERAL"
    if relation == "position":
        match = re.search(r"\b(?:served\s+as|held\s+the\s+position\s+of|position\s+of)\s+([^.;,]+)", text, flags=re.IGNORECASE)
        if match:
            return _hotpot_clean_extracted_value(match.group(1)), "attribute"
    entities = [item for item in _hotpot_entities_in_text(text) if not _hotpot_entities_equivalent(item, subject)]
    if entities:
        return entities[-1], "entity"
    return "unresolved", "unresolved"


def _hotpot_clean_extracted_value(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip(" .,;:()[]{}\"'`")).strip()
    text = re.sub(r"^(?:by|of)\s+", "", text, flags=re.IGNORECASE)
    text = re.split(r"\s+(?:in|on|at|from|during)\s+(?:\d{3,4}|the\b|a\b|an\b)", text, maxsplit=1)[0]
    return text.strip(" .,;:()[]{}\"'`")


def _hotpot_question_side_entity(row: dict[str, Any], fact: dict[str, Any]) -> str:
    title = str(fact.get("title", "")).strip()
    entities = [str(item) for item in row.get("entities", []) if str(item).strip()]
    for entity in entities:
        if _hotpot_entities_equivalent(entity, title):
            return title
    text = f"{fact.get('title', '')} {fact.get('text', '')}"
    for entity in entities:
        if normalize_bridge(entity) and normalize_bridge(entity) in normalize_bridge(text):
            return entity
    return title


def _hotpot_non_equivalent_entity(source: str, entities: list[str]) -> str:
    for entity in entities:
        if str(entity).strip() and not _hotpot_entities_equivalent(source, entity):
            return entity
    return ""


def _hotpot_answer_side_entity(row: dict[str, Any], fact: dict[str, Any]) -> str:
    question = str(row.get("question", ""))
    text = str(fact.get("text", ""))
    if "how many" in question.lower():
        match = re.search(r"\b\d[\d,]*\b", text)
        return match.group(0) if match else ""
    if "when" in question.lower() or "what year" in question.lower():
        match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", text)
        return match.group(0) if match else ""
    entities = _hotpot_entities_in_text(text)
    return entities[-1] if entities else str(fact.get("title", "")).strip()


def _hotpot_relation_label(question: str, text: str) -> str:
    value = str(text).lower()
    if "home games" in value or "home venue" in value or re.search(r"\bplayed\b.*\bat\b", value):
        return "home_venue"
    if "capacity" in value or re.search(r"\b(can seat|seats|seated)\b", value):
        return "capacity"
    if "served as" in value or "position" in value or "chief of protocol" in value:
        return "position"
    if "played by" in value or "portrayed by" in value or "portrayed" in value:
        return "acted_in"
    if "formed" in value:
        return "formed_by"
    if "founded" in value:
        return "formed_by"
    if "created" in value:
        return "created_by"
    if "written" in value or "wrote" in value or "authored" in value:
        return "written_by"
    if "directed" in value or "director" in value:
        return "directed_by"
    if "born" in value:
        return "born_in"
    if "nationality" in value or "american" in value or "british" in value:
        return "nationality"
    if "located" in value or "city" in value or "country" in value:
        return "located_in"
    if "starring" in value or "starred" in value or "actor" in value:
        return "acted_in"
    return "unresolved"


def _hotpot_endpoint_matches_question_side(row: dict[str, Any], entity: str) -> bool:
    if not str(entity).strip():
        return False
    row_entities = [str(item) for item in row.get("entities", []) if str(item).strip()]
    return any(_hotpot_entities_equivalent(entity, candidate) for candidate in row_entities) or bool(
        _hotpot_terms(entity) & _hotpot_terms(str(row.get("question", "")))
    )


def _hotpot_endpoint_matches_bridge(bridge: str, entity: str) -> bool:
    return _hotpot_entities_equivalent(bridge, entity)


def _hotpot_entities_equivalent(left: Any, right: Any) -> bool:
    left_aliases = _hotpot_entity_aliases(left)
    right_aliases = _hotpot_entity_aliases(right)
    return bool(left_aliases and right_aliases and left_aliases & right_aliases)


def _hotpot_best_entity_overlap(target: str, entities: list[str]) -> str:
    for entity in entities:
        if _hotpot_entities_equivalent(target, entity):
            return entity
    return ""


def _hotpot_non_bridge_entity(bridge: str, entities: list[str]) -> str:
    for entity in entities:
        if str(entity).strip() and not _hotpot_entities_equivalent(bridge, entity):
            return entity
    return ""


def _hotpot_fact_entity_profile(fact: dict[str, Any]) -> dict[str, Any]:
    title = str(fact.get("title", "")).strip()
    text = str(fact.get("text", "")).strip()
    referenced = _hotpot_entities_in_text(text)
    aliases: set[str] = set()
    for entity in [title, *referenced]:
        aliases |= _hotpot_entity_aliases(entity)
    return {
        "title_entity": title,
        "referenced_entities": referenced,
        "aliases": aliases,
    }


def _hotpot_entity_aliases(value: Any) -> set[str]:
    base = normalize_bridge(value)
    if not base:
        return set()
    aliases = {base}
    no_parenthetical = re.sub(r"\s*\([^)]*\)", "", str(value or "")).strip()
    no_parenthetical_key = normalize_bridge(no_parenthetical)
    if no_parenthetical_key:
        aliases.add(no_parenthetical_key)
    tokens = [token for token in base.split() if token not in {"the", "a", "an"}]
    if tokens:
        aliases.add(" ".join(tokens))
    if len(tokens) >= 2:
        aliases.add(tokens[-1])
        aliases.add(" ".join(tokens[:2]))
    if len(tokens) >= 3 and tokens[-1] in {"jr", "sr", "ii", "iii", "iv"}:
        aliases.add(tokens[-2])
        aliases.add(" ".join(tokens[:-1]))
    return {alias for alias in aliases if alias and len(alias) >= 3}


def _hotpot_derive_answer_hint(question: str, hop1: dict[str, Any], hop2: dict[str, Any]) -> str:
    text = str(hop2.get("text", "") or hop1.get("text", ""))
    if "how many" in question.lower():
        match = re.search(r"\b\d[\d,]*\b", text)
        return match.group(0) if match else ""
    if "when" in question.lower() or "what year" in question.lower():
        match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", text)
        return match.group(0) if match else ""
    if "yes or no" in question.lower() or question.lower().startswith(("are ", "is ", "was ", "were ", "did ", "do ", "does ")):
        return "yes" if _hotpot_terms(str(hop1.get("text", ""))) & _hotpot_terms(str(hop2.get("text", ""))) else ""
    return ""


def _hotpot_pair_compatibility(row: dict[str, Any], bridge: str, hop1: dict[str, Any], hop2: dict[str, Any]) -> float:
    return _hotpot_chain_validity_score(row, bridge, hop1, hop2)


def _hotpot_scored_fact_as_dict(fact: dict[str, Any]) -> dict[str, Any]:
    return {
        "fact_id": str(fact.get("fact_id", "")),
        "title": str(fact.get("title", "")),
        "sent_id": fact.get("sent_id"),
        "score": float(fact.get("retrieval_score", fact.get("hop_score", 0.0)) or 0.0),
        "retrieval_score": float(fact.get("retrieval_score", fact.get("hop_score", 0.0)) or 0.0),
        "role": str(fact.get("hop_role", "")),
    }


def _hotpot_pair_as_dict(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "hop1_fact_id": str(pair.get("hop1", {}).get("fact_id", "")),
        "hop2_fact_id": str(pair.get("hop2", {}).get("fact_id", "")),
        "compatibility_score": float(pair.get("compatibility_score", 0.0) or 0.0),
        "retrieval_score": float(pair.get("retrieval_score", 0.0) or 0.0),
        "semantic_chain_verification": pair.get("semantic_chain_verification", {}),
        "semantic_chain_score": float(pair.get("semantic_chain_score", pair.get("chain_score", 0.0)) or 0.0),
        "chain_score": float(pair.get("chain_score", 0.0) or 0.0),
    }


def _rank_hotpot_facts_for_terms(facts: list[dict[str, Any]], query: str, *, limit: int) -> list[dict[str, Any]]:
    query_terms = set(re.findall(r"[a-z0-9]+", str(query).lower()))
    scored = []
    for index, fact in enumerate(facts):
        text = f"{fact.get('title', '')} {fact.get('text', '')}".lower()
        terms = set(re.findall(r"[a-z0-9]+", text))
        scored.append((-len(query_terms & terms), index, fact))
    scored.sort()
    return [fact for _, _, fact in scored[:limit]]


def hotpot_endpoint_binder_prompt(row: dict[str, Any], view: Any, fact1: dict[str, Any], fact2: dict[str, Any], reasoning_type: str) -> str:
    return (
        "Bind the semantic endpoints of two HotpotQA evidence facts. "
        "The bridge must be inferred from the evidence endpoints. Do not copy a candidate bridge unless the facts support it. "
        "Return only JSON matching the schema.\n\n"
        f"Question:\n{row.get('question', '')}\n\n"
        f"Reasoning type hint:\n{reasoning_type}\n\n"
        f"Candidate bridge hypothesis, for comparison only:\n{view.bridge_entity}\n\n"
        f"Fact 1 id: {fact1.get('fact_id', '')}\n"
        f"Fact 1 title: {fact1.get('title', '')}\n"
        f"Fact 1 text: {fact1.get('text', '')}\n\n"
        f"Fact 2 id: {fact2.get('fact_id', '')}\n"
        f"Fact 2 title: {fact2.get('title', '')}\n"
        f"Fact 2 text: {fact2.get('text', '')}\n\n"
        "Schema:\n"
        "{\n"
        '  "reasoning_type": "bridge_entity | bridge_attribute | comparison_attribute | shared_property_yesno | set_intersection | multi_constraint_lookup",\n'
        '  "fact1": {"subject": "...", "relation": "...", "object": "...", "subject_role": "entity|attribute|literal|unknown", "object_role": "entity|attribute|literal|unknown", "confidence": 0.0},\n'
        '  "fact2": {"subject": "...", "relation": "...", "object": "...", "subject_role": "entity|attribute|literal|unknown", "object_role": "entity|attribute|literal|unknown", "confidence": 0.0},\n'
        '  "cross_fact": {"inferred_bridge": "...", "bridge_alignment": "exact|alias|inferred|none", "derived_answer": "...", "answer_source": "fact1|fact2|both|comparison|unknown"},\n'
        '  "confidence": 0.0\n'
        "}"
    )


def _hotpot_bind_endpoints_with_model(
    *,
    model: SamplingModel,
    row: dict[str, Any],
    view: Any,
    facts: list[dict[str, Any]],
    reasoning_type: str,
    seed: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> tuple[HotpotEndpointBinding, dict[str, Any]]:
    fact_by_id = {str(fact.get("fact_id") or ""): fact for fact in facts}
    fact1 = fact_by_id.get(str((view.hop1_fact_ids or [""])[0]), {})
    fact2 = fact_by_id.get(str((view.hop2_fact_ids or [""])[0]), {})
    if not fact1 or not fact2:
        return HotpotEndpointBinding(reasoning_type=reasoning_type, source="missing_fact"), {
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_sec": 0.0,
        }
    prompt = hotpot_endpoint_binder_prompt(row, view, fact1, fact2, reasoning_type)
    text, metrics = model.generate(
        prompt,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )
    return parse_hotpot_endpoint_binding(text, reasoning_type=reasoning_type), metrics


def parse_hotpot_endpoint_binding(text: str, *, reasoning_type: str = "") -> HotpotEndpointBinding:
    parsed = _parse_json_payload(text)
    if not isinstance(parsed, dict):
        return HotpotEndpointBinding(reasoning_type=reasoning_type, source="parse_failed", raw_output=str(text or ""))
    fact1 = _hotpot_endpoint_from_payload(parsed.get("fact1"))
    fact2 = _hotpot_endpoint_from_payload(parsed.get("fact2"))
    cross = parsed.get("cross_fact") if isinstance(parsed.get("cross_fact"), dict) else {}
    return HotpotEndpointBinding(
        fact1=fact1,
        fact2=fact2,
        inferred_bridge=str(cross.get("inferred_bridge") or parsed.get("inferred_bridge") or "").strip(),
        bridge_alignment=str(cross.get("bridge_alignment") or parsed.get("bridge_alignment") or "").strip(),
        derived_answer=str(cross.get("derived_answer") or parsed.get("derived_answer") or "").strip(),
        answer_source=str(cross.get("answer_source") or parsed.get("answer_source") or "").strip(),
        reasoning_type=str(parsed.get("reasoning_type") or reasoning_type or "").strip(),
        confidence=_float_value(parsed.get("confidence"), default=0.0),
        source="model",
        raw_output=str(text or ""),
    )


def _hotpot_endpoint_from_payload(value: Any) -> HotpotFactEndpoint:
    payload = value if isinstance(value, dict) else {}
    return HotpotFactEndpoint(
        subject=str(payload.get("subject") or "").strip(),
        relation=str(payload.get("relation") or "").strip(),
        object=str(payload.get("object") or "").strip(),
        subject_role=str(payload.get("subject_role") or "").strip(),
        object_role=str(payload.get("object_role") or "").strip(),
        confidence=_float_value(payload.get("confidence"), default=0.0),
    )


def hotpot_preserve_first_answer_prompt(row: dict[str, Any], view: Any) -> str:
    return (
        "Derive the concise HotpotQA short answer from this preserve-first evidence chain. "
        "Use only the two evidence hops. Return only JSON: {\"answer\":\"...\"}.\n\n"
        f"Question:\n{row.get('question', '')}\n\n"
        f"Bridge entity:\n{view.bridge_entity}\n\n"
        f"Hop 1 ({', '.join(view.hop1_fact_ids)}):\n{view.hop1_key_sentence}\n\n"
        f"Hop 2 ({', '.join(view.hop2_fact_ids)}):\n{view.hop2_key_sentence}\n"
    )


def _hotpot_candidate_bridge(value: Any) -> str:
    parsed = _parse_json_payload(value)
    if isinstance(parsed, dict):
        return str(parsed.get("bridge_entity") or "").strip()
    return ""


def _hotpot_chain_signature(value: Any) -> str:
    parsed = _parse_json_payload(value)
    if not isinstance(parsed, dict):
        return ""
    return "|".join([
        normalize_bridge(parsed.get("bridge_entity")),
        ",".join(sorted(str(item) for item in parsed.get("cited_fact_ids", []) if str(item).strip())),
    ])


def _hotpot_view_signature(view: Any) -> str:
    return "|".join([
        normalize_bridge(view.bridge_entity),
        ",".join(sorted(str(item) for item in view.cited_fact_ids)),
    ])


def mmlu_option_independent_prompt(
    row: dict[str, Any],
    option_text: str,
    *,
    previous_candidates: list[Candidate],
    mode: str = "independent",
    pass_index: int = 0,
) -> str:
    question = str(row.get("question", "")).strip()
    # Hide the original option label from the model; runtime reattaches it.
    question_without_options = question.split("\n\n", 1)[0].strip()
    emphasis = {
        "permutation": (
            "This is an independent scoring pass with permuted option order. "
            "Judge this hypothesis on its content only."
        ),
        "second_pass": (
            "This is pass "
            f"{pass_index + 1}. Produce an independent second hypothesis evaluation. "
            "Use a different evidence emphasis from prior passes: first look for decisive "
            "contradictions, then support, and keep uncertainty explicit."
        ),
    }.get(mode, "This is an independent scoring pass.")
    return (
        "Evaluate one answer hypothesis for a multiple-choice question. "
        "You are not given the option label. Do not infer or mention labels like A, B, C.\n\n"
        f"{emphasis}\n\n"
        f"Question:\n{question_without_options}\n\n"
        "Evaluate this hypothesis only:\n"
        f"{option_text}\n\n"
        "Return only JSON with this schema:\n"
        "{\n"
        '  "support": ["..."],\n'
        '  "contradictions": ["..."],\n'
        '  "support_strength": 0.0,\n'
        '  "contradiction_strength": 0.0,\n'
        '  "uncertainty": 0.0\n'
        "}"
    )


def _deterministic_permutation(items: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    copied = list(items)
    rng.shuffle(copied)
    return copied


def mmlu_score_calibration_prompt(row: dict[str, Any], hypotheses: list[dict[str, Any]]) -> str:
    question = str(row.get("question", "")).split("\n\n", 1)[0].strip()
    compact = []
    for hypothesis in hypotheses:
        compact.append({
            "option_id": hypothesis.get("option_id"),
            "option_text": hypothesis.get("option_text"),
            "support": _string_list(hypothesis.get("support"))[:2],
            "contradictions": _string_list(hypothesis.get("contradictions"))[:2],
            "support_strength": _float_value(hypothesis.get("support_strength"), default=0.0),
            "contradiction_strength": _float_value(hypothesis.get("contradiction_strength"), default=0.0),
            "uncertainty": _float_value(hypothesis.get("uncertainty"), default=0.0),
        })
    return (
        "Calibrate option hypothesis scores for this multiple-choice question. "
        "Do not choose by label position. Use only the question and the listed supports/contradictions.\n\n"
        f"Question:\n{question}\n\n"
        f"Hypotheses:\n{json.dumps(compact, ensure_ascii=False, indent=2)}\n\n"
        "Return only JSON with calibrated_scores. Preserve option ids:\n"
        "{\n"
        '  "calibrated_scores": [\n'
        '    {"option_id": "A", "support_strength": 0.0, "contradiction_strength": 0.0, "uncertainty": 0.0}\n'
        "  ]\n"
        "}"
    )


def parse_mmlu_score_calibration(text: str) -> dict[str, Any]:
    parsed = _parse_json_payload(text)
    scores: list[dict[str, Any]] = []
    if isinstance(parsed, dict):
        items = parsed.get("calibrated_scores") or parsed.get("scores") or parsed.get("options")
        if isinstance(items, list):
            scores = [item for item in items if isinstance(item, dict)]
    elif isinstance(parsed, list):
        scores = [item for item in parsed if isinstance(item, dict)]
    clean = []
    for item in scores:
        option_id = str(item.get("option_id") or item.get("label") or item.get("option") or "").strip().upper()
        if not re.fullmatch(r"[A-J]", option_id):
            continue
        clean.append({
            "option_id": option_id,
            "support_strength": _float_value(item.get("support_strength"), default=0.0),
            "contradiction_strength": _float_value(item.get("contradiction_strength"), default=0.0),
            "uncertainty": _float_value(item.get("uncertainty"), default=0.0),
        })
    return {"calibrated_scores": clean, "parse_fail": not bool(clean)}


def _apply_mmlu_score_calibration(hypotheses: list[dict[str, Any]], calibration: dict[str, Any]) -> None:
    by_id = {
        str(item.get("option_id") or "").strip().upper(): item
        for item in calibration.get("calibrated_scores", [])
        if isinstance(item, dict)
    }
    for hypothesis in hypotheses:
        option_id = str(hypothesis.get("option_id") or "").strip().upper()
        calibrated = by_id.get(option_id)
        if not calibrated:
            continue
        hypothesis["raw_support_strength"] = hypothesis.get("support_strength")
        hypothesis["raw_contradiction_strength"] = hypothesis.get("contradiction_strength")
        hypothesis["raw_uncertainty"] = hypothesis.get("uncertainty")
        hypothesis["support_strength"] = calibrated["support_strength"]
        hypothesis["contradiction_strength"] = calibrated["contradiction_strength"]
        hypothesis["uncertainty"] = calibrated["uncertainty"]
        hypothesis["score_calibrated"] = True


def mmlu_pairwise_option_critic_prompt(
    row: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    top2: list[tuple[str, float]],
) -> str:
    by_id = {
        str(hypothesis.get("option_id") or "").strip().upper(): hypothesis
        for hypothesis in hypotheses
    }
    first = by_id.get(top2[0][0], {})
    second = by_id.get(top2[1][0], {})
    question = str(row.get("question", "")).split("\n\n", 1)[0].strip()
    return (
        "The cheap option-wise scorer is uncertain. Compare only the two listed options.\n\n"
        f"Question:\n{question}\n\n"
        f"Option {top2[0][0]}:\n"
        f"text: {first.get('option_text', '')}\n"
        f"strongest support: {_first_text(first.get('support'))}\n"
        f"strongest contradiction: {_first_text(first.get('contradictions'))}\n"
        f"cheap_score: {top2[0][1]:.3f}\n\n"
        f"Option {top2[1][0]}:\n"
        f"text: {second.get('option_text', '')}\n"
        f"strongest support: {_first_text(second.get('support'))}\n"
        f"strongest contradiction: {_first_text(second.get('contradictions'))}\n"
        f"cheap_score: {top2[1][1]:.3f}\n\n"
        "For each option, identify the strongest support, strongest contradiction, and any condition violated. "
        "Then choose which option better satisfies the question.\n\n"
        "Return only JSON:\n"
        "{\n"
        f'  "selected_option_id": "{top2[0][0]} or {top2[1][0]}",\n'
        '  "reason": "...",\n'
        '  "confidence": 0.0\n'
        "}"
    )


def parse_mmlu_pairwise_critic(text: str, *, allowed_option_ids: list[str]) -> dict[str, Any]:
    allowed = {str(option_id).strip().upper() for option_id in allowed_option_ids}
    parsed = _parse_json_payload(text)
    selected = ""
    reason = ""
    confidence = 0.0
    if isinstance(parsed, dict):
        selected = str(
            parsed.get("selected_option_id")
            or parsed.get("better_option_id")
            or parsed.get("choice")
            or ""
        ).strip().upper()
        reason = str(parsed.get("reason") or parsed.get("rationale") or "")
        confidence = _float_value(parsed.get("confidence"), default=0.0)
    if selected not in allowed:
        labelled = re.findall(
            r"(?:selected_option_id|better_option|choice|answer)\s*[:：]\s*([A-J])\b",
            text,
            flags=re.IGNORECASE,
        )
        selected = labelled[-1].upper() if labelled and labelled[-1].upper() in allowed else ""
    return {
        "selected_option_id": selected,
        "reason": reason,
        "confidence": confidence,
        "allowed_option_ids": sorted(allowed),
    }


def _top2_mmlu_options(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[:2]


def _first_text(value: Any) -> str:
    items = _string_list(value)
    return items[0] if items else "none"


def parse_single_option_hypothesis(text: str) -> dict[str, Any]:
    parsed = _parse_json_payload(text)
    if isinstance(parsed, dict):
        return {
            "support": _string_list(parsed.get("support")),
            "contradictions": _string_list(parsed.get("contradictions", parsed.get("contradiction"))),
            "support_strength": _float_value(parsed.get("support_strength"), default=_option_strength(parsed, "support_strength", "support")),
            "contradiction_strength": _float_value(
                parsed.get("contradiction_strength"),
                default=_option_strength(parsed, "contradiction_strength", "contradictions", "contradiction"),
            ),
            "uncertainty": _float_value(parsed.get("uncertainty"), default=0.0),
        }
    return {
        "support": _extract_lines_after_label(text, "support"),
        "contradictions": _extract_lines_after_label(text, "contradiction"),
        "support_strength": _extract_float_after_label(text, "support_strength"),
        "contradiction_strength": _extract_float_after_label(text, "contradiction_strength"),
        "uncertainty": _extract_float_after_label(text, "uncertainty"),
    }


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def mark_candidate_redundancy(candidate: Candidate, previous: list[Candidate], *, domain: str) -> Candidate:
    if not previous:
        return candidate
    if domain == "hotpotqa":
        current_signature = _hotpot_chain_signature(candidate.text)
        current_bridge = normalize_bridge(_hotpot_candidate_bridge(candidate.text))
        for earlier in previous:
            if current_signature and current_signature == _hotpot_chain_signature(earlier.text):
                candidate.redundant_candidate = True
                candidate.redundancy_reason = "same_bridge_and_same_evidence_chain"
                return candidate
        if current_bridge and any(current_bridge == normalize_bridge(_hotpot_candidate_bridge(earlier.text)) for earlier in previous):
            candidate.redundancy_reason = "same_bridge_different_evidence_chain"
        return candidate
    if domain != "mmlu_pro":
        return candidate
    current_answer = str(candidate_artifact(domain, candidate.text)[2] or "").strip().upper()
    best_similarity = 0.0
    same_answer_high_similarity = False
    for earlier in previous:
        earlier_answer = str(candidate_artifact(domain, earlier.text)[2] or "").strip().upper()
        similarity = _text_jaccard_similarity(candidate.text, earlier.text)
        best_similarity = max(best_similarity, similarity)
        if current_answer and current_answer == earlier_answer and similarity > 0.75:
            same_answer_high_similarity = True
    candidate.rationale_similarity_to_previous = best_similarity
    if same_answer_high_similarity:
        candidate.redundant_candidate = True
        candidate.redundancy_reason = "same_answer_and_high_rationale_similarity"
    return candidate


def _text_jaccard_similarity(left: Any, right: Any) -> float:
    left_tokens = set(_rationale_tokens(left))
    right_tokens = set(_rationale_tokens(right))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _rationale_tokens(value: Any) -> list[str]:
    text = str(value or "").lower()
    text = re.sub(r"final\s+answer\s*[:：]?\s*[a-j]\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return [token for token in text.split() if len(token) > 2]


def run_best_of_n(
    model: SamplingModel,
    rows: list[dict[str, Any]],
    *,
    domain: str,
    n: int,
    seed: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    records = []
    for index, row in enumerate(rows, start=1):
        candidates: list[Candidate] = []
        for attempt in range(n):
            candidates.append(
                generate_attempt_candidate(
                model,
                row,
                domain,
                attempt=attempt,
                previous_candidates=candidates,
                seed=seed + index * 1009 + attempt,
                temperature=temperature,
                top_p=top_p,
                do_sample=attempt > 0,
            )
            )
        correct_any = any(c.score.get("correct") for c in candidates)
        correct_count = sum(int(bool(c.score.get("correct"))) for c in candidates)
        candidate_records = [candidate_record(c, domain=domain, candidate_id=f"c{attempt}") for attempt, c in enumerate(candidates)]
        coverage = coverage_at_k(candidate_records, (1, 2, 3, 4, 8, 16))
        records.append({
            "sample_id": row.get("sample_id"),
            "correct": correct_any,
            "correct_count": correct_count,
            "candidate_count": len(candidates),
            "best_of_n": n,
            **coverage,
            "candidates": candidate_records,
            "candidate_graph": record_candidate_graph(
                row,
                domain=domain,
                group_id=f"{row.get('sample_id', index)}_best_of_{n}",
                candidates=candidate_records,
                selector_policy="verifier_guided_first_verified",
                selected_candidate_id=None,
                metadata={"experiment": "best_of_n", "n": n},
            ),
        })
        print(f"{index}/{len(rows)} {row.get('sample_id')} best_of_{n} correct={correct_any} correct_count={correct_count}", flush=True)
    return summarize(records, domain=domain, experiment="best_of_n", n=n)


def run_fixed_candidate_pool(
    model: SamplingModel,
    rows: list[dict[str, Any]],
    *,
    domain: str,
    n: int,
    seed: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    """Generate exactly K candidates and score pool coverage only.

    This intentionally bypasses selector, repair, and adaptive early-stop so
    candidate ceilings can be measured without controller confounds.
    """
    report = run_best_of_n(
        model,
        rows,
        domain=domain,
        n=n,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
    )
    report["experiment"] = "fixed_candidate_pool"
    report["fixed_k"] = n
    report["selector_disabled"] = True
    report["repair_disabled"] = True
    report["adaptive_early_stop_disabled"] = True
    if domain == "hotpotqa":
        report["hotpot_fixed_k_gate"] = _hotpot_fixed_k_gate(report, n=n)
    return report


def run_oracle_select(
    model: SamplingModel,
    rows: list[dict[str, Any]],
    *,
    domain: str,
    n: int,
    seed: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    report = run_best_of_n(
        model,
        rows,
        domain=domain,
        n=n,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
    )
    report["experiment"] = "oracle_select"
    report["selector"] = "unit_tests" if domain in CODE_DOMAINS else f"gold_{domain}_oracle"
    report["uses_gold_selector"] = domain not in CODE_DOMAINS
    return report


def run_verifier_select(
    model: SamplingModel,
    rows: list[dict[str, Any]],
    *,
    domain: str,
    n: int,
    seed: int,
    temperature: float,
    top_p: float,
    current_baseline: float = 0.0,
) -> dict[str, Any]:
    records = []
    for index, row in enumerate(rows, start=1):
        candidates: list[Candidate] = []
        for attempt in range(n):
            candidates.append(
                generate_attempt_candidate(
                model,
                row,
                domain,
                attempt=attempt,
                previous_candidates=candidates,
                seed=seed + index * 1009 + attempt,
                temperature=temperature,
                top_p=top_p,
                do_sample=attempt > 0,
            )
            )
        candidate_records = [
            candidate_record(candidate, domain=domain, candidate_id=f"c{attempt}")
            for attempt, candidate in enumerate(candidates)
        ]
        deployable_records = deployable_candidate_records(candidate_records, domain=domain, row=row)
        selected, candidate_graph = record_deployable_selection_graph(
            row,
            domain=domain,
            group_id=f"{row.get('sample_id', index)}_verifier_select_{n}",
            candidates=deployable_records,
        )
        selected_original = next(
            item for item in candidate_records
            if item.get("candidate_id") == selected.selected_candidate_id
        )
        correct_any = any(c.get("correct") for c in candidate_records)
        selected_correct = bool(selected_original.get("correct"))
        coverage = coverage_at_k(candidate_records, (1, 2, 3, 4, 8, 16))
        records.append({
            "sample_id": row.get("sample_id"),
            "correct": selected_correct,
            "oracle_has_correct": correct_any,
            "selected_candidate_id": selected.selected_candidate_id,
            "selected_candidate_score": selected.selection_score,
            "selected_candidate_confidence": selected.confidence,
            "selection_reason": list(selected.selection_reason),
            "candidate_count": len(candidate_records),
            "best_of_n": n,
            **coverage,
            "candidates": candidate_records,
            "deployable_candidates": deployable_records,
            "candidate_graph": candidate_graph,
            "selection_trace": list(selected.trace),
        })
        print(
            f"{index}/{len(rows)} {row.get('sample_id')} verifier_select_{n} "
            f"correct={selected_correct} oracle_has_correct={correct_any} selected={selected.selected_candidate_id}",
            flush=True,
        )
    report = summarize(records, domain=domain, experiment="verifier_select", n=n)
    selected_correct = sum(int(bool(record.get("correct"))) for record in records)
    false_positive = sum(int(not bool(record.get("correct"))) for record in records)
    false_negative = sum(
        int((not bool(record.get("correct"))) and bool(record.get("oracle_has_correct")))
        for record in records
    )
    count = len(records)
    oracle = float(report.get(f"coverage@{n}", report.get("accuracy_or_pass_at_n", 0.0)) or 0.0)
    verifier = selected_correct / count if count else 0.0
    report.update({
        "current_baseline": current_baseline,
        "oracle_select_at_n": oracle,
        "verifier_select_at_n": verifier,
        "gap_verifier": oracle - verifier,
        "gap_recovery": (
            (verifier - current_baseline) / (oracle - current_baseline)
            if oracle > current_baseline else 0.0
        ),
        "selection_accuracy": verifier,
        "false_positive_selection_count": false_positive,
        "false_negative_selection_count": false_negative,
        "mean_selection_confidence": (
            sum(float(record.get("selected_candidate_confidence", 0.0) or 0.0) for record in records) / count
            if count else 0.0
        ),
        "passes_phase_gate": verifier >= current_baseline if current_baseline else None,
    })
    return report


def run_select_repair(
    model: SamplingModel,
    rows: list[dict[str, Any]],
    *,
    domain: str,
    n: int,
    seed: int,
    temperature: float,
    top_p: float,
    verifier_baseline: float = 0.0,
    experiment_name: str = "select_repair",
    min_repair_candidates: int = 4,
    selector_communication_policy: str = "deployable_full",
) -> dict[str, Any]:
    records = []
    for index, row in enumerate(rows, start=1):
        prompt = task_prompt(row, domain)
        candidates: list[Candidate] = []
        candidate_records: list[dict[str, Any]] = []
        deployable_records: list[dict[str, Any]] = []
        selected = None
        candidate_graph: dict[str, Any] = {}
        selected_index = 0
        action = "GENERATE"
        actions: list[str] = []
        controller_reasons: list[str] = []
        generate_count = 0
        selector_communication_steps: list[dict[str, Any]] = []

        while len(candidates) < max(1, n):
            attempt = len(candidates)
            generated = generate_attempt_candidate(
                model,
                row,
                domain,
                attempt=attempt,
                previous_candidates=candidates,
                seed=seed + index * 1009 + attempt,
                temperature=temperature,
                top_p=top_p,
                do_sample=attempt > 0,
            )
            generate_count += 1
            actions.append("GENERATE")
            candidates.append(generated)
            candidate_records = [
                candidate_record(candidate, domain=domain, candidate_id=f"c{candidate_index}")
                for candidate_index, candidate in enumerate(candidates)
            ]
            deployable_records = deployable_candidate_records(candidate_records, domain=domain, row=row)
            selected, candidate_graph, selector_comm = select_candidate_for_adaptive_controller(
                row=row,
                domain=domain,
                group_id=f"{row.get('sample_id', index)}_select_repair_{n}_step_{len(candidates)}",
                candidates=deployable_records,
                policy=selector_communication_policy,
                seed=seed + index * 1009 + attempt,
            )
            selector_communication_steps.append(selector_comm)
            selected_index = _candidate_index_from_id(selected.selected_candidate_id, len(candidates))
            selected_deployable = deployable_records[selected_index]
            decision = adaptive_controller_decision(
                domain=domain,
                selected_candidate=selected_deployable,
                selection=selected,
                candidate_count=len(candidates),
                max_candidates=n,
                min_repair_candidates=min_repair_candidates,
            )
            action = decision.action
            controller_reasons.append(decision.reason)
            if action == "REQUEST":
                actions.append("REQUEST")
                action = "GENERATE"
            if action == "SELECT":
                actions.extend(["SELECT", "STOP"])
                break
            if action == "REPAIR":
                actions.append("REPAIR")
                break

        assert selected is not None
        selected_index = _candidate_index_from_id(selected.selected_candidate_id, len(candidates))
        selected_candidate = candidates[selected_index]
        selected_record = candidate_records[selected_index]
        selected_deployable = deployable_records[selected_index]
        selected_correct = bool(selected_record.get("correct"))
        repair_attempt: Candidate | None = None
        repair_feedback: dict[str, Any] | None = None
        final_candidate = selected_candidate
        final_record = selected_record

        if action == "REPAIR" and _has_actionable_failure(domain, selected_deployable):
            repair_feedback = build_repair_feedback(row, domain, selected_candidate)
            repair_attempt = generate_candidate(
                model,
                row,
                domain,
                repair_prompt(row, domain, selected_candidate, repair_feedback, round_id=1),
                seed=seed + index * 1009 + n + 1,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repair_round=1,
                feedback=repair_feedback,
            )
            final_candidate = repair_attempt
            final_record = candidate_record(repair_attempt, domain=domain, candidate_id=f"{selected.selected_candidate_id}_repair_1")
        elif action != "SELECT":
            action = "SELECT"
            actions.extend(["SELECT", "STOP"])

        before_correct = selected_correct
        after_correct = bool(final_record.get("correct"))
        transition = repair_transition(before_correct, after_correct)
        coverage = coverage_at_k(candidate_records, (1, 2, 3, 4, 8, 16))
        record = {
            "sample_id": row.get("sample_id"),
            "correct": after_correct,
            "selected_candidate_id": selected.selected_candidate_id,
            "selected_correct": before_correct,
            "final_candidate_id": final_record.get("candidate_id"),
            "transition": transition,
            "action": action,
            "actions": actions,
            "controller_reasons": controller_reasons,
            "repair_attempted": repair_attempt is not None,
            "repair_success": transition == "W->C",
            "repair_regression": transition == "C->W",
            "extra_generation": generate_count > 1,
            "extra_generation_count": max(0, generate_count - 1),
            "generation_attempt_count": generate_count,
            "candidate_reused": final_record.get("candidate_id") == selected_record.get("candidate_id"),
            "candidate_count": len(candidate_records),
            "max_candidate_budget": n,
            **coverage,
            "selected": selected.as_dict(),
            "candidate_graph": candidate_graph,
            "candidates": candidate_records,
            "deployable_candidates": deployable_records,
            "feedback": repair_feedback,
            "selector_communication_steps": selector_communication_steps,
        }
        if repair_attempt:
            record["repair"] = candidate_record(repair_attempt, domain=domain, candidate_id=f"{selected.selected_candidate_id}_repair_1")
        records.append(record)
        print(
            f"{index}/{len(rows)} {row.get('sample_id')} select_repair_{n} "
            f"actions={','.join(actions)} transition={transition} correct={after_correct}",
            flush=True,
        )
    report = summarize(records, domain=domain, experiment=experiment_name, n=n)
    correct_count = int(report.get("correct_count", 0) or 0)
    repair_attempt_count = sum(int(record.get("repair_attempted", False)) for record in records)
    successful_repair_count = sum(int(record.get("repair_success", False)) for record in records)
    repair_regression_count = sum(int(record.get("repair_regression", False)) for record in records)
    extra_generation_count = sum(int(record.get("extra_generation_count", 0) or 0) for record in records)
    candidate_reuse_count = sum(int(record.get("candidate_reused", False)) for record in records)
    wrong_to_correct = sum(int(record.get("transition") == "W->C") for record in records)
    correct_to_wrong = sum(int(record.get("transition") == "C->W") for record in records)
    controller_action_counts: dict[str, int] = {}
    for record in records:
        for controller_action in record.get("actions", []):
            controller_action_counts[controller_action] = controller_action_counts.get(controller_action, 0) + 1
    report.update({
        "verifier_baseline": verifier_baseline,
        "min_repair_candidates": min_repair_candidates,
        "select_repair_accuracy": report.get("accuracy_or_pass_at_n", 0.0),
        "adaptive_controller_accuracy": report.get("accuracy_or_pass_at_n", 0.0),
        "controller_action_counts": controller_action_counts,
        "select_action_count": controller_action_counts.get("SELECT", 0),
        "request_action_count": controller_action_counts.get("REQUEST", 0),
        "generate_action_count": controller_action_counts.get("GENERATE", 0),
        "repair_action_count": controller_action_counts.get("REPAIR", 0),
        "stop_action_count": controller_action_counts.get("STOP", 0),
        "phase5_gate_select_repair_ge_verifier": (
            float(report.get("accuracy_or_pass_at_n", 0.0) or 0.0) >= verifier_baseline
            if verifier_baseline else None
        ),
        "repair_attempt_count": repair_attempt_count,
        "successful_repair_count": successful_repair_count,
        "repair_regression_count": repair_regression_count,
        "extra_generation_count": extra_generation_count,
        "generation_attempt_count": sum(int(record.get("generation_attempt_count", 0) or 0) for record in records),
        "candidate_reuse_count": candidate_reuse_count,
        "candidate_reuse_rate": candidate_reuse_count / len(records) if records else 0.0,
        "wrong_to_correct_count": wrong_to_correct,
        "correct_to_wrong_count": correct_to_wrong,
        "net_repair_gain": wrong_to_correct - correct_to_wrong,
        "cw_regression_rate": correct_to_wrong / max(1, sum(int(record.get("selected_correct", False)) for record in records)),
        "tokens_per_correct": (
            int(report.get("total_model_tokens", 0) or 0) / correct_count
            if correct_count else None
        ),
        "observed_candidate_model_tokens": sum(
            sum(int(candidate.get("model_tokens", 0) or 0) for candidate in record.get("candidates", []))
            for record in records
        ),
        "estimated_bruteforce_best_of_n_model_tokens": 0,
        "bruteforce_tokens_per_oracle_correct": None,
        "selector_communication_policy": selector_communication_policy,
        "selector_input_tokens": sum(
            sum(int(step.get("selector_input_tokens", 0) or 0) for step in record.get("selector_communication_steps", []))
            for record in records
        ),
        "selector_send_all_equivalent_tokens": sum(
            sum(int(step.get("selector_send_all_equivalent_tokens", 0) or 0) for step in record.get("selector_communication_steps", []))
            for record in records
        ),
    })
    report["selector_token_saving_vs_send_all"] = (
        1.0 - report["selector_input_tokens"] / report["selector_send_all_equivalent_tokens"]
        if report["selector_send_all_equivalent_tokens"] else 0.0
    )
    generated_candidate_tokens = [
        int(candidate.get("model_tokens", 0) or 0)
        for record in records
        for candidate in record.get("candidates", [])
    ]
    avg_candidate_tokens = (
        sum(generated_candidate_tokens) / len(generated_candidate_tokens)
        if generated_candidate_tokens else 0.0
    )
    report["estimated_bruteforce_best_of_n_model_tokens"] = avg_candidate_tokens * n * len(records)
    oracle_correct = sum(int(any(candidate.get("correct") for candidate in record.get("candidates", []))) for record in records)
    if oracle_correct:
        report["bruteforce_tokens_per_oracle_correct"] = report["estimated_bruteforce_best_of_n_model_tokens"] / oracle_correct
        report["phase5_gate_tokens_per_correct_better_than_bruteforce"] = (
            report["tokens_per_correct"] is not None
            and report["tokens_per_correct"] <= report["bruteforce_tokens_per_oracle_correct"]
        )
    else:
        report["phase5_gate_tokens_per_correct_better_than_bruteforce"] = None
    return report


def run_oracle_repair(
    model: SamplingModel,
    rows: list[dict[str, Any]],
    *,
    domain: str,
    seed: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    records = []
    for index, row in enumerate(rows, start=1):
        initial = generate_candidate(
            model,
            row,
            domain,
            generation_prompt(row, domain, attempt=0, previous_candidates=[])[0],
            seed=seed + index * 1009,
            temperature=temperature,
            top_p=top_p,
            do_sample=False,
            generation_role=generation_prompt(row, domain, attempt=0, previous_candidates=[])[1],
        )
        if initial.score.get("correct"):
            records.append({
                "sample_id": row.get("sample_id"),
                "initial_correct": True,
                "correct": True,
                "wrong_to_correct": False,
                "candidate": candidate_record(initial, domain=domain, candidate_id="initial"),
                "candidate_graph": record_candidate_graph(
                    row,
                    domain=domain,
                    group_id=f"{row.get('sample_id', index)}_oracle_repair_initial",
                    candidates=[candidate_record(initial, domain=domain, candidate_id="initial")],
                    selector_policy="verifier_guided_first_verified",
                    selected_candidate_id=None,
                    metadata={"experiment": "oracle_repair", "initial_correct": True},
                ),
            })
            print(f"{index}/{len(rows)} {row.get('sample_id')} initial_correct=True", flush=True)
            continue
        feedback = build_repair_feedback(row, domain, initial)
        repaired = generate_candidate(
            model,
            row,
            domain,
            repair_prompt(row, domain, initial, feedback, round_id=1),
            seed=seed + index * 1009 + 1,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repair_round=1,
            feedback=feedback,
        )
        records.append({
            "sample_id": row.get("sample_id"),
            "initial_correct": False,
            "correct": bool(repaired.score.get("correct")),
            "wrong_to_correct": bool(repaired.score.get("correct")),
            "initial": candidate_record(initial, domain=domain, candidate_id="initial"),
            "repair": candidate_record(repaired, domain=domain, candidate_id="repair_1"),
            "feedback": feedback,
            "candidate_graph": record_candidate_graph(
                row,
                domain=domain,
                group_id=f"{row.get('sample_id', index)}_oracle_repair",
                candidates=[
                    candidate_record(initial, domain=domain, candidate_id="initial"),
                    candidate_record(repaired, domain=domain, candidate_id="repair_1"),
                ],
                selector_policy="verifier_guided_first_verified",
                selected_candidate_id=None,
                metadata={"experiment": "oracle_repair", "initial_correct": False},
            ),
        })
        print(f"{index}/{len(rows)} {row.get('sample_id')} oracle_repair correct={bool(repaired.score.get('correct'))}", flush=True)
    return summarize(records, domain=domain, experiment="oracle_repair")


def run_iterative_repair(
    model: SamplingModel,
    rows: list[dict[str, Any]],
    *,
    domain: str,
    max_rounds: int,
    seed: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    records = []
    for index, row in enumerate(rows, start=1):
        attempts: list[Candidate] = []
        current = generate_candidate(
            model,
            row,
            domain,
            generation_prompt(row, domain, attempt=0, previous_candidates=[])[0],
            seed=seed + index * 1009,
            temperature=temperature,
            top_p=top_p,
            do_sample=False,
            generation_role=generation_prompt(row, domain, attempt=0, previous_candidates=[])[1],
        )
        attempts.append(current)
        first_success_round = 0 if current.score.get("correct") else None
        for round_id in range(1, max_rounds + 1):
            if first_success_round is not None:
                break
            feedback = build_repair_feedback(row, domain, current)
            current = generate_candidate(
                model,
                row,
                domain,
                repair_prompt(row, domain, current, feedback, round_id=round_id),
                seed=seed + index * 1009 + round_id,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repair_round=round_id,
                feedback=feedback,
            )
            attempts.append(current)
            if current.score.get("correct"):
                first_success_round = round_id
                break
        records.append({
            "sample_id": row.get("sample_id"),
            "correct": first_success_round is not None,
            "first_success_round": first_success_round,
            "attempt_count": len(attempts),
            "attempts": [
                candidate_record(c, domain=domain, candidate_id=f"round_{attempt_index}")
                for attempt_index, c in enumerate(attempts)
            ],
            "candidate_graph": record_candidate_graph(
                row,
                domain=domain,
                group_id=f"{row.get('sample_id', index)}_iterative",
                candidates=[
                    candidate_record(c, domain=domain, candidate_id=f"round_{attempt_index}")
                    for attempt_index, c in enumerate(attempts)
                ],
                selector_policy="verifier_guided_first_verified",
                selected_candidate_id=None,
                metadata={"experiment": "iterative_repair", "max_rounds": max_rounds},
            ),
        })
        print(f"{index}/{len(rows)} {row.get('sample_id')} iterative correct={first_success_round is not None} round={first_success_round}", flush=True)
    return summarize(records, domain=domain, experiment="iterative_repair", max_rounds=max_rounds)


def candidate_record(candidate: Candidate, *, domain: str | None = None, candidate_id: str | None = None) -> dict[str, Any]:
    artifact_type = "text"
    artifact = candidate.text
    final_value = None
    if domain is not None:
        artifact_type, artifact, final_value = candidate_artifact(domain, candidate.text)
    extra: dict[str, Any] = {}
    if domain == "mmlu_pro":
        parsed = _parse_json_payload(candidate.text)
        hypotheses = extract_mmlu_option_hypotheses(candidate.text)
        if hypotheses:
            extra["option_hypotheses"] = hypotheses
            extra["option_hypothesis_scores"] = mmlu_option_hypothesis_scores(hypotheses)
            if isinstance(parsed, dict):
                extra["runtime_selection"] = parsed.get("runtime_selection", {})
                extra["pairwise_critic"] = parsed.get("pairwise_critic", {})
                extra["optionwise_mode"] = parsed.get("optionwise_mode", "")
                extra["score_calibration"] = parsed.get("score_calibration", {})
            view = build_mmlu_option_hypothesis_graph_view(
                candidate_id=str(candidate_id or ""),
                hypotheses=hypotheses,
                selected_option_id=str(final_value or ""),
            )
            extra["option_hypothesis_view"] = view.as_dict()
    elif domain == "hotpotqa":
        parsed = _parse_json_payload(candidate.text)
        if isinstance(parsed, dict) and parsed.get("generation_mode") in {
            "preserve_first_evidence_chain_binding",
            "bridge_conditioned_evidence_chain",
        }:
            extra["bridge_entity"] = parsed.get("bridge_entity", "")
            extra["cited_fact_ids"] = parsed.get("cited_fact_ids", [])
            extra["hotpot_chain_signature"] = parsed.get("chain_signature", _hotpot_chain_signature(candidate.text))
            extra["structured_candidate_view"] = parsed.get("structured_candidate_view", {})
            extra["answer_alignment"] = parsed.get("answer_alignment")
            extra["contradiction_signal"] = bool(parsed.get("contradiction_signal", False))
            extra["reasoning_type"] = parsed.get("reasoning_type", "")
            extra["allocation_policy"] = parsed.get("allocation_policy", "")
            extra["constructor_family"] = parsed.get("constructor_family", "")
            extra["bridge_entity_telemetry"] = parsed.get("bridge_entity_telemetry", {})
            extra["bridge_attribute_telemetry"] = parsed.get("bridge_attribute_telemetry", {})
            extra["multi_constraint_telemetry"] = parsed.get("multi_constraint_telemetry", {})
            extra["parallel_retrieval_telemetry"] = parsed.get("parallel_retrieval_telemetry", {})
            extra["bridge_candidate_pool"] = parsed.get("bridge_candidate_pool", [])
            extra["raw_bridge_pool_size"] = int(parsed.get("raw_bridge_pool_size", 0) or 0)
            extra["hop1_pool"] = parsed.get("hop1_pool", [])
            extra["hop2_pool"] = parsed.get("hop2_pool", [])
            extra["chain_pair_pool"] = parsed.get("chain_pair_pool", [])
    return {
        "candidate_id": candidate_id,
        "correct": bool(candidate.score.get("correct")),
        "score": candidate.score,
        "artifact_type": artifact_type,
        "artifact": artifact,
        "final_value": final_value,
        "input_tokens": candidate.input_tokens,
        "output_tokens": candidate.output_tokens,
        "model_tokens": candidate.input_tokens + candidate.output_tokens,
        "latency_sec": candidate.latency_sec,
        "seed": candidate.seed,
        "repair_round": candidate.repair_round,
        "generation_role": candidate.generation_role,
        "redundant_candidate": candidate.redundant_candidate,
        "redundancy_reason": candidate.redundancy_reason,
        "rationale_similarity_to_previous": candidate.rationale_similarity_to_previous,
        "raw_tail": candidate.text[-1000:],
        "feedback": candidate.feedback,
        **extra,
    }


def first_correct_candidate_id(candidates: list[dict[str, Any]]) -> str | None:
    for candidate in candidates:
        if candidate.get("correct"):
            return str(candidate.get("candidate_id"))
    return None


def deployable_candidate_records(
    candidates: list[dict[str, Any]],
    *,
    domain: str,
    row: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    answer_counts: dict[str, int] = {}
    if domain in {"gsm8k", "mmlu_pro", "hotpotqa"}:
        for candidate in candidates:
            if domain == "gsm8k":
                value = str(candidate.get("final_value") or extract_numeric_answer(candidate.get("raw_tail", ""))).strip()
            elif domain == "mmlu_pro":
                value = str(
                    candidate.get("final_value")
                    or select_mmlu_option_from_hypotheses(candidate.get("raw_tail", ""))
                    or extract_mmlu_final_choice(candidate.get("raw_tail", ""))
                ).strip()
            else:
                value = str(candidate.get("final_value") or extract_hotpot_short_answer(candidate.get("raw_tail", ""))).strip()
            if value:
                answer_counts[value] = answer_counts.get(value, 0) + 1
    records = []
    for candidate in candidates:
        if domain == "gsm8k":
            signals = numeric_features(candidate, answer_counts, len(candidates))
            source = "deployable_numeric_check"
        elif domain == "mmlu_pro":
            signals = mmlu_choice_features(candidate, row=row, answer_counts=answer_counts, total=len(candidates))
            source = "deployable_mmlu_option_check"
        elif domain == "hotpotqa":
            signals = hotpot_answer_features(candidate, row=row, answer_counts=answer_counts, total=len(candidates))
            source = "deployable_hotpot_answer_evidence_check"
        else:
            signals = code_features(candidate)
            if domain == "mbpp" and row is not None:
                signals = {**signals, **mbpp_allowed_test_features(row, candidate)}
            source = "deployable_static_generated_tests"
        records.append({
            **candidate,
            "score": signals,
            "verification_signals": signals,
            "verification_source": source,
        })
    return records


def mmlu_choice_features(
    candidate: dict[str, Any],
    *,
    row: dict[str, Any] | None,
    answer_counts: dict[str, int],
    total: int,
) -> dict[str, Any]:
    raw = str(candidate.get("raw_tail") or "")
    hypotheses = candidate.get("option_hypotheses") if isinstance(candidate.get("option_hypotheses"), list) else extract_mmlu_option_hypotheses(raw)
    hypothesis_scores = mmlu_option_hypothesis_scores(hypotheses)
    choice = str(
        candidate.get("final_value")
        or select_mmlu_option_from_hypotheses(raw)
        or extract_mmlu_final_choice(raw)
    ).strip().upper()
    labels = []
    if row is not None:
        labels = [
            str(item.get("label", "")).strip().upper()
            for item in row.get("choices", [])
            if isinstance(item, dict) and str(item.get("label", "")).strip()
        ]
    if not labels:
        labels = list("ABCDEFGHIJ")
    option_valid = bool(choice and choice in set(labels))
    consensus = answer_counts.get(choice, 0) / max(1, total) if choice else 0.0
    features: dict[str, Any] = {
        "option_valid": option_valid,
        "invalid_option": not option_valid,
        "final_value_valid": option_valid,
        "option_hypothesis_parseable": bool(hypotheses),
        "option_hypothesis_count": len(hypotheses),
        "option_score_margin": _top_option_margin(hypothesis_scores),
        "cross_candidate_answer_consensus": consensus,
        "confidence": min(
            0.95,
            (0.30 if option_valid else 0.05)
            + (0.20 if hypotheses else 0.0)
            + 0.30 * consensus
            + min(0.20, max(0.0, _top_option_margin(hypothesis_scores))),
        ),
    }
    return features


def mmlu_option_hypothesis_scores(hypotheses: list[dict[str, Any]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for hypothesis in hypotheses:
        option_id = str(hypothesis.get("option_id") or hypothesis.get("label") or hypothesis.get("option") or "").strip().upper()
        if not re.fullmatch(r"[A-J]", option_id):
            continue
        support = _option_strength(hypothesis, "support_strength", "support")
        contradiction = _option_strength(hypothesis, "contradiction_strength", "contradictions", "contradiction")
        uncertainty = _float_value(hypothesis.get("uncertainty"), default=0.0)
        scores[option_id] = support - contradiction - uncertainty
    return scores


def _top_option_margin(scores: dict[str, float]) -> float:
    if len(scores) < 2:
        return 0.0
    values = sorted(scores.values(), reverse=True)
    return values[0] - values[1]


def hotpot_answer_features(
    candidate: dict[str, Any],
    *,
    row: dict[str, Any] | None,
    answer_counts: dict[str, int],
    total: int,
) -> dict[str, Any]:
    answer = str(candidate.get("final_value") or extract_hotpot_short_answer(candidate.get("raw_tail", ""))).strip()
    normalized = hotpot_normalize(answer)
    answer_valid = bool(normalized)
    consensus = answer_counts.get(answer, 0) / max(1, total) if answer else 0.0
    evidence_covered = _hotpot_answer_covered_by_visible_evidence(row, answer) if row is not None else False
    raw = str(candidate.get("raw_tail") or candidate.get("artifact") or "")
    mentions_evidence = bool(re.search(r"\b(evidence|fact|bridge|because|support)\b", raw, flags=re.IGNORECASE))
    features: dict[str, Any] = {
        "final_value_valid": answer_valid,
        "evidence_covered": evidence_covered,
        "missing_evidence": answer_valid and not evidence_covered and not mentions_evidence,
        "cross_candidate_answer_consensus": consensus,
        "confidence": min(
            0.95,
            (0.20 if answer_valid else 0.0)
            + (0.35 if evidence_covered else 0.0)
            + 0.35 * consensus
            + (0.05 if mentions_evidence else 0.0),
        ),
    }
    return features


def _hotpot_answer_covered_by_visible_evidence(row: dict[str, Any] | None, answer: str) -> bool:
    if row is None:
        return False
    normalized = hotpot_normalize(answer)
    if not normalized:
        return False
    if normalized in {"yes", "no"}:
        return True
    for fact in row.get("supporting_facts", row.get("evidence", [])):
        text = ""
        if isinstance(fact, dict):
            text = f"{fact.get('title', fact.get('entity', ''))} {fact.get('text', '')}"
        else:
            text = str(fact)
        if normalized and normalized in hotpot_normalize(text):
            return True
    return False


def mbpp_allowed_test_features(row: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    code = str(candidate.get("artifact") or candidate.get("raw_tail") or "")
    tests, setup = tests_for_row(row)
    if not tests:
        return {}
    result = execute_python_tests_detailed(code=code, tests=tests, setup=setup, timeout=5)
    detail = dict(result.execution)
    if result.success:
        return {
            "tests_passed": True,
            "public_tests_passed": True,
            "verifier_pass": True,
            "failed_test_count": 0,
            "test_pass_rate": 1.0,
            "confidence": 0.9,
        }
    kind = str(detail.get("kind") or "execution_failure")
    features: dict[str, Any] = {
        "tests_passed": False,
        "public_tests_passed": False,
        "tests_failed": True,
        "failed_test_count": 1,
        "test_pass_rate": 0.0,
        "confidence": 0.1,
        "failure_kind": kind,
        "failed_test": detail.get("failed_test"),
        "test_expression": detail.get("test_expression"),
        "function_name": detail.get("function_name"),
        "input_repr": detail.get("input_repr"),
        "expected": detail.get("expected"),
        "actual": detail.get("actual"),
        "exception_type": detail.get("exception_type"),
        "exception": detail.get("exception"),
    }
    if kind == "assertion_failure":
        features["assertion_failure"] = True
    elif kind == "timeout":
        features["timeout"] = True
    elif kind == "syntax_error":
        features["syntax_error"] = True
        features["compile_error"] = True
    elif kind in {"type_error", "name_error", "import_error", "runtime_exception"}:
        features["runtime_exception"] = True
    return features


def repair_transition(before_correct: bool, after_correct: bool) -> str:
    if before_correct and after_correct:
        return "C->C"
    if before_correct and not after_correct:
        return "C->W"
    if not before_correct and after_correct:
        return "W->C"
    return "W->W"


def _candidate_index_from_id(candidate_id: str, candidate_count: int) -> int:
    try:
        index = int(str(candidate_id).lstrip("c") or 0)
    except ValueError:
        index = 0
    return max(0, min(index, max(0, candidate_count - 1)))


def _is_high_confidence_verifier_pass(domain: str, candidate: dict[str, Any], selection: Any, *, candidate_count: int) -> bool:
    score = candidate.get("score") if isinstance(candidate.get("score"), dict) else {}
    if domain == "gsm8k":
        consensus = float(score.get("cross_candidate_answer_consensus") or 0.0)
        passed = bool(
            candidate_count >= 2
            and
            consensus > 0.5
            and (
                score.get("trace_result_agreement")
                or (
                    score.get("final_value_valid")
                    and score.get("arithmetic_consistency")
                    and score.get("constraint_consistency")
                )
            )
        )
    elif domain == "mmlu_pro":
        consensus = float(score.get("cross_candidate_answer_consensus") or 0.0)
        passed = bool(
            candidate_count >= 2
            and
            score.get("option_valid")
            and not score.get("invalid_option")
            and consensus > 0.5
        )
    elif domain == "hotpotqa":
        consensus = float(score.get("cross_candidate_answer_consensus") or 0.0)
        passed = bool(
            candidate_count >= 2
            and
            score.get("final_value_valid")
            and (
                score.get("evidence_covered")
                or consensus >= 0.5
            )
            and not score.get("missing_evidence")
        )
    else:
        passed = bool(
            score.get("tests_passed")
            or score.get("verifier_pass")
            or score.get("public_tests_passed")
        ) and not bool(
            score.get("syntax_error")
            or score.get("compile_error")
            or score.get("runtime_exception")
            or score.get("timeout")
        )
    feature_confidence = float(score.get("confidence") or 0.0)
    selector_confidence = float(getattr(selection, "confidence", 0.0) or 0.0)
    return passed and max(feature_confidence, selector_confidence) >= 0.2


def adaptive_controller_decision(
    *,
    domain: str,
    selected_candidate: dict[str, Any],
    selection: Any,
    candidate_count: int,
    max_candidates: int,
    min_repair_candidates: int = 4,
) -> AdaptiveDecision:
    """Rule-based controller over SELECT/REQUEST/REPAIR/GENERATE/STOP.

    The controller is intentionally deployable: it only reads verifier features
    and selector metadata, never gold correctness.  Domain priors reflect the
    observed Phase 5 boundary: GSM8K/HumanEval are selection-heavy, while MBPP
    can benefit from targeted repair when a concrete failure signal exists.
    """

    if _is_high_confidence_verifier_pass(
        domain,
        selected_candidate,
        selection,
        candidate_count=candidate_count,
    ):
        return AdaptiveDecision("SELECT", "high_confidence_verifier_pass", should_stop=True)

    if (
        domain == "mbpp"
        and candidate_count >= min_repair_candidates
        and _has_actionable_failure(domain, selected_candidate)
    ):
        return AdaptiveDecision("REPAIR", "mbpp_actionable_failure")

    if candidate_count < max_candidates:
        if _candidate_disagreement_high(domain, selected_candidate, selection):
            return AdaptiveDecision("REQUEST", "insufficient_discrimination_request_more_evidence")
        return AdaptiveDecision("GENERATE", "insufficient_confidence_generate_more")

    if domain == "mbpp" and _has_actionable_failure(domain, selected_candidate):
        return AdaptiveDecision("REPAIR", "mbpp_budget_exhausted_actionable_failure")

    return AdaptiveDecision("SELECT", "budget_exhausted_select_best", should_stop=True)


def _candidate_disagreement_high(domain: str, candidate: dict[str, Any], selection: Any) -> bool:
    score = candidate.get("score") if isinstance(candidate.get("score"), dict) else {}
    if domain in {"gsm8k", "mmlu_pro", "hotpotqa"}:
        return float(score.get("cross_candidate_answer_consensus") or 0.0) < 0.75
    if domain in {"humaneval", "mbpp"}:
        return float(getattr(selection, "confidence", 0.0) or 0.0) < 0.35
    return False


def _has_actionable_failure(domain: str, candidate: dict[str, Any]) -> bool:
    score = candidate.get("score") if isinstance(candidate.get("score"), dict) else {}
    if domain in SELECTION_ONLY_DOMAINS:
        return False
    if domain == "gsm8k":
        return bool(
            score.get("trace_result_disagreement")
            or score.get("constraint_inconsistency")
            or score.get("final_value_valid")
        )
    if score.get("tests_passed") or score.get("verifier_pass") or score.get("public_tests_passed"):
        return False
    return bool(
        score.get("assertion_failure")
        or score.get("runtime_exception")
        or score.get("timeout")
        or score.get("wrong_return_type")
        or score.get("syntax_error")
        or score.get("compile_error")
        or score.get("failed_test_count")
    )


def record_candidate_graph(
    row: dict[str, Any],
    *,
    domain: str,
    group_id: str,
    candidates: list[dict[str, Any]],
    selector_policy: str,
    selected_candidate_id: str | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    store = GraphStore()
    task_id = str(row.get("sample_id") or group_id)
    task = store.add_node(
        task_id=task_id,
        branch_id="main",
        logical_id="task",
        node_type="task",
        content={
            "sample_id": row.get("sample_id"),
            "domain": domain,
            "question": row.get("question"),
        },
        owner="dataset",
        created_by_role="dataset",
    )
    result = materialize_candidate_pool(
        store,
        task_id=task_id,
        branch_id="main",
        group_id=group_id,
        domain=domain,
        candidates=candidates,
        selector_policy=selector_policy if selected_candidate_id is not None else None,
        selected_candidate_id=selected_candidate_id,
        source_node_id=task.node_id,
        metadata=metadata or {},
    )
    if selected_candidate_id is None:
        result = add_verifier_guided_selection(
            store,
            result,
            selection_policy=selector_policy,
            metadata={"source": "candidate_verification_nodes"},
        )
    state = store.snapshot()
    node_type_counts: dict[str, int] = {}
    for node in state.nodes.values():
        node_type_counts[node.type] = node_type_counts.get(node.type, 0) + 1
    return {
        **result.as_dict(),
        "node_count": len(state.nodes),
        "edge_count": len(state.edges),
        "node_type_counts": node_type_counts,
    }


def record_deployable_selection_graph(
    row: dict[str, Any],
    *,
    domain: str,
    group_id: str,
    candidates: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    store = GraphStore()
    task_id = str(row.get("sample_id") or group_id)
    task = store.add_node(
        task_id=task_id,
        branch_id="main",
        logical_id="task",
        node_type="task",
        content={
            "sample_id": row.get("sample_id"),
            "domain": domain,
            "question": row.get("question"),
        },
        owner="dataset",
        created_by_role="dataset",
    )
    pool = materialize_candidate_pool(
        store,
        task_id=task_id,
        branch_id="main",
        group_id=group_id,
        domain=domain,
        candidates=candidates,
        source_node_id=task.node_id,
        metadata={"experiment": "verifier_select"},
    )
    selection = select_candidate_with_deployable_verifier(store, pool)
    result = add_candidate_selection_decision(
        store,
        pool,
        selected_candidate_id=selection.selected_candidate_id,
        selection_policy="deployable_feature_score",
        reason="; ".join(selection.selection_reason),
        metadata=selection.as_dict(),
    )
    state = store.snapshot()
    node_type_counts: dict[str, int] = {}
    for node in state.nodes.values():
        node_type_counts[node.type] = node_type_counts.get(node.type, 0) + 1
    return selection, {
        **result.as_dict(),
        "node_count": len(state.nodes),
        "edge_count": len(state.edges),
        "node_type_counts": node_type_counts,
    }


def select_candidate_for_adaptive_controller(
    *,
    row: dict[str, Any],
    domain: str,
    group_id: str,
    candidates: list[dict[str, Any]],
    policy: str,
    seed: int,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    fragments = build_candidate_fragments(candidates)
    send_all_ids = select_fragment_ids(fragments, policy="selector_send_all", seed=seed)
    send_all_tokens = rendered_fragment_tokens(fragments, send_all_ids)
    if policy == "deployable_full":
        selection, candidate_graph = record_deployable_selection_graph(
            row,
            domain=domain,
            group_id=group_id,
            candidates=candidates,
        )
        return selection, candidate_graph, {
            "policy": policy,
            "selected_candidate_id": selection.selected_candidate_id,
            "selector_input_tokens": send_all_tokens,
            "selector_send_all_equivalent_tokens": send_all_tokens,
            "token_saving_vs_send_all": 0.0,
            "visible_fragment_ids": list(send_all_ids),
        }

    budget_tokens = None
    if policy == "selector_random_same_budget":
        receiver_aware_ids = select_fragment_ids(fragments, policy="selector_receiver_aware", seed=seed)
        budget_tokens = rendered_fragment_tokens(fragments, receiver_aware_ids)
    selection = run_selector_with_fragments(
        candidates,
        policy=policy,
        seed=seed,
        budget_tokens=budget_tokens,
    )
    candidate_graph = record_candidate_graph(
        row,
        domain=domain,
        group_id=group_id,
        candidates=candidates,
        selector_policy=policy,
        selected_candidate_id=selection.selected_candidate_id,
        metadata=selection.as_dict(),
    )
    selector_input_tokens = int(selection.rendered_tokens)
    return selection, candidate_graph, {
        **selection.as_dict(),
        "selector_input_tokens": selector_input_tokens,
        "selector_send_all_equivalent_tokens": send_all_tokens,
        "token_saving_vs_send_all": (
            1.0 - selector_input_tokens / send_all_tokens
            if send_all_tokens else 0.0
        ),
    }


def summarize(records: list[dict[str, Any]], **metadata: Any) -> dict[str, Any]:
    count = len(records)
    correct = sum(int(bool(r.get("correct"))) for r in records)
    initial_correct = sum(int(bool(r.get("initial_correct"))) for r in records)
    wrong_to_correct = sum(int(bool(r.get("wrong_to_correct"))) for r in records)
    coverage_summary: dict[str, float] = {}
    for k in (1, 2, 3, 4, 8, 16):
        key = f"coverage@{k}"
        values = [float(record[key]) for record in records if key in record]
        if values:
            coverage_summary[key] = sum(values) / len(values)
    input_tokens = 0
    output_tokens = 0
    all_candidates: list[dict[str, Any]] = []
    for record in records:
        candidates = []
        if "candidates" in record:
            candidates.extend(record["candidates"])
        if "candidate" in record:
            candidates.append(record["candidate"])
        if "initial" in record:
            candidates.append(record["initial"])
        if "repair" in record:
            candidates.append(record["repair"])
        if "attempts" in record:
            candidates.extend(record["attempts"])
        all_candidates.extend([candidate for candidate in candidates if candidate])
        for candidate in candidates:
            if not candidate:
                continue
            input_tokens += int(candidate.get("input_tokens", 0) or 0)
            output_tokens += int(candidate.get("output_tokens", 0) or 0)
    extra = _phase2_candidate_summary(records, all_candidates)
    if metadata.get("domain") == "mmlu_pro":
        extra["mmlu_coverage_gate"] = _mmlu_coverage_gate(coverage_summary)
    return {
        **metadata,
        "count": count,
        "correct_count": correct,
        "accuracy_or_pass_at_n": correct / count if count else 0.0,
        "initial_correct_count": initial_correct,
        "wrong_to_correct_count": wrong_to_correct,
        **coverage_summary,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_model_tokens": input_tokens + output_tokens,
        **extra,
        "records": records,
    }


def _phase2_candidate_summary(records: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    candidate_counts = [
        int(record.get("candidate_count", 0) or len(record.get("candidates", []) or record.get("attempts", []) or []))
        for record in records
    ]
    if candidate_counts:
        extra["candidate_count_mean"] = sum(candidate_counts) / len(candidate_counts)
        extra["candidate_count_min"] = min(candidate_counts)
        extra["candidate_count_max"] = max(candidate_counts)
    if any(candidate.get("option_hypotheses") for candidate in candidates):
        parseable = sum(int(bool(candidate.get("option_hypotheses"))) for candidate in candidates)
        modes: dict[str, int] = {}
        pairwise_sources: dict[str, int] = {}
        calibration_parse_fail = 0
        margins = []
        for candidate in candidates:
            mode = str(candidate.get("optionwise_mode") or "")
            if not mode:
                role = str(candidate.get("generation_role") or "")
                mode = role.split(":", 1)[1] if ":" in role else ""
            if mode:
                modes[mode] = modes.get(mode, 0) + 1
            runtime = candidate.get("runtime_selection")
            if isinstance(runtime, dict):
                source = str(runtime.get("selection_source") or "")
                if source:
                    pairwise_sources[source] = pairwise_sources.get(source, 0) + 1
                margins.append(float(runtime.get("score_margin", 0.0) or 0.0))
            calibration = candidate.get("score_calibration")
            if isinstance(calibration, dict) and calibration.get("parse_fail"):
                calibration_parse_fail += 1
        extra.update({
            "option_hypothesis_parseable_candidates": parseable,
            "avg_options_per_candidate": (
                sum(len(candidate.get("option_hypotheses") or []) for candidate in candidates) / len(candidates)
                if candidates else 0.0
            ),
            "mmlu_optionwise_mode_counts": modes,
            "runtime_selection_source_counts": pairwise_sources,
            "pairwise_rate": (
                pairwise_sources.get("pairwise_top2_critic", 0) / max(1, sum(pairwise_sources.values()))
                if pairwise_sources else 0.0
            ),
            "avg_option_score_margin": sum(margins) / len(margins) if margins else 0.0,
            "score_calibration_parse_fail": calibration_parse_fail,
        })
    if any(candidate.get("bridge_entity") for candidate in candidates):
        bridges_by_record = []
        chains_by_record = []
        for record in records:
            record_candidates = record.get("candidates") or record.get("attempts") or []
            bridges = {
                normalize_bridge(candidate.get("bridge_entity"))
                for candidate in record_candidates
                if candidate.get("bridge_entity")
            }
            chains = {
                str(candidate.get("hotpot_chain_signature") or "")
                for candidate in record_candidates
                if candidate.get("hotpot_chain_signature")
            }
            bridges_by_record.append(len([item for item in bridges if item]))
            chains_by_record.append(len([item for item in chains if item]))
        extra.update({
            "hotpot_candidate_mode": HOTPOT_CANDIDATE_MODE,
            "avg_unique_bridges": sum(bridges_by_record) / len(bridges_by_record) if bridges_by_record else 0.0,
            "avg_unique_evidence_chains": sum(chains_by_record) / len(chains_by_record) if chains_by_record else 0.0,
            "redundant_candidate_count": sum(int(bool(candidate.get("redundant_candidate"))) for candidate in candidates),
            "same_bridge_different_chain_count": sum(
                int(candidate.get("redundancy_reason") == "same_bridge_different_evidence_chain")
                for candidate in candidates
            ),
        })
        bridge_pool_audit = _hotpot_bridge_pool_audit(records)
        if bridge_pool_audit:
            extra.update(bridge_pool_audit)
    return extra


def _hotpot_bridge_pool_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    audited = []
    for record in records:
        candidates = list(record.get("candidates") or [])
        if not candidates:
            continue
        first_pool = candidates[0].get("bridge_candidate_pool")
        if not isinstance(first_pool, list):
            continue
        gold_answer_correct = any(bool(candidate.get("correct")) for candidate in candidates)
        audited.append({
            "pool": first_pool,
            "raw_size": int(candidates[0].get("raw_bridge_pool_size", 0) or 0),
            "answer_correct": gold_answer_correct,
        })
    if not audited:
        return {}
    return {
        "avg_raw_bridge_pool_size": sum(item["raw_size"] for item in audited) / len(audited),
        "avg_ranked_bridge_pool_size": sum(len(item["pool"]) for item in audited) / len(audited),
        "bridge_pool_source_counts": _bridge_pool_source_counts(audited),
        "bridge_pool_answer_correct_rate": sum(int(item["answer_correct"]) for item in audited) / len(audited),
    }


def _bridge_pool_source_counts(audited: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in audited:
        for record in item["pool"]:
            if isinstance(record, dict):
                source = str(record.get("source_type") or "unknown")
                counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def _mmlu_coverage_gate(coverage_summary: dict[str, float]) -> dict[str, Any]:
    coverage_at_2 = float(coverage_summary.get("coverage@2", 0.0) or 0.0)
    coverage_at_3 = float(coverage_summary.get("coverage@3", 0.0) or 0.0)
    return {
        "minimum_keep_line": "coverage@2 >= 0.60",
        "next_target_coverage_at_2": "coverage@2 >= 0.65",
        "next_target_coverage_at_3": "coverage@3 >= 0.70",
        "coverage@2": coverage_at_2,
        "coverage@3": coverage_at_3,
        "minimum_pass": coverage_at_2 >= 0.60,
        "target_at_2_pass": coverage_at_2 >= 0.65,
        "target_at_3_pass": coverage_at_3 >= 0.70,
        "phase2_mmlu_gate_pass": coverage_at_2 >= 0.60 and (coverage_at_2 >= 0.65 or coverage_at_3 >= 0.70),
    }


def _hotpot_fixed_k_gate(report: dict[str, Any], *, n: int) -> dict[str, Any]:
    coverage_at_4 = float(report.get("coverage@4", 0.0) or 0.0)
    candidate_count_mean = float(report.get("candidate_count_mean", 0.0) or 0.0)
    avg_unique_bridges = float(report.get("avg_unique_bridges", 0.0) or 0.0)
    avg_unique_chains = float(report.get("avg_unique_evidence_chains", 0.0) or 0.0)
    return {
        "fixed_k": n,
        "candidate_count_target": float(n),
        "actual_candidate_count_mean": candidate_count_mean,
        "avg_unique_bridges": avg_unique_bridges,
        "avg_unique_evidence_chains": avg_unique_chains,
        "coverage@4": coverage_at_4,
        "candidate_count_pass": candidate_count_mean >= float(n) - 1e-9,
        "unique_bridge_pass": avg_unique_bridges > 2.0,
        "unique_chain_pass": avg_unique_chains > 2.0,
        "coverage_recovery_pass": coverage_at_4 >= 0.54,
        "phase2_hotpot_fixed_k_gate_pass": (
            candidate_count_mean >= float(n) - 1e-9
            and avg_unique_bridges > 2.0
            and avg_unique_chains > 2.0
            and coverage_at_4 >= 0.54
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=DOMAINS)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--experiment",
        required=True,
        choices=(
            "best_of_n",
            "fixed_candidate_pool",
            "oracle_select",
            "verifier_select",
            "select_repair",
            "adaptive_controller",
            "adaptive_controller_selector_comm",
            "oracle_repair",
            "iterative_repair",
        ),
    )
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--mmlu-optionwise-mode",
        choices=("independent", "permutation", "second_pass"),
        default="independent",
        help="MMLU-Pro Phase 2 option hypothesis variant.",
    )
    parser.add_argument(
        "--hotpot-candidate-mode",
        choices=("default", "preserve_first"),
        default="default",
        help="HotpotQA Phase 2 candidate generator variant.",
    )
    parser.add_argument("--current-baseline", type=float, default=0.0)
    parser.add_argument("--verifier-baseline", type=float, default=0.0)
    parser.add_argument("--min-repair-candidates", type=int, default=4)
    parser.add_argument(
        "--selector-communication-policy",
        choices=SELECTOR_COMMUNICATION_POLICIES,
        default="deployable_full",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    global MMLU_OPTIONWISE_MODE, HOTPOT_CANDIDATE_MODE
    MMLU_OPTIONWISE_MODE = args.mmlu_optionwise_mode
    HOTPOT_CANDIDATE_MODE = args.hotpot_candidate_mode

    rows = read(Path(args.data_path), args.domain, args.limit)
    model = SamplingModel(args.model_path, max_new_tokens=args.max_new_tokens)
    if args.experiment == "best_of_n":
        report = run_best_of_n(
            model,
            rows,
            domain=args.domain,
            n=args.n,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    elif args.experiment == "fixed_candidate_pool":
        report = run_fixed_candidate_pool(
            model,
            rows,
            domain=args.domain,
            n=args.n,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    elif args.experiment == "oracle_select":
        report = run_oracle_select(
            model,
            rows,
            domain=args.domain,
            n=args.n,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    elif args.experiment == "verifier_select":
        report = run_verifier_select(
            model,
            rows,
            domain=args.domain,
            n=args.n,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            current_baseline=args.current_baseline,
        )
    elif args.experiment in {"select_repair", "adaptive_controller", "adaptive_controller_selector_comm"}:
        report = run_select_repair(
            model,
            rows,
            domain=args.domain,
            n=args.n,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            verifier_baseline=args.verifier_baseline,
            experiment_name=args.experiment,
            min_repair_candidates=args.min_repair_candidates,
            selector_communication_policy=args.selector_communication_policy,
        )
    elif args.experiment == "oracle_repair":
        report = run_oracle_repair(
            model,
            rows,
            domain=args.domain,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    else:
        report = run_iterative_repair(
            model,
            rows,
            domain=args.domain,
            max_rounds=args.max_rounds,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    report.update({
        "data_path": args.data_path,
        "model_path": args.model_path,
        "mmlu_optionwise_mode": args.mmlu_optionwise_mode,
        "hotpot_candidate_mode": args.hotpot_candidate_mode,
        "limit": args.limit,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {key: value for key, value in report.items() if key != "records"}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
