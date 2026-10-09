"""Audit whether MMLU-Pro/HotpotQA critics receive enough discriminative input.

The audit keeps candidate generation fixed and varies only the critic input:

* critic_compact: candidate answers plus tiny snippets.
* critic_full_available: task-visible summaries plus candidate rationales.
* critic_full_task: full task/evidence plus full candidate outputs.
* structured_selected_context: domain-specific discriminative packet.
* structured_loop: structured packet plus a receiver request/recheck protocol.

Gold labels are used only after selection for offline scoring.
"""

from __future__ import annotations

import argparse
import inspect
import json
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
    extract_choice,
    hotpot_f1,
    hotpot_normalize,
    read,
    render_native_task,
    score_prediction,
)


DOMAINS = ("mmlu_pro", "hotpotqa")
BASE_SETTINGS = (
    "critic_compact",
    "critic_full_available",
    "critic_full_task",
    "structured_selected_context",
    "structured_loop",
)
HOTPOT_EXTRA_SETTINGS = (
    "per_candidate_evidence",
    "union_evidence_pool",
    "oracle_evidence_chain",
)


def _settings_for_domain(domain: str) -> tuple[str, ...]:
    if domain == "hotpotqa":
        return BASE_SETTINGS + HOTPOT_EXTRA_SETTINGS
    return BASE_SETTINGS


@dataclass
class Candidate:
    candidate_id: str
    text: str
    answer: str
    score: dict[str, Any]
    input_tokens: int
    output_tokens: int
    seed: int

    @property
    def model_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "answer": self.answer,
            "correct": bool(self.score.get("correct")),
            "score": self.score,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_tokens": self.model_tokens,
            "seed": self.seed,
            "raw_tail": self.text[-1500:],
        }


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
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
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
        max_new_tokens: int | None = None,
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
            "max_new_tokens": int(max_new_tokens or self.max_new_tokens),
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
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip(), {
            "input_tokens": int(input_ids.shape[-1]),
            "output_tokens": int(new_ids.shape[-1]),
            "latency_sec": elapsed,
        }


def generation_prompt(row: dict[str, Any], domain: str) -> str:
    if domain == "mmlu_pro":
        return (
            "Answer the multiple-choice question. Explain briefly, then end with "
            "'Final answer: <letter>'.\n\n"
            f"{row['question']}"
        )
    return (
        "Answer the multi-hop question using the evidence. Explain the bridge reasoning "
        "briefly, then end with 'Final answer: <answer>'.\n\n"
        f"{render_native_task(row)}"
    )


def extract_answer(domain: str, text: str) -> str:
    if domain == "mmlu_pro":
        return extract_choice(text)
    labelled = re.findall(r"(?:final\s+answer|answer)\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE)
    if labelled:
        return labelled[-1].strip().splitlines()[0].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def score_answer(row: dict[str, Any], domain: str, answer: str) -> dict[str, Any]:
    if domain == "hotpotqa":
        em = float(hotpot_normalize(answer) == hotpot_normalize(row.get("gold_answer", "")))
        return {
            "correct": bool(em),
            "exact_match": em,
            "f1": hotpot_f1(answer, row.get("gold_answer", "")),
            "predicted_answer": answer,
        }
    return score_prediction(row, answer)


def generate_candidates(
    model: Any,
    row: dict[str, Any],
    *,
    domain: str,
    n: int,
    seed: int,
    temperature: float,
    top_p: float,
) -> list[Candidate]:
    prompt = generation_prompt(row, domain)
    candidates = []
    for index in range(n):
        text, metrics = model.generate(
            prompt,
            seed=seed + index,
            temperature=temperature,
            top_p=top_p,
            do_sample=index > 0,
        )
        answer = extract_answer(domain, text)
        candidates.append(
            Candidate(
                candidate_id=f"c{index}",
                text=text,
                answer=answer,
                score=score_answer(row, domain, answer),
                input_tokens=int(metrics["input_tokens"]),
                output_tokens=int(metrics["output_tokens"]),
                seed=seed + index,
            )
        )
    return candidates


def render_critic_prompt(row: dict[str, Any], domain: str, candidates: list[Candidate], setting: str) -> str:
    if setting == "critic_compact":
        task_context = _compact_task_context(row, domain)
        candidate_block = "\n".join(_candidate_view(candidate, setting) for candidate in candidates)
    elif setting == "critic_full_available":
        task_context = _full_available_context(row, domain)
        candidate_block = "\n".join(_candidate_view(candidate, setting) for candidate in candidates)
    elif setting == "critic_full_task":
        task_context = render_native_task(row)
        candidate_block = "\n".join(_candidate_view(candidate, setting) for candidate in candidates)
    elif setting == "structured_selected_context":
        task_context, candidate_block = _structured_context(row, domain, candidates, loop=False)
    elif setting == "structured_loop":
        task_context, candidate_block = _structured_context(row, domain, candidates, loop=True)
    elif setting == "per_candidate_evidence":
        if domain != "hotpotqa":
            raise ValueError(f"{setting} is only supported for hotpotqa")
        task_context, candidate_block = _hotpotqa_per_candidate_evidence_packet(row, candidates)
    elif setting == "union_evidence_pool":
        if domain != "hotpotqa":
            raise ValueError(f"{setting} is only supported for hotpotqa")
        task_context, candidate_block = _hotpotqa_union_evidence_packet(row, candidates, oracle=False)
    elif setting == "oracle_evidence_chain":
        if domain != "hotpotqa":
            raise ValueError(f"{setting} is only supported for hotpotqa")
        task_context, candidate_block = _hotpotqa_union_evidence_packet(row, candidates, oracle=True)
    else:
        raise ValueError(f"unknown critic setting: {setting}")
    valid_ids = ", ".join(candidate.candidate_id for candidate in candidates)
    return (
        "You are a strict candidate selector. Select the single best candidate using only "
        "the provided information.\n"
        f"Valid candidate IDs: {valid_ids}\n"
        "Output exactly one candidate ID and nothing else. Example: c0\n\n"
        f"Task context:\n{task_context}\n\n"
        f"Candidates:\n{candidate_block}\n\n"
        "Selected candidate ID:"
    )


def _candidate_view(candidate: Candidate, setting: str) -> str:
    if setting == "critic_compact":
        snippet = " ".join(candidate.text.split())[:180]
    elif setting == "critic_full_available":
        snippet = " ".join(candidate.text.split())[:650]
    else:
        snippet = candidate.text
    return f"{candidate.candidate_id}: answer={candidate.answer!r}\nrationale_or_output={snippet}\n"


def _compact_task_context(row: dict[str, Any], domain: str) -> str:
    if domain == "mmlu_pro":
        return str(row.get("question", "")).split("\n\n")[0][:500]
    return f"Question: {row.get('question', '')}\nEntities: {', '.join(row.get('entities', [])[:8])}"


def _full_available_context(row: dict[str, Any], domain: str) -> str:
    if domain == "mmlu_pro":
        return str(row.get("question", ""))
    lines = [f"Question: {row.get('question', '')}", "Evidence summaries:"]
    for fact in row.get("supporting_facts", [])[:8]:
        if isinstance(fact, dict):
            title = fact.get("title", fact.get("entity", ""))
            text = str(fact.get("text", ""))[:350]
            lines.append(f"- {title}: {text}")
        else:
            lines.append(f"- {str(fact)[:350]}")
    return "\n".join(lines)


def _structured_context(
    row: dict[str, Any],
    domain: str,
    candidates: list[Candidate],
    *,
    loop: bool,
) -> tuple[str, str]:
    if domain == "hotpotqa":
        return _hotpotqa_evidence_packet(row, candidates, loop=loop)
    if domain == "mmlu_pro":
        return _mmlu_option_packet(row, candidates, loop=loop)
    raise ValueError(f"unsupported structured context domain: {domain}")


def _hotpotqa_evidence_packet(row: dict[str, Any], candidates: list[Candidate], *, loop: bool) -> tuple[str, str]:
    lines = [
        f"Question: {row.get('question', '')}",
        "Critic task:",
        "1. Check which candidate has a complete bridge chain.",
        "2. Check whether hop1 and hop2 evidence support the answer.",
        "3. Select a candidate only if answer-evidence alignment is strongest.",
    ]
    if loop:
        lines.extend([
            "4. If a hop is missing, treat candidates with complete hop evidence as better.",
            "5. Prefer selected evidence over unrelated full passages.",
        ])
    candidate_blocks = []
    for candidate in candidates:
        packet = _hotpot_candidate_packet(row, candidate, fact_limit=4 if loop else 2)
        candidate_blocks.append(packet)
    return "\n".join(lines), "\n\n".join(candidate_blocks)


def _hotpotqa_per_candidate_evidence_packet(
    row: dict[str, Any],
    candidates: list[Candidate],
) -> tuple[str, str]:
    lines = [
        f"Question: {row.get('question', '')}",
        "Evidence mode: per_candidate_evidence",
        "Critic task:",
        "1. Judge each candidate only from its own ranked evidence packet.",
        "2. Prefer candidates with complete two-hop support and answer alignment.",
        "3. Do not use evidence from one candidate to rescue another candidate.",
    ]
    blocks = [
        _hotpot_candidate_packet(row, candidate, fact_limit=4)
        for candidate in candidates
    ]
    return "\n".join(lines), "\n\n".join(blocks)


def _hotpotqa_union_evidence_packet(
    row: dict[str, Any],
    candidates: list[Candidate],
    *,
    oracle: bool,
) -> tuple[str, str]:
    facts = _oracle_hotpot_facts(row, limit=4) if oracle else _union_ranked_hotpot_facts(row, candidates, limit=4)
    evidence_text = " ".join(_fact_text(fact) for fact in facts)
    evidence_terms = set(_content_words(evidence_text))
    lines = [
        f"Question: {row.get('question', '')}",
        f"Evidence mode: {'oracle_evidence_chain' if oracle else 'union_evidence_pool'}",
        "Shared evidence graph:",
    ]
    for index, fact in enumerate(facts, start=1):
        lines.append(f"E{index}={_fact_label(fact)}: {_fact_text(fact)[:520]}")
    lines.extend([
        "Critic task:",
        "1. Use the shared evidence graph to identify the shortest consistent two-hop chain.",
        "2. Compare every candidate answer against the same shared evidence.",
        "3. Prefer candidates whose answer is directly supported by the shared chain.",
        "4. Reject candidates whose rationale conflicts with the shared evidence.",
    ])
    candidate_blocks = []
    for candidate in candidates:
        answer_terms = set(_content_words(candidate.answer))
        alignment = sorted(answer_terms & evidence_terms)
        bridge = _guess_bridge_entity(row, candidate)
        rationale = " ".join(candidate.text.split())[:520]
        candidate_blocks.append(
            "\n".join([
                f"{candidate.candidate_id}: answer={candidate.answer!r}",
                f"bridge_entity={bridge or 'unknown'}",
                f"shared_answer_alignment={', '.join(alignment[:8]) if alignment else 'none'}",
                f"rationale_summary={rationale}",
            ])
        )
    return "\n".join(lines), "\n\n".join(candidate_blocks)


def _hotpot_candidate_packet(row: dict[str, Any], candidate: Candidate, *, fact_limit: int) -> str:
    facts = _rank_hotpot_facts(row, candidate, limit=fact_limit)
    bridge = _guess_bridge_entity(row, candidate)
    answer_terms = set(_content_words(candidate.answer))
    evidence_text = " ".join(_fact_text(fact) for fact in facts)
    evidence_terms = set(_content_words(evidence_text))
    answer_alignment = sorted(answer_terms & evidence_terms)
    missing_hop = "none" if len(facts) >= 2 and answer_alignment else "missing_or_weak"
    lines = [
        f"{candidate.candidate_id}: answer={candidate.answer!r}",
        f"bridge_entity={bridge or 'unknown'}",
        f"answer_evidence_alignment={', '.join(answer_alignment[:8]) if answer_alignment else 'none'}",
        f"missing_hop_signal={missing_hop}",
    ]
    for index, fact in enumerate(facts[:fact_limit], start=1):
        lines.append(f"hop{index}_supporting_fact={_fact_label(fact)}: {_fact_text(fact)[:420]}")
    rationale = " ".join(candidate.text.split())[:360]
    lines.append(f"candidate_rationale_summary={rationale}")
    return "\n".join(lines)


def _rank_hotpot_facts(row: dict[str, Any], candidate: Candidate, *, limit: int) -> list[Any]:
    facts = list(row.get("supporting_facts", []))
    query = " ".join([str(row.get("question", "")), candidate.answer, candidate.text])
    query_terms = set(_content_words(query))
    ranked = sorted(
        facts,
        key=lambda fact: (
            len(query_terms & set(_content_words(_fact_text(fact)))),
            len(set(_content_words(candidate.answer)) & set(_content_words(_fact_text(fact)))),
        ),
        reverse=True,
    )
    return ranked[:limit]


def _union_ranked_hotpot_facts(row: dict[str, Any], candidates: list[Candidate], *, limit: int) -> list[Any]:
    facts = list(row.get("supporting_facts", []))
    if not facts:
        return []
    question_terms = set(_content_words(row.get("question", "")))
    scored = []
    for index, fact in enumerate(facts):
        fact_terms = set(_content_words(_fact_text(fact)))
        candidate_hits = 0
        answer_hits = 0
        rationale_hits = 0
        for candidate in candidates:
            answer_terms = set(_content_words(candidate.answer))
            rationale_terms = set(_content_words(candidate.text))
            answer_overlap = len(answer_terms & fact_terms)
            rationale_overlap = len(rationale_terms & fact_terms)
            if answer_overlap or rationale_overlap:
                candidate_hits += 1
            answer_hits += answer_overlap
            rationale_hits += rationale_overlap
        question_hits = len(question_terms & fact_terms)
        scored.append((
            4 * candidate_hits + 3 * answer_hits + 2 * question_hits + rationale_hits,
            candidate_hits,
            answer_hits,
            question_hits,
            -index,
            fact,
        ))
    scored.sort(reverse=True, key=lambda item: item[:-1])
    return [item[-1] for item in scored[:limit]]


def _oracle_hotpot_facts(row: dict[str, Any], *, limit: int) -> list[Any]:
    gold_keys = _hotpot_gold_fact_keys(row)
    facts = list(row.get("supporting_facts", []))
    selected = [fact for fact in facts if _fact_key(fact) in gold_keys]
    if len(selected) >= limit:
        return selected[:limit]
    seen = {_fact_key(fact) for fact in selected}
    for fact in facts:
        key = _fact_key(fact)
        if key not in seen:
            selected.append(fact)
            seen.add(key)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _guess_bridge_entity(row: dict[str, Any], candidate: Candidate) -> str:
    text = f"{candidate.answer} {candidate.text}".lower()
    for entity in row.get("entities", []):
        entity_text = str(entity)
        if entity_text and entity_text.lower() in text:
            return entity_text
    return str(row.get("entities", [""])[0] if row.get("entities") else "")


def _fact_label(fact: Any) -> str:
    if isinstance(fact, dict):
        return str(fact.get("title") or fact.get("entity") or "evidence")
    return "evidence"


def _fact_text(fact: Any) -> str:
    if isinstance(fact, dict):
        return str(fact.get("text") or fact)
    return str(fact)


def _mmlu_option_packet(row: dict[str, Any], candidates: list[Candidate], *, loop: bool) -> tuple[str, str]:
    choices = row.get("choices", [])
    answer_counts: dict[str, int] = {}
    for candidate in candidates:
        answer_counts[candidate.answer] = answer_counts.get(candidate.answer, 0) + 1
    disagreement = ", ".join(f"{label}:{count}" for label, count in sorted(answer_counts.items()))
    lines = [
        str(row.get("question", "")),
        "",
        f"candidate_disagreement={disagreement}",
        "Critic task:",
        "Step 1: eliminate impossible options using the full question and all options.",
        "Step 2: compare candidate rationales against the remaining options.",
        "Step 3: select the candidate whose answer is best supported.",
    ]
    if loop:
        lines.extend([
            "If confidence is low, prefer the candidate with the clearest option-specific support.",
            "Treat unsupported rationale or contradicted option support as weaker.",
        ])
    option_lines = ["Option support/contradiction summary:"]
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        label = str(choice.get("label", "")).strip().upper()
        text = str(choice.get("text", ""))
        supporters = [
            candidate.candidate_id
            for candidate in candidates
            if candidate.answer.upper() == label
        ]
        mentions = [
            candidate.candidate_id
            for candidate in candidates
            if label and re.search(rf"\b{re.escape(label)}\b", candidate.text, flags=re.IGNORECASE)
        ]
        option_lines.append(
            f"{label}. {text} | answer_supporters={supporters or 'none'} | rationale_mentions={mentions or 'none'}"
        )
    candidate_blocks = []
    for candidate in candidates:
        summary = " ".join(candidate.text.split())[:700 if loop else 420]
        candidate_blocks.append(
            "\n".join([
                f"{candidate.candidate_id}: answer={candidate.answer!r}",
                f"rationale_summary={summary}",
            ])
        )
    return "\n".join(lines + [""] + option_lines), "\n\n".join(candidate_blocks)


def _content_words(text: Any) -> list[str]:
    stop = {
        "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with",
        "is", "was", "were", "are", "by", "as", "at", "from", "that", "this",
        "which", "who", "what", "when", "where", "how", "it", "its",
    }
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 2 and token not in stop
    ]


def parse_selected_candidate(text: str, candidate_count: int) -> tuple[str, bool]:
    valid_ids = {f"c{index}" for index in range(candidate_count)}
    normalized = text.strip()
    try:
        data = json.loads(_json_object(text))
        candidate_id = str(data.get("selected_candidate_id") or data.get("candidate_id") or "")
    except Exception:
        leading = re.match(r"^\s*`{0,3}\s*c(\d+)\b", text, flags=re.IGNORECASE)
        if leading:
            candidate_id = f"c{leading.group(1)}"
        else:
            labelled = re.search(
                r"(?:selected_candidate_id|candidate_id|selected|answer|candidate)\s*[:=#-]?\s*[\"']?\s*c?(\d+)",
                text,
                flags=re.IGNORECASE,
            )
            if labelled:
                candidate_id = f"c{labelled.group(1)}"
            else:
                bare_number = re.fullmatch(r"(\d+)", normalized)
                candidate_id = f"c{bare_number.group(1)}" if bare_number else ""
    candidate_id = candidate_id.strip().strip("\"'").lower()
    if candidate_id.isdigit():
        candidate_id = f"c{candidate_id}"
    valid = candidate_id in valid_ids
    return (candidate_id if valid else "c0"), valid


def _json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def run_audit(
    *,
    model: Any,
    rows: list[dict[str, Any]],
    domain: str,
    n: int,
    seed: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    records = []
    settings = _settings_for_domain(domain)
    setting_totals = {
        setting: {
            "correct": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "parse_failures": 0,
            "oracle_missed": 0,
        }
        for setting in settings
    }
    for row_index, row in enumerate(rows, start=1):
        candidates = generate_candidates(
            model,
            row,
            domain=domain,
            n=n,
            seed=seed + row_index * 1009,
            temperature=temperature,
            top_p=top_p,
        )
        oracle_has_correct = any(candidate.score.get("correct") for candidate in candidates)
        record = {
            "sample_id": row.get("sample_id"),
            "oracle_has_correct": oracle_has_correct,
            "candidates": [candidate.as_dict() for candidate in candidates],
            "settings": {},
        }
        if domain == "hotpotqa":
            record["evidence_selector_audit"] = _hotpot_evidence_selector_audit(row, candidates)
        for setting_index, setting in enumerate(settings):
            prompt = render_critic_prompt(row, domain, candidates, setting)
            text, metrics = model.generate(
                prompt,
                seed=seed + row_index * 1009 + 10000 + setting_index,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                max_new_tokens=8,
            )
            selected_id, parse_valid = parse_selected_candidate(text, len(candidates))
            selected = candidates[int(selected_id[1:])]
            correct = bool(selected.score.get("correct"))
            setting_totals[setting]["correct"] += int(correct)
            setting_totals[setting]["input_tokens"] += int(metrics["input_tokens"])
            setting_totals[setting]["output_tokens"] += int(metrics["output_tokens"])
            setting_totals[setting]["parse_failures"] += int(not parse_valid)
            setting_totals[setting]["oracle_missed"] += int(oracle_has_correct and not correct)
            record["settings"][setting] = {
                "selected_candidate_id": selected_id,
                "selected_correct": correct,
                "parse_valid": parse_valid,
                "critic_input_tokens": int(metrics["input_tokens"]),
                "critic_output_tokens": int(metrics["output_tokens"]),
                "critic_raw_tail": text[-800:],
            }
        records.append(record)
        print(
            f"{row_index}/{len(rows)} {row.get('sample_id')} "
            + " ".join(
                f"{setting}={record['settings'][setting]['selected_candidate_id']}:{int(record['settings'][setting]['selected_correct'])}"
                for setting in settings
            ),
            flush=True,
        )
    count = len(records)
    policies = {}
    for setting, totals in setting_totals.items():
        policies[setting] = {
            "accuracy": totals["correct"] / count if count else 0.0,
            "correct_count": totals["correct"],
            "critic_input_tokens": totals["input_tokens"],
            "critic_output_tokens": totals["output_tokens"],
            "critic_model_tokens": totals["input_tokens"] + totals["output_tokens"],
            "parse_failure_count": totals["parse_failures"],
            "oracle_missed_count": totals["oracle_missed"],
        }
    report = {
        "domain": domain,
        "experiment": "critic_input_sufficiency_audit",
        "candidate_count": n,
        "count": count,
        "oracle_coverage": sum(int(record["oracle_has_correct"]) for record in records) / count if count else 0.0,
        "policies": policies,
        "records": records,
    }
    if domain == "hotpotqa":
        report["evidence_selector_audit"] = _summarize_hotpot_evidence_audit(records)
    return report


def _hotpot_evidence_selector_audit(row: dict[str, Any], candidates: list[Candidate]) -> dict[str, Any]:
    per_candidate = []
    for candidate in candidates:
        item = {"candidate_id": candidate.candidate_id}
        for k in (2, 4):
            selected = _rank_hotpot_facts(row, candidate, limit=k)
            metrics = _hotpot_evidence_metrics(row, candidate, selected)
            item[f"recall@{k}"] = metrics["recall"]
            item[f"gold_overlap@{k}"] = metrics["gold_overlap"]
            item[f"bridge_coverage@{k}"] = metrics["bridge_coverage"]
            item[f"answer_support@{k}"] = metrics["answer_support"]
            item[f"selected_fact_keys@{k}"] = metrics["selected_fact_keys"]
        per_candidate.append(item)
    summary = {}
    for k in (2, 4):
        recalls = [float(item[f"recall@{k}"]) for item in per_candidate]
        bridge = [float(item[f"bridge_coverage@{k}"]) for item in per_candidate]
        answer = [float(item[f"answer_support@{k}"]) for item in per_candidate]
        summary[f"max_recall@{k}"] = max(recalls) if recalls else 0.0
        summary[f"mean_recall@{k}"] = sum(recalls) / len(recalls) if recalls else 0.0
        summary[f"any_bridge_coverage@{k}"] = bool(any(bridge))
        summary[f"mean_bridge_coverage@{k}"] = sum(bridge) / len(bridge) if bridge else 0.0
        summary[f"any_answer_support@{k}"] = bool(any(answer))
        summary[f"mean_answer_support@{k}"] = sum(answer) / len(answer) if answer else 0.0
    return {"candidates": per_candidate, "summary": summary}


def _hotpot_evidence_metrics(row: dict[str, Any], candidate: Candidate, selected_facts: list[Any]) -> dict[str, Any]:
    gold_keys = _hotpot_gold_fact_keys(row)
    selected_keys = {_fact_key(fact) for fact in selected_facts}
    gold_overlap = len(gold_keys & selected_keys)
    recall = gold_overlap / len(gold_keys) if gold_keys else 0.0
    gold_titles = {title for title, _sent_id in gold_keys if title}
    selected_titles = {title for title, _sent_id in selected_keys if title}
    bridge_coverage = bool(gold_titles and gold_titles.issubset(selected_titles))
    evidence_text = " ".join(_fact_text(fact) for fact in selected_facts)
    answer_terms = set(_content_words(candidate.answer))
    evidence_terms = set(_content_words(evidence_text))
    answer_support = bool(answer_terms and answer_terms <= evidence_terms)
    return {
        "recall": recall,
        "gold_overlap": gold_overlap,
        "bridge_coverage": bridge_coverage,
        "answer_support": answer_support,
        "selected_fact_keys": [list(key) for key in sorted(selected_keys)],
    }


def _hotpot_gold_fact_keys(row: dict[str, Any]) -> set[tuple[str, int]]:
    keys = set()
    for link in row.get("evidence_links", []):
        if not isinstance(link, dict):
            continue
        title = str(link.get("entity", link.get("title", "")))
        try:
            sent_id = int(link.get("sent_id", -1))
        except (TypeError, ValueError):
            sent_id = -1
        keys.add((title, sent_id))
    return keys


def _fact_key(fact: Any) -> tuple[str, int]:
    if not isinstance(fact, dict):
        return ("", -1)
    title = str(fact.get("title") or fact.get("entity") or "")
    try:
        sent_id = int(fact.get("sent_id", -1))
    except (TypeError, ValueError):
        sent_id = -1
    return (title, sent_id)


def _summarize_hotpot_evidence_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for k in (2, 4):
        max_recalls = []
        mean_recalls = []
        any_bridge = []
        mean_bridge = []
        any_answer = []
        mean_answer = []
        for record in records:
            item = record.get("evidence_selector_audit", {}).get("summary", {})
            max_recalls.append(float(item.get(f"max_recall@{k}", 0.0)))
            mean_recalls.append(float(item.get(f"mean_recall@{k}", 0.0)))
            any_bridge.append(float(bool(item.get(f"any_bridge_coverage@{k}", False))))
            mean_bridge.append(float(item.get(f"mean_bridge_coverage@{k}", 0.0)))
            any_answer.append(float(bool(item.get(f"any_answer_support@{k}", False))))
            mean_answer.append(float(item.get(f"mean_answer_support@{k}", 0.0)))
        denom = max(1, len(records))
        summary[f"max_evidence_recall@{k}"] = sum(max_recalls) / denom
        summary[f"mean_evidence_recall@{k}"] = sum(mean_recalls) / denom
        summary[f"bridge_coverage_any_candidate@{k}"] = sum(any_bridge) / denom
        summary[f"bridge_coverage_mean_candidate@{k}"] = sum(mean_bridge) / denom
        summary[f"answer_support_any_candidate@{k}"] = sum(any_answer) / denom
        summary[f"answer_support_mean_candidate@{k}"] = sum(mean_answer) / denom
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=DOMAINS)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read(Path(args.data_path), args.domain, args.limit)
    model = SamplingModel(args.model_path, max_new_tokens=args.max_new_tokens)
    report = run_audit(
        model=model,
        rows=rows,
        domain=args.domain,
        n=args.n,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
