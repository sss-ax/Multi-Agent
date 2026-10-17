"""HotpotQA bridge/evidence-chain listwise selection ablation.

The experiment keeps the delivered bridge candidates fixed and varies only the
selector representation:

* bridge_name_only
* bridge_plus_hop1
* bridge_plus_chain
* bridge_chain_answer
* full_chain_answer
* structured_chain_v1
* semantic_fragments
* semantic_fragments_minus_<fragment_type>
* discriminative_chain_static
* random_same_budget
* receiver_aware_dynamic
* receiver_aware_cheap_selector
* oracle_bridge

Gold labels are used only for offline scoring and the oracle setting.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_domain_workflow import read  # noqa: E402
from workflow_runtime.hotpot_bridge import bridge_matches_any, deduplicate_bridge_candidates, normalize_bridge  # noqa: E402
from workflow_runtime.hotpot_representation import (  # noqa: E402
    SelectorNeed,
    build_hotpot_chain_view_from_facts,
    build_hotpot_semantic_fragments,
    render_hotpot_chain_view,
    render_hotpot_discriminative_chain_static,
    render_hotpot_semantic_fragments,
)


FRAGMENT_TYPES = (
    "bridge_identity",
    "hop1_evidence",
    "hop2_evidence",
    "derived_answer",
    "answer_alignment",
    "contradiction_signal",
)

FRAGMENT_MINUS_SETTINGS = {
    f"semantic_fragments_minus_{fragment_type}": fragment_type
    for fragment_type in FRAGMENT_TYPES
}

SETTINGS = (
    "bridge_name_only",
    "bridge_plus_hop1",
    "bridge_plus_chain",
    "bridge_chain_answer",
    "full_chain_answer",
    "structured_chain_v1",
    "semantic_fragments",
    *FRAGMENT_MINUS_SETTINGS.keys(),
    "discriminative_chain_static",
    "random_same_budget",
    "receiver_aware_dynamic",
    "receiver_aware_cheap_selector",
    "oracle_bridge",
)

DISCRIMINATIVE_STATIC_FRAGMENT_COUNT = 3
COMPARISON_BUNDLES = {
    "B0": ("bridge_identity", "derived_answer"),
    "B1": ("bridge_identity", "hop1_evidence", "hop2_evidence"),
    "B2": ("bridge_identity", "hop1_evidence", "hop2_evidence", "answer_alignment"),
}
RECEIVER_INITIAL_FRAGMENT_TYPES = COMPARISON_BUNDLES["B0"]
RECEIVER_REVEAL_FRAGMENT_TYPES = ("hop1_evidence", "hop2_evidence", "answer_alignment")
RECEIVER_REVEAL_GAP_THRESHOLD = 0.10
RECEIVER_CHEAP_HIGH_CONFIDENCE_THRESHOLD = 0.90
RECEIVER_CHEAP_CONFIDENCE_THRESHOLD = 0.75


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--settings", default=",".join(SETTINGS))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p1-gate", action="store_true")
    parser.add_argument("--p2-gate", action="store_true")
    parser.add_argument("--p3-gate", action="store_true")
    parser.add_argument("--p4-gate", action="store_true")
    parser.add_argument("--receiver-reveal-gap-threshold", type=float, default=RECEIVER_REVEAL_GAP_THRESHOLD)
    parser.add_argument("--receiver-cheap-high-confidence-threshold", type=float, default=RECEIVER_CHEAP_HIGH_CONFIDENCE_THRESHOLD)
    parser.add_argument("--receiver-cheap-confidence-threshold", type=float, default=RECEIVER_CHEAP_CONFIDENCE_THRESHOLD)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    eval_report = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    records = list(eval_report.get("records", []))
    rows = read(Path(args.data_path), "hotpotqa", args.limit or len(records))
    rows_by_sample = {f"eval_hotpotqa_{index}": row for index, row in enumerate(rows)}
    rows_by_sample.update({str(row.get("sample_id")): row for row in rows if str(row.get("sample_id", "")).strip()})
    selected_settings = [item.strip() for item in args.settings.split(",") if item.strip()]
    unknown = sorted(set(selected_settings) - set(SETTINGS))
    if unknown:
        raise SystemExit(f"unknown settings: {', '.join(unknown)}")

    model = None
    if any(setting != "oracle_bridge" for setting in selected_settings):
        if not args.model_path:
            raise SystemExit("--model-path is required for non-oracle settings")
        from workflow_runtime.model_backend import TransformersModel

        model = TransformersModel(args.model_path, max_new_tokens=args.max_new_tokens)

    events = []
    for index, record in enumerate(records):
        row = rows_by_sample.get(str(record.get("sample_id", "")))
        if row is None and index < len(rows):
            row = rows[index]
        events.append(_event(record, row, candidate_limit=args.candidate_limit))
    events = [event for event in events if event]

    setting_rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        for setting in selected_settings:
            if setting == "oracle_bridge":
                selected, raw, metrics = _oracle_select(event), "", {}
                parse_fail = False
                selector_need = {}
            else:
                assert model is not None
                selected, raw, metrics, parse_fail, selector_need = _model_select(
                    model,
                    event,
                    setting=setting,
                    seed=args.seed + event_index,
                    receiver_reveal_gap_threshold=args.receiver_reveal_gap_threshold,
                    receiver_cheap_high_confidence_threshold=args.receiver_cheap_high_confidence_threshold,
                    receiver_cheap_confidence_threshold=args.receiver_cheap_confidence_threshold,
                )
            setting_rows.append({
                "sample_id": event["sample_id"],
                "setting": setting,
                "selected_bridge": selected,
                "selected_bridge_correct": bridge_matches_any(selected, event["gold_bridges"]),
                "gold_in_topk": bool(event["gold_in_topk"]),
                "raw_tail": raw[-1200:],
                "input_tokens": int(metrics.get("physical_input_tokens", 0) or 0),
                "output_tokens": int(metrics.get("output_tokens", 0) or 0),
                "parse_fail": bool(parse_fail),
                "candidate_count": len(event.get("candidates", [])),
                "selector_need": selector_need,
            })

    settings_summary = _summarize(setting_rows)
    summary = {
        "eval_json": args.eval_json,
        "data_path": args.data_path,
        "event_count": len(events),
        "candidate_limit": args.candidate_limit,
        "receiver_reveal_gap_threshold": args.receiver_reveal_gap_threshold,
        "receiver_cheap_high_confidence_threshold": args.receiver_cheap_high_confidence_threshold,
        "receiver_cheap_confidence_threshold": args.receiver_cheap_confidence_threshold,
        "settings": settings_summary,
        "p1_gate": _p1_gate(settings_summary, setting_rows) if args.p1_gate else None,
        "p2_gate": _p2_gate(settings_summary, setting_rows) if args.p2_gate else None,
        "p3_gate": _p3_gate(settings_summary, setting_rows) if args.p3_gate else None,
        "p4_gate": _p4_gate(settings_summary, setting_rows) if args.p4_gate else None,
        "events": events,
        "setting_rows": setting_rows,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"events", "setting_rows"}}, ensure_ascii=False, indent=2))


def _event(record: dict[str, Any], row: dict[str, Any] | None, *, candidate_limit: int) -> dict[str, Any]:
    if row is None:
        return {}
    candidates = _delivered_candidates(record, limit=candidate_limit)
    if not candidates:
        return {}
    facts = list(row.get("supporting_facts", []))
    gold_bridges = _gold_bridge_values(row)
    chains = [
        _chain_for_bridge(
            candidate,
            facts=facts,
            question=str(row.get("question", "")),
            candidate_id=f"c{index + 1}",
        )
        for index, candidate in enumerate(candidates)
    ]
    return {
        "sample_id": str(record.get("sample_id", "")),
        "question": str(row.get("question", "")),
        "gold_answer": row.get("gold_answer", ""),
        "gold_bridges": sorted(gold_bridges),
        "gold_in_topk": any(bridge_matches_any(candidate, gold_bridges) for candidate in candidates),
        "candidates": candidates,
        "candidate_chains": chains,
    }


def _delivered_candidates(record: dict[str, Any], *, limit: int) -> list[str]:
    logs = record.get("runtime_summary", {}).get("records", [])
    for log in logs:
        if log.get("role") != "solver" or log.get("mode") != "repair":
            continue
        stages = _bridge_stage_candidates(log)
        for stage in ("solver_visible", "serialized_packet", "repair_target_filtered", "ranked_top8"):
            candidates = deduplicate_bridge_candidates(stages.get(stage, []), limit=limit)
            if candidates:
                return candidates[:limit]
    return []


def _bridge_stage_candidates(log: dict[str, Any]) -> dict[str, list[str]]:
    for key in ("generation", "telemetry"):
        container = log.get(key)
        if not isinstance(container, dict):
            continue
        audit = container.get("hotpot_repair_packet_audit")
        if not isinstance(audit, dict):
            continue
        stages = audit.get("bridge_candidate_stages")
        if isinstance(stages, dict):
            return {
                str(stage): [str(item).strip() for item in values if str(item).strip()]
                for stage, values in stages.items()
                if isinstance(values, list)
            }
    return {}


def _chain_for_bridge(candidate: str, *, facts: list[dict[str, Any]], question: str, candidate_id: str) -> dict[str, Any]:
    bridge_key = normalize_bridge(candidate)
    bridge_facts = [
        _fact_payload(index, fact)
        for index, fact in enumerate(facts, start=1)
        if normalize_bridge(fact.get("title", fact.get("entity", ""))) == bridge_key
    ]
    if not bridge_facts:
        bridge_facts = _rank_facts(candidate, facts, limit=1)
    hop1 = bridge_facts[:2]
    hop2 = _rank_facts(" ".join([question, candidate, *[fact["text"] for fact in hop1]]), facts, limit=3)
    hop2 = [fact for fact in hop2 if fact not in hop1][:2] or hop2[:2]
    derived_hint = _derived_answer_hint(question, [*hop1, *hop2])
    view = build_hotpot_chain_view_from_facts(
        candidate_id=candidate_id,
        bridge_entity=candidate,
        question=question,
        facts=facts,
        hop1_fact_ids=[fact["fact_id"] for fact in hop1],
        hop2_fact_ids=[fact["fact_id"] for fact in hop2],
        derived_answer=derived_hint,
    )
    return view.as_dict()


def _rank_facts(query: str, facts: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    query_terms = _terms(query)
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for index, fact in enumerate(facts, start=1):
        payload = _fact_payload(index, fact)
        text_terms = _terms(f"{payload['title']} {payload['text']}")
        overlap = len(query_terms & text_terms)
        scored.append((-overlap, index, payload))
    scored.sort()
    return [payload for _, _, payload in scored[:limit]]


def _fact_payload(index: int, fact: dict[str, Any]) -> dict[str, Any]:
    return {
        "fact_id": f"supporting_fact_{index}",
        "title": str(fact.get("title", fact.get("entity", ""))),
        "sent_id": fact.get("sent_id"),
        "text": str(fact.get("text", fact.get("sentence", ""))),
    }


def _derived_answer_hint(question: str, facts: list[dict[str, Any]]) -> str:
    text = " ".join(fact["text"] for fact in facts)
    if re.search(r"\byes\b|\bno\b|same|both|also", question, flags=re.I):
        if re.search(r"\bnot\b|different|whereas|however", text, flags=re.I):
            return "no"
        return "yes_or_no_from_chain"
    date = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", text)
    if date:
        return date.group(1)
    return ""


def _model_select(
    model: Any,
    event: dict[str, Any],
    *,
    setting: str,
    seed: int,
    receiver_reveal_gap_threshold: float = RECEIVER_REVEAL_GAP_THRESHOLD,
    receiver_cheap_high_confidence_threshold: float = RECEIVER_CHEAP_HIGH_CONFIDENCE_THRESHOLD,
    receiver_cheap_confidence_threshold: float = RECEIVER_CHEAP_CONFIDENCE_THRESHOLD,
) -> tuple[str, str, dict[str, Any], bool, dict[str, Any]]:
    if setting == "receiver_aware_cheap_selector":
        return _cheap_receiver_select(
            model,
            event,
            seed=seed,
            receiver_reveal_gap_threshold=receiver_reveal_gap_threshold,
            receiver_cheap_high_confidence_threshold=receiver_cheap_high_confidence_threshold,
            receiver_cheap_confidence_threshold=receiver_cheap_confidence_threshold,
        )
    prompt, selector_need = _selection_prompt(
        event,
        setting,
        seed=seed,
        receiver_reveal_gap_threshold=receiver_reveal_gap_threshold,
    )
    raw, metrics = _call_selector_model(model, prompt, session_key=f"bridge-chain-rerank:{seed}")
    selected, parse_fail = _parse_selected_bridge(raw, event["candidate_chains"])
    return selected, raw, metrics, parse_fail, selector_need


def _cheap_receiver_select(
    model: Any,
    event: dict[str, Any],
    *,
    seed: int,
    receiver_reveal_gap_threshold: float,
    receiver_cheap_high_confidence_threshold: float,
    receiver_cheap_confidence_threshold: float,
) -> tuple[str, str, dict[str, Any], bool, dict[str, Any]]:
    cheap_blocks = [_render_cheap_b0(_chain_view_from_dict(chain), event["question"]) for chain in event["candidate_chains"]]
    cheap_prompt = _cheap_selector_prompt(event, cheap_blocks)
    cheap_raw, cheap_metrics = _call_selector_model(model, cheap_prompt, session_key=f"cheap-provisional:{seed}")
    provisional = _parse_provisional_selection(cheap_raw, event["candidate_chains"])
    selected = provisional["selected_bridge"]
    parse_fail = bool(provisional["parse_fail"])
    gap = float(provisional.get("top1_score", 0.0) or 0.0) - float(provisional.get("top2_score", 0.0) or 0.0)
    confidence = float(provisional.get("confidence", 0.0) or 0.0)
    should_stop = _cheap_should_stop(
        parse_fail=parse_fail,
        confidence=confidence,
        gap=gap,
        tau_high=receiver_cheap_high_confidence_threshold,
        tau_conf=receiver_cheap_confidence_threshold,
        tau_gap=receiver_reveal_gap_threshold,
    )
    if should_stop:
        selector_need = _cheap_selector_telemetry(
            event,
            provisional=provisional,
            receiver_reveal_gap_threshold=receiver_reveal_gap_threshold,
            receiver_cheap_high_confidence_threshold=receiver_cheap_high_confidence_threshold,
            receiver_cheap_confidence_threshold=receiver_cheap_confidence_threshold,
            stopped_after_initial=True,
            bundle_pattern=None,
        )
        return selected, cheap_raw, cheap_metrics, parse_fail, selector_need

    blocks, selector_need = _receiver_aware_cheap_reveal_blocks(
        event,
        provisional=provisional,
        receiver_reveal_gap_threshold=receiver_reveal_gap_threshold,
        receiver_cheap_high_confidence_threshold=receiver_cheap_high_confidence_threshold,
        receiver_cheap_confidence_threshold=receiver_cheap_confidence_threshold,
    )
    final_prompt = _selection_prompt_text(event, blocks)
    final_raw, final_metrics = _call_selector_model(model, final_prompt, session_key=f"cheap-reveal:{seed}")
    final_selected, final_parse_fail = _parse_selected_bridge(final_raw, event["candidate_chains"])
    return (
        final_selected,
        cheap_raw[-800:] + "\n\n--- FINAL ---\n" + final_raw,
        _merge_metrics(cheap_metrics, final_metrics),
        bool(parse_fail or final_parse_fail),
        selector_need,
    )


def _call_selector_model(model: Any, prompt: str, *, session_key: str) -> tuple[str, dict[str, Any]]:
    session_prompt = _chat_prompt(model, prompt)
    request = type("Request", (), {"prompt": prompt, "session_prompt": session_prompt})()
    raw = str(model(request))
    snapshot = model.session_snapshot(session_key) if hasattr(model, "session_snapshot") else {}
    if isinstance(snapshot, dict):
        return raw, dict(snapshot.get("last_call_metrics", {}) or {})
    return raw, {}


def _merge_metrics(*items: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    numeric_keys = {"physical_input_tokens", "output_tokens", "logical_input_tokens", "cached_input_tokens"}
    for item in items:
        for key, value in item.items():
            if key in numeric_keys:
                merged[key] = int(merged.get(key, 0) or 0) + int(value or 0)
            elif key not in merged:
                merged[key] = value
    return merged


def _cheap_should_stop(
    *,
    parse_fail: bool,
    confidence: float,
    gap: float,
    tau_high: float,
    tau_conf: float,
    tau_gap: float,
) -> bool:
    if parse_fail:
        return False
    if confidence >= tau_high:
        return True
    if confidence >= tau_conf and gap >= tau_gap:
        return True
    return False


def _selection_prompt(
    event: dict[str, Any],
    setting: str,
    *,
    seed: int = 0,
    receiver_reveal_gap_threshold: float = RECEIVER_REVEAL_GAP_THRESHOLD,
) -> tuple[str, dict[str, Any]]:
    blocks = []
    selector_need: dict[str, Any] = {}
    random_fragment_types = _random_same_budget_fragment_types(seed) if setting == "random_same_budget" else None
    if setting == "receiver_aware_dynamic":
        blocks, selector_need = _receiver_aware_dynamic_blocks(
            event,
            receiver_reveal_gap_threshold=receiver_reveal_gap_threshold,
        )
    else:
        for chain in event["candidate_chains"]:
            view = _chain_view_from_dict(chain)
            fragment_include_types = _fragment_include_types(setting)
            if random_fragment_types is not None:
                blocks.append(render_hotpot_semantic_fragments(
                    view,
                    event["question"],
                    include_types=random_fragment_types,
                ))
            elif setting == "discriminative_chain_static":
                blocks.append(render_hotpot_discriminative_chain_static(view, event["question"]))
            elif fragment_include_types is not None:
                blocks.append(render_hotpot_semantic_fragments(
                    view,
                    event["question"],
                    include_types=fragment_include_types,
                ))
            elif setting == "structured_chain_v1":
                blocks.append(_render_structured_chain_v1_compact(chain))
            elif setting == "full_chain_answer":
                blocks.append(_render_full_chain_answer(chain))
            else:
                blocks.append(render_hotpot_chain_view(
                    view,
                    include_hop1=setting in {
                        "bridge_plus_hop1",
                        "bridge_plus_chain",
                        "bridge_chain_answer",
                        "structured_chain_v1",
                    },
                    include_hop2=setting in {"bridge_plus_chain", "bridge_chain_answer", "structured_chain_v1"},
                    include_answer=setting in {"bridge_chain_answer", "structured_chain_v1"},
                ))
    prompt = _selection_prompt_text(event, blocks)
    return prompt, selector_need


def _selection_prompt_text(event: dict[str, Any], blocks: list[str]) -> str:
    return (
        "Given the HotpotQA question and candidate bridge/evidence chains, choose the best existing candidate.\n"
        "Check whether the evidence connects the bridge to what the question asks. "
        f"There are exactly {len(event['candidate_chains'])} candidates: "
        + ", ".join(str(chain["candidate_id"]) for chain in event["candidate_chains"])
        + ". Do not create new candidates. Do not repeat or continue the candidate list. "
        "Return one JSON object only with schema: "
        "{\"ranked_candidate_ids\":[\"c1\"],\"selected_candidate_id\":\"c1\","
        "\"confidence\":0.0,\"decisive_evidence\":\"...\"}. "
        "Keep decisive_evidence under 12 words.\n\n"
        f"Question:\n{event['question']}\n\n"
        "Candidates:\n"
        + "\n\n".join(blocks)
        + "\n\nSelection JSON:\n"
    )


def _cheap_selector_prompt(event: dict[str, Any], blocks: list[str]) -> str:
    return (
        "Given the HotpotQA question and minimal candidate summaries, make a provisional ranking.\n"
        "Use only bridge, derived answer, answer type, and alignment-lite. Do not assume missing hop evidence. "
        f"There are exactly {len(event['candidate_chains'])} candidates: "
        + ", ".join(str(chain["candidate_id"]) for chain in event["candidate_chains"])
        + ". Return one JSON object only with schema: "
        "{\"ranking\":[\"c1\",\"c2\"],\"selected_candidate_id\":\"c1\","
        "\"top1_score\":0.0,\"top2_score\":0.0,\"confidence\":0.0,"
        "\"uncertainty_type\":\"low|medium|high\"}. "
        "Scores and confidence must be between 0 and 1.\n\n"
        f"Question:\n{event['question']}\n\n"
        "Minimal candidates:\n"
        + "\n\n".join(blocks)
        + "\n\nProvisional JSON:\n"
    )


def _render_cheap_b0(view: Any, question: str) -> str:
    del question
    alignment = _alignment_label_from_text(str(view.answer_alignment or ""))
    return "\n".join([
        f"[CANDIDATE {view.candidate_id}]",
        f"bridge: {view.bridge_entity}",
        f"derived_answer: {view.derived_answer}",
        f"answer_type: {view.answer_type}",
        f"alignment_lite: {alignment}",
    ]).strip()


def _alignment_label_from_text(text: str) -> str:
    value = str(text or "").lower()
    if "contradict" in value or "not supported" in value:
        return "contradicted"
    if "supported" in value or "derived" in value or "appears" in value:
        return "supported"
    return "unknown"


def _receiver_aware_dynamic_blocks(
    event: dict[str, Any],
    *,
    receiver_reveal_gap_threshold: float = RECEIVER_REVEAL_GAP_THRESHOLD,
) -> tuple[list[str], dict[str, Any]]:
    views = [_chain_view_from_dict(chain) for chain in event["candidate_chains"]]
    need = _diagnose_selector_need(
        views,
        event["question"],
        receiver_reveal_gap_threshold=receiver_reveal_gap_threshold,
    )
    blocks = []
    initial_blocks = []
    extra_blocks = []
    fragment_pattern: dict[str, list[str]] = {}
    bundle_pattern: dict[str, str] = {}
    for view in views:
        bundle = _bundle_for_missing_fragments(need.missing_fragments.get(view.candidate_id, []))
        include_types = COMPARISON_BUNDLES[bundle]
        reveal = tuple(fragment_type for fragment_type in include_types if fragment_type not in RECEIVER_INITIAL_FRAGMENT_TYPES)
        bundle_pattern[view.candidate_id] = bundle
        fragment_pattern[view.candidate_id] = list(include_types)
        blocks.append(render_hotpot_semantic_fragments(view, event["question"], include_types=include_types))
        initial_blocks.append(render_hotpot_semantic_fragments(
            view,
            event["question"],
            include_types=RECEIVER_INITIAL_FRAGMENT_TYPES,
        ))
        if reveal:
            extra_blocks.append(render_hotpot_semantic_fragments(view, event["question"], include_types=reveal))
    telemetry = need.as_dict()
    telemetry.update({
        "fragment_pattern": fragment_pattern,
        "bundle_pattern": bundle_pattern,
        "initial_packet_tokens": _logical_token_count("\n\n".join(initial_blocks)),
        "requested_extra_tokens": _logical_token_count("\n\n".join(extra_blocks)),
        "requested_candidate_count": sum(1 for bundle in bundle_pattern.values() if bundle != "B0"),
        "candidate_count": len(views),
        "visible_fragment_count": sum(len(values) for values in fragment_pattern.values()),
        "full_fallback": all(
            set(values) >= set(FRAGMENT_TYPES)
            for values in fragment_pattern.values()
        ) if fragment_pattern else False,
        "within_sample_dynamic": len({tuple(values) for values in fragment_pattern.values()}) > 1,
        "receiver_reveal_gap_threshold": receiver_reveal_gap_threshold,
    })
    return blocks, telemetry


def _receiver_aware_cheap_reveal_blocks(
    event: dict[str, Any],
    *,
    provisional: dict[str, Any],
    receiver_reveal_gap_threshold: float,
    receiver_cheap_high_confidence_threshold: float,
    receiver_cheap_confidence_threshold: float,
) -> tuple[list[str], dict[str, Any]]:
    views = [_chain_view_from_dict(chain) for chain in event["candidate_chains"]]
    views_by_id = {view.candidate_id: view for view in views}
    ranking = [candidate_id for candidate_id in provisional.get("ranking", []) if candidate_id in views_by_id]
    if not ranking:
        ranking = [view.candidate_id for view in views[:3]]
    top_ids = ranking[:3]
    answers = {_norm_answer(views_by_id[candidate_id].derived_answer) for candidate_id in top_ids if _norm_answer(views_by_id[candidate_id].derived_answer)}
    bundle = "B2" if len(answers) > 1 else "B1"
    bundle_pattern = {
        view.candidate_id: (bundle if view.candidate_id in set(top_ids) else "B0")
        for view in views
    }
    blocks = []
    initial_blocks = []
    extra_blocks = []
    fragment_pattern: dict[str, list[str]] = {}
    for view in views:
        include_types = COMPARISON_BUNDLES[bundle_pattern[view.candidate_id]]
        reveal = tuple(fragment_type for fragment_type in include_types if fragment_type not in RECEIVER_INITIAL_FRAGMENT_TYPES)
        fragment_pattern[view.candidate_id] = list(include_types)
        blocks.append(render_hotpot_semantic_fragments(view, event["question"], include_types=include_types))
        initial_blocks.append(render_hotpot_semantic_fragments(view, event["question"], include_types=RECEIVER_INITIAL_FRAGMENT_TYPES))
        if reveal:
            extra_blocks.append(render_hotpot_semantic_fragments(view, event["question"], include_types=reveal))
    telemetry = _cheap_selector_telemetry(
        event,
        provisional=provisional,
        receiver_reveal_gap_threshold=receiver_reveal_gap_threshold,
        receiver_cheap_high_confidence_threshold=receiver_cheap_high_confidence_threshold,
        receiver_cheap_confidence_threshold=receiver_cheap_confidence_threshold,
        stopped_after_initial=False,
        bundle_pattern=bundle_pattern,
    )
    telemetry.update({
        "unresolved_candidates": top_ids,
        "disagreement_type": "answer_disagreement" if bundle == "B2" else "low_confidence",
        "missing_fragments": {
            candidate_id: [
                fragment_type
                for fragment_type in COMPARISON_BUNDLES[bundle]
                if fragment_type not in RECEIVER_INITIAL_FRAGMENT_TYPES
            ]
            for candidate_id in top_ids
        },
        "fragment_pattern": fragment_pattern,
        "initial_packet_tokens": _logical_token_count("\n\n".join(initial_blocks)),
        "requested_extra_tokens": _logical_token_count("\n\n".join(extra_blocks)),
        "requested_candidate_count": len(top_ids),
        "candidate_count": len(views),
        "visible_fragment_count": sum(len(values) for values in fragment_pattern.values()),
        "full_fallback": False,
        "within_sample_dynamic": len({tuple(values) for values in fragment_pattern.values()}) > 1,
    })
    return blocks, telemetry


def _cheap_selector_telemetry(
    event: dict[str, Any],
    *,
    provisional: dict[str, Any],
    receiver_reveal_gap_threshold: float,
    receiver_cheap_high_confidence_threshold: float,
    receiver_cheap_confidence_threshold: float,
    stopped_after_initial: bool,
    bundle_pattern: dict[str, str] | None,
) -> dict[str, Any]:
    views = [_chain_view_from_dict(chain) for chain in event["candidate_chains"]]
    pattern = bundle_pattern or {view.candidate_id: "B0" for view in views}
    fragment_pattern = {
        candidate_id: list(COMPARISON_BUNDLES[bundle])
        for candidate_id, bundle in pattern.items()
    }
    initial_blocks = [
        render_hotpot_semantic_fragments(view, event["question"], include_types=RECEIVER_INITIAL_FRAGMENT_TYPES)
        for view in views
    ]
    return {
        "mode": "cheap_selector",
        "provisional": provisional,
        "confidence": float(provisional.get("confidence", 0.0) or 0.0),
        "top1_top2_gap": float(provisional.get("top1_score", 0.0) or 0.0) - float(provisional.get("top2_score", 0.0) or 0.0),
        "uncertainty_type": provisional.get("uncertainty_type", ""),
        "stopped_after_initial": stopped_after_initial,
        "receiver_reveal_gap_threshold": receiver_reveal_gap_threshold,
        "receiver_cheap_high_confidence_threshold": receiver_cheap_high_confidence_threshold,
        "receiver_cheap_confidence_threshold": receiver_cheap_confidence_threshold,
        "bundle_pattern": pattern,
        "fragment_pattern": fragment_pattern,
        "initial_packet_tokens": _logical_token_count("\n\n".join(initial_blocks)),
        "requested_extra_tokens": 0,
        "requested_candidate_count": 0,
        "candidate_count": len(views),
        "visible_fragment_count": sum(len(values) for values in fragment_pattern.values()),
        "full_fallback": False,
        "within_sample_dynamic": len({tuple(values) for values in fragment_pattern.values()}) > 1,
        "unresolved_candidates": [],
        "disagreement_type": None,
        "missing_fragments": {},
    }


def _bundle_for_missing_fragments(missing: list[str]) -> str:
    wanted = set(missing)
    if "answer_alignment" in wanted:
        return "B2"
    if {"hop1_evidence", "hop2_evidence"} & wanted:
        return "B1"
    return "B0"


def _diagnose_selector_need(
    views: list[Any],
    question: str,
    *,
    receiver_reveal_gap_threshold: float = RECEIVER_REVEAL_GAP_THRESHOLD,
) -> SelectorNeed:
    scored = sorted(
        ((_initial_candidate_score(view, question), view.candidate_id, view) for view in views),
        key=lambda item: (-item[0], item[1]),
    )
    top_score = scored[0][0] if scored else 0.0
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    gap = top_score - second_score
    confidence = max(0.0, min(1.0, 0.55 + gap))
    answers = {_norm_answer(view.derived_answer) for view in views if _norm_answer(view.derived_answer)}
    top2 = [item[2] for item in scored[:2]]
    uncertain = gap < receiver_reveal_gap_threshold
    unresolved = [item[2] for item in scored[:3]] if uncertain else top2[:1]
    disagreement_type = _selector_disagreement_type(unresolved, answers, gap, uncertain=uncertain)
    missing: dict[str, list[str]] = {view.candidate_id: [] for view in views}
    if uncertain and disagreement_type == "answer_disagreement":
        symmetric = ["hop1_evidence", "hop2_evidence", "answer_alignment"]
        for view in unresolved:
            missing[view.candidate_id].extend(symmetric)
    elif uncertain and len({_norm_answer(view.derived_answer) for view in top2 if _norm_answer(view.derived_answer)}) <= 1 and len(top2) > 1:
        for view in top2:
            missing[view.candidate_id].append("hop2_evidence")
    if uncertain and disagreement_type in {"bridge_conflict", "low_confidence"}:
        for view in unresolved:
            missing[view.candidate_id].extend(["hop1_evidence", "hop2_evidence"])
    for view in views:
        if not uncertain:
            continue
        if disagreement_type == "answer_disagreement" and view in unresolved:
            continue
        if _fragment_value(view, "contradiction_signal", question):
            missing[view.candidate_id].append("answer_alignment")
    missing = {
        candidate_id: sorted(set(values), key=lambda value: RECEIVER_REVEAL_FRAGMENT_TYPES.index(value) if value in RECEIVER_REVEAL_FRAGMENT_TYPES else 99)
        for candidate_id, values in missing.items()
        if values
    }
    return SelectorNeed(
        unresolved_candidates=[view.candidate_id for view in unresolved],
        disagreement_type=disagreement_type,
        missing_fragments=missing,
        confidence=confidence,
    )


def _initial_candidate_score(view: Any, question: str) -> float:
    fragments = {fragment.type: fragment.as_dict() for fragment in build_hotpot_semantic_fragments(view, question)}
    score = 0.0
    if str(fragments.get("bridge_identity", {}).get("bridge", "")).strip():
        score += 0.20
    answer = _norm_answer(fragments.get("derived_answer", {}).get("value", ""))
    if answer:
        score += 0.20
    alignment = str(fragments.get("answer_alignment", {}).get("alignment", "")).lower()
    if alignment == "supported":
        score += 0.25
    elif alignment == "unknown":
        score += 0.05
    if fragments.get("contradiction_signal", {}).get("value"):
        score -= 0.35
    return score


def _selector_disagreement_type(unresolved: list[Any], answers: set[str], gap: float, *, uncertain: bool) -> str | None:
    if not uncertain:
        return None
    if len(answers) > 1:
        return "answer_disagreement"
    if gap < 0.10:
        return "low_confidence"
    bridges = {normalize_bridge(view.bridge_entity) for view in unresolved if str(view.bridge_entity).strip()}
    if len(bridges) > 1:
        return "bridge_conflict"
    return None


def _fragment_value(view: Any, fragment_type: str, question: str) -> Any:
    for fragment in build_hotpot_semantic_fragments(view, question):
        if fragment.type == fragment_type:
            data = fragment.as_dict()
            return data.get("value")
    return None


def _norm_answer(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def _logical_token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _fragment_include_types(setting: str) -> tuple[str, ...] | None:
    if setting == "semantic_fragments":
        return FRAGMENT_TYPES
    removed = FRAGMENT_MINUS_SETTINGS.get(setting)
    if removed:
        return tuple(fragment_type for fragment_type in FRAGMENT_TYPES if fragment_type != removed)
    return None


def _random_same_budget_fragment_types(seed: int) -> tuple[str, ...]:
    rng = random.Random(seed)
    return tuple(sorted(rng.sample(FRAGMENT_TYPES, k=DISCRIMINATIVE_STATIC_FRAGMENT_COUNT)))


def _chat_prompt(model: Any, prompt: str) -> str:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a strict HotpotQA candidate selector. "
                        "Output only the requested JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _render_full_chain_answer(chain: dict[str, Any]) -> str:
    return (
        f"[CANDIDATE {chain['candidate_id']}]\n"
        f"bridge_entity: {chain['bridge_entity']}\n"
        f"hop1_fact_ids: {', '.join(chain.get('hop1_fact_ids', []))}\n"
        f"hop1_evidence_chain: {chain.get('bridge_entity', '')} :: {chain.get('hop1_key_sentence', '')}\n"
        f"hop2_fact_ids: {', '.join(chain.get('hop2_fact_ids', []))}\n"
        f"hop2_evidence_chain: {chain.get('bridge_entity', '')} :: {chain.get('hop2_key_sentence', '')}\n"
        f"derived_answer: {chain.get('derived_answer', '')}\n"
        f"answer_type: {chain.get('answer_type', '')}\n"
        f"answer_alignment: {chain.get('answer_alignment') or ''}\n"
        f"cited_fact_ids: {', '.join(chain.get('cited_fact_ids', []))}"
    )


def _render_structured_chain_v1_compact(chain: dict[str, Any]) -> str:
    return (
        f"[{chain['candidate_id']}] b={chain['bridge_entity']}\n"
        f"h1 {','.join(chain.get('hop1_fact_ids', []))}: {chain.get('hop1_key_sentence', '')}\n"
        f"h2 {','.join(chain.get('hop2_fact_ids', []))}: {chain.get('hop2_key_sentence', '')}\n"
        f"ans={chain.get('derived_answer', '')}; type={chain.get('answer_type', '')}; "
        f"align={chain.get('answer_alignment') or ''}"
    )


def _parse_selected_bridge(raw: str, chains: list[dict[str, Any]]) -> tuple[str, bool]:
    by_id = {str(chain["candidate_id"]): str(chain["bridge_entity"]) for chain in chains}
    try:
        payload = json.loads(_json_object(raw))
        candidate_id = str(payload.get("selected_candidate_id", "")).strip()
        if candidate_id.isdigit():
            candidate_id = f"c{candidate_id}"
        if candidate_id in by_id:
            return by_id[candidate_id], False
        ranked = payload.get("ranked_candidate_ids")
        if isinstance(ranked, list):
            for item in ranked:
                candidate_id = str(item).strip()
                if candidate_id.isdigit():
                    candidate_id = f"c{candidate_id}"
                if candidate_id in by_id:
                    return by_id[candidate_id], False
    except Exception:
        pass
    selected_match = re.search(r"selected_candidate_id['\"]?\s*[:=]\s*['\"]?(c?[1-9])", raw, flags=re.I)
    if selected_match:
        value = selected_match.group(1).lower()
        candidate_id = value if value.startswith("c") else f"c{value}"
        if candidate_id in by_id:
            return by_id[candidate_id], False
    match = re.search(r"\bc([1-9])\b", raw)
    if match:
        candidate_id = f"c{match.group(1)}"
        if candidate_id in by_id:
            return by_id[candidate_id], False
    return (chains[0]["bridge_entity"] if chains else ""), True


def _parse_provisional_selection(raw: str, chains: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(chain["candidate_id"]): str(chain["bridge_entity"]) for chain in chains}
    candidate_ids = list(by_id)
    payload: dict[str, Any] = {}
    parse_fail = False
    try:
        loaded = json.loads(_json_object(raw))
        if isinstance(loaded, dict):
            payload = loaded
    except Exception:
        parse_fail = True
    ranking = _parse_candidate_id_list(payload.get("ranking"), candidate_ids)
    selected_id = _candidate_id(str(payload.get("selected_candidate_id", "")).strip(), candidate_ids)
    if not selected_id and ranking:
        selected_id = ranking[0]
    if not ranking and selected_id:
        ranking = [selected_id]
    if not selected_id:
        selected_id = _candidate_id_from_text(raw, candidate_ids)
    if not selected_id:
        selected_id = candidate_ids[0] if candidate_ids else ""
        parse_fail = True
    if selected_id and selected_id not in ranking:
        ranking = [selected_id, *[candidate_id for candidate_id in ranking if candidate_id != selected_id]]
    ranking = [*ranking, *[candidate_id for candidate_id in candidate_ids if candidate_id not in ranking]]
    top1_score = _score_value(payload.get("top1_score"), default=0.0)
    top2_score = _score_value(payload.get("top2_score"), default=0.0)
    confidence = _score_value(payload.get("confidence"), default=max(0.0, min(1.0, top1_score - top2_score)))
    return {
        "ranking": ranking,
        "selected_candidate_id": selected_id,
        "selected_bridge": by_id.get(selected_id, ""),
        "top1_score": top1_score,
        "top2_score": top2_score,
        "confidence": confidence,
        "uncertainty_type": str(payload.get("uncertainty_type", "")),
        "parse_fail": parse_fail,
    }


def _parse_candidate_id_list(value: Any, valid_ids: list[str]) -> list[str]:
    result: list[str] = []
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = re.findall(r"\bc?[1-9]\b", value)
    else:
        items = []
    for item in items:
        candidate_id = _candidate_id(str(item).strip(), valid_ids)
        if candidate_id and candidate_id not in result:
            result.append(candidate_id)
    return result


def _candidate_id(value: str, valid_ids: list[str]) -> str:
    value = str(value or "").strip().lower()
    if value.isdigit():
        value = f"c{value}"
    return value if value in set(valid_ids) else ""


def _candidate_id_from_text(raw: str, valid_ids: list[str]) -> str:
    selected_match = re.search(r"selected_candidate_id['\"]?\s*[:=]\s*['\"]?(c?[1-9])", raw, flags=re.I)
    if selected_match:
        return _candidate_id(selected_match.group(1), valid_ids)
    match = re.search(r"\bc([1-9])\b", raw)
    if match:
        return _candidate_id(f"c{match.group(1)}", valid_ids)
    return ""


def _score_value(value: Any, *, default: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, score))


def _chain_view_from_dict(value: dict[str, Any]):
    from workflow_runtime.hotpot_representation import HotpotChainView

    return HotpotChainView(
        candidate_id=str(value.get("candidate_id", "")),
        bridge_entity=str(value.get("bridge_entity", "")),
        hop1_fact_ids=[str(item) for item in value.get("hop1_fact_ids", [])],
        hop1_key_sentence=str(value.get("hop1_key_sentence", "")),
        hop2_fact_ids=[str(item) for item in value.get("hop2_fact_ids", [])],
        hop2_key_sentence=str(value.get("hop2_key_sentence", "")),
        derived_answer=str(value.get("derived_answer", "")),
        answer_type=str(value.get("answer_type", "")),
        answer_alignment=value.get("answer_alignment"),
        cited_fact_ids=[str(item) for item in value.get("cited_fact_ids", [])],
    )


def _oracle_select(event: dict[str, Any]) -> str:
    for candidate in event["candidates"]:
        if bridge_matches_any(candidate, event["gold_bridges"]):
            return candidate
    return event["candidates"][0] if event["candidates"] else ""


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_setting: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_setting.setdefault(str(row["setting"]), []).append(row)
    return {
        setting: {
            "count": float(len(items)),
            "selected_bridge_accuracy": _mean(item["selected_bridge_correct"] for item in items),
            "gold_in_topk_rate": _mean(item["gold_in_topk"] for item in items),
            "input_tokens": sum(int(item.get("input_tokens", 0) or 0) for item in items),
            "output_tokens": sum(int(item.get("output_tokens", 0) or 0) for item in items),
            "tokens_per_candidate": (
                sum(int(item.get("input_tokens", 0) or 0) for item in items)
                / max(1, sum(int(item.get("candidate_count", 0) or 0) for item in items))
            ),
            "parse_fail": sum(int(bool(item.get("parse_fail"))) for item in items),
            "candidate_count_mean": (
                sum(int(item.get("candidate_count", 0) or 0) for item in items) / len(items)
                if items else 0.0
            ),
        }
        for setting, items in sorted(by_setting.items())
    }


def _p1_gate(summary: dict[str, dict[str, float]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    full = summary.get("full_chain_answer", {})
    structured = summary.get("structured_chain_v1", {})
    oracle = summary.get("oracle_bridge", {})
    full_acc = float(full.get("selected_bridge_accuracy", 0.0) or 0.0)
    structured_acc = float(structured.get("selected_bridge_accuracy", 0.0) or 0.0)
    full_tokens = int(full.get("input_tokens", 0) or 0)
    structured_tokens = int(structured.get("input_tokens", 0) or 0)
    token_saving = (full_tokens - structured_tokens) / full_tokens if full_tokens else 0.0
    oracle_acc = float(oracle.get("selected_bridge_accuracy", 1.0) or 1.0)
    return {
        "quality_pass": structured_acc >= full_acc - 0.02,
        "exploratory_quality_pass_10": structured_acc >= full_acc - 0.1,
        "cost_pass": token_saving >= 0.20,
        "stability_pass": (
            int(structured.get("parse_fail", 0) or 0) == 0
            and _candidate_counts_identical(rows, "full_chain_answer", "structured_chain_v1")
            and float(structured.get("gold_in_topk_rate", 0.0) or 0.0) == float(full.get("gold_in_topk_rate", 0.0) or 0.0)
        ),
        "p1_pass": structured_acc >= full_acc - 0.02 and token_saving >= 0.20 and int(structured.get("parse_fail", 0) or 0) == 0,
        "full_accuracy": full_acc,
        "structured_accuracy": structured_acc,
        "accuracy_drop": full_acc - structured_acc,
        "full_input_tokens": full_tokens,
        "structured_input_tokens": structured_tokens,
        "token_saving": token_saving,
        "structured_oracle_gap": oracle_acc - structured_acc,
        "full_oracle_gap": oracle_acc - full_acc,
    }


def _p2_gate(summary: dict[str, dict[str, float]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    structured = summary.get("structured_chain_v1", {})
    all_fragments = summary.get("semantic_fragments", {})
    oracle = summary.get("oracle_bridge", {})
    structured_acc = float(structured.get("selected_bridge_accuracy", 0.0) or 0.0)
    all_acc = float(all_fragments.get("selected_bridge_accuracy", 0.0) or 0.0)
    oracle_acc = float(oracle.get("selected_bridge_accuracy", 1.0) or 1.0)
    fragment_importance: dict[str, float] = {}
    minus_accuracy: dict[str, float] = {}
    for setting, fragment_type in FRAGMENT_MINUS_SETTINGS.items():
        item = summary.get(setting, {})
        acc = float(item.get("selected_bridge_accuracy", 0.0) or 0.0)
        minus_accuracy[fragment_type] = acc
        fragment_importance[fragment_type] = all_acc - acc

    strict_low_value = [
        fragment
        for fragment, delta in sorted(fragment_importance.items(), key=lambda item: item[1])
        if delta <= 0.02
    ]
    exploratory_low_value = [
        fragment
        for fragment, delta in sorted(fragment_importance.items(), key=lambda item: item[1])
        if delta <= 0.10
    ]
    quality_pass = all_acc >= structured_acc - 0.02
    relevant_settings = ["structured_chain_v1", "semantic_fragments", *FRAGMENT_MINUS_SETTINGS.keys()]
    parse_fail_total = sum(int(summary.get(setting, {}).get("parse_fail", 0) or 0) for setting in relevant_settings)
    stability_pass = (
        parse_fail_total == 0
        and _candidate_counts_identical_for_settings(rows, relevant_settings)
        and _gold_in_topk_identical(summary, relevant_settings)
    )
    p2_pass = quality_pass and stability_pass and bool(strict_low_value)
    return {
        "quality_pass": quality_pass,
        "stability_pass": stability_pass,
        "p2_pass": p2_pass,
        "p2_stop": quality_pass and stability_pass and not strict_low_value,
        "all_fragments_accuracy": all_acc,
        "structured_accuracy": structured_acc,
        "accuracy_delta_vs_structured": all_acc - structured_acc,
        "oracle_gap": oracle_acc - all_acc,
        "fragment_importance": {
            fragment: fragment_importance[fragment]
            for fragment in sorted(fragment_importance, key=fragment_importance.get, reverse=True)
        },
        "minus_accuracy": minus_accuracy,
        "strict_low_value_fragments": strict_low_value,
        "exploratory_low_value_fragments_10_sample": exploratory_low_value,
        "parse_fail_total": parse_fail_total,
        "all_fragments_parse_fail": int(all_fragments.get("parse_fail", 0) or 0),
    }


def _p3_gate(summary: dict[str, dict[str, float]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    full = summary.get("full_chain_answer", {})
    structured = summary.get("structured_chain_v1", {})
    static = summary.get("discriminative_chain_static", {})
    random_budget = summary.get("random_same_budget", {})
    oracle = summary.get("oracle_bridge", {})
    full_acc = float(full.get("selected_bridge_accuracy", 0.0) or 0.0)
    structured_acc = float(structured.get("selected_bridge_accuracy", 0.0) or 0.0)
    static_acc = float(static.get("selected_bridge_accuracy", 0.0) or 0.0)
    random_acc = float(random_budget.get("selected_bridge_accuracy", 0.0) or 0.0)
    oracle_acc = float(oracle.get("selected_bridge_accuracy", 1.0) or 1.0)
    static_tokens = int(static.get("input_tokens", 0) or 0)
    structured_tokens = int(structured.get("input_tokens", 0) or 0)
    full_tokens = int(full.get("input_tokens", 0) or 0)
    random_tokens = int(random_budget.get("input_tokens", 0) or 0)
    relevant_settings = ["full_chain_answer", "structured_chain_v1", "discriminative_chain_static", "random_same_budget"]
    parse_fail_total = sum(int(summary.get(setting, {}).get("parse_fail", 0) or 0) for setting in relevant_settings)
    stability_pass = (
        parse_fail_total == 0
        and _candidate_counts_identical_for_settings(rows, relevant_settings)
        and _gold_in_topk_identical(summary, relevant_settings)
    )
    quality_pass = static_acc >= structured_acc - 0.02
    random_control_pass = static_acc > random_acc
    return {
        "quality_pass": quality_pass,
        "random_control_pass": random_control_pass,
        "stability_pass": stability_pass,
        "p3_pass": quality_pass and random_control_pass and stability_pass,
        "full_accuracy": full_acc,
        "structured_accuracy": structured_acc,
        "discriminative_static_accuracy": static_acc,
        "random_same_budget_accuracy": random_acc,
        "oracle_accuracy": oracle_acc,
        "static_vs_structured_delta": static_acc - structured_acc,
        "static_vs_random_delta": static_acc - random_acc,
        "oracle_gap": oracle_acc - static_acc,
        "static_input_tokens": static_tokens,
        "structured_input_tokens": structured_tokens,
        "full_input_tokens": full_tokens,
        "random_same_budget_input_tokens": random_tokens,
        "static_token_saving_vs_structured": (
            (structured_tokens - static_tokens) / structured_tokens if structured_tokens else 0.0
        ),
        "static_token_saving_vs_full": (
            (full_tokens - static_tokens) / full_tokens if full_tokens else 0.0
        ),
        "parse_fail_total": parse_fail_total,
    }


def _p4_gate(summary: dict[str, dict[str, float]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    full = summary.get("full_chain_answer", {})
    static = summary.get("discriminative_chain_static", {})
    receiver_setting = (
        "receiver_aware_cheap_selector"
        if "receiver_aware_cheap_selector" in summary
        else "receiver_aware_dynamic"
    )
    dynamic = summary.get(receiver_setting, {})
    random_budget = summary.get("random_same_budget", {})
    oracle = summary.get("oracle_bridge", {})
    full_acc = float(full.get("selected_bridge_accuracy", 0.0) or 0.0)
    static_acc = float(static.get("selected_bridge_accuracy", 0.0) or 0.0)
    dynamic_acc = float(dynamic.get("selected_bridge_accuracy", 0.0) or 0.0)
    random_acc = float(random_budget.get("selected_bridge_accuracy", 0.0) or 0.0)
    oracle_acc = float(oracle.get("selected_bridge_accuracy", 1.0) or 1.0)
    full_tokens = int(full.get("input_tokens", 0) or 0)
    static_tokens = int(static.get("input_tokens", 0) or 0)
    dynamic_tokens = int(dynamic.get("input_tokens", 0) or 0)
    random_tokens = int(random_budget.get("input_tokens", 0) or 0)
    relevant_settings = ["full_chain_answer", "discriminative_chain_static", receiver_setting, "random_same_budget"]
    parse_fail_total = sum(int(summary.get(setting, {}).get("parse_fail", 0) or 0) for setting in relevant_settings)
    reveal_counts: dict[str, int] = {}
    need_types: dict[str, int] = {}
    reveal_events = 0
    dynamic_rows = 0
    full_fallback_count = 0
    within_sample_dynamic_count = 0
    initial_packet_tokens = 0
    requested_extra_tokens = 0
    requested_candidate_count = 0
    candidate_count = 0
    visible_fragment_count = 0
    pattern_counts: dict[str, int] = {}
    threshold_values: set[float] = set()
    high_confidence_threshold_values: set[float] = set()
    confidence_threshold_values: set[float] = set()
    stopped_after_initial_count = 0
    for row in rows:
        if row.get("setting") != receiver_setting:
            continue
        dynamic_rows += 1
        need = row.get("selector_need") if isinstance(row.get("selector_need"), dict) else {}
        if "receiver_reveal_gap_threshold" in need:
            threshold_values.add(float(need.get("receiver_reveal_gap_threshold") or 0.0))
        if "receiver_cheap_high_confidence_threshold" in need:
            high_confidence_threshold_values.add(float(need.get("receiver_cheap_high_confidence_threshold") or 0.0))
        if "receiver_cheap_confidence_threshold" in need:
            confidence_threshold_values.add(float(need.get("receiver_cheap_confidence_threshold") or 0.0))
        if bool(need.get("stopped_after_initial")):
            stopped_after_initial_count += 1
        disagreement = str(need.get("disagreement_type") or "none")
        need_types[disagreement] = need_types.get(disagreement, 0) + 1
        missing = need.get("missing_fragments") if isinstance(need.get("missing_fragments"), dict) else {}
        if missing:
            reveal_events += 1
        if bool(need.get("full_fallback")):
            full_fallback_count += 1
        if bool(need.get("within_sample_dynamic")):
            within_sample_dynamic_count += 1
        initial_packet_tokens += int(need.get("initial_packet_tokens", 0) or 0)
        requested_extra_tokens += int(need.get("requested_extra_tokens", 0) or 0)
        requested_candidate_count += int(need.get("requested_candidate_count", 0) or 0)
        candidate_count += int(need.get("candidate_count", 0) or 0)
        visible_fragment_count += int(need.get("visible_fragment_count", 0) or 0)
        pattern_key = json.dumps(need.get("fragment_pattern", {}), sort_keys=True, ensure_ascii=False)
        pattern_counts[pattern_key] = pattern_counts.get(pattern_key, 0) + 1
        for values in missing.values():
            if not isinstance(values, list):
                continue
            for fragment_type in values:
                key = str(fragment_type)
                reveal_counts[key] = reveal_counts.get(key, 0) + 1
    request_rate = reveal_events / dynamic_rows if dynamic_rows else 0.0
    avg_fragments_per_candidate = visible_fragment_count / candidate_count if candidate_count else 0.0
    full_fallback_rate = full_fallback_count / dynamic_rows if dynamic_rows else 0.0
    within_sample_dynamic_rate = within_sample_dynamic_count / dynamic_rows if dynamic_rows else 0.0
    pattern_variation_rate = (
        1.0 - (max(pattern_counts.values()) / dynamic_rows)
        if dynamic_rows and pattern_counts else 0.0
    )
    quality_pass = dynamic_acc >= full_acc - 0.02
    cost_pass = ((full_tokens - dynamic_tokens) / full_tokens if full_tokens else 0.0) >= 0.40
    dynamicity_pass = within_sample_dynamic_rate >= 0.20
    random_control_pass = dynamic_acc > random_acc
    stability_pass = (
        parse_fail_total == 0
        and _candidate_counts_identical_for_settings(rows, relevant_settings)
        and _gold_in_topk_identical(summary, relevant_settings)
    )
    transitions = _setting_correctness_transitions(rows, "discriminative_chain_static", receiver_setting)
    return {
        "quality_pass": quality_pass,
        "cost_pass": cost_pass,
        "dynamicity_pass": dynamicity_pass,
        "random_control_pass": random_control_pass,
        "stability_pass": stability_pass,
        "p4_pass": quality_pass and cost_pass and dynamicity_pass and stability_pass,
        "receiver_setting": receiver_setting,
        "full_accuracy": full_acc,
        "discriminative_static_accuracy": static_acc,
        "receiver_aware_dynamic_accuracy": dynamic_acc,
        "random_same_budget_accuracy": random_acc,
        "oracle_accuracy": oracle_acc,
        "receiver_reveal_gap_threshold": sorted(threshold_values),
        "receiver_cheap_high_confidence_threshold": sorted(high_confidence_threshold_values),
        "receiver_cheap_confidence_threshold": sorted(confidence_threshold_values),
        "dynamic_vs_full_delta": dynamic_acc - full_acc,
        "dynamic_vs_static_delta": dynamic_acc - static_acc,
        "dynamic_vs_random_delta": dynamic_acc - random_acc,
        "oracle_gap": oracle_acc - dynamic_acc,
        "full_input_tokens": full_tokens,
        "dynamic_input_tokens": dynamic_tokens,
        "static_input_tokens": static_tokens,
        "random_same_budget_input_tokens": random_tokens,
        "dynamic_token_saving_vs_full": (
            (full_tokens - dynamic_tokens) / full_tokens if full_tokens else 0.0
        ),
        "dynamic_token_saving_vs_static": (
            (static_tokens - dynamic_tokens) / static_tokens if static_tokens else 0.0
        ),
        "dynamic_token_saving_vs_random": (
            (random_tokens - dynamic_tokens) / random_tokens if random_tokens else 0.0
        ),
        "parse_fail_total": parse_fail_total,
        "receiver_need_types": need_types,
        "initial_packet_tokens": initial_packet_tokens,
        "requested_extra_tokens": requested_extra_tokens,
        "request_rate": request_rate,
        "stopped_after_initial_rate": stopped_after_initial_count / dynamic_rows if dynamic_rows else 0.0,
        "avg_fragments_per_candidate": avg_fragments_per_candidate,
        "full_fallback_rate": full_fallback_rate,
        "within_sample_dynamic_rate": within_sample_dynamic_rate,
        "pattern_variation_rate": pattern_variation_rate,
        "targeted_reveal_events": reveal_events,
        "targeted_reveal_counts": reveal_counts,
        "static_correct_to_ra_wrong": transitions["left_correct_right_wrong"],
        "ra_correct_static_wrong": transitions["right_correct_left_wrong"],
        "static_ra_both_correct": transitions["both_correct"],
        "static_ra_both_wrong": transitions["both_wrong"],
    }


def _candidate_counts_identical(rows: list[dict[str, Any]], left: str, right: str) -> bool:
    by_sample: dict[str, dict[str, int]] = {}
    for row in rows:
        if row.get("setting") in {left, right}:
            by_sample.setdefault(str(row.get("sample_id", "")), {})[str(row.get("setting"))] = int(row.get("candidate_count", 0) or 0)
    return all(values.get(left) == values.get(right) for values in by_sample.values())


def _setting_correctness_transitions(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, int]:
    by_sample: dict[str, dict[str, bool]] = {}
    for row in rows:
        setting = str(row.get("setting", ""))
        if setting in {left, right}:
            by_sample.setdefault(str(row.get("sample_id", "")), {})[setting] = bool(row.get("selected_bridge_correct"))
    counts = {
        "both_correct": 0,
        "both_wrong": 0,
        "left_correct_right_wrong": 0,
        "right_correct_left_wrong": 0,
    }
    for values in by_sample.values():
        if left not in values or right not in values:
            continue
        left_ok = values[left]
        right_ok = values[right]
        if left_ok and right_ok:
            counts["both_correct"] += 1
        elif not left_ok and not right_ok:
            counts["both_wrong"] += 1
        elif left_ok and not right_ok:
            counts["left_correct_right_wrong"] += 1
        else:
            counts["right_correct_left_wrong"] += 1
    return counts


def _candidate_counts_identical_for_settings(rows: list[dict[str, Any]], settings: list[str]) -> bool:
    wanted = set(settings)
    by_sample: dict[str, dict[str, int]] = {}
    for row in rows:
        setting = str(row.get("setting", ""))
        if setting in wanted:
            by_sample.setdefault(str(row.get("sample_id", "")), {})[setting] = int(row.get("candidate_count", 0) or 0)
    for values in by_sample.values():
        counts = [values.get(setting) for setting in settings if setting in values]
        if counts and any(count != counts[0] for count in counts):
            return False
    return True


def _gold_in_topk_identical(summary: dict[str, dict[str, float]], settings: list[str]) -> bool:
    rates = [
        float(summary.get(setting, {}).get("gold_in_topk_rate", 0.0) or 0.0)
        for setting in settings
        if setting in summary
    ]
    return not rates or all(rate == rates[0] for rate in rates)


def _mean(values: Any) -> float:
    items = [float(bool(value)) for value in values]
    return sum(items) / len(items) if items else 0.0


def _terms(text: str) -> set[str]:
    stop = {"the", "a", "an", "of", "in", "on", "and", "or", "to", "was", "is", "are", "were", "for", "with"}
    return {term for term in re.findall(r"[a-z0-9]+", text.lower()) if len(term) > 2 and term not in stop}


def _json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


def _gold_bridge_values(row: dict[str, Any]) -> set[str]:
    values = set()
    for link in row.get("evidence_links", []):
        if isinstance(link, dict):
            value = str(link.get("entity", link.get("title", ""))).strip()
            if value:
                values.add(value)
    return values


if __name__ == "__main__":
    main()
