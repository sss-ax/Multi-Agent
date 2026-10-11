"""HotpotQA bridge-selection ablations over delivered bridge candidates.

This script intentionally measures bridge selection, not full QA accuracy.  It
uses gold labels only offline after a workflow run has completed.
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
from workflow_runtime.hotpot_bridge import bridge_matches_any, deduplicate_bridge_candidates  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    eval_report = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    records = list(eval_report.get("records", []))
    rows = read(Path(args.data_path), "hotpotqa", args.limit or len(records))
    rows_by_sample = {f"eval_hotpotqa_{index}": row for index, row in enumerate(rows)}
    events = [
        _bridge_selection_event(record, rows_by_sample.get(str(record.get("sample_id", ""))))
        for record in records
    ]
    events = [event for event in events if event]

    model = None
    if args.model_path:
        from workflow_runtime.model_backend import TransformersModel

        model = TransformersModel(args.model_path, max_new_tokens=args.max_new_tokens)

    setting_rows: list[dict[str, Any]] = []
    for event in events:
        candidates = list(event["candidates"])[:8]
        gold = set(event["gold_bridges"])
        setting_rows.append(_setting_row(event, "current_repair", event.get("current_bridge", "")))
        setting_rows.append(_setting_row(event, "bridge_top8_random_select", _random_choice(candidates, args.seed, event["sample_id"])))
        setting_rows.append(_setting_row(event, "oracle_bridge_select", _oracle_choice(candidates, gold)))
        if model is not None:
            selected, raw, metrics = _critic_rerank(model, event, candidates)
            row = _setting_row(event, "bridge_top8_critic_rerank", selected)
            row.update({
                "critic_raw_tail": raw[-1200:],
                "critic_input_tokens": metrics.get("physical_input_tokens", 0),
                "critic_output_tokens": metrics.get("output_tokens", 0),
            })
            setting_rows.append(row)

    summary = {
        "eval_json": args.eval_json,
        "data_path": args.data_path,
        "event_count": len(events),
        "current_repair_exact_match": eval_report.get("exact_match", eval_report.get("accuracy_or_pass_at_1")),
        "current_repair_f1": eval_report.get("f1"),
        "settings": _summarize_settings(setting_rows),
        "events": events,
        "setting_rows": setting_rows,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"events", "setting_rows"}}, ensure_ascii=False, indent=2))


def _bridge_selection_event(record: dict[str, Any], row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    gold_bridges = _gold_bridge_values(row)
    logs = record.get("runtime_summary", {}).get("records", [])
    candidates: list[str] = []
    for log in logs:
        if log.get("role") != "solver" or log.get("mode") != "repair":
            continue
        stages = _bridge_stage_candidates(log)
        for stage in ("solver_visible", "serialized_packet", "repair_target_filtered", "ranked_top8"):
            candidates = deduplicate_bridge_candidates(stages.get(stage, []), limit=8)
            if candidates:
                break
        if candidates:
            break
    if not candidates:
        return {}
    final_value = _final_solver_result(logs)
    current_bridge = str(final_value.get("bridge_entity", "") if isinstance(final_value, dict) else "")
    return {
        "sample_id": str(record.get("sample_id", "")),
        "question": row.get("question", ""),
        "gold_answer": row.get("gold_answer", ""),
        "gold_bridges": sorted(gold_bridges),
        "candidates": candidates,
        "gold_in_top8": any(bridge_matches_any(candidate, gold_bridges) for candidate in candidates[:8]),
        "current_bridge": current_bridge,
    }


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


def _final_solver_result(logs: list[dict[str, Any]]) -> Any:
    value = None
    for log in logs:
        action = log.get("action", {})
        if log.get("role") == "solver" and isinstance(action, dict) and action.get("op") == "set_result":
            value = action.get("value")
    return value


def _setting_row(event: dict[str, Any], setting: str, selected: str) -> dict[str, Any]:
    gold = set(event["gold_bridges"])
    return {
        "sample_id": event["sample_id"],
        "setting": setting,
        "selected_bridge": selected,
        "selected_bridge_correct": bridge_matches_any(selected, gold),
        "gold_in_top8": bool(event["gold_in_top8"]),
    }


def _random_choice(candidates: list[str], seed: int, sample_id: str) -> str:
    if not candidates:
        return ""
    rng = random.Random(f"{seed}:{sample_id}")
    return rng.choice(candidates)


def _oracle_choice(candidates: list[str], gold: set[str]) -> str:
    for candidate in candidates:
        if bridge_matches_any(candidate, gold):
            return candidate
    return candidates[0] if candidates else ""


def _critic_rerank(model: Any, event: dict[str, Any], candidates: list[str]) -> tuple[str, str, dict[str, Any]]:
    numbered = "\n".join(f"{index + 1}. {candidate}" for index, candidate in enumerate(candidates))
    prompt = (
        "Select the single best HotpotQA bridge entity/title for answering the question.\n"
        "Return JSON only: {\"choice\": 1, \"bridge\": \"...\"}.\n\n"
        f"Question:\n{event['question']}\n\n"
        f"Candidate bridges:\n{numbered}\n"
    )
    request = type("Request", (), {"prompt": prompt, "session_prompt": prompt})()
    raw = str(model(request))
    metrics = {}
    snapshot = model.session_snapshot("bridge_rerank") if hasattr(model, "session_snapshot") else {}
    if isinstance(snapshot, dict):
        metrics = dict(snapshot.get("last_call_metrics", {}) or {})
    selected = _parse_choice(raw, candidates)
    return selected, raw, metrics


def _parse_choice(raw: str, candidates: list[str]) -> str:
    try:
        payload = json.loads(_json_object(raw))
        choice = int(payload.get("choice", 0))
        if 1 <= choice <= len(candidates):
            return candidates[choice - 1]
        bridge = str(payload.get("bridge", "")).strip()
        if bridge:
            return bridge
    except Exception:
        pass
    match = re.search(r"\b([1-9])\b", raw)
    if match:
        choice = int(match.group(1))
        if 1 <= choice <= len(candidates):
            return candidates[choice - 1]
    return candidates[0] if candidates else ""


def _json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


def _summarize_settings(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_setting: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_setting.setdefault(str(row["setting"]), []).append(row)
    return {
        setting: {
            "count": float(len(items)),
            "selected_bridge_accuracy": _mean(bool(item["selected_bridge_correct"]) for item in items),
            "gold_in_top8_rate": _mean(bool(item["gold_in_top8"]) for item in items),
        }
        for setting, items in sorted(by_setting.items())
    }


def _mean(values: Any) -> float:
    items = [float(bool(value)) for value in values]
    return sum(items) / len(items) if items else 0.0


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
