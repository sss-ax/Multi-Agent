"""Offline HotpotQA verifier/repair diagnostics from workflow eval logs.

Gold labels are used only after the workflow run has finished.  The script
audits every solver result version, not just the final answer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_domain_workflow import hotpot_f1, hotpot_normalize, read  # noqa: E402
from workflow_runtime.hotpot_bridge import (
    bridge_candidates_from_values,
    bridge_matches_any,
    normalize_bridge,
)

BRIDGE_DELIVERY_STAGES = (
    "offline_top8",
    "raw_pool",
    "ranked_top8",
    "deduplicated_top8",
    "visibility_filtered",
    "budget_filtered",
    "repair_target_filtered",
    "serialized_packet",
    "solver_visible",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()

    report = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    records = list(report.get("records", []))
    rows = read(Path(args.data_path), "hotpotqa", args.limit or len(records))
    rows_by_sample = {
        f"eval_hotpotqa_{index}": row
        for index, row in enumerate(rows)
    }

    candidate_rows: list[dict[str, Any]] = []
    bridge_coverage_rows: list[dict[str, Any]] = []
    bridge_delivery_rows: list[dict[str, Any]] = []
    setting_rows: list[dict[str, Any]] = []
    trajectory_counts: Counter[str] = Counter()
    final_counts: Counter[str] = Counter()
    repair_delta_counts: Counter[str] = Counter()

    for record in records:
        sample_id = str(record.get("sample_id", ""))
        row = rows_by_sample.get(sample_id)
        if row is None:
            continue
        gold_answer = str(row.get("gold_answer", ""))
        gold_fact_ids = _gold_fact_ids(row)
        gold_bridges = _gold_bridge_values(row)
        offline_candidates = _offline_bridge_candidates(row)
        bridge_coverage_rows.append({
            "sample_id": sample_id,
            "gold_bridges": sorted(gold_bridges),
            "offline_bridge_candidates": offline_candidates,
            **_coverage_at_k(offline_candidates, gold_bridges, prefix="offline_bridge_coverage"),
        })

        solver_versions = _solver_versions(record)
        critic_events = _critic_events(record)
        for index, candidate in enumerate(solver_versions):
            next_critic = _next_critic_after(critic_events, candidate["log_index"])
            value = candidate["value"]
            answer = str(value.get("answer", "") if isinstance(value, dict) else value)
            correct = hotpot_normalize(answer) == hotpot_normalize(gold_answer)
            cited = _string_list(value.get("cited_fact_ids") if isinstance(value, dict) else None)
            overlap = len(set(cited) & gold_fact_ids)
            bridge = str(value.get("bridge_entity", "") if isinstance(value, dict) else "")
            repair_packet = _repair_packet(next_critic.get("action", {}) if next_critic else {})
            runtime_candidates = _string_list(repair_packet.get("bridge_candidates") if isinstance(repair_packet, dict) else None)
            repair_stage_event = _next_repair_bridge_stage_event(
                record,
                int(next_critic["log_index"]) if next_critic else None,
            )
            delivery = _bridge_delivery_row(
                sample_id=sample_id,
                candidate_index=index,
                solver_log_index=int(candidate["log_index"]),
                critic_log_index=int(next_critic["log_index"]) if next_critic else None,
                repair_log_index=repair_stage_event.get("log_index"),
                gold_bridges=gold_bridges,
                offline_candidates=offline_candidates,
                runtime_candidates=runtime_candidates,
                stage_candidates=repair_stage_event.get("stage_candidates"),
            )
            bridge_delivery_rows.append(delivery)
            setting_rows.extend(_setting_rows(
                sample_id=sample_id,
                candidate_index=index,
                gold_bridges=gold_bridges,
                offline_candidates=offline_candidates,
                current_candidates=runtime_candidates,
            ))
            candidate_rows.append({
                "sample_id": sample_id,
                "candidate_index": index,
                "mode": candidate["mode"],
                "answer": answer,
                "bridge_entity": bridge,
                "cited_fact_ids": cited,
                "offline_answer_correct": correct,
                "offline_f1": hotpot_f1(answer, gold_answer),
                "offline_supporting_fact_overlap": overlap,
                "offline_supporting_fact_recall": overlap / len(gold_fact_ids) if gold_fact_ids else 0.0,
                "offline_bridge_correct": _matches_any(bridge, gold_bridges),
                "critic_status": (next_critic.get("action", {}) if next_critic else {}).get("status", ""),
                "critic_error_type": (next_critic.get("action", {}) if next_critic else {}).get("error_type", ""),
                "critic_log_index": next_critic.get("log_index") if next_critic else None,
                    "repair_log_index": repair_stage_event.get("log_index"),
                    "event_id": delivery["event_id"],
                "runtime_bridge_candidates": runtime_candidates,
                **_coverage_at_k(runtime_candidates, gold_bridges, prefix="runtime_bridge_coverage"),
                **{
                    key: value
                    for key, value in delivery.items()
                    if key.startswith("gold_in_") or key.startswith("bridge_delivery_recall")
                },
            })

        for prev, nxt in zip(solver_versions, solver_versions[1:]):
            prev_correct = _candidate_correct(prev["value"], gold_answer)
            next_correct = _candidate_correct(nxt["value"], gold_answer)
            transition = ("C" if prev_correct else "W") + "->" + ("C" if next_correct else "W")
            trajectory_counts[transition] += 1
            repair_delta_counts[_repair_delta_label(prev["value"], nxt["value"])] += 1
        if solver_versions:
            initial_correct = _candidate_correct(solver_versions[0]["value"], gold_answer)
            final_correct = _candidate_correct(solver_versions[-1]["value"], gold_answer)
            final_counts[("C" if initial_correct else "W") + "->" + ("C" if final_correct else "W")] += 1

    confusion = _critic_confusion(candidate_rows)
    summary = {
        "eval_json": args.eval_json,
        "data_path": args.data_path,
        "record_count": len(records),
        "candidate_count": len(candidate_rows),
        "critic_confusion": confusion,
        "critic_precision": _ratio(confusion["TP"], confusion["TP"] + confusion["FP"]),
        "critic_recall": _ratio(confusion["TP"], confusion["TP"] + confusion["FN"]),
        "trajectory_transition_counts": dict(sorted(trajectory_counts.items())),
        "initial_to_last_candidate_counts": dict(sorted(final_counts.items())),
        "repair_delta_counts": dict(sorted(repair_delta_counts.items())),
        "bridge_candidate_coverage": _summarize_coverage(bridge_coverage_rows, "offline_bridge_coverage"),
        "runtime_bridge_candidate_coverage": _summarize_coverage(candidate_rows, "runtime_bridge_coverage"),
        "bridge_delivery_recall": _summarize_delivery_recall(candidate_rows),
        "bridge_stage_preservation": _summarize_stage_preservation(bridge_delivery_rows),
        "bridge_delivery_settings": _summarize_setting_rows(setting_rows),
        "bridge_delivery_stage_observability": {
            "offline_top8": "offline_reconstructed_from_dataset",
            "raw_pool": "runtime_observed_when_hotpot_repair_packet_audit_exists",
            "ranked_top8": "runtime_observed_when_hotpot_repair_packet_audit_exists",
            "deduplicated_top8": "runtime_observed_when_hotpot_repair_packet_audit_exists",
            "visibility_filtered": "runtime_observed_when_hotpot_repair_packet_audit_exists",
            "budget_filtered": "runtime_observed_when_hotpot_repair_packet_audit_exists",
            "repair_target_filtered": "runtime_observed_when_hotpot_repair_packet_audit_exists",
            "serialized_packet": "runtime_observed_when_hotpot_repair_packet_audit_exists_else_critic_repair_packet",
            "solver_visible": "runtime_observed_when_hotpot_repair_packet_audit_exists",
        },
        "candidate_error_type_counts": dict(Counter(row["critic_error_type"] or "-" for row in candidate_rows)),
        "candidates": candidate_rows,
        "bridge_coverage_rows": bridge_coverage_rows,
        "bridge_delivery_rows": bridge_delivery_rows,
        "bridge_delivery_setting_rows": setting_rows,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        k: v for k, v in summary.items()
        if k not in {"candidates", "bridge_coverage_rows", "bridge_delivery_rows", "bridge_delivery_setting_rows"}
    }, ensure_ascii=False, indent=2))


def _solver_versions(record: dict[str, Any]) -> list[dict[str, Any]]:
    versions: list[dict[str, Any]] = []
    logs = record.get("runtime_summary", {}).get("records", [])
    for index, log in enumerate(logs):
        action = log.get("action", {})
        if log.get("role") == "solver" and isinstance(action, dict) and action.get("op") == "set_result":
            versions.append({
                "log_index": index,
                "mode": log.get("mode", ""),
                "value": action.get("value"),
            })
    return versions


def _critic_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    logs = record.get("runtime_summary", {}).get("records", [])
    for index, log in enumerate(logs):
        action = log.get("action", {})
        if log.get("role") == "critic" and isinstance(action, dict) and action.get("op") == "verify":
            events.append({"log_index": index, "action": action})
    return events


def _next_critic_after(events: list[dict[str, Any]], log_index: int) -> dict[str, Any] | None:
    for event in events:
        if int(event["log_index"]) > log_index:
            return event
    return None


def _next_repair_bridge_stage_event(record: dict[str, Any], critic_log_index: int | None) -> dict[str, Any]:
    if critic_log_index is None:
        return {}
    logs = record.get("runtime_summary", {}).get("records", [])
    for index, log in enumerate(logs):
        if index <= critic_log_index:
            continue
        if log.get("role") != "solver" or log.get("mode") != "repair":
            continue
        for container_key in ("generation", "telemetry"):
            container = log.get(container_key)
            if not isinstance(container, dict):
                continue
            audit = container.get("hotpot_repair_packet_audit")
            if not isinstance(audit, dict):
                continue
            stages = audit.get("bridge_candidate_stages")
            if isinstance(stages, dict):
                return {
                    "log_index": index,
                    "stage_candidates": {
                        stage: _string_list(stages.get(stage))
                        for stage in BRIDGE_DELIVERY_STAGES
                        if stage in stages
                    },
                }
    return {}


def _critic_confusion(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        correct = bool(row["offline_answer_correct"])
        verified = row["critic_status"] == "verified"
        if correct and verified:
            counts["TP"] += 1
        elif not correct and verified:
            counts["FP"] += 1
        elif correct and not verified:
            counts["FN"] += 1
        else:
            counts["TN"] += 1
    return {key: int(counts[key]) for key in ("TP", "FP", "FN", "TN")}


def _candidate_correct(value: Any, gold_answer: str) -> bool:
    answer = value.get("answer") if isinstance(value, dict) else value
    return hotpot_normalize(answer) == hotpot_normalize(gold_answer)


def _repair_delta_label(old: Any, new: Any) -> str:
    if not isinstance(old, dict) or not isinstance(new, dict):
        return "unknown"
    changed = []
    if str(old.get("answer", "")).strip() != str(new.get("answer", "")).strip():
        changed.append("answer")
    if str(old.get("bridge_entity", "")).strip() != str(new.get("bridge_entity", "")).strip():
        changed.append("bridge")
    if _string_list(old.get("cited_fact_ids")) != _string_list(new.get("cited_fact_ids")):
        changed.append("citations")
    return "changed_" + "_".join(changed) if changed else "noop_answer_bridge_citations"


def _repair_packet(action: dict[str, Any]) -> dict[str, Any]:
    reason = action.get("reason")
    if isinstance(reason, str) and reason.strip().startswith("{"):
        try:
            payload = json.loads(reason)
        except json.JSONDecodeError:
            return {}
        packet = payload.get("repair_packet") if isinstance(payload, dict) else None
        return packet if isinstance(packet, dict) else {}
    return {}


def _gold_fact_ids(row: dict[str, Any]) -> set[str]:
    links = {
        (str(link.get("entity", link.get("title", ""))), int(link.get("sent_id", -1)))
        for link in row.get("evidence_links", [])
        if isinstance(link, dict) and str(link.get("entity", link.get("title", "")))
    }
    ids = set()
    for index, fact in enumerate(row.get("supporting_facts", []), start=1):
        if not isinstance(fact, dict):
            continue
        key = (str(fact.get("title", fact.get("entity", ""))), int(fact.get("sent_id", -1)))
        if key in links:
            ids.add(f"supporting_fact_{index}")
    return ids


def _gold_bridge_values(row: dict[str, Any]) -> set[str]:
    values = set()
    for link in row.get("evidence_links", []):
        if isinstance(link, dict):
            value = str(link.get("entity", link.get("title", ""))).strip()
            if value:
                values.add(value)
    return values


def _offline_bridge_candidates(row: dict[str, Any], *, limit: int = 24) -> list[str]:
    values: list[Any] = []
    for entity in row.get("entities", []):
        values.append(entity)
    for fact in row.get("supporting_facts", []):
        if isinstance(fact, dict):
            values.append(fact.get("title", fact.get("entity", "")))
    return bridge_candidates_from_values(values, limit=limit)


def _coverage_at_k(candidates: list[str], gold: set[str], *, prefix: str) -> dict[str, bool]:
    return {
        f"{prefix}@{k}": any(_matches_any(candidate, gold) for candidate in candidates[:k])
        for k in (2, 4, 8)
    }


def _bridge_delivery_row(
    *,
    sample_id: str,
    candidate_index: int,
    solver_log_index: int | None = None,
    critic_log_index: int | None = None,
    repair_log_index: int | None = None,
    gold_bridges: set[str],
    offline_candidates: list[str],
    runtime_candidates: list[str],
    stage_candidates: Any = None,
) -> dict[str, Any]:
    stage_candidates = stage_candidates if isinstance(stage_candidates, dict) else {}
    stage_map: dict[str, list[str]] = {
        "offline_top8": list(offline_candidates[:8]),
    }
    for stage in BRIDGE_DELIVERY_STAGES:
        if stage in stage_candidates:
            stage_map[stage] = _string_list(stage_candidates.get(stage))
    if "serialized_packet" not in stage_map and runtime_candidates:
        stage_map["serialized_packet"] = list(runtime_candidates)
    observed_stages = sorted(stage_candidates)
    event_id = (
        f"{sample_id}:candidate_{candidate_index}:"
        f"solver_{solver_log_index if solver_log_index is not None else 'na'}:"
        f"critic_{critic_log_index if critic_log_index is not None else 'na'}:"
        f"repair_{repair_log_index if repair_log_index is not None else 'na'}"
    )
    row: dict[str, Any] = {
        "event_id": event_id,
        "sample_id": sample_id,
        "candidate_index": candidate_index,
        "solver_log_index": solver_log_index,
        "critic_log_index": critic_log_index,
        "repair_log_index": repair_log_index,
        "gold_bridges": sorted(gold_bridges),
        "stage_observed": observed_stages,
    }
    for stage in BRIDGE_DELIVERY_STAGES:
        observed = stage in stage_map
        candidates = stage_map.get(stage, [])
        present_key = _stage_present_key(stage)
        row[f"{stage}_observed"] = observed
        row[present_key] = _stage_gold_present(candidates, gold_bridges) if observed else None
        row[f"{stage}_candidate_count"] = len(candidates) if observed else None
    row["solver_visible_context_observed"] = bool(row["solver_visible_observed"])
    row["solver_visible_context_note"] = (
        "observed_from_hotpot_repair_packet_audit"
        if row["solver_visible_context_observed"]
        else "not_observed_in_existing_eval_logs"
    )
    for k in (2, 4, 8):
        offline_hit = any(_matches_any(candidate, gold_bridges) for candidate in offline_candidates[:k])
        runtime_hit = any(_matches_any(candidate, gold_bridges) for candidate in runtime_candidates[:k])
        row[f"gold_in_ranked_top{k}"] = offline_hit
        row[f"gold_in_selected_packet_top{k}"] = runtime_hit
        row[f"bridge_delivery_recall@{k}"] = _ratio(int(runtime_hit), int(offline_hit))
    return row


def _setting_rows(
    *,
    sample_id: str,
    candidate_index: int,
    gold_bridges: set[str],
    offline_candidates: list[str],
    current_candidates: list[str],
) -> list[dict[str, Any]]:
    settings = {
        "current": current_candidates,
        "preserved_top8": offline_candidates[:8],
        "oracle_bridge": sorted(gold_bridges),
    }
    rows: list[dict[str, Any]] = []
    for setting, candidates in settings.items():
        row = {
            "sample_id": sample_id,
            "candidate_index": candidate_index,
            "setting": setting,
            "candidates": list(candidates),
            **_coverage_at_k(list(candidates), gold_bridges, prefix="bridge_coverage"),
        }
        rows.append(row)
    return rows


def _summarize_setting_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_setting: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_setting.setdefault(str(row.get("setting", "")), []).append(row)
    return {
        setting: _summarize_coverage(setting_rows, "bridge_coverage")
        for setting, setting_rows in sorted(by_setting.items())
    }


def _summarize_coverage(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    summary = {}
    for k in (2, 4, 8):
        key = f"{prefix}@{k}"
        values = [bool(row.get(key)) for row in rows if key in row]
        summary[key] = sum(values) / len(values) if values else 0.0
    return summary


def _summarize_delivery_recall(rows: list[dict[str, Any]]) -> dict[str, float]:
    summary = {}
    for k in (2, 4, 8):
        offline_key = f"gold_in_ranked_top{k}"
        runtime_key = f"gold_in_selected_packet_top{k}"
        offline_hits = sum(bool(row.get(offline_key)) for row in rows)
        runtime_hits = sum(bool(row.get(runtime_key)) for row in rows if row.get(offline_key))
        summary[f"bridge_delivery_recall@{k}"] = runtime_hits / offline_hits if offline_hits else 0.0
    return summary


def _summarize_stage_preservation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage_keys = {
        stage: _stage_present_key(stage)
        for stage in BRIDGE_DELIVERY_STAGES
    }
    coverage: dict[str, float | None] = {}
    observed_counts: dict[str, int] = {}
    present_counts: dict[str, int] = {}
    for stage, key in stage_keys.items():
        values = [row.get(key) for row in rows if row.get(key) is not None]
        observed_counts[stage] = len(values)
        present_counts[stage] = sum(bool(value) for value in values)
        coverage[stage] = present_counts[stage] / len(values) if values else None

    transitions: dict[str, float | None] = {}
    transition_denominators: dict[str, int] = {}
    transition_retained_counts: dict[str, int] = {}
    for before, after in zip(BRIDGE_DELIVERY_STAGES, BRIDGE_DELIVERY_STAGES[1:]):
        before_key = stage_keys[before]
        after_key = stage_keys[after]
        eligible = [
            row for row in rows
            if row.get(before_key) is True and row.get(after_key) is not None
        ]
        retained = sum(row.get(after_key) is True for row in eligible)
        name = f"{after}_over_{before}"
        transitions[name] = retained / len(eligible) if eligible else None
        transition_denominators[name] = len(eligible)
        transition_retained_counts[name] = retained
    return {
        "coverage": coverage,
        "present_counts": present_counts,
        "observed_counts": observed_counts,
        "observation_rate": {
            stage: observed_counts[stage] / len(rows) if rows else 0.0
            for stage in stage_keys
        },
        "transition_recall": transitions,
        "transition_denominators": transition_denominators,
        "transition_retained_counts": transition_retained_counts,
        "offline_to_runtime_raw_recall": transitions.get("raw_pool_over_offline_top8"),
    }


def _stage_gold_present(candidates: list[str] | None, gold_bridges: set[str]) -> bool:
    return any(_matches_any(candidate, gold_bridges) for candidate in (candidates or []))


def _stage_present_key(stage: str) -> str:
    if stage == "ranked_top8":
        return "gold_in_runtime_ranked_top8"
    return f"gold_in_{stage}"


def _matches_any(value: str, gold: set[str]) -> bool:
    return bridge_matches_any(value, gold)


def _norm(value: Any) -> str:
    return normalize_bridge(value)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


if __name__ == "__main__":
    main()
