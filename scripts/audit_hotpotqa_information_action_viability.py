"""Audit whether HotpotQA information actions are ready for policy learning.

This is deliberately an offline audit.  It joins an existing HotpotQA workflow
run (which contains real intermediate solver states and repair attempts) with a
bridge/evidence selection ablation produced by
``run_hotpotqa_bridge_evidence_selection_ablation.py``.

The report separates three questions that are easy to conflate:

1. Did the candidate pool contain a gold bridge/evidence source?
2. Could a small selector choose it, especially on initially-wrong states?
3. Did the existing repair trajectory actually improve the final answer?

No gold label is exposed to a non-oracle selector by this audit.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_domain_workflow import hotpot_f1, hotpot_normalize, read


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    report = audit(
        eval_report=json.loads(Path(args.eval_json).read_text(encoding="utf-8")),
        selection_report=json.loads(Path(args.selection_json).read_text(encoding="utf-8")),
        rows=read(Path(args.data_path), "hotpotqa", args.limit or 10**9),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report.update({
        "eval_json": args.eval_json,
        "selection_json": args.selection_json,
        "data_path": args.data_path,
    })
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def audit(
    *,
    eval_report: dict[str, Any],
    selection_report: dict[str, Any],
    rows: list[dict[str, Any]],
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    rows_by_sample = {f"eval_hotpotqa_{index}": row for index, row in enumerate(rows)}
    rows_by_sample.update({str(row.get("sample_id", "")): row for row in rows})

    trajectory_rows = []
    for record in eval_report.get("records", []):
        sample_id = str(record.get("sample_id", ""))
        row = rows_by_sample.get(sample_id)
        if row is None:
            continue
        answers = _solver_answers(record)
        if not answers:
            continue
        gold = row.get("gold_answer", "")
        first, last = answers[0], answers[-1]
        first_em = _em(first, gold)
        last_em = _em(last, gold)
        trajectory_rows.append({
            "sample_id": sample_id,
            "gold_answer": gold,
            "first_answer": first,
            "last_answer": last,
            "solver_answer_count": len(answers),
            "first_em": first_em,
            "last_em": last_em,
            "first_f1": hotpot_f1(first, gold),
            "last_f1": hotpot_f1(last, gold),
            "repaired": (not first_em) and last_em,
            "regressed": first_em and (not last_em),
        })

    wrong_ids = {item["sample_id"] for item in trajectory_rows if not item["first_em"]}
    all_ids = {item["sample_id"] for item in trajectory_rows}
    selection_rows = [
        item for item in selection_report.get("setting_rows", [])
        if str(item.get("sample_id", "")) in all_ids
    ]
    selection_all = _selection_summary(
        selection_rows,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    selection_wrong = _selection_summary(
        [item for item in selection_rows if str(item.get("sample_id", "")) in wrong_ids],
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    trajectory = _trajectory_summary(trajectory_rows)
    oracle_wrong = selection_wrong.get("oracle_bridge", {}).get("accuracy", 0.0)
    deployable = _best_deployable(selection_wrong)
    deployable_accuracy = float(deployable.get("accuracy", 0.0)) if deployable else 0.0
    random_accuracy = float(selection_wrong.get("random_same_budget", {}).get("accuracy", 0.0))
    paired_delta = _paired_selection_delta(
        selection_rows,
        sample_ids=wrong_ids,
        setting=str(deployable.get("setting", "")),
        baseline="random_same_budget",
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    repair_uplift = float(trajectory.get("last_em", 0.0)) - float(trajectory.get("first_em", 0.0))

    gates = {
        "candidate_quality_pass": oracle_wrong >= 0.90,
        "selection_learnability_pass": (
            deployable_accuracy >= 0.60
            and paired_delta.get("delta", 0.0) > 0.0
            and paired_delta.get("bootstrap_95ci", [0.0])[0] > 0.0
        ),
        "realized_reward_pass": (
            trajectory.get("wrong_first_repair_rate", 0.0) > 0.0 and repair_uplift > 0.0
        ),
    }
    gates["rl_ready"] = all(gates.values())
    priorities = []
    if not gates["candidate_quality_pass"]:
        priorities.append("improve_candidate_recall")
    if not gates["selection_learnability_pass"]:
        priorities.append("improve_state_representation_and_selector")
    if not gates["realized_reward_pass"]:
        priorities.append("fix_evidence_to_answer_transition_and_stop_policy")
    recommendation = (
        "before_rl:" + ",".join(priorities)
        if priorities else
        "proceed_to_policy_learning_with_measured_answer_delta_reward"
    )

    return {
        "sample_count": len(trajectory_rows),
        "initially_wrong_count": len(wrong_ids),
        "trajectory": trajectory,
        "selection_all_states": selection_all,
        "selection_initially_wrong_states": selection_wrong,
        "best_deployable_selector_on_wrong_states": deployable,
        "best_deployable_vs_random_on_wrong_states": paired_delta,
        "gates": gates,
        "recommendation": recommendation,
        "notes": [
            "Selection correctness is gold-bridge accuracy, not final-answer reward.",
            "Trajectory reward uses actual first and last solver answers from the workflow log.",
            "Bootstrap intervals resample samples, preserving paired method outcomes.",
        ],
        "trajectory_rows": trajectory_rows,
    }


def _solver_answers(record: dict[str, Any]) -> list[str]:
    answers = []
    for event in record.get("runtime_summary", {}).get("records", []):
        if event.get("role") != "solver":
            continue
        action = event.get("action", {})
        if not isinstance(action, dict) or action.get("op") != "answer_with_evidence":
            continue
        answer = action.get("answer")
        if answer is not None and str(answer).strip():
            answers.append(str(answer).strip())
    return answers


def _em(prediction: Any, gold: Any) -> bool:
    return hotpot_normalize(prediction) == hotpot_normalize(gold)


def _trajectory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    wrong = [row for row in rows if not row["first_em"]]
    return {
        "count": count,
        "first_em": _mean(row["first_em"] for row in rows),
        "last_em": _mean(row["last_em"] for row in rows),
        "em_delta": _mean(row["last_em"] for row in rows) - _mean(row["first_em"] for row in rows),
        "first_f1": _mean(row["first_f1"] for row in rows),
        "last_f1": _mean(row["last_f1"] for row in rows),
        "f1_delta": _mean(row["last_f1"] for row in rows) - _mean(row["first_f1"] for row in rows),
        "initially_wrong_count": len(wrong),
        "repaired_count": sum(bool(row["repaired"]) for row in rows),
        "wrong_first_repair_rate": _mean(row["repaired"] for row in wrong),
        "regressed_count": sum(bool(row["regressed"]) for row in rows),
        "regression_rate": _mean(row["regressed"] for row in rows),
    }


def _selection_summary(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    by_setting: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_setting.setdefault(str(row.get("setting", "")), []).append(row)
    result = {}
    for setting, items in sorted(by_setting.items()):
        values = [float(bool(item.get("selected_bridge_correct"))) for item in items]
        result[setting] = {
            "count": len(items),
            "accuracy": _mean(values),
            "accuracy_bootstrap_95ci": _bootstrap_ci(values, bootstrap_samples, seed),
            "gold_in_topk_rate": _mean(bool(item.get("gold_in_topk")) for item in items),
            "parse_fail_rate": _mean(bool(item.get("parse_fail")) for item in items),
            "input_tokens": sum(int(item.get("input_tokens", 0) or 0) for item in items),
            "request_rate": _request_rate(items),
            "stop_rate": _stop_rate(items),
        }
    return result


def _best_deployable(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        {"setting": setting, **metrics}
        for setting, metrics in summary.items()
        if setting not in {"oracle_bridge", "random_same_budget"}
    ]
    return max(candidates, key=lambda item: (item["accuracy"], -item["input_tokens"]), default={})


def _paired_selection_delta(
    rows: list[dict[str, Any]],
    *,
    sample_ids: set[str],
    setting: str,
    baseline: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    by_setting: dict[str, dict[str, float]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in sample_ids:
            continue
        by_setting.setdefault(str(row.get("setting", "")), {})[sample_id] = float(
            bool(row.get("selected_bridge_correct"))
        )
    treatment = by_setting.get(setting, {})
    control = by_setting.get(baseline, {})
    common = sorted(set(treatment) & set(control))
    differences = [treatment[sample_id] - control[sample_id] for sample_id in common]
    return {
        "setting": setting,
        "baseline": baseline,
        "paired_count": len(differences),
        "delta": _mean(differences),
        "bootstrap_95ci": _bootstrap_ci(differences, bootstrap_samples, seed),
        "treatment_only_wins": sum(value > 0 for value in differences),
        "baseline_only_wins": sum(value < 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
    }


def _request_rate(items: list[dict[str, Any]]) -> float | None:
    diagnosed = [item.get("selector_need") for item in items if item.get("selector_need")]
    if not diagnosed:
        return None
    return _mean(
        bool(need.get("requested_candidate_count", 0)) or bool(need.get("missing_fragments"))
        for need in diagnosed
    )


def _stop_rate(items: list[dict[str, Any]]) -> float | None:
    diagnosed = [item.get("selector_need") for item in items if item.get("selector_need")]
    if not diagnosed or not any("stopped_after_initial" in need for need in diagnosed):
        return None
    return _mean(bool(need.get("stopped_after_initial")) for need in diagnosed)


def _bootstrap_ci(values: list[float], samples: int, seed: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    estimates = []
    for _ in range(max(1, samples)):
        estimates.append(sum(rng.choice(values) for _ in values) / len(values))
    estimates.sort()
    low = estimates[int(0.025 * (len(estimates) - 1))]
    high = estimates[int(0.975 * (len(estimates) - 1))]
    return [low, high]


def _mean(values: Iterable[Any]) -> float:
    materialized = [float(value) for value in values]
    return sum(materialized) / len(materialized) if materialized else 0.0


if __name__ == "__main__":
    main()
