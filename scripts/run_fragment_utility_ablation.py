"""Run paired fragment utility ablations over one benchmark split.

The script evaluates the same ordered samples under:
core_only, selected single-fragment restorations, and send_all.  It then
computes sample-level utility:

    u_i(f) = Correct_i(core + f) - Correct_i(core)

with positive/negative/neutral counts and an oracle targeted upper bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARMS = (
    "core_only",
    "support_dependencies",
    "full_plan",
    "result_metadata",
    "calculation_trace",
    "full_feedback",
    "send_all",
)
POLICY_BASELINE_ARMS = (
    "core_only",
    "random_same_budget",
    "static_utility",
    "receiver_aware_heuristic",
    "send_all",
)
POLICY_BY_ARM = {
    "core_only": "fragment_ablation_core_only",
    "support_dependencies": "fragment_ablation_support_dependencies",
    "full_plan": "fragment_ablation_full_plan",
    "result_metadata": "fragment_ablation_result_metadata",
    "calculation_trace": "fragment_ablation_calculation_trace",
    "full_feedback": "fragment_ablation_full_feedback",
    "send_all": "send_all",
    "random_same_budget": "random_same_budget",
    "static_utility": "static_utility",
    "receiver_aware_heuristic": "receiver_aware_heuristic",
}


def fragment_arms(arms: list[str] | tuple[str, ...]) -> list[str]:
    return [arm for arm in arms if arm not in {"core_only", "send_all"}]


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_ablation(reports_by_arm: dict[str, dict[str, Any]], *, domain: str = "", seed: int = 0) -> dict[str, Any]:
    if "core_only" not in reports_by_arm:
        raise ValueError("core_only report is required")
    core_records = _records_by_sample(reports_by_arm["core_only"])
    sample_ids = list(core_records)
    if not sample_ids:
        raise ValueError("core_only report has no records")

    arms: dict[str, Any] = {}
    for arm, report in reports_by_arm.items():
        records = _records_by_sample(report)
        paired_ids = [sample_id for sample_id in sample_ids if sample_id in records]
        positives: list[str] = []
        negatives: list[str] = []
        neutrals: list[str] = []
        for sample_id in paired_ids:
            core_correct = bool(core_records[sample_id].get("correct"))
            arm_correct = bool(records[sample_id].get("correct"))
            delta = int(arm_correct) - int(core_correct)
            if delta > 0:
                positives.append(sample_id)
            elif delta < 0:
                negatives.append(sample_id)
            else:
                neutrals.append(sample_id)
        input_delta = int(report.get("total_input_tokens", 0) or 0) - int(
            reports_by_arm["core_only"].get("total_input_tokens", 0) or 0
        )
        model_delta = int(report.get("total_model_tokens", 0) or 0) - int(
            reports_by_arm["core_only"].get("total_model_tokens", 0) or 0
        )
        sent_delta = int(report.get("graph_delta_sent_tokens", 0) or 0) - int(
            reports_by_arm["core_only"].get("graph_delta_sent_tokens", 0) or 0
        )
        net_gain = len(positives) - len(negatives)
        arms[arm] = {
            "count": len(paired_ids),
            "accuracy_or_pass_at_1": report.get("accuracy_or_pass_at_1"),
            "failure_count": report.get("failure_count", 0),
            "positive": len(positives),
            "negative": len(negatives),
            "neutral": len(neutrals),
            "positive_rate": len(positives) / len(paired_ids) if paired_ids else 0.0,
            "negative_rate": len(negatives) / len(paired_ids) if paired_ids else 0.0,
            "net_gain": net_gain,
            "net_gain_rate": net_gain / len(paired_ids) if paired_ids else 0.0,
            "positive_sample_ids": positives,
            "negative_sample_ids": negatives,
            "input_token_delta_vs_core": input_delta,
            "model_token_delta_vs_core": model_delta,
            "graph_delta_sent_token_delta_vs_core": sent_delta,
            "utility_per_input_token": net_gain / input_delta if input_delta > 0 else 0.0,
            "utility_per_model_token": net_gain / model_delta if model_delta > 0 else 0.0,
            "utility_per_graph_sent_token": net_gain / sent_delta if sent_delta > 0 else 0.0,
        }

    oracle_targeted_correct_ids: list[str] = []
    oracle_targeted_sources: dict[str, list[str]] = {}
    oracle_any_correct_ids: list[str] = []
    oracle_any_sources: dict[str, list[str]] = {}
    train_examples = build_fragment_utility_examples(reports_by_arm, domain=domain)
    utility_tables = build_utility_tables(train_examples)
    policy_baselines = evaluate_policy_baselines(
        reports_by_arm,
        utility_tables=utility_tables,
        domain=domain,
        seed=seed,
    )
    targeted_arms = fragment_arms(tuple(reports_by_arm))
    non_core_arms = [arm for arm in reports_by_arm if arm != "core_only"]
    record_maps = {arm: _records_by_sample(report) for arm, report in reports_by_arm.items()}
    for sample_id in sample_ids:
        targeted_sources = [
            arm for arm in targeted_arms
            if sample_id in record_maps[arm] and bool(record_maps[arm][sample_id].get("correct"))
        ]
        any_sources = [
            arm for arm in non_core_arms
            if sample_id in record_maps[arm] and bool(record_maps[arm][sample_id].get("correct"))
        ]
        if bool(core_records[sample_id].get("correct")) or targeted_sources:
            oracle_targeted_correct_ids.append(sample_id)
        if targeted_sources:
            oracle_targeted_sources[sample_id] = targeted_sources
        if bool(core_records[sample_id].get("correct")) or any_sources:
            oracle_any_correct_ids.append(sample_id)
        if any_sources:
            oracle_any_sources[sample_id] = any_sources
    core_correct = sum(int(bool(record.get("correct"))) for record in core_records.values())
    oracle_targeted_correct = len(oracle_targeted_correct_ids)
    oracle_any_correct = len(oracle_any_correct_ids)
    send_all_records = record_maps.get("send_all", {})
    send_all_correct_core_wrong = [
        sample_id for sample_id in sample_ids
        if sample_id in send_all_records
        and bool(send_all_records[sample_id].get("correct"))
        and not bool(core_records[sample_id].get("correct"))
    ]
    targeted_correct_core_wrong = [
        sample_id for sample_id in sample_ids
        if sample_id in oracle_targeted_sources
        and not bool(core_records[sample_id].get("correct"))
    ]
    send_all_only_rescues = [
        sample_id for sample_id in send_all_correct_core_wrong
        if sample_id not in oracle_targeted_sources
    ]
    positive_examples = [example for example in train_examples if example["utility"] > 0]
    negative_examples = [example for example in train_examples if example["utility"] < 0]
    neutral_examples = [example for example in train_examples if example["utility"] == 0]
    return {
        "count": len(sample_ids),
        "arms": arms,
        "core_correct": core_correct,
        "core_accuracy": core_correct / len(sample_ids),
        "oracle_targeted_correct": oracle_targeted_correct,
        "oracle_targeted_accuracy": oracle_targeted_correct / len(sample_ids),
        "oracle_targeted_gain": oracle_targeted_correct - core_correct,
        "oracle_targeted_gain_rate": (oracle_targeted_correct - core_correct) / len(sample_ids),
        "oracle_targeted_gt_core_only": oracle_targeted_correct > core_correct,
        "oracle_targeted_sources_by_sample": oracle_targeted_sources,
        "oracle_any_correct": oracle_any_correct,
        "oracle_any_accuracy": oracle_any_correct / len(sample_ids),
        "oracle_any_gain": oracle_any_correct - core_correct,
        "oracle_any_sources_by_sample": oracle_any_sources,
        # Backward-compatible aliases now refer to the proper targeted oracle.
        "oracle_correct": oracle_targeted_correct,
        "oracle_accuracy": oracle_targeted_correct / len(sample_ids),
        "oracle_gain": oracle_targeted_correct - core_correct,
        "oracle_gain_rate": (oracle_targeted_correct - core_correct) / len(sample_ids),
        "oracle_sources_by_sample": oracle_targeted_sources,
        "targeted_correct_core_wrong_count": len(targeted_correct_core_wrong),
        "targeted_correct_core_wrong_sample_ids": targeted_correct_core_wrong,
        "send_all_correct_core_wrong_count": len(send_all_correct_core_wrong),
        "send_all_correct_core_wrong_sample_ids": send_all_correct_core_wrong,
        "send_all_only_rescue_count": len(send_all_only_rescues),
        "send_all_only_rescue_sample_ids": send_all_only_rescues,
        "training_example_count": len(train_examples),
        "positive_training_example_count": len(positive_examples),
        "negative_training_example_count": len(negative_examples),
        "neutral_training_example_count": len(neutral_examples),
        "fragment_policy_has_positive_utility": bool(positive_examples),
        "utility_tables": utility_tables,
        "policy_baselines": policy_baselines,
        "heuristic_beats_random_same_budget": (
            policy_baselines["receiver_aware_heuristic"]["accuracy"]
            > policy_baselines["random_same_budget"]["accuracy"]
        ),
    }


def build_fragment_utility_examples(reports_by_arm: dict[str, dict[str, Any]], *, domain: str = "") -> list[dict[str, Any]]:
    if "core_only" not in reports_by_arm:
        raise ValueError("core_only report is required")
    core_report = reports_by_arm["core_only"]
    core_records = _records_by_sample(core_report)
    examples: list[dict[str, Any]] = []
    for arm in fragment_arms(tuple(reports_by_arm)):
        report = reports_by_arm[arm]
        arm_records = _records_by_sample(report)
        input_delta = int(report.get("total_input_tokens", 0) or 0) - int(core_report.get("total_input_tokens", 0) or 0)
        model_delta = int(report.get("total_model_tokens", 0) or 0) - int(core_report.get("total_model_tokens", 0) or 0)
        sent_delta = int(report.get("graph_delta_sent_tokens", 0) or 0) - int(core_report.get("graph_delta_sent_tokens", 0) or 0)
        for sample_id, core_record in core_records.items():
            if sample_id not in arm_records:
                continue
            core_correct = bool(core_record.get("correct"))
            arm_correct = bool(arm_records[sample_id].get("correct"))
            utility = int(arm_correct) - int(core_correct)
            receivers = _selected_receivers(arm_records[sample_id])
            receiver_key = receivers[0] if len(receivers) == 1 else "mixed" if receivers else "none"
            examples.append({
                "sample_id": sample_id,
                "domain": domain or str(report.get("domain", "")),
                "receiver": receiver_key,
                "receivers": receivers,
                "fragment_arm": arm,
                "core_correct": core_correct,
                "fragment_correct": arm_correct,
                "utility": utility,
                "label": "send" if utility > 0 else "drop",
                "harmful": utility < 0,
                "input_token_delta_vs_core": input_delta,
                "model_token_delta_vs_core": model_delta,
                "graph_delta_sent_token_delta_vs_core": sent_delta,
                "utility_per_input_token": utility / input_delta if input_delta > 0 else 0.0,
                "utility_per_model_token": utility / model_delta if model_delta > 0 else 0.0,
                "utility_per_graph_sent_token": utility / sent_delta if sent_delta > 0 else 0.0,
            })
    return examples


def build_utility_tables(examples: list[dict[str, Any]]) -> dict[str, Any]:
    global_groups: dict[str, list[dict[str, Any]]] = {}
    conditional_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for example in examples:
        arm = str(example.get("fragment_arm", ""))
        receiver = str(example.get("receiver", "none"))
        domain = str(example.get("domain", ""))
        global_groups.setdefault(arm, []).append(example)
        conditional_groups.setdefault((arm, receiver, domain), []).append(example)
    return {
        "global": {
            arm: _utility_stats(items)
            for arm, items in sorted(global_groups.items())
        },
        "conditioned": {
            f"{arm}|{receiver}|{domain}": _utility_stats(items)
            for (arm, receiver, domain), items in sorted(conditional_groups.items())
        },
    }


def evaluate_policy_baselines(
    reports_by_arm: dict[str, dict[str, Any]],
    *,
    utility_tables: dict[str, Any],
    domain: str = "",
    seed: int = 0,
) -> dict[str, Any]:
    core_records = _records_by_sample(reports_by_arm["core_only"])
    sample_ids = list(core_records)
    targeted_arms = fragment_arms(tuple(reports_by_arm))
    record_maps = {arm: _records_by_sample(report) for arm, report in reports_by_arm.items()}
    static_arm = _best_arm(utility_tables.get("global", {}), targeted_arms)
    receiver_arm = _best_conditioned_arm(
        utility_tables.get("conditioned", {}),
        targeted_arms,
        domain=domain,
    ) or static_arm
    budget = 1 if receiver_arm else 0
    random_choices = {
        sample_id: _random_arm_same_budget(targeted_arms, sample_id=sample_id, seed=seed)
        for sample_id in sample_ids
    } if budget else {}
    return {
        "core_only": _policy_eval_from_records(core_records, sample_ids, selected_arm="core_only"),
        "random_same_budget": _policy_eval_from_selector(
            record_maps,
            sample_ids,
            lambda sample_id: random_choices.get(sample_id, ""),
            selected_arm="random_same_budget",
        ),
        "static_utility": _policy_eval_from_selector(
            record_maps,
            sample_ids,
            lambda _sample_id: static_arm,
            selected_arm=static_arm or "core_only",
        ),
        "receiver_aware_heuristic": _policy_eval_from_selector(
            record_maps,
            sample_ids,
            lambda _sample_id: receiver_arm,
            selected_arm=receiver_arm or "core_only",
        ),
        "oracle": _policy_eval_oracle(record_maps, sample_ids, targeted_arms),
        "send_all": _policy_eval_from_records(
            record_maps.get("send_all", {}),
            sample_ids,
            selected_arm="send_all",
        ),
        "selected_static_arm": static_arm,
        "selected_receiver_aware_arm": receiver_arm,
        "random_same_budget_selected_arms": random_choices,
    }


def _records_by_sample(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("sample_id")): record
        for record in report.get("records", [])
        if str(record.get("sample_id", "")).strip()
    }


def _selected_receivers(record: dict[str, Any]) -> list[str]:
    receivers: list[str] = []
    for log in record.get("runtime_summary", {}).get("records", []):
        for event in log.get("communication", []):
            if event.get("selected_optional_fragment_ids"):
                receiver = str(event.get("receiver", "")).strip()
                if receiver:
                    receivers.append(receiver)
    return sorted(set(receivers))


def _utility_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(items)
    positive = sum(1 for item in items if int(item.get("utility", 0)) > 0)
    negative = sum(1 for item in items if int(item.get("utility", 0)) < 0)
    neutral = count - positive - negative
    net_gain = positive - negative
    mean_utility = sum(int(item.get("utility", 0)) for item in items) / count if count else 0.0
    return {
        "count": count,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "positive_rate": positive / count if count else 0.0,
        "negative_rate": negative / count if count else 0.0,
        "net_gain": net_gain,
        "mean_utility": mean_utility,
    }


def _best_arm(table: dict[str, dict[str, Any]], arms: list[str]) -> str:
    available = [arm for arm in arms if arm in table]
    if not available:
        return ""
    best = max(
        available,
        key=lambda arm: (
            float(table[arm].get("mean_utility", 0.0)),
            int(table[arm].get("positive", 0)),
            -int(table[arm].get("negative", 0)),
            arm,
        ),
    )
    return best if float(table[best].get("mean_utility", 0.0)) > 0 else ""


def _best_conditioned_arm(table: dict[str, dict[str, Any]], arms: list[str], *, domain: str) -> str:
    candidates: dict[str, dict[str, Any]] = {}
    for key, stats in table.items():
        try:
            arm, _receiver, item_domain = key.split("|", 2)
        except ValueError:
            continue
        if arm in arms and item_domain == domain:
            current = candidates.get(arm)
            if current is None or float(stats.get("mean_utility", 0.0)) > float(current.get("mean_utility", 0.0)):
                candidates[arm] = stats
    return _best_arm(candidates, arms)


def _random_arm_same_budget(arms: list[str], *, sample_id: str, seed: int) -> str:
    if not arms:
        return ""
    key = f"{seed}|{sample_id}|random_same_budget"
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    return arms[value % len(arms)]


def _policy_eval_from_records(records: dict[str, dict[str, Any]], sample_ids: list[str], *, selected_arm: str) -> dict[str, Any]:
    correct_ids = [sample_id for sample_id in sample_ids if bool(records.get(sample_id, {}).get("correct"))]
    return {
        "selected_arm": selected_arm,
        "correct": len(correct_ids),
        "accuracy": len(correct_ids) / len(sample_ids) if sample_ids else 0.0,
        "correct_sample_ids": correct_ids,
    }


def _policy_eval_from_selector(
    record_maps: dict[str, dict[str, dict[str, Any]]],
    sample_ids: list[str],
    selector,
    *,
    selected_arm: str,
) -> dict[str, Any]:
    correct_ids: list[str] = []
    chosen_by_sample: dict[str, str] = {}
    core_records = record_maps.get("core_only", {})
    for sample_id in sample_ids:
        arm = selector(sample_id)
        chosen_by_sample[sample_id] = arm or "core_only"
        records = record_maps.get(arm, core_records) if arm else core_records
        if bool(records.get(sample_id, {}).get("correct")):
            correct_ids.append(sample_id)
    return {
        "selected_arm": selected_arm,
        "correct": len(correct_ids),
        "accuracy": len(correct_ids) / len(sample_ids) if sample_ids else 0.0,
        "correct_sample_ids": correct_ids,
        "chosen_arm_by_sample": chosen_by_sample,
    }


def _policy_eval_oracle(
    record_maps: dict[str, dict[str, dict[str, Any]]],
    sample_ids: list[str],
    targeted_arms: list[str],
) -> dict[str, Any]:
    correct_ids: list[str] = []
    chosen_by_sample: dict[str, str] = {}
    core_records = record_maps.get("core_only", {})
    for sample_id in sample_ids:
        if bool(core_records.get(sample_id, {}).get("correct")):
            correct_ids.append(sample_id)
            chosen_by_sample[sample_id] = "core_only"
            continue
        chosen = ""
        for arm in targeted_arms:
            if bool(record_maps.get(arm, {}).get(sample_id, {}).get("correct")):
                chosen = arm
                break
        if chosen:
            correct_ids.append(sample_id)
            chosen_by_sample[sample_id] = chosen
        else:
            chosen_by_sample[sample_id] = "core_only"
    return {
        "selected_arm": "per_sample_oracle",
        "correct": len(correct_ids),
        "accuracy": len(correct_ids) / len(sample_ids) if sample_ids else 0.0,
        "correct_sample_ids": correct_ids,
        "chosen_arm_by_sample": chosen_by_sample,
    }


def run_arm(args: argparse.Namespace, *, arm: str, output_path: Path) -> None:
    policy = POLICY_BY_ARM[arm]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_domain_workflow.py"),
        "--domain",
        args.domain,
        "--data-path",
        args.data_path,
        "--model-path",
        args.model_path,
        "--execution-mode",
        "optimized",
        "--communication-policy",
        policy,
        "--limit",
        str(args.limit),
        "--max-rounds",
        str(args.max_rounds),
        "--communication-seed",
        str(args.communication_seed),
        "--output",
        str(output_path),
    ]
    if args.communication_budget_tokens is not None:
        cmd.extend(["--communication-budget-tokens", str(args.communication_budget_tokens)])
    if args.fragment_utility_table:
        cmd.extend(["--fragment-utility-table", args.fragment_utility_table])
    cmd.extend(["--task-family", args.domain])
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("gsm8k", "tatqa", "hotpotqa", "mbpp", "humaneval", "mmlu_pro"), required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--communication-seed", type=int, default=0)
    parser.add_argument("--communication-budget-tokens", type=int, default=None)
    parser.add_argument("--fragment-utility-table", default="")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--run-policy-baselines", action="store_true")
    args = parser.parse_args()

    arms = tuple(dict.fromkeys(item.strip() for item in args.arms.split(",") if item.strip()))
    unknown = [arm for arm in arms if arm not in POLICY_BY_ARM]
    if unknown:
        raise SystemExit(f"unknown ablation arms: {', '.join(unknown)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, Any]] = {}
    for arm in arms:
        path = output_dir / f"{args.domain}_{arm}_{args.limit}.json"
        if not args.reuse_existing or not path.exists():
            run_arm(args, arm=arm, output_path=path)
        reports[arm] = load_report(path)

    utility_table_for_baselines = args.fragment_utility_table
    summary = {
        "domain": args.domain,
        "data_path": args.data_path,
        "limit": args.limit,
        "arms": list(arms),
        "arm_outputs": {
            arm: str(output_dir / f"{args.domain}_{arm}_{args.limit}.json")
            for arm in arms
        },
        **summarize_ablation(reports, domain=args.domain, seed=args.communication_seed),
    }
    training_examples = build_fragment_utility_examples(reports, domain=args.domain)
    summary_path = output_dir / f"{args.domain}_fragment_utility_ablation_{args.limit}.json"
    training_path = output_dir / f"{args.domain}_fragment_utility_examples_{args.limit}.jsonl"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with training_path.open("w", encoding="utf-8") as handle:
        for example in training_examples:
            handle.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary["training_examples_path"] = str(training_path)
    if not utility_table_for_baselines:
        utility_table_for_baselines = str(summary_path)
    if args.run_policy_baselines:
        policy_outputs: dict[str, str] = {}
        for arm in POLICY_BASELINE_ARMS:
            path = output_dir / f"{args.domain}_{arm}_policy_{args.limit}.json"
            baseline_args = argparse.Namespace(**vars(args))
            baseline_args.fragment_utility_table = utility_table_for_baselines
            if not args.reuse_existing or not path.exists():
                run_arm(baseline_args, arm=arm, output_path=path)
            policy_outputs[arm] = str(path)
        summary["policy_baseline_outputs"] = policy_outputs
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hidden_keys = {"oracle_sources_by_sample", "oracle_targeted_sources_by_sample", "oracle_any_sources_by_sample"}
    print(json.dumps({k: v for k, v in summary.items() if k not in hidden_keys}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
