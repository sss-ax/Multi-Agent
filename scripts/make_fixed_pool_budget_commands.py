"""Generate fixed-candidate-pool commands and post-hoc budget summaries."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


POLICIES = (
    "selector_send_all",
    "selector_core_only",
    "selector_receiver_aware",
    "selector_random_same_budget",
)


def build_fixed_pool_commands(args: argparse.Namespace) -> list[dict[str, Any]]:
    _validate_model_path(args.model_path)
    commands: list[dict[str, Any]] = []
    for domain in args.domains:
        pool_path = Path(args.output_dir) / domain / f"candidate_pool_k{args.k}_limit{args.limit}_seed{args.seed}.json"
        selector_path = Path(args.output_dir) / domain / f"selector_policies_k{args.k}_limit{args.limit}_seed{args.seed}.json"
        pool_command = [
            args.python,
            "scripts/run_reasoning_upper_bounds.py",
            "--domain",
            domain,
            "--data-path",
            args.data_paths[domain],
            "--model-path",
            args.model_path,
            "--experiment",
            "fixed_candidate_pool",
            "--n",
            str(args.k),
            "--limit",
            str(args.limit),
            "--seed",
            str(args.seed),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--output",
            str(pool_path),
        ]
        selector_command = [
            args.python,
            "scripts/evaluate_selector_communication.py",
            "--input",
            str(pool_path),
            "--domain",
            domain,
            "--seed",
            str(args.seed),
            "--output",
            str(selector_path),
        ]
        budget_command = [
            args.python,
            "scripts/make_fixed_pool_budget_commands.py",
            "summarize-budget",
            "--candidate-report",
            str(pool_path),
            "--selector-report",
            str(selector_path),
            "--budgets",
            *[str(item) for item in args.budgets],
            "--output",
            str(Path(args.output_dir) / domain / f"budget_curve_k{args.k}_limit{args.limit}_seed{args.seed}.json"),
        ]
        commands.extend(
            [
                {"domain": domain, "stage": "candidate_pool", "output": str(pool_path), "command": pool_command, "shell": shlex.join(pool_command)},
                {"domain": domain, "stage": "selector_policies", "output": str(selector_path), "command": selector_command, "shell": shlex.join(selector_command)},
                {"domain": domain, "stage": "budget_summary", "output": budget_command[-1], "command": budget_command, "shell": shlex.join(budget_command)},
            ]
        )
    return commands


def summarize_budget(candidate_report: Path, selector_report: Path, budgets: list[int]) -> dict[str, Any]:
    candidates = json.loads(candidate_report.read_text(encoding="utf-8"))
    selector = json.loads(selector_report.read_text(encoding="utf-8"))
    candidate_costs = {
        str(record.get("sample_id")): sum(
            int(candidate.get("input_tokens", 0) or 0) + int(candidate.get("output_tokens", 0) or 0)
            for candidate in record.get("candidates", [])
        )
        for record in candidates.get("records", [])
    }
    count = int(selector.get("count", 0) or len(candidate_costs))
    curves: dict[str, Any] = {}
    for policy in POLICIES:
        records = selector.get("records_by_policy", {}).get(policy, [])
        points = []
        for budget in budgets:
            eligible = []
            for record in records:
                sample_id = str(record.get("sample_id"))
                total_tokens = candidate_costs.get(sample_id, 0) + int(record.get("selector_input_tokens", 0) or 0)
                eligible.append(
                    {
                        "sample_id": sample_id,
                        "correct": bool(record.get("correct")),
                        "total_tokens": total_tokens,
                        "within_budget": total_tokens <= budget,
                    }
                )
            solved = sum(int(item["correct"] and item["within_budget"]) for item in eligible)
            within = sum(int(item["within_budget"]) for item in eligible)
            points.append(
                {
                    "budget": budget,
                    "solved_count": solved,
                    "within_budget_count": within,
                    "accuracy_under_budget": solved / count if count else 0.0,
                    "coverage_under_budget": within / count if count else 0.0,
                }
            )
        curves[policy] = {
            "total_selector_input_tokens": int(selector.get("policies", {}).get(policy, {}).get("selector_input_tokens", 0) or 0),
            "accuracy_without_budget": float(selector.get("policies", {}).get(policy, {}).get("accuracy", 0.0) or 0.0),
            "points": points,
        }
    return {
        "candidate_report": str(candidate_report),
        "selector_report": str(selector_report),
        "count": count,
        "budgets": budgets,
        "policies": curves,
        "note": "Post-hoc budget curve: candidate generation cost plus selector-visible input. Runtime budget stopping is not enforced here.",
    }


def _domain_data_paths(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--data-paths entries must be domain=path, got {value!r}")
        domain, path = value.split("=", 1)
        parsed[domain.strip()] = path.strip()
    return parsed


def _validate_model_path(model_path: str) -> None:
    if model_path in {"/path/to/qwen", "/path/to/Qwen2.5-1.5B-Instruct", "/path/to/Qwen2.5-7B-Instruct"}:
        raise SystemExit(
            "Replace --model-path with a real local model directory or HuggingFace id; "
            f"got placeholder {model_path!r}."
        )
    path = Path(model_path)
    if path.is_absolute() and not path.exists():
        raise SystemExit(
            f"--model-path points to a missing absolute path: {model_path}. "
            "Use an existing local model directory or a HuggingFace id such as Qwen/Qwen2.5-7B-Instruct."
        )


def command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed candidate pool and budget-matched experiment utilities.")
    sub = parser.add_subparsers(dest="mode", required=True)

    make = sub.add_parser("make-commands", help="Generate fixed-pool, selector-policy, and budget-summary commands.")
    make.add_argument("--python", default=".venv/bin/python")
    make.add_argument("--model-path", required=True)
    make.add_argument("--data-paths", nargs="+", required=True, help="Entries like gsm8k=data/gsm8k.jsonl")
    make.add_argument("--domains", nargs="+", default=["gsm8k", "humaneval", "mbpp", "mmlu_pro", "hotpotqa"])
    make.add_argument("--output-dir", default=".runtime/fixed_pool_budget")
    make.add_argument("--k", type=int, default=4)
    make.add_argument("--limit", type=int, default=100)
    make.add_argument("--seed", type=int, default=0)
    make.add_argument("--temperature", type=float, default=0.7)
    make.add_argument("--top-p", type=float, default=0.95)
    make.add_argument("--max-new-tokens", type=int, default=512)
    make.add_argument("--budgets", nargs="+", type=int, default=[1000, 2000, 3000, 5000, 8000])
    make.add_argument("--commands-output", default="")
    make.add_argument("--json-output", default="")

    summarize = sub.add_parser("summarize-budget", help="Build post-hoc budget curves from fixed-pool selector reports.")
    summarize.add_argument("--candidate-report", required=True)
    summarize.add_argument("--selector-report", required=True)
    summarize.add_argument("--budgets", nargs="+", type=int, default=[1000, 2000, 3000, 5000, 8000])
    summarize.add_argument("--output", required=True)
    return parser


def main() -> None:
    parser = command_parser()
    args = parser.parse_args()
    if args.mode == "make-commands":
        args.data_paths = _domain_data_paths(args.data_paths)
        missing = [domain for domain in args.domains if domain not in args.data_paths]
        if missing:
            raise SystemExit(f"Missing --data-paths entries for: {', '.join(missing)}")
        commands = build_fixed_pool_commands(args)
        if args.commands_output:
            path = Path(args.commands_output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(item["shell"] for item in commands) + "\n", encoding="utf-8")
        if args.json_output:
            path = Path(args.json_output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"commands": commands}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for item in commands:
            print(item["shell"])
        return

    payload = summarize_budget(Path(args.candidate_report), Path(args.selector_report), args.budgets)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "policies"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
