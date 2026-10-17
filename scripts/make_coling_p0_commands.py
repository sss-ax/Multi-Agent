"""Generate commands for the COLING P0/P1 experiment suite.

The generated shell script is intentionally explicit.  It first builds
fragment-utility tables, then runs receiver-aware/semantic-feedback arms,
fixed candidate-pool comparisons, latest five-domain main runs, and finally
summary/audit commands.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any


DATA_PATH_DEFAULTS = {
    "gsm8k": "data/gsm8k/test.jsonl",
    "humaneval": "data/humaneval/HumanEval.jsonl.gz",
    "mbpp": "data/mbpp/mbpp.jsonl",
    "mmlu_pro": "data/mmlu_pro/test.jsonl",
    "hotpotqa": "data/hotpotqa/hotpot_dev_distractor_v1.jsonl",
}

P0_POLICIES = (
    "send_all",
    "random_same_budget",
    "receiver_aware_no_closure",
    "receiver_aware_heuristic",
    "receiver_aware_targeted_feedback",
    "minimal_no_feedback",
    "minimal_sendall_fallback",
    "minimal_targeted_feedback",
)


def build_commands(args: argparse.Namespace) -> list[str]:
    _validate_model_path(args.model_path)
    data_paths = _domain_data_paths(args.data_paths)
    domains = list(args.domains)
    missing = [domain for domain in domains if domain not in data_paths]
    if missing:
        raise SystemExit(f"Missing data paths for: {', '.join(missing)}")

    commands: list[str] = []
    output_dir = Path(args.output_dir)
    utility_dir = output_dir / "fragment_utility"
    ra_dir = output_dir / "receiver_aware"
    semantic_dir = output_dir / "semantic_feedback"
    latest_dir = output_dir / "latest_five_domain"
    fixed_dir = output_dir / "fixed_pool_budget"
    stats_dir = output_dir / "stats"

    for domain in domains:
        utility_out = utility_dir / domain
        commands.append(_join([
            args.python,
            "scripts/run_fragment_utility_ablation.py",
            "--domain",
            domain,
            "--data-path",
            data_paths[domain],
            "--model-path",
            args.model_path,
            "--limit",
            str(args.utility_limit),
            "--max-rounds",
            str(args.max_rounds),
            "--communication-seed",
            str(args.seed),
            "--arms",
            "core_only,support_dependencies,full_plan,result_metadata,calculation_trace,full_feedback,send_all",
            "--output-dir",
            str(utility_out),
            "--run-policy-baselines",
        ]))
        utility_table = utility_out / f"{domain}_fragment_utility_ablation_{args.utility_limit}.json"
        for policy in P0_POLICIES:
            out_dir = semantic_dir if policy.startswith("minimal_") else ra_dir
            commands.append(_join([
                args.python,
                "scripts/evaluate_domain_workflow.py",
                "--domain",
                domain,
                "--data-path",
                data_paths[domain],
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
                str(args.seed),
                "--fragment-utility-table",
                str(utility_table),
                "--task-family",
                domain,
                "--output",
                str(out_dir / domain / f"{domain}_{policy}_limit{args.limit}_seed{args.seed}.json"),
            ]))

        for policy in ("selector_send_all", "selector_core_only", "selector_verifier_only", "selector_receiver_aware", "selector_random_same_budget"):
            if policy:
                pass
        commands.append(_join([
            args.python,
            "scripts/run_reasoning_upper_bounds.py",
            "--domain",
            domain,
            "--data-path",
            data_paths[domain],
            "--model-path",
            args.model_path,
            "--experiment",
            "adaptive_controller_selector_comm",
            "--selector-communication-policy",
            "selector_receiver_aware",
            "--n",
            str(args.k),
            "--limit",
            str(args.limit),
            "--seed",
            str(args.seed),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--output",
            str(latest_dir / domain / f"{domain}_latest_full_k{args.k}_limit{args.limit}_seed{args.seed}.json"),
        ]))

    commands.append(_join([
        args.python,
        "scripts/make_fixed_pool_budget_commands.py",
        "make-commands",
        "--python",
        args.python,
        "--model-path",
        args.model_path,
        "--data-paths",
        *[f"{domain}={data_paths[domain]}" for domain in domains],
        "--domains",
        *domains,
        "--output-dir",
        str(fixed_dir),
        "--k",
        str(args.k),
        "--limit",
        str(args.limit),
        "--seed",
        str(args.seed),
        "--commands-output",
        str(output_dir / "fixed_pool_budget_commands.sh"),
        "--json-output",
        str(output_dir / "fixed_pool_budget_commands.json"),
    ]))
    commands.append(_join(["bash", str(output_dir / "fixed_pool_budget_commands.sh")]))

    semantic_inputs = [
        str(semantic_dir / domain / f"{domain}_{policy}_limit{args.limit}_seed{args.seed}.json")
        for domain in domains
        for policy in ("minimal_no_feedback", "minimal_sendall_fallback", "minimal_targeted_feedback")
    ]
    semantic_inputs.extend(
        str(ra_dir / domain / f"{domain}_receiver_aware_targeted_feedback_limit{args.limit}_seed{args.seed}.json")
        for domain in domains
    )
    commands.append(_join([
        args.python,
        "scripts/summarize_semantic_feedback_ablation.py",
        "--inputs",
        *semantic_inputs,
        "--output",
        str(stats_dir / "semantic_feedback_summary.json"),
    ]))

    audit_globs = [
        str(ra_dir / "*" / "*.json"),
        str(semantic_dir / "*" / "*.json"),
        str(latest_dir / "*" / "*.json"),
        str(fixed_dir / "*" / "candidate_pool_*.json"),
    ]
    commands.append(
        f"{shlex.quote(args.python)} scripts/audit_token_cost_accounting.py --inputs "
        + " ".join(audit_globs)
        + f" --output {shlex.quote(str(stats_dir / 'token_cost_audit.json'))}"
    )

    if "hotpotqa" in domains:
        hotpot_latest = latest_dir / "hotpotqa" / f"hotpotqa_latest_full_k{args.k}_limit{args.limit}_seed{args.seed}.json"
        commands.append(_join([
            args.python,
            "scripts/run_hotpot_topology_verifier_diagnostic.py",
            "--output",
            str(stats_dir / "hotpot_topology_verifier_diagnostic.json"),
        ]))
        commands.append(_join([
            args.python,
            "scripts/run_hotpotqa_bridge_evidence_selection_ablation.py",
            "--eval-json",
            str(hotpot_latest),
            "--data-path",
            data_paths["hotpotqa"],
            "--model-path",
            args.model_path,
            "--limit",
            str(args.hotpot_diagnostic_limit),
            "--seed",
            str(args.seed),
            "--output",
            str(stats_dir / "hotpot_dependency_analysis.json"),
        ]))
    return commands


def _join(parts: list[str]) -> str:
    return shlex.join(parts)


def _domain_data_paths(values: list[str]) -> dict[str, str]:
    parsed = dict(DATA_PATH_DEFAULTS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"--data-paths entries must be domain=path, got {value!r}")
        domain, path = value.split("=", 1)
        parsed[domain.strip()] = path.strip()
    return parsed


def _validate_model_path(model_path: str) -> None:
    if model_path in {"/path/to/qwen", "/path/to/Qwen2.5-1.5B-Instruct", "/path/to/Qwen2.5-7B-Instruct"}:
        raise SystemExit(f"Replace placeholder --model-path with a real model path; got {model_path!r}")
    path = Path(model_path)
    if path.is_absolute() and not path.exists():
        raise SystemExit(f"--model-path points to a missing absolute path: {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate COLING P0/P1 experiment commands.")
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-paths", nargs="*", default=[])
    parser.add_argument("--domains", nargs="+", default=["gsm8k", "humaneval", "mbpp", "mmlu_pro", "hotpotqa"])
    parser.add_argument("--output-dir", default=".runtime/coling_p0")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--utility-limit", type=int, default=50)
    parser.add_argument("--hotpot-diagnostic-limit", type=int, default=50)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--commands-output", default="")
    args = parser.parse_args()

    commands = build_commands(args)
    script = "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + "\n"
    if args.commands_output:
        output = Path(args.commands_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(script, encoding="utf-8")
    print(script, end="")


if __name__ == "__main__":
    main()
