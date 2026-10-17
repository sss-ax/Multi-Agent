"""Generate runnable commands for controller/selector/RA module ablations."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


SETTINGS = (
    {
        "setting": "full",
        "controller": True,
        "selector": True,
        "receiver_aware": True,
        "experiment": "adaptive_controller_selector_comm",
        "selector_policy": "selector_receiver_aware",
    },
    {
        "setting": "no_ra_send_all",
        "controller": True,
        "selector": True,
        "receiver_aware": False,
        "experiment": "adaptive_controller_selector_comm",
        "selector_policy": "selector_send_all",
    },
    {
        "setting": "no_controller_fixed_k",
        "controller": False,
        "selector": True,
        "receiver_aware": False,
        "experiment": "verifier_select",
        "selector_policy": None,
    },
    {
        "setting": "no_selector_adaptive",
        "controller": True,
        "selector": False,
        "receiver_aware": True,
        "experiment": "adaptive_controller",
        "selector_policy": None,
    },
    {
        "setting": "base_single_candidate",
        "controller": False,
        "selector": False,
        "receiver_aware": False,
        "experiment": "best_of_n",
        "selector_policy": None,
        "n_override": 1,
    },
)


def build_commands(args: argparse.Namespace) -> list[dict[str, Any]]:
    _validate_model_path(args.model_path)
    commands: list[dict[str, Any]] = []
    for domain in args.domains:
        data_path = args.data_paths[domain]
        for setting in SETTINGS:
            n = int(setting.get("n_override") or args.n)
            output = Path(args.output_dir) / domain / f"{setting['setting']}_n{n}_limit{args.limit}_seed{args.seed}.json"
            command = [
                args.python,
                "scripts/run_reasoning_upper_bounds.py",
                "--domain",
                domain,
                "--data-path",
                data_path,
                "--model-path",
                args.model_path,
                "--experiment",
                str(setting["experiment"]),
                "--n",
                str(n),
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
                "--min-repair-candidates",
                str(args.min_repair_candidates),
                "--output",
                str(output),
            ]
            if setting.get("selector_policy"):
                command.extend(["--selector-communication-policy", str(setting["selector_policy"])])
            commands.append(
                {
                    "domain": domain,
                    "setting": setting["setting"],
                    "controller": setting["controller"],
                    "selector": setting["selector"],
                    "receiver_aware": setting["receiver_aware"],
                    "output": str(output),
                    "command": command,
                    "shell": shlex.join(command),
                }
            )
    return commands


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate module-ablation experiment commands.")
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-paths", nargs="+", required=True, help="Entries like gsm8k=data/gsm8k.jsonl")
    parser.add_argument("--domains", nargs="+", default=["gsm8k", "humaneval", "mbpp", "mmlu_pro", "hotpotqa"])
    parser.add_argument("--output-dir", default=".runtime/module_ablations")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-repair-candidates", type=int, default=4)
    parser.add_argument("--commands-output", default="")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()
    args.data_paths = _domain_data_paths(args.data_paths)

    missing = [domain for domain in args.domains if domain not in args.data_paths]
    if missing:
        raise SystemExit(f"Missing --data-paths entries for: {', '.join(missing)}")

    commands = build_commands(args)
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


if __name__ == "__main__":
    main()
