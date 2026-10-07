"""Evaluate receiver-aware communication policies for candidate selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.run_reasoning_upper_bounds import deployable_candidate_records  # noqa: E402
from workflow_runtime.candidate_selector_communication import (  # noqa: E402
    build_candidate_fragments,
    rendered_fragment_tokens,
    run_selector_with_fragments,
    select_fragment_ids,
)


POLICIES = (
    "selector_send_all",
    "selector_core_only",
    "selector_verifier_only",
    "selector_receiver_aware",
    "selector_random_same_budget",
    "selector_oracle_minimal",
)


def evaluate_selector_communication(path: Path, *, domain: str, seed: int = 0) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    summaries: dict[str, dict[str, Any]] = {
        policy: {
            "correct_count": 0,
            "selector_input_tokens": 0,
            "records": [],
        }
        for policy in POLICIES
    }
    for record in source.get("records", []):
        deployable = deployable_candidate_records(list(record.get("candidates", [])), domain=domain)
        oracle_candidate_id = _first_correct_candidate_id(deployable)
        fragments = build_candidate_fragments(deployable)
        receiver_aware_ids = select_fragment_ids(fragments, policy="selector_receiver_aware", seed=seed)
        receiver_aware_budget = rendered_fragment_tokens(fragments, receiver_aware_ids)
        for policy in POLICIES:
            kwargs = {"policy": policy, "seed": seed}
            if policy == "selector_random_same_budget":
                kwargs["budget_tokens"] = receiver_aware_budget
            if policy == "selector_oracle_minimal":
                kwargs["oracle_candidate_id"] = oracle_candidate_id or deployable[0]["candidate_id"]
            result = run_selector_with_fragments(deployable, **kwargs)
            selected = next(item for item in deployable if item["candidate_id"] == result.selected_candidate_id)
            correct = bool(selected.get("correct"))
            summaries[policy]["correct_count"] += int(correct)
            summaries[policy]["selector_input_tokens"] += result.rendered_tokens
            summaries[policy]["records"].append({
                "sample_id": record.get("sample_id"),
                "selected_candidate_id": result.selected_candidate_id,
                "correct": correct,
                "selector_input_tokens": result.rendered_tokens,
                "visible_fragment_ids": list(result.visible_fragment_ids),
                "selection_reason": list(result.selection_reason),
            })
    count = len(source.get("records", []))
    compact: dict[str, Any] = {
        "domain": domain,
        "source_path": str(path),
        "count": count,
        "policies": {},
    }
    send_all_tokens = max(1, int(summaries["selector_send_all"]["selector_input_tokens"]))
    send_all_quality = summaries["selector_send_all"]["correct_count"] / count if count else 0.0
    for policy, payload in summaries.items():
        quality = payload["correct_count"] / count if count else 0.0
        tokens = int(payload["selector_input_tokens"])
        compact["policies"][policy] = {
            "accuracy": quality,
            "correct_count": payload["correct_count"],
            "selector_input_tokens": tokens,
            "token_saving_vs_send_all": 1.0 - (tokens / send_all_tokens),
            "quality_delta_vs_send_all": quality - send_all_quality,
        }
    compact["records_by_policy"] = {
        policy: payload["records"] for policy, payload in summaries.items()
    }
    compact["phase4_gate"] = {
        "epsilon": 0.02,
        "receiver_aware_quality_ok": (
            compact["policies"]["selector_receiver_aware"]["accuracy"]
            >= compact["policies"]["selector_send_all"]["accuracy"] - 0.02
        ),
        "receiver_aware_cost_ok": (
            compact["policies"]["selector_receiver_aware"]["selector_input_tokens"]
            < compact["policies"]["selector_send_all"]["selector_input_tokens"]
        ),
        "receiver_aware_token_saving_at_least_20pct": (
            compact["policies"]["selector_receiver_aware"]["token_saving_vs_send_all"] >= 0.2
        ),
    }
    return compact


def _first_correct_candidate_id(candidates: list[dict[str, Any]]) -> str | None:
    for candidate in candidates:
        if candidate.get("correct"):
            return str(candidate.get("candidate_id"))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--domain", required=True, choices=("gsm8k", "humaneval", "mbpp"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = evaluate_selector_communication(Path(args.input), domain=args.domain, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records_by_policy"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
