"""Stage 5 diagnostic for topology-specific Hotpot reasoning verifiers.

This audit deliberately does not feed candidates back into ranking.  It builds a
small controlled gold candidate and several wrong candidates for each reasoning
topology, then measures whether the topology-specific verifier separates them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts import run_reasoning_upper_bounds as rub  # noqa: E402
from workflow_runtime.hotpot_topology import REASONING_TYPES  # noqa: E402


def diagnostic_cases() -> dict[str, dict[str, Any]]:
    return {
        "bridge_entity": {
            "row": {
                "question": "Who connects Alice to Carol?",
                "entities": ["Alice", "Carol"],
            },
            "bridge": "Bob",
            "gold": (
                {"fact_id": "be1", "title": "Alice", "text": "Alice was created by Bob."},
                {"fact_id": "be2", "title": "Bob", "text": "Bob created Carol."},
            ),
            "wrongs": [
                (
                    {"fact_id": "be1w1", "title": "Alice", "text": "Alice was created by Dana."},
                    {"fact_id": "be2w1", "title": "Evan", "text": "Evan created Carol."},
                ),
                (
                    {"fact_id": "be1w2", "title": "Alice", "text": "Alice was created by Bob."},
                    {"fact_id": "be2w2", "title": "Dana", "text": "Dana created Carol."},
                ),
                (
                    {"fact_id": "be1w3", "title": "Alice", "text": "Alice is a fictional character."},
                    {"fact_id": "be2w3", "title": "Bob", "text": "Bob was born in Paris."},
                ),
            ],
        },
        "bridge_attribute": {
            "row": {
                "question": "What company formed the group that created 2014 S/S?",
                "entities": ["2014 S/S"],
            },
            "bridge": "WINNER",
            "gold": (
                {"fact_id": "ba1", "title": "2014 S/S", "text": "2014 S/S was created by WINNER."},
                {"fact_id": "ba2", "title": "WINNER", "text": "WINNER was formed by YG Entertainment."},
            ),
            "wrongs": [
                (
                    {"fact_id": "ba1w1", "title": "2014 S/S", "text": "2014 S/S was created by Team A."},
                    {"fact_id": "ba2w1", "title": "WINNER", "text": "WINNER was formed by YG Entertainment."},
                ),
                (
                    {"fact_id": "ba1w2", "title": "2014 S/S", "text": "2014 S/S was created by WINNER."},
                    {"fact_id": "ba2w2", "title": "Team A", "text": "Team A was formed by YG Entertainment."},
                ),
                (
                    {"fact_id": "ba1w3", "title": "2014 S/S", "text": "2014 S/S was created by WINNER."},
                    {"fact_id": "ba2w3", "title": "WINNER", "text": "WINNER is a South Korean boy band."},
                ),
            ],
        },
        "comparison_attribute": {
            "row": {
                "question": "Are Museum A and Museum B located in the same neighborhood?",
                "entities": ["Museum A", "Museum B"],
            },
            "bridge": "",
            "gold": (
                {"fact_id": "ca1", "title": "Museum A", "text": "Museum A is located in Laleli."},
                {"fact_id": "ca2", "title": "Museum B", "text": "Museum B is located in Laleli."},
            ),
            "wrongs": [
                (
                    {"fact_id": "ca1w1", "title": "Museum A", "text": "Museum A is located in Laleli."},
                    {"fact_id": "ca2w1", "title": "Museum B", "text": "Museum B is located in Ortakoy."},
                ),
                (
                    {"fact_id": "ca1w2", "title": "Museum A", "text": "Museum A is located in Laleli."},
                    {"fact_id": "ca2w2", "title": "Museum C", "text": "Museum C is located in Laleli."},
                ),
                (
                    {"fact_id": "ca1w3", "title": "Museum A", "text": "Museum A is located in Laleli."},
                    {"fact_id": "ca2w3", "title": "Museum B", "text": "Museum B opened in 1900."},
                ),
            ],
        },
        "shared_property_yesno": {
            "row": {
                "question": "Are both Alpha and Beta genus plants?",
                "entities": ["Alpha", "Beta"],
            },
            "bridge": "",
            "gold": (
                {"fact_id": "sp1", "title": "Alpha", "text": "Alpha is a genus of flowering plants."},
                {"fact_id": "sp2", "title": "Beta", "text": "Beta is a genus of flowering plants."},
            ),
            "wrongs": [
                (
                    {"fact_id": "sp1w1", "title": "Alpha", "text": "Alpha is a genus of flowering plants."},
                    {"fact_id": "sp2w1", "title": "Beta", "text": "Beta is a city in Europe."},
                ),
                (
                    {"fact_id": "sp1w2", "title": "Alpha", "text": "Alpha is a city in Europe."},
                    {"fact_id": "sp2w2", "title": "Beta", "text": "Beta is a genus of flowering plants."},
                ),
                (
                    {"fact_id": "sp1w3", "title": "Gamma", "text": "Gamma is a genus of flowering plants."},
                    {"fact_id": "sp2w3", "title": "Beta", "text": "Beta is a genus of flowering plants."},
                ),
            ],
        },
        "set_intersection": {
            "row": {
                "question": "What genus property do Alpha and Beta share in common?",
                "entities": ["Alpha", "Beta"],
            },
            "bridge": "",
            "gold": (
                {"fact_id": "si1", "title": "Alpha", "text": "Alpha is a genus of flowering plants."},
                {"fact_id": "si2", "title": "Beta", "text": "Beta is a genus of flowering plants."},
            ),
            "wrongs": [
                (
                    {"fact_id": "si1w1", "title": "Alpha", "text": "Alpha is a genus of flowering plants."},
                    {"fact_id": "si2w1", "title": "Beta", "text": "Beta is a city in Europe."},
                ),
                (
                    {"fact_id": "si1w2", "title": "Alpha", "text": "Alpha is located in Europe."},
                    {"fact_id": "si2w2", "title": "Beta", "text": "Beta is a genus of flowering plants."},
                ),
                (
                    {"fact_id": "si1w3", "title": "Alpha", "text": "Alpha is a genus of flowering plants."},
                    {"fact_id": "si2w3", "title": "Gamma", "text": "Gamma is a genus of flowering plants."},
                ),
            ],
        },
        "multi_constraint_lookup": {
            "row": {
                "question": "Who wrote Novel X and directed Film Y?",
                "entities": ["Novel X", "Film Y"],
            },
            "bridge": "Jordan Lee",
            "gold": (
                {"fact_id": "mc1", "title": "Novel X", "text": "Novel X was created by Jordan Lee."},
                {"fact_id": "mc2", "title": "Film Y", "text": "Film Y was directed by Jordan Lee."},
            ),
            "wrongs": [
                (
                    {"fact_id": "mc1w1", "title": "Novel X", "text": "Novel X was created by Jordan Lee."},
                    {"fact_id": "mc2w1", "title": "Film Y", "text": "Film Y was directed by Morgan Park."},
                ),
                (
                    {"fact_id": "mc1w2", "title": "Novel X", "text": "Novel X was published in 2010."},
                    {"fact_id": "mc2w2", "title": "Film Y", "text": "Film Y was directed by Jordan Lee."},
                ),
                (
                    {"fact_id": "mc1w3", "title": "Novel Z", "text": "Novel Z was created by Jordan Lee."},
                    {"fact_id": "mc2w3", "title": "Film Y", "text": "Film Y was directed by Jordan Lee."},
                ),
            ],
        },
    }


def _candidate_valid(verdict: dict[str, Any]) -> bool:
    result = verdict["verification_result"]
    return bool(result["evidence_valid"] and result["reasoning_complete"] and result["answer_supported"])


def run_diagnostic() -> dict[str, Any]:
    rows = []
    summary: dict[str, Any] = {}
    for reasoning_type in REASONING_TYPES:
        case = diagnostic_cases()[reasoning_type]
        row = dict(case["row"])
        gold = rub._hotpot_verify_semantic_chain(row, case["bridge"], case["gold"][0], case["gold"][1])
        gold_result = gold["verification_result"]
        gold_score = float(gold_result["score"])
        wrong_records = []
        for wrong_index, wrong_pair in enumerate(case["wrongs"], start=1):
            wrong = rub._hotpot_verify_semantic_chain(row, case["bridge"], wrong_pair[0], wrong_pair[1])
            wrong_result = wrong["verification_result"]
            wrong_score = float(wrong_result["score"])
            wrong_records.append({
                "wrong_id": f"{reasoning_type}_wrong_{wrong_index}",
                "valid": _candidate_valid(wrong),
                "score": wrong_score,
                "gold_score_gt_wrong": gold_score > wrong_score,
                "candidate": wrong["reasoning_candidate"],
                "verification": wrong_result,
            })
        gold_valid = _candidate_valid(gold)
        wrong_valid_count = sum(int(item["valid"]) for item in wrong_records)
        comparison_count = len(wrong_records)
        gold_score_wins = sum(int(item["gold_score_gt_wrong"]) for item in wrong_records)
        entry = {
            "reasoning_type": reasoning_type,
            "gold_valid": gold_valid,
            "gold_valid_rate": 1.0 if gold_valid else 0.0,
            "wrong_valid_rate": wrong_valid_count / comparison_count if comparison_count else 0.0,
            "p_gold_score_gt_wrong": gold_score_wins / comparison_count if comparison_count else 0.0,
            "gold_score": gold_score,
            "wrong_count": comparison_count,
            "wrong_valid_count": wrong_valid_count,
            "gold_candidate": gold["reasoning_candidate"],
            "gold_verification": gold_result,
            "wrong_candidates": wrong_records,
        }
        rows.append(entry)
        summary[reasoning_type] = {
            "gold_valid_rate": entry["gold_valid_rate"],
            "wrong_valid_rate": entry["wrong_valid_rate"],
            "p_gold_score_gt_wrong": entry["p_gold_score_gt_wrong"],
            "gold_valid": entry["gold_valid"],
            "wrong_count": entry["wrong_count"],
        }
    return {
        "count": len(rows),
        "summary_by_type": summary,
        "gate_pass": all(
            item["gold_valid_rate"] > item["wrong_valid_rate"]
            and item["p_gold_score_gt_wrong"] == 1.0
            for item in rows
        ),
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 5 topology-specific verifier diagnostics.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_diagnostic()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary_by_type"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
