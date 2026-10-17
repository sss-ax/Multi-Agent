"""Paired bootstrap and exact McNemar summaries for two reports."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def load_correct_by_id(path: Path) -> dict[str, bool]:
    report = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, bool] = {}
    for index, record in enumerate(report.get("records", [])):
        sample_id = str(record.get("sample_id") or index)
        result[sample_id] = bool(record.get("correct") or record.get("selected_correct"))
    return result


def paired_stats(left_path: Path, right_path: Path, *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    left = load_correct_by_id(left_path)
    right = load_correct_by_id(right_path)
    ids = sorted(set(left) & set(right))
    if not ids:
        raise ValueError("reports have no paired sample ids")
    deltas = [int(right[item]) - int(left[item]) for item in ids]
    n01 = sum(1 for item in ids if not left[item] and right[item])
    n10 = sum(1 for item in ids if left[item] and not right[item])
    rng = random.Random(seed)
    draws = []
    for _ in range(bootstrap_samples):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    return {
        "left_report": str(left_path),
        "right_report": str(right_path),
        "paired_count": len(ids),
        "left_accuracy": sum(int(left[item]) for item in ids) / len(ids),
        "right_accuracy": sum(int(right[item]) for item in ids) / len(ids),
        "accuracy_delta_right_minus_left": sum(deltas) / len(ids),
        "wrong_to_right_count": n01,
        "right_to_wrong_count": n10,
        "mcnemar_exact_p": _mcnemar_exact_p(n01, n10),
        "bootstrap_ci_95": [
            draws[int(0.025 * (len(draws) - 1))],
            draws[int(0.975 * (len(draws) - 1))],
        ],
    }


def _mcnemar_exact_p(n01: int, n10: int) -> float:
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    cdf = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return min(1.0, 2.0 * cdf)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired bootstrap and exact McNemar for two reports.")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = paired_stats(
        Path(args.left),
        Path(args.right),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
