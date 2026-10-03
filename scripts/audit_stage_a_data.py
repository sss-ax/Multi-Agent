"""Audit sample-id leakage and prepare a fixed 200-question Stage A dev set."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ids_in(path: Path) -> Set[str]:
    return {str(row["sample_id"]) for row in load_jsonl(path) if row.get("sample_id")}


def role_splits(role_dir: Path) -> Dict[str, Set[str]]:
    result: Dict[str, Set[str]] = {}
    for split in ("train", "validation"):
        path = role_dir / f"{split}.jsonl"
        result[split] = ids_in(path) if path.is_file() else set()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-data-path", required=True)
    parser.add_argument("--dev-out", default="data/stage_a_dev_200.jsonl")
    parser.add_argument("--report-out", default="logs/stage_a_data_audit.json")
    parser.add_argument("--dev-limit", type=int, default=200)
    parser.add_argument("--role-data-dir", action="append", nargs=2, metavar=("ROLE", "DIR"), required=True)
    args = parser.parse_args()

    dev_rows = load_jsonl(Path(args.dev_data_path))[: args.dev_limit]
    if len(dev_rows) != args.dev_limit:
        raise ValueError(f"expected at least {args.dev_limit} dev rows, found {len(dev_rows)}")
    dev_rows_out = []
    dev_ids: Set[str] = set()
    for row in dev_rows:
        question = str(row.get("question", "")).strip()
        if not question:
            raise ValueError("dev row has no question")
        sample_id = hashlib.sha256(question.encode("utf-8")).hexdigest()
        if sample_id in dev_ids:
            raise ValueError(f"duplicate dev sample_id: {sample_id}")
        dev_ids.add(sample_id)
        dev_rows_out.append({**row, "sample_id": sample_id})

    role_assignments: Dict[str, Dict[str, Set[str]]] = {}
    all_training_ids: Set[str] = set()
    for role, directory in args.role_data_dir:
        splits = role_splits(Path(directory))
        if splits["train"] & splits["validation"]:
            raise ValueError(f"{role}: train/validation sample_id overlap")
        role_assignments[role] = splits
        all_training_ids |= splits["train"] | splits["validation"]

    role_consistency = {}
    known_roles = list(role_assignments)
    for role in known_roles:
        other_roles = [other for other in known_roles if other != role]
        mismatches = []
        for sample_id in sorted(set().union(*role_assignments[role].values(), *[set().union(*role_assignments[o].values()) for o in other_roles])):
            assignment = "train" if sample_id in role_assignments[role]["train"] else "validation" if sample_id in role_assignments[role]["validation"] else "absent"
            for other in other_roles:
                other_assignment = "train" if sample_id in role_assignments[other]["train"] else "validation" if sample_id in role_assignments[other]["validation"] else "absent"
                if assignment != other_assignment and assignment != "absent" and other_assignment != "absent":
                    mismatches.append({"sample_id": sample_id, "other_role": other, "this": assignment, "other": other_assignment})
        role_consistency[role] = {"mismatch_count": len(mismatches), "examples": mismatches[:20]}

    report = {
        "dev_samples": len(dev_rows_out),
        "dev_sample_ids": sorted(dev_ids),
        "roles": {
            role: {split: len(ids) for split, ids in splits.items()}
            for role, splits in role_assignments.items()
        },
        "dev_overlap": {
            role: sorted(dev_ids & (splits["train"] | splits["validation"]))
            for role, splits in role_assignments.items()
        },
        "shared_split_consistency": role_consistency,
        "clean_dev": not any(dev_ids & (splits["train"] | splits["validation"]) for splits in role_assignments.values()),
    }
    Path(args.dev_out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.dev_out).open("w", encoding="utf-8") as handle:
        for row in dev_rows_out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
