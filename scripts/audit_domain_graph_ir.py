"""Audit canonical normalized domain Graph IR split consistency."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--report-out", default="")
    args = parser.parse_args()
    root = Path(args.data_dir)
    split_ids: Dict[str, Set[str]] = {}
    counts: Dict[str, int] = {}
    errors: List[str] = []
    required = {"sample_id", "domain", "task_type", "question", "facts", "evidence", "query_spec", "gold_plan", "gold_answer", "source"}
    for split in ("train", "validation", "test"):
        path = root / f"{split}.jsonl"
        if not path.is_file():
            continue
        ids: Set[str] = set()
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                count += 1
                missing = required - set(row)
                if missing:
                    errors.append(f"{path}:{line_no}: missing {sorted(missing)}")
                if row.get("domain") != args.domain:
                    errors.append(f"{path}:{line_no}: domain={row.get('domain')!r}")
                sample_id = row.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    errors.append(f"{path}:{line_no}: invalid sample_id")
                elif sample_id in ids:
                    errors.append(f"{path}:{line_no}: duplicate sample_id within split")
                ids.add(sample_id)
        split_ids[split] = ids
        counts[split] = count
    for left, left_ids in split_ids.items():
        for right, right_ids in split_ids.items():
            if left < right and left_ids & right_ids:
                errors.append(f"split overlap {left}/{right}: {len(left_ids & right_ids)}")
    report = {"domain": args.domain, "data_dir": str(root), "counts": counts, "unique_ids": len(set().union(*split_ids.values())) if split_ids else 0, "errors": errors}
    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
