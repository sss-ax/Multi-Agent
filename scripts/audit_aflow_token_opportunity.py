#!/usr/bin/env python3
"""Report per-operator token opportunity for Artifact routing.

This is a read-only audit over a reproduction JSON result; it does not invoke
models and deliberately keeps optimization sources separate.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True)
    p.add_argument("--output")
    args = p.parse_args()
    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    run = next(iter(data.get("runs", {}).values()))
    run = next(iter(run.values())) if isinstance(run, dict) and "records" not in run else run
    by_operator = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "duplicate_candidate_tokens": 0})
    for row in run.get("model_call_records", []):
        op = str(row.get("operator", row.get("name", "unknown")))
        bucket = by_operator[op]
        bucket["calls"] += 1
        bucket["input_tokens"] += int(row.get("input_tokens", row.get("prompt_tokens", 0)) or 0)
        bucket["output_tokens"] += int(row.get("output_tokens", 0) or 0)
    summary = run.get("graph_telemetry", []) or []
    for item in summary:
        for key, value in item.items():
            if key in {"aflow_duplicate_candidate_occurrences", "aflow_dedup_saved_tokens"}:
                by_operator["__dedup__"][key] = by_operator["__dedup__"].get(key, 0) + int(value or 0)
    total = sum(v["input_tokens"] for k, v in by_operator.items() if k != "__dedup__")
    for value in by_operator.values():
        value["input_share"] = value.get("input_tokens", 0) / max(1, total)
    out = {"sample_count": len(run.get("records", [])), "total_input_tokens": total, "operators": dict(sorted(by_operator.items()))}
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output: Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
if __name__ == "__main__": main()
