#!/usr/bin/env python3
"""Paired bootstrap comparison for HotpotQA AFlow context experiments."""
from __future__ import annotations
import argparse, json, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "AFlow"))
from benchmarks.hotpotqa import HotpotQABenchmark

def _unwrap(path):
    data = json.loads(Path(path).read_text())
    run = next(iter(data["runs"].values()))
    run = next(iter(run.values())) if isinstance(run, dict) and "score" not in run else run
    return data, run


def _sample_scores(data, run):
    """Return per-sample scores without applying a Hotpot scorer to MBPP."""
    records = run.get("records", [])
    record_ids = run.get("record_ids", [])
    dataset = str(data.get("dataset", run.get("dataset", ""))).lower()
    scores = {}
    for i, row in enumerate(records):
        sample_id = str(record_ids[i] if i < len(record_ids) else i)
        if dataset == "mbpp":
            # MBPP records store the task score in column 3.  Do not pass
            # code/critique through the HotpotQA scorer.
            try:
                scores[sample_id] = float(row[3])
            except (IndexError, TypeError, ValueError):
                scores[sample_id] = 0.0
        else:
            scorer = HotpotQABenchmark("HotpotQA", "", "")
            scores[sample_id] = float(scorer.calculate_score(str(row[3]), str(row[2]))[0])
    return scores


def _merge_numeric_maps(target, value):
    if not isinstance(value, dict):
        return
    for key, amount in value.items():
        try:
            target[str(key)] = target.get(str(key), 0) + float(amount)
        except (TypeError, ValueError):
            continue


def run(path):
    data, run = _unwrap(path)
    records = run.get("records", [])
    record_ids = run.get("record_ids", [])
    scores = _sample_scores(data, run)
    toks = {}
    for row in run.get("model_call_records", []):
        sample_id = str(row.get("problem_id", row.get("problem_index", 0)))
        toks[sample_id] = toks.get(sample_id, 0) + int(row.get("input_tokens", 0))
    plans = {}
    artifact = {
        "route_events_by_operator": {},
        "original_tokens_by_operator": {},
        "sendall_tokens_by_operator": {},
        "minctx_tokens_by_operator": {},
        "saved_tokens_by_operator": {},
        "fallbacks_by_operator": {},
        "contract_failures_by_operator": {},
        "route_count": 0,
        "fallback_count": 0,
        "contract_failure_count": 0,
        "multiplicity_expected": 0.0,
        "multiplicity_delivered": 0.0,
        "compile_ms": 0.0,
        "route_ms": 0.0,
        "compression_opportunity": None,
    }
    for summary in run.get("graph_telemetry", []) or []:
        for plan in summary.get("aflow_context_plans", []) or []:
            plans[str(plan.get("sample_id", ""))] = plan
        for field in (
            "aflow_artifact_route_events_by_operator",
            "aflow_artifact_original_tokens_by_operator",
            "aflow_artifact_sendall_tokens_by_operator",
            "aflow_artifact_minctx_tokens_by_operator",
            "aflow_artifact_saved_tokens_by_operator",
            "aflow_artifact_fallbacks_by_operator",
            "aflow_artifact_contract_failures_by_operator",
        ):
            _merge_numeric_maps(artifact[field.removeprefix("aflow_artifact_")], summary.get(field))
        artifact["route_count"] += int(summary.get("aflow_artifact_route_count", 0) or 0)
        artifact["fallback_count"] += int(summary.get("aflow_artifact_fallback_count", 0) or 0)
        artifact["contract_failure_count"] += int(summary.get("aflow_artifact_contract_failures", 0) or 0)
        artifact["multiplicity_expected"] += float(summary.get("aflow_artifact_multiplicity_expected", 0) or 0)
        artifact["multiplicity_delivered"] += float(summary.get("aflow_artifact_multiplicity_delivered", 0) or 0)
        artifact["compile_ms"] += float(summary.get("aflow_artifact_compile_ms", 0) or 0)
        artifact["route_ms"] += float(summary.get("aflow_artifact_selection_overhead_ms", 0) or 0)
        opportunity = summary.get("aflow_artifact_compression_opportunity")
        if opportunity is not None:
            artifact["compression_opportunity"] = bool(opportunity) if artifact["compression_opportunity"] is None else artifact["compression_opportunity"] or bool(opportunity)
    return data, run, scores, toks, plans, artifact

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True); p.add_argument("--candidate", required=True)
    p.add_argument("--bootstrap-samples", type=int, default=10000); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output")
    a = p.parse_args(); bd, b, bf, bt, bp, ba = run(a.baseline); cd, c, cf, ct, cp, ca = run(a.candidate)
    if set(bf) != set(cf):
        raise SystemExit(f"record_ids mismatch: baseline={len(bf)} candidate={len(cf)}")
    ids = sorted(bf); deltas = [cf[i] - bf[i] for i in ids]
    savings = [1.0 - ct.get(i, 0) / max(1, bt.get(i, 0)) for i in ids]
    rng = random.Random(a.seed); boot_d = []; boot_s = []
    for _ in range(a.bootstrap_samples):
        sample = [rng.choice(range(len(ids))) for _ in ids]
        boot_d.append(sum(deltas[j] for j in sample) / len(sample)); boot_s.append(sum(savings[j] for j in sample) / len(sample))
    def ci(values):
        values = sorted(values); return [values[int(.025 * len(values))], values[int(.975 * len(values))]]
    candidate_summaries = c.get("graph_telemetry", []) or []
    candidate_sample_plans = {str(t.get("sample_id")): (t.get("aflow_context_plans", []) or [])[-1]
                              for t in candidate_summaries if t.get("sample_id") and t.get("aflow_context_plans")}
    levels = {str(p.get("effective_level")) for p in candidate_sample_plans.values()}
    level_names = tuple(sorted(levels | {"L0", "L1", "L2", "S0", "S1", "S2"}))
    level_usage = {k: sum(candidate_sample_plans.get(i, {}).get("effective_level") == k for i in ids) for k in level_names}
    fallback_samples = sum(bool(candidate_sample_plans.get(i, {}).get("fallback")) for i in ids)
    # The Hotpot fields remain for backwards compatibility.  For MBPP the
    # same fields now contain true task-score deltas rather than Hotpot F1.
    out = {
        "dataset": cd.get("dataset", c.get("dataset", "")),
        "score_metric": "mbpp_task_score" if str(cd.get("dataset", c.get("dataset", ""))).lower() == "mbpp" else "hotpotqa_f1",
        "baseline_score": float(b.get("score", 0)), "candidate_score": float(c.get("score", 0)),
        "delta_score": float(sum(deltas) / max(1, len(deltas))),
        "delta_f1": float(sum(deltas) / max(1, len(deltas))), "delta_f1_ci95": ci(boot_d),
        "token_saving": 1.0 - c.get("input_tokens", 0) / max(1, b.get("input_tokens", 0)),
        "physical_prefill_tokens_baseline": int(b.get("physical_prefill_tokens", 0)),
        "physical_prefill_tokens_candidate": int(c.get("physical_prefill_tokens", 0)),
        "physical_prefill_saving": 1.0 - c.get("physical_prefill_tokens", 0) / max(1, b.get("physical_prefill_tokens", 0)),
        "llm_calls_baseline": int(b.get("model_calls", 0)),
        "llm_calls_candidate": int(c.get("model_calls", 0)),
        "wall_time_baseline": float(b.get("wall_time_seconds", 0.0)),
        "wall_time_candidate": float(c.get("wall_time_seconds", 0.0)),
        "token_saving_ci95": ci(boot_s), "noninferiority_margin": 0.02,
        "probability_delta_ge_negative_margin": sum(x >= -.02 for x in boot_d) / max(1, len(boot_d)),
        "correct_to_wrong": sum(bf[i] > 0 and cf[i] == 0 for i in ids),
        "wrong_to_correct": sum(bf[i] == 0 and cf[i] > 0 for i in ids),
        "fallback_rate": fallback_samples / max(1, len(ids)),
        "artifact_fallback_rate": ca["fallback_count"] / max(1, len(ids)),
        "sample_fallback_count": fallback_samples,
        "adaptive_recovery_rate": sum(len(candidate_sample_plans.get(i, {}).get("attempted_levels", [])) > 1 for i in ids) / max(1, len(ids)),
        "selector_overhead_ms": sum(float(t.get("aflow_context_selection_overhead_ms", 0) or 0) for t in c.get("graph_telemetry", [])),
        "artifact_compile_route_overhead_ms": ca["compile_ms"] + ca["route_ms"],
        "level_usage": level_usage, "sample_count": len(ids),
        "l2_subset_count": sum(candidate_sample_plans.get(i, {}).get("effective_level") in {"L2", "S2"} for i in ids),
        "l2_subset_f1": sum(cf[i] for i in ids if candidate_sample_plans.get(i, {}).get("effective_level") in {"L2", "S2"}) / max(1, sum(candidate_sample_plans.get(i, {}).get("effective_level") in {"L2", "S2"} for i in ids)),
        "l2_subset_baseline_f1": sum(bf[i] for i in ids if candidate_sample_plans.get(i, {}).get("effective_level") in {"L2", "S2"}) / max(1, sum(candidate_sample_plans.get(i, {}).get("effective_level") in {"L2", "S2"} for i in ids)),
        "artifact": {
            "baseline": ba,
            "candidate": ca,
            "operator_token_saving": {
                op: 1.0 - ca["minctx_tokens_by_operator"].get(op, ca["original_tokens_by_operator"].get(op, 0.0)) / max(1.0, ca["original_tokens_by_operator"].get(op, 0.0))
                for op in sorted(ca["original_tokens_by_operator"])
            },
            "multiplicity_preservation": (
                None if ca["multiplicity_expected"] <= 0
                else ca["multiplicity_delivered"] / ca["multiplicity_expected"]
            ),
        },
    }
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if a.output: Path(a.output).write_text(text + "\n")
    print(text)
if __name__ == "__main__": main()
