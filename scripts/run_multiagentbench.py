#!/usr/bin/env python3
"""Run the current workflow against a MultiAgentBench/MARBLE task manifest."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.multiagentbench import load_multiagentbench_tasks, summarize_results
from benchmarks.marble_adapter import (
    MARBLEEvaluatorBridge,
    official_trajectory_record,
)
from main import run_workflow


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value[:100] or "task"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _latest_workflow_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            record = json.loads(line)
            if record.get("event") == "workflow_summary":
                return record
    except (OSError, json.JSONDecodeError):
        return {}
    return {}


def _record_failure(task: Any, mode: str, error: Exception, *, log_path: Path | None = None) -> dict[str, Any]:
    summary = _latest_workflow_summary(log_path) if log_path is not None else {}
    quality = summary.get("quality") or {}
    physical_input_tokens = int(summary.get("physical_input_tokens", 0) or 0)
    output_tokens = int(summary.get("output_tokens", 0) or 0)
    return {
        "task_id": task.task_id,
        "execution_mode": mode,
        "task_type": task.task_type,
        "domain": task.domain,
        "coordination_category": task.coordination_category,
        "topology": task.topology,
        "runtime_success": False,
        "answer_status": "not_available",
        "answer_correct": False if task.has_reference else None,
        "physical_input_tokens": physical_input_tokens,
        "output_tokens": output_tokens,
        "total_model_tokens": physical_input_tokens + output_tokens,
        "graph_delta_candidate_tokens": int(summary.get("graph_delta_candidate_tokens", 0) or 0),
        "graph_delta_sent_tokens": int(summary.get("graph_delta_sent_tokens", 0) or 0),
        "graph_delta_sent_nodes": int(summary.get("graph_delta_sent_nodes", 0) or 0),
        "forward_calls": int(summary.get("forward_calls", 0) or 0),
        "retry_count": int(summary.get("retry_count", 0) or 0),
        "completed_stages": int(quality.get("completed_stages", 0) or 0),
        "failed_stages": int(quality.get("failed_stages", 0) or 0),
        "duration_sec": float(summary.get("duration_sec", 0) or 0),
        "error": f"{type(error).__name__}: {error}",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    tasks = load_multiagentbench_tasks(args.input, limit=args.limit, task_ids=set(args.task_id or []))
    if not tasks:
        raise ValueError("no tasks selected from the benchmark manifest")
    modes = tuple(dict.fromkeys(args.execution_mode))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.jsonl"
    if args.overwrite and result_path.exists():
        result_path.unlink()
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    official_marble_requested = bool(getattr(args, "official_marble", False))
    marble_bridge = (
        MARBLEEvaluatorBridge(marble_root=getattr(args, "marble_root", "MARBLE"))
        if official_marble_requested
        else None
    )
    for task in tasks:
        for mode in modes:
            base = output_dir / mode / _safe_name(task.task_id)
            if not task.supported:
                skipped.append({
                    "task_id": task.task_id,
                    "execution_mode": mode,
                    "task_type": task.task_type,
                    "domain": task.domain,
                    "skipped": True,
                    "reason": task.unsupported_reason,
                })
                continue
            try:
                outcome = run_workflow(
                    task.runtime_prompt(),
                    task_type=task.task_type,
                    model_path=args.model_path,
                    max_rounds=args.max_rounds,
                    log_path=str(base / "workflow.jsonl"),
                    reference_answer=task.reference_answer if task.has_reference else None,
                    answer_tolerance=task.answer_tolerance,
                    enable_action_constraints=not args.disable_action_constraints,
                    execution_mode=mode,
                    communication_policy=args.communication_policy,
                    communication_seed=args.communication_seed,
                    communication_budget_tokens=args.communication_budget_tokens,
                    task_id=f"mab_{_safe_name(task.task_id)}",
                )
                summary = dict(outcome.get("telemetry") or {})
                evaluation = dict(outcome.get("evaluation") or {})
                native_trajectory = official_trajectory_record(task, outcome)
                official_marble: dict[str, Any] | None = None
                official_marble_error: str | None = None
                if marble_bridge is not None:
                    if mode != "native_langgraph":
                        official_marble_error = (
                            "official MARBLE graph scoring requires native_langgraph; "
                            "the optimized protocol does not emit a MARBLE message trajectory"
                        )
                    else:
                        try:
                            official_marble = marble_bridge.evaluate(
                                task,
                                outcome,
                                evaluator_model=getattr(args, "marble_evaluator_model", None),
                            )
                        except Exception as error:  # preserve runtime output for diagnosis
                            official_marble_error = f"{type(error).__name__}: {error}"
                            if getattr(args, "strict_official_marble", False):
                                raise
                record = {
                    "task_id": task.task_id,
                    "execution_mode": mode,
                    "task_type": task.task_type,
                    "domain": task.domain,
                    "coordination_category": task.coordination_category,
                    "topology": task.topology,
                    "runtime_success": summary.get("status") == "success",
                    "answer_status": evaluation.get("status"),
                    "answer_correct": evaluation.get("correct"),
                    "final_answer_present": summary.get("final_answer_present", False),
                    "physical_input_tokens": summary.get("physical_input_tokens", 0),
                    "logical_input_tokens": summary.get("logical_input_tokens", 0),
                    "output_tokens": summary.get("output_tokens", 0),
                    "total_model_tokens": int(summary.get("physical_input_tokens", 0) or 0)
                    + int(summary.get("output_tokens", 0) or 0),
                    "graph_delta_candidate_tokens": summary.get("graph_delta_candidate_tokens", 0),
                    "graph_delta_sent_tokens": summary.get("graph_delta_sent_tokens", 0),
                    "graph_delta_sent_nodes": summary.get("graph_delta_sent_nodes", 0),
                    "graph_communication_events": summary.get("graph_communication_events", 0),
                    "retry_count": summary.get("retry_count", 0),
                    "forward_calls": summary.get("forward_calls", 0),
                    "duration_sec": summary.get("duration_sec", 0),
                    "completed_stages": (summary.get("quality") or {}).get("completed_stages", 0),
                    "failed_stages": (summary.get("quality") or {}).get("failed_stages", 0),
                    "milestone_count": len(task.milestones),
                    "milestone_proxy_score": (
                        min(1.0, float((summary.get("quality") or {}).get("completed_stages", 0)) / 4.0)
                        if summary.get("status") == "success" else 0.0
                    ),
                    "quality_source": "reference_answer" if task.has_reference else "runtime_milestone_proxy",
                    "official_marble_status": (
                        "scored" if official_marble is not None
                        else "error" if official_marble_error
                        else "not_requested"
                    ),
                    "official_marble": official_marble,
                    "official_marble_error": official_marble_error,
                    "log_path": str(base / "workflow.jsonl"),
                    "task": task.as_dict(),
                }
                trajectory_payload = {
                    "benchmark": "MultiAgentBench/MARBLE",
                    "task": task.as_dict(),
                    "execution_mode": mode,
                    "final_answer": outcome.get("final_answer"),
                    "evaluation": evaluation,
                    "telemetry": summary,
                    "graph": outcome.get("graph"),
                    "native_state": outcome.get("state"),
                    "native_agents": outcome.get("agents"),
                    "marble_trajectory": native_trajectory,
                    "official_marble": official_marble,
                    "official_marble_error": official_marble_error,
                }
                _write_json(base / "trajectory.json", trajectory_payload)
                final_answer = outcome.get("final_answer")
                if isinstance(final_answer, dict) and isinstance(final_answer.get("code"), str):
                    (base / "solution.py").write_text(final_answer["code"], encoding="utf-8")
            except Exception as error:  # keep the paired comparison running
                record = _record_failure(task, mode, error, log_path=base / "workflow.jsonl")
                record["log_path"] = str(base / "workflow.jsonl")
            with result_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":")) + "\n")
            records.append(record)
    report = {
        "benchmark": "MultiAgentBench/MARBLE",
        "input": str(args.input),
        "model_path": args.model_path,
        "official_marble_requested": official_marble_requested,
        "marble_root": str(getattr(args, "marble_root", "MARBLE")),
        "execution_modes": list(modes),
        "task_count": len(tasks),
        "supported_task_count": sum(int(task.supported) for task in tasks),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "summary": summarize_results(records),
        "results_path": str(result_path),
    }
    _write_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired MultiAgentBench/MARBLE experiments")
    parser.add_argument("--input", required=True, help="official benchmark.jsonl or normalized JSON/JSONL manifest")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--execution-mode",
        action="append",
        choices=("native_langgraph", "optimized"),
        default=None,
        help="repeat to select modes; defaults to both",
    )
    parser.add_argument("--output-dir", default=".runtime/multiagentbench")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-id", action="append", default=None)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--disable-action-constraints", action="store_true")
    parser.add_argument(
        "--communication-policy",
        choices=("send_all", "minimal_no_feedback", "minimal_sendall_fallback", "minimal_targeted_feedback", "random_keep_75", "random_keep_50", "random_keep_25", "closure_aware_heuristic"),
        default="closure_aware_heuristic",
    )
    parser.add_argument("--communication-seed", type=int, default=0)
    parser.add_argument("--communication-budget-tokens", type=int, default=None)
    parser.add_argument(
        "--official-marble",
        action="store_true",
        help=(
            "invoke the official MARBLE Evaluator on native message trajectories; "
            "requires MARBLE dependencies and a judge LLM"
        ),
    )
    parser.add_argument(
        "--marble-root",
        default="MARBLE",
        help="root directory containing the official MARBLE/marble package",
    )
    parser.add_argument(
        "--marble-evaluator-model",
        default=None,
        help="judge model passed to official MARBLE Evaluator (defaults to task config or gpt-4o)",
    )
    parser.add_argument(
        "--strict-official-marble",
        action="store_true",
        help="fail a task when official MARBLE scoring cannot be completed",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.execution_mode is None:
        args.execution_mode = (
            ["native_langgraph"] if args.official_marble
            else ["native_langgraph", "optimized"]
        )
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
