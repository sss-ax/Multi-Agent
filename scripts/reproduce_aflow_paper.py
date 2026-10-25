"""Replay AFlow's published workflows with the official benchmark evaluators.

This is intentionally separate from ``evaluate_external_baselines.py``.  The
external-baseline command uses this repository's normalized task schema and
scorer; this command uses AFlow's own dataset files, workflow entry points, and
benchmark implementations so paper-style results remain auditable.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import re
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
AFLOW_ROOT = ROOT / "third_party" / "AFlow"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.aflow_runner import (  # noqa: E402
    _AFlowLLMBridge,
    _aflow_import_context,
    _patched_aflow_llm,
)
from workflow_runtime.model_backend import DirectTransformersModel  # noqa: E402


DATASETS = ("DROP", "HotpotQA", "MATH", "GSM8K", "MBPP", "HumanEval")
MATH_PAPER_CATEGORIES = {
    "counting and probability",
    "number theory",
    "prealgebra",
    "precalculus",
}
PAPER_TEST_ROUNDS = {
    "DROP": 3,
    "HotpotQA": 3,
    "MATH": 5,
    "GSM8K": 10,
    "MBPP": 14,
    "HumanEval": 5,
}


def _benchmark_class(name: str) -> type[Any]:
    from benchmarks.drop import DROPBenchmark
    from benchmarks.gsm8k import GSM8KBenchmark
    from benchmarks.hotpotqa import HotpotQABenchmark
    from benchmarks.humaneval import HumanEvalBenchmark
    from benchmarks.math import MATHBenchmark
    from benchmarks.mbpp import MBPPBenchmark

    return {
        "DROP": DROPBenchmark,
        "HotpotQA": HotpotQABenchmark,
        "MATH": MATHBenchmark,
        "GSM8K": GSM8KBenchmark,
        "MBPP": MBPPBenchmark,
        "HumanEval": HumanEvalBenchmark,
    }[name]


def _data_path(dataset: str, split: str) -> Path:
    return AFLOW_ROOT / "data" / "datasets" / f"{dataset.lower()}_{split}.jsonl"


def _filter_rows(dataset: str, rows: list[dict[str, Any]], *, math_paper_subset: bool) -> list[dict[str, Any]]:
    if dataset != "MATH" or not math_paper_subset:
        return rows
    filtered: list[dict[str, Any]] = []
    for row in rows:
        level = str(row.get("level", "")).strip().lower()
        category = str(row.get("type", row.get("category", ""))).strip().lower().replace("&", "and")
        if level in {"level 5", "5"} and category in MATH_PAPER_CATEGORIES:
            filtered.append(row)
    return filtered


def _paper_workflow_dir(dataset: str, workflow_round: int) -> Path:
    return AFLOW_ROOT / "data" / "results" / "results" / dataset / "graphs_test" / f"round_{workflow_round}"


def _load_paper_workflow(dataset: str, workflow_round: int, bridge: Any) -> tuple[Any, str]:
    """Load a downloaded paper graph while using the repository's local operators.

    The archived graphs import the older MetaGPT package layout.  Their graph
    logic and prompts are still usable: the current vendored AFlow workspace
    provides compatible operators, so only those package imports are remapped.
    """
    workflow_dir = _paper_workflow_dir(dataset, workflow_round)
    graph_path = workflow_dir / "graph.py"
    prompt_path = workflow_dir / "prompt.py"
    if not graph_path.exists() or not prompt_path.exists():
        raise FileNotFoundError(f"Paper Workflow files are missing under {workflow_dir}")

    prompt_name = f"_aflow_paper_prompt_{dataset}_{workflow_round}"
    prompt_spec = importlib.util.spec_from_file_location(prompt_name, prompt_path)
    if prompt_spec is None or prompt_spec.loader is None:
        raise ImportError(f"Cannot load paper prompt: {prompt_path}")
    prompt_module = importlib.util.module_from_spec(prompt_spec)
    sys.modules[prompt_name] = prompt_module
    prompt_spec.loader.exec_module(prompt_module)

    local_operator = f"from workspace.{dataset}.workflows.template import operator"
    transformed: list[str] = []
    for line in graph_path.read_text(encoding="utf-8").splitlines():
        if re.match(r"^import .* as operator$", line.strip()):
            transformed.append(local_operator)
        elif re.match(r"^import .* as prompt_custom$", line.strip()):
            transformed.append(f"import {prompt_name} as prompt_custom")
        elif line.strip() == "from metagpt.provider.llm_provider_registry import create_llm_instance":
            transformed.append("from scripts.async_llm import create_llm_instance")
        elif line.strip() == "from metagpt.utils.cost_manager import CostManager":
            transformed.extend(["class CostManager:", "    def __init__(self):", "        self.total_cost = 0.0"])
        elif line.strip().startswith("from metagpt") or line.strip().startswith("import metagpt"):
            raise ImportError(f"Unsupported MetaGPT import in paper Workflow: {line.strip()}")
        else:
            transformed.append(line)

    module_name = f"_aflow_paper_graph_{dataset}_{workflow_round}"
    module = types.ModuleType(module_name)
    module.__file__ = str(graph_path)
    module.__package__ = ""
    sys.modules[module_name] = module
    exec(compile("\n".join(transformed) + "\n", str(graph_path), "exec"), module.__dict__)
    module.create_llm_instance = lambda _config: bridge
    return module, str(graph_path)


async def _load_rows(benchmark: Any, limit: int | None) -> list[dict[str, Any]]:
    rows = await benchmark.load_data()
    return rows if limit is None else rows[:limit]


async def _run_one(
    *,
    dataset: str,
    split: str,
    workflow_round: int,
    model: Any,
    limit: int | None,
    math_paper_subset: bool,
    log_dir: Path,
    generation_timeout_seconds: float | None,
    resume_result: dict[str, Any] | None,
    save_progress: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    data_path = _data_path(dataset, split)
    if not data_path.exists():
        raise FileNotFoundError(
            f"AFlow data file is missing: {data_path}. "
            "Run `PYTHONPATH=third_party/AFlow .venv/bin/python -m data.download_data` "
            "from third_party/AFlow or use the download command in the README."
        )

    progress: Any = None
    log_dir.mkdir(parents=True, exist_ok=True)
    call_log_path = log_dir / "model_calls.jsonl"
    if resume_result is None:
        call_log_path.write_text("", encoding="utf-8")

    def report_model_call(event: dict[str, Any]) -> None:
        persisted_event = {"timestamp": time.time(), **event}
        with call_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(persisted_event, ensure_ascii=False, default=str) + "\n")
        event_name = str(event["event"])
        problem_number = int(event.get("problem_index", 0)) + 1
        problem_call = int(event["call_number"]) - int(event.get("call_offset", 0))
        if event_name == "start":
            stage = f"item {problem_number}: LLM {problem_call} generating"
            timeout = event.get("timeout_seconds")
            timeout_text = f"{timeout}s" if timeout else "off"
            detail = (
                f"[LLM start] {dataset} item={problem_number} call={problem_call} "
                f"input_tokens={event['input_tokens']} timeout={timeout_text}"
            )
        else:
            status = event_name if event_name in {"timeout", "error"} else "done"
            stage = f"item {problem_number}: LLM {problem_call} {status}"
            detail = (
                f"[LLM {status}] {dataset} item={problem_number} call={problem_call} "
                f"elapsed={float(event['elapsed_seconds']):.2f}s "
                f"output_tokens={event['output_tokens']}"
            )
            if event.get("error"):
                detail += f" error={event['error']}"
        if progress is not None:
            progress.set_postfix_str(stage, refresh=True)
            progress.write(detail)
        else:
            print(detail, flush=True)

    bridge = _AFlowLLMBridge(
        model,
        model.tokenizer,
        call_timeout_seconds=generation_timeout_seconds,
        event_callback=report_model_call,
    )
    with _aflow_import_context(AFLOW_ROOT), _patched_aflow_llm(bridge):
        print(f"[setup] {dataset}: loading benchmark rows", flush=True)
        benchmark_type = _benchmark_class(dataset)
        benchmark = benchmark_type(name=dataset, file_path=str(data_path), log_path=str(log_dir))
        rows = _filter_rows(dataset, await _load_rows(benchmark, None), math_paper_subset=math_paper_subset)
        if limit is not None:
            rows = rows[:limit]
        if not rows:
            raise RuntimeError(f"No rows selected for {dataset} {split}")
        graph_path_hint = _paper_workflow_dir(dataset, workflow_round) / "graph.py"
        result = _initial_run_result(
            dataset=dataset,
            split=split,
            workflow_round=workflow_round,
            graph_path=str(graph_path_hint),
            data_path=data_path,
            resume_result=resume_result,
        )
        completed = int(result["completed_rows"])
        expected_ids = [_row_identifier(row, index) for index, row in enumerate(rows[:completed])]
        if result["record_ids"] != expected_ids:
            raise RuntimeError(
                f"Checkpoint rows do not match the selected {dataset} data. "
                "Use a different --output path or restart without --resume."
            )
        result["setup_stage"] = "loading_workflow"
        result["model_call_log"] = str(call_log_path)
        save_progress(result)
        print(f"[setup] {dataset}: loading paper workflow round {workflow_round}", flush=True)
        workflow_module, graph_path = _load_paper_workflow(dataset, workflow_round, bridge)
        workflow = workflow_module.Workflow(
            name=f"aflow_paper_replay_{dataset}",
            llm_config={"model": "local-qwen"},
            dataset=dataset,
        )
        result["workflow_module"] = graph_path
        result["setup_stage"] = "evaluating"
        print(f"[setup] {dataset}: workflow ready", flush=True)

        started = time.time()
        prior_wall_time = float(result.get("wall_time_seconds", 0.0))
        prior_usage = {
            "model_calls": int(result.get("model_calls", 0)),
            "timeout_count": int(result.get("timeout_count", 0)),
            "input_tokens": int(result.get("input_tokens", 0)),
            "output_tokens": int(result.get("output_tokens", 0)),
        }
        save_progress(result)
        progress = tqdm(
            total=len(rows),
            initial=completed,
            desc=f"Evaluating {dataset} problems",
            unit="item",
            dynamic_ncols=True,
        )
        try:
            for row_index in range(completed, len(rows)):
                call_offset = len(bridge.calls)
                bridge.set_call_context(
                    dataset=dataset,
                    problem_index=row_index,
                    problem_id=_row_identifier(rows[row_index], row_index),
                    call_offset=call_offset,
                )
                progress.set_postfix_str(f"item {row_index + 1}: evaluating", refresh=True)
                item_started = time.time()
                record = await benchmark.evaluate_problem(rows[row_index], workflow)
                item_calls = bridge.calls[call_offset:]
                result["records"].append(record)
                result["record_ids"].append(_row_identifier(rows[row_index], row_index))
                result["model_call_records"].extend(_compact_call_record(call) for call in item_calls)
                result["item_wall_times_seconds"].append(time.time() - item_started)
                result["completed_rows"] = row_index + 1
                result["count"] = row_index + 1
                _refresh_run_totals(result, benchmark, bridge, prior_usage, prior_wall_time, started)
                save_progress(result)
                progress.update(1)
                progress.set_postfix_str(f"item {row_index + 1}: checkpoint saved", refresh=True)
        finally:
            progress.close()

    result["status"] = "complete"
    result["setup_stage"] = "complete"
    _refresh_run_totals(result, benchmark, bridge, prior_usage, prior_wall_time, started)
    save_progress(result)
    return result


def _row_identifier(row: dict[str, Any], index: int) -> str:
    for key in ("id", "_id", "task_id", "question_id"):
        if key in row:
            return f"{key}:{row[key]}"
    canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _compact_call_record(call: dict[str, Any]) -> dict[str, Any]:
    return {
        key: call[key]
        for key in (
            "dataset",
            "problem_index",
            "problem_id",
            "input_tokens",
            "output_tokens",
            "elapsed_seconds",
            "timed_out",
            "error",
        )
        if key in call
    }


def _score_metric(dataset: str) -> str:
    if dataset in {"MBPP", "HumanEval"}:
        return "pass@1"
    if dataset in {"DROP", "HotpotQA"}:
        return "f1"
    return "solve_rate"


def _initial_run_result(
    *,
    dataset: str,
    split: str,
    workflow_round: int,
    graph_path: str,
    data_path: Path,
    resume_result: dict[str, Any] | None,
) -> dict[str, Any]:
    if resume_result is not None:
        result = dict(resume_result)
        result.setdefault("status", "complete" if "score" in result else "running")
        result.setdefault("completed_rows", len(result.get("records", [])))
        result.setdefault("record_ids", [])
        result.setdefault("model_call_records", [])
        result.setdefault("item_wall_times_seconds", [])
        return result
    return {
        "status": "running",
        "dataset": dataset,
        "split": split,
        "workflow_round": workflow_round,
        "workflow_module": graph_path,
        "data_path": str(data_path),
        "count": 0,
        "completed_rows": 0,
        "score": 0.0,
        "score_metric": _score_metric(dataset),
        "model_calls": 0,
        "timeout_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_model_tokens": 0,
        "wall_time_seconds": 0.0,
        "records": [],
        "record_ids": [],
        "model_call_records": [],
        "item_wall_times_seconds": [],
    }


def _refresh_run_totals(
    result: dict[str, Any],
    benchmark: Any,
    bridge: _AFlowLLMBridge,
    prior_usage: dict[str, int],
    prior_wall_time: float,
    started: float,
) -> None:
    columns = benchmark.get_result_columns()
    score_index = columns.index("score")
    scores = [float(record[score_index]) for record in result["records"]]
    usage = bridge.get_usage_summary()
    result["score"] = sum(scores) / len(scores) if scores else 0.0
    result["model_calls"] = prior_usage["model_calls"] + int(usage["call_count"])
    result["timeout_count"] = prior_usage["timeout_count"] + int(usage["timeout_count"])
    result["input_tokens"] = prior_usage["input_tokens"] + int(usage["total_input_tokens"])
    result["output_tokens"] = prior_usage["output_tokens"] + int(usage["total_output_tokens"])
    result["total_model_tokens"] = result["input_tokens"] + result["output_tokens"]
    result["wall_time_seconds"] = prior_wall_time + time.time() - started


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS + ("all",), default="all")
    parser.add_argument("--split", choices=("validate", "test"), default="test")
    parser.add_argument(
        "--workflow-round",
        type=int,
        default=0,
        help="0 selects the downloaded paper test round for each dataset",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows in the selected official split")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--math-paper-subset", action="store_true")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="Maximum generated tokens per call; 0 leaves output uncapped within the model context window",
    )
    parser.add_argument(
        "--chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render each AFlow user prompt with the model tokenizer's chat template",
    )
    parser.add_argument(
        "--generation-timeout-seconds",
        type=float,
        default=300.0,
        help=(
            "Stop an individual Transformers generation after this many seconds "
            "(checked between generated tokens); 0 disables the deadline"
        ),
    )
    parser.add_argument("--output", default=".runtime/aflow_paper_reproduction/results.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.runs < 1:
        raise SystemExit("--runs must be positive")
    if args.generation_timeout_seconds < 0:
        raise SystemExit("--generation-timeout-seconds cannot be negative")
    if args.max_new_tokens < 0:
        raise SystemExit("--max-new-tokens cannot be negative")
    selected = list(DATASETS if args.dataset == "all" else (args.dataset,))
    limit = args.limit or None
    output = Path(args.output)
    missing = [_data_path(dataset, args.split) for dataset in selected if not _data_path(dataset, args.split).exists()]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(
            "AFlow official data is missing:\n"
            f"{missing_text}\n"
            "Download it before loading the model. See third_party/AFlow/data/download_data.py."
        )
    payload: dict[str, Any] = {
        "experiment": "aflow_paper_reproduction",
        "model_path": args.model_path,
        "split": args.split,
        "seed": 42,
        "validation_fraction": 0.2,
        "test_fraction": 0.8,
        "workflow_round": args.workflow_round,
        "workflow_rounds": {
            dataset: args.workflow_round or PAPER_TEST_ROUNDS[dataset] for dataset in selected
        },
        "math_paper_subset": args.math_paper_subset,
        "max_new_tokens": args.max_new_tokens or None,
        "chat_template": args.chat_template,
        "generation_timeout_seconds": args.generation_timeout_seconds or None,
        "requested_limit": limit,
        "runs": {},
    }
    if args.resume and output.exists():
        saved_payload = json.loads(output.read_text(encoding="utf-8"))
        expected_config = {
            "model_path": args.model_path,
            "split": args.split,
            "math_paper_subset": args.math_paper_subset,
            "max_new_tokens": args.max_new_tokens or None,
            "chat_template": args.chat_template,
            "requested_limit": limit,
        }
        mismatches = [
            key for key, expected in expected_config.items()
            if saved_payload.get(key) != expected
        ]
        if mismatches:
            details = ", ".join(
                f"{key}: checkpoint={saved_payload.get(key)!r}, requested={expected_config[key]!r}"
                for key in mismatches
            )
            raise SystemExit(f"Cannot resume with different experiment settings ({details})")
        payload.update(saved_payload)
        payload["generation_timeout_seconds"] = args.generation_timeout_seconds or None
        payload.setdefault("runs", {})

    model = DirectTransformersModel(
        args.model_path,
        max_new_tokens=args.max_new_tokens or None,
        use_chat_template=args.chat_template,
    )
    for dataset in selected:
        dataset_runs = payload["runs"].setdefault(dataset, {})
        workflow_round = args.workflow_round or PAPER_TEST_ROUNDS[dataset]
        for run_index in range(1, args.runs + 1):
            key = str(run_index)
            saved_run = dataset_runs.get(key) if args.resume else None
            if saved_run is not None and (
                saved_run.get("status") == "complete" or "status" not in saved_run
            ):
                print(f"skip {dataset} run={run_index} (checkpoint)", flush=True)
                continue
            print(f"start {dataset} run={run_index}", flush=True)

            def save_run_progress(result: dict[str, Any]) -> None:
                dataset_runs[key] = result
                _write_checkpoint(output, payload)

            result = asyncio.run(
                _run_one(
                    dataset=dataset,
                    split=args.split,
                    workflow_round=workflow_round,
                    model=model,
                    limit=limit,
                    math_paper_subset=args.math_paper_subset,
                    log_dir=output.parent / "aflow_logs" / dataset / f"run_{run_index}",
                    generation_timeout_seconds=args.generation_timeout_seconds or None,
                    resume_result=saved_run,
                    save_progress=save_run_progress,
                )
            )
            dataset_runs[key] = result
            _write_checkpoint(output, payload)
            print(f"done {dataset} run={run_index} score={result['score']:.4f}", flush=True)

    for dataset in selected:
        completed = [
            item["score"]
            for item in payload["runs"].get(dataset, {}).values()
            if item.get("status") == "complete" or "status" not in item
        ]
        if completed:
            payload.setdefault("summary", {})[dataset] = {
                "runs_completed": len(completed),
                "mean_score": sum(completed) / len(completed),
                "scores": completed,
            }
    _write_checkpoint(output, payload)
    print(json.dumps(payload.get("summary", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
