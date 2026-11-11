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
from baselines.aflow_graph_adapter import (  # noqa: E402
    AFlowGraphSession,
    make_aflow_operator_proxy_module,
)
from baselines.aflow_profiles import aflow_profile_names, get_aflow_profile  # noqa: E402
from workflow_runtime.communication import make_communication_policy  # noqa: E402
from workflow_runtime.model_backend import DirectTransformersModel  # noqa: E402
from workflow_runtime.telemetry import WorkflowTelemetry  # noqa: E402


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


def _load_paper_workflow(
    dataset: str,
    workflow_round: int,
    bridge: Any,
    *,
    graph_session: AFlowGraphSession | None = None,
    communication_mode: str = "original",
) -> tuple[Any, str]:
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
    proxy_module_name = ""
    if graph_session is not None and communication_mode != "original":
        original_operator = importlib.import_module(
            f"workspace.{dataset}.workflows.template.operator"
        )
        safe_task_id = re.sub(r"[^a-zA-Z0-9_]+", "_", graph_session.task_id)
        proxy_module_name = (
            f"_aflow_operator_proxy_{dataset}_{workflow_round}_{safe_task_id}"
        )
        proxy_operator = make_aflow_operator_proxy_module(
            original_operator,
            session=graph_session,
            communication_mode=communication_mode,
        )
        sys.modules[proxy_module_name] = proxy_operator
        local_operator = f"import {proxy_module_name} as operator"
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
    # Read the exact benchmark JSONL through the local filesystem.  AFlow's
    # aiofiles loader can stall before evaluation in some environments; the
    # official scorer still receives the same parsed rows.
    data_path = Path(benchmark.file_path)
    with data_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
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
    communication_mode: str,
    communication_policy: str,
    model_fingerprint: str,
    aflow_content_dedup: bool,
    aflow_invocation_reuse: bool,
    aflow_dedup_mode: str,
    aflow_profile: Any,
    aflow_profile_explicit: bool,
    aflow_context_policy: str,
    aflow_context_budget_tokens: int,
    aflow_context_top_k_documents: int,
    aflow_context_fallback: str,
    aflow_context_recovery: str,
    aflow_context_recovery_top_k_documents: int,
    aflow_semantic_max_paths: int,
    aflow_semantic_max_document_hops: int,
    aflow_semantic_budget_tokens: int,
    aflow_semantic_entity_policy: str,
    artifact_routing_policy: str,
    artifact_fallback: str,
    artifact_preserve_multiplicity: bool,
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
    graph_log_path = log_dir / "graph_1.jsonl"
    if resume_result is None:
        call_log_path.write_text("", encoding="utf-8")
        # A graph telemetry file is append-only by design, so explicitly
        # truncate it at the beginning of a fresh run.  Otherwise a later
        # audit can mix events from multiple experiments.
        graph_log_path.write_text("", encoding="utf-8")

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
                f"output_tokens={event['output_tokens']} "
                f"prefill_tokens={event.get('physical_prefill_tokens', '-') } "
                f"cache_hit_tokens={event.get('prefix_cache_hit_tokens', '-') }"
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
            communication_mode=communication_mode,
            communication_policy=communication_policy,
            aflow_profile=aflow_profile.profile_name if aflow_profile else "none",
            profile_validated=bool(aflow_profile and aflow_profile.validated),
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
        graph_path = str(_paper_workflow_dir(dataset, workflow_round) / "graph.py")
        workflow = None
        if communication_mode == "original":
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
            "physical_prefill_tokens": int(result.get("physical_prefill_tokens", 0)),
            "prefix_cache_hit_tokens": int(result.get("prefix_cache_hit_tokens", 0)),
            "prefix_cache_miss_tokens": int(result.get("prefix_cache_miss_tokens", 0)),
            "prefix_cache_warmup_tokens": int(result.get("prefix_cache_warmup_tokens", 0)),
            "prefix_cache_hits": int(result.get("prefix_cache_hits", 0)),
            "prefix_cache_misses": int(result.get("prefix_cache_misses", 0)),
            "transformer_forward_calls": int(result.get("transformer_forward_calls", 0)),
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
                    cache_namespace=f"aflow:{dataset}:{_row_identifier(rows[row_index], row_index)}",
                )
                progress.set_postfix_str(f"item {row_index + 1}: evaluating", refresh=True)
                item_started = time.time()
                graph_session = None
                item_workflow = workflow
                if communication_mode != "original":
                    row_id = _row_identifier(rows[row_index], row_index)
                    graph_log_path = log_dir / f"graph_{row_index + 1}.jsonl"
                    if resume_result is None:
                        graph_log_path.write_text("", encoding="utf-8")
                    graph_session = AFlowGraphSession(
                        task_id=row_id,
                        task=_row_task_text(rows[row_index]),
                        dataset=dataset,
                        workflow_round=workflow_round,
                        task_metadata={
                            "entry_point": rows[row_index].get("entry_point", ""),
                            # Public HotpotQA context is used only to build
                            # source artifacts.  Gold supporting_facts and
                            # answer labels are intentionally not copied.
                            "question": rows[row_index].get("question", ""),
                            "context": rows[row_index].get("context", []) if dataset == "HotpotQA" else [],
                        },
                        profile=aflow_profile,
                        profile_explicit=aflow_profile_explicit,
                        run_id=f"aflow:{dataset}:{row_id}",
                        policy=make_communication_policy(
                            communication_policy,
                            task_family=dataset,
                        ),
                        telemetry=WorkflowTelemetry(
                            graph_log_path
                        ),
                        content_dedup=aflow_content_dedup,
                        invocation_reuse=aflow_invocation_reuse,
                        dedup_mode=aflow_dedup_mode,
                        model_fingerprint=model_fingerprint,
                        generation_config={
                            "max_new_tokens": getattr(model, "max_new_tokens", None),
                            "use_chat_template": getattr(model, "use_chat_template", None),
                            "do_sample": False,
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "seed": 42,
                        },
                        tokenizer=model.tokenizer,
                        context_mode=communication_mode,
                        context_policy=aflow_context_policy,
                        context_budget_tokens=aflow_context_budget_tokens,
                        context_top_k_documents=aflow_context_top_k_documents,
                        context_fallback=aflow_context_fallback,
                        context_recovery=aflow_context_recovery,
                        context_recovery_top_k_documents=aflow_context_recovery_top_k_documents,
                        semantic_max_paths=aflow_semantic_max_paths,
                        semantic_max_document_hops=aflow_semantic_max_document_hops,
                        semantic_budget_tokens=aflow_semantic_budget_tokens,
                        semantic_entity_policy=aflow_semantic_entity_policy,
                        artifact_routing_policy=artifact_routing_policy,
                        artifact_fallback=artifact_fallback,
                        artifact_preserve_multiplicity=artifact_preserve_multiplicity,
                    )
                    result["aflow_effective_policy"] = graph_session.optimization_status
                    workflow_module, graph_path = _load_paper_workflow(
                        dataset,
                        workflow_round,
                        bridge,
                        graph_session=graph_session,
                        communication_mode=communication_mode,
                    )
                    item_workflow = workflow_module.Workflow(
                        name=f"aflow_graph_{dataset}_{row_index + 1}",
                        llm_config={"model": "local-qwen"},
                        dataset=dataset,
                    )
                record = await benchmark.evaluate_problem(rows[row_index], item_workflow)
                item_calls = bridge.calls[call_offset:]
                if graph_session is not None:
                    graph_session.telemetry.record_aflow_metric(
                        aflow_actual_llm_calls=len(item_calls)
                    )
                    graph_summary = graph_session.telemetry.summary()
                    graph_summary["sample_id"] = row_id
                    result.setdefault("graph_telemetry", []).append(graph_summary)
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


def _row_task_text(row: dict[str, Any]) -> str:
    for key in ("question", "problem", "prompt", "input"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _compact_call_record(call: dict[str, Any]) -> dict[str, Any]:
    return {
        key: call[key]
        for key in (
            "dataset",
            "problem_index",
            "problem_id",
            # Preserve the native AFlow operator trace so communication-cost
            # audits can identify which intermediate outputs are copied into
            # later operator prompts.  Token totals alone cannot recover this.
            "prompt",
            "output",
            "input_tokens",
            "output_tokens",
            "elapsed_seconds",
            "timed_out",
            "error",
            "logical_input_tokens",
            "physical_prefill_tokens",
            "prefix_cache_hit_tokens",
            "prefix_cache_miss_tokens",
            "prefix_cache_warmup_tokens",
            "prefix_cache_hit",
            "prefix_cache_hits",
            "prefix_cache_misses",
            "transformer_forward_calls",
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
    communication_mode: str = "original",
    communication_policy: str = "send_all",
    aflow_profile: str = "none",
    profile_validated: bool = False,
) -> dict[str, Any]:
    if resume_result is not None:
        result = dict(resume_result)
        result.setdefault("status", "complete" if "score" in result else "running")
        result.setdefault("completed_rows", len(result.get("records", [])))
        result.setdefault("record_ids", [])
        result.setdefault("model_call_records", [])
        result.setdefault("item_wall_times_seconds", [])
        result.setdefault("graph_telemetry", [])
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
        "physical_prefill_tokens": 0,
        "prefix_cache_hit_tokens": 0,
        "prefix_cache_miss_tokens": 0,
        "prefix_cache_warmup_tokens": 0,
        "prefix_cache_hits": 0,
        "prefix_cache_misses": 0,
        "transformer_forward_calls": 0,
        "output_tokens": 0,
        "total_model_tokens": 0,
        "wall_time_seconds": 0.0,
        "records": [],
        "record_ids": [],
        "model_call_records": [],
        "item_wall_times_seconds": [],
        "communication_mode": communication_mode,
        "communication_policy": communication_policy,
        "aflow_profile": aflow_profile,
        "profile_validated": profile_validated,
        "input_binding_version": "v1",
        "candidate_policy_version": "v1",
        "reuse_policy_version": "v1",
        "graph_telemetry": [],
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
    result["physical_prefill_tokens"] = prior_usage["physical_prefill_tokens"] + int(usage["total_physical_prefill_tokens"])
    result["prefix_cache_hit_tokens"] = prior_usage["prefix_cache_hit_tokens"] + int(usage["total_prefix_cache_hit_tokens"])
    result["prefix_cache_miss_tokens"] = prior_usage["prefix_cache_miss_tokens"] + int(usage["total_prefix_cache_miss_tokens"])
    result["prefix_cache_warmup_tokens"] = prior_usage["prefix_cache_warmup_tokens"] + int(usage["total_prefix_cache_warmup_tokens"])
    result["prefix_cache_hits"] = prior_usage["prefix_cache_hits"] + int(usage["prefix_cache_hits"])
    result["prefix_cache_misses"] = prior_usage["prefix_cache_misses"] + int(usage["prefix_cache_misses"])
    result["transformer_forward_calls"] = prior_usage["transformer_forward_calls"] + int(usage["transformer_forward_calls"])
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
    parser.add_argument(
        "--prefix-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse identical prompt prefixes across calls within one problem.",
    )
    parser.add_argument(
        "--prefix-cache-max-entries",
        type=int,
        default=16,
        help="Maximum cached prompt-prefix KV states per model instance.",
    )
    parser.add_argument(
        "--communication-mode",
        choices=("original", "graph_shadow", "graph_sendall", "graph_ra", "graph_minctx", "artifact_shadow", "artifact_sendall", "artifact_minctx"),
        default="original",
        help="Keep the paper Workflow topology and optionally route Operator context through GraphStore.",
    )
    parser.add_argument(
        "--communication-policy",
        choices=("send_all", "closure_aware_heuristic", "receiver_aware_heuristic", "receiver_aware_no_closure"),
        default="send_all",
        help="Graph communication policy used by graph_* modes.",
    )
    parser.add_argument(
        "--aflow-context-policy",
        choices=("full", "topk_documents", "flat_topk_documents", "dependency_minimal", "semantic_minimal", "semantic_shadow"),
        default="full",
        help="HotpotQA source-artifact context policy used by graph_minctx.",
    )
    parser.add_argument(
        "--aflow-context-budget-tokens",
        type=int,
        default=0,
        help="Optional hard budget for a compressed HotpotQA Operator input; 0 disables it.",
    )
    parser.add_argument(
        "--aflow-context-top-k-documents",
        type=int,
        default=4,
        help="Number of deterministic HotpotQA documents considered by top-k policies.",
    )
    parser.add_argument(
        "--aflow-context-fallback",
        choices=("original",),
        default="original",
        help="Fallback policy when a compressed context fails its safety contract.",
    )
    parser.add_argument("--aflow-context-recovery", choices=("none", "adaptive", "semantic_adaptive"), default="none")
    parser.add_argument("--aflow-context-recovery-top-k-documents", type=int, default=6)
    parser.add_argument("--aflow-semantic-max-paths", type=int, default=2)
    parser.add_argument("--aflow-semantic-max-document-hops", type=int, default=2)
    parser.add_argument("--aflow-semantic-budget-tokens", type=int, default=512)
    parser.add_argument("--aflow-semantic-entity-policy", choices=("deterministic_v1",), default="deterministic_v1")
    parser.add_argument("--artifact-routing-policy", choices=("send_all", "contract_minimal"), default="send_all")
    parser.add_argument("--artifact-fallback", choices=("original",), default="original")
    parser.add_argument("--artifact-preserve-multiplicity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--aflow-content-dedup",
        action="store_true",
        help="Collapse exact duplicate AFlow outputs into one canonical GraphStore artifact.",
    )
    parser.add_argument(
        "--aflow-invocation-reuse",
        action="store_true",
        help="Reuse deterministic duplicate Operator invocations with async single-flight.",
    )
    parser.add_argument(
        "--aflow-dedup-mode",
        choices=("exact", "structured"),
        default="exact",
        help="Canonical output deduplication mode; structured is reserved for conservative typed extraction.",
    )
    parser.add_argument(
        "--aflow-profile",
        choices=("none", "auto", *aflow_profile_names()),
        default="auto",
        help="Dataset/workflow-specific profile; auto enables validated profiles only.",
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
    if args.prefix_cache_max_entries < 1:
        raise SystemExit("--prefix-cache-max-entries must be positive")
    if args.aflow_context_budget_tokens < 0:
        raise SystemExit("--aflow-context-budget-tokens cannot be negative")
    if args.aflow_context_top_k_documents < 1:
        raise SystemExit("--aflow-context-top-k-documents must be positive")
    if args.aflow_context_recovery_top_k_documents < 1:
        raise SystemExit("--aflow-context-recovery-top-k-documents must be positive")
    if args.aflow_semantic_max_paths < 1 or args.aflow_semantic_max_document_hops < 1 or args.aflow_semantic_budget_tokens < 1:
        raise SystemExit("AFlow semantic limits and budget must be positive")
    if args.communication_mode == "original" and (
        args.aflow_content_dedup or args.aflow_invocation_reuse
    ):
        raise SystemExit(
            "AFlow reuse flags require graph_sendall or graph_ra; "
            "original mode is intentionally untouched"
        )
    if args.aflow_dedup_mode == "structured":
        raise SystemExit(
            "--aflow-dedup-mode structured is not implemented yet; use exact"
        )
    if args.aflow_invocation_reuse and not args.aflow_content_dedup:
        raise SystemExit(
            "--aflow-invocation-reuse requires --aflow-content-dedup"
        )
    communication_policy = args.communication_policy
    if args.communication_mode == "graph_ra" and communication_policy == "send_all":
        # AFlow outputs are whole semantic nodes rather than fragment-labelled
        # native Actions.  Use the existing closure-aware structural policy
        # until an AFlow fragment utility table is supplied; this keeps the
        # default RA smoke test from selecting an empty set.
        communication_policy = "closure_aware_heuristic"
    selected = list(DATASETS if args.dataset == "all" else (args.dataset,))
    selected_rounds = {dataset: args.workflow_round or PAPER_TEST_ROUNDS[dataset] for dataset in selected}
    selected_profiles: dict[str, Any] = {}
    for dataset in selected:
        round_index = selected_rounds[dataset]
        profile = None if args.aflow_profile == "none" else get_aflow_profile(dataset, round_index)
        if args.aflow_profile not in {"none", "auto"}:
            if profile is None or profile.profile_name != args.aflow_profile:
                raise SystemExit(f"Profile {args.aflow_profile!r} does not match {dataset} round {round_index}")
        if args.communication_mode not in {"original", "graph_shadow"}:
            if profile is None:
                raise SystemExit(f"No registered AFlow profile for {dataset} round {round_index}; use original or graph_shadow")
            if args.aflow_profile == "auto" and not profile.validated:
                raise SystemExit(
                    f"Profile {profile.profile_name} is not validated; explicitly pass --aflow-profile {profile.profile_name} for experimental use"
                )
        selected_profiles[dataset] = profile
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
            dataset: selected_rounds[dataset] for dataset in selected
        },
        "math_paper_subset": args.math_paper_subset,
        "max_new_tokens": args.max_new_tokens or None,
        "chat_template": args.chat_template,
        "generation_timeout_seconds": args.generation_timeout_seconds or None,
        "prefix_cache": args.prefix_cache,
        "prefix_cache_max_entries": args.prefix_cache_max_entries,
        "communication_mode": args.communication_mode,
        "communication_policy": communication_policy,
        "aflow_content_dedup": args.aflow_content_dedup,
        "aflow_invocation_reuse": args.aflow_invocation_reuse,
        "aflow_dedup_mode": args.aflow_dedup_mode,
        "aflow_profile": args.aflow_profile,
        "aflow_context_policy": args.aflow_context_policy,
        "aflow_context_budget_tokens": args.aflow_context_budget_tokens,
        "aflow_context_top_k_documents": args.aflow_context_top_k_documents,
        "aflow_context_fallback": args.aflow_context_fallback,
        "aflow_context_recovery": args.aflow_context_recovery,
        "aflow_context_recovery_top_k_documents": args.aflow_context_recovery_top_k_documents,
        "aflow_semantic_max_paths": args.aflow_semantic_max_paths,
        "aflow_semantic_max_document_hops": args.aflow_semantic_max_document_hops,
        "aflow_semantic_budget_tokens": args.aflow_semantic_budget_tokens,
        "aflow_semantic_entity_policy": args.aflow_semantic_entity_policy,
        "artifact_routing_policy": args.artifact_routing_policy,
        "artifact_fallback": args.artifact_fallback,
        "artifact_preserve_multiplicity": args.artifact_preserve_multiplicity,
        "profile_validated": all(bool(profile and profile.validated) for profile in selected_profiles.values()),
        "input_binding_version": "v1",
        "candidate_policy_version": "v1",
        "reuse_policy_version": "v1",
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
            "prefix_cache": args.prefix_cache,
            "prefix_cache_max_entries": args.prefix_cache_max_entries,
            "communication_mode": args.communication_mode,
            "communication_policy": communication_policy,
            "aflow_content_dedup": args.aflow_content_dedup,
            "aflow_invocation_reuse": args.aflow_invocation_reuse,
            "aflow_dedup_mode": args.aflow_dedup_mode,
            "aflow_profile": args.aflow_profile,
            "aflow_context_policy": args.aflow_context_policy,
            "aflow_context_budget_tokens": args.aflow_context_budget_tokens,
            "aflow_context_top_k_documents": args.aflow_context_top_k_documents,
            "aflow_context_fallback": args.aflow_context_fallback,
            "aflow_context_recovery": args.aflow_context_recovery,
            "aflow_context_recovery_top_k_documents": args.aflow_context_recovery_top_k_documents,
            "aflow_semantic_max_paths": args.aflow_semantic_max_paths,
            "aflow_semantic_max_document_hops": args.aflow_semantic_max_document_hops,
            "aflow_semantic_budget_tokens": args.aflow_semantic_budget_tokens,
            "aflow_semantic_entity_policy": args.aflow_semantic_entity_policy,
            "artifact_routing_policy": args.artifact_routing_policy,
            "artifact_fallback": args.artifact_fallback,
            "artifact_preserve_multiplicity": args.artifact_preserve_multiplicity,
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
        prefix_cache=args.prefix_cache,
        prefix_cache_max_entries=args.prefix_cache_max_entries,
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
                    communication_mode=args.communication_mode,
                    communication_policy=communication_policy,
                    model_fingerprint=args.model_path,
                    aflow_content_dedup=args.aflow_content_dedup,
                    aflow_invocation_reuse=args.aflow_invocation_reuse,
                    aflow_dedup_mode=args.aflow_dedup_mode,
                    aflow_profile=selected_profiles[dataset],
                    aflow_profile_explicit=args.aflow_profile not in {"none", "auto"},
                    aflow_context_policy=args.aflow_context_policy,
                    aflow_context_budget_tokens=args.aflow_context_budget_tokens,
                    aflow_context_top_k_documents=args.aflow_context_top_k_documents,
                    aflow_context_fallback=args.aflow_context_fallback,
                    aflow_context_recovery=args.aflow_context_recovery,
                    aflow_context_recovery_top_k_documents=args.aflow_context_recovery_top_k_documents,
                    aflow_semantic_max_paths=args.aflow_semantic_max_paths,
                    aflow_semantic_max_document_hops=args.aflow_semantic_max_document_hops,
                    aflow_semantic_budget_tokens=args.aflow_semantic_budget_tokens,
                    aflow_semantic_entity_policy=args.aflow_semantic_entity_policy,
                    artifact_routing_policy=args.artifact_routing_policy,
                    artifact_fallback=args.artifact_fallback,
                    artifact_preserve_multiplicity=args.artifact_preserve_multiplicity,
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
