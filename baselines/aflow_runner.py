"""AFlow replay/search external baseline adapters."""

from __future__ import annotations

import asyncio
import importlib
import subprocess
import sys
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .common import (
    BaselineMessage,
    BaselineResult,
    BaselineTokenUsage,
    communication_token_usage,
    count_tokens,
)


@dataclass(frozen=True)
class AFlowModelRequest:
    prompt: str
    session_prompt: str
    session_id: str
    session_reset: bool = True
    session_rollback: bool = False
    timeout_seconds: float | None = None
    cache_namespace: str = ""


@dataclass(frozen=True)
class AFlowStep:
    """One replayed operator in an AFlow-style searched workflow."""

    name: str
    role: str
    instruction: str
    visible_history: str = "all"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "instruction": self.instruction,
            "visible_history": self.visible_history,
        }


class AFlowReplayRunner:
    """Replay fixed AFlow-style workflows under the repository evaluator.

    This runner deliberately does not perform workflow search.  It executes a
    domain-specific searched-workflow template so AFlow can enter the same
    quality/token reporting path before the expensive search/checkpoint phase.
    """

    name = "aflow_replay"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        workflows: dict[str, list[AFlowStep]] | None = None,
        prefer_vendored: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.workflows = workflows or _default_workflows()
        self.prefer_vendored = prefer_vendored

    def run(self, row: dict[str, Any]) -> BaselineResult:
        if self.prefer_vendored and _official_dataset(row) is not None:
            return self._run_vendored(row)
        return self._run_local(row)

    def _run_local(self, row: dict[str, Any]) -> BaselineResult:
        sample_id = str(row.get("sample_id", "sample"))
        domain = str(row.get("domain", ""))
        task_type = str(row.get("task_type", ""))
        workflow_key = domain or task_type
        steps = self.workflows.get(workflow_key) or self.workflows.get(task_type) or self.workflows["default"]
        task_prompt = _task_prompt(row)
        started = time.time()
        trace: list[BaselineMessage] = [
            BaselineMessage(sender="user", receiver="aflow", content=task_prompt, role="user", message_type="task")
        ]
        input_tokens = 0
        output_tokens = 0
        step_records: list[dict[str, Any]] = []
        for index, step in enumerate(steps, start=1):
            visible_messages = _visible_messages(trace, step.visible_history)
            prompt = _render_step_prompt(step, row, task_prompt, visible_messages)
            output = str(
                self.model(
                    AFlowModelRequest(
                        prompt=prompt,
                        session_prompt=prompt,
                        session_id=f"external:aflow:{sample_id}:{step.name}",
                        cache_namespace=f"external:aflow:{sample_id}",
                    )
                )
            )
            prompt_tokens = count_tokens(self.tokenizer, prompt)
            completion_tokens = count_tokens(self.tokenizer, output)
            input_tokens += prompt_tokens
            output_tokens += completion_tokens
            trace.append(BaselineMessage(sender=step.name, receiver="aflow", content=output, role=step.role))
            step_records.append({
                "index": index,
                "name": step.name,
                "role": step.role,
                "visible_history": step.visible_history,
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
            })

        communication_tokens, communication_messages = communication_token_usage(self.tokenizer, trace)
        usage = BaselineTokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            communication_tokens=communication_tokens,
            communication_messages=communication_messages,
            forward_calls=len(steps),
            system_prompt_tokens=sum(count_tokens(self.tokenizer, step.instruction) for step in steps),
            task_prompt_tokens=count_tokens(self.tokenizer, task_prompt),
        )
        final_answer = trace[-1].content if trace else ""
        return BaselineResult(
            sample_id=sample_id,
            method=self.name,
            domain=domain,
            final_answer=final_answer,
            trace=trace,
            token_usage=usage,
            raw_output=final_answer,
            latency_sec=time.time() - started,
            metadata={
                "aflow_mode": "local_replay",
                "aflow_workflow_key": workflow_key,
                "aflow_workflow": [step.as_dict() for step in steps],
                "aflow_step_records": step_records,
            },
        )

    def _run_vendored(self, row: dict[str, Any]) -> BaselineResult:
        sample_id = str(row.get("sample_id", "sample"))
        domain = str(row.get("domain", ""))
        dataset = _official_dataset(row)
        if dataset is None:
            return self._run_local(row)
        started = time.time()
        bridge = _AFlowLLMBridge(self.model, self.tokenizer)
        root = _ensure_aflow_importable()
        task_prompt = _task_prompt(row)
        with _aflow_import_context(root), _patched_aflow_llm(bridge):
            workflow_module = importlib.import_module(f"workspace.{dataset}.workflows.round_1.graph")
            setattr(workflow_module, "create_llm_instance", lambda _config: bridge)
            workflow = workflow_module.Workflow(
                name=f"{self.name}_{dataset}",
                llm_config={"model": "local-qwen"},
                dataset=dataset,
            )
            output = asyncio.run(_call_official_workflow(workflow, row, dataset, task_prompt))
        trace = [BaselineMessage(sender="user", receiver="aflow", content=task_prompt, role="user", message_type="task")]
        for index, call in enumerate(bridge.calls, start=1):
            trace.append(BaselineMessage(sender=f"aflow_op_{index}", receiver="aflow", content=call["output"]))
        final_answer = str(output)
        if not trace or trace[-1].content != final_answer:
            trace.append(BaselineMessage(sender="aflow_final", receiver="aflow", content=final_answer))
        communication_tokens, communication_messages = communication_token_usage(self.tokenizer, trace)
        usage = BaselineTokenUsage(
            input_tokens=sum(int(call["input_tokens"]) for call in bridge.calls),
            output_tokens=sum(int(call["output_tokens"]) for call in bridge.calls),
            communication_tokens=communication_tokens,
            communication_messages=communication_messages,
            forward_calls=len(bridge.calls),
            task_prompt_tokens=count_tokens(self.tokenizer, task_prompt),
        )
        return BaselineResult(
            sample_id=sample_id,
            method=self.name,
            domain=domain,
            final_answer=final_answer,
            trace=trace,
            token_usage=usage,
            raw_output=final_answer,
            latency_sec=time.time() - started,
            metadata={
                "aflow_mode": "vendored_replay",
                "aflow_repo_root": str(root),
                "aflow_dataset": dataset,
                "aflow_workflow_module": f"workspace.{dataset}.workflows.round_1.graph",
                "aflow_call_records": list(bridge.calls),
            },
        )


class AFlowSearchRunner:
    """Explicit boundary for full AFlow workflow search."""

    name = "aflow_search"

    def __init__(
        self,
        *_: Any,
        dataset: str = "GSM8K",
        sample: int = 4,
        initial_round: int = 1,
        max_rounds: int = 1,
        validation_rounds: int = 1,
        optimized_path: str = ".runtime/aflow_search",
        opt_model_name: str = "gpt-4o-mini",
        exec_model_name: str = "gpt-4o-mini",
        run_search: bool = False,
        **__: Any,
    ) -> None:
        self.root = _ensure_aflow_importable()
        self.dataset = dataset
        self.sample = int(sample)
        self.initial_round = int(initial_round)
        self.max_rounds = int(max_rounds)
        self.validation_rounds = int(validation_rounds)
        self.optimized_path = optimized_path
        self.opt_model_name = opt_model_name
        self.exec_model_name = exec_model_name
        self.run_search = bool(run_search)
        self.last_search: dict[str, Any] | None = None

    def run(self, row: dict[str, Any]) -> BaselineResult:
        if self.last_search is None:
            self.last_search = self.run_once()
        return BaselineResult(
            sample_id=str(row.get("sample_id", "sample")),
            method=self.name,
            domain=str(row.get("domain", "")),
            final_answer=None,
            failure="" if self.last_search.get("returncode") == 0 else str(self.last_search.get("stderr", ""))[:2000],
            metadata={"aflow_search": self.last_search},
        )

    def command(self) -> list[str]:
        return [
            sys.executable,
            "run.py",
            "--dataset",
            self.dataset,
            "--sample",
            str(self.sample),
            "--optimized_path",
            self.optimized_path,
            "--initial_round",
            str(self.initial_round),
            "--max_rounds",
            str(self.max_rounds),
            "--validation_rounds",
            str(self.validation_rounds),
            "--opt_model_name",
            self.opt_model_name,
            "--exec_model_name",
            self.exec_model_name,
            "--if_force_download",
            "false",
        ]

    def run_once(self) -> dict[str, Any]:
        command = self.command()
        if not self.run_search:
            return {
                "status": "dry_run",
                "returncode": None,
                "command": command,
                "cwd": str(self.root),
                "note": "Set run_search=True from code after configuring AFlow config/config2.yaml and API/backend access.",
            }
        completed = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=None)
        return {
            "status": "executed",
            "returncode": completed.returncode,
            "command": command,
            "cwd": str(self.root),
            "stdout": completed.stdout[-8000:],
            "stderr": completed.stderr[-8000:],
        }


class _AFlowLLMBridge:
    """AFlow AsyncLLM-compatible wrapper around the repository model callable."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        call_timeout_seconds: float | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.calls: list[dict[str, Any]] = []
        self.call_timeout_seconds = call_timeout_seconds
        self.event_callback = event_callback
        self.call_context: dict[str, Any] = {}

    def set_call_context(self, **context: Any) -> None:
        self.call_context = dict(context)

    def _emit(self, event: dict[str, Any]) -> None:
        if self.event_callback is not None:
            self.event_callback(event)

    async def __call__(self, prompt: str) -> str:
        call_number = len(self.calls) + 1
        render_prompt = getattr(self.model, "render_prompt", None)
        rendered_prompt = render_prompt(prompt) if callable(render_prompt) else prompt
        input_tokens = count_tokens(self.tokenizer, rendered_prompt)
        started = time.monotonic()
        self._emit({
            "event": "start",
            "call_number": call_number,
            "input_tokens": input_tokens,
            "timeout_seconds": self.call_timeout_seconds,
            **self.call_context,
        })
        try:
            output = str(
                self.model(
                    AFlowModelRequest(
                        prompt=prompt,
                        session_prompt=prompt,
                        session_id=f"external:aflow:{call_number}",
                        timeout_seconds=self.call_timeout_seconds,
                        cache_namespace=str(
                            self.call_context.get("cache_namespace", "")
                            or f"external:aflow:{call_number}"
                        ),
                    )
                )
            )
        except Exception as exc:
            elapsed = time.monotonic() - started
            call = {
                "prompt": prompt,
                "output": "",
                "input_tokens": input_tokens,
                "output_tokens": 0,
                "elapsed_seconds": elapsed,
                "timed_out": False,
                "error": f"{type(exc).__name__}: {exc}",
                **self.call_context,
            }
            self.calls.append(call)
            self._emit({
                "event": "error",
                "call_number": call_number,
                "input_tokens": input_tokens,
                "output_tokens": 0,
                "elapsed_seconds": elapsed,
                "error": call["error"],
                **self.call_context,
            })
            raise
        elapsed = time.monotonic() - started
        model_metrics = dict(getattr(self.model, "last_call_metrics", {}) or {})
        input_tokens = int(model_metrics.get("physical_input_tokens", input_tokens))
        timed_out = bool(model_metrics.get("timed_out", False))
        call = {
            "prompt": prompt,
            "output": output,
            "input_tokens": input_tokens,
            "output_tokens": count_tokens(self.tokenizer, output),
            "elapsed_seconds": elapsed,
            "timed_out": timed_out,
            "logical_input_tokens": int(model_metrics.get("logical_input_tokens", input_tokens)),
            "physical_prefill_tokens": int(model_metrics.get("physical_prefill_tokens", input_tokens)),
            "prefix_cache_hit_tokens": int(model_metrics.get("prefix_cache_hit_tokens", 0)),
            "prefix_cache_miss_tokens": int(model_metrics.get("prefix_cache_miss_tokens", input_tokens)),
            "prefix_cache_warmup_tokens": int(model_metrics.get("prefix_cache_warmup_tokens", 0)),
            "prefix_cache_hit": bool(model_metrics.get("prefix_cache_hit", False)),
            "prefix_cache_hits": int(model_metrics.get("prefix_cache_hits", 0)),
            "prefix_cache_misses": int(model_metrics.get("prefix_cache_misses", 0)),
            "transformer_forward_calls": int(model_metrics.get("transformer_forward_calls", 1)),
            "chat_template_applied": bool(model_metrics.get("chat_template_applied", False)),
            **self.call_context,
        }
        self.calls.append(call)
        self._emit({
            "event": "timeout" if timed_out else "end",
            "call_number": call_number,
            "input_tokens": call["input_tokens"],
            "output_tokens": call["output_tokens"],
            "physical_prefill_tokens": call["physical_prefill_tokens"],
            "prefix_cache_hit_tokens": call["prefix_cache_hit_tokens"],
            "prefix_cache_hit": call["prefix_cache_hit"],
            "elapsed_seconds": elapsed,
            "timed_out": timed_out,
            **self.call_context,
        })
        return output

    async def call_with_format(self, prompt: str, formatter: Any) -> Any:
        formatted_prompt = formatter.prepare_prompt(prompt)
        response = await self.__call__(formatted_prompt)
        is_valid, parsed = formatter.validate_response(response)
        if is_valid:
            return parsed
        return {"response": response, "format_error": formatter.format_error_message()}

    def get_usage_summary(self) -> dict[str, Any]:
        input_tokens = sum(int(call["input_tokens"]) for call in self.calls)
        output_tokens = sum(int(call["output_tokens"]) for call in self.calls)
        return {
            "total_input_tokens": input_tokens,
            "total_output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "total_logical_input_tokens": sum(int(call.get("logical_input_tokens", call["input_tokens"])) for call in self.calls),
            "total_physical_prefill_tokens": sum(int(call.get("physical_prefill_tokens", call["input_tokens"])) for call in self.calls),
            "total_prefix_cache_hit_tokens": sum(int(call.get("prefix_cache_hit_tokens", 0)) for call in self.calls),
            "total_prefix_cache_miss_tokens": sum(int(call.get("prefix_cache_miss_tokens", call["input_tokens"])) for call in self.calls),
            "total_prefix_cache_warmup_tokens": sum(int(call.get("prefix_cache_warmup_tokens", 0)) for call in self.calls),
            "prefix_cache_hits": sum(int(call.get("prefix_cache_hits", 0)) for call in self.calls),
            "prefix_cache_misses": sum(int(call.get("prefix_cache_misses", 0)) for call in self.calls),
            "transformer_forward_calls": sum(int(call.get("transformer_forward_calls", 1)) for call in self.calls),
            "total_cost": 0.0,
            "call_count": len(self.calls),
            "timeout_count": sum(bool(call.get("timed_out")) for call in self.calls),
            "history": list(self.calls),
        }


def _task_prompt(row: dict[str, Any]) -> str:
    from scripts.evaluate_domain_workflow import render_native_task

    return render_native_task(row)


async def _call_official_workflow(workflow: Any, row: dict[str, Any], dataset: str, task_prompt: str) -> Any:
    if dataset in {"MBPP", "HumanEval"}:
        requirements = row.get("requirements", {})
        entry_point = ""
        if isinstance(requirements, dict):
            entry_point = str(requirements.get("entry_point", ""))
        result = await workflow(task_prompt, entry_point)
    else:
        result = await workflow(task_prompt)
    if isinstance(result, tuple):
        return result[0]
    return result


def _official_dataset(row: dict[str, Any]) -> str | None:
    domain = str(row.get("domain", "")).lower()
    task_type = str(row.get("task_type", "")).lower()
    mapping = {
        "gsm8k": "GSM8K",
        "hotpotqa": "HotpotQA",
        "mbpp": "MBPP",
        "humaneval": "HumanEval",
        "numeric_solve": "GSM8K",
        "multihop_qa": "HotpotQA",
        "code_generation": "MBPP",
    }
    if domain == "humaneval":
        return "HumanEval"
    if domain == "mbpp":
        return "MBPP"
    if domain == "mmlu_pro":
        return None
    return mapping.get(domain) or mapping.get(task_type)


def _ensure_aflow_importable() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "third_party" / "AFlow",
        repo_root.parent / "AFlow",
    ]
    for candidate in candidates:
        if (candidate / "run.py").exists() and (candidate / "workspace").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError(
        "Upstream AFlow is not available. Clone https://github.com/FoundationAgents/AFlow "
        "to third_party/AFlow before running aflow_replay/aflow_search."
    )


@contextmanager
def _patched_aflow_llm(bridge: _AFlowLLMBridge):
    root = _ensure_aflow_importable()
    async_llm = importlib.import_module("scripts.async_llm")
    old_factory = getattr(async_llm, "create_llm_instance")
    setattr(async_llm, "create_llm_instance", lambda _config: bridge)
    try:
        yield root
    finally:
        setattr(async_llm, "create_llm_instance", old_factory)


@contextmanager
def _aflow_import_context(root: Path):
    """Temporarily route AFlow's top-level packages to the vendored repo."""

    package_names = ("scripts", "benchmarks", "workspace", "data")
    saved = {name: sys.modules.get(name) for name in package_names}
    missing = {name for name in package_names if name not in sys.modules}
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        for name in package_names:
            package = types.ModuleType(name)
            package.__package__ = name
            package.__path__ = [str(root / name)]
            package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            sys.modules[name] = package
        yield
    finally:
        for name, module in saved.items():
            if name in missing:
                sys.modules.pop(name, None)
            elif module is not None:
                sys.modules[name] = module


def _visible_messages(trace: list[BaselineMessage], mode: str) -> list[BaselineMessage]:
    if mode == "last":
        return trace[-1:]
    if mode == "task":
        return trace[:1]
    return list(trace)


def _render_step_prompt(
    step: AFlowStep,
    row: dict[str, Any],
    task_prompt: str,
    visible_messages: list[BaselineMessage],
) -> str:
    parts = [
        step.instruction,
        "",
        f"Domain: {row.get('domain', '')}",
        f"Task type: {row.get('task_type', '')}",
        "",
        "Task:",
        task_prompt,
        "",
        "Visible workflow state:",
    ]
    parts.extend(f"{message.sender}: {message.content}" for message in visible_messages)
    parts.extend(["", _final_output_instruction(row, step)])
    return "\n".join(parts)


def _final_output_instruction(row: dict[str, Any], step: AFlowStep) -> str:
    task_type = str(row.get("task_type", ""))
    if step.role != "finalizer":
        return "Produce the intermediate artifact requested by this workflow step."
    if task_type == "code_generation":
        return "Return only executable Python code, with no Markdown."
    if task_type == "multiple_choice":
        return "Return only the final option letter."
    return "Return only the final answer."


def _default_workflows() -> dict[str, list[AFlowStep]]:
    numeric = [
        AFlowStep("plan_solver", "solver", "Solve the problem step by step and propose an answer.", "task"),
        AFlowStep("consistency_checker", "critic", "Check the proposed solution for arithmetic or reasoning errors.", "all"),
        AFlowStep("final_numeric_answer", "finalizer", "Revise if needed and emit the final numeric answer.", "all"),
    ]
    code = [
        AFlowStep("code_writer", "solver", "Write a complete Python solution for the task.", "task"),
        AFlowStep("test_reasoner", "critic", "Inspect the code against the visible tests and identify likely failures.", "all"),
        AFlowStep("code_repair_final", "finalizer", "Repair the solution and emit final executable Python code.", "all"),
    ]
    multihop = [
        AFlowStep("question_decomposer", "solver", "Decompose the question into evidence needs.", "task"),
        AFlowStep("evidence_synthesizer", "solver", "Use the provided evidence to synthesize a concise rationale.", "all"),
        AFlowStep("hotpot_final_answer", "finalizer", "Return the final short answer supported by the evidence.", "all"),
    ]
    multiple_choice = [
        AFlowStep("choice_reasoner", "solver", "Reason over the choices and select the most likely answer.", "task"),
        AFlowStep("choice_verifier", "critic", "Check whether the selected option follows from the question.", "all"),
        AFlowStep("choice_finalizer", "finalizer", "Emit only the final option letter.", "all"),
    ]
    return {
        "gsm8k": numeric,
        "numeric_solve": numeric,
        "humaneval": code,
        "mbpp": code,
        "code_generation": code,
        "hotpotqa": multihop,
        "multihop_qa": multihop,
        "mmlu_pro": multiple_choice,
        "multiple_choice": multiple_choice,
        "default": [
            AFlowStep("initial_solver", "solver", "Solve the task and propose an answer.", "task"),
            AFlowStep("reviewer", "critic", "Review the proposed answer for mistakes.", "all"),
            AFlowStep("finalizer", "finalizer", "Emit the final answer.", "all"),
        ],
    }
