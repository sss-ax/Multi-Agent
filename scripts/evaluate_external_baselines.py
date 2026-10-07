"""Evaluate black-box external baselines with the repository benchmark schema."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.common import (  # noqa: E402
    BaselineResult,
    BaselineTokenUsage,
    aggregate_external_records,
    count_tokens,
)
from scripts.evaluate_domain_workflow import read, render_native_task, score_prediction  # noqa: E402


@dataclass(frozen=True)
class ModelRequest:
    prompt: str
    session_prompt: str
    session_id: str
    session_reset: bool = True
    session_rollback: bool = False


class SingleAgentBaselineRunner:
    """A minimal lower-bound external runner used to validate the common path."""

    name = "single_agent"

    def __init__(self, model: Any, tokenizer: Any, *, max_new_tokens: int = 512) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def run(self, row: dict[str, Any]) -> BaselineResult:
        sample_id = str(row.get("sample_id", "sample"))
        task_prompt = render_native_task(row)
        system_prompt = (
            "You are a careful benchmark solver. Return only the final answer unless the task asks for code."
        )
        prompt = f"{system_prompt}\n\nTask:\n{task_prompt}"
        started = time.time()
        output = self.model(ModelRequest(prompt=prompt, session_prompt=prompt, session_id=f"external:{sample_id}"))
        latency = time.time() - started
        input_tokens = count_tokens(self.tokenizer, prompt)
        output_tokens = count_tokens(self.tokenizer, output)
        usage = BaselineTokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            communication_tokens=0,
            communication_messages=0,
            forward_calls=1,
            system_prompt_tokens=count_tokens(self.tokenizer, system_prompt),
            task_prompt_tokens=count_tokens(self.tokenizer, task_prompt),
        )
        return BaselineResult(
            sample_id=sample_id,
            method=self.name,
            domain=str(row.get("domain", "")),
            final_answer=output,
            token_usage=usage,
            raw_output=output,
            latency_sec=latency,
        )


def make_runner(method: str, *, model_path: str, max_new_tokens: int) -> Any:
    from workflow_runtime.model_backend import DirectTransformersModel

    model = DirectTransformersModel(model_path, max_new_tokens=max_new_tokens)
    if method == "single_agent":
        return SingleAgentBaselineRunner(model, model.tokenizer, max_new_tokens=max_new_tokens)
    if method == "autogen_roundrobin":
        from baselines.autogen_runner import AutoGenRoundRobinRunner

        return AutoGenRoundRobinRunner(model, model.tokenizer)
    if method == "agentprune_autogen_local":
        from baselines.agentprune_runner import AgentPruneAutoGenLocalRunner

        return AgentPruneAutoGenLocalRunner(model, model.tokenizer)
    if method == "agentprune_autogen_faithful":
        from baselines.agentprune_runner import AgentPruneAutoGenFaithfulRunner

        return AgentPruneAutoGenFaithfulRunner(model, model.tokenizer)
    if method == "aflow_replay":
        from baselines.aflow_runner import AFlowReplayRunner

        return AFlowReplayRunner(model, model.tokenizer)
    if method == "aflow_search":
        from baselines.aflow_runner import AFlowSearchRunner

        return AFlowSearchRunner(model, model.tokenizer)
    raise ValueError(f"unsupported external baseline method: {method}")


def evaluate_rows(
    *,
    rows: list[dict[str, Any]],
    runner: Any,
    domain: str,
    data_path: str,
    model_path: str,
    limit: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", f"eval_{domain}_{index}"))
        row = {**row, "sample_id": sample_id, "domain": domain}
        try:
            result = runner.run(row)
            scores = score_prediction(row, result.final_answer) if not result.failure else {
                "correct": False,
                "predicted_answer": result.final_answer,
            }
        except Exception as exc:  # pragma: no cover - exercised through CLI failures.
            result = BaselineResult(
                sample_id=sample_id,
                method=getattr(runner, "name", "external"),
                domain=domain,
                failure=f"{type(exc).__name__}: {exc}",
            )
            scores = {"correct": False, "predicted_answer": None}
        records.append(result.as_record(scores=scores, gold_answer=row.get("gold_answer")))
        print(f"{index + 1}/{len(rows)} method={result.method} correct={records[-1]['correct']} failure={result.failure or '-'}", flush=True)

    summary = aggregate_external_records(records)
    return {
        "domain": domain,
        "data_path": data_path,
        "method": getattr(runner, "name", "external"),
        "execution_mode": "external_baseline",
        "model_path": model_path,
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        **summary,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=(
            "single_agent",
            "autogen_roundrobin",
            "agentprune_autogen_local",
            "agentprune_autogen_faithful",
            "aflow_replay",
            "aflow_search",
        ),
        default="single_agent",
    )
    parser.add_argument("--domain", choices=("gsm8k", "tatqa", "hotpotqa", "mbpp", "humaneval", "mmlu_pro"), required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows = read(Path(args.data_path), args.domain, args.limit)
    if not rows:
        raise SystemExit("No evaluation rows loaded")
    runner = make_runner(args.method, model_path=args.model_path, max_new_tokens=args.max_new_tokens)
    report = evaluate_rows(
        rows=rows,
        runner=runner,
        domain=args.domain,
        data_path=args.data_path,
        model_path=args.model_path,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
