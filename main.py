"""LangGraph entry point for the versioned multi-agent workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict
from typing import Any

from workflow_runtime.graph_store import GraphStore
from workflow_runtime.communication import make_communication_policy
from workflow_runtime.langgraph_workflow import LangGraphWorkflow, NativeLangGraphWorkflow
from workflow_runtime.model_backend import DirectTransformersModel, TransformersModel
from workflow_runtime.telemetry import WorkflowTelemetry
from workflow_runtime.evaluator import TaskEvaluator, parse_reference_answer


_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def _numeric_value(raw: str) -> int | float:
    normalized = raw.replace(",", "")
    return float(normalized) if "." in normalized else int(normalized)


def _seed_numeric_facts(graph: GraphStore, task: str, *, task_id: str, task_node_id: str) -> None:
    facts: list[dict[str, Any]] = []
    for index, match in enumerate(_NUMBER_PATTERN.finditer(task), start=1):
        start = max(0, match.start() - 48)
        end = min(len(task), match.end() + 48)
        fact = {
            "id": f"N{index}",
            "value": _numeric_value(match.group(0)),
            "text": task[start:end].strip(),
        }
        node = graph.add_node(
            task_id=task_id,
            branch_id="main",
            logical_id=f"fact_{fact['id']}",
            node_type="fact",
            content=fact,
            owner="dataset",
            validation={"schema_valid": True},
            created_by_role="dataset",
        )
        graph.add_edge(source=task_node_id, target=node.node_id, relation="depends_on", created_by_role="dataset")
        facts.append({"id": fact["id"], "value": fact["value"], "text": fact["text"]})
    if not facts:
        return
    aggregate = graph.add_node(
        task_id=task_id,
        branch_id="main",
        logical_id="facts",
        node_type="facts",
        content=facts,
        owner="dataset",
        validation={"schema_valid": True},
        created_by_role="dataset",
    )
    for fact in facts:
        fact_node = graph.latest_valid(task_id, "main", f"fact_{fact['id']}")
        if fact_node is not None:
            graph.add_edge(source=fact_node.node_id, target=aggregate.node_id, relation="input_to", created_by_role="dataset")


def init_workflow_graph(task: str, *, task_id: str, task_type: str = "numeric_solve") -> GraphStore:
    graph = GraphStore()
    task_node = graph.add_node(
        task_id=task_id,
        branch_id="main",
        logical_id="task",
        node_type="task",
        content=f"[domain={task_type}]\n{task}",
        owner="user",
        validation={"schema_valid": True},
        created_by_role="user",
        provenance={"communication_scope": "MANDATORY"},
    )
    query = graph.add_node(
        task_id=task_id,
        branch_id="main",
        logical_id="query_spec",
        node_type="query_spec",
        content={"question": task, "task_type": task_type},
        owner="runtime",
        validation={"schema_valid": True},
        created_by_role="runtime",
        provenance={"communication_scope": "MANDATORY"},
    )
    graph.add_edge(source=task_node.node_id, target=query.node_id, relation="depends_on", created_by_role="runtime")
    if task_type in {"code_generation", "marble_database"}:
        requirements = graph.add_node(
            task_id=task_id,
            branch_id="main",
            logical_id="requirements",
            node_type="requirements",
            content={"text": task},
            owner="user",
            validation={"schema_valid": True},
            created_by_role="user",
        )
        graph.add_edge(source=task_node.node_id, target=requirements.node_id, relation="depends_on", created_by_role="user")
    if task_type in {"numeric_solve", "numeric_comparison"}:
        _seed_numeric_facts(graph, task, task_id=task_id, task_node_id=task_node.node_id)
    return graph


def latest_node_by_type(graph: GraphStore, node_type: str) -> Any:
    candidates = [
        node for node in graph.snapshot().nodes.values()
        if node.type == node_type and node.is_operationally_valid()
    ]
    return max(candidates, key=lambda node: (node.version, node.created_at), default=None)


def run_workflow(
    task: str,
    *,
    task_type: str = "numeric_solve",
    model_path: str,
    max_rounds: int = 3,
    max_new_tokens: int | None = None,
    log_path: str | None = ".runtime/logs/workflow.jsonl",
    reference_answer: Any = None,
    answer_tolerance: float = 1e-6,
    enable_action_constraints: bool = True,
    execution_mode: str = "optimized",
    communication_policy: str = "closure_aware_heuristic",
    communication_seed: int = 0,
    communication_budget_tokens: int | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    if execution_mode not in {"optimized", "native_langgraph"}:
        raise ValueError(f"unsupported execution_mode: {execution_mode}")
    optimized = execution_mode == "optimized"
    task_digest = hashlib.sha256(f"{task_type}\n{task}".encode("utf-8")).hexdigest()[:16]
    task_id = task_id or f"task_cli_{task_digest}"
    telemetry = WorkflowTelemetry(log_path)
    evaluator = TaskEvaluator(
        task_type,
        reference_answer,
        tolerance=answer_tolerance,
    )
    # The native baseline has no logical graph protocol.  Its only graph is
    # LangGraph's runtime control-flow graph, created by NativeLangGraphWorkflow.
    graph = init_workflow_graph(task, task_id=task_id, task_type=task_type) if optimized else None
    model = (
        TransformersModel(model_path, max_new_tokens=max_new_tokens)
        if optimized
        else DirectTransformersModel(model_path, max_new_tokens=max_new_tokens)
    )
    if optimized:
        workflow = LangGraphWorkflow(
            store=graph,
            model=model,
            task_id=task_id,
            task_type=task_type,
            tokenizer=model.tokenizer,
            max_rounds=max_rounds,
            telemetry=telemetry,
            enable_action_constraints=bool(enable_action_constraints),
            communication_policy=make_communication_policy(communication_policy, seed=communication_seed),
            communication_budget_tokens=communication_budget_tokens,
        )
    else:
        # Native is a separate baseline: same LangGraph topology, but no
        # GraphStore/Action protocol/constraints at all.
        workflow = NativeLangGraphWorkflow(
            model=model,
            task_id=task_id,
            task=task,
            task_type=task_type,
            max_rounds=max_rounds,
            telemetry=telemetry,
        )
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError as exc:
        raise RuntimeError("Install LangGraph first: pip install langgraph") from exc
    app = workflow.compile(checkpointer=MemorySaver())
    try:
        result = app.invoke(
            workflow.initial_state(),
            {"configurable": {"thread_id": task_id}},
        )
        if optimized:
            assert graph is not None
            final = graph.latest_valid(task_id, "main", "final_answer")
            verification = graph.latest_valid(task_id, "main", "verification")
            final_answer = final.content if final is not None else None
            session_snapshot = model.session_snapshot(workflow.session_id)
            graph_payload = asdict(graph.snapshot())
        else:
            final_answer = result.get("final_answer")
            verification = None
            final_agent = workflow.agents["final_solver"]
            session_snapshot = model.session_snapshot(final_agent.session_id)
            graph_payload = {}
            agent_payload = result.get("agent_snapshots", workflow.agent_snapshot())
        if optimized:
            agent_payload = {}
        evaluation = evaluator.evaluate(final_answer)
        telemetry.record_evaluation(evaluation.as_dict())
        telemetry_summary = telemetry.finish(
            status="success",
            task_id=task_id,
            branch_id="main",
            final_answer_present=final_answer is not None,
            verification_status=verification.status if verification is not None else None,
            action_constraints_enabled=bool(enable_action_constraints and optimized),
            execution_mode=execution_mode,
            evaluation=evaluation.as_dict(),
        )
        return {
            "state": result,
            "final_answer": final_answer,
            "graph": graph_payload,
            "agents": agent_payload,
            "session": session_snapshot,
            "telemetry": telemetry_summary,
            "evaluation": evaluation.as_dict(),
        }
    except Exception as error:
        evaluation = evaluator.evaluate(None)
        telemetry.record_evaluation(evaluation.as_dict())
        telemetry.finish(
            status="error",
            error=f"{type(error).__name__}: {error}",
            task_id=task_id,
            branch_id="main",
            action_constraints_enabled=bool(enable_action_constraints and optimized),
            execution_mode=execution_mode,
            evaluation=evaluation.as_dict(),
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LangGraph multi-agent workflow")
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-type", default="numeric_solve", choices=(
        "numeric_solve", "numeric_comparison", "table_qa", "multihop_qa", "code_generation",
        "multiple_choice", "marble_research", "marble_bargaining", "marble_database",
    ))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument(
        "--log-path",
        default=os.getenv("WORKFLOW_LOG_PATH", ".runtime/logs/workflow.jsonl"),
        help="JSONL quality/cost log path; pass an empty string to disable file logging",
    )
    parser.add_argument(
        "--reference-answer",
        default=None,
        help="Expected final answer; accepts JSON values such as 5, true, or a JSON object",
    )
    parser.add_argument(
        "--answer-tolerance",
        type=float,
        default=1e-6,
        help="Absolute/relative tolerance for numeric reference answers",
    )
    parser.add_argument(
        "--disable-action-constraints",
        action="store_true",
        help="Disable state-aware logits masking and dynamic Action guidance for A/B comparison",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("optimized", "native_langgraph"),
        default="optimized",
        help="optimized graph/action workflow or unoptimized direct LangGraph baseline",
    )
    parser.add_argument(
        "--communication-policy",
        choices=(
            "send_all",
            "minimal_no_feedback",
            "minimal_sendall_fallback",
            "minimal_targeted_feedback",
            "random_keep_75",
            "random_keep_50",
            "random_keep_25",
            "closure_aware_heuristic",
        ),
        default="closure_aware_heuristic",
        help="Graph-delta communication policy for optimized mode",
    )
    parser.add_argument("--communication-seed", type=int, default=0)
    parser.add_argument(
        "--communication-budget-tokens",
        type=int,
        default=None,
        help="Optional root-selection token budget for budgeted graph communication",
    )
    args = parser.parse_args()
    print(json.dumps(run_workflow(
        args.task,
        task_type=args.task_type,
        model_path=args.model_path,
        max_rounds=args.max_rounds,
        log_path=args.log_path or None,
        reference_answer=parse_reference_answer(args.reference_answer) if args.reference_answer is not None else None,
        answer_tolerance=args.answer_tolerance,
        enable_action_constraints=not args.disable_action_constraints,
        execution_mode=args.execution_mode,
        communication_policy=args.communication_policy,
        communication_seed=args.communication_seed,
        communication_budget_tokens=args.communication_budget_tokens,
    ), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
