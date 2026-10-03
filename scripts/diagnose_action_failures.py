#!/usr/bin/env python3
"""Diagnose strict Action failures and graph-delta communication health."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.checkpoint.memory import MemorySaver

from workflow_runtime.action_compiler import ActionCompiler
from workflow_runtime.agent_graph_view import AgentGraphViewManager
from workflow_runtime.communication import make_communication_policy
from workflow_runtime.delta_closure import dependency_closure
from workflow_runtime.delta_extractor import extract_delta_candidates
from workflow_runtime.graph_store import GraphStore
from workflow_runtime.langgraph_workflow import LangGraphWorkflow
from workflow_runtime.model_backend import TransformersModel
from workflow_runtime.protocol import validate_action
from workflow_runtime.telemetry import WorkflowTelemetry
import main as workflow_entry


def _load_eval_module() -> Any:
    path = ROOT / "scripts" / "evaluate_domain_workflow.py"
    spec = importlib.util.spec_from_file_location("domain_eval_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LoggingModel:
    """Record every model raw output while preserving the backend interface."""

    def __init__(self, backend: TransformersModel) -> None:
        self.backend = backend
        self.tokenizer = backend.tokenizer
        self.calls: list[dict[str, Any]] = []

    def __call__(self, request: Any) -> str:
        started = time.time()
        raw = self.backend(request)
        self.calls.append({
            "role": getattr(request, "role", ""),
            "mode": getattr(request, "mode", ""),
            "raw_output": raw,
            "raw_output_chars": len(str(raw)),
            "latency_sec": time.time() - started,
            "metrics": self.backend.session_snapshot(getattr(request, "session_id", "")).get("last_call_metrics", {}),
        })
        return raw

    def session_snapshot(self, session_id: str) -> dict[str, Any]:
        return self.backend.session_snapshot(session_id)

    def rollback_session(self, session_id: str) -> None:
        return self.backend.rollback_session(session_id)


def _extract_first_json_object(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        suffix = text[index + end:].strip()
        if isinstance(value, dict):
            return value, text[index:index + end], suffix
    return None, "", ""


def classify_action_text(raw: Any, *, role: str, task_type: str) -> dict[str, Any]:
    text = "" if raw is None else str(raw)
    stripped = text.strip()
    if not stripped:
        return {"category": "parse_empty", "detail": "empty output", "json_object": None}
    try:
        value = json.loads(stripped)
        exact_json = True
        trailing_text = ""
    except json.JSONDecodeError as exc:
        value, _, trailing_text = _extract_first_json_object(stripped)
        exact_json = False
        if value is None:
            return {
                "category": "parse_invalid_json",
                "detail": f"{exc.msg} at pos {exc.pos}",
                "json_object": None,
                "preview": stripped[:500],
            }
    if not isinstance(value, dict):
        return {"category": "parse_schema_error", "detail": "JSON value is not an object", "json_object": value}
    op = value.get("op")
    errors = validate_action(role, value, task_type=task_type)
    if errors:
        joined = "; ".join(errors)
        if any("unknown action op" in item for item in errors):
            category = "parse_unknown_op"
        elif any("keys must be exactly" in item or "must not be null" in item or "non-empty" in item for item in errors):
            category = "parse_missing_field"
        elif any("cannot emit action" in item or "unsupported task_type" in item for item in errors):
            category = "parse_semantic_error"
        else:
            category = "parse_schema_error"
        return {
            "category": category,
            "detail": joined,
            "json_object": value,
            "exact_json": exact_json,
            "trailing_text": trailing_text[:200],
        }
    if not exact_json or trailing_text:
        return {
            "category": "parse_wrapped_valid_json",
            "detail": "valid Action object appears inside extra text",
            "json_object": value,
            "op": op,
            "exact_json": exact_json,
            "trailing_text": trailing_text[:200],
        }
    return {"category": "parse_valid", "detail": "valid exact Action", "json_object": value, "op": op}


def run_action_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    helpers = _load_eval_module()
    rows = helpers.read(Path(args.data_path), args.domain, args.limit)
    if not rows:
        raise SystemExit("No evaluation rows loaded")
    backend = TransformersModel(args.model_path, max_new_tokens=args.max_new_tokens)
    model = LoggingModel(backend)
    records: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    by_role: dict[str, Counter[str]] = {}

    for index, row in enumerate(rows):
        sample_id = f"diag_{args.domain}_{index}"
        task = f"[domain={row['task_type']}]\n{row['question']}"
        graph = workflow_entry.init_workflow_graph(task, task_id=sample_id, task_type=row["task_type"])
        helpers.add_source_nodes(workflow_entry, graph, row, sample_id)
        telemetry_path = Path(args.output).with_suffix("") / f"{sample_id}.jsonl" if args.output else None
        telemetry = WorkflowTelemetry(telemetry_path)
        workflow = LangGraphWorkflow(
            store=graph,
            model=model,
            task_id=sample_id,
            task_type=row["task_type"],
            tokenizer=model.tokenizer,
            max_rounds=args.max_rounds,
            telemetry=telemetry,
            communication_policy=make_communication_policy(args.communication_policy, seed=args.communication_seed),
            communication_budget_tokens=args.communication_budget_tokens,
        )
        before = len(model.calls)
        failure = ""
        try:
            app = workflow.compile(checkpointer=MemorySaver())
            app.invoke(workflow.initial_state(), {"configurable": {"thread_id": sample_id}})
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        calls = model.calls[before:]
        call_records = []
        for call in calls:
            role = str(call.get("role", ""))
            classified = classify_action_text(call.get("raw_output", ""), role=role, task_type=row["task_type"])
            categories[classified["category"]] += 1
            by_role.setdefault(role, Counter())[classified["category"]] += 1
            call_records.append({**call, "classification": classified})
        records.append({
            "sample_id": sample_id,
            "task_type": row["task_type"],
            "failure": failure,
            "call_count": len(calls),
            "calls": call_records,
        })
        first = call_records[0]["classification"]["category"] if call_records else "no_model_call"
        print(f"{index + 1}/{len(rows)} failure={failure or '-'} first_category={first}", flush=True)

    return {
        "mode": "action",
        "domain": args.domain,
        "data_path": args.data_path,
        "count": len(records),
        "category_counts": dict(categories),
        "category_counts_by_role": {role: dict(counter) for role, counter in by_role.items()},
        "records": records,
    }


def run_communication_smoke(args: argparse.Namespace) -> dict[str, Any]:
    store = GraphStore()
    store.add_node(task_id="smoke", branch_id="main", logical_id="task", node_type="task", content="A=5 B=3 sum", owner="user")
    compiler = ActionCompiler(store, task_id="smoke", task_type="numeric_solve")
    views = AgentGraphViewManager(store, task_id="smoke", branch_id="main", agents=("planner", "solver", "critic"))
    policy = make_communication_policy(args.communication_policy, seed=args.communication_seed)
    steps = [
        ("planner", "solver", {"op": "declare_query", "content": {"question": "sum A and B", "task_type": "numeric_solve"}}),
        ("planner", "solver", {"op": "add_fact", "id": "A", "value": 5}),
        ("planner", "solver", {"op": "add_fact", "id": "B", "value": 3}),
        ("planner", "solver", {"op": "add_plan_step", "id": "R1", "operation": "add", "inputs": ["A", "B"]}),
        ("solver", "critic", {"op": "calculate", "id": "R1", "expression": "5+3", "value": 8}),
        ("solver", "critic", {"op": "set_result", "id": "R1", "value": 8}),
    ]
    events = []
    for sender, receiver, action in steps:
        result = compiler.apply(sender, action)
        views.grant(sender, node_ids=result.node_ids, local=True)
        state = store.snapshot()
        candidates = extract_delta_candidates(
            state,
            node_ids=result.node_ids,
            sender=sender,
            receiver=receiver,
            receiver_view=views.view(receiver),
        )
        roots = policy.select_roots(
            state=state,
            sender_view=views.view(sender),
            receiver_view=views.view(receiver),
            candidates=candidates,
            budget_tokens=args.communication_budget_tokens,
        )
        delta = dependency_closure(
            state,
            root_node_ids=roots,
            sender=sender,
            receiver=receiver,
            receiver_view=views.view(receiver),
            policy=policy.name,
            candidate_token_cost=sum(candidate.token_cost for candidate in candidates),
        )
        views.grant(receiver, node_ids=delta.node_ids, edge_ids=delta.edge_ids, delta_id=f"{sender}->{receiver}:{len(events)+1}")
        events.append({
            "sender": sender,
            "receiver": receiver,
            "action": action,
            "compiled_node_ids": list(result.node_ids),
            "candidate_node_ids": [candidate.node_id for candidate in candidates],
            "candidate_categories": [candidate.novelty_score for candidate in candidates],
            "selected_roots": roots,
            "sent_node_ids": list(delta.node_ids),
            "closure_added_node_ids": list(delta.closure_added_node_ids),
            "sent_edge_ids": list(delta.edge_ids),
            "sent_tokens": delta.token_cost,
        })
    solver_visible = views.view("solver").visible_node_ids
    critic_visible = views.view("critic").visible_node_ids
    result = store.latest_valid("smoke", "main", "result")
    facts = store.latest_valid("smoke", "main", "facts")
    plan = store.latest_valid("smoke", "main", "plan")
    return {
        "mode": "communication",
        "policy": policy.name,
        "ok": bool(result and result.node_id in critic_visible and facts and facts.node_id in critic_visible),
        "solver_visible_count": len(solver_visible),
        "critic_visible_count": len(critic_visible),
        "critic_has_result": bool(result and result.node_id in critic_visible),
        "critic_has_facts": bool(facts and facts.node_id in critic_visible),
        "critic_has_plan": bool(plan and plan.node_id in critic_visible),
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("action", "communication", "all"), default="all")
    parser.add_argument("--domain", default="gsm8k")
    parser.add_argument("--data-path", default="data/gsm8k/test.jsonl")
    parser.add_argument("--model-path", default="/root/models/models/qwen--Qwen2.5-1.5B-Instruct/snapshots/master")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--communication-policy",
        choices=("send_all", "minimal_no_feedback", "minimal_sendall_fallback", "minimal_targeted_feedback", "random_keep_75", "random_keep_50", "random_keep_25", "closure_aware_heuristic"),
        default="closure_aware_heuristic",
    )
    parser.add_argument("--communication-seed", type=int, default=0)
    parser.add_argument("--communication-budget-tokens", type=int, default=None)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    report: dict[str, Any] = {}
    if args.mode in {"communication", "all"}:
        report["communication"] = run_communication_smoke(args)
    if args.mode in {"action", "all"}:
        report["action"] = run_action_diagnosis(args)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
