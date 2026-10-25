"""Generate Phase 2B five-action counterfactual teachers from Hotpot snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_domain_workflow import read, score_prediction, score_record  # noqa: E402
from workflow_runtime import (  # noqa: E402
    CommunicationAction,
    CommunicationActionExecutor,
    CommunicationActionType,
    CommunicationReplayBuffer,
    CounterfactualReplayRunner,
    GraphStore,
    HotpotLocalInformationAdapter,
    LangGraphWorkflow,
    TransformersModel,
)
from workflow_runtime.communication import make_communication_policy  # noqa: E402
from workflow_runtime.learning.artifacts import Phase2ArtifactWriter  # noqa: E402
from workflow_runtime.learning.error_features import build_dual_risk_example  # noqa: E402
from workflow_runtime.learning.labeling import label_current_answer  # noqa: E402
from workflow_runtime.learning.teacher_search import InformationAcquisitionTeacher  # noqa: E402


def _row_index(task_id: str) -> int:
    prefix = "eval_hotpotqa_"
    if not task_id.startswith(prefix):
        raise ValueError(f"cannot map snapshot task_id to HotpotQA row: {task_id}")
    return int(task_id[len(prefix):])


def _candidate_value(candidate: Any) -> Any:
    if candidate is None:
        return None
    value = candidate.content
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit-snapshots", type=int, default=0)
    parser.add_argument("--retrieval-limit", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--communication-policy", default="closure_aware_heuristic")
    parser.add_argument("--communication-budget-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    replay_path = Path(args.replay_path)
    if not replay_path.is_file():
        raise SystemExit(
            f"Replay file not found: {replay_path}. Run snapshot collection successfully before teacher generation."
        )
    snapshots = [
        snapshot
        for snapshot in CommunicationReplayBuffer.from_jsonl(replay_path).snapshots()
        if snapshot.task_type == "multihop_qa"
        and isinstance(snapshot.external_state.get("hotpot_hidden_evidence"), dict)
    ]
    if args.limit_snapshots:
        snapshots = snapshots[: args.limit_snapshots]
    if not snapshots:
        raise SystemExit("No HotpotQA hidden-evidence snapshots found")
    max_row = max(_row_index(snapshot.task_id) for snapshot in snapshots)
    rows = read(Path(args.data_path), "hotpotqa", max_row + 1)
    model = TransformersModel(args.model_path, max_new_tokens=args.max_new_tokens)
    writer = Phase2ArtifactWriter(args.output_dir)
    summary = {"snapshot_count": len(snapshots), "teacher_found": 0, "no_successful_action": 0, "errors": 0}

    for index, snapshot in enumerate(snapshots, start=1):
        row = rows[_row_index(snapshot.task_id)]

        def workflow_factory(current_snapshot):
            return LangGraphWorkflow(
                store=GraphStore(),
                model=model,
                task_id=current_snapshot.task_id,
                branch_id=current_snapshot.branch_id,
                task_type=current_snapshot.task_type,
                tokenizer=model.tokenizer,
                max_rounds=args.max_rounds,
                communication_policy=make_communication_policy(
                    args.communication_policy,
                    seed=args.seed,
                    task_family="hotpotqa",
                ),
                communication_budget_tokens=args.communication_budget_tokens,
            )

        def current_labeler(current_snapshot):
            return label_current_answer(
                current_snapshot,
                lambda candidate: (
                    bool(score_prediction(row, _candidate_value(candidate)).get("correct"))
                    if candidate is not None else None
                ),
                label_source="hotpotqa_answer_em",
            )

        def continuation(workflow, state, action):
            from langgraph.checkpoint.memory import MemorySaver

            before = dict(model.total_call_metrics)
            failure = ""
            terminal: dict[str, Any] = {}
            try:
                app = workflow.compile_from_communication_snapshot(snapshot, checkpointer=MemorySaver())
                terminal = app.invoke(state, {"configurable": {"thread_id": f"{snapshot.snapshot_id}:{action.action_type.value}"}})
            except Exception as exc:  # Preserve failed actions as negative outcomes.
                failure = str(exc)
                partial = getattr(exc, "partial_state", None)
                if isinstance(partial, dict):
                    terminal = partial
            import main as graph_helpers

            scores = score_record(graph_helpers, row, workflow.store)
            after = model.total_call_metrics
            logs = list(terminal.get("logs", ()))
            return {
                "final_success": bool(not failure and scores.get("correct")),
                "final_answer": scores.get("predicted_answer"),
                "policy_name": action.action_type.value,
                "evaluator_source": "hotpotqa_answer_em",
                "model_input_tokens": int(after["physical_input_tokens"] - before["physical_input_tokens"]),
                "model_output_tokens": int(after["output_tokens"] - before["output_tokens"]),
                "round_count": int(terminal.get("round_id", 0) or 0),
                "failure_reason": failure,
                "metadata": {"scores": scores, "continuation_log_count": len(logs)},
            }

        runner = CounterfactualReplayRunner(
            snapshot,
            workflow_factory=workflow_factory,
            continuation=continuation,
            current_answer_labeler=current_labeler,
            action_executor_factory=lambda: CommunicationActionExecutor(
                information_adapter=HotpotLocalInformationAdapter.from_snapshot(
                    snapshot,
                    retrieval_limit=args.retrieval_limit,
                )
            ),
        )
        optional = tuple(
            str(fragment.get("fragment_id", ""))
            for fragment in snapshot.fragments
            if fragment.get("scope") == "OPTIONAL_POLICY" and fragment.get("fragment_id")
        )
        question = str(snapshot.external_state["hotpot_hidden_evidence"].get("question", ""))
        actions = (
            CommunicationAction(CommunicationActionType.STOP),
            CommunicationAction(CommunicationActionType.SEND, selected_fragment_ids=optional),
            CommunicationAction(CommunicationActionType.REQUEST, requested_semantics=("missing_evidence", "second_hop")),
            CommunicationAction(CommunicationActionType.RETRIEVE, query=question),
            CommunicationAction(CommunicationActionType.REVISE, target="current_answer"),
        )
        try:
            result = InformationAcquisitionTeacher().search(snapshot, actions, runner)
            writer.write_teacher(result)
            for outcome in result.evaluated_outcomes:
                writer.write_risk(build_dual_risk_example(
                    snapshot,
                    current_answer=outcome.current_answer,
                    final_outcome=outcome.final_outcome,
                    policy_features={"action_type": outcome.action.action_type.value},
                ))
            summary[result.status] = int(summary.get(result.status, 0)) + 1
            print(f"{index}/{len(snapshots)} {snapshot.stage} status={result.status}", flush=True)
        except Exception as exc:
            summary["errors"] += 1
            print(f"{index}/{len(snapshots)} {snapshot.stage} error={exc}", flush=True)

    destination = Path(args.output_dir) / "summary.json"
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
