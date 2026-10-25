"""Audit hidden-corpus isolation and RETRIEVE viability in Phase 2B snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflow_runtime import (  # noqa: E402
    CommunicationAction,
    CommunicationActionType,
    CommunicationReplayBuffer,
    HotpotLocalInformationAdapter,
)


def audit(path: Path, *, retrieval_limit: int = 4) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(
            f"replay file not found: {path}; the snapshot-collection run must finish past its first communication decision"
        )
    snapshots = CommunicationReplayBuffer.from_jsonl(path).snapshots()
    rows: list[dict[str, object]] = []
    for snapshot in snapshots:
        payload = snapshot.external_state.get("hotpot_hidden_evidence")
        if not isinstance(payload, dict):
            continue
        serialized_graph = json.dumps(
            {node_id: node.content for node_id, node in snapshot.graph_state.nodes.items()},
            ensure_ascii=False,
        )
        hidden = list(payload.get("hidden_facts", ()))
        leaked_ids = [
            str(fact.get("fact_id", ""))
            for fact in hidden
            if str(fact.get("text", "")) and str(fact.get("text")) in serialized_graph
        ]
        oracle_keys = sorted({"gold_answer", "answer", "supporting_facts"} & set(payload))
        action = CommunicationAction(
            CommunicationActionType.RETRIEVE,
            query=str(payload.get("question", "")),
        )
        try:
            preview = HotpotLocalInformationAdapter.from_snapshot(
                snapshot,
                retrieval_limit=retrieval_limit,
            ).preview(snapshot, action)
            retrieve_legal = preview.legal
            retrieve_reason = preview.reason
        except ValueError as exc:
            retrieve_legal = False
            retrieve_reason = str(exc)
        rows.append({
            "snapshot_id": snapshot.snapshot_id,
            "task_id": snapshot.task_id,
            "stage": snapshot.stage,
            "visible_fact_count": len(payload.get("visible_facts", ())),
            "hidden_fact_count": len(hidden),
            "hidden_fact_leak_ids": leaked_ids,
            "oracle_keys": oracle_keys,
            "retrieve_legal": retrieve_legal,
            "retrieve_reason": retrieve_reason,
        })
    failures = [
        row for row in rows
        if row["hidden_fact_leak_ids"] or row["oracle_keys"] or not row["retrieve_legal"]
    ]
    return {
        "replay_path": str(path),
        "snapshot_count": len(snapshots),
        "hotpot_hidden_snapshot_count": len(rows),
        "audit_passed": bool(rows) and not failures,
        "failure_count": len(failures),
        "failures": failures,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-path", required=True)
    parser.add_argument("--retrieval-limit", type=int, default=4)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    try:
        report = audit(Path(args.replay_path), retrieval_limit=args.retrieval_limit)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2))
    if not report["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
