from scripts.audit_hotpot_phase2b_replay import audit
from tests.test_hotpot_hidden_evidence import _facts, _snapshot
from workflow_runtime import build_hotpot_hidden_environment


def test_phase2b_replay_audit_accepts_isolated_hidden_corpus(tmp_path):
    environment = build_hotpot_hidden_environment(
        "What position was held by the woman who portrayed Corliss Archer?",
        _facts(),
        initial_title_count=1,
    )
    snapshot, _ = _snapshot(environment)
    path = tmp_path / "replay.jsonl"
    import json

    path.write_text(json.dumps(snapshot.to_dict()) + "\n", encoding="utf-8")
    report = audit(path)

    assert report["audit_passed"] is True
    assert report["hotpot_hidden_snapshot_count"] == 1
