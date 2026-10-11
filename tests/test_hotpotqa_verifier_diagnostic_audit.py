import json

from scripts.audit_hotpotqa_verifier_diagnostics import (
    _bridge_delivery_row,
    _critic_confusion,
    _gold_fact_ids,
    _offline_bridge_candidates,
    _repair_delta_label,
    _setting_rows,
    _summarize_delivery_recall,
    _summarize_setting_rows,
    _summarize_stage_preservation,
)


def test_hotpot_audit_confusion_counts_candidate_versions():
    rows = [
        {"offline_answer_correct": True, "critic_status": "verified"},
        {"offline_answer_correct": False, "critic_status": "verified"},
        {"offline_answer_correct": True, "critic_status": "need_fix"},
        {"offline_answer_correct": False, "critic_status": "need_fix"},
    ]

    assert _critic_confusion(rows) == {"TP": 1, "FP": 1, "FN": 1, "TN": 1}


def test_hotpot_audit_maps_gold_supporting_fact_ids():
    row = {
        "supporting_facts": [
            {"title": "Ada", "sent_id": 0, "text": "Ada met Charles."},
            {"title": "Charles", "sent_id": 1, "text": "Charles was born in London."},
        ],
        "evidence_links": [
            {"entity": "Charles", "sent_id": 1},
        ],
    }

    assert _gold_fact_ids(row) == {"supporting_fact_2"}


def test_hotpot_audit_bridge_candidates_and_repair_delta():
    row = {
        "entities": ["Ada", "Charles"],
        "supporting_facts": [
            {"title": "Ada", "sent_id": 0, "text": "Ada met Charles."},
            {"title": "Charles", "sent_id": 1, "text": "Charles was born in London."},
        ],
    }

    assert _offline_bridge_candidates(row)[:2] == ["Ada", "Charles"]
    old = {"answer": "Paris", "bridge_entity": "Ada", "cited_fact_ids": ["supporting_fact_1"]}
    new = {"answer": "London", "bridge_entity": "Charles", "cited_fact_ids": ["supporting_fact_2"]}
    assert _repair_delta_label(old, new) == "changed_answer_bridge_citations"


def test_hotpot_audit_bridge_delivery_recall_is_conditioned_on_offline_hit():
    row = _bridge_delivery_row(
        sample_id="s",
        candidate_index=0,
        gold_bridges={"Charles"},
        offline_candidates=["Ada", "Charles", "London"],
        runtime_candidates=["Ada", "London"],
    )

    assert row["gold_in_ranked_top2"] is True
    assert row["gold_in_selected_packet_top2"] is False
    assert row["bridge_delivery_recall@2"] == 0.0
    assert row["gold_in_solver_visible"] is None
    assert row["solver_visible_context_note"] == "not_observed_in_existing_eval_logs"
    assert _summarize_delivery_recall([row])["bridge_delivery_recall@2"] == 0.0


def test_hotpot_audit_stage_preservation_pinpoints_delivery_loss():
    row = _bridge_delivery_row(
        sample_id="s",
        candidate_index=0,
        gold_bridges={"Charles"},
        offline_candidates=["Ada", "Charles", "London"],
        runtime_candidates=["Ada"],
        stage_candidates={
            "raw_pool": ["Ada", "Charles", "Ada"],
            "ranked_top8": ["Ada", "Charles", "Ada"],
            "deduplicated_top8": ["Ada", "Charles"],
            "visibility_filtered": ["Ada", "Charles"],
            "budget_filtered": ["Ada", "Charles"],
            "repair_target_filtered": ["Ada", "Charles"],
            "serialized_packet": ["Ada"],
            "solver_visible": ["Ada"],
        },
    )
    summary = _summarize_stage_preservation([row])

    assert row["gold_in_repair_target_filtered"] is True
    assert row["gold_in_serialized_packet"] is False
    assert summary["coverage"]["serialized_packet"] == 0.0
    assert summary["transition_recall"]["serialized_packet_over_repair_target_filtered"] == 0.0
    assert summary["transition_denominators"]["serialized_packet_over_repair_target_filtered"] == 1


def test_hotpot_audit_stage_preservation_reports_offline_to_raw_loss_and_observation_rate():
    lost = _bridge_delivery_row(
        sample_id="s",
        candidate_index=0,
        gold_bridges={"Charles"},
        offline_candidates=["Ada", "Charles", "London"],
        runtime_candidates=[],
        stage_candidates={
            "raw_pool": ["Ada", "London"],
            "ranked_top8": ["Ada", "London"],
        },
    )
    unobserved = _bridge_delivery_row(
        sample_id="s",
        candidate_index=1,
        gold_bridges={"Charles"},
        offline_candidates=["Ada", "Charles", "London"],
        runtime_candidates=[],
    )
    summary = _summarize_stage_preservation([lost, unobserved])

    assert summary["coverage"]["offline_top8"] == 1.0
    assert summary["coverage"]["raw_pool"] == 0.0
    assert summary["observation_rate"]["raw_pool"] == 0.5
    assert summary["transition_recall"]["raw_pool_over_offline_top8"] == 0.0
    assert summary["transition_denominators"]["raw_pool_over_offline_top8"] == 1


def test_hotpot_audit_bridge_delivery_matches_json_string_entity_candidates():
    row = _bridge_delivery_row(
        sample_id="s",
        candidate_index=0,
        gold_bridges={"Scott Derrickson"},
        offline_candidates=["Scott Derrickson"],
        runtime_candidates=[],
        stage_candidates={
            "raw_pool": ['{"name":"Scott Derrickson"}'],
            "ranked_top8": ['{"name":"Scott Derrickson"}'],
        },
    )
    summary = _summarize_stage_preservation([row])

    assert row["gold_in_raw_pool"] is True
    assert row["gold_in_runtime_ranked_top8"] is True
    assert summary["offline_to_runtime_raw_recall"] == 1.0


def test_hotpot_audit_simulates_current_preserved_and_oracle_settings():
    rows = _setting_rows(
        sample_id="s",
        candidate_index=0,
        gold_bridges={"Charles"},
        offline_candidates=["Ada", "Charles", "London"],
        current_candidates=["Ada"],
    )
    summary = _summarize_setting_rows(rows)

    assert summary["current"]["bridge_coverage@2"] == 0.0
    assert summary["preserved_top8"]["bridge_coverage@2"] == 1.0
    assert summary["oracle_bridge"]["bridge_coverage@2"] == 1.0
