import json
from pathlib import Path

from scripts.audit_hotpotqa_fixed_k_diagnostics import (
    audit_answer_evidence_coverage,
    audit_bridge_attribute_hop2_retrieval,
    audit_bridge_entity_construction,
    audit_multi_constraint_target_intersection,
    audit_parallel_entity_side_retrieval,
    audit_topology_construction,
)
from scripts.audit_hotpotqa_fixed_k_diagnostics import summarize as summarize_hotpot_fixed_k
from scripts.audit_token_cost_accounting import aggregate_audits, audit_report
from scripts.make_coling_p0_commands import build_commands
from scripts.make_fixed_pool_budget_commands import summarize_budget
from scripts.make_module_ablation_commands import _validate_model_path
from scripts.audit_hotpotqa_topology_retrieval import diagnose_row as diagnose_hotpot_retrieval_row
from scripts.run_hotpot_topology_verifier_diagnostic import run_diagnostic
from workflow_runtime.communication import make_communication_policy


def test_audit_separates_billed_and_selector_tokens(tmp_path: Path):
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "experiment": "adaptive_controller_selector_comm",
                "domain": "gsm8k",
                "count": 1,
                "accuracy": 1.0,
                "total_input_tokens": 100,
                "total_output_tokens": 20,
                "total_model_tokens": 120,
                "selector_input_tokens": 30,
                "records": [
                    {
                        "sample_id": "s1",
                        "candidates": [{"input_tokens": 70, "output_tokens": 10}],
                        "selector_communication_steps": [{"selector_input_tokens": 30}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    audit = audit_report(path)

    assert audit["total_billed_tokens"] == 120
    assert audit["selector_input_tokens"] == 30
    assert audit["analysis_total_model_plus_logical_comm"] == 120
    assert audit["audit"]["do_not_sum_selector_into_billed_total"] is True


def test_aggregate_audits_weighted_quality():
    aggregate = aggregate_audits(
        [
            {"count": 2, "quality": 0.5, "total_billed_tokens": 100},
            {"count": 1, "quality": 1.0, "total_billed_tokens": 50},
        ]
    )

    assert aggregate["count"] == 3
    assert aggregate["quality"] == 2 / 3
    assert aggregate["per_sample"]["total_billed_tokens"] == 50


def test_budget_summary_counts_samples_under_budget(tmp_path: Path):
    candidate_report = tmp_path / "pool.json"
    selector_report = tmp_path / "selector.json"
    candidate_report.write_text(
        json.dumps(
            {
                "records": [
                    {"sample_id": "a", "candidates": [{"input_tokens": 80, "output_tokens": 20}]},
                    {"sample_id": "b", "candidates": [{"input_tokens": 140, "output_tokens": 20}]},
                ]
            }
        ),
        encoding="utf-8",
    )
    selector_report.write_text(
        json.dumps(
            {
                "count": 2,
                "policies": {
                    "selector_receiver_aware": {"selector_input_tokens": 90, "accuracy": 0.5},
                    "selector_send_all": {"selector_input_tokens": 110, "accuracy": 0.5},
                },
                "records_by_policy": {
                    "selector_receiver_aware": [
                        {"sample_id": "a", "selector_input_tokens": 40, "correct": True},
                        {"sample_id": "b", "selector_input_tokens": 50, "correct": True},
                    ],
                    "selector_send_all": [
                        {"sample_id": "a", "selector_input_tokens": 80, "correct": True},
                        {"sample_id": "b", "selector_input_tokens": 90, "correct": True},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_budget(candidate_report, selector_report, [150, 220])

    ra_points = summary["policies"]["selector_receiver_aware"]["points"]
    assert ra_points[0]["solved_count"] == 1
    assert ra_points[1]["solved_count"] == 2


def test_model_path_validation_rejects_placeholder():
    try:
        _validate_model_path("/path/to/qwen")
    except SystemExit as exc:
        assert "placeholder" in str(exc)
    else:
        raise AssertionError("placeholder model path should be rejected")


def test_receiver_aware_policy_variants_are_registered():
    no_closure = make_communication_policy("receiver_aware_no_closure")
    targeted = make_communication_policy("receiver_aware_targeted_feedback")

    assert no_closure.name == "receiver_aware_no_closure"
    assert no_closure.include_dependency_closure is False
    assert targeted.name == "receiver_aware_targeted_feedback"
    assert targeted.include_dependency_closure is True


def test_hotpot_stage6_separates_answer_evidence_and_grounded_coverage():
    audit = audit_answer_evidence_coverage([
        {
            "candidate_id": "c0",
            "answer_correct": True,
            "evidence_correct": False,
            "grounded_correct": False,
        },
        {
            "candidate_id": "c1",
            "answer_correct": False,
            "evidence_correct": True,
            "grounded_correct": False,
        },
        {
            "candidate_id": "c2",
            "answer_correct": True,
            "evidence_correct": True,
            "grounded_correct": True,
        },
    ])

    assert audit["answer_oracle_coverage"] is True
    assert audit["answer_oracle_rank"] == 1
    assert audit["evidence_oracle_coverage"] is True
    assert audit["evidence_oracle_rank"] == 2
    assert audit["grounded_answer_oracle_coverage"] is True
    assert audit["gold_candidate_rank"] == 3
    assert audit["gold_candidate_in_full_pool"] is True
    assert audit["gold_candidate_in_top8"] is True
    assert audit["gold_candidate_in_top4"] is True
    assert audit["gold_candidate_selected"] is False
    assert audit["gold_candidate_top2"] is False
    assert audit["gold_candidate_top4"] is True


def test_hotpot_stage8_reports_selection_oracle_decomposition():
    summary = summarize_hotpot_fixed_k([
        {
            "gold_candidate_in_full_pool": True,
            "gold_candidate_in_top8": True,
            "gold_candidate_in_top4": True,
            "gold_candidate_selected": False,
        },
        {
            "gold_candidate_in_full_pool": True,
            "gold_candidate_in_top8": True,
            "gold_candidate_in_top4": False,
            "gold_candidate_selected": False,
        },
        {
            "gold_candidate_in_full_pool": False,
            "gold_candidate_in_top8": False,
            "gold_candidate_in_top4": False,
            "gold_candidate_selected": False,
        },
    ])

    assert summary["R_construct"] == 2 / 3
    assert summary["R_top8"] == 2 / 3
    assert summary["R_top4"] == 1 / 3
    assert summary["R_selector"] == 0.0
    assert summary["P_top8_given_pool"] == 1.0
    assert summary["P_top4_given_top8"] == 0.5
    assert summary["P_selected_given_top4"] == 0.0


def test_hotpot_topology_construction_audit_tracks_fact_availability_and_binding():
    audit = audit_topology_construction(
        [
            {
                "hop1_pool": [{"fact_id": "supporting_fact_1"}],
                "hop2_pool": [{"fact_id": "supporting_fact_2"}],
                "chain_pair_pool": [
                    {"hop1_fact_id": "supporting_fact_1", "hop2_fact_id": "supporting_fact_2"}
                ],
                "cited_fact_ids": ["supporting_fact_1", "supporting_fact_2"],
                "answer_correct": False,
                "evidence_correct": True,
            }
        ],
        ["supporting_fact_1", "supporting_fact_2"],
        "bridge_attribute",
    )

    assert audit["gold_support_fact1_available"] is True
    assert audit["gold_support_fact2_available"] is True
    assert audit["both_gold_facts_available"] is True
    assert audit["gold_reasoning_structure_constructible"] is True
    assert audit["gold_evidence_binding_constructed"] is True
    assert audit["answer_correct_given_gold_evidence"] is False


def test_hotpot_bridge_entity_audit_tracks_conditional_fact2_recall():
    audit = audit_bridge_entity_construction(
        [
            {
                "bridge_entity_telemetry": {
                    "source_fact_id": "supporting_fact_1",
                    "inferred_bridge": "Jane Doe",
                    "hop2_pool_fact_ids": ["supporting_fact_2", "supporting_fact_9"],
                }
            }
        ],
        ["supporting_fact_1", "supporting_fact_2"],
        "bridge_entity",
    )

    assert audit["bridge_entity_gold_fact1_recall"] is True
    assert audit["bridge_entity_inferred_bridge_correct"] is True
    assert audit["bridge_entity_gold_fact2_recall_after_correct_bridge"] is True


def test_hotpot_bridge_attribute_audit_tracks_relation_conditioned_hop2_retrieval():
    audit = audit_bridge_attribute_hop2_retrieval(
        [
            {
                "bridge_attribute_telemetry": {
                    "hop2_retrieved_any_about_bridge": True,
                    "hop2_retrieved_requested_relation": True,
                    "hop2_pool_fact_ids": ["supporting_fact_2", "supporting_fact_9"],
                }
            }
        ],
        ["supporting_fact_1", "supporting_fact_2"],
        "bridge_attribute",
    )

    assert audit["bridge_attribute_hop2_retrieved_any_about_bridge"] is True
    assert audit["bridge_attribute_hop2_retrieved_requested_relation"] is True
    assert audit["bridge_attribute_hop2_retrieved_gold_fact"] is True


def test_hotpot_multi_constraint_audit_tracks_target_intersection():
    audit = audit_multi_constraint_target_intersection(
        [
            {
                "multi_constraint_telemetry": {
                    "target_set_sizes": [2, 1],
                    "shared_target_count": 1,
                    "shared_targets": ["Jordan Lee"],
                }
            }
        ],
        "multi_constraint_lookup",
    )

    assert audit["multi_constraint_constraint_1_target_retrieved"] is True
    assert audit["multi_constraint_constraint_2_target_retrieved"] is True
    assert audit["multi_constraint_shared_target_found"] is True


def test_hotpot_parallel_audit_tracks_entity_side_retrieval():
    audit = audit_parallel_entity_side_retrieval(
        [
            {
                "parallel_retrieval_telemetry": {
                    "entity_a_fact_count": 2,
                    "entity_b_fact_count": 1,
                    "entity_a_relation_compatible_count": 1,
                    "entity_b_relation_compatible_count": 1,
                    "entity_a_value_count": 1,
                    "entity_b_value_count": 1,
                }
            }
        ],
        "comparison_attribute",
    )

    assert audit["parallel_entity_a_retrieved"] is True
    assert audit["parallel_entity_b_retrieved"] is True
    assert audit["parallel_both_entities_retrieved"] is True
    assert audit["parallel_both_relation_compatible"] is True
    assert audit["parallel_both_values_present"] is True


def test_hotpot_stage9a_retrieval_audit_tracks_gold_fact_recall():
    row = {
        "sample_id": "s1",
        "question": "Are both Dictyosperma and Huernia described as a genus?",
        "entities": ["Dictyosperma", "Huernia"],
        "gold_answer": "yes",
        "supporting_facts": [
            {"fact_id": "supporting_fact_1", "title": "Dictyosperma", "sent_id": 0, "text": "Dictyosperma is a monotypic genus of flowering plant."},
            {"fact_id": "supporting_fact_2", "title": "Huernia", "sent_id": 0, "text": "The genus Huernia consists of stem succulents."},
            {"fact_id": "supporting_fact_3", "title": "Other", "sent_id": 0, "text": "Other evidence."},
        ],
        "evidence_links": [
            {"title": "Dictyosperma", "sent_id": 0},
            {"title": "Huernia", "sent_id": 0},
        ],
    }

    diag = diagnose_hotpot_retrieval_row(row)

    assert diag["reasoning_type"] == "shared_property_yesno"
    assert diag["fact1_recall@10"] is True
    assert diag["fact2_recall@10"] is True
    assert diag["both_fact_recall@10"] is True
    assert diag["gold_fact1_source"]["retrieval_source"] != "miss"
    assert diag["gold_fact2_source"]["retrieval_source"] != "miss"


def test_coling_p0_command_generation_includes_key_experiments(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    args = type(
        "Args",
        (),
        {
            "python": ".venv/bin/python",
            "model_path": str(model),
            "data_paths": ["gsm8k=data/gsm8k/test.jsonl"],
            "domains": ["gsm8k"],
            "output_dir": str(tmp_path / "out"),
            "limit": 2,
            "utility_limit": 2,
            "hotpot_diagnostic_limit": 2,
            "k": 4,
            "seed": 0,
            "max_rounds": 3,
            "max_new_tokens": 512,
        },
    )()

    commands = "\n".join(build_commands(args))

    assert "receiver_aware_no_closure" in commands
    assert "receiver_aware_targeted_feedback" in commands
    assert "summarize_semantic_feedback_ablation.py" in commands
    assert "audit_token_cost_accounting.py" in commands


def test_hotpot_topology_diagnostic_separates_gold_from_wrong():
    report = run_diagnostic()

    assert report["count"] == 6
    assert isinstance(report["gate_pass"], bool)
    assert set(report["summary_by_type"]) == {
        "bridge_entity",
        "bridge_attribute",
        "comparison_attribute",
        "shared_property_yesno",
        "set_intersection",
        "multi_constraint_lookup",
    }
    for summary in report["summary_by_type"].values():
        assert summary["gold_valid_rate"] == 1.0
        assert 0.0 <= summary["wrong_valid_rate"] <= 1.0
        assert 0.0 <= summary["p_gold_score_gt_wrong"] <= 1.0
        assert summary["wrong_count"] >= 3
