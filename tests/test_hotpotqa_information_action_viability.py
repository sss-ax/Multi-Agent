from scripts.audit_hotpotqa_information_action_viability import audit


def test_audit_separates_selection_from_realized_answer_reward() -> None:
    rows = [
        {"sample_id": "row0", "gold_answer": "yes"},
        {"sample_id": "row1", "gold_answer": "Paris"},
    ]
    eval_report = {
        "records": [
            {
                "sample_id": "eval_hotpotqa_0",
                "runtime_summary": {"records": [
                    {"role": "solver", "action": {"op": "answer_with_evidence", "answer": "no"}},
                    {"role": "solver", "action": {"op": "answer_with_evidence", "answer": "yes"}},
                ]},
            },
            {
                "sample_id": "eval_hotpotqa_1",
                "runtime_summary": {"records": [
                    {"role": "solver", "action": {"op": "answer_with_evidence", "answer": "Paris"}},
                    {"role": "solver", "action": {"op": "answer_with_evidence", "answer": "London"}},
                ]},
            },
        ]
    }
    selection_report = {
        "setting_rows": [
            {"sample_id": sample, "setting": setting, "selected_bridge_correct": correct,
             "gold_in_topk": True, "parse_fail": False, "input_tokens": 10, "selector_need": {}}
            for sample, deployable_correct, random_correct in [
                ("eval_hotpotqa_0", True, False),
                ("eval_hotpotqa_1", True, True),
            ]
            for setting, correct in [
                ("static", deployable_correct),
                ("random_same_budget", random_correct),
                ("oracle_bridge", True),
            ]
        ]
    }

    result = audit(
        eval_report=eval_report,
        selection_report=selection_report,
        rows=rows,
        bootstrap_samples=100,
    )

    assert result["trajectory"]["first_em"] == 0.5
    assert result["trajectory"]["last_em"] == 0.5
    assert result["trajectory"]["repaired_count"] == 1
    assert result["trajectory"]["regressed_count"] == 1
    assert result["selection_initially_wrong_states"]["static"]["accuracy"] == 1.0
    assert result["gates"]["rl_ready"] is False
    assert result["recommendation"] == "before_rl:fix_evidence_to_answer_transition_and_stop_policy"
