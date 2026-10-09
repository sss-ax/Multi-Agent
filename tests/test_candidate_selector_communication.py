from workflow_runtime.candidate_selector_communication import (
    build_candidate_fragments,
    rendered_fragment_tokens,
    render_visible_candidate_fragments,
    run_selector_with_fragments,
    select_fragment_ids,
)


def _candidates():
    return [
        {
            "candidate_id": "c1",
            "artifact": "full reasoning for 48",
            "final_value": "48",
            "score": {
                "final_value_valid": True,
                "arithmetic_consistency": True,
                "constraint_consistency": True,
                "trace_result_agreement": True,
                "independent_recompute_consistency": True,
            },
            "raw_tail": "24*2 = 48\nFinal answer: 48",
        },
        {
            "candidate_id": "c2",
            "artifact": "full reasoning for 42",
            "final_value": "42",
            "score": {
                "final_value_valid": True,
                "constraint_inconsistency": True,
                "trace_result_disagreement": True,
            },
            "raw_tail": "wrong trace",
        },
    ]


def test_selector_cannot_access_untransmitted_fragments():
    fragments = build_candidate_fragments(_candidates())
    visible_ids = tuple(
        fragment.fragment_id for fragment in fragments
        if fragment.fragment_name in {"final_value", "status"}
    )
    visible = render_visible_candidate_fragments(fragments, visible_ids)

    assert "verifier_summary" not in visible["c1"]
    assert "full_artifact" not in visible["c1"]
    assert "trace" not in visible["c1"]


def test_canonical_hidden_candidate_info_is_not_read_without_visibility():
    full = run_selector_with_fragments(_candidates(), policy="selector_send_all")
    core = run_selector_with_fragments(_candidates(), policy="selector_core_only")

    assert full.rendered_tokens > core.rendered_tokens
    assert any("constraint_consistency" in reason for reason in full.selection_reason)
    assert not any("constraint_consistency" in reason for reason in core.selection_reason)


def test_requested_fragment_next_round_is_visible():
    fragments = build_candidate_fragments(_candidates())
    initial_ids = select_fragment_ids(fragments, policy="selector_receiver_aware")
    visible = render_visible_candidate_fragments(fragments, initial_ids)

    assert "expression" in visible["c2"]
    assert "trace" not in visible["c2"]


def test_unrequested_fragment_not_rendered():
    fragments = build_candidate_fragments(_candidates())
    ids = select_fragment_ids(fragments, policy="selector_verifier_only")
    visible = render_visible_candidate_fragments(fragments, ids)

    assert "verifier_summary" in visible["c1"]
    assert "full_artifact" not in visible["c1"]
    assert "trace" not in visible["c1"]


def test_token_accounting_matches_rendered_candidate_fragments():
    fragments = build_candidate_fragments(_candidates())
    ids = select_fragment_ids(fragments, policy="selector_verifier_only")
    result = run_selector_with_fragments(_candidates(), policy="selector_verifier_only")

    assert result.rendered_tokens == rendered_fragment_tokens(fragments, ids)


def test_selector_send_all_matches_full_candidate_reading():
    fragments = build_candidate_fragments(_candidates())
    ids = select_fragment_ids(fragments, policy="selector_send_all")

    assert set(ids) == {fragment.fragment_id for fragment in fragments}


def test_receiver_aware_matches_send_all_quality_with_lower_cost():
    send_all = run_selector_with_fragments(_candidates(), policy="selector_send_all")
    receiver_aware = run_selector_with_fragments(_candidates(), policy="selector_receiver_aware")

    assert receiver_aware.selected_candidate_id == send_all.selected_candidate_id == "c1"
    assert receiver_aware.rendered_tokens < send_all.rendered_tokens
    assert receiver_aware.rendered_tokens <= int(send_all.rendered_tokens * 0.8)


def test_random_same_budget_uses_same_or_lower_budget_than_receiver_aware():
    fragments = build_candidate_fragments(_candidates())
    receiver_aware_ids = select_fragment_ids(fragments, policy="selector_receiver_aware")
    budget = rendered_fragment_tokens(fragments, receiver_aware_ids)
    random_ids = select_fragment_ids(
        fragments,
        policy="selector_random_same_budget",
        seed=7,
        budget_tokens=budget,
    )

    assert rendered_fragment_tokens(fragments, random_ids) <= budget


def test_oracle_minimal_sends_only_target_core_and_verifier_summary():
    fragments = build_candidate_fragments(_candidates())
    ids = select_fragment_ids(
        fragments,
        policy="selector_oracle_minimal",
        oracle_candidate_id="c1",
    )

    assert set(ids) == {"c1#final_value", "c1#status", "c1#verifier_summary"}


def test_receiver_aware_tie_break_prefers_earlier_generation():
    candidates = [
        {
            "candidate_id": "c0",
            "artifact": "def f(): return 1",
            "final_value": "def f(): return 1",
            "score": {
                "syntax_valid": True,
                "compile_success": True,
                "runtime_ok": True,
                "no_exception": True,
                "test_pass_rate": 0.0,
                "failed_test_count": 0,
                "confidence": 0.25,
            },
        },
        {
            "candidate_id": "c3",
            "artifact": "def f(): return 1",
            "final_value": "def f(): return 1",
            "score": {
                "syntax_valid": True,
                "compile_success": True,
                "runtime_ok": True,
                "no_exception": True,
                "test_pass_rate": 0.0,
                "failed_test_count": 0,
                "confidence": 0.25,
            },
        },
    ]

    result = run_selector_with_fragments(candidates, policy="selector_receiver_aware")

    assert result.selected_candidate_id == "c0"
