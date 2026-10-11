from workflow_runtime.hotpot_representation import HotpotChainView

from scripts.run_hotpotqa_bridge_evidence_selection_ablation import (
    _bundle_for_missing_fragments,
    _cheap_should_stop,
    _parse_provisional_selection,
    _cheap_selector_telemetry,
    _diagnose_selector_need,
)


def test_answer_disagreement_uses_symmetric_reveal_for_unresolved_candidates() -> None:
    views = [
        HotpotChainView(
            candidate_id="c1",
            bridge_entity="Scott Derrickson",
            hop1_fact_ids=["supporting_fact_1"],
            hop1_key_sentence="Scott Derrickson is an American film director.",
            hop2_fact_ids=["supporting_fact_2"],
            hop2_key_sentence="Ed Wood was an American filmmaker.",
            derived_answer="yes",
            answer_type="yes_no",
            answer_alignment="supported",
            cited_fact_ids=["supporting_fact_1", "supporting_fact_2"],
        ),
        HotpotChainView(
            candidate_id="c2",
            bridge_entity="Ed Wood",
            hop1_fact_ids=["supporting_fact_3"],
            hop1_key_sentence="Ed Wood was an American filmmaker.",
            hop2_fact_ids=["supporting_fact_4"],
            hop2_key_sentence="However, another source indicates a different nationality.",
            derived_answer="no",
            answer_type="yes_no",
            answer_alignment="supported",
            cited_fact_ids=["supporting_fact_3", "supporting_fact_4"],
        ),
        HotpotChainView(
            candidate_id="c3",
            bridge_entity="Tim Burton",
            hop1_fact_ids=["supporting_fact_5"],
            hop1_key_sentence="Tim Burton directed Ed Wood.",
            hop2_fact_ids=["supporting_fact_6"],
            hop2_key_sentence="Scott Derrickson directed another film.",
            derived_answer="unknown",
            answer_type="yes_no",
            answer_alignment="supported",
            cited_fact_ids=["supporting_fact_5", "supporting_fact_6"],
        ),
    ]

    need = _diagnose_selector_need(
        views,
        "Were Scott Derrickson and Ed Wood of the same nationality?",
    )

    assert need.disagreement_type == "answer_disagreement"
    revealed = [tuple(need.missing_fragments[candidate_id]) for candidate_id in need.unresolved_candidates]
    assert len(set(revealed)) == 1
    assert revealed[0] == ("hop1_evidence", "hop2_evidence", "answer_alignment")
    assert {_bundle_for_missing_fragments(list(value)) for value in revealed} == {"B2"}


def test_answer_disagreement_does_not_trigger_reveal_when_comparison_is_confident() -> None:
    views = [
        HotpotChainView(
            candidate_id="c1",
            bridge_entity="Scott Derrickson",
            hop1_fact_ids=["supporting_fact_1"],
            hop1_key_sentence="Scott Derrickson is an American film director.",
            hop2_fact_ids=["supporting_fact_2"],
            hop2_key_sentence="Ed Wood was an American filmmaker.",
            derived_answer="yes",
            answer_type="yes_no",
            answer_alignment="supported",
            cited_fact_ids=["supporting_fact_1", "supporting_fact_2"],
        ),
        HotpotChainView(
            candidate_id="c2",
            bridge_entity="Ed Wood",
            hop1_fact_ids=["supporting_fact_3"],
            hop1_key_sentence="Ed Wood was an American filmmaker.",
            hop2_fact_ids=["supporting_fact_4"],
            hop2_key_sentence="However, another source indicates a different nationality.",
            derived_answer="no",
            answer_type="yes_no",
            answer_alignment="contradicted",
            cited_fact_ids=["supporting_fact_3", "supporting_fact_4"],
        ),
    ]

    need = _diagnose_selector_need(
        views,
        "Were Scott Derrickson and Ed Wood of the same nationality?",
    )

    assert need.disagreement_type is None
    assert need.missing_fragments == {}

    calibrated_need = _diagnose_selector_need(
        views,
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        receiver_reveal_gap_threshold=0.70,
    )

    assert calibrated_need.disagreement_type == "answer_disagreement"
    assert calibrated_need.missing_fragments


def test_reveal_policy_uses_comparison_bundles_not_partial_fragments() -> None:
    assert _bundle_for_missing_fragments([]) == "B0"
    assert _bundle_for_missing_fragments(["hop1_evidence"]) == "B1"
    assert _bundle_for_missing_fragments(["hop2_evidence"]) == "B1"
    assert _bundle_for_missing_fragments(["hop1_evidence", "answer_alignment"]) == "B2"


def test_parse_provisional_selection_reads_scores_and_ranking() -> None:
    chains = [
        {"candidate_id": "c1", "bridge_entity": "A"},
        {"candidate_id": "c2", "bridge_entity": "B"},
    ]

    parsed = _parse_provisional_selection(
        '{"ranking":["c2","c1"],"selected_candidate_id":"c2",'
        '"top1_score":0.81,"top2_score":0.42,"confidence":0.77,'
        '"uncertainty_type":"low"}',
        chains,
    )

    assert parsed["selected_bridge"] == "B"
    assert parsed["ranking"][:2] == ["c2", "c1"]
    assert parsed["top1_score"] == 0.81
    assert parsed["top2_score"] == 0.42
    assert parsed["confidence"] == 0.77
    assert parsed["parse_fail"] is False


def test_cheap_selector_stop_uses_b0_for_all_candidates() -> None:
    event = {
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "candidate_chains": [
            HotpotChainView(
                candidate_id="c1",
                bridge_entity="Scott Derrickson",
                hop1_fact_ids=[],
                hop1_key_sentence="",
                hop2_fact_ids=[],
                hop2_key_sentence="",
                derived_answer="yes",
                answer_type="yes_no",
                answer_alignment="supported",
                cited_fact_ids=[],
            ).as_dict(),
            HotpotChainView(
                candidate_id="c2",
                bridge_entity="Ed Wood",
                hop1_fact_ids=[],
                hop1_key_sentence="",
                hop2_fact_ids=[],
                hop2_key_sentence="",
                derived_answer="no",
                answer_type="yes_no",
                answer_alignment="contradicted",
                cited_fact_ids=[],
            ).as_dict(),
        ],
    }

    telemetry = _cheap_selector_telemetry(
        event,
        provisional={
            "ranking": ["c1", "c2"],
            "selected_candidate_id": "c1",
            "top1_score": 0.9,
            "top2_score": 0.2,
            "confidence": 0.85,
            "uncertainty_type": "low",
        },
        receiver_reveal_gap_threshold=0.25,
        receiver_cheap_high_confidence_threshold=0.85,
        receiver_cheap_confidence_threshold=0.7,
        stopped_after_initial=True,
        bundle_pattern=None,
    )

    assert telemetry["stopped_after_initial"] is True
    assert telemetry["bundle_pattern"] == {"c1": "B0", "c2": "B0"}
    assert telemetry["requested_extra_tokens"] == 0
    assert telemetry["requested_candidate_count"] == 0


def test_cheap_selector_uses_three_level_stop_rule() -> None:
    assert _cheap_should_stop(
        parse_fail=False,
        confidence=0.90,
        gap=0.01,
        tau_high=0.85,
        tau_conf=0.70,
        tau_gap=0.15,
    )
    assert _cheap_should_stop(
        parse_fail=False,
        confidence=0.80,
        gap=0.15,
        tau_high=0.85,
        tau_conf=0.70,
        tau_gap=0.15,
    )
    assert not _cheap_should_stop(
        parse_fail=False,
        confidence=0.80,
        gap=0.10,
        tau_high=0.85,
        tau_conf=0.70,
        tau_gap=0.15,
    )
    assert not _cheap_should_stop(
        parse_fail=True,
        confidence=0.95,
        gap=0.30,
        tau_high=0.85,
        tau_conf=0.70,
        tau_gap=0.15,
    )
