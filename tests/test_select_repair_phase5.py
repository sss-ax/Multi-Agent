import scripts.run_reasoning_upper_bounds as rub
from scripts.run_reasoning_upper_bounds import (
    DOMAINS,
    candidate_artifact,
    deployable_candidate_records,
    repair_transition,
    run_best_of_n,
    run_fixed_candidate_pool,
    run_select_repair,
)
from workflow_runtime.candidate_selector_communication import build_candidate_fragments, select_fragment_ids


class QueueModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def generate(self, prompt, *, seed, temperature, top_p, do_sample):
        self.prompts.append(prompt)
        text = self.outputs.pop(0)
        return text, {"input_tokens": 10, "output_tokens": 5, "latency_sec": 0.0}


def _row(sample_id="r0"):
    return {
        "sample_id": sample_id,
        "task_type": "numeric_solve",
        "question": "What is 2+2?",
        "gold_answer": "4",
    }


def _code_row(sample_id="c0"):
    return {
        "sample_id": sample_id,
        "task_type": "code_generation",
        "question": "Write a function add_one(x) that returns x + 1.",
        "requirements": {"text": "def add_one(x): ..."},
        "tests": [{"text": "assert add_one(1) == 2"}],
    }


def _mmlu_row(sample_id="m0"):
    return {
        "sample_id": sample_id,
        "task_type": "multiple_choice",
        "question": "Which option is correct?\n\nA. wrong\nB. right\n\nAnswer with only the option letter.",
        "choices": [{"label": "A", "text": "wrong"}, {"label": "B", "text": "right"}],
        "gold_answer": "B",
    }


def _hotpot_row(sample_id="h0"):
    return {
        "sample_id": sample_id,
        "task_type": "multihop_qa",
        "question": "Where was Ada born?",
        "entities": ["Ada", "London"],
        "supporting_facts": [
            {"fact_id": "supporting_fact_1", "title": "Ada", "text": "Ada was born in London."},
            {"fact_id": "supporting_fact_2", "title": "London", "text": "London is a city in England."},
        ],
        "gold_answer": "London",
    }


def test_phase5_gsm8k_does_not_repair_by_default():
    model = QueueModel([
        "2+2 = 5\nFinal answer: 5",
        "2+2 = 6\nFinal answer: 6",
    ])

    report = run_select_repair(
        model,
        [_row()],
        domain="gsm8k",
        n=2,
        seed=0,
        temperature=0.7,
        top_p=0.95,
        verifier_baseline=0.0,
    )

    record = report["records"][0]
    assert record["action"] == "SELECT"
    assert record["transition"] == "W->W"
    assert report["repair_attempt_count"] == 0
    assert report["net_repair_gain"] == 0


def test_phase5_mbpp_wrong_to_correct_repair_transition():
    model = QueueModel([
        "def add_one(x):\n    return x\n\nassert add_one(1) == 2",
        "def add_one(x):\n    return x\n\nassert add_one(1) == 2",
        "def add_one(x):\n    return x + 1",
    ])

    report = run_select_repair(
        model,
        [_code_row()],
        domain="mbpp",
        n=4,
        seed=0,
        temperature=0.7,
        top_p=0.95,
        verifier_baseline=0.0,
        min_repair_candidates=2,
    )

    record = report["records"][0]
    assert record["action"] == "REPAIR"
    assert record["transition"] == "W->C"
    assert report["net_repair_gain"] == 1
    assert report["repair_attempt_count"] == 1
    assert report["successful_repair_count"] == 1


def test_phase5_mbpp_waits_for_min_repair_candidates():
    model = QueueModel([
        "def add_one(x):\n    return x\n\nassert add_one(1) == 2",
        "def add_one(x):\n    return x + 1",
        "def add_one(x):\n    return x + 1",
    ])

    report = run_select_repair(
        model,
        [_code_row()],
        domain="mbpp",
        n=4,
        seed=0,
        temperature=0.7,
        top_p=0.95,
        verifier_baseline=0.0,
        min_repair_candidates=2,
    )

    record = report["records"][0]
    assert record["generation_attempt_count"] == 2
    assert record["action"] == "SELECT"
    assert record["actions"] == ["GENERATE", "GENERATE", "SELECT", "STOP"]
    assert record["controller_reasons"][0] == "insufficient_confidence_generate_more"


def test_phase5_humaneval_does_not_repair_without_strong_gate():
    model = QueueModel([
        "def add_one(x):\n    return x",
        "def add_one(x):\n    return x",
    ])

    report = run_select_repair(
        model,
        [_code_row()],
        domain="humaneval",
        n=2,
        seed=0,
        temperature=0.7,
        top_p=0.95,
        verifier_baseline=0.0,
    )

    record = report["records"][0]
    assert record["action"] == "SELECT"
    assert record["transition"] == "W->W"
    assert report["repair_attempt_count"] == 0


def test_phase5_correct_to_correct_selects_without_repair():
    model = QueueModel([
        "2+2 = 4\nFinal answer: 4",
        "2+2 = 4\nFinal answer: 4",
    ])

    report = run_select_repair(
        model,
        [_row()],
        domain="gsm8k",
        n=2,
        seed=0,
        temperature=0.7,
        top_p=0.95,
        verifier_baseline=1.0,
    )

    record = report["records"][0]
    assert record["action"] == "SELECT"
    assert record["transition"] == "C->C"
    assert report["repair_attempt_count"] == 0
    assert report["phase5_gate_select_repair_ge_verifier"] is True


def test_phase5_can_use_receiver_aware_selector_communication():
    model = QueueModel([
        "2+2 = 4\nFinal answer: 4",
        "2+2 = 5\nFinal answer: 5",
    ])

    report = run_select_repair(
        model,
        [_row()],
        domain="gsm8k",
        n=2,
        seed=0,
        temperature=0.7,
        top_p=0.95,
        verifier_baseline=0.0,
        selector_communication_policy="selector_receiver_aware",
    )

    record = report["records"][0]
    assert report["selector_communication_policy"] == "selector_receiver_aware"
    assert report["selector_input_tokens"] > 0
    assert report["selector_send_all_equivalent_tokens"] >= report["selector_input_tokens"]
    assert report["selector_token_saving_vs_send_all"] >= 0
    assert record["selector_communication_steps"][0]["policy"] == "selector_receiver_aware"


def test_phase5_mmlu_pro_uses_selector_comm_schema_without_repair():
    wrong_a = '{"support": ["plausible"], "contradictions": [], "support_strength": 0.8, "contradiction_strength": 0.0, "uncertainty": 0.1}'
    weak_b = '{"support": [], "contradictions": ["weak"], "support_strength": 0.1, "contradiction_strength": 0.4, "uncertainty": 0.2}'
    weak_a = '{"support": [], "contradictions": ["conflict"], "support_strength": 0.1, "contradiction_strength": 0.6, "uncertainty": 0.1}'
    right_b = '{"support": ["matches"], "contradictions": [], "support_strength": 0.9, "contradiction_strength": 0.0, "uncertainty": 0.1}'
    model = QueueModel([
        wrong_a, weak_b,
        weak_a, right_b,
        weak_a, right_b,
    ])

    report = run_select_repair(
        model,
        [_mmlu_row()],
        domain="mmlu_pro",
        n=3,
        seed=0,
        temperature=0.7,
        top_p=0.95,
        verifier_baseline=0.0,
        selector_communication_policy="selector_receiver_aware",
        experiment_name="adaptive_controller_selector_comm",
    )

    record = report["records"][0]
    assert report["experiment"] == "adaptive_controller_selector_comm"
    assert report["selector_input_tokens"] > 0
    assert report["selector_send_all_equivalent_tokens"] >= report["selector_input_tokens"]
    assert report["repair_attempt_count"] == 0
    assert record["action"] == "SELECT"
    assert record["correct"] is True
    assert record["deployable_candidates"][0]["verification_source"] == "deployable_mmlu_option_check"


def test_phase5_hotpotqa_uses_selector_comm_schema_without_repair():
    model = QueueModel([
        "Final answer: Paris",
        "Final answer: London",
        "Final answer: London",
    ])

    report = run_select_repair(
        model,
        [_hotpot_row()],
        domain="hotpotqa",
        n=3,
        seed=0,
        temperature=0.7,
        top_p=0.95,
        verifier_baseline=0.0,
        selector_communication_policy="selector_receiver_aware",
        experiment_name="adaptive_controller_selector_comm",
    )

    record = report["records"][0]
    assert report["experiment"] == "adaptive_controller_selector_comm"
    assert report["selector_input_tokens"] > 0
    assert report["repair_attempt_count"] == 0
    assert record["correct"] is True
    assert record["deployable_candidates"][1]["verification_source"] == "deployable_hotpot_answer_evidence_check"


def test_reasoning_upper_bounds_domain_artifacts_and_deployable_signals():
    assert "mmlu_pro" in DOMAINS
    assert "hotpotqa" in DOMAINS
    assert candidate_artifact("mmlu_pro", "After analysis, final answer: B") == ("choice", "B", "B")
    option_json = (
        '{"option_hypotheses": ['
        '{"option_id": "A", "support": [], "contradictions": ["conflict"], '
        '"support_strength": 0.1, "contradiction_strength": 0.8, "uncertainty": 0.1},'
        '{"option_id": "B", "support": ["matches"], "contradictions": [], '
        '"support_strength": 0.9, "contradiction_strength": 0.0, "uncertainty": 0.1}'
        "]}"
    )
    assert candidate_artifact("mmlu_pro", option_json) == ("choice", "B", "B")
    assert candidate_artifact("hotpotqa", "Reasoning...\nFinal answer: London") == ("answer", "London", "London")

    mmlu_candidate = {
        "candidate_id": "c0",
        "artifact_type": "choice",
        "artifact": "B",
        "final_value": "B",
        "raw_tail": "Final answer: B",
    }
    hotpot_candidate = {
        "candidate_id": "c0",
        "artifact_type": "answer",
        "artifact": "London",
        "final_value": "London",
        "raw_tail": "Final answer: London",
    }

    assert deployable_candidate_records([mmlu_candidate], domain="mmlu_pro", row=_mmlu_row())[0]["score"]["option_valid"]
    assert deployable_candidate_records([hotpot_candidate], domain="hotpotqa", row=_hotpot_row())[0]["score"]["evidence_covered"]


def test_mmlu_pro_generation_uses_option_hypothesis_builder_and_marks_redundancy():
    wrong_a = '{"support": ["looks plausible"], "contradictions": [], "support_strength": 0.8, "contradiction_strength": 0.0, "uncertainty": 0.1}'
    weak_b = '{"support": [], "contradictions": ["less likely"], "support_strength": 0.1, "contradiction_strength": 0.4, "uncertainty": 0.2}'
    right_a_weak = '{"support": [], "contradictions": ["conflict"], "support_strength": 0.1, "contradiction_strength": 0.7, "uncertainty": 0.1}'
    right_b = '{"support": ["matches"], "contradictions": [], "support_strength": 0.9, "contradiction_strength": 0.0, "uncertainty": 0.1}'
    model = QueueModel([
        wrong_a, weak_b,
        wrong_a, weak_b,
        right_a_weak, right_b,
    ])

    report = run_best_of_n(
        model,
        [_mmlu_row()],
        domain="mmlu_pro",
        n=3,
        seed=0,
        temperature=0.7,
        top_p=0.95,
    )

    candidates = report["records"][0]["candidates"]
    assert candidates[0]["generation_role"] == "option_independent_hypothesis_builder:independent"
    assert candidates[1]["generation_role"] == "option_independent_hypothesis_builder:independent"
    assert "Evaluate this hypothesis only:" in model.prompts[0]
    assert "option_id" not in model.prompts[0]
    assert "A. wrong" not in model.prompts[0]
    assert "B. right" not in model.prompts[0]
    assert candidates[0]["final_value"] == "A"
    assert candidates[2]["final_value"] == "B"
    assert candidates[2]["option_hypotheses"][0]["option_id"] == "A"
    assert candidates[2]["option_hypotheses"][1]["option_id"] == "B"
    assert candidates[2]["option_hypothesis_view"]["selected_option_id"] == "B"
    assert candidates[2]["option_hypothesis_view"]["options"][1]["support_fragments"] == ["matches"]
    assert candidates[2]["option_hypothesis_view"]["options"][1]["support_score"] == 0.9
    assert candidates[2]["option_hypothesis_view"]["options"][1]["contradiction_score"] == 0.0
    assert candidates[2]["option_hypothesis_view"]["options"][1]["aggregate_score"] == 0.8
    fragments = build_candidate_fragments([candidates[2]])
    selected = select_fragment_ids(fragments, policy="selector_receiver_aware")
    assert "c2#option_hypothesis_graph" in selected
    assert candidates[1]["redundant_candidate"] is True


def test_mmlu_pro_low_margin_uses_pairwise_top2_critic():
    option_a = (
        '{"support": ["partially fits"], "contradictions": [], '
        '"support_strength": 0.55, "contradiction_strength": 0.10, "uncertainty": 0.10}'
    )
    option_b = (
        '{"support": ["also fits"], "contradictions": [], '
        '"support_strength": 0.50, "contradiction_strength": 0.10, "uncertainty": 0.10}'
    )
    pairwise = '{"selected_option_id": "B", "reason": "B better satisfies the question", "confidence": 0.8}'
    model = QueueModel([option_a, option_b, pairwise])

    report = run_best_of_n(
        model,
        [_mmlu_row()],
        domain="mmlu_pro",
        n=1,
        seed=0,
        temperature=0.7,
        top_p=0.95,
    )

    candidate = report["records"][0]["candidates"][0]
    assert candidate["final_value"] == "B"
    assert candidate["runtime_selection"]["selection_source"] == "pairwise_top2_critic"
    assert candidate["pairwise_critic"]["selected_option_id"] == "B"
    assert "Compare only the two listed options" in model.prompts[-1]


def test_phase2_mmlu_permutation_mode_records_structured_candidate():
    old_mode = rub.MMLU_OPTIONWISE_MODE
    rub.MMLU_OPTIONWISE_MODE = "permutation"
    try:
        model = QueueModel([
            '{"support": ["right"], "contradictions": [], "support_strength": 0.9, "contradiction_strength": 0.0, "uncertainty": 0.1}',
            '{"support": [], "contradictions": ["wrong"], "support_strength": 0.1, "contradiction_strength": 0.6, "uncertainty": 0.1}',
        ])
        report = run_best_of_n(
            model,
            [_mmlu_row()],
            domain="mmlu_pro",
            n=1,
            seed=0,
            temperature=0.7,
            top_p=0.95,
        )
    finally:
        rub.MMLU_OPTIONWISE_MODE = old_mode

    candidate = report["records"][0]["candidates"][0]
    assert candidate["correct"] is True
    assert candidate["generation_role"] == "option_independent_hypothesis_builder:permutation"
    assert candidate["option_hypothesis_view"]["selected_option_id"] == "B"
    assert report["mmlu_optionwise_mode_counts"] == {"permutation": 1}


def test_phase2_mmlu_second_pass_mode_uses_independent_emphasis():
    old_mode = rub.MMLU_OPTIONWISE_MODE
    rub.MMLU_OPTIONWISE_MODE = "second_pass"
    try:
        model = QueueModel([
            '{"support": [], "contradictions": ["wrong"], "support_strength": 0.1, "contradiction_strength": 0.6, "uncertainty": 0.1}',
            '{"support": ["right"], "contradictions": [], "support_strength": 0.9, "contradiction_strength": 0.0, "uncertainty": 0.1}',
        ])
        report = run_best_of_n(
            model,
            [_mmlu_row()],
            domain="mmlu_pro",
            n=1,
            seed=0,
            temperature=0.7,
            top_p=0.95,
        )
    finally:
        rub.MMLU_OPTIONWISE_MODE = old_mode

    candidate = report["records"][0]["candidates"][0]
    assert candidate["correct"] is True
    assert candidate["final_value"] == "B"
    assert candidate["generation_role"] == "option_independent_hypothesis_builder:second_pass"
    assert "Produce an independent second hypothesis evaluation" in model.prompts[0]
    assert report["score_calibration_parse_fail"] == 0


def test_phase2_hotpot_preserve_first_generator_records_chain_binding():
    old_mode = rub.HOTPOT_CANDIDATE_MODE
    rub.HOTPOT_CANDIDATE_MODE = "preserve_first"
    try:
        model = QueueModel([
            '{"reasoning_type":"bridge_entity","fact1":{"subject":"Ada","relation":"born_in","object":"London","subject_role":"entity","object_role":"entity","confidence":1.0},"fact2":{"subject":"London","relation":"located_in","object":"England","subject_role":"entity","object_role":"entity","confidence":1.0},"cross_fact":{"inferred_bridge":"London","bridge_alignment":"inferred","derived_answer":"England","answer_source":"fact2"},"confidence":0.9}',
            '{"answer":"London"}',
            '{"reasoning_type":"bridge_entity","fact1":{"subject":"Ada","relation":"born_in","object":"London","subject_role":"entity","object_role":"entity","confidence":1.0},"fact2":{"subject":"London","relation":"located_in","object":"England","subject_role":"entity","object_role":"entity","confidence":1.0},"cross_fact":{"inferred_bridge":"London","bridge_alignment":"inferred","derived_answer":"England","answer_source":"fact2"},"confidence":0.9}',
            '{"answer":"England"}',
        ])
        report = run_best_of_n(
            model,
            [_hotpot_row()],
            domain="hotpotqa",
            n=2,
            seed=0,
            temperature=0.7,
            top_p=0.95,
        )
    finally:
        rub.HOTPOT_CANDIDATE_MODE = old_mode

    candidates = report["records"][0]["candidates"]
    assert candidates[0]["generation_role"] == "preserve_first_evidence_chain_binding"
    assert candidates[0]["bridge_entity"]
    assert candidates[0]["cited_fact_ids"]
    assert candidates[0]["structured_candidate_view"]["bridge_entity"] == candidates[0]["bridge_entity"]
    assert report["hotpot_candidate_mode"] == "preserve_first"
    assert report["avg_unique_bridges"] >= 1
    assert report["avg_unique_evidence_chains"] >= 1


def test_phase2_hotpot_fixed_candidate_pool_disables_controller_and_generates_k():
    old_mode = rub.HOTPOT_CANDIDATE_MODE
    rub.HOTPOT_CANDIDATE_MODE = "preserve_first"
    try:
        model = QueueModel([
            '{"cross_fact":{"inferred_bridge":"London","derived_answer":"England"}}',
            '{"answer":"London"}',
            '{"cross_fact":{"inferred_bridge":"London","derived_answer":"England"}}',
            '{"answer":"England"}',
            '{"cross_fact":{"inferred_bridge":"London","derived_answer":"England"}}',
            '{"answer":"London"}',
            '{"cross_fact":{"inferred_bridge":"London","derived_answer":"England"}}',
            '{"answer":"England"}',
        ])
        report = run_fixed_candidate_pool(
            model,
            [_hotpot_row()],
            domain="hotpotqa",
            n=4,
            seed=0,
            temperature=0.7,
            top_p=0.95,
        )
    finally:
        rub.HOTPOT_CANDIDATE_MODE = old_mode

    record = report["records"][0]
    assert report["experiment"] == "fixed_candidate_pool"
    assert report["selector_disabled"] is True
    assert report["repair_disabled"] is True
    assert report["adaptive_early_stop_disabled"] is True
    assert record["candidate_count"] == 4
    assert report["candidate_count_mean"] == 4
    assert report["hotpot_fixed_k_gate"]["candidate_count_pass"] is True


def test_phase2_hotpot_pair_construction_keeps_full_3x3_grid():
    hop1_pool = [
        {"fact_id": f"h1_{index}", "title": "Ada", "text": "Ada evidence", "hop_score": 1.0}
        for index in range(3)
    ]
    hop2_pool = [
        {"fact_id": f"h2_{index}", "title": "London", "text": "London evidence", "hop_score": 1.0}
        for index in range(3)
    ]

    pairs = rub._hotpot_rank_chain_pairs(_hotpot_row(), "Ada", hop1_pool, hop2_pool)

    assert len(pairs) == 9
    assert {
        (pair["hop1"]["fact_id"], pair["hop2"]["fact_id"])
        for pair in pairs
    } == {
        (f"h1_{left}", f"h2_{right}")
        for left in range(3)
        for right in range(3)
    }
    assert all("retrieval_score" in pair for pair in pairs)
    assert all("semantic_chain_score" in pair for pair in pairs)
    assert all("semantic_chain_verification" in pair for pair in pairs)


def test_phase2_hotpot_semantic_chain_score_is_separate_from_retrieval_score():
    low_retrieval_pair = rub._hotpot_rank_chain_pairs(
        _hotpot_row(),
        "Ada",
        [{"fact_id": "h1", "title": "Ada", "text": "Ada evidence", "retrieval_score": 0.1}],
        [{"fact_id": "h2", "title": "London", "text": "London evidence", "retrieval_score": 0.1}],
    )[0]
    high_retrieval_pair = rub._hotpot_rank_chain_pairs(
        _hotpot_row(),
        "Ada",
        [{"fact_id": "h1", "title": "Ada", "text": "Ada evidence", "retrieval_score": 9.0}],
        [{"fact_id": "h2", "title": "London", "text": "London evidence", "retrieval_score": 9.0}],
    )[0]

    assert high_retrieval_pair["retrieval_score"] > low_retrieval_pair["retrieval_score"]
    assert high_retrieval_pair["semantic_chain_score"] == low_retrieval_pair["semantic_chain_score"]


def test_phase2_hotpot_semantic_chain_verifier_schema_and_score():
    pair = rub._hotpot_rank_chain_pairs(
        _hotpot_row(),
        "Ada",
        [{"fact_id": "h1", "title": "Ada", "text": "Ada was born in London.", "retrieval_score": 1.0}],
        [{"fact_id": "h2", "title": "London", "text": "London is a city in England.", "retrieval_score": 1.0}],
    )[0]

    verdict = pair["semantic_chain_verification"]
    assert {
        "reasoning_type",
        "candidate_bridge",
        "inferred_bridge",
        "bridge_hypothesis_match",
        "evidence_chain_valid",
        "evidence_valid",
        "answer_supported",
        "failure_reason",
        "type_specific_details",
        "hop1_relation_valid",
        "hop2_relation_valid",
        "bridge_entity_aligned",
        "bridge_match_type",
        "chain_complete",
        "question_relation_supported",
        "derived_answer",
        "contradiction",
        "confidence",
        "entity_resolution",
        "relation_endpoints",
        "reasoning_ir",
        "programmatic_gates",
        "gate_results",
    }.issubset(set(verdict))
    assert pair["semantic_chain_score"] == rub._hotpot_semantic_chain_score(verdict)
    ir = verdict["reasoning_ir"]
    assert {"topology", "question", "constraints", "facts", "bindings", "derived"}.issubset(set(ir))
    assert {"endpoint_grounding", "relation_compatibility", "topology_consistency", "answer_grounding"}.issubset(set(verdict["gate_results"]))


def test_phase7_hotpot_topology_specific_scores_use_constraint_bindings():
    good = {
        "reasoning_type": "multi_constraint_lookup",
        "type_specific_details": {
            "constraint_bindings": [
                {"anchor_match": True, "relation_match": True, "target_valid": True},
                {"anchor_match": True, "relation_match": True, "target_valid": True},
            ],
            "same_target": True,
        },
    }
    wrong_anchor = {
        "reasoning_type": "multi_constraint_lookup",
        "type_specific_details": {
            "constraint_bindings": [
                {"anchor_match": False, "relation_match": True, "target_valid": True},
                {"anchor_match": True, "relation_match": True, "target_valid": True},
            ],
            "same_target": True,
        },
    }

    assert rub._hotpot_semantic_chain_score(good) > rub._hotpot_semantic_chain_score(wrong_anchor)


def test_phase2_hotpot_bridge_entity_resolution_matches_aliases():
    pair = rub._hotpot_rank_chain_pairs(
        _hotpot_row(),
        "Robert Downey Jr.",
        [{"fact_id": "h1", "title": "Downey", "text": "Downey starred in the film.", "retrieval_score": 1.0}],
        [{"fact_id": "h2", "title": "Robert Downey Jr.", "text": "The actor was born in New York.", "retrieval_score": 1.0}],
    )[0]

    verdict = pair["semantic_chain_verification"]
    assert verdict["entity_resolution"]["hop1_bridge_match"]["matched"] is True
    assert verdict["entity_resolution"]["hop2_bridge_match"]["matched"] is True
    assert "downey" in verdict["entity_resolution"]["hop1_bridge_match"]["matched_aliases"]
    assert verdict["bridge_match_type"] in {"alias", "title_entity", "exact"}


def test_phase2_hotpot_relation_endpoint_matching_uses_bridge_endpoint_equivalence():
    pair = rub._hotpot_rank_chain_pairs(
        _hotpot_row(),
        "Robert Downey Jr.",
        [{"fact_id": "h1", "title": "Tony Stark", "text": "Tony Stark was played by Downey.", "retrieval_score": 1.0}],
        [{"fact_id": "h2", "title": "Robert Downey Jr.", "text": "Robert Downey Jr. was born in New York.", "retrieval_score": 1.0}],
    )[0]

    endpoints = pair["semantic_chain_verification"]["relation_endpoints"]
    assert endpoints["hop1"]["target_entity"] == "Downey"
    assert endpoints["hop2"]["source_entity"] == "Robert Downey Jr."
    assert endpoints["bridge_endpoint_equivalent"] is True


def test_phase2_hotpot_candidate_bridge_is_hypothesis_not_verifier_anchor():
    row = {
        **_hotpot_row(),
        "question": "2014 S/S is the debut album of a South Korean boy group that was formed by who?",
        "entities": ["2014 S/S", "Winner (band)"],
    }
    pair = rub._hotpot_rank_chain_pairs(
        row,
        "2014 S/S",
        [{"fact_id": "h1", "title": "2014 S/S", "text": "2014 S/S is the debut album of South Korean group WINNER.", "retrieval_score": 1.0}],
        [{"fact_id": "h2", "title": "Winner (band)", "text": "Winner is a South Korean boy group formed in 2013 by YG Entertainment.", "retrieval_score": 1.0}],
    )[0]

    verdict = pair["semantic_chain_verification"]
    assert verdict["reasoning_type"] == "bridge_attribute"
    assert verdict["inferred_bridge"] == "Winner (band)"
    assert verdict["bridge_hypothesis_match"] is False
    assert verdict["evidence_chain_valid"] is True


def test_phase2_hotpot_chain_complete_does_not_automatically_support_question_relation():
    pair = rub._hotpot_rank_chain_pairs(
        _hotpot_row(),
        "Ada",
        [{"fact_id": "h1", "title": "Ada", "text": "Ada worked with Ada.", "retrieval_score": 1.0}],
        [{"fact_id": "h2", "title": "Ada", "text": "Ada wrote a paper.", "retrieval_score": 1.0}],
    )[0]

    verdict = pair["semantic_chain_verification"]
    assert verdict["relation_endpoints"]["hop1"]["relation"] == "unresolved"
    assert verdict["relation_endpoints"]["hop1"]["target_entity"] == "unresolved"
    assert verdict["chain_complete"] is False
    assert verdict["question_relation_supported"] is False


def test_phase2_hotpot_reasoning_type_routes_parallel_comparison_without_bridge_alignment():
    row = {
        **_hotpot_row(),
        "question": "Are both Dictyosperma and Huernia described as a genus?",
        "entities": ["Dictyosperma", "Huernia"],
    }
    pair = rub._hotpot_rank_chain_pairs(
        row,
        "genus",
        [{"fact_id": "h1", "title": "Dictyosperma", "text": "Dictyosperma is a monotypic genus of flowering plant.", "retrieval_score": 1.0}],
        [{"fact_id": "h2", "title": "Huernia", "text": "The genus Huernia consists of stem succulents.", "retrieval_score": 1.0}],
    )[0]

    verdict = pair["semantic_chain_verification"]
    assert verdict["reasoning_type"] == "shared_property_yesno"
    assert verdict["evidence_valid"] is True
    assert verdict["answer_supported"] is True
    assert verdict["bridge_entity_aligned"] is False


def test_phase2_hotpot_topology_specific_construction_bypasses_bridge_pool_for_parallel_type():
    row = {
        **_hotpot_row(),
        "question": "Are both Dictyosperma and Huernia described as a genus?",
        "entities": ["Dictyosperma", "Huernia"],
    }
    facts = [
        {"fact_id": "f1", "title": "Dictyosperma", "text": "Dictyosperma is a monotypic genus of flowering plant."},
        {"fact_id": "f2", "title": "Huernia", "text": "The genus Huernia consists of stem succulents."},
        {"fact_id": "f3", "title": "Unrelated", "text": "Unrelated was founded by Someone."},
    ]

    options = rub._hotpot_hierarchical_chain_options(
        row=row,
        facts=facts,
        bridge_records=[],
        previous_signatures=set(),
    )

    assert options
    assert options[0]["allocation_policy"] == "topology_specific_shared_property_yesno"
    pair = options[0]["binding"]["chain_pair_pool"][0]
    verifier = pair["semantic_chain_verification"]
    assert verifier["reasoning_candidate"]["reasoning_type"] == "shared_property_yesno"
    assert verifier["reasoning_candidate"]["evidence_bindings"]["entity_a"] == "Dictyosperma"
    assert verifier["reasoning_candidate"]["evidence_bindings"]["entity_b"] == "Huernia"


def test_phase9_hotpot_parallel_retrieval_is_entity_side_and_relation_compatible():
    row = {
        **_hotpot_row(),
        "question": "Are both Dictyosperma and Huernia described as a genus?",
        "entities": ["Dictyosperma", "Huernia"],
    }
    facts = [
        {"fact_id": "f1", "title": "Dictyosperma", "text": "Dictyosperma is a monotypic genus of flowering plant."},
        {"fact_id": "f2", "title": "Huernia", "text": "The genus Huernia consists of stem succulents."},
        {"fact_id": "f3", "title": "Dictyosperma", "text": "Dictyosperma grows on islands."},
        {"fact_id": "f4", "title": "Huernia", "text": "Huernia has angular stems."},
    ]

    options = rub._hotpot_hierarchical_chain_options(
        row=row,
        facts=facts,
        bridge_records=[],
        previous_signatures=set(),
    )

    assert options
    assert options[0]["constructor_family"] == "entity_side_parallel_retrieval"
    assert options[0]["view"].cited_fact_ids == ["f1", "f2"]
    telemetry = options[0]["parallel_retrieval_telemetry"]
    assert telemetry["entity_a"] == "Dictyosperma"
    assert telemetry["entity_b"] == "Huernia"
    assert telemetry["entity_a_relation_compatible_count"] >= 1
    assert telemetry["entity_b_relation_compatible_count"] >= 1
    assert telemetry["entity_a_value_count"] >= 1
    assert telemetry["entity_b_value_count"] >= 1


def test_phase9_hotpot_bridge_family_construction_derives_bridge_from_evidence():
    row = {
        **_hotpot_row(),
        "question": "What company formed the group that created 2014 S/S?",
        "entities": ["2014 S/S"],
    }
    facts = [
        {"fact_id": "f1", "title": "2014 S/S", "text": "2014 S/S was created by WINNER."},
        {"fact_id": "f2", "title": "WINNER", "text": "WINNER was formed by YG Entertainment."},
        {"fact_id": "f3", "title": "WINNER", "text": "WINNER is a South Korean boy band."},
    ]

    options = rub._hotpot_hierarchical_chain_options(
        row=row,
        facts=facts,
        bridge_records=[],
        previous_signatures=set(),
    )

    assert options
    assert options[0]["allocation_policy"] == "topology_native_bridge_attribute"
    assert options[0]["constructor_family"] == "evidence_derived_bridge"
    assert options[0]["view"].cited_fact_ids == ["f1", "f2"]
    verifier = options[0]["binding"]["chain_pair_pool"][0]["semantic_chain_verification"]
    assert verifier["inferred_bridge"] == "WINNER"
    assert verifier["evidence_valid"] is True


def test_phase9_hotpot_bridge_attribute_hop2_is_relation_conditioned():
    row = {
        **_hotpot_row(),
        "question": "What company formed the group that created 2014 S/S?",
        "entities": ["2014 S/S"],
    }
    facts = [
        {"fact_id": "f_wrong", "title": "WINNER", "text": "WINNER is a South Korean boy band."},
        {"fact_id": "f_gold", "title": "WINNER", "text": "WINNER was formed by YG Entertainment."},
    ]

    hop2_pool = rub._hotpot_rank_relation_specific_hop2(
        row,
        "WINNER",
        facts,
        reasoning_type="bridge_attribute",
        limit=2,
    )

    assert hop2_pool[0]["fact_id"] == "f_gold"
    assert hop2_pool[0]["about_bridge"] is True
    assert hop2_pool[0]["requested_relation_match"] is True
    assert hop2_pool[0]["target_role_match"] is True
    assert hop2_pool[1]["requested_relation_match"] is False


def test_phase9_hotpot_constraint_construction_intersects_targets():
    row = {
        **_hotpot_row(),
        "question": "Who wrote Novel X and directed Film Y?",
        "entities": ["Novel X", "Film Y"],
    }
    facts = [
        {"fact_id": "f1", "title": "Novel X", "text": "Novel X was written by Jordan Lee."},
        {"fact_id": "f2", "title": "Film Y", "text": "Film Y was directed by Jordan Lee."},
        {"fact_id": "f3", "title": "Novel Z", "text": "Novel Z was written by Jordan Lee."},
        {"fact_id": "f4", "title": "Film Y", "text": "Film Y was directed by Morgan Park."},
    ]

    options = rub._hotpot_hierarchical_chain_options(
        row=row,
        facts=facts,
        bridge_records=[],
        previous_signatures=set(),
    )

    assert options
    assert options[0]["allocation_policy"] == "topology_native_multi_constraint_lookup"
    assert options[0]["constructor_family"] == "constraint_target_intersection"
    assert options[0]["view"].cited_fact_ids == ["f1", "f2"]
    details = options[0]["binding"]["chain_pair_pool"][0]["semantic_chain_verification"]["type_specific_details"]
    assert details["same_target"] is True
    assert details["constraint_1_supported"] is True
    assert details["constraint_2_supported"] is True


def test_phase9_hotpot_constraint_construction_requires_shared_target_before_pairing():
    row = {
        **_hotpot_row(),
        "question": "Who wrote Novel X and directed Film Y?",
        "entities": ["Novel X", "Film Y"],
    }
    facts = [
        {"fact_id": "f1", "title": "Novel X", "text": "Novel X was written by Jordan Lee."},
        {"fact_id": "f2", "title": "Film Y", "text": "Film Y was directed by Morgan Park."},
        {"fact_id": "f3", "title": "Film Y", "text": "Film Y was directed by Jordan Lee."},
    ]

    options = rub._hotpot_hierarchical_chain_options(
        row=row,
        facts=facts,
        bridge_records=[],
        previous_signatures=set(),
    )

    assert options
    assert options[0]["view"].cited_fact_ids == ["f1", "f3"]
    telemetry = options[0]["multi_constraint_telemetry"]
    assert telemetry["target_set_sizes"] == [1, 2]
    assert telemetry["shared_target_count"] == 1
    assert telemetry["shared_targets"] == ["Jordan Lee"]


def test_phase2_hotpot_reasoning_type_routes_bridge_attribute():
    row = {
        **_hotpot_row(),
        "question": "2014 S/S is the debut album of a South Korean boy group that was formed by who?",
        "entities": ["2014 S/S", "Winner (band)"],
    }
    assert rub._hotpot_reasoning_type(row) == "bridge_attribute"


def test_phase2_hotpot_bridge_pool_filters_attribute_values_by_role():
    row = {
        **_hotpot_row(),
        "question": "Are both Dictyosperma and Huernia described as a genus?",
        "entities": ["Dictyosperma", "Huernia"],
    }
    facts = [
        {"fact_id": "f1", "title": "Dictyosperma", "text": "Dictyosperma is a South Korean genus from Eastern islands."},
        {"fact_id": "f2", "title": "Huernia", "text": "The genus Huernia is found in Southern Africa."},
    ]

    pool = rub._hotpot_raw_bridge_pool(row, facts=facts)
    names = {item.name for item in pool}

    assert "Dictyosperma" in names
    assert "Huernia" in names
    assert "South Korean" not in names
    assert "Eastern" not in names
    assert "Southern" not in names
    assert rub._hotpot_ontology_role("South Korean") == "ATTRIBUTE_VALUE"
    assert rub._hotpot_ontology_role("genus") == "ATTRIBUTE_VALUE"


def test_phase2_hotpot_fact_semantics_keeps_attribute_value_role():
    row = {
        **_hotpot_row(),
        "question": "Are both Dictyosperma and Huernia described as a genus?",
        "entities": ["Dictyosperma", "Huernia"],
    }
    semantics = rub._hotpot_fact_semantics(
        row,
        {"title": "Dictyosperma", "text": "Dictyosperma is a monotypic genus of flowering plant."},
    )

    assert semantics["object"] == "genus"
    assert semantics["object_role"] == "attribute"


def test_phase3_hotpot_semantic_chain_records_reasoning_candidate_and_verification_result():
    row = {
        "question": "What company formed the group that created 2014 S/S?",
        "entities": ["2014 S/S"],
    }
    pair = rub._hotpot_rank_chain_pairs(
        row,
        "WINNER",
        [{"fact_id": "f1", "title": "2014 S/S", "text": "2014 S/S was created by WINNER."}],
        [{"fact_id": "f2", "title": "WINNER", "text": "WINNER was formed by YG Entertainment."}],
    )[0]

    verdict = pair["semantic_chain_verification"]
    candidate = verdict["reasoning_candidate"]
    result = verdict["verification_result"]

    assert candidate["reasoning_type"] == "bridge_attribute"
    assert candidate["evidence_bindings"]["inferred_bridge"] == "WINNER"
    assert candidate["metadata"]["candidate_bridge_is_hypothesis_only"] is True
    assert result["reasoning_type"] == "bridge_attribute"
    assert result["evidence_valid"] is True
    assert result["reasoning_complete"] is True
    assert result["answer_supported"] is True


def test_phase3_hotpot_bridge_attribute_rejects_wrong_requested_relation():
    row = {
        "question": "What company formed the group that created 2014 S/S?",
        "entities": ["2014 S/S"],
    }
    pair = rub._hotpot_rank_chain_pairs(
        row,
        "WINNER",
        [{"fact_id": "f1", "title": "2014 S/S", "text": "2014 S/S was created by WINNER."}],
        [{"fact_id": "f2", "title": "WINNER", "text": "WINNER is a South Korean boy band."}],
    )[0]

    verifier = pair["semantic_chain_verification"]
    details = verifier["type_specific_details"]

    assert verifier["reasoning_type"] == "bridge_attribute"
    assert details["hop2_subject_matches_bridge"] is True
    assert details["requested_relation"] == "formed_by"
    assert details["fact_relation"] == "unresolved"
    assert details["hop2_relation_matches_request"] is False
    assert details["hop2_target_type_matches"] is False
    assert verifier["evidence_valid"] is False
    assert verifier["failure_code"] == "requested_relation_mismatch"
    assert verifier["gate_results"]["relation_compatibility"]["pass"] is False


def test_phase3_hotpot_bridge_attribute_accepts_compatible_relation_alias():
    row = {
        "question": "What company founded the group that created 2014 S/S?",
        "entities": ["2014 S/S"],
    }
    pair = rub._hotpot_rank_chain_pairs(
        row,
        "WINNER",
        [{"fact_id": "f1", "title": "2014 S/S", "text": "2014 S/S was created by WINNER."}],
        [{"fact_id": "f2", "title": "WINNER", "text": "WINNER was formed by YG Entertainment."}],
    )[0]

    verifier = pair["semantic_chain_verification"]
    details = verifier["type_specific_details"]

    assert details["requested_relation"] == "formed_by"
    assert details["fact_relation"] == "formed_by"
    assert details["relation_match_type"] == "exact"
    assert verifier["evidence_valid"] is True
    assert verifier["answer_supported"] is True


def test_phase3_hotpot_multi_constraint_rejects_wrong_anchor_even_with_shared_target():
    row = {
        "question": "Who wrote Novel X and directed Film Y?",
        "entities": ["Novel X", "Film Y"],
    }
    pair = rub._hotpot_rank_chain_pairs(
        row,
        "Jordan Lee",
        [{"fact_id": "f1", "title": "Novel Z", "text": "Novel Z was written by Jordan Lee."}],
        [{"fact_id": "f2", "title": "Film Y", "text": "Film Y was directed by Jordan Lee."}],
    )[0]

    verifier = pair["semantic_chain_verification"]
    details = verifier["type_specific_details"]

    assert verifier["reasoning_type"] == "multi_constraint_lookup"
    assert details["constraint_1_supported"] is False
    assert details["constraint_2_supported"] is True
    assert details["same_target"] is True
    assert verifier["evidence_valid"] is False
    assert verifier["failure_code"] == "constraint_not_satisfied"


def test_phase3_hotpot_multi_constraint_requires_anchor_relation_and_same_target():
    row = {
        "question": "Who wrote Novel X and directed Film Y?",
        "entities": ["Novel X", "Film Y"],
    }
    pair = rub._hotpot_rank_chain_pairs(
        row,
        "Jordan Lee",
        [{"fact_id": "f1", "title": "Novel X", "text": "Novel X was written by Jordan Lee."}],
        [{"fact_id": "f2", "title": "Film Y", "text": "Film Y was directed by Jordan Lee."}],
    )[0]

    verifier = pair["semantic_chain_verification"]
    details = verifier["type_specific_details"]

    assert verifier["reasoning_type"] == "multi_constraint_lookup"
    assert details["constraint_1_supported"] is True
    assert details["constraint_2_supported"] is True
    assert details["same_target"] is True
    assert verifier["evidence_valid"] is True
    assert verifier["answer_supported"] is True


def test_phase9_hotpot_question_semantics_validates_multi_constraint_schema():
    row = {
        "question": "Who wrote Novel X and directed Film Y?",
        "entities": ["Novel X", "Film Y"],
        "supporting_facts": [
            {"fact_id": "f1", "title": "Novel X", "text": "Novel X was written by Jordan Lee."},
            {"fact_id": "f2", "title": "Film Y", "text": "Film Y was directed by Jordan Lee."},
        ],
    }

    qs = rub._hotpot_question_semantics(row)

    assert qs.reasoning_type == "multi_constraint_lookup"
    assert qs.valid is True
    assert [constraint.anchor for constraint in qs.constraints] == ["Novel X", "Film Y"]
    assert [constraint.relation for constraint in qs.constraints] == ["written_by", "directed_by"]
    assert qs.constraint_graph_type == "convergent_shared_target"
    assert qs.answer_var == "X"
    assert [constraint.target_var for constraint in qs.constraints] == ["X", "X"]
    assert all(constraint.direction == "anchor_to_target" for constraint in qs.constraints)
    assert [constraint.context_title for constraint in qs.constraints] == ["Novel X", "Film Y"]
    assert all(constraint.resolution_type == "exact_title" for constraint in qs.constraints)


def test_phase9_hotpot_question_semantics_supports_serial_constraints():
    row = {
        "question": "What government position was held by the woman who portrayed Corliss Archer?",
        "entities": ["Corliss Archer", "Shirley Temple"],
        "supporting_facts": [
            {"fact_id": "f1", "title": "Shirley Temple", "text": "Shirley Temple portrayed Corliss Archer in the radio series."},
            {"fact_id": "f2", "title": "Shirley Temple", "text": "Shirley Temple served as Chief of Protocol of the United States."},
        ],
    }

    qs = rub._hotpot_question_semantics(row)

    assert qs.reasoning_type == "multi_constraint_lookup"
    assert qs.constraint_graph_type == "serial"
    assert qs.answer_var == "Y"
    assert [constraint.anchor for constraint in qs.constraints] == ["Corliss Archer", "X"]
    assert [constraint.relation for constraint in qs.constraints] == ["acted_in", "position"]
    assert [constraint.target_var for constraint in qs.constraints] == ["X", "Y"]
    assert qs.constraints[1].depends_on == "X"


def test_phase9_hotpot_serial_constraint_constructor_binds_intermediate_then_answer():
    row = {
        "question": "What government position was held by the woman who portrayed Corliss Archer?",
        "entities": ["Corliss Archer", "Shirley Temple"],
    }
    facts = [
        {"fact_id": "f1", "title": "Shirley Temple", "text": "Shirley Temple portrayed Corliss Archer in the radio series."},
        {"fact_id": "f2", "title": "Shirley Temple", "text": "Shirley Temple served as Chief of Protocol of the United States."},
        {"fact_id": "f3", "title": "Other Person", "text": "Other Person served as mayor."},
    ]

    options = rub._hotpot_hierarchical_chain_options(
        row=row,
        facts=facts,
        bridge_records=[],
        previous_signatures=set(),
    )

    assert options
    assert options[0]["constructor_family"] == "serial_constraint_binding"
    assert options[0]["view"].cited_fact_ids == ["f1", "f2"]
    telemetry = options[0]["multi_constraint_telemetry"]
    assert telemetry["constraint_graph_type"] == "serial"
    assert telemetry["answer_var"] == "Y"
    assert telemetry["shared_targets"] == ["Chief of Protocol of the United States"]
    details = options[0]["binding"]["chain_pair_pool"][0]["semantic_chain_verification"]["type_specific_details"]
    assert details["variable_bindings"]["X"] == "Shirley Temple"
    assert details["variable_bindings"]["Y"] == "Chief of Protocol of the United States"
    assert details["answer_supported"] is True


def test_phase9_hotpot_constraint_retrieval_prefers_bound_title_before_global_distractor():
    row = {
        "question": "Who wrote Novel X and directed Film Y?",
        "entities": ["Novel X", "Film Y"],
        "supporting_facts": [
            {"fact_id": "gold", "title": "Novel X", "text": "Novel X was written by Jordan Lee."},
            {"fact_id": "distractor", "title": "Novel Z", "text": "Novel Z was written by Jordan Lee."},
        ],
    }
    facts = rub._hotpot_facts(row)
    index = rub._hotpot_evidence_index(row, facts)
    constraint = rub._hotpot_question_constraints(row)[0]

    targets = rub._hotpot_retrieve_relation_targets(
        row,
        constraint,
        facts,
        role="constraint_1",
        limit=5,
        evidence_index=index,
    )

    assert targets
    assert targets[0]["fact_id"] == "gold"
    assert targets[0]["context_title"] == "Novel X"
    assert targets[0]["anchor_tier"] == 0


def test_phase9_hotpot_relation_semantics_preserves_written_by_direction():
    row = {
        "question": "Who wrote Novel X?",
        "entities": ["Novel X"],
    }

    passive = rub._hotpot_fact_semantics(
        row,
        {"title": "Novel X", "text": "Novel X was written by Jordan Lee."},
    )
    active = rub._hotpot_fact_semantics(
        row,
        {"title": "Novel X", "text": "Jordan Lee wrote Novel X."},
    )

    assert passive["relation"] == "written_by"
    assert passive["subject"] == "Novel X"
    assert passive["object"] == "Jordan Lee"
    assert active["relation"] == "written_by"
    assert active["subject"] == "Novel X"
    assert active["object"] == "Jordan Lee"


def test_phase9_hotpot_bridge_inference_uses_relation_endpoint_not_ner_order():
    row = {
        "question": "Where was the director of Film A born?",
        "entities": ["Film A"],
    }
    fact = {"fact_id": "f1", "title": "Film A", "text": "Film A was directed by Jane Doe in 2011."}

    assert rub._hotpot_inferred_intermediate_from_fact(row, fact) == "Jane Doe"


def test_phase3_hotpot_bridge_entity_uses_source_fact_to_infer_bridge_then_answer_fact():
    row = {
        "question": "Where was the director of Film A born?",
        "entities": ["Film A"],
    }
    facts = [
        {"fact_id": "supporting_fact_1", "title": "Film A", "text": "Film A was directed by Jane Doe."},
        {"fact_id": "supporting_fact_2", "title": "Jane Doe", "text": "Jane Doe was born in Paris."},
        {"fact_id": "supporting_fact_3", "title": "Film A", "text": "Film A was released in 2014."},
    ]

    options = rub._hotpot_bridge_entity_chain_options(
        row=row,
        facts=facts,
        previous_signatures=set(),
    )

    assert options
    first = options[0]
    assert first["constructor_family"] == "evidence_derived_bridge_entity"
    assert first["bridge"] == "Jane Doe"
    assert first["view"].hop1_fact_ids == ["supporting_fact_1"]
    assert first["view"].hop2_fact_ids == ["supporting_fact_2"]
    telemetry = first["bridge_entity_telemetry"]
    assert telemetry["source_entity"] == "Film A"
    assert telemetry["source_fact_id"] == "supporting_fact_1"
    assert telemetry["inferred_bridge"] == "Jane Doe"
    assert "supporting_fact_2" in telemetry["hop2_pool_fact_ids"]


def test_phase9_hotpot_bridge_entity_does_not_infer_answer_side_location_as_bridge():
    row = {
        "question": "Big Stone Gap's director is based in what New York city?",
        "entities": ["Big Stone Gap"],
    }
    director_fact = {
        "fact_id": "f_director",
        "title": "Big Stone Gap",
        "text": "Big Stone Gap was written and directed by Adriana Trigiani.",
    }
    based_fact = {
        "fact_id": "f_based",
        "title": "Adriana Trigiani",
        "text": "Adriana Trigiani is based in Greenwich Village, New York City.",
    }

    assert rub._hotpot_question_relation_label(row) == "located_in"
    assert rub._hotpot_bridge_source_relation_hint(row) == "directed_by"
    assert rub._hotpot_inferred_intermediate_from_fact(row, director_fact) == "Adriana Trigiani"
    assert rub._hotpot_inferred_intermediate_from_fact(row, based_fact) == ""

    pair = rub._hotpot_rank_chain_pairs(row, "Adriana Trigiani", [director_fact], [based_fact])[0]
    verdict = pair["semantic_chain_verification"]
    assert verdict["bridge_entity_aligned"] is True
    assert verdict["evidence_valid"] is True
    assert verdict["answer_supported"] is True


def test_phase9_hotpot_home_venue_capacity_bridge_uses_arena_not_location():
    row = {
        "question": "The Lewiston Maineiacs home games arena can seat how many?",
        "entities": ["Lewiston Maineiacs"],
    }
    team_fact = {
        "fact_id": "f_team",
        "title": "Lewiston Maineiacs",
        "text": "The Lewiston Maineiacs played their home games at Androscoggin Bank Colisée.",
    }
    capacity_fact = {
        "fact_id": "f_arena",
        "title": "Androscoggin Bank Colisée",
        "text": "Androscoggin Bank Colisée can seat 3,677 spectators.",
    }

    assert rub._hotpot_question_relation_label(row) == "capacity"
    assert rub._hotpot_bridge_source_relation_hint(row) == "home_venue"
    assert rub._hotpot_inferred_intermediate_from_fact(row, team_fact) == "Androscoggin Bank Colisée"

    pair = rub._hotpot_rank_chain_pairs(row, "Androscoggin Bank Colisée", [team_fact], [capacity_fact])[0]
    verdict = pair["semantic_chain_verification"]
    assert verdict["bridge_entity_aligned"] is True
    assert verdict["evidence_valid"] is True
    assert verdict["derived_answer"] == "3,677"


def test_phase9_hotpot_yes_no_both_routes_to_parallel_not_bridge_entity():
    row = {
        "question": "Are Giuseppe Verdi and Ambroise Thomas both Opera composers?",
        "entities": ["Giuseppe Verdi", "Ambroise Thomas"],
    }

    assert rub._hotpot_reasoning_type(row) == "shared_property_yesno"
    classification = rub._hotpot_topology_classification(row)
    assert classification["reasoning_family"] == "parallel"
    assert classification["reasoning_type"] == "shared_property_yesno"
    assert classification["answer_mode"] == "yes_no"
    assert "shared_property_yesno" in classification["compatible_topologies"]


def test_phase9_hotpot_low_confidence_compositional_keeps_compatible_backups():
    row = {
        "question": "What company formed the group that created 2014 S/S?",
        "entities": ["2014 S/S"],
    }

    classification = rub._hotpot_topology_classification(row)

    assert classification["reasoning_family"] == "compositional"
    assert classification["reasoning_type"] == "bridge_attribute"
    assert classification["confidence"] < 0.8
    assert classification["compatible_topologies"][:2] == ("bridge_attribute", "bridge_entity")
    assert "multi_constraint_lookup" in classification["compatible_topologies"]


def test_phase9_hotpot_constraint_binding_can_invert_portrayed_relation():
    row = {
        "question": "What government position was held by the woman who portrayed Corliss Archer?",
        "entities": ["Corliss Archer", "Shirley Temple"],
    }
    constraint = {"anchor": "Corliss Archer", "relation": "acted_in"}
    fact = {
        "fact_id": "f_actor",
        "title": "Shirley Temple",
        "text": "Shirley Temple portrayed Corliss Archer in the radio series.",
    }

    binding = rub._hotpot_best_constraint_binding(row, constraint, [fact], used_fact_ids=set())

    assert binding["anchor_match"] is True
    assert binding["relation_match"] is True
    assert binding["target"] == "Shirley Temple"
    assert binding["satisfied"] is True


def test_phase5_mbpp_missing_public_tests_passed_does_not_trigger_repair():
    model = QueueModel([
        "def add_one(x):\n    return x + 1",
        "def add_one(x):\n    return x + 1",
    ])

    report = run_select_repair(
        model,
        [_code_row()],
        domain="mbpp",
        n=2,
        seed=0,
        temperature=0.7,
        top_p=0.95,
        verifier_baseline=0.0,
        min_repair_candidates=2,
        selector_communication_policy="selector_receiver_aware",
    )

    record = report["records"][0]
    assert record["action"] == "SELECT"
    assert record["transition"] == "C->C"
    assert report["repair_attempt_count"] == 0


def test_phase5_transition_labels_include_regression():
    assert repair_transition(True, True) == "C->C"
    assert repair_transition(True, False) == "C->W"
    assert repair_transition(False, True) == "W->C"
    assert repair_transition(False, False) == "W->W"
