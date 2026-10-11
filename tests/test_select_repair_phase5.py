from scripts.run_reasoning_upper_bounds import (
    DOMAINS,
    candidate_artifact,
    deployable_candidate_records,
    repair_transition,
    run_best_of_n,
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
        "supporting_facts": [
            {"fact_id": "supporting_fact_1", "title": "Ada", "text": "Ada was born in London."}
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
    assert candidates[0]["generation_role"] == "option_independent_hypothesis_builder"
    assert candidates[1]["generation_role"] == "option_independent_hypothesis_builder"
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
