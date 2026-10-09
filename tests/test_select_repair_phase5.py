from scripts.run_reasoning_upper_bounds import repair_transition, run_select_repair


class QueueModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def generate(self, prompt, *, seed, temperature, top_p, do_sample):
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
