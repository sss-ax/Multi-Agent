from scripts.evaluate_domain_workflow import evaluate_verifier_guided_candidates


class FakeModel:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, *, seed, temperature, top_p, do_sample):
        self.calls += 1
        text = "2+2 = 5\nFinal answer: 5" if self.calls % 2 else "2+2 = 4\nFinal answer: 4"
        return text, {"input_tokens": 10, "output_tokens": 6, "latency_sec": 0.0}


def test_verifier_guided_candidate_mode_report_schema_and_selection():
    rows = [
        {
            "sample_id": "fake_gsm8k_0",
            "task_type": "numeric_solve",
            "question": "What is 2+2?",
            "gold_answer": "4",
        }
    ]

    report = evaluate_verifier_guided_candidates(
        rows=rows,
        domain="gsm8k",
        model=FakeModel(),
        candidate_count=2,
        seed=0,
        temperature=0.7,
        top_p=0.95,
        current_baseline=0.0,
        data_path="fake.jsonl",
    )

    assert report["execution_mode"] == "verifier_guided_candidates"
    assert report["candidate_count"] == 2
    assert report["accuracy_or_pass_at_1"] == 1.0
    assert report["verifier_select_at_n"] == 1.0
    assert report["oracle_select_at_n"] == 1.0
    assert report["candidate_graph_invariant_error_count"] == 0
    assert report["records"][0]["selected_candidate_id"] == "c1"
    assert report["records"][0]["candidate_graph"]["invariant_errors"] == []


def test_verifier_guided_candidate_mode_rejects_unsupported_domain():
    try:
        evaluate_verifier_guided_candidates(
            rows=[],
            domain="hotpotqa",
            model=FakeModel(),
            candidate_count=2,
            seed=0,
            temperature=0.7,
            top_p=0.95,
            current_baseline=0.0,
        )
    except ValueError as exc:
        assert "supports gsm8k" in str(exc)
    else:
        raise AssertionError("unsupported verifier-guided candidate domain was accepted")
