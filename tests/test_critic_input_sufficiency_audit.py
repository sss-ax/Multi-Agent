from scripts.run_critic_input_sufficiency_audit import (
    _hotpot_evidence_selector_audit,
    parse_selected_candidate,
    render_critic_prompt,
    run_audit,
)


class FakeAuditModel:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, *, seed, temperature, top_p, do_sample, max_new_tokens=None):
        self.calls += 1
        if self.calls == 1:
            text = "I think the answer is B.\nFinal answer: B"
        elif self.calls == 2:
            text = "The correct option is A.\nFinal answer: A"
        elif "Task context:" in prompt and "A. correct option" in prompt and "rationale_or_output" in prompt:
            text = '{"selected_candidate_id":"c1","reason":"full task shows A is correct"}'
        else:
            text = '{"selected_candidate_id":"c0","reason":"compact is ambiguous"}'
        return text, {"input_tokens": len(prompt.split()), "output_tokens": len(text.split()), "latency_sec": 0.0}


def test_critic_input_sufficiency_audit_reports_setting_accuracy():
    rows = [
        {
            "sample_id": "m0",
            "domain": "mmlu_pro",
            "task_type": "multiple_choice",
            "question": "Pick the correct option.\n\nA. correct option\nB. distractor\n\nAnswer with only the option letter.",
            "choices": [{"label": "A", "text": "correct option"}, {"label": "B", "text": "distractor"}],
            "gold_answer": "A",
        }
    ]

    report = run_audit(
        model=FakeAuditModel(),
        rows=rows,
        domain="mmlu_pro",
        n=2,
        seed=0,
        temperature=0.7,
        top_p=0.95,
    )

    assert report["oracle_coverage"] == 1.0
    assert report["policies"]["critic_compact"]["accuracy"] == 0.0
    assert report["policies"]["critic_full_task"]["accuracy"] == 1.0
    assert "structured_selected_context" in report["policies"]
    assert "structured_loop" in report["policies"]
    assert report["records"][0]["settings"]["critic_full_task"]["selected_candidate_id"] == "c1"


def test_parse_selected_candidate_accepts_common_short_forms():
    for text in [
        "c2",
        "C2",
        "2",
        "candidate: 2",
        "selected_candidate_id: c2",
        '{"selected_candidate_id":"c2"}',
        "c2\nThe selected candidate ID is c2 because...",
        "```c2\n```",
        "c2. Final answer: something",
    ]:
        assert parse_selected_candidate(text, 4) == ("c2", True)


def test_parse_selected_candidate_falls_back_to_c0_on_invalid_output():
    assert parse_selected_candidate("not sure", 4) == ("c0", False)


def test_hotpotqa_structured_prompt_contains_evidence_packet_fields():
    row = {
        "sample_id": "h0",
        "domain": "hotpotqa",
        "task_type": "multihop_qa",
        "question": "What city is connected to the bridge entity?",
        "entities": ["Bridge Entity"],
        "supporting_facts": [
            {"title": "Bridge Entity", "text": "Bridge Entity was born in Paris."},
            {"title": "Paris", "text": "Paris is a city in France."},
        ],
        "gold_answer": "Paris",
    }
    candidates = [
        type("C", (), {
            "candidate_id": "c0",
            "answer": "Paris",
            "text": "Bridge Entity links to Paris. Final answer: Paris",
        })()
    ]

    prompt = render_critic_prompt(row, "hotpotqa", candidates, "structured_selected_context")

    assert "bridge_entity" in prompt
    assert "hop1_supporting_fact" in prompt
    assert "answer_evidence_alignment" in prompt
    assert "missing_hop_signal" in prompt


def test_hotpotqa_audit_reports_cross_candidate_evidence_settings():
    row = {
        "sample_id": "h2",
        "domain": "hotpotqa",
        "task_type": "multihop_qa",
        "question": "What city is connected to the bridge entity?",
        "entities": ["Bridge Entity", "Paris"],
        "supporting_facts": [
            {"title": "Bridge Entity", "sent_id": 0, "text": "Bridge Entity was born in Paris."},
            {"title": "Paris", "sent_id": 0, "text": "Paris is a city in France."},
        ],
        "evidence_links": [
            {"entity": "Bridge Entity", "sent_id": 0},
            {"entity": "Paris", "sent_id": 0},
        ],
        "gold_answer": "Paris",
    }

    report = run_audit(
        model=FakeAuditModel(),
        rows=[row],
        domain="hotpotqa",
        n=2,
        seed=0,
        temperature=0.7,
        top_p=0.95,
    )

    assert "per_candidate_evidence" in report["policies"]
    assert "union_evidence_pool" in report["policies"]
    assert "oracle_evidence_chain" in report["policies"]
    assert "evidence_selector_audit" in report


def test_hotpotqa_union_and_oracle_prompts_use_shared_evidence_graph():
    row = {
        "sample_id": "h3",
        "domain": "hotpotqa",
        "task_type": "multihop_qa",
        "question": "Where is Bridge Entity's city?",
        "entities": ["Bridge Entity", "Paris"],
        "supporting_facts": [
            {"title": "Bridge Entity", "sent_id": 0, "text": "Bridge Entity was born in Paris."},
            {"title": "Paris", "sent_id": 0, "text": "Paris is a city in France."},
            {"title": "Noise", "sent_id": 0, "text": "Unrelated sentence."},
        ],
        "evidence_links": [
            {"entity": "Bridge Entity", "sent_id": 0},
            {"entity": "Paris", "sent_id": 0},
        ],
        "gold_answer": "Paris",
    }
    candidates = [
        type("C", (), {"candidate_id": "c0", "answer": "Paris", "text": "Bridge Entity links to Paris."})(),
        type("C", (), {"candidate_id": "c1", "answer": "France", "text": "Paris is in France."})(),
    ]

    union_prompt = render_critic_prompt(row, "hotpotqa", candidates, "union_evidence_pool")
    oracle_prompt = render_critic_prompt(row, "hotpotqa", candidates, "oracle_evidence_chain")

    assert "Evidence mode: union_evidence_pool" in union_prompt
    assert "Shared evidence graph" in union_prompt
    assert "E1=" in union_prompt
    assert "Evidence mode: oracle_evidence_chain" in oracle_prompt
    assert "Bridge Entity was born in Paris" in oracle_prompt
    assert "Paris is a city in France" in oracle_prompt


def test_mmlu_structured_prompt_contains_option_elimination_fields():
    row = {
        "sample_id": "m1",
        "domain": "mmlu_pro",
        "task_type": "multiple_choice",
        "question": "Pick one.\n\nA. Alpha\nB. Beta\n\nAnswer with only the option letter.",
        "choices": [{"label": "A", "text": "Alpha"}, {"label": "B", "text": "Beta"}],
        "gold_answer": "A",
    }
    candidates = [
        type("C", (), {"candidate_id": "c0", "answer": "A", "text": "A is supported."})(),
        type("C", (), {"candidate_id": "c1", "answer": "B", "text": "B is supported."})(),
    ]

    prompt = render_critic_prompt(row, "mmlu_pro", candidates, "structured_loop")

    assert "candidate_disagreement" in prompt
    assert "eliminate impossible options" in prompt
    assert "Option support/contradiction summary" in prompt


def test_hotpot_evidence_selector_audit_reports_recall_and_bridge_coverage():
    row = {
        "sample_id": "h1",
        "domain": "hotpotqa",
        "task_type": "multihop_qa",
        "question": "Where is Bridge Entity's city?",
        "entities": ["Bridge Entity", "Paris"],
        "supporting_facts": [
            {"title": "Bridge Entity", "sent_id": 0, "text": "Bridge Entity was born in Paris."},
            {"title": "Paris", "sent_id": 0, "text": "Paris is a city in France."},
            {"title": "Noise", "sent_id": 0, "text": "Unrelated sentence."},
        ],
        "evidence_links": [
            {"entity": "Bridge Entity", "sent_id": 0},
            {"entity": "Paris", "sent_id": 0},
        ],
        "gold_answer": "Paris",
    }
    candidate = type("C", (), {
        "candidate_id": "c0",
        "answer": "Paris",
        "text": "Bridge Entity was born in Paris and Paris is in France.",
    })()

    audit = _hotpot_evidence_selector_audit(row, [candidate])

    assert audit["summary"]["max_recall@2"] == 1.0
    assert audit["summary"]["any_bridge_coverage@2"] is True
    assert audit["candidates"][0]["gold_overlap@2"] == 2
