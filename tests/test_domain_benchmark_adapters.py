import gzip
import json

from scripts.evaluate_domain_workflow import action_token_totals, hotpot_f1, hotpot_normalize, optional_consumption_totals, read


def test_read_humaneval_jsonl_gz(tmp_path):
    path = tmp_path / "HumanEval.jsonl.gz"
    row = {
        "task_id": "HumanEval/0",
        "prompt": "def add(a, b):\n",
        "entry_point": "add",
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3",
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")

    loaded = read(path, "humaneval", 1)

    assert loaded[0]["task_type"] == "code_generation"
    assert loaded[0]["tests"][0]["text"].endswith("check(add)")


def test_read_mmlu_pro_options_and_answer(tmp_path):
    path = tmp_path / "mmlu_pro.jsonl"
    row = {
        "question_id": 7,
        "question": "Which letter is correct?",
        "options": ["wrong", "right"],
        "answer": "B",
        "category": "toy",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    loaded = read(path, "mmlu_pro", 1)

    assert loaded[0]["task_type"] == "multiple_choice"
    assert loaded[0]["choices"][1] == {"label": "B", "text": "right"}
    assert loaded[0]["gold_answer"] == "B"


def test_read_hotpotqa_json_array_and_metrics(tmp_path):
    path = tmp_path / "hotpot_dev.json"
    row = {
        "_id": "abc",
        "question": "Where was Ada born?",
        "answer": "London",
        "context": [["Ada", ["Ada was born in London.", "She worked on engines."]]],
        "supporting_facts": [["Ada", 0]],
    }
    path.write_text(json.dumps([row]), encoding="utf-8")

    loaded = read(path, "hotpotqa", 1)

    assert loaded[0]["task_type"] == "multihop_qa"
    assert loaded[0]["entities"] == ["Ada"]
    assert loaded[0]["supporting_facts"][0]["text"] == "Ada was born in London."
    assert hotpot_normalize("The London!") == "london"
    assert hotpot_f1("London", "The London") == 1.0


def test_action_token_totals_aggregate_action_level_telemetry():
    records = [
        {
            "runtime_summary": {
                "records": [
                    {
                        "context_tokens": 12,
                        "graph_update_tokens": 7,
                        "telemetry": {
                            "physical_input_tokens": 100,
                            "logical_input_tokens": 110,
                            "output_tokens": 9,
                            "forward_calls": 1,
                            "graph_read_context_tokens": 12,
                            "graph_update_tokens": 7,
                        },
                    },
                    {
                        "context_tokens": 18,
                        "telemetry": {
                            "physical_input_tokens": 80,
                            "logical_input_tokens": 90,
                            "output_tokens": 5,
                            "forward_calls": 1,
                            "graph_read_context_tokens": 18,
                            "graph_update_tokens": 4,
                        },
                    },
                ],
            },
        }
    ]

    totals = action_token_totals(records)

    assert totals["total_input_tokens"] == 180
    assert totals["total_output_tokens"] == 14
    assert totals["total_model_tokens"] == 194
    assert totals["graph_read_context_tokens"] == 30
    assert totals["graph_update_tokens"] == 11
    assert totals["peak_context_tokens"] == 18


def test_optional_consumption_totals_track_future_receiver_context():
    records = [
        {
            "runtime_summary": {
                "records": [
                    {
                        "role": "critic",
                        "context_tokens": 10,
                        "telemetry": {"physical_input_tokens": 20},
                        "communication": [
                            {
                                "sender": "critic",
                                "receiver": "solver",
                                "selected_optional_root_node_ids": ["verification@v1", "note@v1"],
                                "selected_optional_root_token_by_node": {
                                    "verification@v1": 3,
                                    "note@v1": 5,
                                },
                            }
                        ],
                    },
                    {
                        "role": "solver",
                        "context_tokens": 30,
                        "telemetry": {"physical_input_tokens": 40},
                        "context_slice_node_ids": ["task@v1", "verification@v1"],
                    },
                ],
            },
        }
    ]

    totals = optional_consumption_totals(records)

    assert totals["optional_transmitted_roots"] == 2
    assert totals["optional_transmitted_root_tokens"] == 8
    assert totals["optional_consumed_roots"] == 1
    assert totals["optional_consumed_root_tokens"] == 3
    assert totals["optional_consumption_rate"] == 0.5
    assert totals["optimization_headroom"] == 3 / 40
    assert totals["oracle_saving_upper_bound"] == 3 / 60
    assert totals["optional_consumption_by_pair"]["critic->solver"]["consumed_node_ids"] == ["verification@v1"]
