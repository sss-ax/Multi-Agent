import gzip
import json

from scripts.evaluate_domain_workflow import (
    action_token_totals,
    add_source_nodes,
    communication_totals,
    hotpot_f1,
    hotpot_normalize,
    optional_consumption_totals,
    read,
)
from workflow_runtime import GraphStore


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


def test_multiple_choice_source_nodes_include_allowed_label_schema(tmp_path):
    store = GraphStore()
    store.add_node(task_id="mmlu_1", branch_id="main", logical_id="task", node_type="task", content="pick", owner="user")
    row = {
        "task_type": "multiple_choice",
        "choices": [{"label": "A", "text": "alpha"}, {"label": "B", "text": "beta"}],
    }

    add_source_nodes(None, store, row, "mmlu_1")

    schema = store.latest_valid("mmlu_1", "main", "choice_schema")
    assert json.loads(schema.content)["allowed_labels"] == ["A", "B"]


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
                        "role": "solver",
                        "mode": "normal",
                        "context_tokens": 12,
                        "graph_update_tokens": 7,
                        "action": {"op": "set_result", "id": "R1", "value": 5},
                        "telemetry": {
                            "role": "solver",
                            "physical_input_tokens": 100,
                            "logical_input_tokens": 110,
                            "output_tokens": 9,
                            "forward_calls": 1,
                            "graph_read_context_tokens": 12,
                            "graph_update_tokens": 7,
                            "persistent_context_tokens": 4,
                            "incremental_context_tokens": 8,
                            "logical_communication_tokens": 8,
                            "system_prompt_tokens": 6,
                            "prompt_wrapper_tokens": 92,
                            "reusable_prefix_tokens": 10,
                            "reusable_prefix_key": "solver:normal:numeric_solve",
                        },
                    },
                    {
                        "role": "solver",
                        "mode": "repair",
                        "context_tokens": 18,
                        "action": {"op": "set_result", "id": "R1", "value": 6},
                        "telemetry": {
                            "role": "solver",
                            "physical_input_tokens": 80,
                            "logical_input_tokens": 90,
                            "output_tokens": 5,
                            "forward_calls": 1,
                            "graph_read_context_tokens": 18,
                            "graph_update_tokens": 4,
                            "persistent_context_tokens": 5,
                            "incremental_context_tokens": 13,
                            "logical_communication_tokens": 13,
                            "system_prompt_tokens": 6,
                            "prompt_wrapper_tokens": 66,
                            "reusable_prefix_tokens": 11,
                            "reusable_prefix_key": "solver:normal:numeric_solve",
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
    assert totals["reasoning_round_count"] == 2
    assert totals["solver_revision_round_count"] == 1
    assert totals["physical_llm_input_tokens"] == 180
    assert totals["prefill_cost_tokens"] == 180
    assert totals["decode_cost_tokens"] == 14
    assert totals["persistent_context_tokens"] == 9
    assert totals["incremental_context_tokens"] == 21
    assert totals["logical_communication_tokens"] == 21
    assert totals["unique_prefix_tokens"] == 10
    assert totals["repeated_prefix_tokens"] == 11
    assert totals["simulated_context_reuse_input_tokens"] == 169
    assert totals["context_reuse_saving_ratio"] == 11 / 180
    assert totals["logical_vs_physical_input_gap_tokens"] == 159


def test_communication_totals_aggregate_phase7_metrics():
    records = [
        {
            "runtime_summary": {
                "records": [
                    {
                        "communication": [
                            {
                                "sender": "solver",
                                "receiver": "critic",
                                "candidate_tokens": 20,
                                "sent_tokens": 10,
                                "sent_count": 2,
                                "total_comm_tokens": 10,
                                "core_comm_tokens": 3,
                                "delta_comm_tokens": 2,
                                "verification_comm_tokens": 1,
                                "quality_comm_tokens": 1,
                                "control_comm_tokens": 3,
                                "unique_comm_tokens": 7,
                                "repeated_comm_tokens": 3,
                                "receiver_seen_hit_count": 2,
                                "state_delta_tokens": 10,
                                "full_state_equivalent_tokens": 20,
                                "revision_success_count": 1,
                                "feedback_sent_tokens": 5,
                                "feedback_newly_rendered_tokens": 4,
                                "communication_token_accounting_ok": True,
                                "communication_token_breakdown_ok": True,
                                "feedback_render_accounting_ok": True,
                                "unique_repeated_accounting_ok": True,
                            }
                        ],
                    }
                ],
            },
        }
    ]

    totals = communication_totals(records)
    pair = totals["graph_delta_by_pair"]["solver->critic"]

    assert totals["communication_round_count"] == 1
    assert totals["total_comm_tokens"] == 10
    assert totals["duplicate_ratio"] == 0.3
    assert totals["incremental_saving"] == 0.5
    assert totals["communication_token_breakdown_ok"] is True
    assert totals["feedback_render_accounting_ok"] is True
    assert totals["unique_repeated_accounting_ok"] is True
    assert totals["communication_cost_tokens"] == 10
    assert totals["communication_token_accounting_errors"] == 0
    assert pair["communication_round_count"] == 1
    assert pair["duplicate_ratio"] == 0.3
    assert pair["incremental_saving"] == 0.5
    assert pair["communication_cost_tokens"] == 10


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
