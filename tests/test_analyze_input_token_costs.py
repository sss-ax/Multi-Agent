import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_input_token_costs import analyze


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def test_analyze_input_token_costs_splits_current_telemetry_fields(tmp_path: Path) -> None:
    telemetry = tmp_path / "workflow.jsonl"
    _write_jsonl(
        telemetry,
        [
            {
                "event": "model_action",
                "role": "solver",
                "mode": "normal",
                "action_index": 0,
                "physical_input_tokens": 100,
                "logical_input_tokens": 90,
                "system_prompt_tokens": 10,
                "persistent_context_tokens": 20,
                "incremental_context_tokens": 30,
                "prompt_wrapper_tokens": 15,
                "persistent_context_node_ids": ["task@v1", "fact@v1"],
            },
            {
                "event": "model_action",
                "role": "solver",
                "mode": "normal",
                "action_index": 1,
                "physical_input_tokens": 80,
                "logical_input_tokens": 75,
                "system_prompt_tokens": 10,
                "persistent_context_tokens": 20,
                "incremental_context_tokens": 25,
                "prompt_wrapper_tokens": 15,
                "persistent_context_node_ids": ["task@v1", "fact@v2"],
            },
            {
                "event": "graph_delta_communication",
                "sender": "solver",
                "receiver": "critic",
                "sent_tokens": 40,
                "repeated_comm_tokens": 12,
                "receiver_seen_hit_count": 2,
                "sent_node_ids": ["fact@v1", "result@v1"],
                "receiver_visible_before_node_ids": ["fact@v1"],
            },
        ],
    )

    report = analyze([telemetry])

    assert report["overall"]["calls"] == 2
    assert report["overall"]["categories"]["system"]["tokens"] == 20
    assert report["overall"]["categories"]["task"]["tokens"] == 40
    assert report["overall"]["categories"]["graph"]["tokens"] == 55
    assert report["overall"]["categories"]["protocol"]["tokens"] == 30
    assert report["by_agent"]["solver"]["physical_input_tokens"] == 180
    assert report["graph_duplicates"]["communication"]["repeated_comm_tokens"] == 12
    assert report["graph_duplicates"]["communication"]["repeated_sent_node_count"] == 1
    assert report["graph_duplicates"]["repeated_context_nodes"]["solver:task@v1"] == 1
    assert report["dominant_category"] == "graph"


def test_analyze_input_token_costs_reads_domain_workflow_report_json(tmp_path: Path) -> None:
    report_path = tmp_path / "gsm8k.json"
    report_path.write_text(
        json.dumps(
            {
                "domain": "gsm8k",
                "records": [
                    {
                        "runtime_summary": {
                            "records": [
                                {
                                    "role": "planner",
                                    "mode": "normal",
                                    "action_index": 0,
                                    "rendered_context_node_ids": ["task@v1"],
                                    "telemetry": {
                                        "physical_input_tokens": 50,
                                        "logical_input_tokens": 45,
                                        "system_prompt_tokens": 8,
                                        "persistent_context_tokens": 15,
                                        "incremental_context_tokens": 12,
                                        "prompt_wrapper_tokens": 5,
                                    },
                                    "communication": [
                                        {
                                            "sender": "planner",
                                            "receiver": "solver",
                                            "sent_tokens": 10,
                                            "repeated_comm_tokens": 3,
                                        }
                                    ],
                                }
                            ]
                        }
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = analyze([report_path])

    assert report["overall"]["calls"] == 1
    assert report["by_agent"]["planner"]["physical_input_tokens"] == 50
    assert report["overall"]["categories"]["task"]["tokens"] == 15
    assert report["graph_duplicates"]["communication"]["repeated_comm_tokens"] == 3
