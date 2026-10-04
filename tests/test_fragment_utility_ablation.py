from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fragment_utility_ablation import (
    build_fragment_utility_examples,
    build_utility_tables,
    evaluate_policy_baselines,
    summarize_ablation,
)
from workflow_runtime import AgentGraphViewManager, GraphStore
from workflow_runtime.communication import make_communication_policy
from workflow_runtime.langgraph_workflow import LangGraphWorkflow


def _store_and_workflow(policy_name: str) -> tuple[GraphStore, LangGraphWorkflow]:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="solve", owner="user")
    workflow = LangGraphWorkflow(
        store=store,
        model=lambda request: {},
        task_id="t",
        communication_policy=make_communication_policy(policy_name),
    )
    return store, workflow


def test_fragment_ablation_policy_selects_only_named_optional_fragment() -> None:
    store, workflow = _store_and_workflow("fragment_ablation_full_plan")
    plan = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="plan",
        node_type="plan",
        content={"steps": [{"id": "R1", "operation": "add"}], "rationale": "because"},
        owner="planner",
        created_by_role="planner",
    )
    workflow.agent_views.grant("planner", node_ids=[plan.node_id], local=True)

    event = workflow._communicate_nodes(sender="planner", node_ids=[plan.node_id], branch_id="main")[0]

    assert event["policy"] == "fragment_ablation_full_plan"
    assert f"{plan.node_id}#full_plan" in event["selected_optional_fragment_ids"]
    assert all("rationale" not in item for item in event["selected_optional_fragment_ids"])
    assert f"{plan.node_id}#full_plan" in event["sent_fragment_ids"]


def test_core_only_fragment_ablation_sends_no_optional_fragments() -> None:
    store, workflow = _store_and_workflow("fragment_ablation_core_only")
    plan = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="plan",
        node_type="plan",
        content={"steps": [{"id": "R1", "operation": "add"}], "rationale": "because"},
        owner="planner",
        created_by_role="planner",
    )
    workflow.agent_views.grant("planner", node_ids=[plan.node_id], local=True)

    event = workflow._communicate_nodes(sender="planner", node_ids=[plan.node_id], branch_id="main")[0]

    assert event["selected_optional_fragment_ids"] == []
    assert all("#full_plan" not in item for item in event["sent_fragment_ids"])


def test_fragment_utility_summary_counts_sample_level_gain_and_oracle() -> None:
    reports = {
        "core_only": {
            "accuracy_or_pass_at_1": 0.5,
            "total_input_tokens": 100,
            "total_model_tokens": 120,
            "graph_delta_sent_tokens": 10,
            "records": [
                {"sample_id": "s1", "correct": False},
                {"sample_id": "s2", "correct": True},
                {"sample_id": "s3", "correct": True},
            ],
        },
        "full_plan": {
            "accuracy_or_pass_at_1": 2 / 3,
            "total_input_tokens": 130,
            "total_model_tokens": 155,
            "graph_delta_sent_tokens": 25,
            "records": [
                {"sample_id": "s1", "correct": True},
                {"sample_id": "s2", "correct": True},
                {"sample_id": "s3", "correct": False},
            ],
        },
        "send_all": {
            "accuracy_or_pass_at_1": 1.0,
            "total_input_tokens": 150,
            "total_model_tokens": 180,
            "graph_delta_sent_tokens": 40,
            "records": [
                {"sample_id": "s1", "correct": True},
                {"sample_id": "s2", "correct": True},
                {"sample_id": "s3", "correct": True},
            ],
        },
    }

    summary = summarize_ablation(reports, domain="humaneval", seed=7)

    assert summary["arms"]["full_plan"]["positive"] == 1
    assert summary["arms"]["full_plan"]["negative"] == 1
    assert summary["arms"]["full_plan"]["neutral"] == 1
    assert summary["arms"]["full_plan"]["net_gain"] == 0
    assert summary["oracle_correct"] == 3
    assert summary["oracle_gain"] == 1
    assert summary["oracle_targeted_gt_core_only"] is True
    assert summary["send_all_correct_core_wrong_sample_ids"] == ["s1"]
    assert set(summary["policy_baselines"]) >= {
        "core_only",
        "random_same_budget",
        "static_utility",
        "receiver_aware_heuristic",
        "oracle",
        "send_all",
    }


def test_oracle_targeted_excludes_send_all_only_rescues() -> None:
    reports = {
        "core_only": {
            "records": [
                {"sample_id": "s1", "correct": False},
                {"sample_id": "s2", "correct": True},
            ],
        },
        "full_feedback": {
            "records": [
                {"sample_id": "s1", "correct": False},
                {"sample_id": "s2", "correct": True},
            ],
        },
        "send_all": {
            "records": [
                {"sample_id": "s1", "correct": True},
                {"sample_id": "s2", "correct": True},
            ],
        },
    }

    summary = summarize_ablation(reports, domain="humaneval")

    assert summary["core_correct"] == 1
    assert summary["oracle_targeted_correct"] == 1
    assert summary["oracle_any_correct"] == 2
    assert summary["oracle_targeted_gt_core_only"] is False
    assert summary["send_all_only_rescue_sample_ids"] == ["s1"]


def test_fragment_utility_examples_are_supervised_selector_labels() -> None:
    reports = {
        "core_only": {
            "total_input_tokens": 10,
            "total_model_tokens": 12,
            "graph_delta_sent_tokens": 3,
            "records": [
                {"sample_id": "s1", "correct": False},
                {"sample_id": "s2", "correct": True},
            ],
        },
        "calculation_trace": {
            "total_input_tokens": 20,
            "total_model_tokens": 25,
            "graph_delta_sent_tokens": 8,
            "records": [
                {"sample_id": "s1", "correct": True},
                {"sample_id": "s2", "correct": False},
            ],
        },
    }

    examples = build_fragment_utility_examples(reports, domain="gsm8k")

    assert examples == [
        {
            "sample_id": "s1",
            "domain": "gsm8k",
            "receiver": "none",
            "receivers": [],
            "fragment_arm": "calculation_trace",
            "core_correct": False,
            "fragment_correct": True,
            "utility": 1,
            "label": "send",
            "harmful": False,
            "input_token_delta_vs_core": 10,
            "model_token_delta_vs_core": 13,
            "graph_delta_sent_token_delta_vs_core": 5,
            "utility_per_input_token": 0.1,
            "utility_per_model_token": 1 / 13,
            "utility_per_graph_sent_token": 0.2,
        },
        {
            "sample_id": "s2",
            "domain": "gsm8k",
            "receiver": "none",
            "receivers": [],
            "fragment_arm": "calculation_trace",
            "core_correct": True,
            "fragment_correct": False,
            "utility": -1,
            "label": "drop",
            "harmful": True,
            "input_token_delta_vs_core": 10,
            "model_token_delta_vs_core": 13,
            "graph_delta_sent_token_delta_vs_core": 5,
            "utility_per_input_token": -0.1,
            "utility_per_model_token": -1 / 13,
            "utility_per_graph_sent_token": -0.2,
        },
    ]


def test_utility_tables_are_conditioned_on_fragment_receiver_and_domain() -> None:
    examples = [
        {"fragment_arm": "full_plan", "receiver": "solver", "domain": "humaneval", "utility": 1},
        {"fragment_arm": "full_plan", "receiver": "solver", "domain": "gsm8k", "utility": -1},
        {"fragment_arm": "full_feedback", "receiver": "final_solver", "domain": "humaneval", "utility": 0},
    ]

    tables = build_utility_tables(examples)

    assert tables["global"]["full_plan"]["mean_utility"] == 0.0
    assert tables["conditioned"]["full_plan|solver|humaneval"]["mean_utility"] == 1.0
    assert tables["conditioned"]["full_plan|solver|gsm8k"]["mean_utility"] == -1.0


def test_receiver_aware_heuristic_and_random_same_budget_are_reported() -> None:
    reports = {
        "core_only": {
            "records": [
                {"sample_id": "s1", "correct": False},
                {"sample_id": "s2", "correct": False},
            ],
        },
        "full_plan": {
            "records": [
                {"sample_id": "s1", "correct": True},
                {"sample_id": "s2", "correct": True},
            ],
        },
        "full_feedback": {
            "records": [
                {"sample_id": "s1", "correct": False},
                {"sample_id": "s2", "correct": False},
            ],
        },
        "send_all": {
            "records": [
                {"sample_id": "s1", "correct": True},
                {"sample_id": "s2", "correct": True},
            ],
        },
    }
    examples = build_fragment_utility_examples(reports, domain="humaneval")
    tables = build_utility_tables(examples)

    baselines = evaluate_policy_baselines(reports, utility_tables=tables, domain="humaneval", seed=0)

    assert baselines["static_utility"]["selected_arm"] == "full_plan"
    assert baselines["receiver_aware_heuristic"]["selected_arm"] == "full_plan"
    assert baselines["receiver_aware_heuristic"]["accuracy"] == 1.0
    assert baselines["random_same_budget"]["selected_arm"] == "random_same_budget"
    assert baselines["oracle"]["accuracy"] == 1.0
