from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_runtime import AgentGraphViewManager, GraphStore
from workflow_runtime.communication import make_communication_policy
from workflow_runtime.langgraph_workflow import LangGraphWorkflow


def _workflow(policy_name: str, *, table: dict, task_family: str = "humaneval"):
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="solve", owner="user")
    workflow = LangGraphWorkflow(
        store=store,
        model=lambda request: {},
        task_id="t",
        task_type=task_family,
        communication_policy=make_communication_policy(
            policy_name,
            utility_table_path=None,
            task_family=task_family,
        ),
    )
    workflow.communication_policy.utility_table = workflow.communication_policy.utility_table.__class__(
        table,
        task_family=task_family,
    )
    return store, workflow


def test_policy_factory_creates_utility_policies() -> None:
    assert make_communication_policy("static_utility").name == "static_utility"
    assert make_communication_policy("receiver_aware_heuristic").name == "receiver_aware_heuristic"
    assert make_communication_policy("random_same_budget").name == "random_same_budget"


def test_receiver_aware_heuristic_uses_conditioned_fragment_receiver_domain_utility() -> None:
    table = {
        "utility_tables": {
            "global": {
                "full_plan": {"mean_utility": -1.0},
                "result_metadata": {"mean_utility": 0.1},
            },
            "conditioned": {
                "full_plan|solver|humaneval": {"mean_utility": 1.0},
                "result_metadata|solver|humaneval": {"mean_utility": -1.0},
            },
        }
    }
    store, workflow = _workflow("receiver_aware_heuristic", table=table, task_family="humaneval")
    plan = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="plan",
        node_type="plan",
        content={"steps": [{"id": "R1", "operation": "write_code"}]},
        owner="planner",
        created_by_role="planner",
    )
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"value": "x", "explanation": "metadata"},
        owner="planner",
        created_by_role="planner",
    )
    workflow.agent_views.grant("planner", node_ids=[plan.node_id, result.node_id], local=True)

    event = workflow._communicate_nodes(sender="planner", node_ids=[plan.node_id, result.node_id], branch_id="main")[0]

    assert f"{plan.node_id}#full_plan" in event["selected_optional_fragment_ids"]
    assert f"{result.node_id}#result_metadata" not in event["selected_optional_fragment_ids"]


def test_static_utility_uses_global_fragment_utility() -> None:
    table = {
        "utility_tables": {
            "global": {
                "full_plan": {"mean_utility": -1.0},
                "result_metadata": {"mean_utility": 1.0},
            },
            "conditioned": {
                "full_plan|solver|humaneval": {"mean_utility": 1.0},
                "result_metadata|solver|humaneval": {"mean_utility": -1.0},
            },
        }
    }
    store, workflow = _workflow("static_utility", table=table, task_family="humaneval")
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"value": "x", "explanation": "metadata"},
        owner="planner",
        created_by_role="solver",
    )
    workflow.agent_views.grant("solver", node_ids=[result.node_id], local=True)

    event = workflow._communicate_nodes(sender="solver", node_ids=[result.node_id], branch_id="main")[0]

    assert f"{result.node_id}#result_metadata" in event["selected_optional_fragment_ids"]


def test_random_same_budget_never_exceeds_receiver_aware_sent_fragment_budget() -> None:
    table = {
        "utility_tables": {
            "global": {
                "full_plan": {"mean_utility": 1.0},
                "result_metadata": {"mean_utility": 1.0},
            },
            "conditioned": {},
        }
    }
    store, workflow = _workflow("random_same_budget", table=table, task_family="humaneval")
    plan = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="plan",
        node_type="plan",
        content={"steps": [{"id": "R1", "operation": "write_code"}]},
        owner="planner",
        created_by_role="planner",
    )
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"value": "x", "explanation": "metadata"},
        owner="planner",
        created_by_role="planner",
    )
    workflow.agent_views.grant("planner", node_ids=[plan.node_id, result.node_id], local=True)

    event = workflow._communicate_nodes(sender="planner", node_ids=[plan.node_id, result.node_id], branch_id="main")[0]

    selected_tokens = sum(
        event["fragment_token_by_id"][fragment_id]
        for fragment_id in event["selected_optional_fragment_ids"]
    )
    positive_budget = sum(
        event["fragment_token_by_id"][fragment_id]
        for fragment_id in (f"{plan.node_id}#full_plan", f"{result.node_id}#result_metadata")
        if fragment_id in event["fragment_token_by_id"]
    )
    assert selected_tokens <= positive_budget
