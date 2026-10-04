from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_runtime import GraphStore
from workflow_runtime.action_compiler import ActionCompiler
from workflow_runtime.agent_graph_view import AgentGraphViewManager
from workflow_runtime.communication import ClosureAwareHeuristicPolicy, MinimalNoFeedbackPolicy, MinimalSendAllFallbackPolicy, MinimalTargetedFeedbackPolicy, RandomKeepPolicy, SendAllPolicy, make_communication_policy
from workflow_runtime.delta_closure import dependency_closure
from workflow_runtime.delta_extractor import extract_delta_candidates


def _numeric_graph():
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="x=5 y=3", owner="user")
    compiler = ActionCompiler(store, task_id="t", task_type="numeric_solve")
    compiler.apply("planner", {"op": "declare_query", "content": {"question": "sum"}})
    compiler.apply("planner", {"op": "add_fact", "id": "A", "value": 5})
    compiler.apply("planner", {"op": "add_fact", "id": "B", "value": 3})
    compiler.apply("planner", {"op": "add_plan_step", "id": "R1", "operation": "add", "inputs": ["A", "B"]})
    compiler.apply("solver", {"op": "calculate", "id": "R1", "expression": "5+3", "value": 8})
    compiler.apply("solver", {"op": "set_result", "id": "R1", "value": 8})
    return store


def test_receiver_relative_closure_adds_missing_dependencies_only():
    store = _numeric_graph()
    views = AgentGraphViewManager(store, task_id="t", branch_id="main", agents=("solver", "critic"))
    state = store.snapshot()
    result = store.latest_valid("t", "main", "result")
    facts = store.latest_valid("t", "main", "facts")
    assert result is not None and facts is not None

    # If the receiver already knows facts, closure should not retransmit them.
    views.grant("critic", node_ids=[facts.node_id])
    delta = dependency_closure(
        state,
        root_node_ids=[result.node_id],
        sender="solver",
        receiver="critic",
        receiver_view=views.view("critic"),
        policy="test",
    )
    assert result.node_id in delta.node_ids
    assert facts.node_id not in delta.node_ids

    # Without facts, the same selected root becomes a dependency-closed message.
    views = AgentGraphViewManager(store, task_id="t", branch_id="main", agents=("solver", "critic"))
    delta = dependency_closure(
        state,
        root_node_ids=[result.node_id],
        sender="solver",
        receiver="critic",
        receiver_view=views.view("critic"),
        policy="test",
    )
    assert result.node_id in delta.node_ids
    assert facts.node_id in delta.node_ids
    assert delta.closure_added_node_ids


def test_closure_aware_policy_skips_semantically_known_candidates():
    store = _numeric_graph()
    views = AgentGraphViewManager(store, task_id="t", branch_id="main", agents=("planner", "solver"))
    state = store.snapshot()
    fact_a = store.latest_valid("t", "main", "fact_A")
    assert fact_a is not None
    views.grant("solver", node_ids=[fact_a.node_id])

    candidates = extract_delta_candidates(
        state,
        node_ids=[fact_a.node_id],
        sender="planner",
        receiver="solver",
        receiver_view=views.view("solver"),
    )
    assert candidates[0].novelty_score == 0.0
    assert ClosureAwareHeuristicPolicy().select_roots(
        state=state,
        sender_view=views.view("planner"),
        receiver_view=views.view("solver"),
        candidates=candidates,
    ) == []


def test_query_spec_is_mandatory_baseline_not_delta_candidate():
    store = _numeric_graph()
    views = AgentGraphViewManager(store, task_id="t", branch_id="main", agents=("planner", "solver", "critic"))
    query = store.latest_valid("t", "main", "query_spec")
    assert query is not None
    views.grant_global_visibility()

    assert query.node_id in views.view("critic").visible_node_ids
    candidates = extract_delta_candidates(
        store.snapshot(),
        node_ids=[query.node_id],
        sender="planner",
        receiver="critic",
        receiver_view=views.view("critic"),
    )
    assert candidates == []


def test_policy_factory_selects_supported_policies():
    assert make_communication_policy("send_all").name == SendAllPolicy.name
    assert make_communication_policy("minimal_no_feedback").name == MinimalNoFeedbackPolicy.name
    assert make_communication_policy("minimal_sendall_fallback").name == MinimalSendAllFallbackPolicy.name
    assert make_communication_policy("minimal_targeted_feedback").name == MinimalTargetedFeedbackPolicy.name
    assert make_communication_policy("random_keep_75").name == "random_keep_75"
    assert make_communication_policy("random_keep_50").name == "random_keep_50"
    assert make_communication_policy("random_keep_25").name == "random_keep_25"
    assert make_communication_policy("closure_aware_heuristic").name == "closure_aware_heuristic"
    assert make_communication_policy("random_same_budget").name == "random_same_budget"
    assert make_communication_policy("static_utility").name == "static_utility"
    assert make_communication_policy("receiver_aware_heuristic").name == "receiver_aware_heuristic"
    for removed in ("novelty_only", "novelty_closure", "budgeted_graph_heuristic", "random_keep_100"):
        with pytest.raises(ValueError):
            make_communication_policy(removed)


def test_random_keep_policy_is_seed_deterministic():
    store = _numeric_graph()
    views = AgentGraphViewManager(store, task_id="t", branch_id="main", agents=("planner", "solver"))
    state = store.snapshot()
    fact_a = store.latest_valid("t", "main", "fact_A")
    fact_b = store.latest_valid("t", "main", "fact_B")
    assert fact_a is not None and fact_b is not None
    candidates = extract_delta_candidates(
        state,
        node_ids=[fact_a.node_id, fact_b.node_id],
        sender="planner",
        receiver="solver",
        receiver_view=views.view("solver"),
    )
    left = RandomKeepPolicy(ratio=0.5, seed=7).select_roots(
        state=state,
        sender_view=views.view("planner"),
        receiver_view=views.view("solver"),
        candidates=candidates,
    )
    right = RandomKeepPolicy(ratio=0.5, seed=7).select_roots(
        state=state,
        sender_view=views.view("planner"),
        receiver_view=views.view("solver"),
        candidates=candidates,
    )
    assert left == right
