from workflow_runtime.candidate_graph import (
    add_verifier_guided_selection,
    coverage_at_k,
    dereference_selected_candidate_value,
    materialize_candidate_pool,
    oracle_select_candidate_id,
    safe_id,
    select_candidate_by_deployable_verification,
    select_candidate_by_verification,
)
from workflow_runtime.graph_store import GraphStore


def _task_store():
    store = GraphStore()
    task = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content={"question": "2+2?"},
        owner="dataset",
    )
    return store, task


def test_materialize_candidate_pool_creates_auditable_candidate_graph():
    store, task = _task_store()

    result = materialize_candidate_pool(
        store,
        task_id="t",
        branch_id="main",
        group_id="sample-1",
        domain="gsm8k",
        source_node_id=task.node_id,
        selector_policy="oracle_correct_if_available",
        selected_candidate_id="c1",
        candidates=[
            {
                "candidate_id": "c0",
                "artifact_type": "result",
                "artifact": "5",
                "final_value": "5",
                "correct": False,
                "score": {"correct": False},
                "seed": 1,
            },
            {
                "candidate_id": "c1",
                "artifact_type": "result",
                "artifact": "4",
                "final_value": "4",
                "correct": True,
                "score": {"correct": True},
                "seed": 2,
            },
        ],
    )

    state = store.snapshot()
    type_counts = {}
    for node in state.nodes.values():
        type_counts[node.type] = type_counts.get(node.type, 0) + 1

    assert result.invariant_errors == []
    assert type_counts["candidate_group"] == 1
    assert type_counts["candidate"] == 2
    assert type_counts["candidate_artifact"] == 2
    assert type_counts["candidate_verification"] == 2
    assert type_counts["selection_decision"] == 1
    assert result.selected_candidate_id == "c1"
    assert result.selected_artifact_node_id == result.artifact_node_ids["c1"]
    assert result.selected_verification_node_id == result.verification_node_ids["c1"]

    correct_artifact = state.nodes[result.artifact_node_ids["c1"]]
    wrong_artifact = state.nodes[result.artifact_node_ids["c0"]]
    assert correct_artifact.validation["model_judged_correct"] is True
    assert "model_judged_correct" not in wrong_artifact.validation

    edge_triples = {(edge.source, edge.relation, edge.target) for edge in state.edges}
    assert (
        result.verification_node_ids["c1"],
        "verifies",
        result.artifact_node_ids["c1"],
    ) in edge_triples
    assert (
        result.verification_node_ids["c0"],
        "invalidates",
        result.artifact_node_ids["c0"],
    ) in edge_triples

    selection = state.nodes[result.selection_node_id]
    assert selection.content["selected_candidate_id"] == "c1"
    assert selection.content["selected_artifact_node_id"] == result.artifact_node_ids["c1"]


def test_candidate_identity_is_local_and_collision_safe():
    store, task = _task_store()

    result = materialize_candidate_pool(
        store,
        task_id="t",
        branch_id="main",
        group_id="sample 2",
        domain="mbpp",
        source_node_id=task.node_id,
        selected_candidate_id="Candidate A",
        candidates=[
            {"candidate_id": "Candidate A", "artifact_type": "code", "artifact": "def f(): pass"},
            {"candidate_id": "Candidate A", "artifact_type": "code", "artifact": "def f(): return 1"},
        ],
    )

    assert safe_id("Candidate A") == "candidate_a"
    assert set(result.candidate_node_ids) == {"Candidate A", "Candidate A_2"}
    assert result.selected_candidate_id == "Candidate A"
    assert len(set(result.candidate_node_ids.values())) == 2
    assert result.invariant_errors == []


def test_phase1_gate_selected_candidate_dereferences_without_regeneration():
    store, task = _task_store()
    result = materialize_candidate_pool(
        store,
        task_id="t",
        branch_id="main",
        group_id="manual-gate",
        domain="gsm8k",
        source_node_id=task.node_id,
        selector_policy="manual_select_c2",
        selected_candidate_id="c2",
        candidates=[
            {
                "candidate_id": "c1",
                "artifact_type": "result",
                "artifact": "42",
                "final_value": "42",
                "correct": False,
                "score": {"correct": False},
            },
            {
                "candidate_id": "c2",
                "artifact_type": "result",
                "artifact": "48",
                "final_value": "48",
                "correct": True,
                "score": {"correct": True},
            },
        ],
    )

    state = store.snapshot()
    final_answer = dereference_selected_candidate_value(store, result.selection_node_id)

    assert result.as_dict()["candidate_count"] == 2
    assert result.invariant_errors == []
    assert {node.branch_id for node in state.nodes.values()} == {"main"}
    assert result.selected_candidate_id == "c2"
    assert final_answer == "48"

    selected_artifact = state.nodes[result.selected_artifact_node_id]
    selection = state.nodes[result.selection_node_id]
    assert selected_artifact.content["final_value"] == "48"
    assert selection.content["selected_artifact_node_id"] == result.artifact_node_ids["c2"]
    assert selection.content.get("value") is None
    assert selection.content.get("final_answer") is None


def test_phase2_verifier_guided_selector_uses_bound_verification_nodes():
    store, task = _task_store()
    pool = materialize_candidate_pool(
        store,
        task_id="t",
        branch_id="main",
        group_id="phase2-gate",
        domain="gsm8k",
        source_node_id=task.node_id,
        candidates=[
            {
                "candidate_id": "c1",
                "artifact_type": "result",
                "artifact": "42",
                "final_value": "42",
                "correct": False,
                "score": {"correct": False},
            },
            {
                "candidate_id": "c2",
                "artifact_type": "result",
                "artifact": "48",
                "final_value": "48",
                "correct": True,
                "score": {"correct": True},
            },
        ],
    )

    decision = select_candidate_by_verification(store, pool)
    selected = add_verifier_guided_selection(store, pool)
    state = store.snapshot()

    assert decision.selected_candidate_id == "c2"
    assert decision.verified_candidate_ids == ("c2",)
    assert decision.rejected_candidate_ids == ("c1",)
    assert selected.selected_candidate_id == "c2"
    assert dereference_selected_candidate_value(store, selected.selection_node_id) == "48"
    assert {node.branch_id for node in state.nodes.values()} == {"main"}

    edge_triples = {(edge.source, edge.relation, edge.target) for edge in state.edges}
    assert (
        selected.selected_verification_node_id,
        "input_to",
        selected.selection_node_id,
    ) in edge_triples
    selection = state.nodes[selected.selection_node_id]
    assert selection.content["metadata"]["source"] == "candidate_verification_nodes"
    assert selection.content["selection_reason"] == "verified_candidate"


def test_coverage_definition_any_correct_candidate_covers_prefix():
    candidates = [
        {"candidate_id": "c1", "correct": False},
        {"candidate_id": "c2", "correct": False},
        {"candidate_id": "c3", "correct": True},
    ]

    assert coverage_at_k(candidates, (1, 2, 4, 8, 16)) == {
        "coverage@1": 0.0,
        "coverage@2": 0.0,
        "coverage@4": 1.0,
        "coverage@8": 1.0,
        "coverage@16": 1.0,
    }


def test_oracle_select_correct_when_available_equals_coverage():
    candidates = [
        {"candidate_id": "c1", "correct": False},
        {"candidate_id": "c2", "correct": True},
        {"candidate_id": "c3", "correct": False},
    ]

    selected = oracle_select_candidate_id(candidates)

    assert selected == "c2"
    assert bool(next(c for c in candidates if c["candidate_id"] == selected)["correct"])
    assert coverage_at_k(candidates, (3,))["coverage@3"] == 1.0


def test_oracle_fallback_when_none_correct_is_deterministic():
    candidates = [
        {"candidate_id": "c1", "correct": False},
        {"candidate_id": "c2", "correct": False},
    ]

    assert oracle_select_candidate_id(candidates) == "c1"
    assert oracle_select_candidate_id(candidates) == "c1"
    assert coverage_at_k(candidates, (2,))["coverage@2"] == 0.0


def test_oracle_no_mutation():
    candidates = [
        {"candidate_id": "c1", "correct": False, "score": {"correct": False}},
        {"candidate_id": "c2", "correct": True, "score": {"correct": True}},
    ]
    before = [dict(candidate, score=dict(candidate["score"])) for candidate in candidates]

    assert oracle_select_candidate_id(candidates) == "c2"

    assert candidates == before


def test_oracle_selection_traceable():
    store, task = _task_store()
    pool = materialize_candidate_pool(
        store,
        task_id="t",
        branch_id="main",
        group_id="oracle-trace",
        domain="gsm8k",
        source_node_id=task.node_id,
        selector_policy="oracle_select",
        selected_candidate_id="c2",
        candidates=[
            {"candidate_id": "c1", "artifact_type": "result", "artifact": "42", "correct": False},
            {"candidate_id": "c2", "artifact_type": "result", "artifact": "48", "correct": True},
        ],
    )

    selection = store.snapshot().nodes[pool.selection_node_id]

    assert selection.content["selected_candidate_id"] == "c2"
    assert selection.content["selected_candidate_node_id"] == pool.candidate_node_ids["c2"]
    assert selection.content["selected_artifact_node_id"] == pool.artifact_node_ids["c2"]
    assert selection.content["selected_verification_node_id"] == pool.verification_node_ids["c2"]


def test_oracle_select_at_n_equals_coverage_at_n_with_gold_evaluator():
    candidates = [
        {"candidate_id": "c1", "correct": False},
        {"candidate_id": "c2", "correct": False},
        {"candidate_id": "c3", "correct": True},
        {"candidate_id": "c4", "correct": False},
    ]
    selected = oracle_select_candidate_id(candidates)

    assert selected == "c3"
    assert coverage_at_k(candidates, (4,))["coverage@4"] == 1.0
    assert bool(next(c for c in candidates if c["candidate_id"] == selected)["correct"]) == bool(
        coverage_at_k(candidates, (4,))["coverage@4"]
    )


def test_oracle_uses_gold_only_in_oracle_path_deployable_rejects_leakage():
    store, task = _task_store()
    pool = materialize_candidate_pool(
        store,
        task_id="t",
        branch_id="main",
        group_id="leakage-gate",
        domain="gsm8k",
        source_node_id=task.node_id,
        candidates=[
            {
                "candidate_id": "c1",
                "artifact_type": "result",
                "artifact": "42",
                "correct": True,
                "verification_source": "oracle_evaluator",
                "score": {
                    "correct": True,
                    "gold_answer": "42",
                    "hidden_test_result": "pass",
                },
            },
        ],
    )

    try:
        select_candidate_by_deployable_verification(store, pool)
    except ValueError as exc:
        assert "deployable" in str(exc)
    else:
        raise AssertionError("deployable selector accessed oracle-only verifier data")


def test_deployable_selector_can_use_non_oracle_verification_payload():
    store, task = _task_store()
    pool = materialize_candidate_pool(
        store,
        task_id="t",
        branch_id="main",
        group_id="deployable-ok",
        domain="gsm8k",
        source_node_id=task.node_id,
        candidates=[
            {
                "candidate_id": "c1",
                "artifact_type": "result",
                "artifact": "42",
                "correct": True,
                "verification_source": "public_unit_test",
                "score": {"passed_public_check": True},
            },
        ],
    )

    decision = select_candidate_by_deployable_verification(store, pool)

    assert decision.selected_candidate_id == "c1"
    assert decision.reason == "deployable_verified_candidate"
