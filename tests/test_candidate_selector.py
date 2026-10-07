from workflow_runtime.candidate_graph import materialize_candidate_pool
from workflow_runtime.candidate_selector import (
    assert_no_oracle_features,
    build_deployable_selector_examples,
    build_oracle_selector_examples,
    select_candidate_with_deployable_verifier,
    score_candidate_features,
    select_candidate_by_feature_score,
)
from workflow_runtime.graph_store import GraphStore


def _store_with_task():
    store = GraphStore()
    task = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="select",
        owner="dataset",
    )
    return store, task


def test_oracle_labels_are_not_deployable_features():
    store, task = _store_with_task()
    pool = materialize_candidate_pool(
        store,
        task_id="t",
        branch_id="main",
        group_id="selector-train",
        domain="gsm8k",
        source_node_id=task.node_id,
        candidates=[
            {
                "candidate_id": "c1",
                "artifact_type": "result",
                "artifact": "42",
                "final_value": "42",
                "correct": False,
                "score": {"correct": False, "gold_answer": "48"},
            },
            {
                "candidate_id": "c2",
                "artifact_type": "result",
                "artifact": "48",
                "final_value": "48",
                "correct": True,
                "score": {"correct": True, "gold_answer": "48"},
            },
        ],
    )

    examples = build_oracle_selector_examples(store, pool)

    assert [example.label for example in examples] == [0, 1]
    for example in examples:
        assert "correct" not in example.features
        assert "gold_answer" not in example.features
        assert "final_value" not in example.features
        assert "artifact" not in example.features


def test_deployable_examples_reject_oracle_source():
    store, task = _store_with_task()
    pool = materialize_candidate_pool(
        store,
        task_id="t",
        branch_id="main",
        group_id="selector-leak",
        domain="gsm8k",
        source_node_id=task.node_id,
        candidates=[
            {
                "candidate_id": "c1",
                "artifact_type": "result",
                "artifact": "42",
                "correct": True,
                "verification_source": "oracle_evaluator",
            },
        ],
    )

    try:
        build_deployable_selector_examples(store, pool)
    except ValueError as exc:
        assert "deployable selector cannot use verifier source" in str(exc)
    else:
        raise AssertionError("deployable example builder accepted oracle verifier source")


def test_assert_no_oracle_features_blocks_value_and_hidden_fields():
    for forbidden in ("correct", "gold_answer", "hidden_test_result", "final_value", "artifact"):
        try:
            assert_no_oracle_features({forbidden: "x"})
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted forbidden selector feature: {forbidden}")


def test_feature_score_prefers_public_pass_signal_over_failure_signal():
    passing = {
        "generation_index": 1,
        "artifact_type": "code",
        "public_tests_passed": True,
        "verifier_confidence": 0.7,
    }
    failing = {
        "generation_index": 0,
        "artifact_type": "code",
        "assertion_failure": True,
        "verifier_confidence": 0.9,
    }

    assert score_candidate_features(passing) > score_candidate_features(failing)


def test_deployable_feature_selector_is_deterministic_and_traceable():
    store, task = _store_with_task()
    pool = materialize_candidate_pool(
        store,
        task_id="t",
        branch_id="main",
        group_id="selector-deployable",
        domain="mbpp",
        source_node_id=task.node_id,
        candidates=[
            {
                "candidate_id": "c1",
                "artifact_type": "code",
                "artifact": "def f(): return 1",
                "correct": False,
                "verification_source": "public_unit_test",
                "score": {"public_tests_passed": False, "assertion_failure": True, "confidence": 0.9},
            },
            {
                "candidate_id": "c2",
                "artifact_type": "code",
                "artifact": "def f(): return 2",
                "correct": True,
                "verification_source": "public_unit_test",
                "score": {"public_tests_passed": True, "confidence": 0.7},
            },
        ],
    )

    selected_a, trace_a = select_candidate_by_feature_score(store, pool)
    selected_b, trace_b = select_candidate_by_feature_score(store, pool)

    assert selected_a == "c2"
    assert selected_b == "c2"
    assert trace_a == trace_b
    assert trace_a[0]["candidate_id"] == "c2"
    assert "feature_score" in trace_a[0]
    assert trace_a[0]["label"] is None


def test_selector_output_has_required_fields():
    store, task = _store_with_task()
    pool = materialize_candidate_pool(
        store,
        task_id="t",
        branch_id="main",
        group_id="selector-output",
        domain="gsm8k",
        source_node_id=task.node_id,
        candidates=[
            {
                "candidate_id": "c1",
                "artifact_type": "result",
                "artifact": "48",
                "correct": True,
                "verification_source": "public_math_check",
                "score": {
                    "final_value_valid": True,
                    "arithmetic_consistency": True,
                    "constraint_consistency": True,
                    "trace_result_agreement": True,
                    "independent_recompute_consistency": True,
                },
            }
        ],
    )

    result = select_candidate_with_deployable_verifier(store, pool).as_dict()

    assert result["selected_candidate_id"] == "c1"
    assert isinstance(result["selection_score"], float)
    assert result["selection_reason"]
    assert 0.0 <= result["confidence"] <= 1.0


def _select_from_candidates(candidates, *, domain="gsm8k"):
    store, task = _store_with_task()
    pool = materialize_candidate_pool(
        store,
        task_id="t",
        branch_id="main",
        group_id=f"{domain}-pool",
        domain=domain,
        source_node_id=task.node_id,
        candidates=candidates,
    )
    return select_candidate_with_deployable_verifier(store, pool)


def test_gsm8k_internal_consistency_wrong_equation_loses_to_constraint_consistent_candidate():
    result = _select_from_candidates([
        {
            "candidate_id": "c1",
            "artifact_type": "result",
            "artifact": "42",
            "correct": False,
            "verification_source": "public_math_check",
            "score": {
                "final_value_valid": True,
                "expression_parseable": True,
                "arithmetic_consistency": True,
                "trace_result_agreement": True,
                "constraint_consistency": False,
                "independent_recompute_consistency": False,
            },
        },
        {
            "candidate_id": "c2",
            "artifact_type": "result",
            "artifact": "48",
            "correct": True,
            "verification_source": "public_math_check",
            "score": {
                "final_value_valid": True,
                "expression_parseable": True,
                "arithmetic_consistency": True,
                "trace_result_agreement": True,
                "constraint_consistency": True,
                "independent_recompute_consistency": True,
            },
        },
    ])

    assert result.selected_candidate_id == "c2"
    assert "+constraint_consistency" in result.selection_reason


def test_gsm8k_trace_correct_but_final_value_copied_wrong_is_penalized():
    result = _select_from_candidates([
        {
            "candidate_id": "c1",
            "artifact_type": "result",
            "artifact": "42",
            "correct": False,
            "verification_source": "public_math_check",
            "score": {
                "final_value_valid": True,
                "expression_parseable": True,
                "arithmetic_consistency": True,
                "constraint_consistency": True,
                "trace_result_disagreement": True,
            },
        },
        {
            "candidate_id": "c2",
            "artifact_type": "result",
            "artifact": "48",
            "correct": True,
            "verification_source": "public_math_check",
            "score": {
                "final_value_valid": True,
                "expression_parseable": True,
                "arithmetic_consistency": True,
                "constraint_consistency": True,
                "trace_result_agreement": True,
            },
        },
    ])

    assert result.selected_candidate_id == "c2"


def test_gsm8k_wrong_majority_does_not_override_verified_minority():
    candidates = [
        {
            "candidate_id": f"wrong_{idx}",
            "artifact_type": "result",
            "artifact": "42",
            "correct": False,
            "verification_source": "public_math_check",
            "score": {
                "final_value_valid": True,
                "cross_candidate_answer_consensus": 0.75,
                "constraint_inconsistency": True,
            },
        }
        for idx in range(3)
    ]
    candidates.append(
        {
            "candidate_id": "right_minority",
            "artifact_type": "result",
            "artifact": "48",
            "correct": True,
            "verification_source": "public_math_check",
            "score": {
                "final_value_valid": True,
                "arithmetic_consistency": True,
                "constraint_consistency": True,
                "trace_result_agreement": True,
                "independent_recompute_consistency": True,
            },
        }
    )

    result = _select_from_candidates(candidates)

    assert result.selected_candidate_id == "right_minority"


def test_gsm8k_same_answer_prefers_better_reasoning_quality():
    result = _select_from_candidates([
        {
            "candidate_id": "weak_reasoning",
            "artifact_type": "result",
            "artifact": "48",
            "correct": True,
            "verification_source": "public_math_check",
            "score": {"final_value_valid": True},
        },
        {
            "candidate_id": "strong_reasoning",
            "artifact_type": "result",
            "artifact": "48",
            "correct": True,
            "verification_source": "public_math_check",
            "score": {
                "final_value_valid": True,
                "arithmetic_consistency": True,
                "constraint_consistency": True,
                "trace_result_agreement": True,
            },
        },
    ])

    assert result.selected_candidate_id == "strong_reasoning"


def test_code_selector_ranks_public_test_pass_over_shorter_wrong_candidate():
    result = _select_from_candidates([
        {
            "candidate_id": "short_wrong",
            "artifact_type": "code",
            "artifact": "def f(): return 1",
            "correct": False,
            "verification_source": "public_unit_test",
            "score": {
                "syntax_valid": True,
                "compile_success": True,
                "no_exception": True,
                "test_pass_rate": 0.25,
                "failed_test_count": 3,
            },
        },
        {
            "candidate_id": "passes_public",
            "artifact_type": "code",
            "artifact": "def f(): return 2",
            "correct": True,
            "verification_source": "public_unit_test",
            "score": {
                "syntax_valid": True,
                "compile_success": True,
                "public_tests_passed": True,
                "test_pass_rate": 1.0,
                "execution_diversity": 2,
            },
        },
    ], domain="mbpp")

    assert result.selected_candidate_id == "passes_public"


def test_code_selector_penalizes_runtime_exception_timeout_and_wrong_return_type():
    result = _select_from_candidates([
        {
            "candidate_id": "runtime_error",
            "artifact_type": "code",
            "artifact": "def f(): raise RuntimeError()",
            "correct": False,
            "verification_source": "public_unit_test",
            "score": {"syntax_valid": True, "runtime_exception": True, "exception_severity": 4},
        },
        {
            "candidate_id": "timeout",
            "artifact_type": "code",
            "artifact": "def f():\n while True: pass",
            "correct": False,
            "verification_source": "public_unit_test",
            "score": {"syntax_valid": True, "timeout": True, "exception_severity": 8},
        },
        {
            "candidate_id": "public_pass",
            "artifact_type": "code",
            "artifact": "def f(): return 2",
            "correct": True,
            "verification_source": "public_unit_test",
            "score": {"public_tests_passed": True, "test_pass_rate": 1.0},
        },
    ], domain="humaneval")

    assert result.selected_candidate_id == "public_pass"


def test_code_hidden_evaluator_signal_is_inaccessible_to_deployable_selector():
    store, task = _store_with_task()
    pool = materialize_candidate_pool(
        store,
        task_id="t",
        branch_id="main",
        group_id="hidden-code",
        domain="humaneval",
        source_node_id=task.node_id,
        candidates=[
            {
                "candidate_id": "overfit",
                "artifact_type": "code",
                "artifact": "def f(): return 2",
                "correct": False,
                "verification_source": "public_unit_test",
                "score": {
                    "public_tests_passed": True,
                    "hidden_test_result": "fail",
                },
            },
        ],
    )

    try:
        select_candidate_with_deployable_verifier(store, pool)
    except ValueError as exc:
        assert "oracle-only" in str(exc)
    else:
        raise AssertionError("deployable selector accessed hidden evaluator signal")


def test_code_selector_tiebreak_is_deterministic():
    candidates = [
        {
            "candidate_id": "c1",
            "artifact_type": "code",
            "artifact": "def f(): return 1",
            "correct": False,
            "verification_source": "public_unit_test",
            "score": {"syntax_valid": True},
        },
        {
            "candidate_id": "c2",
            "artifact_type": "code",
            "artifact": "def f(): return 2",
            "correct": False,
            "verification_source": "public_unit_test",
            "score": {"syntax_valid": True},
        },
    ]

    first = _select_from_candidates(candidates, domain="mbpp")
    second = _select_from_candidates(candidates, domain="mbpp")

    assert first.selected_candidate_id == second.selected_candidate_id == "c1"


def test_oracle_training_examples_do_not_mutate_candidate_graph():
    store, task = _store_with_task()
    pool = materialize_candidate_pool(
        store,
        task_id="t",
        branch_id="main",
        group_id="selector-no-mutation",
        domain="gsm8k",
        source_node_id=task.node_id,
        candidates=[
            {"candidate_id": "c1", "artifact_type": "result", "artifact": "42", "correct": False},
            {"candidate_id": "c2", "artifact_type": "result", "artifact": "48", "correct": True},
        ],
    )
    before = store.snapshot()

    examples = build_oracle_selector_examples(store, pool)

    after = store.snapshot()
    assert [example.label for example in examples] == [0, 1]
    assert before.nodes == after.nodes
    assert before.edges == after.edges
