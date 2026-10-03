from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pytest
except ImportError:  # pragma: no cover - lets this file run in minimal envs
    class _Raises:
        def __init__(self, expected):
            self.expected = expected

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                raise AssertionError(f"expected {self.expected.__name__} to be raised")
            return issubclass(exc_type, self.expected)

    class _PytestShim:
        @staticmethod
        def raises(expected):
            return _Raises(expected)

    pytest = _PytestShim()

from workflow_runtime import (
    GraphStore,
    InMemoryResultCache,
    RuntimeFingerprint,
    SliceError,
    build_context_slice,
    dependency_digest,
    node_content_digest,
)
from workflow_runtime.invalidation import invalidate_downstream
from main import init_workflow_graph


def runtime() -> RuntimeFingerprint:
    return RuntimeFingerprint(model_id="m", tokenizer_id="t", prompt_template_digest="p")


def test_digest_stability_and_dependency_changes() -> None:
    left = node_content_digest("facts", {"b": 2, "a": 1})
    right = node_content_digest("facts", {"a": 1, "b": 2})
    assert left == right
    assert left != node_content_digest("facts", {"a": 1, "b": 3})

    assert dependency_digest({"facts": "facts@v1"}) != dependency_digest({"facts": "facts@v2"})


def test_dependency_closure_context_slice() -> None:
    store = GraphStore()
    task = store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="q", owner="user")
    query = store.add_node(task_id="t", branch_id="main", logical_id="query_spec", node_type="query_spec", content="q", owner="planner")
    facts = store.add_node(task_id="t", branch_id="main", logical_id="facts", node_type="facts", content="f", owner="planner")
    plan = store.add_node(task_id="t", branch_id="main", logical_id="plan", node_type="plan", content="p", owner="planner")
    steps = store.add_node(task_id="t", branch_id="main", logical_id="plan_steps", node_type="plan_steps", content="[]", owner="planner")
    store.add_edge(source=task.node_id, relation="derived_from", target=query.node_id)
    store.add_edge(source=task.node_id, relation="derived_from", target=facts.node_id)
    store.add_edge(source=facts.node_id, relation="derived_from", target=plan.node_id)
    store.add_edge(source=plan.node_id, relation="derived_from", target=steps.node_id)

    context_slice = build_context_slice(store, task_id="t", branch_id="main", role="solver", policy="dependency_closure")

    assert query.node_id in context_slice.visible_node_ids
    assert facts.node_id in context_slice.visible_node_ids
    assert plan.node_id in context_slice.visible_node_ids
    assert steps.node_id in context_slice.visible_node_ids
    assert task.node_id in context_slice.visible_node_ids


def test_minimal_verified_rejects_unverified_result() -> None:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="query_spec", node_type="query_spec", content="q", owner="planner")
    store.add_node(task_id="t", branch_id="main", logical_id="result", node_type="result", content="2", owner="solver", validation={"schema_valid": True})
    store.add_node(task_id="t", branch_id="main", logical_id="verification", node_type="verification", content="ok", owner="critic", status="verified")

    with pytest.raises(SliceError):
        build_context_slice(store, task_id="t", branch_id="main", role="final_solver", policy="minimal_verified")


def test_minimal_verified_rejects_verification_for_old_result() -> None:
    store = GraphStore()
    query = store.add_node(task_id="t", branch_id="main", logical_id="query_spec", node_type="query_spec", content="q", owner="planner")
    old_result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content="2",
        owner="solver",
        validation={"schema_valid": True, "model_judged_correct": True},
    )
    verification = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="verification",
        node_type="verification",
        content="ok",
        owner="critic",
        status="verified",
        validation={"schema_valid": True, "model_judged_correct": True},
    )
    store.add_edge(source=verification.node_id, relation="verifies", target=old_result.node_id)
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content="3",
        owner="solver",
        validation={"schema_valid": True, "model_judged_correct": True},
    )
    assert query.node_id

    with pytest.raises(SliceError):
        build_context_slice(store, task_id="t", branch_id="main", role="final_solver", policy="minimal_verified")


def test_minimal_verified_accepts_current_verified_result() -> None:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="query_spec", node_type="query_spec", content="q", owner="planner")
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content="2",
        owner="solver",
        validation={"schema_valid": True, "model_judged_correct": True},
    )
    verification = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="verification",
        node_type="verification",
        content="ok",
        owner="critic",
        status="verified",
        validation={"schema_valid": True, "model_judged_correct": True},
    )
    store.add_edge(source=verification.node_id, relation="verifies", target=result.node_id)

    context_slice = build_context_slice(store, task_id="t", branch_id="main", role="final_solver", policy="minimal_verified")
    assert result.node_id in context_slice.visible_node_ids
    assert verification.node_id in context_slice.visible_node_ids


def test_code_generation_final_context_uses_code_boundary() -> None:
    store = GraphStore()
    task = store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="[domain=code_generation]\nwrite code", owner="user")
    query = store.add_node(task_id="t", branch_id="main", logical_id="query_spec", node_type="query_spec", content={"task_type": "code_generation"}, owner="planner")
    code = store.add_node(task_id="t", branch_id="main", logical_id="code", node_type="code", content="def f():\n    return 1\n", owner="solver")
    verification = store.add_node(task_id="t", branch_id="main", logical_id="verification", node_type="verification", content={"target": "code", "status": "verified"}, owner="critic", status="verified")
    store.add_edge(source=task.node_id, target=query.node_id, relation="depends_on")
    store.add_edge(source=query.node_id, target=code.node_id, relation="depends_on")
    store.add_edge(source=verification.node_id, target=code.node_id, relation="verifies")

    context_slice = build_context_slice(store, task_id="t", branch_id="main", role="final_solver", policy="dependency_closure")

    assert code.node_id in context_slice.visible_node_ids
    assert verification.node_id in context_slice.visible_node_ids


def test_numeric_initial_graph_seeds_facts_without_solving() -> None:
    store = init_workflow_graph("Janet has 16 eggs. She eats 3 and uses 4.", task_id="gsm", task_type="numeric_solve")

    query = store.latest_valid("gsm", "main", "query_spec")
    assert query is not None
    assert query.content["task_type"] == "numeric_solve"
    assert query.provenance["communication_scope"] == "MANDATORY"
    facts = store.latest_valid("gsm", "main", "facts")
    assert facts is not None
    assert [item["id"] for item in facts.content] == ["N1", "N2", "N3"]
    assert [item["value"] for item in facts.content] == [16, 3, 4]
    assert store.latest_valid("gsm", "main", "calculation") is None
    assert store.latest_valid("gsm", "main", "result") is None


def test_new_versions_are_append_only_and_supersede_previous_version() -> None:
    store = GraphStore()
    first = store.add_node(task_id="t", branch_id="main", logical_id="result", node_type="result", content="2", owner="solver")
    second = store.add_node(task_id="t", branch_id="main", logical_id="result", node_type="result", content="3", owner="solver")

    state = store.snapshot()
    assert state.nodes[first.node_id].content == "2"
    assert state.nodes[first.node_id].status == "superseded"
    assert state.nodes[second.node_id].version == 2
    assert store.latest_valid("t", "main", "result").node_id == second.node_id


def test_dependency_digest_tracks_upstream_content() -> None:
    store = GraphStore()
    source = store.add_node(task_id="t", branch_id="main", logical_id="facts", node_type="facts", content="first", owner="planner")
    target = store.add_node(task_id="t", branch_id="main", logical_id="calculation", node_type="calculation", content="x", owner="solver")
    store.add_edge(source=source.node_id, relation="input_to", target=target.node_id)
    first_digest = store.node(target.node_id).dependency_digest

    replacement = store.add_node(task_id="t", branch_id="main", logical_id="facts", node_type="facts", content="second", owner="planner")
    store.add_edge(source=replacement.node_id, relation="input_to", target=target.node_id)
    content_digest = store.node(target.node_id).dependency_digest
    assert content_digest != first_digest


def test_context_slice_digest_tracks_validation_status_changes() -> None:
    store = GraphStore()
    for logical_id in ("query_spec", "facts", "plan", "plan_steps"):
        store.add_node(task_id="t", branch_id="main", logical_id=logical_id, node_type=logical_id, content=logical_id, owner="planner")
    before = build_context_slice(store, task_id="t", branch_id="main", role="solver").dependency_digest
    store.mark_status(store.latest_valid("t", "main", "facts").node_id, "need_fix", reason="test status change")
    after = build_context_slice(store, task_id="t", branch_id="main", role="solver").dependency_digest
    assert before != after


def test_native_validation_edge_marks_the_validated_artifact() -> None:
    store = GraphStore()
    result = store.add_node(task_id="t", branch_id="main", logical_id="result", node_type="result", content="2", owner="solver")
    verification = store.add_node(task_id="t", branch_id="main", logical_id="verification", node_type="verification", content="correct", owner="critic", status="verified")
    store.add_edge(source=verification.node_id, relation="verifies", target=result.node_id)

    current_result = store.node(result.node_id)
    assert current_result.validation["model_judged_correct"] is True


def test_cache_requires_same_runtime_and_dependency_digest() -> None:
    store = GraphStore()
    node = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content="2",
        owner="solver",
        dependency_versions={"calculation": "calculation@v1"},
        validation={"schema_valid": True, "execution_valid": True},
        runtime=runtime(),
    )
    cache = InMemoryResultCache()
    key = cache.put(node, role="solver", runtime=runtime())
    assert key
    assert cache.get(
        task_id="t",
        branch_id="main",
        logical_id="result",
        dependency_digest=node.dependency_digest,
        role="solver",
        runtime=runtime(),
    ).hit
    changed_runtime = RuntimeFingerprint(model_id="m2", tokenizer_id="t", prompt_template_digest="p")
    assert not cache.get(
        task_id="t",
        branch_id="main",
        logical_id="result",
        dependency_digest=node.dependency_digest,
        role="solver",
        runtime=changed_runtime,
    ).hit
    assert not cache.get(
        task_id="t",
        branch_id="main",
        logical_id="result",
        dependency_digest=dependency_digest({"calculation": "calculation@v2"}),
        role="solver",
        runtime=runtime(),
    ).hit


def test_invalidation_propagates_data_and_validation_dependents() -> None:
    store = GraphStore()
    facts = store.add_node(task_id="t", branch_id="main", logical_id="facts", node_type="facts", content="f", owner="planner")
    calc = store.add_node(task_id="t", branch_id="main", logical_id="calculation", node_type="calculation", content="c", owner="solver")
    result = store.add_node(task_id="t", branch_id="main", logical_id="result", node_type="result", content="r", owner="solver")
    verification = store.add_node(task_id="t", branch_id="main", logical_id="verification", node_type="verification", content="ok", owner="critic", status="verified")
    final = store.add_node(task_id="t", branch_id="main", logical_id="final_answer", node_type="final_answer", content="r", owner="final_solver")
    store.add_edge(source=facts.node_id, relation="input_to", target=calc.node_id)
    store.add_edge(source=calc.node_id, relation="produces", target=result.node_id)
    store.add_edge(source=verification.node_id, relation="verifies", target=result.node_id)
    store.add_edge(source=result.node_id, relation="derived_from", target=final.node_id)

    report = invalidate_downstream(store, [facts.node_id], reason="facts_changed")

    assert calc.node_id in report.stale_node_ids
    assert result.node_id in report.stale_node_ids
    assert verification.node_id in report.stale_node_ids
    assert final.node_id in report.stale_node_ids


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
    print("runtime core tests passed")
