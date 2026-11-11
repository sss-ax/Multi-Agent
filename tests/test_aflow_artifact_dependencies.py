from types import SimpleNamespace

from baselines.aflow_artifact_dependencies import (
    capture_invocation_input_bindings,
    materialize_dependency_edges,
)
from baselines.aflow_artifact_graph import ArtifactOccurrence
from baselines.aflow_artifact_router import build_receiver_artifact_closure
from baselines.aflow_artifact_renderers import render_mbpp_fix, render_mbpp_scensemble, render_mbpp_test


def test_runtime_binding_creates_dependent_to_prerequisite_edge():
    bindings = capture_invocation_input_bindings(
        task_id="t1", operator="Test", invocation_id="i2",
        kwargs={"solution": "code"},
        artifact_binding_hints={"solution": ("occ:code",)},
    )
    output = ArtifactOccurrence("occ:test", "canon:test", "Test", "i2", 0, task_id="t1")
    edges = materialize_dependency_edges(
        dataset="MBPP", operator="Test", invocation_id="i2",
        bindings=bindings, output_occurrences=(output,),
    )
    assert len(edges) == 1
    assert edges[0].dependent_node_id == "occ:test"
    assert edges[0].prerequisite_node_id == "occ:code"
    assert edges[0].relation == "test_evaluates"
    assert edges[0].hard is True


def test_closure_does_not_cross_task_or_batch():
    output = ArtifactOccurrence("occ:test", "canon:test", "Test", "i2", 0, task_id="t1", candidate_batch_id="b1")
    session = SimpleNamespace(
        task_id="t1",
        _artifact_occurrences={"occ:test": output},
        _artifact_records={"canon:test": object()},
        _artifact_dependency_edges={},
        _hotpot_source_node_ids=(),
        store=SimpleNamespace(snapshot=lambda: SimpleNamespace(nodes={})),
    )
    closure = build_receiver_artifact_closure(session, "Test", ("canon:test",), task_id="t1", candidate_batch_id="b1")
    assert closure.complete is True
    assert closure.replayable is True
    assert closure.selected_canonical_ids == ()


def test_occurrence_multiplicity_is_not_collapsed_by_binding():
    bindings = capture_invocation_input_bindings(
        task_id="t1", operator="ScEnsemble", invocation_id="i3",
        kwargs={"solutions": ["a", "a"]},
        artifact_binding_hints={"solutions": ("occ:a1", "occ:a2")},
        candidate_batch_id="batch-1",
    )
    assert bindings[0].artifact_occurrence_ids == ("occ:a1", "occ:a2")
    assert bindings[0].candidate_batch_id == "batch-1"


def test_mbpp_renderers_do_not_emit_graph_metadata():
    values = [SimpleNamespace(value="a"), SimpleNamespace(value="a"), SimpleNamespace(value="b")]
    assert render_mbpp_scensemble(values) == {"solutions": ["a", "a", "b"]}
    test_input = render_mbpp_test("problem", "f(x)", "code")
    assert test_input == {"problem": "problem", "entry_point": "f(x)", "solution": "code"}
    fix_input = render_mbpp_fix("problem", "bad", "NameError")
    assert "artifact" not in str(fix_input).lower()
