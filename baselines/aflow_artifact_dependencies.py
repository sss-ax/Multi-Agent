"""Runtime-bound hard dependencies for Dataset-Aware Artifact routing.

This index is intentionally separate from the GraphStore data DAG.  Artifact
edges are receiver-relative evidence for routing and must not participate in
stale propagation or invocation reuse invalidation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .aflow_artifact_graph import (
    ArtifactDependencyEdge,
    ArtifactOccurrence,
    InvocationInputBinding,
    stable_digest,
)

DATA_DEPENDENCY_RELATIONS = frozenset({
    "artifact_depends_on", "decision_depends_on", "verification_depends_on",
    "test_evaluates", "fix_depends_on_error", "evidence_supports",
})


def _relation(operator: str, field: str) -> str:
    op = str(operator).casefold()
    name = str(field).casefold()
    if op == "scensemble" or name in {"solutions", "candidates"}:
        return "decision_depends_on"
    if op == "test" or name in {"solution", "selected_code"}:
        return "test_evaluates"
    if name in {"error", "test_result", "feedback", "failure"}:
        return "fix_depends_on_error"
    if name in {"evidence", "evidence_ref", "citations"}:
        return "evidence_supports"
    if name in {"verification", "tests", "public_tests"}:
        return "verification_depends_on"
    return "artifact_depends_on"


def materialize_dependency_edges(
    *,
    dataset: str,
    operator: str,
    invocation_id: str,
    bindings: Iterable[InvocationInputBinding],
    output_occurrences: Iterable[ArtifactOccurrence],
) -> tuple[ArtifactDependencyEdge, ...]:
    """Create hard edges from actual input bindings to output occurrences."""
    outputs = tuple(output_occurrences)
    edges: list[ArtifactDependencyEdge] = []
    for binding in bindings:
        prerequisites = tuple(binding.artifact_occurrence_ids) + tuple(binding.source_node_ids)
        if not prerequisites:
            continue
        relation = _relation(operator, binding.field_name)
        for output in outputs:
            for prerequisite in prerequisites:
                if not prerequisite or prerequisite == output.occurrence_id:
                    continue
                edge_id = "edge:artifact:{}".format(stable_digest({
                    "task": binding.task_id, "dependent": output.occurrence_id,
                    "prerequisite": prerequisite, "field": binding.field_name,
                    "invocation": invocation_id,
                })[:20])
                edges.append(ArtifactDependencyEdge(
                    edge_id=edge_id,
                    task_id=binding.task_id,
                    relation=relation,
                    dependent_node_id=output.occurrence_id,
                    prerequisite_node_id=str(prerequisite),
                    producer_invocation_id=output.invocation_id,
                    consumer_invocation_id=invocation_id,
                    input_field=binding.field_name,
                    candidate_batch_id=binding.candidate_batch_id,
                    hard=True,
                    provenance=f"{dataset}:{operator}:runtime_binding",
                ))
    return tuple(dict((edge.edge_id, edge) for edge in edges).values())


def capture_invocation_input_bindings(
    *,
    task_id: str,
    operator: str,
    invocation_id: str,
    kwargs: Mapping[str, Any],
    artifact_binding_hints: Mapping[str, tuple[str, ...]] | None = None,
    source_node_ids: Iterable[str] = (),
    candidate_batch_id: str | None = None,
) -> tuple[InvocationInputBinding, ...]:
    """Capture explicit hidden binding hints without parsing model prompts."""
    hints = artifact_binding_hints or {}
    source_ids = tuple(str(x) for x in source_node_ids)
    bindings: list[InvocationInputBinding] = []
    fields = list(kwargs.items())
    for virtual_field in hints:
        if virtual_field not in kwargs:
            fields.append((virtual_field, None))
    for field, value in fields:
        occurrence_ids = tuple(str(x) for x in hints.get(str(field), ()))
        field_sources = source_ids if not occurrence_ids and str(field) in {"problem", "question", "entry_point", "public_tests", "tests"} else ()
        resolution = "exact_occurrence" if occurrence_ids else ("source_match" if field_sources else "raw_unresolved")
        bindings.append(InvocationInputBinding(
            binding_id=f"binding:{stable_digest((task_id, invocation_id, field, value))[:20]}",
            task_id=str(task_id), consumer_invocation_id=str(invocation_id),
            consumer_operator=str(operator), field_name=str(field),
            artifact_occurrence_ids=occurrence_ids, source_node_ids=field_sources,
            raw_value_digest=stable_digest(value), resolution=resolution,
            candidate_batch_id=candidate_batch_id,
        ))
    return tuple(bindings)
