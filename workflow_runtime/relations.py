"""Canonical relation semantics for the runtime graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


# All relations in this set are oriented producer/input -> consumer/output.
DATA_DEPENDENCY_RELATIONS: FrozenSet[str] = frozenset(
    {"input_to", "derived_from", "produces", "depends_on", "grounded_in"}
)

# These relations are oriented validator -> validated artifact.  If the
# validated artifact changes, the validator becomes stale through the reverse
# direction; this is handled by dependent_node_ids().
VALIDATION_RELATIONS: FrozenSet[str] = frozenset(
    {"verifies", "contradicts", "invalidates"}
)

LINEAGE_RELATIONS: FrozenSet[str] = frozenset({"supersedes"})

# Audit/provenance links that must not be traversed as model-visible data
# dependencies.  In particular, an internal AFlow invocation materializes a
# canonical artifact but is not itself part of the receiver context.
TRACE_RELATIONS: FrozenSet[str] = frozenset({"materializes"})

# Structural source-order links are auditable context metadata, not data
# dependencies.  They must not force a full sentence chain into a closure.
CONTEXT_RELATIONS: FrozenSet[str] = frozenset({
    "next_sentence", "query_mentions_entity", "title_identifies_entity",
    "sentence_mentions_entity", "entity_co_occurs", "query_requires_subgoal",
    "subgoal_requires_relation", "sentence_supports_claim", "claim_subject",
    "claim_object", "claim_depends_on", "claim_answers_subgoal",
    "produces_artifact", "artifact_depends_on", "artifact_supports",
    "artifact_contradicts", "artifact_refines", "artifact_supersedes",
    "artifact_cites",
})

ALL_RUNTIME_RELATIONS: FrozenSet[str] = (
    DATA_DEPENDENCY_RELATIONS
    | VALIDATION_RELATIONS
    | LINEAGE_RELATIONS
    | TRACE_RELATIONS
    | CONTEXT_RELATIONS
)


@dataclass(frozen=True)
class RelationSpec:
    name: str
    source_role: str
    target_role: str
    propagates_source_change: bool
    propagates_target_change: bool
    participates_in_dag: bool


RELATION_SPECS = {
    relation: RelationSpec(
        name=relation,
        source_role="producer",
        target_role="consumer",
        propagates_source_change=True,
        propagates_target_change=False,
        participates_in_dag=True,
    )
    for relation in DATA_DEPENDENCY_RELATIONS
}
RELATION_SPECS.update(
    {
        relation: RelationSpec(
            name=relation,
            source_role="validator",
            target_role="validated",
            propagates_source_change=False,
            propagates_target_change=True,
            participates_in_dag=False,
        )
        for relation in VALIDATION_RELATIONS
    }
)
RELATION_SPECS.update(
    {
        relation: RelationSpec(
            name=relation,
            source_role="predecessor",
            target_role="successor",
            propagates_source_change=False,
            propagates_target_change=False,
            participates_in_dag=False,
        )
        for relation in CONTEXT_RELATIONS
    }
)
RELATION_SPECS.update(
    {
        relation: RelationSpec(
            name=relation,
            source_role="invocation",
            target_role="artifact",
            propagates_source_change=False,
            propagates_target_change=False,
            participates_in_dag=False,
        )
        for relation in TRACE_RELATIONS
    }
)
RELATION_SPECS.update(
    {
        relation: RelationSpec(
            name=relation,
            source_role="new_version",
            target_role="old_version",
            propagates_source_change=False,
            propagates_target_change=False,
            participates_in_dag=False,
        )
        for relation in LINEAGE_RELATIONS
    }
)


def is_data_relation(relation: str) -> bool:
    return relation in DATA_DEPENDENCY_RELATIONS


def is_validation_relation(relation: str) -> bool:
    return relation in VALIDATION_RELATIONS
