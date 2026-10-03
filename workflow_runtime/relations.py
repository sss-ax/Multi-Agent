"""Canonical relation semantics for the runtime graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


# All relations in this set are oriented producer/input -> consumer/output.
DATA_DEPENDENCY_RELATIONS: FrozenSet[str] = frozenset(
    {"input_to", "derived_from", "produces", "depends_on"}
)

# These relations are oriented validator -> validated artifact.  If the
# validated artifact changes, the validator becomes stale through the reverse
# direction; this is handled by dependent_node_ids().
VALIDATION_RELATIONS: FrozenSet[str] = frozenset(
    {"verifies", "contradicts", "invalidates"}
)

LINEAGE_RELATIONS: FrozenSet[str] = frozenset({"supersedes"})

ALL_RUNTIME_RELATIONS: FrozenSet[str] = (
    DATA_DEPENDENCY_RELATIONS | VALIDATION_RELATIONS | LINEAGE_RELATIONS
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
