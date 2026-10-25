"""Two-phase, atomic compiler for Semantic Delta proposals."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .graph_store import GraphStore
from .models import NodeOrigin, VisibilityClass
from .semantic_delta_ir import SemanticDelta
from .semantic_delta_validator import (
    ValidatedSemanticDelta,
    validate_semantic_delta,
)
from .task_spec import TaskSpec, get_task_spec


_SINGLETON_LOGICAL_IDS = {
    "calculation": "calculation",
    "candidate_answer": "candidate_answer",
    "code": "code",
    "slot_analysis": "slot_analysis",
    "repair_patch": "repair_patch",
}


@dataclass(frozen=True)
class GraphMutationPlan:
    """Validated, kernel-owned mutation description before commit."""

    role: str
    node_specs: tuple[dict[str, Any], ...]
    edge_specs: tuple[dict[str, Any], ...]
    send_hints: tuple[dict[str, Any], ...] = ()
    temp_to_logical_id: dict[str, str] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "nodes": [copy.deepcopy(item) for item in self.node_specs],
            "edges": [copy.deepcopy(item) for item in self.edge_specs],
            "send_hints": [copy.deepcopy(item) for item in self.send_hints],
            "temp_to_logical_id": dict(self.temp_to_logical_id or {}),
        }


@dataclass(frozen=True)
class CompiledSemanticDelta:
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    send_hints: tuple[dict[str, Any], ...]
    temp_to_node_id: dict[str, str]

    @property
    def all_ids(self) -> tuple[str, ...]:
        return self.node_ids + self.edge_ids


class SemanticDeltaCompiler:
    """Validate and atomically commit a model-proposed semantic batch."""

    def __init__(
        self,
        store: GraphStore,
        *,
        task_id: str,
        branch_id: str = "main",
        task_type: str,
        task_spec: TaskSpec | None = None,
    ) -> None:
        self.store = store
        self.task_id = task_id
        self.branch_id = branch_id
        self.task_type = task_type
        self.task_spec = task_spec or get_task_spec(task_type)

    def build_plan(
        self,
        role: str,
        payload: SemanticDelta | dict[str, Any],
        *,
        visible_node_ids: Iterable[str] = (),
        known_fragment_ids: Iterable[str] = (),
        allowed_receivers: Iterable[str] = (),
        node_ref_context: dict[str, Any] | None = None,
    ) -> GraphMutationPlan:
        """Validate a proposal and return a no-side-effect mutation plan."""
        validated = validate_semantic_delta(
            payload,
            role=role,
            task_id=self.task_id,
            branch_id=self.branch_id,
            store=self.store,
            task_spec=self.task_spec,
            visible_node_ids=visible_node_ids,
            known_fragment_ids=known_fragment_ids,
            allowed_receivers=allowed_receivers,
            node_ref_context=node_ref_context,
        )
        used = {
            node.logical_id
            for node in self.store.snapshot().nodes.values()
            if node.task_id == self.task_id and node.branch_id == self.branch_id
        }
        node_specs: list[dict[str, Any]] = []
        temp_to_logical: dict[str, str] = {}
        for proposed in validated.delta.nodes:
            logical_id = self._allocate_logical_id(proposed.node_type, proposed.temp_id, used)
            temp_to_logical[proposed.temp_id] = logical_id
            node_specs.append({
                "temp_id": proposed.temp_id,
                "logical_id": logical_id,
                "type": proposed.node_type,
                "content": copy.deepcopy(proposed.content),
                "grounding_node_ids": list(validated.grounding_nodes.get(proposed.temp_id, ())),
            })
        return GraphMutationPlan(
            role=role,
            node_specs=tuple(node_specs),
            edge_specs=tuple(edge.as_dict() for edge in validated.delta.edges),
            send_hints=tuple(item.as_dict() for item in validated.delta.send_hints),
            temp_to_logical_id=temp_to_logical,
        )

    def apply(
        self,
        role: str,
        payload: SemanticDelta | dict[str, Any],
        *,
        visible_node_ids: Iterable[str] = (),
        known_fragment_ids: Iterable[str] = (),
        allowed_receivers: Iterable[str] = (),
        node_ref_context: dict[str, Any] | None = None,
    ) -> CompiledSemanticDelta:
        validated = validate_semantic_delta(
            payload,
            role=role,
            task_id=self.task_id,
            branch_id=self.branch_id,
            store=self.store,
            task_spec=self.task_spec,
            visible_node_ids=visible_node_ids,
            known_fragment_ids=known_fragment_ids,
            allowed_receivers=allowed_receivers,
            node_ref_context=node_ref_context,
        )
        # Compile against an isolated clone.  No mutation reaches the live
        # store until every node and edge has been accepted by GraphStore.
        working = self.store.clone()
        temp_to_node: dict[str, Any] = {}
        node_ids: list[str] = []
        used_logical_ids = {
            node.logical_id
            for node in working.snapshot().nodes.values()
            if node.task_id == self.task_id and node.branch_id == self.branch_id
        }
        for proposed in validated.delta.nodes:
            logical_id = self._allocate_logical_id(proposed.node_type, proposed.temp_id, used_logical_ids)
            grounding_ids = list(validated.grounding_nodes.get(proposed.temp_id, ()))
            communication_scope = "OPTIONAL_POLICY"
            if (role == "planner" and proposed.node_type in {"fact", "plan_step", "plan", "entity_hypothesis", "retrieval_query"}) or (
                role == "solver" and proposed.node_type in {"calculation", "candidate_answer", "code", "selected_evidence", "bridge_hypothesis"}
            ):
                communication_scope = "MANDATORY_STAGE"
            provenance = {
                "origin": NodeOrigin.MODEL,
                "trust": "untrusted_generated_test" if proposed.node_type == "generated_test" else "model_generated",
                "visibility_class": VisibilityClass.ROLE_SCOPED,
                "communication_scope": communication_scope,
                "task_type": self.task_type,
                "semantic_delta": True,
                "temp_id": proposed.temp_id,
                "grounding_node_ids": grounding_ids,
            }
            node = working.add_node(
                task_id=self.task_id,
                branch_id=self.branch_id,
                logical_id=logical_id,
                node_type=proposed.node_type,
                content=copy.deepcopy(proposed.content),
                owner=role,
                status="ready",
                validation={"schema_valid": True, "grounding_checked": bool(grounding_ids) or not proposed.grounding},
                source_refs=grounding_ids,
                provenance=provenance,
                created_by_role=role,
            )
            temp_to_node[proposed.temp_id] = node
            node_ids.append(node.node_id)

        edge_ids: list[str] = []
        candidate_nodes = [
            (proposed, temp_to_node[proposed.temp_id])
            for proposed in validated.delta.nodes
            if proposed.node_type == "candidate_answer"
        ]
        if candidate_nodes:
            proposed, candidate = candidate_nodes[-1]
            candidate_value = copy.deepcopy(proposed.content)
            if self.task_type in {"numeric_solve", "numeric_comparison"} and isinstance(candidate_value, dict):
                if "value" in candidate_value:
                    candidate_value = candidate_value["value"]
                elif "answer" in candidate_value:
                    candidate_value = candidate_value["answer"]
            result = working.add_node(
                task_id=self.task_id,
                branch_id=self.branch_id,
                logical_id="result",
                node_type="result",
                content={"id": proposed.temp_id, "value": candidate_value},
                owner="runtime",
                status="ready",
                source_refs=[candidate.node_id],
                provenance={
                    "origin": NodeOrigin.RUNTIME,
                    "trust": "runtime_boundary",
                    "visibility_class": VisibilityClass.ROLE_SCOPED,
                    "semantic_delta_result": True,
                },
                created_by_role="runtime",
            )
            runtime_edge = working.add_edge(
                source=candidate.node_id,
                target=result.node_id,
                relation="produces",
                created_by_role="runtime",
                metadata={"semantic_delta": True, "runtime_boundary": True},
            )
            edge_ids.append(runtime_edge.edge_id)
            node_ids.append(result.node_id)
        if validated.delta.complete:
            proposed_types = {node.node_type for node in validated.delta.nodes}
            missing_outputs = [
                node_type
                for node_type in self.task_spec.required_for(role)
                if node_type not in proposed_types
            ]
            if missing_outputs:
                raise ValueError(
                    f"complete Semantic Delta is missing required {role} outputs: "
                    + ", ".join(missing_outputs)
                )
            boundary = working.add_node(
                task_id=self.task_id,
                branch_id=self.branch_id,
                logical_id=f"semantic_delta_{role}_boundary",
                node_type="semantic_delta_boundary",
                content={"role": role, "complete": True},
                owner="runtime",
                status="verified",
                provenance={
                    "origin": NodeOrigin.RUNTIME,
                    "trust": "runtime_boundary",
                    "visibility_class": VisibilityClass.PUBLIC,
                    "communication_scope": "MANDATORY",
                    "semantic_delta_boundary": True,
                },
                created_by_role="runtime",
            )
            node_ids.append(boundary.node_id)
        for proposed in validated.delta.nodes:
            target = temp_to_node[proposed.temp_id]
            for source_id in validated.grounding_nodes.get(proposed.temp_id, ()):
                edge = working.add_edge(
                    source=source_id,
                    target=target.node_id,
                    relation="grounded_in",
                    created_by_role="runtime",
                    metadata={"semantic_delta": True, "grounding": True},
                )
                edge_ids.append(edge.edge_id)
        for proposed_edge in validated.delta.edges:
            source = self._resolve_endpoint(working, temp_to_node, proposed_edge.source, node_ref_context)
            target = self._resolve_endpoint(working, temp_to_node, proposed_edge.target, node_ref_context)
            if source is None or target is None:
                # Validation should have caught this; keep the commit atomic if
                # a future resolver changes independently.
                raise ValueError(
                    f"Semantic Delta endpoint disappeared: {proposed_edge.source} -> {proposed_edge.target}"
                )
            edge = working.add_edge(
                source=source.node_id,
                target=target.node_id,
                relation=proposed_edge.relation,
                created_by_role=role,
                metadata={**dict(proposed_edge.metadata), "semantic_delta": True},
            )
            edge_ids.append(edge.edge_id)

        # This is the commit point.  A failed validation/compile above leaves
        # the canonical store byte-for-byte unchanged.
        self.store.replace_from(working)
        return CompiledSemanticDelta(
            node_ids=tuple(node_ids),
            edge_ids=tuple(edge_ids),
            send_hints=tuple(item.as_dict() for item in validated.delta.send_hints),
            temp_to_node_id={temp_id: node.node_id for temp_id, node in temp_to_node.items()},
        )

    @staticmethod
    def _allocate_logical_id(node_type: str, temp_id: str, used: set[str]) -> str:
        if node_type in _SINGLETON_LOGICAL_IDS:
            # Stable logical boundaries intentionally create new versions on
            # repair.  Downstream tools resolve `code`, `candidate_answer`,
            # etc. without knowing model-selected temporary IDs.
            return _SINGLETON_LOGICAL_IDS[node_type]
        safe_type = re.sub(r"[^a-zA-Z0-9_]+", "_", node_type).strip("_").lower() or "node"
        safe_temp = re.sub(r"[^a-zA-Z0-9_]+", "_", temp_id).strip("_").lower() or "item"
        base = f"{safe_type}_{safe_temp}"
        candidate = base
        index = 2
        while candidate in used:
            candidate = f"{base}_{index}"
            index += 1
        used.add(candidate)
        return candidate

    @staticmethod
    def _resolve_endpoint(
        working: GraphStore,
        temp_to_node: dict[str, Any],
        ref: str,
        node_ref_context: dict[str, Any] | None = None,
    ):
        if ref in temp_to_node:
            return temp_to_node[ref]
        resolved_ref = ref
        if isinstance(node_ref_context, dict):
            for index_name in ("local_node_index", "fragment_index", "semantic_type_index"):
                index = node_ref_context.get(index_name, {})
                if not isinstance(index, dict) or ref not in index:
                    continue
                item = index[ref]
                if isinstance(item, dict):
                    resolved_ref = str(item.get("source_node_id", item.get("node_id", item.get("source_logical_id", ref))))
                elif isinstance(item, str):
                    resolved_ref = item
                break
        state = working.snapshot()
        if resolved_ref in state.nodes:
            return state.nodes[resolved_ref]
        # A model may use a logical ID for an existing visible node.
        candidates = [node for node in state.nodes.values() if node.logical_id == resolved_ref and node.is_operationally_valid()]
        return max(candidates, key=lambda node: (node.version, node.created_at, node.node_id), default=None)
