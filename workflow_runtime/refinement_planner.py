"""Deterministic targeted refinement planning for semantic NACKs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .agent_graph_view import AgentGraphView
from .delta_closure import dependency_closure
from .models import GraphState, NodeVersion
from .relations import DATA_DEPENDENCY_RELATIONS, VALIDATION_RELATIONS
from .semantic_feedback import SemanticNACK

REFINEMENT_TARGETED = "TARGETED"
REFINEMENT_UNRESOLVABLE = "UNRESOLVABLE"


@dataclass(frozen=True)
class RefinementPlan:
    sender: str
    receiver: str
    missing_semantics: tuple[str, ...]
    root_node_ids: tuple[str, ...]
    reason_by_node: dict[str, str]
    status: str = REFINEMENT_TARGETED

    @property
    def is_empty(self) -> bool:
        return not self.root_node_ids

    @property
    def is_unresolvable(self) -> bool:
        return self.status == REFINEMENT_UNRESOLVABLE

    def as_dict(self) -> dict[str, object]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "missing_semantics": list(self.missing_semantics),
            "root_node_ids": list(self.root_node_ids),
            "reason_by_node": dict(self.reason_by_node),
            "status": self.status,
            "is_unresolvable": self.is_unresolvable,
            "is_empty": self.is_empty,
        }


class RefinementPlanner:
    """Choose minimal sender-side roots for missing receiver semantics."""

    def __init__(
        self,
        state: GraphState,
        *,
        sender: str,
        receiver: str,
        sender_view: AgentGraphView,
        receiver_view: AgentGraphView,
    ) -> None:
        self.state = state
        self.sender = sender
        self.receiver = receiver
        self.sender_view = sender_view
        self.receiver_view = receiver_view

    def plan_for_nack(self, nack: SemanticNACK) -> RefinementPlan:
        return self.plan(nack.missing_semantics)

    def plan(self, missing_semantics: Iterable[str]) -> RefinementPlan:
        semantics = tuple(dict.fromkeys(str(item) for item in missing_semantics))
        roots: list[str] = []
        reasons: dict[str, str] = {}
        for semantic in semantics:
            selected = self._select_lowest_cost_root(semantic)
            if selected is None:
                return RefinementPlan(
                    sender=self.sender,
                    receiver=self.receiver,
                    missing_semantics=semantics,
                    root_node_ids=(),
                    reason_by_node={},
                    status=REFINEMENT_UNRESOLVABLE,
                )
            node_id, reason = selected
            if node_id not in roots:
                roots.append(node_id)
                reasons[node_id] = reason
        return RefinementPlan(
            sender=self.sender,
            receiver=self.receiver,
            missing_semantics=semantics,
            root_node_ids=tuple(roots),
            reason_by_node=reasons,
            status=REFINEMENT_TARGETED,
        )

    def _select_lowest_cost_root(self, semantic: str) -> tuple[str, str] | None:
        candidates = self._roots_for_semantic(semantic)
        if not candidates:
            return None
        ranked = []
        for node_id, reason in candidates:
            delta = dependency_closure(
                self.state,
                root_node_ids=(node_id,),
                sender=self.sender,
                receiver=self.receiver,
                receiver_view=self.receiver_view,
                policy="refinement_planner:cost_probe",
            )
            if not delta.node_ids:
                continue
            ranked.append((delta.token_cost, _type_priority(self.state.nodes[node_id].type), node_id, reason))
        if not ranked:
            return None
        _cost, _priority, node_id, reason = min(ranked)
        return node_id, reason

    def _roots_for_semantic(self, semantic: str) -> tuple[tuple[str, str], ...]:
        if semantic in {"candidate_answer_or_code", "final_candidate"}:
            return tuple(
                (node.node_id, f"missing {semantic}: send candidate artifact")
                for node in self._latest_sender_nodes(("result", "code", "final_answer", "execution", "test_result"))
            )
        if semantic == "candidate_code":
            return tuple(
                (node.node_id, "missing candidate_code: send code artifact")
                for node in self._latest_sender_nodes(("code", "execution"))
            )
        if semantic == "validation_signal":
            return tuple(
                (node.node_id, "missing validation_signal: send validation artifact")
                for node in self._latest_sender_nodes(("verification", "test_result"))
            )
        if semantic == "support_dependencies":
            candidate_ids = [node.node_id for node in self._latest_sender_nodes(("result", "code", "execution", "test_result"))]
            dependency_ids = self._direct_dependencies(candidate_ids)
            if not dependency_ids:
                dependency_ids = tuple(
                    node.node_id
                    for node in self._latest_sender_nodes(("calculation", "plan_steps", "fact", "facts", "requirements", "test"))
                    if node.node_id not in self.receiver_view.visible_node_ids
                )
            return tuple(
                (node_id, "missing support_dependencies: send direct dependency")
                for node_id in dependency_ids
            )
        if semantic in {"plan_or_operation", "task_inputs"}:
            types = ("plan", "plan_steps") if semantic == "plan_or_operation" else (
                "facts", "fact", "requirements", "test", "table", "table_cell",
                "evidence", "entity", "supporting_fact", "evidence_link", "choice",
            )
            return tuple(
                (node.node_id, f"missing {semantic}: send typed support")
                for node in self._latest_sender_nodes(types)
            )
        return ()

    def _latest_sender_nodes(self, node_types: tuple[str, ...]) -> list[NodeVersion]:
        visible = self.sender_view.visible_node_ids
        seen: dict[tuple[str, str], NodeVersion] = {}
        for node in self.state.nodes.values():
            if node.node_id not in visible or not node.is_operationally_valid() or node.type not in node_types:
                continue
            key = (node.logical_id, node.type)
            current = seen.get(key)
            if current is None or (node.version, node.created_at, node.node_id) > (current.version, current.created_at, current.node_id):
                seen[key] = node
        return sorted(seen.values(), key=lambda node: (_type_priority(node.type), node.logical_id, node.version, node.node_id))

    def _direct_dependencies(self, root_node_ids: Iterable[str]) -> tuple[str, ...]:
        visible = self.sender_view.visible_node_ids
        receiver_visible = self.receiver_view.visible_node_ids
        dependencies: list[str] = []
        for root_id in root_node_ids:
            for edge in self.state.edges:
                if edge.relation in DATA_DEPENDENCY_RELATIONS and edge.target == root_id:
                    dep_id = edge.source
                elif edge.relation in VALIDATION_RELATIONS and edge.source == root_id:
                    dep_id = edge.target
                else:
                    continue
                if dep_id in visible and dep_id not in receiver_visible and dep_id not in dependencies:
                    dependencies.append(dep_id)
        return tuple(dependencies)


def plan_refinement(
    state: GraphState,
    *,
    sender: str,
    receiver: str,
    sender_view: AgentGraphView,
    receiver_view: AgentGraphView,
    missing_semantics: Iterable[str],
) -> RefinementPlan:
    return RefinementPlanner(
        state,
        sender=sender,
        receiver=receiver,
        sender_view=sender_view,
        receiver_view=receiver_view,
    ).plan(missing_semantics)


def _type_priority(node_type: str) -> int:
    priorities = {
        "result": 0,
        "code": 1,
        "final_answer": 2,
        "verification": 3,
        "test_result": 4,
        "execution": 5,
        "calculation": 6,
        "plan": 7,
        "plan_steps": 8,
        "facts": 9,
        "fact": 10,
    }
    return priorities.get(node_type, 100)
