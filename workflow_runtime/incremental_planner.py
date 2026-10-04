"""Receiver-aware incremental request planning at fragment granularity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .agent_graph_view import AgentGraphView
from .delta_closure import dependency_closure
from .models import GraphState, NodeVersion
from .receiver_need import ReceiverNeed, ReceiverNeedLevel
from .relations import DATA_DEPENDENCY_RELATIONS, VALIDATION_RELATIONS
from .semantic_fragments import (
    MANDATORY_FRAGMENT,
    OPTIONAL_FRAGMENT,
    fragment_token_cost,
    fragments_for_node,
)

INCREMENTAL_TARGETED = "TARGETED"
INCREMENTAL_UNRESOLVABLE = "UNRESOLVABLE"


@dataclass(frozen=True)
class IncrementalRequestPlan:
    sender: str
    receiver: str
    level: str
    target_requirements: tuple[str, ...]
    root_node_ids: tuple[str, ...]
    requested_fragment_ids: tuple[str, ...]
    candidate_fragment_ids: tuple[str, ...]
    selected_fragment_ids: tuple[str, ...]
    rejected_fragment_ids: tuple[str, ...]
    reason_by_node: dict[str, str]
    reason_by_fragment: dict[str, str]
    token_cost: int
    estimated_cost: int
    estimated_utility: float = 0.0
    utility_per_token: float = 0.0
    status: str = INCREMENTAL_TARGETED

    @property
    def is_empty(self) -> bool:
        return not self.root_node_ids and not self.requested_fragment_ids

    @property
    def is_unresolvable(self) -> bool:
        return self.status == INCREMENTAL_UNRESOLVABLE

    def as_dict(self) -> dict[str, object]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "level": self.level,
            "target_requirements": list(self.target_requirements),
            "root_node_ids": list(self.root_node_ids),
            "requested_fragment_ids": list(self.requested_fragment_ids),
            "candidate_fragments": list(self.candidate_fragment_ids),
            "selected_fragments": list(self.selected_fragment_ids),
            "rejected_fragments": list(self.rejected_fragment_ids),
            "reason_by_node": dict(self.reason_by_node),
            "reason_by_fragment": dict(self.reason_by_fragment),
            "token_cost": self.token_cost,
            "estimated_cost": self.estimated_cost,
            "estimated_utility": self.estimated_utility,
            "utility_per_token": self.utility_per_token,
            "reason": dict(self.reason_by_fragment),
            "status": self.status,
            "is_empty": self.is_empty,
            "is_unresolvable": self.is_unresolvable,
        }


class IncrementalRequestPlanner:
    """Choose the smallest visible sender fragments for a receiver need."""

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

    def plan_need(self, need: ReceiverNeed) -> IncrementalRequestPlan:
        if need.has_hard_missing:
            return self.plan(need.hard_missing, level=ReceiverNeedLevel.HARD.value)
        if need.has_verification_missing:
            return self.plan(need.verification_missing, level=ReceiverNeedLevel.VERIFICATION.value)
        if need.has_quality_gap:
            return self.plan(need.quality_gaps, level=ReceiverNeedLevel.QUALITY.value)
        return IncrementalRequestPlan(
            sender=self.sender,
            receiver=self.receiver,
            level=ReceiverNeedLevel.SATISFIED.value,
            target_requirements=(),
            root_node_ids=(),
            requested_fragment_ids=(),
            candidate_fragment_ids=(),
            selected_fragment_ids=(),
            rejected_fragment_ids=(),
            reason_by_node={},
            reason_by_fragment={},
            token_cost=0,
            estimated_cost=0,
            estimated_utility=0.0,
            utility_per_token=0.0,
        )

    def plan(self, requirements: Iterable[str], *, level: str) -> IncrementalRequestPlan:
        targets = tuple(dict.fromkeys(str(item) for item in requirements if str(item).strip()))
        roots: list[str] = []
        fragments: list[str] = []
        candidate_fragments: list[str] = []
        reasons_by_node: dict[str, str] = {}
        reasons_by_fragment: dict[str, str] = {}
        for requirement in targets:
            selected = self._select_lowest_cost_candidate(requirement, level=level)
            if selected is None:
                return self._unresolvable(targets, level)
            node_id, fragment_ids, node_reason, fragment_reason, requirement_candidates, utility = selected
            candidate_fragments.extend(requirement_candidates)
            if node_id not in roots:
                roots.append(node_id)
                reasons_by_node[node_id] = node_reason
            for fragment_id in fragment_ids:
                if fragment_id not in fragments:
                    fragments.append(fragment_id)
                    reasons_by_fragment[fragment_id] = fragment_reason
        estimated_cost = fragment_token_cost(fragments, self.state.nodes)
        estimated_utility = sum(
            _quality_utility(requirement) if level == ReceiverNeedLevel.QUALITY.value else 1.0
            for requirement in targets
        )
        return IncrementalRequestPlan(
            sender=self.sender,
            receiver=self.receiver,
            level=level,
            target_requirements=targets,
            root_node_ids=tuple(roots),
            requested_fragment_ids=tuple(fragments),
            candidate_fragment_ids=tuple(dict.fromkeys(candidate_fragments)),
            selected_fragment_ids=tuple(fragments),
            rejected_fragment_ids=tuple(
                fragment_id for fragment_id in dict.fromkeys(candidate_fragments)
                if fragment_id not in set(fragments)
            ),
            reason_by_node=reasons_by_node,
            reason_by_fragment=reasons_by_fragment,
            token_cost=estimated_cost,
            estimated_cost=estimated_cost,
            estimated_utility=estimated_utility,
            utility_per_token=estimated_utility / max(1, estimated_cost),
        )

    def _unresolvable(self, targets: tuple[str, ...], level: str) -> IncrementalRequestPlan:
        return IncrementalRequestPlan(
            sender=self.sender,
            receiver=self.receiver,
            level=level,
            target_requirements=targets,
            root_node_ids=(),
            requested_fragment_ids=(),
            candidate_fragment_ids=(),
            selected_fragment_ids=(),
            rejected_fragment_ids=(),
            reason_by_node={},
            reason_by_fragment={},
            token_cost=0,
            estimated_cost=0,
            estimated_utility=0.0,
            utility_per_token=0.0,
            status=INCREMENTAL_UNRESOLVABLE,
        )

    def _select_lowest_cost_candidate(
        self,
        requirement: str,
        *,
        level: str,
    ) -> tuple[str, tuple[str, ...], str, str, tuple[str, ...], float] | None:
        candidates = self._candidates_for_requirement(requirement, level=level)
        ranked = []
        candidate_fragments: list[str] = []
        for node_id, fragment_ids, reason in candidates:
            candidate_fragments.extend(fragment_ids)
            delta = dependency_closure(
                self.state,
                root_node_ids=(node_id,),
                sender=self.sender,
                receiver=self.receiver,
                receiver_view=self.receiver_view,
                policy="incremental_request_planner:cost_probe",
            )
            if not delta.node_ids:
                continue
            cost = fragment_token_cost(fragment_ids, self.state.nodes)
            utility = _quality_utility(requirement) if level == ReceiverNeedLevel.QUALITY.value else 1.0
            if level == ReceiverNeedLevel.QUALITY.value:
                # Quality requests optimize expected marginal utility per rendered
                # token. Hard and verification requests stay cost-minimal.
                ranked.append((-utility / max(1, cost), cost, delta.token_cost, _type_priority(self.state.nodes[node_id].type), node_id, fragment_ids, reason, utility))
            else:
                ranked.append((cost, delta.token_cost, _type_priority(self.state.nodes[node_id].type), node_id, fragment_ids, reason, utility))
        if not ranked:
            return None
        selected = min(ranked)
        if level == ReceiverNeedLevel.QUALITY.value:
            _score, cost, _closure_cost, _priority, node_id, fragment_ids, reason, utility = selected
            return (
                node_id,
                fragment_ids,
                reason,
                f"{reason}; utility/cost={utility / max(1, cost):.6f}",
                tuple(dict.fromkeys(candidate_fragments)),
                float(utility),
            )
        _cost, _closure_cost, _priority, node_id, fragment_ids, reason, utility = selected
        return node_id, fragment_ids, reason, f"{reason}; minimum closure cost", tuple(dict.fromkeys(candidate_fragments)), float(utility)

    def _candidates_for_requirement(
        self,
        requirement: str,
        *,
        level: str,
    ) -> tuple[tuple[str, tuple[str, ...], str], ...]:
        if level == ReceiverNeedLevel.QUALITY.value:
            return self._quality_candidates(requirement)
        roots = self._root_candidates(requirement)
        return tuple(
            (node.node_id, self._core_fragments(node), reason)
            for node, reason in roots
            if self._core_fragments(node)
        )

    def _root_candidates(self, requirement: str) -> tuple[tuple[NodeVersion, str], ...]:
        if requirement in {"candidate_answer_or_code", "final_candidate"}:
            return tuple(
                (node, f"missing {requirement}: send candidate artifact")
                for node in self._latest_sender_nodes(("result", "code", "final_answer", "execution", "test_result"))
            )
        if requirement == "candidate_code":
            return tuple(
                (node, "missing candidate_code: send code artifact")
                for node in self._latest_sender_nodes(("code", "execution"))
            )
        if requirement == "validation_signal":
            return tuple(
                (node, "missing validation_signal: send validation artifact")
                for node in self._latest_sender_nodes(("verification", "test_result"))
            )
        if requirement == "support_dependencies":
            candidate_ids = [node.node_id for node in self._latest_sender_nodes(("result", "code", "execution", "test_result"))]
            dependency_ids = self._direct_dependencies(candidate_ids)
            if not dependency_ids:
                dependency_ids = tuple(
                    node.node_id
                    for node in self._latest_sender_nodes(("calculation", "plan_steps", "fact", "facts", "requirements", "test"))
                    if node.node_id not in self.receiver_view.visible_node_ids
                )
            return tuple(
                (self.state.nodes[node_id], "missing support_dependencies: request direct support core")
                for node_id in dependency_ids
                if node_id in self.state.nodes
            )
        if requirement in {"plan_or_operation", "task_inputs"}:
            types = ("plan", "plan_steps") if requirement == "plan_or_operation" else (
                "facts", "fact", "requirements", "test", "table", "table_cell",
                "evidence", "entity", "supporting_fact", "evidence_link", "choice",
            )
            return tuple(
                (node, f"missing {requirement}: request typed input core")
                for node in self._latest_sender_nodes(types)
            )
        return ()

    def _quality_candidates(self, requirement: str) -> tuple[tuple[str, tuple[str, ...], str], ...]:
        fragment_names = _quality_fragment_names(requirement)
        if not fragment_names:
            return ()
        candidates: list[tuple[str, tuple[str, ...], str]] = []
        for node in self._latest_sender_nodes(("verification", "calculation", "result", "final_answer", "execution", "test_result", "plan", "plan_steps")):
            fragments = tuple(
                fragment.fragment_id
                for fragment in fragments_for_node(
                    node,
                    sender=self.sender,
                    receiver=self.receiver,
                    node_scope="MANDATORY_STAGE",
                )
                if fragment.scope == OPTIONAL_FRAGMENT and fragment.name in fragment_names
                and fragment.fragment_id not in self.receiver_view.visible_fragment_ids
            )
            if fragments:
                candidates.append((node.node_id, fragments, f"quality request {requirement}: request optional fragment"))
        return tuple(candidates)

    def _core_fragments(self, node: NodeVersion) -> tuple[str, ...]:
        return tuple(
            fragment.fragment_id
            for fragment in fragments_for_node(
                node,
                sender=self.sender,
                receiver=self.receiver,
                node_scope="MANDATORY_STAGE",
            )
            if fragment.scope == MANDATORY_FRAGMENT and fragment.fragment_id not in self.receiver_view.visible_fragment_ids
        )

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


def plan_incremental_request(
    state: GraphState,
    *,
    sender: str,
    receiver: str,
    sender_view: AgentGraphView,
    receiver_view: AgentGraphView,
    receiver_need: ReceiverNeed,
) -> IncrementalRequestPlan:
    return IncrementalRequestPlanner(
        state,
        sender=sender,
        receiver=receiver,
        sender_view=sender_view,
        receiver_view=receiver_view,
    ).plan_need(receiver_need)


def _quality_fragment_names(requirement: str) -> tuple[str, ...]:
    mapping = {
        "full_feedback": ("full_feedback",),
        "critic_confidence_low": ("full_feedback", "repair_hint", "error_type", "error_location"),
        "insufficient_derivation": ("calculation_trace", "full_plan", "result_metadata", "execution_detail"),
        "result_metadata": ("result_metadata",),
        "calculation_trace": ("calculation_trace",),
        "full_plan": ("full_plan", "rationale"),
        "execution_detail": ("execution_detail",),
    }
    return mapping.get(requirement, (requirement,))


def _quality_utility(requirement: str) -> float:
    utilities = {
        "critic_confidence_low": 0.9,
        "insufficient_derivation": 0.8,
        "calculation_trace": 0.7,
        "execution_detail": 0.7,
        "repair_hint": 0.6,
        "error_location": 0.5,
        "error_type": 0.45,
        "full_plan": 0.4,
        "result_metadata": 0.3,
        "full_feedback": 0.25,
    }
    return utilities.get(requirement, 0.2)


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
