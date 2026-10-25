"""Typed, previewable actions for learned communication and acquisition policies."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol

from .agent_graph_view import AgentGraphViewManager
from .communication_replay import CommunicationDecisionSnapshot, graph_state_to_dict
from .context_slicer import build_context_slice, render_compact_context_slice
from .delta_closure import dependency_closure
from .delta_scoring import estimate_edge_tokens
from .graph_store import GraphStore
from .semantic_fragments import (
    MANDATORY_FRAGMENT,
    fragment_token_cost,
    fragments_for_node,
    node_id_from_fragment,
)


class CommunicationActionType(str, Enum):
    STOP = "STOP"
    SEND = "SEND"
    REQUEST = "REQUEST"
    RETRIEVE = "RETRIEVE"
    REVISE = "REVISE"


@dataclass(frozen=True)
class CommunicationAction:
    action_type: CommunicationActionType
    selected_fragment_ids: tuple[str, ...] = ()
    requested_semantics: tuple[str, ...] = ()
    query: str = ""
    target: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "selected_fragment_ids": list(self.selected_fragment_ids),
            "requested_semantics": list(self.requested_semantics),
            "query": self.query,
            "target": self.target,
            "payload": copy.deepcopy(self.payload),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CommunicationAction":
        return cls(
            action_type=CommunicationActionType(str(payload.get("action_type", ""))),
            selected_fragment_ids=tuple(str(item) for item in payload.get("selected_fragment_ids", ())),
            requested_semantics=tuple(str(item) for item in payload.get("requested_semantics", ())),
            query=str(payload.get("query", "")),
            target=str(payload.get("target", "")),
            payload=copy.deepcopy(dict(payload.get("payload", {}))),
        )


@dataclass(frozen=True)
class CommunicationPreview:
    action: CommunicationAction
    legal: bool
    reason: str = ""
    root_node_ids: tuple[str, ...] = ()
    closure_node_ids: tuple[str, ...] = ()
    closure_edge_ids: tuple[str, ...] = ()
    delivered_fragment_ids: tuple[str, ...] = ()
    communication_tokens: int = 0
    rendered_context_tokens_before: int = 0
    rendered_context_tokens_after: int = 0
    rendered_context_increment_tokens: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.as_dict(),
            "legal": self.legal,
            "reason": self.reason,
            "root_node_ids": list(self.root_node_ids),
            "closure_node_ids": list(self.closure_node_ids),
            "closure_edge_ids": list(self.closure_edge_ids),
            "delivered_fragment_ids": list(self.delivered_fragment_ids),
            "communication_tokens": self.communication_tokens,
            "rendered_context_tokens_before": self.rendered_context_tokens_before,
            "rendered_context_tokens_after": self.rendered_context_tokens_after,
            "rendered_context_increment_tokens": self.rendered_context_increment_tokens,
        }


class InformationActionAdapter(Protocol):
    """Environment-owned implementation for REQUEST/RETRIEVE/REVISE."""

    def preview(
        self,
        snapshot: CommunicationDecisionSnapshot,
        action: CommunicationAction,
    ) -> CommunicationPreview: ...

    def apply(
        self,
        snapshot: CommunicationDecisionSnapshot,
        action: CommunicationAction,
        *,
        store: GraphStore,
        views: AgentGraphViewManager,
    ) -> CommunicationPreview: ...


class CommunicationActionExecutor:
    """Preview and apply actions without allowing oracle-dependent mutations."""

    def __init__(
        self,
        *,
        token_counter: Callable[[str], int] | None = None,
        information_adapter: InformationActionAdapter | None = None,
    ) -> None:
        self.token_counter = token_counter or (lambda text: max(0, len(text.split())))
        self.information_adapter = information_adapter

    def preview(
        self,
        snapshot: CommunicationDecisionSnapshot,
        action: CommunicationAction,
    ) -> CommunicationPreview:
        leaked = _oracle_action_fields(action.payload)
        if leaked:
            return CommunicationPreview(
                action=action,
                legal=False,
                reason=f"action payload contains oracle-only fields: {sorted(leaked)}",
            )
        if action.action_type in {
            CommunicationActionType.REQUEST,
            CommunicationActionType.RETRIEVE,
            CommunicationActionType.REVISE,
        }:
            if self.information_adapter is None:
                return CommunicationPreview(
                    action=action,
                    legal=False,
                    reason=f"{action.action_type.value} requires an InformationActionAdapter",
                )
            return self.information_adapter.preview(snapshot, action)
        if action.action_type == CommunicationActionType.STOP:
            if action.selected_fragment_ids:
                return CommunicationPreview(action=action, legal=False, reason="STOP cannot select fragments")
            visible_fragments = set(
                snapshot.agent_views.get("views", {})
                .get(snapshot.receiver, {})
                .get("visible_fragment_ids", ())
            )
            missing_mandatory = sorted(
                str(fragment.get("fragment_id", ""))
                for fragment in snapshot.fragments
                if str(fragment.get("scope", "")) == MANDATORY_FRAGMENT
                and str(fragment.get("fragment_id", "")) not in visible_fragments
            )
            if missing_mandatory:
                return CommunicationPreview(
                    action=action,
                    legal=False,
                    reason=f"STOP would suppress mandatory fragments: {missing_mandatory}",
                )
            before = self._rendered_context_tokens(snapshot)
            return CommunicationPreview(
                action=action,
                legal=True,
                reason="no communication",
                rendered_context_tokens_before=before,
                rendered_context_tokens_after=before,
            )
        if action.action_type != CommunicationActionType.SEND:
            return CommunicationPreview(action=action, legal=False, reason="unsupported action type")
        return self._preview_send(snapshot, action)

    def apply(
        self,
        snapshot: CommunicationDecisionSnapshot,
        action: CommunicationAction,
        *,
        store: GraphStore,
        views: AgentGraphViewManager,
    ) -> CommunicationPreview:
        if (
            graph_state_to_dict(store.snapshot()) != graph_state_to_dict(snapshot.graph_state)
            or views.snapshot() != snapshot.agent_views
        ):
            return CommunicationPreview(
                action=action,
                legal=False,
                reason="live runtime does not match the pre-action snapshot",
            )
        if action.action_type in {
            CommunicationActionType.REQUEST,
            CommunicationActionType.RETRIEVE,
            CommunicationActionType.REVISE,
        }:
            information_preview = self.preview(snapshot, action)
            if not information_preview.legal:
                return information_preview
            assert self.information_adapter is not None
            return self.information_adapter.apply(snapshot, action, store=store, views=views)
        preview = self.preview(snapshot, action)
        if not preview.legal or action.action_type == CommunicationActionType.STOP:
            return preview
        views.grant(
            snapshot.receiver,
            node_ids=preview.closure_node_ids,
            edge_ids=preview.closure_edge_ids,
            fragment_ids=preview.delivered_fragment_ids,
            delta_id=f"replay:{snapshot.snapshot_id}:{action.action_type.value}",
            packet_id=f"replay-packet:{snapshot.snapshot_id}",
        )
        return preview

    def _preview_send(
        self,
        snapshot: CommunicationDecisionSnapshot,
        action: CommunicationAction,
    ) -> CommunicationPreview:
        fragment_by_id = {
            str(fragment.get("fragment_id", "")): fragment
            for fragment in snapshot.fragments
            if str(fragment.get("fragment_id", ""))
        }
        selected = tuple(dict.fromkeys(action.selected_fragment_ids))
        unknown = sorted(set(selected) - set(fragment_by_id))
        if unknown:
            return CommunicationPreview(action=action, legal=False, reason=f"unknown fragments: {unknown}")
        non_optional = sorted(
            fragment_id for fragment_id in selected
            if str(fragment_by_id[fragment_id].get("scope", "")) != "OPTIONAL_POLICY"
        )
        if non_optional:
            return CommunicationPreview(
                action=action,
                legal=False,
                reason=f"policy may select only optional fragments: {non_optional}",
            )
        mandatory = tuple(
            fragment_id for fragment_id, fragment in fragment_by_id.items()
            if str(fragment.get("scope", "")) == MANDATORY_FRAGMENT
        )
        delivered_roots = tuple(dict.fromkeys((*mandatory, *selected)))
        root_node_ids = tuple(dict.fromkeys(node_id_from_fragment(item) for item in delivered_roots))

        store, views = self._isolated_runtime(snapshot)
        receiver_view = views.view(snapshot.receiver)
        before = self._rendered_context_tokens(snapshot, store=store, views=views)
        delta = dependency_closure(
            store.snapshot(),
            root_node_ids=root_node_ids,
            sender=snapshot.sender,
            receiver=snapshot.receiver,
            receiver_view=receiver_view,
            policy=f"replay:{snapshot.policy_name}",
        )
        delivered = list(
            fragment_id for fragment_id in delivered_roots
            if node_id_from_fragment(fragment_id) in set(delta.node_ids)
        )
        closure_added = set(delta.closure_added_node_ids)
        graph = store.snapshot()
        for node_id in sorted(closure_added):
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            delivered.extend(
                fragment.fragment_id
                for fragment in fragments_for_node(
                    node,
                    sender=snapshot.sender,
                    receiver=snapshot.receiver,
                    node_scope="MANDATORY_STAGE",
                )
                if fragment.scope == MANDATORY_FRAGMENT
            )
        delivered_fragment_ids = tuple(dict.fromkeys(delivered))
        communication_tokens = fragment_token_cost(delivered_fragment_ids, graph.nodes)
        communication_tokens += sum(
            estimate_edge_tokens(edge)
            for edge in graph.edges
            if edge.edge_id in set(delta.edge_ids)
        )
        budget = snapshot.remaining_budget
        if budget is not None and communication_tokens > budget:
            return CommunicationPreview(
                action=action,
                legal=False,
                reason=f"closure-rendered cost {communication_tokens} exceeds budget {budget}",
                root_node_ids=root_node_ids,
                closure_node_ids=delta.node_ids,
                closure_edge_ids=delta.edge_ids,
                delivered_fragment_ids=delivered_fragment_ids,
                communication_tokens=communication_tokens,
                rendered_context_tokens_before=before,
            )
        views.grant(
            snapshot.receiver,
            node_ids=delta.node_ids,
            edge_ids=delta.edge_ids,
            fragment_ids=delivered_fragment_ids,
        )
        after = self._rendered_context_tokens(snapshot, store=store, views=views)
        return CommunicationPreview(
            action=action,
            legal=True,
            reason="action satisfies protocol and closure-rendered budget",
            root_node_ids=root_node_ids,
            closure_node_ids=delta.node_ids,
            closure_edge_ids=delta.edge_ids,
            delivered_fragment_ids=delivered_fragment_ids,
            communication_tokens=communication_tokens,
            rendered_context_tokens_before=before,
            rendered_context_tokens_after=after,
            rendered_context_increment_tokens=max(0, after - before),
        )

    @staticmethod
    def _isolated_runtime(
        snapshot: CommunicationDecisionSnapshot,
    ) -> tuple[GraphStore, AgentGraphViewManager]:
        store = GraphStore(snapshot.graph_state)
        raw_views = snapshot.agent_views.get("views", {})
        views = AgentGraphViewManager(
            store,
            task_id=snapshot.task_id,
            branch_id=snapshot.branch_id,
            agents=tuple(raw_views),
        )
        views.restore(snapshot.agent_views)
        return store, views

    def _rendered_context_tokens(
        self,
        snapshot: CommunicationDecisionSnapshot,
        *,
        store: GraphStore | None = None,
        views: AgentGraphViewManager | None = None,
    ) -> int:
        if store is None or views is None:
            store, views = self._isolated_runtime(snapshot)
        receiver_view = views.view(snapshot.receiver)
        visible_store = views.visible_store(snapshot.receiver)
        context = build_context_slice(
            visible_store,
            task_id=snapshot.task_id,
            branch_id=snapshot.branch_id,
            role=snapshot.receiver,
            policy=(
                "planner_state" if snapshot.receiver == "planner"
                else "solver_state" if snapshot.receiver == "solver"
                else "dependency_closure"
            ),
            allow_missing=True,
            visible_fragment_ids=receiver_view.visible_fragment_ids,
        )
        rendered = render_compact_context_slice(context, visible_store.snapshot())
        return int(self.token_counter(rendered))


def _oracle_action_fields(payload: Mapping[str, Any]) -> set[str]:
    forbidden = {
        "gold_answer", "gold_label", "hidden_test_result", "hidden_tests",
        "oracle_correct", "reference_answer", "final_task_success", "reward",
    }
    found: set[str] = set()

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                name = str(key).casefold()
                child = f"{path}.{name}" if path else name
                if name in forbidden:
                    found.add(child)
                visit(item, child)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(payload, "")
    return found
