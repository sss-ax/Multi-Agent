"""Field-level semantic fragments for communication and rendering.

The canonical graph remains node-versioned.  This module gives the
communication layer a finer action space by splitting a node into stable,
auditable fragments.  Core fragments are sufficient for downstream contracts;
refinement fragments are optional context that policies may drop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from .delta_scoring import estimate_node_tokens
from .models import NodeVersion


MANDATORY_FRAGMENT = "MANDATORY_STAGE"
OPTIONAL_FRAGMENT = "OPTIONAL_POLICY"


@dataclass(frozen=True)
class SemanticFragment:
    fragment_id: str
    node_id: str
    name: str
    scope: str
    content: Any
    token_cost: int


def fragment_id(node_id: str, name: str) -> str:
    return f"{node_id}#{name}"


def node_id_from_fragment(fragment: str) -> str:
    return str(fragment).split("#", 1)[0]


def fragments_for_node(
    node: NodeVersion,
    *,
    sender: str = "",
    receiver: str = "",
    node_scope: str = "OPTIONAL_POLICY",
) -> tuple[SemanticFragment, ...]:
    """Return stable fragments for a node under one stage communication scope."""
    if node_scope == "LOCAL":
        return ()
    specs = _fragment_specs(node, sender=sender, receiver=receiver, node_scope=node_scope)
    fragments: list[SemanticFragment] = []
    for name, scope, content in specs:
        if content in (None, "", [], {}):
            continue
        text = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        fragments.append(SemanticFragment(
            fragment_id=fragment_id(node.node_id, name),
            node_id=node.node_id,
            name=name,
            scope=scope,
            content=content,
            token_cost=max(1, len(text.split())),
        ))
    if not fragments:
        scope = MANDATORY_FRAGMENT if node_scope == "MANDATORY_STAGE" else OPTIONAL_FRAGMENT
        fragments.append(SemanticFragment(
            fragment_id=fragment_id(node.node_id, "content"),
            node_id=node.node_id,
            name="content",
            scope=scope,
            content=node.content,
            token_cost=estimate_node_tokens(node),
        ))
    return tuple(fragments)


def mandatory_fragment_ids(
    node: NodeVersion,
    *,
    sender: str = "",
    receiver: str = "",
    node_scope: str = "OPTIONAL_POLICY",
) -> tuple[str, ...]:
    return tuple(
        fragment.fragment_id
        for fragment in fragments_for_node(node, sender=sender, receiver=receiver, node_scope=node_scope)
        if fragment.scope == MANDATORY_FRAGMENT
    )


def optional_fragment_ids(
    node: NodeVersion,
    *,
    sender: str = "",
    receiver: str = "",
    node_scope: str = "OPTIONAL_POLICY",
) -> tuple[str, ...]:
    return tuple(
        fragment.fragment_id
        for fragment in fragments_for_node(node, sender=sender, receiver=receiver, node_scope=node_scope)
        if fragment.scope == OPTIONAL_FRAGMENT
    )


def all_fragment_ids(
    node: NodeVersion,
    *,
    sender: str = "",
    receiver: str = "",
    node_scope: str = "OPTIONAL_POLICY",
) -> tuple[str, ...]:
    return tuple(
        fragment.fragment_id
        for fragment in fragments_for_node(node, sender=sender, receiver=receiver, node_scope=node_scope)
    )


def render_fragmented_content(node: NodeVersion, visible_fragment_ids: Iterable[str]) -> Any:
    """Return only visible fields for a node.

    An empty fragment set means the caller is using legacy node-level
    visibility and should receive the full content.
    """
    visible = {str(item) for item in visible_fragment_ids}
    if not visible:
        return node.content
    all_fragments = _all_render_fragments(node)
    fragments = [fragment for fragment in all_fragments if fragment.fragment_id in visible]
    if not fragments:
        return {}
    return {
        fragment.name: fragment.content
        for fragment in fragments
    }


def fragment_token_cost(fragment_ids: Iterable[str], nodes: dict[str, NodeVersion]) -> int:
    total = 0
    by_node: dict[str, set[str]] = {}
    for item in fragment_ids:
        by_node.setdefault(node_id_from_fragment(str(item)), set()).add(str(item))
    for node_id, ids in by_node.items():
        node = nodes.get(node_id)
        if node is None:
            continue
        for fragment in _all_render_fragments(node):
            if fragment.fragment_id in ids:
                total += fragment.token_cost
    return total


def _fragment_specs(
    node: NodeVersion,
    *,
    sender: str,
    receiver: str,
    node_scope: str,
) -> list[tuple[str, str, Any]]:
    if node_scope != "MANDATORY_STAGE":
        return [("content", OPTIONAL_FRAGMENT, node.content)]

    content = node.content
    if node.type in {"result", "final_answer"}:
        value = content.get("value") if isinstance(content, dict) and "value" in content else content
        extras = {k: v for k, v in content.items() if k != "value"} if isinstance(content, dict) else None
        return [
            ("final_value", MANDATORY_FRAGMENT, value),
            ("result_metadata", OPTIONAL_FRAGMENT, extras),
        ]
    if node.type == "calculation":
        if isinstance(content, dict):
            return [
                ("final_value", MANDATORY_FRAGMENT, content.get("value")),
                ("key_operation", MANDATORY_FRAGMENT, content.get("expression")),
                ("calculation_trace", OPTIONAL_FRAGMENT, content),
            ]
        return [("calculation_core", MANDATORY_FRAGMENT, content)]
    if node.type == "code":
        return [
            ("code_body", MANDATORY_FRAGMENT, content),
        ]
    if node.type == "verification":
        verdict = content.get("status") if isinstance(content, dict) else content
        error_type = content.get("error_type") if isinstance(content, dict) else None
        error_location = content.get("error_location") if isinstance(content, dict) else None
        repair_hint = content.get("repair_hint") if isinstance(content, dict) else None
        return [
            ("verdict", MANDATORY_FRAGMENT, verdict),
            ("error_type", OPTIONAL_FRAGMENT, error_type),
            ("error_location", OPTIONAL_FRAGMENT, error_location),
            ("repair_hint", OPTIONAL_FRAGMENT, repair_hint),
            ("full_feedback", OPTIONAL_FRAGMENT, content),
        ]
    if node.type in {"plan", "plan_steps"}:
        if isinstance(content, list):
            first = content[0] if content else None
            return [
                ("next_action", MANDATORY_FRAGMENT, first),
                ("full_plan", OPTIONAL_FRAGMENT, content),
            ]
        if isinstance(content, dict):
            steps = content.get("steps")
            first = steps[0] if isinstance(steps, list) and steps else content
            return [
                ("next_action", MANDATORY_FRAGMENT, first),
                ("dependencies", MANDATORY_FRAGMENT, content.get("inputs")),
                ("full_plan", OPTIONAL_FRAGMENT, content),
                ("rationale", OPTIONAL_FRAGMENT, content.get("rationale")),
            ]
        return [("next_action", MANDATORY_FRAGMENT, content)]
    if node.type in {"facts", "fact", "requirements", "test", "table", "table_cell", "evidence", "entity", "supporting_fact", "evidence_link", "choice"}:
        return [("task_input", MANDATORY_FRAGMENT, content)]
    if node.type in {"execution", "test_result", "tool_result"}:
        return [
            ("execution_status", MANDATORY_FRAGMENT, _status_like(content)),
            ("execution_detail", OPTIONAL_FRAGMENT, content),
        ]
    return [("content", MANDATORY_FRAGMENT, content)]


def _all_render_fragments(node: NodeVersion) -> tuple[SemanticFragment, ...]:
    fragments = [
        *fragments_for_node(node, node_scope="MANDATORY_STAGE"),
        *fragments_for_node(node, node_scope="OPTIONAL_POLICY"),
    ]
    return tuple(dict((fragment.fragment_id, fragment) for fragment in fragments).values())


def _status_like(content: Any) -> Any:
    if not isinstance(content, dict):
        return content
    for key in ("success", "ok", "tests_passed", "status"):
        if key in content:
            return {key: content[key]}
    value = content.get("value")
    if isinstance(value, dict):
        for key in ("success", "ok", "tests_passed", "status"):
            if key in value:
                return {key: value[key]}
    return content
