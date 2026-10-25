"""Model-facing batch IR for semantic graph mutations."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class GroundingRef:
    source: Any
    quote: str = ""
    fields: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Any) -> "GroundingRef":
        if isinstance(payload, Mapping):
            raw_fields = payload.get("fields", ())
            return cls(
                source=copy.deepcopy(payload.get("source", payload.get("source_ref", ""))),
                quote=str(payload.get("quote", "")),
                fields=tuple(str(item) for item in raw_fields) if isinstance(raw_fields, (list, tuple)) else (),
            )
        return cls(source=copy.deepcopy(payload))

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": copy.deepcopy(self.source),
            "quote": self.quote,
            "fields": list(self.fields),
        }


@dataclass(frozen=True)
class ProposedNode:
    temp_id: str
    node_type: str
    content: Any
    grounding: tuple[GroundingRef, ...] = ()

    @classmethod
    def from_payload(cls, payload: Any) -> "ProposedNode":
        if not isinstance(payload, Mapping):
            raise ValueError("Semantic Delta node must be an object")
        grounding = payload.get("grounding", ())
        if not isinstance(grounding, (list, tuple)):
            raise ValueError("Semantic Delta node grounding must be an array")
        return cls(
            temp_id=str(payload.get("temp_id", payload.get("id", ""))),
            node_type=str(payload.get("type", payload.get("node_type", ""))),
            content=copy.deepcopy(payload.get("content")),
            grounding=tuple(GroundingRef.from_payload(item) for item in grounding),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "temp_id": self.temp_id,
            "type": self.node_type,
            "content": copy.deepcopy(self.content),
            "grounding": [item.as_dict() for item in self.grounding],
        }


@dataclass(frozen=True)
class ProposedEdge:
    source: str
    relation: str
    target: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> "ProposedEdge":
        if not isinstance(payload, Mapping):
            raise ValueError("Semantic Delta edge must be an object")
        return cls(
            source=str(payload.get("source", "")),
            relation=str(payload.get("relation", "")),
            target=str(payload.get("target", "")),
            metadata=copy.deepcopy(dict(payload.get("metadata", {}))) if isinstance(payload.get("metadata", {}), Mapping) else {},
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "metadata": copy.deepcopy(dict(self.metadata)),
        }


@dataclass(frozen=True)
class SendHint:
    receiver: str
    fragments: tuple[str, ...] = ()
    reason: str = ""

    @classmethod
    def from_payload(cls, payload: Any) -> "SendHint":
        if not isinstance(payload, Mapping):
            raise ValueError("Semantic Delta send_hint must be an object")
        fragments = payload.get("fragments", payload.get("fragment_ids", ()))
        if not isinstance(fragments, (list, tuple)):
            raise ValueError("Semantic Delta send_hint fragments must be an array")
        return cls(
            receiver=str(payload.get("receiver", "")),
            fragments=tuple(str(item) for item in fragments),
            reason=str(payload.get("reason", "")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {"receiver": self.receiver, "fragments": list(self.fragments), "reason": self.reason}


@dataclass(frozen=True)
class SemanticDelta:
    nodes: tuple[ProposedNode, ...] = ()
    edges: tuple[ProposedEdge, ...] = ()
    send_hints: tuple[SendHint, ...] = ()
    complete: bool = False

    @classmethod
    def from_payload(cls, payload: Any) -> "SemanticDelta":
        if not isinstance(payload, Mapping):
            raise ValueError("Semantic Delta payload must be an object")
        raw_nodes = payload.get("nodes", ())
        raw_edges = payload.get("edges", ())
        raw_hints = payload.get("send_hints", payload.get("sendHints", ()))
        if not all(isinstance(value, (list, tuple)) for value in (raw_nodes, raw_edges, raw_hints)):
            raise ValueError("Semantic Delta nodes, edges, and send_hints must be arrays")
        return cls(
            nodes=tuple(ProposedNode.from_payload(item) for item in raw_nodes),
            edges=tuple(ProposedEdge.from_payload(item) for item in raw_edges),
            send_hints=tuple(SendHint.from_payload(item) for item in raw_hints),
            complete=bool(payload.get("complete", False)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": [item.as_dict() for item in self.nodes],
            "edges": [item.as_dict() for item in self.edges],
            "send_hints": [item.as_dict() for item in self.send_hints],
            "complete": self.complete,
        }
