"""Unified HotpotQA semantic verification helpers.

The verifier is intentionally split into two layers:

1. Programmatic structure checks: citation resolution, minimum hop count, and
   bridge/evidence connectivity.
2. Constrained semantic entailment contract: the result is only finally trusted
   when a downstream critic/NLI verdict explicitly supports the answer.  In the
   runtime verifier this second layer is represented as a required delegated
   check, preserving the old critic contract while making the boundary explicit.
"""

from __future__ import annotations

import re
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .context_slicer import _node_sort_key
from .hotpot_bridge import bridge_candidates_from_values, normalize_bridge


@dataclass(frozen=True)
class HotpotVerificationSignal:
    status: str
    checks: tuple[str, ...]
    confidence: float
    error_type: str = ""
    error_location: str = ""
    reason: str = ""
    repair_instruction: str = ""
    missing_evidence: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class HotpotCitationResolver:
    """Resolve local/model citation aliases to canonical supporting fact ids."""

    def __init__(self, facts: list[Any], *, visible_nodes: list[Any] | None = None):
        self.facts = sorted(facts, key=_node_sort_key)
        self.fact_index: dict[str, Any] = {}
        self.alias_to_canonical: dict[str, str] = {}
        local_index_by_node = {
            node.node_id: f"n{index}"
            for index, node in enumerate(sorted(visible_nodes or self.facts, key=_node_sort_key), start=1)
        }
        for index, node in enumerate(self.facts, start=1):
            canonical = str(getattr(node, "logical_id", "")).strip()
            if not canonical:
                continue
            self.fact_index[canonical] = node
            for alias in self._aliases(node, index, local_index_by_node.get(node.node_id, f"n{index}")):
                cleaned = self._clean(alias)
                if cleaned:
                    self.alias_to_canonical.setdefault(cleaned, canonical)
            self.alias_to_canonical.setdefault(self._clean(canonical), canonical)

    def resolve_all(self, citations: list[str]) -> list[str | None]:
        return [self.resolve(citation) for citation in citations]

    def resolve(self, citation: Any) -> str | None:
        text = str(citation or "").strip()
        for candidate in self._candidate_forms(text):
            resolved = self.alias_to_canonical.get(self._clean(candidate))
            if resolved is not None:
                return resolved
        return None

    def _candidate_forms(self, text: str) -> list[str]:
        forms = [text]
        if text.startswith("title:"):
            forms.append(text.removeprefix("title:"))
        forms.append(re.sub(r":sent_id:", ":", text))
        forms.append(re.sub(r":sent(?:ence)?_?", ":", text))
        forms.append(re.sub(r"^fact(\d+)", r"supporting_fact_\1", text))
        forms.append(re.sub(r"^f(\d+)$", r"supporting_fact_\1", text))
        forms.append(re.sub(r"^n(\d+)(?::.*)?$", r"n\1", text))
        forms.append(re.sub(r"^title:(.*):sent_id:(\d+)$", r"\1:\2", text))
        forms.append(re.sub(r"^title:(.*):sent(\d+)$", r"\1:\2", text))
        return list(dict.fromkeys(forms))

    def _aliases(self, node: Any, index: int, local_node_id: str) -> set[str]:
        content = _json_value(_node_content(node))
        keys = {
            str(getattr(node, "logical_id", "")).strip(),
            str(getattr(node, "node_id", "")).strip(),
            local_node_id,
        }
        logical_id = str(getattr(node, "logical_id", "")).strip()
        match = re.fullmatch(r"supporting_fact_(\d+)", logical_id)
        if match:
            keys.add(f"f{match.group(1)}")
            keys.add(f"fact{match.group(1)}")
        if isinstance(content, dict):
            for key in ("id", "fact_id", "logical_id"):
                if content.get(key):
                    keys.add(str(content[key]).strip())
            title = str(content.get("title", content.get("entity", ""))).strip()
            sent_id = content.get("sent_id")
            if title:
                keys.add(title)
                keys.add(f"title:{title}")
                if sent_id is not None:
                    keys.update({
                        f"{title}:{sent_id}",
                        f"{title}#{sent_id}",
                        f"title:{title}:{sent_id}",
                        f"title:{title}:sent_id:{sent_id}",
                        f"{title}:sent_id:{sent_id}",
                        f"{title}:sent{sent_id}",
                        f"{local_node_id}:{sent_id}",
                        f"{local_node_id}:sent_id:{sent_id}",
                        f"{local_node_id}:sent{sent_id}",
                    })
        return keys

    @staticmethod
    def _clean(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())


class HotpotSemanticVerifier:
    def __init__(self, state_nodes: list[Any]):
        self.state_nodes = sorted(state_nodes, key=_node_sort_key)
        self.facts = [
            node for node in self.state_nodes
            if getattr(node, "type", "") in {"supporting_fact", "evidence", "evidence_link"}
        ]
        self.resolver = HotpotCitationResolver(self.facts, visible_nodes=self.state_nodes)
        self.bridge_candidates = self._bridge_candidates()

    def verify(self, payload: Any) -> HotpotVerificationSignal:
        if not self.facts:
            return HotpotVerificationSignal(
                status="need_fix",
                checks=("missing_evidence",),
                confidence=1.0,
                error_type="evidence_insufficiency",
                error_location="supporting evidence",
                reason="No supporting evidence is visible for the answer.",
                repair_instruction="Request or use supporting evidence before finalizing the answer.",
                missing_evidence=("supporting_fact",),
            )
        value = _json_value(payload)
        if isinstance(value, dict) and "value" in value:
            value = _json_value(value.get("value"))
        answer = _structured_answer(value)
        cited_ids = _string_list(value.get("cited_fact_ids") if isinstance(value, dict) else None)
        bridge = str(value.get("bridge_entity", "")).strip() if isinstance(value, dict) else ""
        chain = _string_list(value.get("reasoning_chain") if isinstance(value, dict) else None)
        missing = self._required_missing(value, answer=answer, cited_ids=cited_ids, bridge=bridge, chain=chain)
        if missing:
            return self._need_fix(
                checks=("evidence_present", "missing_structured_citations"),
                error_type="missing_evidence_chain",
                error_location="result",
                reason="Evidence exists, but the candidate is missing the required structured evidence chain.",
                repair_instruction=(
                    "Return a structured set_result.value with answer, bridge_entity, "
                    "cited_fact_ids, and reasoning_chain grounded in visible supporting facts."
                ),
                missing=missing,
                packet={
                    "status": "need_fix",
                    "error_type": "missing_evidence_chain",
                    "known": {"answer": answer},
                    "required": ["answer", "bridge_entity", "cited_fact_ids", "reasoning_chain"],
                    "preserve": ["question", "visible evidence"],
                },
            )
        resolved = self.resolver.resolve_all(cited_ids)
        unresolved = [fact_id for fact_id, canonical in zip(cited_ids, resolved) if canonical is None]
        if unresolved:
            return self._need_fix(
                checks=("invalid_citation",),
                error_type="invalid_citation",
                error_location="cited_fact_ids",
                reason=f"Candidate cites unknown supporting facts: {', '.join(unresolved)}.",
                repair_instruction="Replace invalid citations with visible supporting_fact ids, titles, or title:sent_id references.",
                missing=unresolved,
                packet={
                    "status": "need_fix",
                    "error_type": "invalid_citation",
                    "known": {"answer": answer, "bridge_entity": bridge, "invalid_citations": unresolved},
                    "required": ["visible cited_fact_ids"],
                    "preserve": ["answer if still entailed", "visible evidence"],
                },
            )
        canonical_ids = [item for item in resolved if item is not None]
        if len(canonical_ids) < 2:
            return self._need_fix(
                checks=("citation_valid", "missing_second_hop"),
                error_type="missing_hop",
                error_location="reasoning_chain",
                reason="A multihop answer should cite at least two connected facts.",
                repair_instruction="Preserve useful valid citations, then add one visible supporting_fact for the missing hop.",
                missing=("second_hop",),
                packet={
                    "status": "need_fix",
                    "error_type": "missing_hop",
                    "known": {"answer": answer, "bridge_entity": bridge, "valid_cited_fact_ids": canonical_ids},
                    "required": [f"find one fact that connects {bridge or 'the bridge entity'} to {answer or 'the answer'}"],
                    "repair_target": "cited_fact_ids",
                    "must_change": {"cited_fact_ids": True},
                    "bridge_candidates": self.bridge_candidates,
                    "preserve": ["valid citations"],
                },
            )
        cited_text = " ".join(_fact_text(self.resolver.fact_index[fact_id]) for fact_id in canonical_ids)
        if not _contains_normalized(cited_text, bridge):
            return self._need_fix(
                checks=("citation_valid", "incomplete_chain"),
                error_type="wrong_bridge_entity",
                error_location="bridge_entity",
                reason="The cited facts do not ground the claimed bridge entity.",
                repair_instruction="Choose the bridge entity supported by cited facts, or replace the citations with facts that support the claimed bridge.",
                missing=("bridge_entity",),
                packet={
                    "status": "need_fix",
                    "error_type": "wrong_bridge_entity",
                    "known": {"answer": answer, "claimed_bridge_entity": bridge, "valid_cited_fact_ids": canonical_ids},
                    "required": ["select one bridge entity from bridge_candidates that is grounded in cited facts"],
                    "repair_target": "bridge_entity",
                    "must_change": {"bridge_entity": True},
                    "forbidden_values": {"bridge_entity": [bridge]},
                    "bridge_candidates": self.bridge_candidates,
                    "preserve": ["valid citations only if they support the corrected bridge"],
                },
            )
        return HotpotVerificationSignal(
            status="verified",
            checks=("citation_valid", "chain_complete", "semantic_entailment_delegated_to_critic"),
            confidence=0.0,
            metadata={
                "answer": answer,
                "bridge_entity": bridge,
                "cited_fact_ids": canonical_ids,
                "raw_cited_fact_ids": cited_ids,
                "semantic_entailment_required_from": "critic",
                "programmatic_verifier": "HotpotSemanticVerifier",
            },
        )

    def _required_missing(self, value: Any, *, answer: str, cited_ids: list[str], bridge: str, chain: list[str]) -> tuple[str, ...]:
        if not isinstance(value, dict):
            return ("answer", "bridge_entity", "cited_fact_ids", "reasoning_chain")
        missing = [
            key for key in ("answer", "bridge_entity", "cited_fact_ids", "reasoning_chain")
            if key not in value
        ]
        if not answer:
            missing.append("answer_value")
        if not cited_ids:
            missing.append("cited_fact_ids_value")
        if not bridge:
            missing.append("bridge_entity_value")
        if not chain:
            missing.append("reasoning_chain_value")
        return tuple(dict.fromkeys(missing))

    def _need_fix(
        self,
        *,
        checks: tuple[str, ...],
        error_type: str,
        error_location: str,
        reason: str,
        repair_instruction: str,
        missing: Iterable[str],
        packet: dict[str, Any],
    ) -> HotpotVerificationSignal:
        return HotpotVerificationSignal(
            status="need_fix",
            checks=checks,
            confidence=1.0,
            error_type=error_type,
            error_location=error_location,
            reason=reason,
            repair_instruction="Typed repair request: status=need_fix error_type=" + error_type + ". " + repair_instruction,
            missing_evidence=tuple(dict.fromkeys(str(item) for item in missing)),
            metadata={"repair_packet": packet},
        )

    def _bridge_candidates(self, *, limit: int = 24) -> list[str]:
        values: list[Any] = []
        for node in self.state_nodes:
            content = _json_value(_node_content(node))
            if getattr(node, "type", "") == "entity":
                if isinstance(content, dict):
                    values.extend(content.get(key) for key in ("name", "entity", "title", "value"))
                else:
                    values.append(content)
            elif getattr(node, "type", "") in {"supporting_fact", "evidence", "evidence_link"} and isinstance(content, dict):
                values.extend(content.get(key) for key in ("title", "entity"))
            if len(values) >= limit * 2:
                break
        return bridge_candidates_from_values(values, limit=limit)


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _node_content(node: Any) -> Any:
    return getattr(node, "content", None)


def _structured_answer(value: Any) -> str:
    value = _json_value(value)
    if isinstance(value, dict):
        for key in ("answer", "final_answer", "value"):
            if key in value:
                return _structured_answer(value[key])
        return ""
    return str(value or "").strip()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _fact_text(node: Any) -> str:
    content = _json_value(_node_content(node))
    if isinstance(content, dict):
        title = str(content.get("title", content.get("entity", ""))).strip()
        text = str(content.get("text", content.get("sentence", ""))).strip()
        return " ".join(part for part in (title, text) if part)
    return str(content or "")


def _contains_normalized(text: str, needle: str) -> bool:
    normalized_text = normalize_bridge(text)
    normalized_needle = normalize_bridge(needle)
    return bool(normalized_needle and normalized_needle in normalized_text)
