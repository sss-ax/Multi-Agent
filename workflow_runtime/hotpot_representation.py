"""Lossless HotpotQA candidate chain views.

These views are runtime renderings over existing graph/data facts.  They are
not canonical graph nodes and deliberately avoid LLM-generated relation
triples in P1.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .context_slicer import _node_sort_key
from .hotpot_bridge import normalize_bridge


@dataclass(frozen=True)
class HotpotChainView:
    candidate_id: str
    bridge_entity: str
    hop1_fact_ids: list[str]
    hop1_key_sentence: str
    hop2_fact_ids: list[str]
    hop2_key_sentence: str
    derived_answer: str
    answer_type: str
    answer_alignment: str | None
    cited_fact_ids: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HotpotSemanticFragment:
    type: str
    candidate_id: str
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.type, "candidate_id": self.candidate_id, **dict(self.payload)}


@dataclass(frozen=True)
class SelectorNeed:
    unresolved_candidates: list[str]
    disagreement_type: str | None
    missing_fragments: dict[str, list[str]]
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_hotpot_chain_view(candidate: Any, graph_store: Any, question: str) -> HotpotChainView:
    """Build a deterministic runtime view from a candidate and graph facts."""
    value = candidate.get("value", candidate) if isinstance(candidate, dict) else {}
    if not isinstance(value, dict):
        value = {}
    facts = _facts_from_graph(graph_store)
    bridge = str(value.get("bridge_entity", "")).strip()
    cited = _string_list(value.get("cited_fact_ids"))
    selected = [fact for fact in facts if fact["fact_id"] in set(cited)]
    if not selected and bridge:
        selected = _facts_for_bridge(bridge, facts)[:2]
    hop1 = selected[:1] or _facts_for_bridge(bridge, facts)[:1]
    hop2 = selected[1:2] or _rank_facts(" ".join([question, bridge, *_fact_texts(hop1)]), facts, limit=1, exclude=hop1)
    derived = str(value.get("answer") or value.get("derived_answer") or "").strip()
    return HotpotChainView(
        candidate_id=str(candidate.get("candidate_id", candidate.get("id", "candidate")) if isinstance(candidate, dict) else "candidate"),
        bridge_entity=bridge,
        hop1_fact_ids=[fact["fact_id"] for fact in hop1],
        hop1_key_sentence=_join_fact_sentences(hop1),
        hop2_fact_ids=[fact["fact_id"] for fact in hop2],
        hop2_key_sentence=_join_fact_sentences(hop2),
        derived_answer=derived,
        answer_type=infer_hotpot_answer_type(question, derived),
        answer_alignment=deterministic_answer_alignment(question, derived, [*hop1, *hop2]),
        cited_fact_ids=cited,
    )


def build_hotpot_chain_view_from_facts(
    *,
    candidate_id: str,
    bridge_entity: str,
    question: str,
    facts: list[dict[str, Any]],
    hop1_fact_ids: Iterable[str] = (),
    hop2_fact_ids: Iterable[str] = (),
    derived_answer: str = "",
) -> HotpotChainView:
    indexed = [_fact_payload(index, fact) for index, fact in enumerate(facts, start=1)]
    hop1 = _select_fact_ids(indexed, hop1_fact_ids) or _facts_for_bridge(bridge_entity, indexed)[:1]
    hop2 = (
        _select_fact_ids(indexed, hop2_fact_ids)
        or _rank_facts(" ".join([question, bridge_entity, *_fact_texts(hop1)]), indexed, limit=1, exclude=hop1)
    )
    cited = [fact["fact_id"] for fact in [*hop1, *hop2]]
    return HotpotChainView(
        candidate_id=candidate_id,
        bridge_entity=bridge_entity,
        hop1_fact_ids=[fact["fact_id"] for fact in hop1],
        hop1_key_sentence=_join_fact_sentences(hop1),
        hop2_fact_ids=[fact["fact_id"] for fact in hop2],
        hop2_key_sentence=_join_fact_sentences(hop2),
        derived_answer=str(derived_answer or _derived_answer_hint(question, [*hop1, *hop2])).strip(),
        answer_type=infer_hotpot_answer_type(question, derived_answer),
        answer_alignment=deterministic_answer_alignment(question, derived_answer, [*hop1, *hop2]),
        cited_fact_ids=cited,
    )


def render_hotpot_chain_view(view: HotpotChainView, *, include_hop1: bool, include_hop2: bool, include_answer: bool) -> str:
    lines = [
        f"[CANDIDATE {view.candidate_id}]",
        "",
        "Bridge:",
        view.bridge_entity,
    ]
    if include_hop1:
        lines.extend([
            "",
            "Hop 1:",
            f"fact_id: {', '.join(view.hop1_fact_ids)}",
            f"evidence: {view.hop1_key_sentence}",
        ])
    if include_hop2:
        lines.extend([
            "",
            "Hop 2:",
            f"fact_id: {', '.join(view.hop2_fact_ids)}",
            f"evidence: {view.hop2_key_sentence}",
        ])
    if include_answer:
        lines.extend([
            "",
            "Derived answer:",
            view.derived_answer,
            "",
            "Answer type:",
            view.answer_type,
            "",
            "Answer alignment:",
            view.answer_alignment or "",
        ])
    return "\n".join(lines).strip()


def build_hotpot_semantic_fragments(view: HotpotChainView, question: str) -> list[HotpotSemanticFragment]:
    """Split a structured chain view into deterministic semantic fragments."""
    question_target = infer_hotpot_question_target(question)
    contradiction = _has_contradiction_signal(view)
    return [
        HotpotSemanticFragment(
            "bridge_identity",
            view.candidate_id,
            {"bridge": view.bridge_entity},
        ),
        HotpotSemanticFragment(
            "hop1_evidence",
            view.candidate_id,
            {
                "fact_id": _first_or_empty(view.hop1_fact_ids),
                "fact_ids": list(view.hop1_fact_ids),
                "text": view.hop1_key_sentence,
            },
        ),
        HotpotSemanticFragment(
            "hop2_evidence",
            view.candidate_id,
            {
                "fact_id": _first_or_empty(view.hop2_fact_ids),
                "fact_ids": list(view.hop2_fact_ids),
                "text": view.hop2_key_sentence,
            },
        ),
        HotpotSemanticFragment(
            "derived_answer",
            view.candidate_id,
            {"value": view.derived_answer},
        ),
        HotpotSemanticFragment(
            "answer_alignment",
            view.candidate_id,
            {
                "question_target": question_target,
                "derived_answer": view.derived_answer,
                "alignment": _alignment_label(view),
            },
        ),
        HotpotSemanticFragment(
            "contradiction_signal",
            view.candidate_id,
            {"value": contradiction},
        ),
    ]


def render_hotpot_semantic_fragments(
    view: HotpotChainView,
    question: str,
    *,
    include_types: Iterable[str] | None = None,
) -> str:
    allowed = set(include_types) if include_types is not None else None
    fragments = [
        fragment.as_dict()
        for fragment in build_hotpot_semantic_fragments(view, question)
        if allowed is None or fragment.type in allowed
    ]
    lines = [f"[CANDIDATE {view.candidate_id}]"]
    for fragment in fragments:
        payload = {key: value for key, value in fragment.items() if key not in {"candidate_id"}}
        lines.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines)


def render_hotpot_discriminative_chain_static(view: HotpotChainView, question: str = "") -> str:
    """Render the P3 static discriminative chain.

    P2 identified hop evidence as the decision-critical signal.  This view
    keeps the bridge plus compact, deterministic hop clues and drops the
    low-value alignment/contradiction/derived-answer fragments.
    """
    del question
    return "\n".join([
        f"[CANDIDATE {view.candidate_id}]",
        f"bridge: {view.bridge_entity}",
        f"hop1 {','.join(view.hop1_fact_ids)}: {_compact_fact_claim(view.hop1_key_sentence, view.bridge_entity)}",
        f"hop2 {','.join(view.hop2_fact_ids)}: {_compact_fact_claim(view.hop2_key_sentence, '')}",
    ]).strip()


def infer_hotpot_answer_type(question: str, answer: str = "") -> str:
    q = question.lower()
    if q.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "has ", "have ", "had ", "can ", "could ")):
        return "yes_no"
    if "which country" in q:
        return "country"
    if "which city" in q:
        return "city"
    if q.startswith("when ") or "what year" in q:
        return "date"
    if q.startswith("how many"):
        return "number"
    if q.startswith("who "):
        return "person_or_entity"
    if q.startswith("where "):
        return "location"
    return "entity_or_short_answer" if str(answer or "").strip() else "unknown"


def infer_hotpot_question_target(question: str) -> str:
    q = question.lower()
    if "same nationality" in q or "nationality" in q:
        return "same nationality" if "same" in q else "nationality"
    if "same country" in q:
        return "same country"
    if "which country" in q:
        return "country"
    if "which city" in q:
        return "city"
    if "what year" in q:
        return "year"
    if q.startswith("when "):
        return "date"
    if q.startswith("who "):
        return "person/entity"
    if q.startswith("where "):
        return "location"
    words = re.findall(r"[a-z0-9]+", q)
    return " ".join(words[:6]) if words else "answer"


def deterministic_answer_alignment(question: str, derived_answer: str, facts: list[dict[str, Any]]) -> str | None:
    answer = str(derived_answer or "").strip()
    if not answer:
        return None
    answer_type = infer_hotpot_answer_type(question, answer)
    text = " ".join(_fact_texts(facts))
    if answer_type == "yes_no":
        return "yes/no answer derived from whether both evidence hops satisfy the question relation"
    if answer and _normalized(answer) in _normalized(text):
        return "derived answer appears verbatim in cited evidence"
    return "derived answer is a short answer candidate associated with the cited evidence"


def _alignment_label(view: HotpotChainView) -> str:
    text = str(view.answer_alignment or "").lower()
    if "contradict" in text or "not supported" in text:
        return "contradicted"
    if view.derived_answer:
        return "supported"
    return "unknown"


def _has_contradiction_signal(view: HotpotChainView) -> bool:
    text = " ".join([
        str(view.answer_alignment or ""),
        str(view.hop1_key_sentence or ""),
        str(view.hop2_key_sentence or ""),
    ]).lower()
    return bool(re.search(r"\bcontradict|whereas|however|not supported\b", text))


def _first_or_empty(values: list[str]) -> str:
    return str(values[0]) if values else ""


def _facts_from_graph(graph_store: Any) -> list[dict[str, Any]]:
    state = graph_store.snapshot() if hasattr(graph_store, "snapshot") else graph_store
    nodes = list(getattr(state, "nodes", {}).values()) if state is not None else []
    facts = []
    for node in sorted(nodes, key=_node_sort_key):
        if getattr(node, "type", "") != "supporting_fact":
            continue
        if hasattr(node, "is_operationally_valid") and not node.is_operationally_valid():
            continue
        facts.append(_node_fact_payload(node))
    return facts


def _node_fact_payload(node: Any) -> dict[str, Any]:
    content = _json_value(getattr(node, "content", {}))
    if not isinstance(content, dict):
        content = {"text": str(content)}
    return {
        "fact_id": str(getattr(node, "logical_id", "")),
        "title": str(content.get("title", content.get("entity", ""))),
        "sent_id": content.get("sent_id"),
        "text": str(content.get("text", content.get("sentence", ""))),
    }


def _fact_payload(index: int, fact: dict[str, Any]) -> dict[str, Any]:
    return {
        "fact_id": str(fact.get("fact_id") or f"supporting_fact_{index}"),
        "title": str(fact.get("title", fact.get("entity", ""))),
        "sent_id": fact.get("sent_id"),
        "text": str(fact.get("text", fact.get("sentence", ""))),
    }


def _facts_for_bridge(bridge: str, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key = normalize_bridge(bridge)
    return [fact for fact in facts if normalize_bridge(fact.get("title", fact.get("entity", ""))) == key]


def _rank_facts(query: str, facts: list[dict[str, Any]], *, limit: int, exclude: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    excluded = {fact["fact_id"] for fact in (exclude or [])}
    query_terms = _terms(query)
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for fact in facts:
        if fact["fact_id"] in excluded:
            continue
        text_terms = _terms(f"{fact['title']} {fact['text']}")
        scored.append((-len(query_terms & text_terms), fact["fact_id"], fact))
    scored.sort()
    return [fact for _, _, fact in scored[:limit]]


def _select_fact_ids(facts: list[dict[str, Any]], ids: Iterable[str]) -> list[dict[str, Any]]:
    wanted = {str(item) for item in ids}
    return [fact for fact in facts if fact["fact_id"] in wanted]


def _derived_answer_hint(question: str, facts: list[dict[str, Any]]) -> str:
    text = " ".join(_fact_texts(facts))
    if re.search(r"\byes\b|\bno\b|same|both|also", question, flags=re.I):
        if re.search(r"\bnot\b|different|whereas|however", text, flags=re.I):
            return "no"
        return "yes_or_no_from_chain"
    date = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", text)
    if date:
        return date.group(1)
    return ""


def _join_fact_sentences(facts: list[dict[str, Any]]) -> str:
    return " ".join(fact["text"] for fact in facts if str(fact.get("text", "")).strip())


def _fact_texts(facts: list[dict[str, Any]]) -> list[str]:
    return [str(fact.get("text", "")) for fact in facts]


def _terms(text: str) -> set[str]:
    stop = {"the", "a", "an", "of", "in", "on", "and", "or", "to", "was", "is", "are", "were", "for", "with"}
    return {term for term in re.findall(r"[a-z0-9]+", text.lower()) if len(term) > 2 and term not in stop}


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def _compact_fact_claim(text: str, title: str) -> str:
    sentence = re.sub(r"\s+", " ", str(text or "")).strip()
    sentence = sentence.rstrip(".")
    title = re.sub(r"\s+", " ", str(title or "")).strip()
    if not sentence:
        return ""
    if title:
        escaped = re.escape(title)
        match = re.match(rf"(?i)^({escaped})\s+(?:is|was|are|were|has|have|had)\s+(.*)$", sentence)
        if match:
            return f"{title}: {_truncate_words(match.group(2), 14)}"
        if normalize_bridge(title) in normalize_bridge(sentence):
            return f"{title}: {_truncate_words(sentence, 16)}"
    match = re.match(r"(?i)^([A-Z][^,.;:]{1,80}?)\s+(?:is|was|are|were|has|have|had)\s+(.*)$", sentence)
    if match and len(match.group(1).split()) <= 6:
        return f"{match.group(1)}: {_truncate_words(match.group(2), 14)}"
    return _truncate_words(sentence, 18)


def _truncate_words(text: str, limit: int) -> str:
    words = str(text or "").split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit])
