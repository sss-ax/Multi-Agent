"""Deterministic public-source Entity–Evidence graph for HotpotQA."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

STOP = {"what", "where", "which", "who", "when", "why", "how", "the", "a", "an", "is", "are", "was", "were", "did", "does"}
RELATIONS = {"born", "located", "founded", "directed", "member", "capital", "year", "place", "same", "both", "earlier", "later"}
PRONOUN_PREFIXES = ("he ", "she ", "it ", "they ", "this ", "that ", "these ", "those ", "former ", "latter ", "the company ", "the group ", "the city ", "the film ", "the album ")


def normalize_entity(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value)
    value = re.sub(r"[^\w\s-]", " ", value, flags=re.UNICODE)
    return " ".join(value.split()).strip(" -_")


def _spans(text: str) -> set[str]:
    values = set()
    for token in re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", str(text)):
        value = normalize_entity(token)
        if value and value not in STOP:
            values.add(value)
    for match in re.findall(r"(?:\b[A-Z][A-Za-z0-9_-]*\b(?:\s+|$)){1,6}|\b[A-Z]{2,}\b|[\"“][^\"”]{3,}[\"”]", str(text)):
        value = normalize_entity(match.strip(' "“”'))
        if value and value not in STOP and (len(value) >= 3 or any(ch.isdigit() for ch in value)):
            values.add(value)
    return values


@dataclass(frozen=True)
class EntityArtifact:
    entity_id: str
    canonical_name: str
    surface_forms: tuple[str, ...]
    entity_kind: str
    document_frequency: int
    source_sentence_ids: tuple[str, ...]
    generic: bool = False


@dataclass(frozen=True)
class HotpotSemanticPath:
    path_id: str
    path_type: str
    query_entity_ids: tuple[str, ...]
    bridge_entity_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    sentence_ids: tuple[str, ...]
    entity_coverage: float
    relation_coverage: float
    token_cost: int
    complete: bool
    sentence_score: float = 0.0


@dataclass(frozen=True)
class HotpotSemanticSelectionResult:
    selected_paths: tuple[HotpotSemanticPath, ...]
    candidate_paths: tuple[HotpotSemanticPath, ...]
    selected_document_ids: tuple[str, ...]
    selected_sentence_ids: tuple[str, ...]
    selected_entity_ids: tuple[str, ...]
    neighbor_sentence_ids: tuple[str, ...]
    closure_node_ids: tuple[str, ...]
    contract_satisfied: bool
    missing_requirements: tuple[str, ...]
    token_cost: int
    query_entity_ids: tuple[str, ...] = ()
    entity_coverage: float = 0.0
    relation_coverage: float = 0.0


@dataclass(frozen=True)
class SemanticGraphBuild:
    entities: tuple[EntityArtifact, ...]
    query_entity_ids: tuple[str, ...]
    query_relations: tuple[str, ...]
    sentence_mentions: Mapping[str, tuple[str, ...]]
    title_entities: Mapping[str, str]
    entity_documents: Mapping[str, tuple[str, ...]]


def build_semantic_graph(question: str, documents: Sequence[Mapping[str, Any]], sentences: Sequence[Mapping[str, Any]]) -> SemanticGraphBuild:
    canonical: dict[str, dict[str, Any]] = {}
    title_entities: dict[str, str] = {}
    doc_by_id = {str(d["node_id"]): d for d in documents}

    def add(name: str, surface: str, kind: str) -> str | None:
        name = normalize_entity(name)
        if not name or name in STOP:
            return None
        entity_id = "entity_" + hashlib.sha256(name.encode()).hexdigest()[:16]
        item = canonical.setdefault(name, {"id": entity_id, "surfaces": set(), "kind": kind, "sentences": set(), "docs": set()})
        item["surfaces"].add(str(surface))
        if kind == "title": item["kind"] = "title"
        return entity_id

    for document in documents:
        title = str(document["title"])
        eid = add(title, title, "title")
        if eid:
            title_entities[str(document["node_id"])] = eid
            canonical[normalize_entity(title)]["docs"].add(str(document["node_id"]))
    for value in _spans(question): add(value, value, "named_span")
    for sentence in sentences:
        for value in _spans(str(sentence["text"])): add(value, value, "named_span")

    sentence_mentions: dict[str, tuple[str, ...]] = {}
    for sentence in sentences:
        text_norm = normalize_entity(str(sentence["text"]))
        mentions = []
        for name in sorted(canonical, key=lambda x: (text_norm.find(x) if text_norm.find(x) >= 0 else 10**9, -len(x), x)):
            if re.search(rf"\b{re.escape(name)}\b", text_norm):
                item = canonical[name]; mentions.append(item["id"]); item["sentences"].add(str(sentence["node_id"])); item["docs"].add(str(sentence["document_id"]))
        sentence_mentions[str(sentence["node_id"])] = tuple(dict.fromkeys(mentions))

    qnorm = normalize_entity(question)
    query_ids = tuple(item["id"] for name, item in sorted(canonical.items()) if re.search(rf"\b{re.escape(name)}\b", qnorm))
    entities = tuple(EntityArtifact(item["id"], name, tuple(sorted(item["surfaces"])), item["kind"], len(item["docs"]), tuple(sorted(item["sentences"])), len(item["docs"]) > 3) for name, item in sorted(canonical.items()))
    entity_documents = {item["id"]: tuple(sorted(item["docs"])) for item in canonical.values()}
    return SemanticGraphBuild(entities, query_ids, tuple(sorted(RELATIONS & set(qnorm.split()))), sentence_mentions, title_entities, entity_documents)


def select_semantic_paths(build: SemanticGraphBuild, documents: Sequence[Mapping[str, Any]], sentences: Sequence[Mapping[str, Any]], *, token_counter: Callable[[Any], int], max_paths: int = 2, max_document_hops: int = 2, budget_tokens: int = 512) -> HotpotSemanticSelectionResult:
    sent = {str(s["node_id"]): s for s in sentences}
    entity = {e.entity_id: e for e in build.entities}
    query = set(build.query_entity_ids)
    tokens = set(normalize_entity(" ".join(entity[e].canonical_name for e in query if e in entity)).split())

    def relevance(sid: str) -> float:
        text = set(normalize_entity(str(sent[sid]["text"])).split())
        return float(len(tokens & text) + 2 * len(query & set(build.sentence_mentions.get(sid, ()))))

    paths: list[HotpotSemanticPath] = []
    ids = sorted(sent, key=lambda sid: (int(sent[sid]["document_index"]), int(sent[sid]["sentence_index"])))
    for i, left in enumerate(ids):
        lm = set(build.sentence_mentions.get(left, ()))
        if not (lm & query): continue
        for right in ids[i + 1:]:
            if sent[left]["document_id"] == sent[right]["document_id"]: continue
            if max_document_hops < 2: continue
            rm = set(build.sentence_mentions.get(right, ()))
            bridge = tuple(sorted(e for e in lm & rm if e in entity and not entity[e].generic))
            title_bridge = tuple(sorted((lm & set(build.title_entities.values())) | (rm & set(build.title_entities.values()))))
            bridges = tuple(dict.fromkeys((*bridge, *title_bridge)))
            comparison = bool((lm & query) and (rm & query))
            if not bridges and not comparison: continue
            qids = tuple(sorted(query & (lm | rm)))
            docs = (str(sent[left]["document_id"]), str(sent[right]["document_id"]))
            sentence_ids = (left, right)
            coverage = len(qids) / max(1, len(query))
            relation_cov = 1.0 if build.query_relations and any(r in normalize_entity(str(sent[x]["text"])).split() for r in build.query_relations for x in sentence_ids) else 0.0
            cost = sum(token_counter(sent[x]["text"]) for x in sentence_ids)
            path_type = "comparison" if comparison and len(qids) >= 2 else "bridge"
            complete = len(docs) == 2 and coverage > 0 and bool(bridges or path_type == "comparison")
            pid = "path_" + hashlib.sha256((left + right + "|".join(bridges)).encode()).hexdigest()[:16]
            paths.append(HotpotSemanticPath(pid, path_type, qids, bridges, docs, sentence_ids, coverage, relation_cov, cost, complete, relevance(left) + relevance(right)))

    paths.sort(key=lambda p: (not p.complete, -len(p.query_entity_ids), -len(p.document_ids), -len(p.bridge_entity_ids), -p.relation_coverage, -p.sentence_score, p.token_cost, p.sentence_ids))
    selected: list[HotpotSemanticPath] = []
    selected_sentences: list[str] = []
    for path in paths:
        new_ids = [sid for sid in path.sentence_ids if sid not in selected_sentences]
        if sum(token_counter(sent[sid]["text"]) for sid in (*selected_sentences, *new_ids)) > budget_tokens: continue
        selected.append(path); selected_sentences.extend(new_ids)
        if len(selected) >= max_paths: break

    neighbor_ids: list[str] = []
    by_pos = {(int(s["document_index"]), int(s["sentence_index"])): str(s["node_id"]) for s in sentences}
    for sid in list(selected_sentences):
        item = sent[sid]; text = str(item["text"]).strip(); lower = text.casefold()
        candidates = []
        if lower.startswith(PRONOUN_PREFIXES): candidates.append(by_pos.get((int(item["document_index"]), int(item["sentence_index"]) - 1)))
        if len(normalize_entity(text).split()) < 8 or text.endswith(":"): candidates.append(by_pos.get((int(item["document_index"]), int(item["sentence_index"]) + 1)))
        for neighbor in candidates:
            if neighbor and neighbor not in selected_sentences and neighbor not in neighbor_ids:
                proposed = (*selected_sentences, *neighbor_ids, neighbor)
                if sum(token_counter(sent[x]["text"]) for x in proposed) <= budget_tokens: neighbor_ids.append(neighbor)

    all_sentences = tuple(sorted((*selected_sentences, *neighbor_ids), key=lambda sid: (int(sent[sid]["document_index"]), int(sent[sid]["sentence_index"]))))
    docs = tuple(dict.fromkeys(str(sent[sid]["document_id"]) for sid in all_sentences))
    entities = tuple(sorted({eid for sid in all_sentences for eid in build.sentence_mentions.get(sid, ())}))
    missing = []
    if not build.query_entity_ids: missing.append("missing_query_root")
    if not selected or not any(p.complete for p in selected): missing.append("missing_complete_path")
    if len(docs) < 2: missing.append("fewer_than_two_documents")
    if len(all_sentences) < 2: missing.append("fewer_than_two_sentences")
    coverage = len(set(build.query_entity_ids) & set(entities)) / max(1, len(build.query_entity_ids))
    if build.query_entity_ids and coverage == 0: missing.append("zero_entity_coverage")
    return HotpotSemanticSelectionResult(tuple(selected), tuple(paths), docs, all_sentences, entities, tuple(neighbor_ids), docs, not missing, tuple(missing), sum(token_counter(sent[x]["text"]) for x in all_sentences), build.query_entity_ids, coverage, max((p.relation_coverage for p in selected), default=0.0))
