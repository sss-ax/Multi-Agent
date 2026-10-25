"""Query-guided graph retrieval for HotpotQA evidence paths.

This module deliberately stops at candidate recall.  It does not decide whether
an answer is entailed; :mod:`hotpot_verifier` remains the final authority.
The graph is a lightweight evidence graph built from sentence-level entity
mentions, so it can be used with the existing supporting-fact representation.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


_STOP = {"the", "a", "an", "of", "and", "or", "was", "were", "is", "are", "what", "who", "which", "where", "by", "in", "to"}


def _terms(value: Any) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]+", str(value or "").lower()) if term not in _STOP and len(term) > 1}


def _aliases(value: Any) -> set[str]:
    text = str(value or "").strip().lower()
    if not text:
        return set()
    base = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = [token for token in base.split() if token not in {"the", "a", "an"}]
    values = {" ".join(tokens)} if tokens else set()
    if len(tokens) >= 2:
        values.add(tokens[-1])
        values.add(" ".join(tokens[:2]))
    return {value for value in values if len(value) >= 3}


@dataclass(frozen=True)
class GraphSeed:
    fact_id: str
    score: float
    sources: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidencePath:
    path_id: str
    evidence_ids: tuple[str, ...]
    entities: tuple[str, ...]
    relations: tuple[str, ...]
    retrieval_score: float
    connectivity_score: float
    uncertainty: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class GraphSeedRetriever:
    """Select query-related sentence seeds using lexical/entity signals."""

    def select(self, question: str, facts: Iterable[dict[str, Any]], *, limit: int = 8) -> list[GraphSeed]:
        question_terms = _terms(question)
        question_aliases = _aliases(question)
        ranked: list[GraphSeed] = []
        for fact in facts:
            fact_id = str(fact.get("fact_id", ""))
            text = f"{fact.get('title', '')} {fact.get('text', '')}"
            fact_terms = _terms(text)
            overlap = len(question_terms & fact_terms) / max(1, len(question_terms))
            entity_overlap = 1.0 if any(alias in question_aliases for alias in _aliases(fact.get("title", ""))) else 0.0
            relation_hint = 1.0 if any(term in fact_terms for term in {"born", "portrayed", "played", "held", "directed", "written", "located", "position"}) else 0.0
            score = min(1.0, 0.55 * overlap + 0.35 * entity_overlap + 0.10 * relation_hint)
            sources = []
            if overlap > 0:
                sources.append("lexical_overlap")
            if entity_overlap:
                sources.append("entity_link")
            if relation_hint:
                sources.append("relation_hint")
            ranked.append(GraphSeed(fact_id=fact_id, score=score, sources=tuple(sources)))
        ranked.sort(key=lambda seed: (-seed.score, seed.fact_id))
        return ranked[:limit]


class MultiPathExplorer:
    """Diffuse seed mass and enumerate short, connected evidence paths."""

    def __init__(self, *, alpha: float = 0.35, beam_size: int = 16, max_hops: int = 2):
        self.alpha = max(0.0, min(1.0, alpha))
        self.beam_size = max(1, int(beam_size))
        self.max_hops = max(1, int(max_hops))

    def explore(self, question: str, facts: list[dict[str, Any]], seeds: list[GraphSeed], *, limit: int = 16) -> list[EvidencePath]:
        if not facts:
            return []
        by_id = {str(fact.get("fact_id", "")): fact for fact in facts}
        ids = list(by_id)
        seed_mass = {fact_id: 0.0 for fact_id in ids}
        for seed in seeds:
            if seed.fact_id in seed_mass:
                seed_mass[seed.fact_id] = seed.score
        adjacency = {fact_id: set() for fact_id in ids}
        profiles = {fact_id: self._profile(fact) for fact_id, fact in by_id.items()}
        for index, left_id in enumerate(ids):
            for right_id in ids[index + 1:]:
                if profiles[left_id]["aliases"] & profiles[right_id]["aliases"]:
                    adjacency[left_id].add(right_id)
                    adjacency[right_id].add(left_id)
        ppr = self._diffuse(ids, adjacency, seed_mass)
        question_terms = _terms(question)
        paths: list[EvidencePath] = []
        for start_id in ids:
            start_score = ppr.get(start_id, 0.0)
            if start_score <= 0:
                continue
            paths.append(self._path((start_id,), by_id, profiles, ppr, question_terms))
            if self.max_hops >= 2:
                for next_id in adjacency[start_id]:
                    paths.append(self._path((start_id, next_id), by_id, profiles, ppr, question_terms))
        paths.sort(key=lambda path: (-path.retrieval_score, path.path_id))
        dedup: dict[tuple[str, ...], EvidencePath] = {}
        for path in paths:
            dedup.setdefault(path.evidence_ids, path)
        return list(dedup.values())[: min(limit, self.beam_size)]

    def _diffuse(self, ids: list[str], adjacency: dict[str, set[str]], seed_mass: dict[str, float]) -> dict[str, float]:
        total = sum(seed_mass.values()) or 1.0
        p0 = {fact_id: value / total for fact_id, value in seed_mass.items()}
        current = dict(p0)
        for _ in range(8):
            propagated = {fact_id: 0.0 for fact_id in ids}
            for fact_id in ids:
                neighbors = adjacency[fact_id] or {fact_id}
                share = current[fact_id] / len(neighbors)
                for neighbor in neighbors:
                    propagated[neighbor] += share
            current = {fact_id: self.alpha * p0[fact_id] + (1.0 - self.alpha) * propagated[fact_id] for fact_id in ids}
        return current

    def _profile(self, fact: dict[str, Any]) -> dict[str, Any]:
        title = str(fact.get("title", ""))
        text = str(fact.get("text", ""))
        aliases = _aliases(title) | _aliases(text)
        relation = str(fact.get("explicit_relation") or fact.get("relation") or fact.get("fact_semantics", {}).get("relation", ""))
        return {"title": title, "aliases": aliases, "entities": tuple(sorted(_aliases(title))), "relation": relation}

    def _path(self, ids: tuple[str, ...], by_id: dict[str, dict[str, Any]], profiles: dict[str, dict[str, Any]], ppr: dict[str, float], question_terms: set[str]) -> EvidencePath:
        entities: list[str] = []
        relations: list[str] = []
        lexical = 0.0
        for fact_id in ids:
            profile = profiles[fact_id]
            entities.extend(profile["entities"])
            if profile["relation"]:
                relations.append(profile["relation"])
            lexical += len(question_terms & _terms(f"{profile['title']} {by_id[fact_id].get('text', '')}")) / max(1, len(question_terms))
        connectivity = 1.0 if len(ids) == 1 else len(profiles[ids[0]]["aliases"] & profiles[ids[1]]["aliases"]) / max(1, len(profiles[ids[0]]["aliases"] | profiles[ids[1]]["aliases"]))
        retrieval = min(1.0, 0.45 * sum(ppr[fact_id] for fact_id in ids) + 0.35 * min(1.0, lexical / len(ids)) + 0.20 * connectivity)
        return EvidencePath(
            path_id="p:" + ":".join(ids),
            evidence_ids=ids,
            entities=tuple(dict.fromkeys(entities)),
            relations=tuple(dict.fromkeys(relations)),
            retrieval_score=retrieval,
            connectivity_score=connectivity,
            uncertainty=1.0 - retrieval,
            metadata={"ppr_scores": {fact_id: ppr[fact_id] for fact_id in ids}},
        )


class PathReranker:
    """Rerank paths with question/path semantics and evidence grounding.

    This is intentionally a ranking signal, not a verifier: an explicit
    downstream HotpotQA verifier must still approve the answer.
    """

    def rerank(self, question: str, paths: Iterable[EvidencePath], facts: list[dict[str, Any]], *, limit: int = 16) -> list[EvidencePath]:
        fact_by_id = {str(fact.get("fact_id", "")): fact for fact in facts}
        question_terms = _terms(question)
        ranked: list[tuple[float, EvidencePath]] = []
        for path in paths:
            text = " ".join(str(fact_by_id.get(fact_id, {}).get("text", "")) for fact_id in path.evidence_ids)
            lexical = len(question_terms & _terms(text)) / max(1, len(question_terms))
            grounded = sum(1 for fact_id in path.evidence_ids if fact_id in fact_by_id) / max(1, len(path.evidence_ids))
            score = 0.45 * path.retrieval_score + 0.25 * path.connectivity_score + 0.20 * lexical + 0.10 * grounded
            ranked.append((score, EvidencePath(
                path_id=path.path_id,
                evidence_ids=path.evidence_ids,
                entities=path.entities,
                relations=path.relations,
                retrieval_score=max(0.0, min(1.0, score)),
                connectivity_score=path.connectivity_score,
                uncertainty=max(0.0, min(1.0, 1.0 - score)),
                metadata={**path.metadata, "rerank_score": score, "grounding_score": grounded},
            )))
        ranked.sort(key=lambda item: (-item[0], item[1].path_id))
        return [path for _, path in ranked[:limit]]


class QueryGuidedGraphRetriever:
    """Seed selection + diffusion + beam exploration + path reranking."""

    def __init__(self, *, alpha: float = 0.35, beam_size: int = 16):
        self.seeds = GraphSeedRetriever()
        self.explorer = MultiPathExplorer(alpha=alpha, beam_size=beam_size)
        self.reranker = PathReranker()

    def retrieve(self, question: str, facts: list[dict[str, Any]], *, limit: int = 16) -> dict[str, Any]:
        seeds = self.seeds.select(question, facts, limit=min(8, len(facts)))
        paths = self.reranker.rerank(question, self.explorer.explore(question, facts, seeds, limit=limit), facts, limit=limit)
        return {"seeds": [seed.as_dict() for seed in seeds], "paths": [path.as_dict() for path in paths], "retrieval_method": "query_guided_ppr_beam"}

