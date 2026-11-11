"""High-precision deterministic Query–Claim–Evidence graph for HotpotQA."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .hotpot_semantic_graph import SemanticGraphBuild, normalize_entity

RELATION_PATTERNS = (
    ("founded_by", ("founded by", "co-founded by", "established by", "founded")),
    ("located_in", ("located in", "based in", "situated in", "from")),
    ("born_in", ("born in",)),
    ("directed_by", ("directed by", "directed")),
    ("member_of", ("member of", "part of")),
    ("capital_of", ("capital of",)),
    ("released_in", ("released in", "published in")),
    ("created_by", ("created by", "created")),
    ("same_as", ("also known as", "known as")),
)
COMPARISON_WORDS = {"earlier", "later", "older", "younger", "first", "same", "both"}


@dataclass(frozen=True)
class QuerySubgoal:
    subgoal_id: str
    subject_ref: str
    relation: str
    object_ref: str


@dataclass(frozen=True)
class QuerySemanticSpec:
    question_type: str
    query_entity_ids: tuple[str, ...]
    required_relations: tuple[str, ...]
    answer_type: str
    subgoals: tuple[QuerySubgoal, ...]


@dataclass(frozen=True)
class ClaimArtifact:
    claim_id: str
    subject_entity_id: str
    relation: str
    object_entity_id: str | None
    object_value: str | None
    evidence_sentence_id: str
    confidence: float
    extractor_version: str = "deterministic_claim_v1"


@dataclass(frozen=True)
class ClaimGraphBuild:
    query_spec: QuerySemanticSpec
    claims: tuple[ClaimArtifact, ...]
    rejected_claims: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ClaimProofSelection:
    selected_claims: tuple[ClaimArtifact, ...]
    selected_sentence_ids: tuple[str, ...]
    selected_document_ids: tuple[str, ...]
    contract_satisfied: bool
    missing_requirements: tuple[str, ...]
    relation_coverage: float
    candidate_proof_count: int


def _answer_type(question: str) -> str:
    q = normalize_entity(question).split()
    if not q: return "unknown"
    if q[0] == "where": return "location"
    if q[0] in {"when", "year"}: return "date"
    if q[0] == "who": return "person"
    if q[:2] == ["how", "many"]: return "number"
    if q[0] == "which": return "entity"
    return "unknown"


def _relation_requirements(question: str, answer_type: str) -> tuple[str, ...]:
    q = normalize_entity(question)
    found = []
    for relation, patterns in RELATION_PATTERNS:
        if any(pattern in q for pattern in patterns): found.append(relation)
    if answer_type == "location" and "located_in" not in found: found.append("located_in")
    if answer_type == "date" and not found: found.append("released_in")
    return tuple(dict.fromkeys(found))


def build_query_spec(question: str, semantic: SemanticGraphBuild) -> QuerySemanticSpec:
    q = normalize_entity(question)
    answer_type = _answer_type(question)
    required = _relation_requirements(question, answer_type)
    comparison = bool(COMPARISON_WORDS & set(q.split())) and len(semantic.query_entity_ids) >= 2
    qtype = "comparison" if comparison else ("bridge" if semantic.query_entity_ids and required else "unknown")
    subgoals = []
    if qtype == "comparison":
        for index, entity_id in enumerate(semantic.query_entity_ids[:2], 1):
            relation = required[0] if required else "comparison_date"
            subgoals.append(QuerySubgoal(f"subgoal_{index}", entity_id, relation, f"$value_{index}"))
    elif qtype == "bridge" and len(required) >= 1:
        subgoals.append(QuerySubgoal("subgoal_1", semantic.query_entity_ids[0], required[0], "$intermediate"))
        if len(required) >= 2:
            subgoals.append(QuerySubgoal("subgoal_2", "$intermediate", required[1], "$answer"))
    return QuerySemanticSpec(qtype, tuple(semantic.query_entity_ids[:4]), required, answer_type, tuple(subgoals))


def extract_claims(question: str, semantic: SemanticGraphBuild, sentences: Sequence[Mapping[str, Any]]) -> ClaimGraphBuild:
    spec = build_query_spec(question, semantic)
    entity_by_name = {normalize_entity(e.canonical_name): e.entity_id for e in semantic.entities}
    claims = []; rejected = []
    for sentence in sentences:
        text = normalize_entity(str(sentence["text"]))
        mentioned = [eid for eid in semantic.sentence_mentions.get(str(sentence["node_id"]), ())]
        for relation, patterns in RELATION_PATTERNS:
            if not any(pattern in text for pattern in patterns): continue
            object_value = None
            if len(mentioned) < 2:
                if relation == "located_in":
                    match = re.search(r"(?:located in|based in|situated in|from)\s+([a-z][a-z0-9_-]*)", text)
                    if match:
                        object_value = match.group(1)
                if not mentioned or object_value is None:
                    rejected.append({"sentence_id": sentence["node_id"], "relation": relation, "reason": "missing_subject_or_object"}); continue
            subject, object_entity = mentioned[0], mentioned[1] if len(mentioned) > 1 else None
            claim_id = "claim_" + hashlib.sha256(f"{sentence['node_id']}|{relation}|{subject}|{object_entity}".encode()).hexdigest()[:16]
            claims.append(ClaimArtifact(claim_id, subject, relation, object_entity, object_value, str(sentence["node_id"]), 0.95))
    return ClaimGraphBuild(spec, tuple(dict.fromkeys(claims)), tuple(rejected))


def select_claim_proof(build: ClaimGraphBuild, sentences: Sequence[Mapping[str, Any]], *, token_counter: Callable[[Any], int], budget_tokens: int = 512, max_candidate_proofs: int = 32, max_selected_proofs: int = 2) -> ClaimProofSelection:
    claims = list(build.claims)
    by_relation = {relation: [claim for claim in claims if claim.relation == relation] for relation in build.query_spec.required_relations}
    candidates: list[tuple[ClaimArtifact, ...]] = []
    spec = build.query_spec
    if spec.question_type == "bridge" and len(spec.subgoals) >= 2:
        first = by_relation.get(spec.subgoals[0].relation, [])
        second = by_relation.get(spec.subgoals[1].relation, [])
        for left in first:
            for right in second:
                if left.object_entity_id and left.object_entity_id == right.subject_entity_id and left.evidence_sentence_id != right.evidence_sentence_id:
                    candidates.append((left, right))
    elif spec.question_type == "comparison":
        relation = spec.subgoals[0].relation if spec.subgoals else ""
        grouped = by_relation.get(relation, [])
        for index, left in enumerate(grouped):
            for right in grouped[index + 1:]:
                if left.subject_entity_id != right.subject_entity_id and left.evidence_sentence_id != right.evidence_sentence_id:
                    candidates.append((left, right))
    candidates = candidates[:max_candidate_proofs]
    sent = {str(item["node_id"]): item for item in sentences}
    candidates.sort(key=lambda proof: (len({c.evidence_sentence_id for c in proof}), sum(token_counter(sent[c.evidence_sentence_id]["text"]) for c in proof), tuple(c.evidence_sentence_id for c in proof)))
    selected_claims: list[ClaimArtifact] = []
    for proof in candidates[:max_selected_proofs]:
        sentence_ids = list(dict.fromkeys(c.evidence_sentence_id for c in (*selected_claims, *proof)))
        if sum(token_counter(sent[sid]["text"]) for sid in sentence_ids) <= budget_tokens:
            selected_claims.extend(proof)
    selected_claims = list(dict.fromkeys(selected_claims))
    selected_sentence_ids = tuple(dict.fromkeys(c.evidence_sentence_id for c in selected_claims))
    docs = tuple(dict.fromkeys(str(sent[sid]["document_id"]) for sid in selected_sentence_ids))
    covered = {c.relation for c in selected_claims}
    required = set(spec.required_relations)
    relation_coverage = len(covered & required) / max(1, len(required))
    missing = []
    if spec.question_type == "unknown": missing.append("unknown_question_type")
    if not spec.subgoals: missing.append("missing_subgoals")
    if not selected_claims: missing.append("missing_grounded_claims")
    if required and relation_coverage < 1.0: missing.append("incomplete_relation_coverage")
    if len(docs) < 2: missing.append("fewer_than_two_documents")
    if len(selected_sentence_ids) < 2: missing.append("fewer_than_two_sentences")
    return ClaimProofSelection(tuple(selected_claims), selected_sentence_ids, docs, not missing, tuple(missing), relation_coverage, len(candidates))
