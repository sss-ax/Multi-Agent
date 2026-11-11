from baselines.hotpot_semantic_graph import build_semantic_graph
from baselines.hotpot_claim_graph import build_query_spec, extract_claims, select_claim_proof


def _data():
    docs = [{"node_id": "d1", "title": "Alice", "index": 0}, {"node_id": "d2", "title": "Acme", "index": 1}]
    sentences = [
        {"node_id": "s1", "text": "Alice founded Acme.", "document_id": "d1", "document_index": 0, "sentence_index": 0},
        {"node_id": "s2", "text": "Acme is based in Paris.", "document_id": "d2", "document_index": 1, "sentence_index": 0},
    ]
    return docs, sentences


def test_query_spec_and_claims_are_grounded():
    docs, sentences = _data()
    semantic = build_semantic_graph("Where is the company founded by Alice based?", docs, sentences)
    spec = build_query_spec("Where is the company founded by Alice based?", semantic)
    claims = extract_claims("Where is the company founded by Alice based?", semantic, sentences)
    assert spec.question_type == "bridge"
    assert {claim.relation for claim in claims.claims} >= {"founded_by", "located_in"}
    assert all(claim.evidence_sentence_id in {"s1", "s2"} for claim in claims.claims)


def test_claim_proof_requires_relation_complete_two_hop_path():
    docs, sentences = _data()
    semantic = build_semantic_graph("Where is the company founded by Alice based?", docs, sentences)
    claims = extract_claims("Where is the company founded by Alice based?", semantic, sentences)
    proof = select_claim_proof(claims, sentences, token_counter=lambda x: len(str(x).split()))
    assert proof.contract_satisfied
    assert proof.relation_coverage == 1.0
    assert set(proof.selected_sentence_ids) == {"s1", "s2"}


def test_claim_proof_rejects_entity_cooccurrence_without_relation():
    docs = [{"node_id": "d1", "title": "Alice", "index": 0}, {"node_id": "d2", "title": "Acme", "index": 1}]
    sentences = [
        {"node_id": "s1", "text": "Alice visited Acme.", "document_id": "d1", "document_index": 0, "sentence_index": 0},
        {"node_id": "s2", "text": "Acme was mentioned by Alice.", "document_id": "d2", "document_index": 1, "sentence_index": 0},
    ]
    question = "Where is the company founded by Alice based?"
    semantic = build_semantic_graph(question, docs, sentences)
    claims = extract_claims(question, semantic, sentences)
    proof = select_claim_proof(claims, sentences, token_counter=lambda x: len(str(x).split()))
    assert proof.contract_satisfied is False
