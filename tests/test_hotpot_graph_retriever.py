from workflow_runtime.hotpot_graph_retriever import (
    GraphSeedRetriever,
    MultiPathExplorer,
    PathReranker,
    QueryGuidedGraphRetriever,
)


def _facts():
    return [
        {"fact_id": "f1", "title": "Corliss Archer", "text": "Corliss Archer was portrayed by Shirley Temple."},
        {"fact_id": "f2", "title": "Shirley Temple", "text": "Shirley Temple held the position of Chief of Protocol."},
        {"fact_id": "f3", "title": "Unrelated", "text": "Unrelated was released in 1998."},
    ]


def test_seed_retriever_keeps_entity_and_relation_seeds():
    seeds = GraphSeedRetriever().select("What position was held by the woman who portrayed Corliss Archer?", _facts())
    assert seeds[0].fact_id in {"f1", "f2"}
    assert any("entity_link" in seed.sources for seed in seeds)


def test_graph_diffusion_recovers_connected_second_hop():
    result = QueryGuidedGraphRetriever(alpha=0.35, beam_size=8).retrieve(
        "What government position was held by the woman who portrayed Corliss Archer?",
        _facts(),
    )
    paths = result["paths"]
    assert paths
    assert any(path["evidence_ids"] == ("f1", "f2") or path["evidence_ids"] == ("f2", "f1") for path in paths)
    assert all(0.0 <= path["uncertainty"] <= 1.0 for path in paths)


def test_graph_retriever_is_recall_only_and_returns_provenance():
    result = QueryGuidedGraphRetriever().retrieve("Who portrayed Corliss Archer?", _facts())
    assert result["retrieval_method"] == "query_guided_ppr_beam"
    assert result["seeds"]
    assert all("metadata" in path for path in result["paths"])


def test_path_reranker_adds_grounding_signal_without_verifying_answer():
    result = QueryGuidedGraphRetriever().retrieve("Who portrayed Corliss Archer?", _facts())
    from workflow_runtime.hotpot_graph_retriever import EvidencePath
    reranked = PathReranker().rerank("Who portrayed Corliss Archer?", [
        EvidencePath(**path) for path in result["paths"]
    ], _facts())
    assert reranked
    assert "grounding_score" in reranked[0].metadata
