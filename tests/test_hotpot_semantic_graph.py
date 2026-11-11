from baselines.aflow_graph_adapter import AFlowGraphSession
from baselines.hotpot_semantic_graph import build_semantic_graph, normalize_entity, select_semantic_paths


def _source():
    documents = [
        {"node_id": "d1", "title": "Alice", "index": 0},
        {"node_id": "d2", "title": "Acme Corporation", "index": 1},
    ]
    sentences = [
        {"node_id": "s1", "text": "Alice founded Acme Corporation.", "document_id": "d1", "document_index": 0, "sentence_index": 0},
        {"node_id": "s2", "text": "Acme Corporation is located in Paris.", "document_id": "d2", "document_index": 1, "sentence_index": 0},
    ]
    return documents, sentences


def test_semantic_entity_normalization_and_title_nodes_are_deterministic():
    documents, sentences = _source()
    left = build_semantic_graph("Where is Acme Corporation located?", documents, sentences)
    right = build_semantic_graph("Where is Acme Corporation located?", documents, sentences)
    assert left == right
    assert normalize_entity("Acme Corporation (company)") == "acme corporation"
    assert len(left.title_entities) == 2


def test_semantic_bridge_path_is_complete_and_cross_document():
    documents, sentences = _source()
    graph = build_semantic_graph("Where is Acme Corporation located?", documents, sentences)
    selected = select_semantic_paths(graph, documents, sentences, token_counter=lambda text: len(str(text).split()))
    assert selected.contract_satisfied
    assert selected.selected_paths[0].path_type in {"bridge", "comparison"}
    assert set(selected.selected_document_ids) == {"d1", "d2"}


def test_semantic_graph_materializes_public_entity_edges():
    context = [
        ["Alice", ["Alice founded Acme Corporation."]],
        ["Acme Corporation", ["Acme Corporation is located in Paris."]],
    ]
    question = "Where is Acme Corporation located?"
    session = AFlowGraphSession(
        task_id="semantic", task=question, dataset="HotpotQA", workflow_round=3,
        profile_explicit=True, context_mode="graph_minctx", context_policy="semantic_minimal",
        context_recovery="semantic_adaptive", task_metadata={"question": question, "context": context},
    )
    state = session.store.snapshot()
    assert any(node.type == "entity_source" for node in state.nodes.values())
    assert {edge.relation for edge in state.edges} >= {"query_mentions_entity", "title_identifies_entity", "sentence_mentions_entity"}
    assert not any("supporting_facts" in str(node.content) for node in state.nodes.values())


def test_semantic_adaptive_uses_s0_and_reuses_plan_cache():
    context = [
        ["Alice", ["Alice founded Acme Corporation."]],
        ["Acme Corporation", ["Acme Corporation is located in Paris."]],
    ]
    question = "Where is Acme Corporation located?"
    original = "Context: " + "\n".join(" ".join(x[1]) for x in context) + f"\n\nQuestion: {question}\n\nAnswer:"
    session = AFlowGraphSession(
        task_id="semantic-run", task=question, dataset="HotpotQA", workflow_round=3,
        profile_explicit=True, context_mode="graph_minctx", context_policy="semantic_minimal",
        context_recovery="semantic_adaptive", task_metadata={"question": question, "context": context},
    )
    first = session.prepare_operator_input(operator="AnswerGenerate", kwargs={"input": original}, communication_mode="graph_minctx")
    second = session.prepare_operator_input(operator="AnswerGenerate", kwargs={"input": original}, communication_mode="graph_minctx")
    assert first.kwargs == second.kwargs
    assert session._context_plans[-1].effective_level in {"S0", "S1"}
    assert session._context_plans[-1].cache_hit is True


def test_semantic_shadow_keeps_dependency_k4_input():
    context = [["Alice", ["Alice founded Acme Corporation."]], ["Acme Corporation", ["Acme Corporation is located in Paris."]]]
    question = "Where is Acme Corporation located?"
    original = "Context: " + "\n".join(" ".join(x[1]) for x in context) + f"\n\nQuestion: {question}\n\nAnswer:"
    common = dict(task=question, dataset="HotpotQA", workflow_round=3, profile_explicit=True, context_mode="graph_minctx", context_top_k_documents=4, task_metadata={"question": question, "context": context})
    baseline = AFlowGraphSession(task_id="baseline", context_policy="dependency_minimal", **common)
    shadow = AFlowGraphSession(task_id="shadow", context_policy="semantic_shadow", **common)
    left = baseline.prepare_operator_input(operator="AnswerGenerate", kwargs={"input": original}, communication_mode="graph_minctx")
    right = shadow.prepare_operator_input(operator="AnswerGenerate", kwargs={"input": original}, communication_mode="graph_minctx")
    assert left.kwargs == right.kwargs
