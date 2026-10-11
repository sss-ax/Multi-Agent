import json

from workflow_runtime import GraphStore
from workflow_runtime.hotpot_representation import (
    build_hotpot_semantic_fragments,
    build_hotpot_chain_view,
    build_hotpot_chain_view_from_facts,
    render_hotpot_chain_view,
    render_hotpot_discriminative_chain_static,
    render_hotpot_semantic_fragments,
)


def test_hotpot_chain_view_from_graph_is_lossless_structured_view() -> None:
    store = GraphStore()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="supporting_fact_1",
        node_type="supporting_fact",
        content=json.dumps({
            "title": "Scott Derrickson",
            "sent_id": 0,
            "text": "Scott Derrickson is an American film director.",
        }),
        owner="dataset",
    )
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="supporting_fact_2",
        node_type="supporting_fact",
        content=json.dumps({
            "title": "Ed Wood",
            "sent_id": 0,
            "text": "Ed Wood was an American filmmaker.",
        }),
        owner="dataset",
    )
    candidate = {
        "candidate_id": "c1",
        "value": {
            "answer": "yes",
            "bridge_entity": "Scott Derrickson",
            "cited_fact_ids": ["supporting_fact_1", "supporting_fact_2"],
        },
    }

    view = build_hotpot_chain_view(candidate, store, "Were Scott Derrickson and Ed Wood of the same nationality?")

    assert view.candidate_id == "c1"
    assert view.bridge_entity == "Scott Derrickson"
    assert view.hop1_fact_ids == ["supporting_fact_1"]
    assert view.hop1_key_sentence == "Scott Derrickson is an American film director."
    assert view.hop2_fact_ids == ["supporting_fact_2"]
    assert view.hop2_key_sentence == "Ed Wood was an American filmmaker."
    assert view.derived_answer == "yes"
    assert view.answer_type == "yes_no"


def test_hotpot_chain_view_renderer_uses_structured_v1_format() -> None:
    facts = [
        {"title": "Scott Derrickson", "sent_id": 0, "text": "Scott Derrickson is an American film director."},
        {"title": "Ed Wood", "sent_id": 0, "text": "Ed Wood was an American filmmaker."},
    ]
    view = build_hotpot_chain_view_from_facts(
        candidate_id="c3",
        bridge_entity="Scott Derrickson",
        question="Were Scott Derrickson and Ed Wood of the same nationality?",
        facts=facts,
        hop1_fact_ids=["supporting_fact_1"],
        hop2_fact_ids=["supporting_fact_2"],
        derived_answer="yes",
    )

    rendered = render_hotpot_chain_view(view, include_hop1=True, include_hop2=True, include_answer=True)

    assert "[CANDIDATE c3]" in rendered
    assert "Bridge:\nScott Derrickson" in rendered
    assert "fact_id: supporting_fact_1" in rendered
    assert "evidence: Scott Derrickson is an American film director." in rendered
    assert "Derived answer:\nyes" in rendered
    assert "Answer alignment:" in rendered
    assert "relation:" not in rendered


def test_hotpot_semantic_fragments_split_chain_without_graph_nodes() -> None:
    facts = [
        {"title": "Scott Derrickson", "sent_id": 0, "text": "Scott Derrickson is an American film director."},
        {"title": "Ed Wood", "sent_id": 0, "text": "Ed Wood was an American filmmaker."},
    ]
    view = build_hotpot_chain_view_from_facts(
        candidate_id="c3",
        bridge_entity="Scott Derrickson",
        question="Were Scott Derrickson and Ed Wood of the same nationality?",
        facts=facts,
        hop1_fact_ids=["supporting_fact_1"],
        hop2_fact_ids=["supporting_fact_2"],
        derived_answer="yes",
    )

    fragments = [fragment.as_dict() for fragment in build_hotpot_semantic_fragments(
        view,
        "Were Scott Derrickson and Ed Wood of the same nationality?",
    )]

    assert [fragment["type"] for fragment in fragments] == [
        "bridge_identity",
        "hop1_evidence",
        "hop2_evidence",
        "derived_answer",
        "answer_alignment",
        "contradiction_signal",
    ]
    assert fragments[0] == {
        "type": "bridge_identity",
        "candidate_id": "c3",
        "bridge": "Scott Derrickson",
    }
    assert fragments[1]["fact_id"] == "supporting_fact_1"
    assert fragments[1]["text"] == "Scott Derrickson is an American film director."
    assert fragments[4]["question_target"] == "same nationality"
    assert fragments[4]["alignment"] == "supported"
    assert fragments[5]["value"] is False


def test_hotpot_semantic_fragment_renderer_groups_candidate_fragments() -> None:
    facts = [
        {"title": "Scott Derrickson", "sent_id": 0, "text": "Scott Derrickson is an American film director."},
        {"title": "Ed Wood", "sent_id": 0, "text": "Ed Wood was an American filmmaker."},
    ]
    view = build_hotpot_chain_view_from_facts(
        candidate_id="c3",
        bridge_entity="Scott Derrickson",
        question="Were Scott Derrickson and Ed Wood of the same nationality?",
        facts=facts,
        hop1_fact_ids=["supporting_fact_1"],
        hop2_fact_ids=["supporting_fact_2"],
        derived_answer="yes",
    )

    rendered = render_hotpot_semantic_fragments(view, "Were Scott Derrickson and Ed Wood of the same nationality?")

    assert "[CANDIDATE c3]" in rendered
    assert '"type":"bridge_identity"' in rendered
    assert '"type":"hop1_evidence"' in rendered
    assert '"type":"contradiction_signal"' in rendered
    assert "candidate_hop1_node" not in rendered


def test_hotpot_discriminative_chain_static_uses_compact_hop_clues() -> None:
    facts = [
        {"title": "Scott Derrickson", "sent_id": 0, "text": "Scott Derrickson is an American film director."},
        {"title": "Ed Wood", "sent_id": 0, "text": "Ed Wood was an American filmmaker."},
    ]
    view = build_hotpot_chain_view_from_facts(
        candidate_id="c3",
        bridge_entity="Scott Derrickson",
        question="Were Scott Derrickson and Ed Wood of the same nationality?",
        facts=facts,
        hop1_fact_ids=["supporting_fact_1"],
        hop2_fact_ids=["supporting_fact_2"],
        derived_answer="yes",
    )

    rendered = render_hotpot_discriminative_chain_static(
        view,
        "Were Scott Derrickson and Ed Wood of the same nationality?",
    )

    assert "[CANDIDATE c3]" in rendered
    assert "bridge: Scott Derrickson" in rendered
    assert "hop1 supporting_fact_1: Scott Derrickson: an American film director" in rendered
    assert "hop2 supporting_fact_2: Ed Wood: an American filmmaker" in rendered
    assert "answer_alignment" not in rendered
    assert "contradiction_signal" not in rendered
