from workflow_runtime.hotpot_bridge import (
    bridge_candidates_from_values,
    bridge_matches_any,
    extract_bridge_candidate,
    normalize_bridge,
)


def test_hotpot_bridge_extracts_entity_from_json_string_and_dict() -> None:
    assert extract_bridge_candidate('{"name":"Scott Derrickson"}') == "Scott Derrickson"
    assert extract_bridge_candidate({"title": "Ed Wood (film)", "text": "A film."}) == "Ed Wood (film)"


def test_hotpot_bridge_prefers_entity_fields_over_text() -> None:
    value = {"text": "A long sentence about Paris.", "label": "Paris"}

    assert extract_bridge_candidate(value) == "Paris"


def test_hotpot_bridge_candidates_deduplicate_after_canonical_extraction() -> None:
    candidates = bridge_candidates_from_values([
        '{"name":"Scott Derrickson"}',
        {"name": "Scott Derrickson"},
        {"title": "Ed Wood"},
    ])

    assert candidates == ["Scott Derrickson", "Ed Wood"]


def test_hotpot_bridge_match_handles_json_string_candidate() -> None:
    assert normalize_bridge('{"name":"Scott Derrickson"}') == "scott derrickson"
    assert bridge_matches_any('{"name":"Scott Derrickson"}', {"Scott Derrickson"})
