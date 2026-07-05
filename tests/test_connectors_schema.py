from bukmeker.connectors.schema import (
    CANONICAL_FIELDS,
    FieldMapping,
    apply_mapping,
    find_record_list,
    get_by_path,
)


def test_get_by_path_resolves_nested_dict():
    record = {"teams": {"home": {"name": "Arsenal"}}}
    assert get_by_path(record, "teams.home.name") == "Arsenal"


def test_get_by_path_resolves_list_index():
    record = {"odds": [{"price": 1.75}, {"price": 3.60}]}
    assert get_by_path(record, "odds.1.price") == 3.60


def test_get_by_path_returns_none_for_missing_key():
    record = {"a": {"b": 1}}
    assert get_by_path(record, "a.c") is None


def test_get_by_path_returns_none_for_out_of_range_index():
    record = {"odds": [{"price": 1.75}]}
    assert get_by_path(record, "odds.5.price") is None


def test_find_record_list_locates_nested_array():
    raw = {"response": {"data": {"matches": [{"id": 1}, {"id": 2}]}}}
    records = find_record_list(raw)
    assert records == [{"id": 1}, {"id": 2}]


def test_find_record_list_handles_top_level_array():
    raw = [{"id": 1}, {"id": 2}]
    assert find_record_list(raw) == raw


def test_find_record_list_returns_empty_for_no_arrays():
    assert find_record_list({"status": "ok"}) == []


def test_apply_mapping_produces_canonical_match_with_typed_odds():
    record = {
        "fixture": {"id": 42},
        "teams": {"home": {"name": "Arsenal"}, "away": {"name": "Chelsea"}},
        "odds": {"home": "1.75", "draw": "3.6", "away": "4.75"},
    }
    mapping = FieldMapping(
        paths={
            "match_id": "fixture.id",
            "sport": None,
            "league": None,
            "home_team": "teams.home.name",
            "away_team": "teams.away.name",
            "start_time": None,
            "home_odds": "odds.home",
            "draw_odds": "odds.draw",
            "away_odds": "odds.away",
        }
    )
    match = apply_mapping(record, mapping)
    assert match.home_team == "Arsenal"
    assert match.away_team == "Chelsea"
    assert match.home_odds == 1.75
    assert match.draw_odds == 3.6
    assert match.away_odds == 4.75
    assert match.sport is None
    assert match.raw is record


def test_all_canonical_fields_covered_by_mapping_with_nulls():
    mapping = FieldMapping(paths={f: None for f in CANONICAL_FIELDS})
    match = apply_mapping({"anything": True}, mapping)
    assert match.match_id is None
    assert match.home_odds is None


def test_apply_mapping_treats_non_string_hallucinated_path_as_unmappable():
    # ClaudeFieldMapper's output is untrusted LLM structured output -- a
    # non-string "path" (int, list, dict) must not crash `path.split(".")`.
    record = {"teams": {"home": {"name": "Arsenal"}}}
    mapping = FieldMapping(
        paths={
            **{f: None for f in CANONICAL_FIELDS},
            "home_team": 123,
            "away_team": ["teams", "home", "name"],
            "match_id": {"nested": "object"},
        }
    )
    match = apply_mapping(record, mapping)
    assert match.home_team is None
    assert match.away_team is None
    assert match.match_id is None
