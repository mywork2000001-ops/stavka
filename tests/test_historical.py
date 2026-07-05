import numpy as np

from bukmeker.connectors.historical import (
    HISTORICAL_FIELDS,
    apply_historical_mapping,
    to_rating_arrays,
)
from bukmeker.connectors.schema import CANONICAL_FIELDS, FieldMapping


def _mapping(**overrides):
    paths = {f: None for f in HISTORICAL_FIELDS}
    paths.update(overrides)
    return FieldMapping(paths=paths)


def test_historical_fields_distinct_from_betting_canonical_fields():
    # sanity check the two schemas don't silently collapse into one
    assert set(HISTORICAL_FIELDS) != set(CANONICAL_FIELDS)
    assert "home_goals" in HISTORICAL_FIELDS
    assert "home_goals" not in CANONICAL_FIELDS


def test_apply_historical_mapping_coerces_goals_to_int():
    record = {
        "teams": {"home": "Arsenal", "away": "Chelsea"},
        "score": {"home": "2", "away": "1"},
        "fixture": {"date": "2026-01-15"},
    }
    mapping = _mapping(
        home_team="teams.home", away_team="teams.away",
        home_goals="score.home", away_goals="score.away", date="fixture.date",
    )
    match = apply_historical_mapping(record, mapping)
    assert match.home_team == "Arsenal"
    assert match.home_goals == 2
    assert match.away_goals == 1
    assert isinstance(match.home_goals, int)


def test_apply_historical_mapping_sets_none_for_nonnumeric_goals():
    record = {"score": {"home": "postponed", "away": None}}
    mapping = _mapping(home_goals="score.home", away_goals="score.away")
    match = apply_historical_mapping(record, mapping)
    assert match.home_goals is None
    assert match.away_goals is None


def test_apply_historical_mapping_treats_non_string_path_as_unmappable():
    record = {"a": 1}
    mapping = _mapping(home_team=123, away_team=["a", "b"])
    match = apply_historical_mapping(record, mapping)
    assert match.home_team is None
    assert match.away_team is None


def test_to_rating_arrays_filters_incomplete_records():
    matches = [
        apply_historical_mapping(
            {"h": "Arsenal", "a": "Chelsea", "hg": 2, "ag": 1},
            _mapping(home_team="h", away_team="a", home_goals="hg", away_goals="ag"),
        ),
        # missing away_goals -> must be excluded, not crash
        apply_historical_mapping(
            {"h": "Liverpool", "a": "Everton", "hg": 3},
            _mapping(home_team="h", away_team="a", home_goals="hg", away_goals="ag"),
        ),
    ]
    home_ids, away_ids, home_goals, away_goals, teams = to_rating_arrays(matches)
    assert len(home_ids) == 1
    assert home_ids[0] == "Arsenal"
    assert teams == ["Arsenal", "Chelsea"]


def test_to_rating_arrays_empty_input_returns_empty_arrays_not_crash():
    home_ids, away_ids, home_goals, away_goals, teams = to_rating_arrays([])
    assert len(home_ids) == 0
    assert teams == []


def test_to_rating_arrays_output_is_directly_usable_by_poisson_strength_fit():
    from bukmeker.ratings import PoissonStrength

    rng = np.random.default_rng(0)
    teams_pool = ["A", "B", "C", "D"]
    matches = []
    for _ in range(200):
        h, a = rng.choice(teams_pool, size=2, replace=False)
        matches.append(
            apply_historical_mapping(
                {"h": h, "a": a, "hg": int(rng.poisson(1.4)), "ag": int(rng.poisson(1.1))},
                _mapping(home_team="h", away_team="a", home_goals="hg", away_goals="ag"),
            )
        )

    home_ids, away_ids, home_goals, away_goals, teams = to_rating_arrays(matches)
    fitted = PoissonStrength.fit(home_ids, away_ids, home_goals, away_goals, teams)
    lam_home, lam_away = fitted.expected_goals("A", "B")
    assert lam_home > 0 and lam_away > 0
