"""Direct unit tests for dashboard business logic that doesn't need a running
Streamlit script context -- faster and more precise than driving it through
AppTest, and covers cases AppTest can't (st.data_editor isn't drivable via
AppTest in this Streamlit version)."""

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from bukmeker.dashboard import parse_bet_rows  # noqa: E402


def _rows(*records):
    return pd.DataFrame.from_records(records)


def test_parse_bet_rows_all_valid_rows_produce_no_skips():
    df = _rows(
        {"bet_id": 1, "match_id": 100, "league_id": 1, "team_ids": "1,2", "prob": 0.55, "odds": 2.0},
        {"bet_id": 2, "match_id": 101, "league_id": 1, "team_ids": "3,4", "prob": 0.4, "odds": 2.5},
    )
    candidates, skipped = parse_bet_rows(df)
    assert len(candidates) == 2
    assert skipped == []
    assert candidates[0].team_ids == (1, 2)


def test_parse_bet_rows_reports_skipped_row_with_reason_instead_of_dropping_silently():
    df = _rows(
        {"bet_id": 1, "match_id": 100, "league_id": 1, "team_ids": "1,2", "prob": 0.55, "odds": 2.0},
        {"bet_id": 2, "match_id": 101, "league_id": 1, "team_ids": "x,y", "prob": 0.4, "odds": 2.5},
    )
    candidates, skipped = parse_bet_rows(df)
    assert len(candidates) == 1  # only the valid row
    assert len(skipped) == 1
    row_index, reason = skipped[0]
    assert row_index == 1
    assert reason  # a non-empty explanation, not a silently swallowed error


def test_parse_bet_rows_skips_row_with_missing_required_field():
    df = _rows({"bet_id": 1, "match_id": 100, "league_id": 1, "team_ids": "1,2", "prob": None, "odds": 2.0})
    candidates, skipped = parse_bet_rows(df)
    assert candidates == []
    assert len(skipped) == 1


def test_parse_bet_rows_handles_single_team_id_without_comma():
    df = _rows({"bet_id": 1, "match_id": 100, "league_id": 1, "team_ids": "5", "prob": 0.5, "odds": 2.0})
    candidates, skipped = parse_bet_rows(df)
    assert skipped == []
    assert candidates[0].team_ids == (5,)


def test_parse_bet_rows_empty_dataframe_returns_empty_results():
    df = pd.DataFrame(columns=["bet_id", "match_id", "league_id", "team_ids", "prob", "odds"])
    candidates, skipped = parse_bet_rows(df)
    assert candidates == []
    assert skipped == []
