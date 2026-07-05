"""Historical match RESULTS (final scores), as opposed to `schema.CanonicalMatch`
which represents future fixtures + bookmaker odds.

This is the missing piece between "a math engine that computes optimal stakes
given a probability" and "a system that produces its own probability": fitting
`bukmeker.ratings.PoissonStrength` requires real past scores, not just team
names or upcoming odds. Reuses the same AI-mapping mechanism as the betting
schema (`ClaudeFieldMapper`/`AIDataConnector`), just targeting a different
canonical field list.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import FieldMapping, get_by_path

HISTORICAL_FIELDS = ["match_id", "league", "home_team", "away_team", "date", "home_goals", "away_goals"]


@dataclass(frozen=True)
class HistoricalMatch:
    match_id: str | None
    league: str | None
    home_team: str | None
    away_team: str | None
    date: str | None
    home_goals: int | None
    away_goals: int | None
    raw: dict


def apply_historical_mapping(record: dict, mapping: FieldMapping) -> HistoricalMatch:
    """Mirrors `schema.apply_mapping`, but for `HISTORICAL_FIELDS` and with
    goal counts coerced to `int` rather than odds to `float`. Also treats a
    non-string mapped path as unmappable (same untrusted-LLM-output guard as
    `schema.apply_mapping`)."""
    values: dict = {}
    for field_name in HISTORICAL_FIELDS:
        path = mapping.get(field_name)
        values[field_name] = get_by_path(record, path) if isinstance(path, str) and path else None

    for goals_field in ("home_goals", "away_goals"):
        raw_value = values.get(goals_field)
        if raw_value is not None:
            try:
                values[goals_field] = int(raw_value)
            except (TypeError, ValueError):
                values[goals_field] = None

    return HistoricalMatch(raw=record, **values)


def to_rating_arrays(
    matches: list[HistoricalMatch],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Filters to matches with a complete (team names + both scores) record and
    returns `(home_ids, away_ids, home_goals, away_goals, teams)`, ready for
    `bukmeker.ratings.PoissonStrength.fit()`. Incomplete records (the AI mapper
    couldn't locate a field, or a provider left it null) are silently excluded
    from fitting rather than crashing -- fitting on 950 of 1000 matches is
    normal; a `KeyError` on the first bad record is not useful."""
    complete = [
        m
        for m in matches
        if m.home_team and m.away_team and m.home_goals is not None and m.away_goals is not None
    ]
    if not complete:
        return np.array([]), np.array([]), np.array([]), np.array([]), []

    home_ids = np.array([m.home_team for m in complete])
    away_ids = np.array([m.away_team for m in complete])
    home_goals = np.array([m.home_goals for m in complete])
    away_goals = np.array([m.away_goals for m in complete])
    teams = sorted(set(home_ids) | set(away_ids))
    return home_ids, away_ids, home_goals, away_goals, teams
