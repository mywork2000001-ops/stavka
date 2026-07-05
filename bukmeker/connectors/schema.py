"""Canonical match schema and the helpers used to map arbitrary provider JSON
onto it (bukmeker.txt §1.4 attribute list, generalised across providers)."""

from __future__ import annotations

from dataclasses import dataclass

CANONICAL_FIELDS = [
    "match_id",
    "sport",
    "league",
    "home_team",
    "away_team",
    "start_time",
    "home_odds",
    "draw_odds",
    "away_odds",
]


@dataclass(frozen=True)
class FieldMapping:
    """canonical field name -> dotted path within a raw provider record
    (e.g. {"home_team": "teams.home.name"}); `None` means "not found"."""

    paths: dict[str, str | None]

    def get(self, field_name: str) -> str | None:
        return self.paths.get(field_name)


@dataclass(frozen=True)
class CanonicalMatch:
    match_id: str | None
    sport: str | None
    league: str | None
    home_team: str | None
    away_team: str | None
    start_time: str | None
    home_odds: float | None
    draw_odds: float | None
    away_odds: float | None
    raw: dict


def get_by_path(record, path: str):
    """Resolve a dotted path, e.g. 'teams.home.name' or 'odds.0.price'
    (integer segments index into lists)."""
    current = record
    for part in path.split("."):
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        else:
            return None
    return current


def find_record_list(raw, min_len: int = 1) -> list[dict]:
    """Heuristically locate the list of match-like records inside an arbitrary
    provider JSON payload. Many APIs wrap arrays under keys like "response",
    "data", "matches", "results" — this does a breadth-first search for the
    first list of dicts, so it works without knowing the provider in advance."""
    if isinstance(raw, list):
        return raw if raw and all(isinstance(x, dict) for x in raw) else []
    if not isinstance(raw, dict):
        return []

    queue: list = [raw]
    while queue:
        node = queue.pop(0)
        if isinstance(node, dict):
            for value in node.values():
                if isinstance(value, list) and len(value) >= min_len and all(
                    isinstance(x, dict) for x in value
                ):
                    return value
                if isinstance(value, (dict, list)):
                    queue.append(value)
        elif isinstance(node, list):
            queue.extend(item for item in node if isinstance(item, (dict, list)))
    return []


def apply_mapping(record: dict, mapping: FieldMapping) -> CanonicalMatch:
    values: dict = {}
    for field_name in CANONICAL_FIELDS:
        path = mapping.get(field_name)
        values[field_name] = get_by_path(record, path) if path else None

    for odds_field in ("home_odds", "draw_odds", "away_odds"):
        raw_value = values.get(odds_field)
        if raw_value is not None:
            try:
                values[odds_field] = float(raw_value)
            except (TypeError, ValueError):
                values[odds_field] = None

    return CanonicalMatch(raw=record, **values)
