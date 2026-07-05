"""Sync the entity registry (leagues/competitors) from live data pulled
through an `AIDataConnector`.

This is the actual mechanism for growing beyond the curated seed in
`bukmeker.entities`: real breadth comes from pointing a connector at a real
data source (by API key), not from hardcoding more names into the seed.
"""

from __future__ import annotations

from dataclasses import dataclass

from bukmeker.entities import Competitor, CompetitorKind, EntityRegistry, League

from .schema import CanonicalMatch


@dataclass(frozen=True)
class SyncReport:
    matches_processed: int
    leagues_added: int
    competitors_added: int
    skipped_incomplete: int


def sync_registry_from_matches(
    registry: EntityRegistry,
    matches: list[CanonicalMatch],
    sport_id: int,
    fallback_country_id: int,
) -> SyncReport:
    """Idempotently merge league/team names observed in `matches` into
    `registry`, in place. Leagues are matched by `(sport_id, league name)`;
    competitors by `(league_id, name)` -- running this twice on the same
    `matches` adds nothing the second time.

    `CanonicalMatch` carries a league *name* but not a country (most sports
    data providers don't expose one uniformly), so a newly discovered league
    is placed under `fallback_country_id` (e.g. the country associated with
    the data source/API key) rather than guessed from the team names.
    Matches missing `home_team`/`away_team` (the AI field mapper could not
    locate them in the source schema) are skipped and counted, not silently
    dropped.
    """
    next_league_id = max(registry.leagues, default=0) + 1
    next_competitor_id = max(registry.competitors, default=0) + 1
    leagues_added = 0
    competitors_added = 0
    skipped = 0

    league_by_key = {(lg.sport_id, lg.name): lg for lg in registry.leagues.values()}
    competitor_by_key = {(c.league_id, c.name): c for c in registry.competitors.values()}

    for match in matches:
        if not match.home_team or not match.away_team:
            skipped += 1
            continue

        league_name = match.league or "Unknown league (synced)"
        league_key = (sport_id, league_name)
        league = league_by_key.get(league_key)
        if league is None:
            league = League(next_league_id, sport_id, fallback_country_id, league_name)
            registry.add_league(league)
            league_by_key[league_key] = league
            next_league_id += 1
            leagues_added += 1

        for team_name in (match.home_team, match.away_team):
            competitor_key = (league.id, team_name)
            if competitor_key in competitor_by_key:
                continue
            competitor = Competitor(next_competitor_id, league.id, team_name, CompetitorKind.CLUB)
            registry.add_competitor(competitor)
            competitor_by_key[competitor_key] = competitor
            next_competitor_id += 1
            competitors_added += 1

    return SyncReport(
        matches_processed=len(matches),
        leagues_added=leagues_added,
        competitors_added=competitors_added,
        skipped_incomplete=skipped,
    )
