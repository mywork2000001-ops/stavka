"""Universal, sport/country-agnostic entity model.

Design decision: the platform does not hard-code "all clubs in the world" — that
is a data-ingestion problem, not a schema problem. Instead this module defines a
generic `Sport -> Country -> League -> Competitor` hierarchy that any sport, any
country, and any competition (team-based or individual) fits into. Real-world
breadth ("all countries, all clubs") is achieved by feeding this schema from live
data connectors (see `bukmeker.connectors`), not by enumerating entities in code.

The seed data below is a small, illustrative multi-sport / multi-country sample
used by the demo and tests — not a claim of exhaustive coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ScoringModel(str, Enum):
    """Which probability model family applies to a sport."""

    POISSON_GOALS = "poisson_goals"  # low-scoring team sports: football, hockey, handball
    POINT_SPREAD = "point_spread"  # high-scoring team sports: basketball, American football
    SET_BASED = "set_based"  # individual/racket sports decided by a race to N sets: tennis, volleyball


class CompetitorKind(str, Enum):
    CLUB = "club"
    PLAYER = "player"
    NATIONAL_TEAM = "national_team"


@dataclass(frozen=True)
class Sport:
    id: int
    name: str
    scoring_model: ScoringModel


@dataclass(frozen=True)
class Country:
    id: int
    name: str
    iso_code: str


@dataclass(frozen=True)
class League:
    id: int
    sport_id: int
    country_id: int
    name: str


@dataclass(frozen=True)
class Competitor:
    id: int
    league_id: int
    name: str
    kind: CompetitorKind


@dataclass
class EntityRegistry:
    """In-memory lookup over the Sport/Country/League/Competitor hierarchy."""

    sports: dict[int, Sport] = field(default_factory=dict)
    countries: dict[int, Country] = field(default_factory=dict)
    leagues: dict[int, League] = field(default_factory=dict)
    competitors: dict[int, Competitor] = field(default_factory=dict)

    def add_sport(self, sport: Sport) -> None:
        self.sports[sport.id] = sport

    def add_country(self, country: Country) -> None:
        self.countries[country.id] = country

    def add_league(self, league: League) -> None:
        self.leagues[league.id] = league

    def add_competitor(self, competitor: Competitor) -> None:
        self.competitors[competitor.id] = competitor

    def leagues_for_sport(self, sport_id: int) -> list[League]:
        return [lg for lg in self.leagues.values() if lg.sport_id == sport_id]

    def leagues_for_country(self, country_id: int) -> list[League]:
        return [lg for lg in self.leagues.values() if lg.country_id == country_id]

    def competitors_for_league(self, league_id: int) -> list[Competitor]:
        return [c for c in self.competitors.values() if c.league_id == league_id]

    def sport_of_league(self, league_id: int) -> Sport:
        return self.sports[self.leagues[league_id].sport_id]


def build_seed_registry() -> EntityRegistry:
    """A small illustrative multi-sport, multi-country registry (not exhaustive)."""
    reg = EntityRegistry()

    football = Sport(1, "Football", ScoringModel.POISSON_GOALS)
    basketball = Sport(2, "Basketball", ScoringModel.POINT_SPREAD)
    tennis = Sport(3, "Tennis", ScoringModel.SET_BASED)
    for s in (football, basketball, tennis):
        reg.add_sport(s)

    england = Country(1, "England", "ENG")
    spain = Country(2, "Spain", "ESP")
    germany = Country(3, "Germany", "GER")
    brazil = Country(4, "Brazil", "BRA")
    usa = Country(5, "USA", "USA")
    for c in (england, spain, germany, brazil, usa):
        reg.add_country(c)

    premier_league = League(1, football.id, england.id, "Premier League")
    la_liga = League(2, football.id, spain.id, "La Liga")
    bundesliga = League(3, football.id, germany.id, "Bundesliga")
    brasileirao = League(4, football.id, brazil.id, "Brasileirao Serie A")
    nba = League(5, basketball.id, usa.id, "NBA")
    atp_tour = League(6, tennis.id, usa.id, "ATP Tour")
    for lg in (premier_league, la_liga, bundesliga, brasileirao, nba, atp_tour):
        reg.add_league(lg)

    seed_competitors = [
        (1, premier_league.id, "Arsenal"),
        (2, premier_league.id, "Chelsea"),
        (3, premier_league.id, "Liverpool"),
        (4, premier_league.id, "Manchester City"),
        (5, la_liga.id, "Real Madrid"),
        (6, la_liga.id, "Barcelona"),
        (7, la_liga.id, "Atletico Madrid"),
        (8, bundesliga.id, "Bayern Munich"),
        (9, bundesliga.id, "Borussia Dortmund"),
        (10, brasileirao.id, "Flamengo"),
        (11, brasileirao.id, "Palmeiras"),
        (12, nba.id, "Los Angeles Lakers"),
        (13, nba.id, "Boston Celtics"),
        (14, nba.id, "Golden State Warriors"),
        (15, atp_tour.id, "Novak Djokovic"),
        (16, atp_tour.id, "Carlos Alcaraz"),
    ]
    kind_by_league = {nba.id: CompetitorKind.CLUB, atp_tour.id: CompetitorKind.PLAYER}
    for cid, league_id, name in seed_competitors:
        kind = kind_by_league.get(league_id, CompetitorKind.CLUB)
        reg.add_competitor(Competitor(cid, league_id, name, kind))

    return reg
