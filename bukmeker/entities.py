"""Universal, sport/country-agnostic entity model.

Countries are the REAL, complete ISO 3166-1 list (all 249 officially assigned
countries/territories), sourced from `pycountry` -- a maintained, offline
package with no network calls at runtime, so "all countries in the world" is
a verifiable fact rather than a hand-typed guess.

Leagues/competitors are NOT exhaustive -- there is no honest way to hardcode
"every club in the world for every sport" (that requires a licensed data
provider, updated continuously). What's seeded here is a much larger, real
(not fictional) curated set: the top-flight football league + several
well-known real clubs for ~35 footballing nations, the complete NBA (30 real
teams), and a set of well-known real ATP players. Anything beyond this seed
is reached through `bukmeker.connectors` (live data via an AI-assisted
connector), not by hardcoding more entries here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pycountry


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
    id: int  # ISO 3166-1 numeric code, as int
    name: str
    iso_code: str  # ISO 3166-1 alpha-3


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

    def country_by_alpha3(self, iso_code: str) -> Country:
        return next(c for c in self.countries.values() if c.iso_code == iso_code)

    def all_countries_sorted(self) -> list[Country]:
        return sorted(self.countries.values(), key=lambda c: c.name)

    def leagues_for_sport(self, sport_id: int) -> list[League]:
        return [lg for lg in self.leagues.values() if lg.sport_id == sport_id]

    def leagues_for_country(self, country_id: int) -> list[League]:
        return [lg for lg in self.leagues.values() if lg.country_id == country_id]

    def leagues_for_country_and_sport(self, country_id: int, sport_id: int) -> list[League]:
        return [
            lg for lg in self.leagues.values() if lg.country_id == country_id and lg.sport_id == sport_id
        ]

    def competitors_for_league(self, league_id: int) -> list[Competitor]:
        return [c for c in self.competitors.values() if c.league_id == league_id]

    def sport_of_league(self, league_id: int) -> Sport:
        return self.sports[self.leagues[league_id].sport_id]

    def has_league_data(self, country_id: int) -> bool:
        return len(self.leagues_for_country(country_id)) > 0


# Real, verifiable, top-flight-only football data (league name + a handful of
# well-known real clubs) for a curated set of footballing nations, keyed by
# ISO 3166-1 alpha-3. Several countries have more than one notable league
# (e.g. GBR covers both England and Scotland, which are separate football
# associations despite sharing one ISO country entry).
_FOOTBALL_LEAGUES: dict[str, list[tuple[str, list[str]]]] = {
    "GBR": [
        ("Premier League (England)", ["Arsenal", "Chelsea", "Liverpool", "Manchester City",
                                       "Manchester United", "Tottenham Hotspur"]),
        ("Scottish Premiership", ["Celtic", "Rangers"]),
    ],
    "ESP": [("La Liga", ["Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla"])],
    "DEU": [("Bundesliga", ["Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen"])],
    "ITA": [("Serie A", ["Juventus", "AC Milan", "Inter Milan", "Napoli", "AS Roma"])],
    "FRA": [("Ligue 1", ["Paris Saint-Germain", "Marseille", "Lyon", "Monaco"])],
    "PRT": [("Primeira Liga", ["Benfica", "Porto", "Sporting CP"])],
    "NLD": [("Eredivisie", ["Ajax", "PSV Eindhoven", "Feyenoord"])],
    "BEL": [("Belgian Pro League", ["Club Brugge", "Anderlecht"])],
    "TUR": [("Super Lig", ["Galatasaray", "Fenerbahce", "Besiktas"])],
    "RUS": [("Russian Premier League", ["Zenit Saint Petersburg", "Spartak Moscow", "CSKA Moscow"])],
    "UKR": [("Ukrainian Premier League", ["Shakhtar Donetsk", "Dynamo Kyiv"])],
    "GRC": [("Super League Greece", ["Olympiacos", "Panathinaikos", "AEK Athens"])],
    "CHE": [("Swiss Super League", ["FC Basel", "Young Boys"])],
    "AUT": [("Austrian Bundesliga", ["Red Bull Salzburg", "Rapid Wien"])],
    "POL": [("Ekstraklasa", ["Legia Warsaw"])],
    "HRV": [("HNL", ["Dinamo Zagreb"])],
    "SRB": [("Serbian SuperLiga", ["Red Star Belgrade", "Partizan"])],
    "DNK": [("Danish Superliga", ["FC Copenhagen"])],
    "SWE": [("Allsvenskan", ["Malmo FF"])],
    "NOR": [("Eliteserien", ["Rosenborg"])],
    "BRA": [("Brasileirao Serie A", ["Flamengo", "Palmeiras", "Corinthians", "Sao Paulo", "Santos"])],
    "ARG": [("Primera Division", ["Boca Juniors", "River Plate"])],
    "USA": [("MLS", ["LA Galaxy", "Inter Miami", "Seattle Sounders"])],
    "MEX": [("Liga MX", ["Club America", "Chivas Guadalajara"])],
    "JPN": [("J1 League", ["Kashima Antlers", "Urawa Red Diamonds"])],
    "KOR": [("K League 1", ["Jeonbuk Hyundai Motors"])],
    "SAU": [("Saudi Pro League", ["Al Hilal", "Al Nassr", "Al Ittihad"])],
    "CHN": [("Chinese Super League", ["Shanghai Port"])],
    "AUS": [("A-League", ["Melbourne Victory", "Sydney FC"])],
    "EGY": [("Egyptian Premier League", ["Al Ahly", "Zamalek"])],
    "MAR": [("Botola Pro", ["Raja Casablanca", "Wydad Casablanca"])],
    "ZAF": [("Premier Soccer League", ["Kaizer Chiefs", "Orlando Pirates"])],
    "COL": [("Categoria Primera A", ["Millonarios", "Atletico Nacional"])],
    "CHL": [("Primera Division de Chile", ["Colo-Colo", "Universidad de Chile"])],
    "URY": [("Primera Division de Uruguay", ["Penarol", "Nacional"])],
    "IND": [("Indian Super League", ["Mumbai City FC", "Bengaluru FC"])],
    "CIV": [("Ligue 1 (Cote d'Ivoire)", ["ASEC Mimosas", "Africa Sports"])],
}

# All 30 NBA teams -- this list genuinely is exhaustive for the league.
_NBA_TEAMS = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls",
    "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons",
    "Golden State Warriors", "Houston Rockets", "Indiana Pacers", "LA Clippers",
    "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat", "Milwaukee Bucks",
    "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
    "Utah Jazz", "Washington Wizards",
]

# Well-known real ATP players -- illustrative, not a live/current ranking
# (rankings change continuously; this is not a substitute for live data).
_ATP_PLAYERS = [
    "Novak Djokovic", "Carlos Alcaraz", "Jannik Sinner", "Daniil Medvedev",
    "Alexander Zverev", "Stefanos Tsitsipas", "Andrey Rublev", "Casper Ruud",
    "Rafael Nadal", "Andy Murray",
]


def build_seed_registry() -> EntityRegistry:
    """All real ISO countries, plus a curated (real, not exhaustive) set of
    football leagues/clubs, the complete NBA, and well-known ATP players."""
    reg = EntityRegistry()

    for iso_country in pycountry.countries:
        reg.add_country(
            Country(id=int(iso_country.numeric), name=iso_country.name, iso_code=iso_country.alpha_3)
        )

    football = Sport(1, "Football", ScoringModel.POISSON_GOALS)
    basketball = Sport(2, "Basketball", ScoringModel.POINT_SPREAD)
    tennis = Sport(3, "Tennis", ScoringModel.SET_BASED)
    for s in (football, basketball, tennis):
        reg.add_sport(s)

    next_league_id = 1
    next_competitor_id = 1

    for alpha3, league_specs in _FOOTBALL_LEAGUES.items():
        country = reg.country_by_alpha3(alpha3)
        for league_name, clubs in league_specs:
            league = League(next_league_id, football.id, country.id, league_name)
            reg.add_league(league)
            next_league_id += 1
            for club_name in clubs:
                reg.add_competitor(Competitor(next_competitor_id, league.id, club_name, CompetitorKind.CLUB))
                next_competitor_id += 1

    usa = reg.country_by_alpha3("USA")

    nba = League(next_league_id, basketball.id, usa.id, "NBA")
    reg.add_league(nba)
    next_league_id += 1
    for team_name in _NBA_TEAMS:
        reg.add_competitor(Competitor(next_competitor_id, nba.id, team_name, CompetitorKind.CLUB))
        next_competitor_id += 1

    atp_tour = League(next_league_id, tennis.id, usa.id, "ATP Tour")
    reg.add_league(atp_tour)
    next_league_id += 1
    for player_name in _ATP_PLAYERS:
        reg.add_competitor(Competitor(next_competitor_id, atp_tour.id, player_name, CompetitorKind.PLAYER))
        next_competitor_id += 1

    return reg
