from bukmeker.entities import CompetitorKind, ScoringModel, build_seed_registry


def test_seed_registry_has_all_real_iso_countries():
    reg = build_seed_registry()
    # ISO 3166-1 currently assigns 249 countries/territories (via pycountry) --
    # this is a completeness check against the real standard, not a guess.
    assert len(reg.countries) == 249


def test_country_names_and_iso_codes_are_populated_and_unique():
    reg = build_seed_registry()
    iso_codes = [c.iso_code for c in reg.countries.values()]
    assert len(iso_codes) == len(set(iso_codes))
    assert all(len(code) == 3 for code in iso_codes)
    assert all(c.name for c in reg.countries.values())


def test_seed_registry_has_three_sports():
    reg = build_seed_registry()
    assert len(reg.sports) == 3
    names = {s.name for s in reg.sports.values()}
    assert names == {"Football", "Basketball", "Tennis"}


def test_leagues_for_sport_returns_only_matching_sport():
    reg = build_seed_registry()
    football_id = next(s.id for s in reg.sports.values() if s.name == "Football")
    leagues = reg.leagues_for_sport(football_id)
    assert len(leagues) >= 35
    assert all(lg.sport_id == football_id for lg in leagues)


def test_competitors_for_league_are_non_empty_for_every_league():
    reg = build_seed_registry()
    for league in reg.leagues.values():
        competitors = reg.competitors_for_league(league.id)
        assert len(competitors) > 0, f"league {league.name} has no competitors"


def test_sport_of_league_matches_scoring_model():
    reg = build_seed_registry()
    nba = next(lg for lg in reg.leagues.values() if lg.name == "NBA")
    assert reg.sport_of_league(nba.id).scoring_model == ScoringModel.POINT_SPREAD

    atp = next(lg for lg in reg.leagues.values() if lg.name == "ATP Tour")
    assert reg.sport_of_league(atp.id).scoring_model == ScoringModel.SET_BASED

    premier_league = next(lg for lg in reg.leagues.values() if "Premier League" in lg.name)
    assert reg.sport_of_league(premier_league.id).scoring_model == ScoringModel.POISSON_GOALS


def test_nba_league_is_exhaustive_with_thirty_teams():
    reg = build_seed_registry()
    nba = next(lg for lg in reg.leagues.values() if lg.name == "NBA")
    teams = reg.competitors_for_league(nba.id)
    assert len(teams) == 30
    assert all(c.kind == CompetitorKind.CLUB for c in teams)


def test_atp_tour_players_are_marked_as_players_not_clubs():
    reg = build_seed_registry()
    atp = next(lg for lg in reg.leagues.values() if lg.name == "ATP Tour")
    players = reg.competitors_for_league(atp.id)
    assert len(players) > 0
    assert all(c.kind == CompetitorKind.PLAYER for c in players)


def test_country_by_alpha3_resolves_known_codes():
    reg = build_seed_registry()
    assert reg.country_by_alpha3("USA").name == "United States"
    assert reg.country_by_alpha3("GBR").iso_code == "GBR"


def test_gbr_has_both_english_and_scottish_football_leagues():
    reg = build_seed_registry()
    gbr = reg.country_by_alpha3("GBR")
    league_names = {lg.name for lg in reg.leagues_for_country(gbr.id)}
    assert any("England" in name for name in league_names)
    assert any("Scottish" in name for name in league_names)


def test_most_countries_have_no_seeded_league_data():
    # The seed is a curated real subset, not an exhaustive club database --
    # most of the 249 real countries should have zero leagues seeded.
    reg = build_seed_registry()
    countries_without_data = [c for c in reg.countries.values() if not reg.has_league_data(c.id)]
    assert len(countries_without_data) > 200


def test_leagues_for_country_and_sport_filters_both_dimensions():
    reg = build_seed_registry()
    usa = reg.country_by_alpha3("USA")
    football_id = next(s.id for s in reg.sports.values() if s.name == "Football")
    basketball_id = next(s.id for s in reg.sports.values() if s.name == "Basketball")

    football_leagues = reg.leagues_for_country_and_sport(usa.id, football_id)
    basketball_leagues = reg.leagues_for_country_and_sport(usa.id, basketball_id)
    assert {lg.name for lg in football_leagues} == {"MLS"}
    assert {lg.name for lg in basketball_leagues} == {"NBA"}


def test_all_countries_sorted_is_alphabetical_by_name():
    reg = build_seed_registry()
    names = [c.name for c in reg.all_countries_sorted()]
    assert names == sorted(names)
