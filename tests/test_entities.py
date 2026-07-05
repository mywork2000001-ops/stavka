from bukmeker.entities import ScoringModel, build_seed_registry


def test_seed_registry_has_multiple_sports_and_countries():
    reg = build_seed_registry()
    assert len(reg.sports) >= 3
    assert len(reg.countries) >= 4


def test_leagues_for_sport_returns_only_matching_sport():
    reg = build_seed_registry()
    football_id = next(s.id for s in reg.sports.values() if s.name == "Football")
    leagues = reg.leagues_for_sport(football_id)
    assert len(leagues) >= 3
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

    premier_league = next(lg for lg in reg.leagues.values() if lg.name == "Premier League")
    assert reg.sport_of_league(premier_league.id).scoring_model == ScoringModel.POISSON_GOALS


def test_leagues_for_country_covers_multiple_countries():
    reg = build_seed_registry()
    for country in reg.countries.values():
        assert len(reg.leagues_for_country(country.id)) >= 1
