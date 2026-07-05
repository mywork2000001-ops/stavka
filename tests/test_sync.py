from bukmeker.connectors.schema import CanonicalMatch
from bukmeker.connectors.sync import sync_registry_from_matches
from bukmeker.entities import build_seed_registry


def _match(home, away, league="Test League"):
    return CanonicalMatch(
        match_id="1", sport=None, league=league, home_team=home, away_team=away,
        start_time=None, home_odds=None, draw_odds=None, away_odds=None, raw={},
    )


def test_sync_adds_new_league_and_competitors():
    reg = build_seed_registry()
    football_id = next(s.id for s in reg.sports.values() if s.name == "Football")
    usa_id = reg.country_by_alpha3("USA").id
    before_leagues = len(reg.leagues)
    before_competitors = len(reg.competitors)

    matches = [_match("Real Salt Lake", "Austin FC", league="Brand New League")]
    report = sync_registry_from_matches(reg, matches, sport_id=football_id, fallback_country_id=usa_id)

    assert report.leagues_added == 1
    assert report.competitors_added == 2
    assert report.matches_processed == 1
    assert report.skipped_incomplete == 0
    assert len(reg.leagues) == before_leagues + 1
    assert len(reg.competitors) == before_competitors + 2

    new_league = next(lg for lg in reg.leagues.values() if lg.name == "Brand New League")
    assert new_league.country_id == usa_id
    team_names = {c.name for c in reg.competitors_for_league(new_league.id)}
    assert team_names == {"Real Salt Lake", "Austin FC"}


def test_sync_reuses_existing_league_by_name_and_sport():
    reg = build_seed_registry()
    football_id = next(s.id for s in reg.sports.values() if s.name == "Football")
    usa_id = reg.country_by_alpha3("USA").id
    mls = next(lg for lg in reg.leagues.values() if lg.name == "MLS")
    before_leagues = len(reg.leagues)

    matches = [_match("Some New MLS Team", "Another New Team", league="MLS")]
    report = sync_registry_from_matches(reg, matches, sport_id=football_id, fallback_country_id=usa_id)

    assert report.leagues_added == 0  # MLS already existed
    assert report.competitors_added == 2
    assert len(reg.leagues) == before_leagues  # no duplicate MLS league created
    assert {c.name for c in reg.competitors_for_league(mls.id)} >= {"Some New MLS Team", "Another New Team"}


def test_sync_is_idempotent_on_repeated_calls_with_same_matches():
    reg = build_seed_registry()
    football_id = next(s.id for s in reg.sports.values() if s.name == "Football")
    usa_id = reg.country_by_alpha3("USA").id
    matches = [_match("Team X", "Team Y", league="Repeat League")]

    first = sync_registry_from_matches(reg, matches, sport_id=football_id, fallback_country_id=usa_id)
    second = sync_registry_from_matches(reg, matches, sport_id=football_id, fallback_country_id=usa_id)

    assert first.leagues_added == 1
    assert first.competitors_added == 2
    assert second.leagues_added == 0
    assert second.competitors_added == 0


def test_sync_skips_matches_missing_team_names():
    reg = build_seed_registry()
    football_id = next(s.id for s in reg.sports.values() if s.name == "Football")
    usa_id = reg.country_by_alpha3("USA").id

    incomplete = CanonicalMatch(
        match_id="1", sport=None, league="X", home_team=None, away_team="Team Y",
        start_time=None, home_odds=None, draw_odds=None, away_odds=None, raw={},
    )
    report = sync_registry_from_matches(reg, [incomplete], sport_id=football_id, fallback_country_id=usa_id)

    assert report.skipped_incomplete == 1
    assert report.leagues_added == 0
    assert report.competitors_added == 0


def test_sync_uses_placeholder_league_name_when_league_field_missing():
    reg = build_seed_registry()
    football_id = next(s.id for s in reg.sports.values() if s.name == "Football")
    usa_id = reg.country_by_alpha3("USA").id

    match = CanonicalMatch(
        match_id="1", sport=None, league=None, home_team="A", away_team="B",
        start_time=None, home_odds=None, draw_odds=None, away_odds=None, raw={},
    )
    report = sync_registry_from_matches(reg, [match], sport_id=football_id, fallback_country_id=usa_id)

    assert report.leagues_added == 1
    placeholder = next(lg for lg in reg.leagues.values() if lg.name == "Unknown league (synced)")
    assert placeholder.sport_id == football_id
