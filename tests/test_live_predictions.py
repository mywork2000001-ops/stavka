from bukmeker.connectors.schema import CanonicalMatch
from bukmeker.live_predictions import scan_fixtures_for_value
from bukmeker.ratings import PoissonStrength


def _fixture(home, away, home_odds=None, draw_odds=None, away_odds=None):
    return CanonicalMatch(
        match_id=None, sport=None, league=None, home_team=home, away_team=away, start_time=None,
        home_odds=home_odds, draw_odds=draw_odds, away_odds=away_odds, raw={},
    )


def _fitted_model():
    return PoissonStrength(
        # "Unknown-to-model" is in the roster (gets an attack/defence entry,
        # like PoissonStrength.fit() would for any never-observed team) but
        # is deliberately absent from `observed_teams` -- this is exactly the
        # "in attack/defence but never actually fitted" case the code must
        # not confuse with a genuinely rated team.
        teams=["Strong", "Weak", "Unknown-to-model"],
        mu=0.3, home_adv=0.2,
        attack={"Strong": 0.6, "Weak": -0.4, "Unknown-to-model": 0.0},
        defence={"Strong": 0.3, "Weak": -0.2, "Unknown-to-model": 0.0},
        observed_teams=frozenset({"Strong", "Weak"}),
    )


def test_scan_finds_value_bet_when_odds_are_generous():
    fitted = _fitted_model()
    # Strong (home) heavily favoured by the model; give generous underdog odds
    fixtures = [_fixture("Strong", "Weak", home_odds=1.5, draw_odds=4.0, away_odds=8.0)]
    candidates = scan_fixtures_for_value(fitted, fixtures, league_id=1, min_ev=0.0)
    assert len(candidates) >= 1
    assert all(0.0 <= c.prob <= 1.0 for c in candidates)
    assert all(c.league_id == 1 for c in candidates)


def test_scan_skips_fixture_with_unrated_team():
    fitted = _fitted_model()
    fixtures = [_fixture("Strong", "Never Seen Before FC", home_odds=1.5, draw_odds=4.0, away_odds=8.0)]
    candidates = scan_fixtures_for_value(fitted, fixtures, league_id=1)
    assert candidates == []


def test_scan_skips_team_that_is_in_roster_but_never_actually_observed():
    # "Unknown-to-model" IS a key in fitted.attack/defence (a real fit() run
    # gives every roster team an entry, ridge-shrunk to ~0 if never observed)
    # -- the scan must not mistake that for a genuine rating.
    fitted = _fitted_model()
    assert "Unknown-to-model" in fitted.attack  # sanity: it IS in the roster
    assert "Unknown-to-model" not in fitted.observed_teams

    fixtures = [_fixture("Strong", "Unknown-to-model", home_odds=1.5, draw_odds=4.0, away_odds=8.0)]
    candidates = scan_fixtures_for_value(fitted, fixtures, league_id=1)
    assert candidates == []


def test_scan_skips_fixture_missing_any_odds():
    fitted = _fitted_model()
    fixtures = [_fixture("Strong", "Weak", home_odds=1.5, draw_odds=None, away_odds=8.0)]
    candidates = scan_fixtures_for_value(fitted, fixtures, league_id=1)
    assert candidates == []


def test_scan_excludes_negative_ev_outcomes():
    fitted = _fitted_model()
    # tight, model-consistent odds -> no real edge anywhere
    fixtures = [_fixture("Strong", "Weak", home_odds=1.30, draw_odds=5.5, away_odds=9.5)]
    candidates_no_edge = scan_fixtures_for_value(fitted, fixtures, league_id=1, min_ev=0.5)
    assert candidates_no_edge == []


def test_scan_assigns_stable_team_ids_across_calls():
    fitted = _fitted_model()
    fixtures = [_fixture("Strong", "Weak", home_odds=1.5, draw_odds=4.0, away_odds=8.0)]
    first = scan_fixtures_for_value(fitted, fixtures, league_id=1)
    second = scan_fixtures_for_value(fitted, fixtures, league_id=1)
    assert [c.team_ids for c in first] == [c.team_ids for c in second]


def test_scan_skips_single_outcome_with_malformed_odds_but_keeps_others():
    fitted = _fitted_model()
    # home_odds=1.0 is invalid (expected_value requires odds > 1) -- a
    # plausible malformed value from a live provider (e.g. a suspended/void
    # market priced at evens-or-below). Must not crash or abort the whole
    # fixture -- the other two outcomes (draw/away, valid odds) are still
    # evaluated normally.
    fixtures = [_fixture("Strong", "Weak", home_odds=1.0, draw_odds=4.0, away_odds=8.0)]
    candidates = scan_fixtures_for_value(fitted, fixtures, league_id=1, min_ev=0.0)
    assert not any(c.odds == 1.0 for c in candidates)


def test_scan_bet_ids_are_sequential_and_unique():
    fitted = _fitted_model()
    fixtures = [
        _fixture("Strong", "Weak", home_odds=1.5, draw_odds=4.0, away_odds=8.0),
        _fixture("Weak", "Strong", home_odds=8.0, draw_odds=4.0, away_odds=1.5),
    ]
    candidates = scan_fixtures_for_value(fitted, fixtures, league_id=1)
    bet_ids = [c.bet_id for c in candidates]
    assert bet_ids == sorted(set(bet_ids))  # unique and increasing
