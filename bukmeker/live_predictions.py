"""Turns a fitted, backtested rating model + a batch of upcoming fixtures
(with bookmaker odds) into value-bet candidates ready for
`bukmeker.coupon.generate_coupons`. This is the last link in the chain from
"real historical data" to "an actually-generated coupon" -- not just a
calculator that trusts a hand-typed probability.
"""

from __future__ import annotations

from .connectors.schema import CanonicalMatch
from .coupon import ValueBetCandidate
from .ratings import PoissonStrength
from .sports.football import predict_1x2
from .value_betting import expected_value


def scan_fixtures_for_value(
    fitted: PoissonStrength,
    fixtures: list[CanonicalMatch],
    league_id: int,
    min_ev: float = 0.0,
    rho: float = -0.08,
) -> list[ValueBetCandidate]:
    """For each fixture where both teams are known to `fitted` and all three
    1X2 odds are present, computes model probabilities via Dixon-Coles and
    keeps any outcome with `EV > min_ev` as a `ValueBetCandidate`.

    Fixtures involving a team the model was never fitted on are skipped (no
    rating to compute a probability from) rather than guessed. Team IDs are
    assigned stably from `fitted.teams` (index-based), so repeated calls
    against the same fitted model produce the same `team_ids` -- required for
    `generate_coupons`'s correlation check to recognise repeat teams/matches.
    """
    team_id_by_name = {name: i + 1 for i, name in enumerate(fitted.teams)}
    candidates: list[ValueBetCandidate] = []
    next_bet_id = 1

    for match_index, fixture in enumerate(fixtures):
        home, away = fixture.home_team, fixture.away_team
        # `fitted.teams`/`team_id_by_name` cover the full roster passed to
        # `PoissonStrength.fit()`, including names that never actually
        # appeared in the fitted match data (see its docstring) -- those get
        # a meaningless ridge-shrunk "average" rating, not a real one, so the
        # real gate here is `observed_teams`, not roster membership.
        if home not in fitted.observed_teams or away not in fitted.observed_teams:
            continue
        odds = {"home_win": fixture.home_odds, "draw": fixture.draw_odds, "away_win": fixture.away_odds}
        if any(v is None for v in odds.values()):
            continue

        lam_home, lam_away = fitted.expected_goals(home, away)
        probs = predict_1x2(lam_home, lam_away, rho=rho)

        for outcome in ("home_win", "draw", "away_win"):
            try:
                ev = expected_value(probs[outcome], odds[outcome])
            except ValueError:
                continue  # a malformed odds value from the provider -- skip just this outcome
            if ev > min_ev:
                candidates.append(
                    ValueBetCandidate(
                        bet_id=next_bet_id,
                        match_id=match_index + 1,
                        league_id=league_id,
                        team_ids=(team_id_by_name[home], team_id_by_name[away]),
                        prob=probs[outcome],
                        odds=odds[outcome],
                    )
                )
                next_bet_id += 1

    return candidates
