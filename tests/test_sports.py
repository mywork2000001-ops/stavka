import pytest

from bukmeker.sports.basketball import predict_moneyline, predict_spread_cover, predict_total
from bukmeker.sports.football import predict_1x2
from bukmeker.sports.tennis import predict_match_win_prob, race_to_win_prob


def test_football_predict_1x2_sums_to_one():
    probs = predict_1x2(home_lambda=1.4, away_lambda=1.1)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)


def test_basketball_moneyline_even_margin_is_pick_em():
    probs = predict_moneyline(mu_margin=0.0, sigma_margin=10.0)
    assert probs["home_win"] == pytest.approx(0.5)


def test_basketball_moneyline_favours_home_for_positive_margin():
    probs = predict_moneyline(mu_margin=5.0, sigma_margin=10.0)
    assert probs["home_win"] > 0.5


def test_basketball_moneyline_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        predict_moneyline(mu_margin=1.0, sigma_margin=0.0)


def test_basketball_spread_cover_at_zero_matches_moneyline():
    ml = predict_moneyline(mu_margin=4.0, sigma_margin=12.0)
    cover = predict_spread_cover(mu_margin=4.0, sigma_margin=12.0, spread=0.0)
    assert cover["home_covers"] == pytest.approx(ml["home_win"])


def test_basketball_total_over_under_sums_to_one():
    result = predict_total(mu_total=220.0, sigma_total=15.0, line=225.5)
    assert result["over"] + result["under"] == pytest.approx(1.0)


def test_basketball_total_over_probability_decreases_with_higher_line():
    low_line = predict_total(mu_total=220.0, sigma_total=15.0, line=200.0)
    high_line = predict_total(mu_total=220.0, sigma_total=15.0, line=250.0)
    assert low_line["over"] > high_line["over"]


def test_tennis_race_to_win_matches_known_best_of_3_formula():
    p = 0.6
    formula = p**2 * (3 - 2 * p)
    assert race_to_win_prob(p, wins_needed=2) == pytest.approx(formula)


def test_tennis_predict_match_win_prob_best_of_3_matches_race_to_win():
    probs = predict_match_win_prob(p_set=0.6, best_of=3)
    assert probs["player1_win"] == pytest.approx(race_to_win_prob(0.6, 2))
    assert probs["player1_win"] + probs["player2_win"] == pytest.approx(1.0)


def test_tennis_predict_match_win_prob_best_of_5_higher_favours_stronger_player():
    # a set-favourite should have a *higher* match win prob over best-of-5 than best-of-3
    bo3 = predict_match_win_prob(p_set=0.6, best_of=3)["player1_win"]
    bo5 = predict_match_win_prob(p_set=0.6, best_of=5)["player1_win"]
    assert bo5 > bo3


def test_tennis_rejects_even_best_of():
    with pytest.raises(ValueError):
        predict_match_win_prob(p_set=0.5, best_of=4)


def test_tennis_fifty_fifty_set_prob_gives_fifty_fifty_match_prob():
    probs = predict_match_win_prob(p_set=0.5, best_of=3)
    assert probs["player1_win"] == pytest.approx(0.5)
