import numpy as np
import pytest

from bukmeker.value_betting import (
    apply_global_limits,
    bootstrap_probability_ci,
    expected_value,
    kelly_stake,
    overlay,
    probability_edge,
    value_percentage,
)


def test_expected_value_positive_for_underpriced_favourite():
    ev = expected_value(model_prob=0.6, odds=2.0)
    assert ev == pytest.approx(0.2)


def test_expected_value_zero_at_fair_odds():
    # fair odds for p=0.5 is 2.0
    assert expected_value(0.5, 2.0) == pytest.approx(0.0)


def test_expected_value_rejects_invalid_prob():
    with pytest.raises(ValueError):
        expected_value(1.5, 2.0)


def test_expected_value_rejects_odds_below_one():
    with pytest.raises(ValueError):
        expected_value(0.5, 1.0)


def test_value_percentage_matches_manual_formula():
    assert value_percentage(0.55, 2.1) == pytest.approx(0.55 * 2.1 - 1.0)


def test_overlay_and_probability_edge():
    assert overlay(0.55, 0.50) == pytest.approx(0.05)
    assert probability_edge(0.55, 0.50) == pytest.approx(0.10)


def test_overlay_rejects_model_prob_outside_unit_interval():
    # a plausible real mistake: passing odds (e.g. 2.5) where a probability was expected
    with pytest.raises(ValueError):
        overlay(2.5, 0.50)


def test_probability_edge_rejects_fair_prob_outside_unit_interval():
    with pytest.raises(ValueError):
        probability_edge(0.55, 1.5)


def test_kelly_stake_zero_when_no_edge():
    stake = kelly_stake(bankroll=1000, prob=0.4, odds=2.0, fraction=1.0)
    assert stake == pytest.approx(0.0)


def test_kelly_stake_matches_manual_formula_full_kelly():
    bankroll, prob, odds = 1000.0, 0.6, 2.0
    stake = kelly_stake(bankroll, prob, odds, fraction=1.0)
    f_star = (prob * (odds - 1) - (1 - prob)) / (odds - 1)
    assert stake == pytest.approx(bankroll * f_star)


def test_half_kelly_is_half_of_full_kelly():
    full = kelly_stake(1000, 0.6, 2.0, fraction=1.0)
    half = kelly_stake(1000, 0.6, 2.0, fraction=0.5)
    assert half == pytest.approx(full / 2)


def test_apply_global_limits_caps_single_stake():
    stake = apply_global_limits(requested_stake=500, bankroll=1000, max_stake_pct=0.02)
    assert stake == pytest.approx(20.0)


def test_apply_global_limits_respects_daily_exposure_cap():
    stake = apply_global_limits(
        requested_stake=100,
        bankroll=1000,
        max_stake_pct=0.5,
        daily_exposure_so_far=95,
        max_daily_exposure_pct=0.10,
    )
    assert stake == pytest.approx(5.0)


def test_bootstrap_probability_ci_contains_true_mean():
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=0.6, scale=0.05, size=500)

    def fit_predict(idx):
        return float(np.mean(sample[idx]))

    lower, upper = bootstrap_probability_ci(fit_predict, n_samples=500, n_bootstrap=500, seed=1)
    assert lower < 0.6 < upper
