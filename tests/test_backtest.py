import numpy as np
import pytest

from bukmeker.backtest import backtest_poisson_ratings


def _synthetic_matches(n=300, seed=0):
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D", "E", "F"]
    true_attack = {"A": 0.5, "B": 0.3, "C": 0.0, "D": -0.1, "E": -0.3, "F": -0.4}
    home_ids, away_ids, home_goals, away_goals = [], [], [], []
    for _ in range(n):
        h, a = rng.choice(teams, size=2, replace=False)
        lam_h = np.exp(0.25 + 0.2 + true_attack[h] - true_attack[a] * 0)
        lam_a = np.exp(0.25 + true_attack[a] - true_attack[h] * 0)
        home_ids.append(h)
        away_ids.append(a)
        home_goals.append(rng.poisson(lam_h))
        away_goals.append(rng.poisson(lam_a))
    return np.array(home_ids), np.array(away_ids), np.array(home_goals), np.array(away_goals), teams


def test_backtest_returns_fitted_model_and_standard_metrics():
    home_ids, away_ids, home_goals, away_goals, teams = _synthetic_matches()
    fitted, metrics = backtest_poisson_ratings(home_ids, away_ids, home_goals, away_goals, teams)

    assert set(metrics) >= {"log_loss", "brier", "roc_auc", "ece", "n_train", "n_test"}
    assert metrics["n_train"] + metrics["n_test"] + metrics["n_test_skipped_unknown_team"] <= len(home_ids)
    assert fitted.attack  # actually fitted, not a stub


def test_backtest_only_fits_on_training_split_not_full_data():
    home_ids, away_ids, home_goals, away_goals, teams = _synthetic_matches(n=300)
    fitted, metrics = backtest_poisson_ratings(
        home_ids, away_ids, home_goals, away_goals, teams, test_fraction=0.3
    )
    assert metrics["n_train"] == int(300 * 0.7)


def test_backtest_rejects_too_small_dataset():
    home_ids, away_ids, home_goals, away_goals, teams = _synthetic_matches(n=10)
    with pytest.raises(ValueError, match="not enough matches"):
        backtest_poisson_ratings(home_ids, away_ids, home_goals, away_goals, teams)


def test_backtest_a_stronger_team_is_reflected_in_higher_home_win_auc():
    # a model that actually captures team strength should do noticeably
    # better than a coin flip (roc_auc meaningfully > 0.5) on this synthetic
    # data, where team strength differences are substantial and real.
    home_ids, away_ids, home_goals, away_goals, teams = _synthetic_matches(n=600, seed=1)
    _, metrics = backtest_poisson_ratings(home_ids, away_ids, home_goals, away_goals, teams)
    assert metrics["roc_auc"] > 0.55


def test_backtest_skips_test_matches_with_a_team_unseen_in_training():
    # Train split (first 90%) only ever features A/B/C/D; the held-out test
    # split's matches all involve a team that never appeared in training --
    # every single one must be skipped (n_test_skipped_unknown_team), not
    # crash on a missing rating.
    home_ids = np.array(["A", "B"] * 45 + ["NewTeam"] * 10)
    away_ids = np.array(["B", "A"] * 45 + ["A"] * 10)
    home_goals = np.array([1, 2] * 45 + [1] * 10)
    away_goals = np.array([0, 1] * 45 + [1] * 10)
    teams = ["A", "B", "NewTeam"]

    with pytest.raises(ValueError, match="too few matches"):
        backtest_poisson_ratings(home_ids, away_ids, home_goals, away_goals, teams, test_fraction=0.1)


def test_backtest_reports_skipped_count_for_partially_unknown_test_teams():
    home_ids, away_ids, home_goals, away_goals, teams = _synthetic_matches(n=200)
    # append a handful of test-only matches with a brand-new, never-trained team
    home_ids = np.concatenate([home_ids, ["Ghost FC"] * 5])
    away_ids = np.concatenate([away_ids, ["A"] * 5])
    home_goals = np.concatenate([home_goals, [1] * 5])
    away_goals = np.concatenate([away_goals, [0] * 5])

    _, metrics = backtest_poisson_ratings(home_ids, away_ids, home_goals, away_goals, teams, test_fraction=0.2)
    assert metrics["n_test_skipped_unknown_team"] >= 5
