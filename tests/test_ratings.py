import numpy as np
import pytest

from bukmeker.ratings import BayesianRating, PoissonStrength, elo_expected_score, elo_update


def test_elo_expected_score_equal_ratings_no_home_advantage():
    e = elo_expected_score(1500, 1500, home_advantage=0.0)
    assert e == pytest.approx(0.5)


def test_elo_expected_score_home_advantage_favours_home():
    e_with_ha = elo_expected_score(1500, 1500, home_advantage=65.0)
    e_without = elo_expected_score(1500, 1500, home_advantage=0.0)
    assert e_with_ha > e_without


def test_elo_update_winner_rating_increases():
    new_home, new_away = elo_update(1500, 1500, actual_home_score=1.0, k=20, home_advantage=0.0)
    assert new_home > 1500
    assert new_away < 1500
    # zero-sum
    assert (new_home - 1500) == pytest.approx(-(new_away - 1500))


def test_bayesian_rating_update_moves_toward_observation():
    prior = BayesianRating(mu=0.0, sigma2=1.0)
    posterior = prior.update(observation=2.0, observation_var=0.5)
    assert 0.0 < posterior.mu < 2.0
    assert posterior.sigma2 < prior.sigma2  # uncertainty shrinks


def test_poisson_strength_fit_recovers_stronger_attacker():
    rng = np.random.default_rng(42)
    teams = ["A", "B", "C", "D"]
    # Team A scores more than others regardless of opponent
    true_attack = {"A": 0.6, "B": 0.0, "C": -0.2, "D": -0.3}
    home_ids, away_ids, home_goals, away_goals = [], [], [], []
    for _ in range(500):
        h, a = rng.choice(teams, size=2, replace=False)
        lam_h = np.exp(0.3 + 0.2 + true_attack[h] - true_attack[a] * 0)
        lam_a = np.exp(0.3 + true_attack[a] - true_attack[h] * 0)
        home_ids.append(h)
        away_ids.append(a)
        home_goals.append(rng.poisson(lam_h))
        away_goals.append(rng.poisson(lam_a))

    fitted = PoissonStrength.fit(
        np.array(home_ids), np.array(away_ids), np.array(home_goals), np.array(away_goals), teams
    )
    assert fitted.attack["A"] > fitted.attack["D"]


def test_poisson_strength_fit_rejects_unknown_team_with_clear_error():
    teams = ["A", "B"]
    with pytest.raises(ValueError, match="not present in .teams."):
        PoissonStrength.fit(
            home_ids=np.array(["A"]),
            away_ids=np.array(["Unknown Team"]),
            home_goals=np.array([1]),
            away_goals=np.array([0]),
            teams=teams,
        )


def test_poisson_strength_expected_goals_positive():
    teams = ["A", "B"]
    fitted = PoissonStrength(
        teams=teams, mu=0.3, home_adv=0.2, attack={"A": 0.1, "B": -0.1}, defence={"A": 0.0, "B": 0.0}
    )
    lam_home, lam_away = fitted.expected_goals("A", "B")
    assert lam_home > 0 and lam_away > 0
