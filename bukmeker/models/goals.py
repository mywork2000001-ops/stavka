"""Goal-scoring probability models (bukmeker.txt §1.8, §3.4).

Poisson, bivariate Poisson, Dixon-Coles, Negative Binomial, Skellam, Monte Carlo.
"""

from __future__ import annotations

from math import factorial

import numpy as np
from scipy.stats import nbinom, poisson, skellam


def poisson_pmf(k: int, lam: float) -> float:
    """P(X=k) = lambda^k * e^(-lambda) / k!"""
    return float(poisson.pmf(k, lam))


def poisson_matrix(home_lambda: float, away_lambda: float, max_goals: int = 10) -> np.ndarray:
    """Independent-Poisson joint score matrix, matrix[i, j] = P(home=i, away=j)."""
    home_probs = poisson.pmf(np.arange(max_goals + 1), home_lambda)
    away_probs = poisson.pmf(np.arange(max_goals + 1), away_lambda)
    return np.outer(home_probs, away_probs)


def bivariate_poisson_matrix(
    home_lambda: float, away_lambda: float, lambda3: float, max_goals: int = 10
) -> np.ndarray:
    """Karlis & Ntzoufras bivariate Poisson: X = Z1+Z3, Y = Z2+Z3, Z3 the shared
    covariance component capturing correlation between home/away goals."""
    l1, l2, l3 = home_lambda, away_lambda, lambda3
    n = max_goals + 1
    matrix = np.zeros((n, n))
    prefactor = np.exp(-(l1 + l2 + l3))
    for x in range(n):
        for y in range(n):
            total = 0.0
            for k in range(min(x, y) + 1):
                total += (
                    l1 ** (x - k) / factorial(x - k)
                    * l2 ** (y - k) / factorial(y - k)
                    * l3**k / factorial(k)
                )
            matrix[x, y] = prefactor * total
    return matrix / matrix.sum()


def dixon_coles_matrix(
    home_lambda: float, away_lambda: float, rho: float, max_goals: int = 10
) -> np.ndarray:
    """Independent-Poisson matrix with the Dixon-Coles low-score correction tau(i,j),
    which adjusts the (0,0), (1,0), (0,1), (1,1) cells for the empirical
    under/over-dispersion observed at low scores. `rho` is typically in [-0.2, 0.2];
    values outside that range risk negative probabilities and are not guaranteed valid."""
    matrix = poisson_matrix(home_lambda, away_lambda, max_goals)

    def tau(i: int, j: int) -> float:
        if i == 0 and j == 0:
            return 1 - home_lambda * away_lambda * rho
        if i == 0 and j == 1:
            return 1 + home_lambda * rho
        if i == 1 and j == 0:
            return 1 + away_lambda * rho
        if i == 1 and j == 1:
            return 1 - rho
        return 1.0

    for i in range(min(2, max_goals + 1)):
        for j in range(min(2, max_goals + 1)):
            matrix[i, j] *= tau(i, j)

    matrix = np.clip(matrix, 0.0, None)
    return matrix / matrix.sum()


def negative_binomial_pmf(k: np.ndarray, mu: float, dispersion: float) -> np.ndarray:
    """NB parametrised by mean `mu` and dispersion `dispersion` (= r, the size
    parameter); variance = mu + mu^2/dispersion. Models goal-count overdispersion
    that plain Poisson (mean=variance) cannot capture."""
    p = dispersion / (dispersion + mu)
    return nbinom.pmf(k, dispersion, p)


def skellam_probs(home_lambda: float, away_lambda: float) -> dict:
    """Exact P(home win / draw / away win) from the Skellam distribution of the
    goal difference X-Y, independent of building a full score matrix."""
    home_win = 1.0 - skellam.cdf(0, home_lambda, away_lambda)
    draw = skellam.pmf(0, home_lambda, away_lambda)
    away_win = skellam.cdf(-1, home_lambda, away_lambda)
    return {"home_win": float(home_win), "draw": float(draw), "away_win": float(away_win)}


def monte_carlo_outcome_probs(
    home_lambda: float, away_lambda: float, n_sim: int = 100_000, seed: int | None = None
) -> dict:
    """Simulate n_sim independent matches and return empirical outcome frequencies."""
    rng = np.random.default_rng(seed)
    home_goals = rng.poisson(lam=home_lambda, size=n_sim)
    away_goals = rng.poisson(lam=away_lambda, size=n_sim)
    diff = home_goals - away_goals
    return {
        "home_win": float(np.mean(diff > 0)),
        "draw": float(np.mean(diff == 0)),
        "away_win": float(np.mean(diff < 0)),
    }


def outcome_probs_from_matrix(matrix: np.ndarray) -> dict:
    """Collapse a home/away score matrix (matrix[i, j] = P(home=i, away=j)) into
    1X2 outcome probabilities."""
    home_win = float(np.sum(np.tril(matrix, k=-1)))  # row index i > col index j
    away_win = float(np.sum(np.triu(matrix, k=1)))  # row index i < col index j
    draw = float(np.trace(matrix))
    return {"home_win": home_win, "draw": draw, "away_win": away_win}
