"""Bookmaker margin removal (bukmeker.txt §2.4)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def _validate_odds(odds: np.ndarray) -> np.ndarray:
    odds = np.asarray(odds, dtype=float)
    if np.any(odds <= 1.0):
        raise ValueError(f"all odds must be > 1.0, got {odds.tolist()}")
    return odds


def multiplicative_margin_removal(odds: np.ndarray) -> np.ndarray:
    """Baseline/fallback: normalise implied probabilities so they sum to 1."""
    odds = _validate_odds(odds)
    q = 1.0 / odds
    return q / q.sum()


def shin_margin_removal(odds: np.ndarray) -> np.ndarray:
    """Shin's (1993) method: models the margin as coming from a proportion `z`
    of informed bettors, solving for `lambda` such that the implied probabilities
    sum to 1 once the favourite-longshot bias is corrected.

        sum_i  q_i / (1 + lambda*(1 - q_i))  =  1,   q_i = 1/odds_i
        p_i = q_i / (1 + lambda*(1 - q_i)),  renormalised to sum to 1
    """
    odds = _validate_odds(odds)
    q = 1.0 / odds
    if q.sum() <= 1.0:
        raise ValueError("odds imply no bookmaker margin (sum of implied probs <= 1)")

    def objective(lam: float) -> float:
        return np.sum(q / (1.0 + lam * (1.0 - q))) - 1.0

    # Expand the search bracket until it contains a sign change, rather than
    # assuming [-0.999, 10.0] always brackets the root -- markets with many
    # outcomes and a very high overround can need a larger lambda.
    lo, hi = -0.999, 10.0
    f_lo, f_hi = objective(lo), objective(hi)
    while f_lo * f_hi > 0:
        hi *= 2
        f_hi = objective(hi)
        if hi > 1e9:
            raise ValueError("could not bracket a root for Shin's lambda -- check input odds")

    lam = brentq(objective, lo, hi)
    p = q / (1.0 + lam * (1.0 - q))
    return p / p.sum()
