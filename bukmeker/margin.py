"""Bookmaker margin removal (bukmeker.txt §2.4)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def multiplicative_margin_removal(odds: np.ndarray) -> np.ndarray:
    """Baseline/fallback: normalise implied probabilities so they sum to 1."""
    q = 1.0 / np.asarray(odds, dtype=float)
    return q / q.sum()


def shin_margin_removal(odds: np.ndarray) -> np.ndarray:
    """Shin's (1993) method: models the margin as coming from a proportion `z`
    of informed bettors, solving for `lambda` such that the implied probabilities
    sum to 1 once the favourite-longshot bias is corrected.

        sum_i  q_i / (1 + lambda*(1 - q_i))  =  1,   q_i = 1/odds_i
        p_i = q_i / (1 + lambda*(1 - q_i)),  renormalised to sum to 1
    """
    q = 1.0 / np.asarray(odds, dtype=float)
    if q.sum() <= 1.0:
        raise ValueError("odds imply no bookmaker margin (sum of implied probs <= 1)")

    def objective(lam: float) -> float:
        return np.sum(q / (1.0 + lam * (1.0 - q))) - 1.0

    lam = brentq(objective, -0.999, 10.0)
    p = q / (1.0 + lam * (1.0 - q))
    return p / p.sum()
