"""Value detection and bankroll management (bukmeker.txt §2.5, §3.1)."""

from __future__ import annotations

from typing import Callable

import numpy as np


def expected_value(model_prob: float, odds: float) -> float:
    """EV = p*(odds-1) - (1-p)."""
    _validate_prob_odds(model_prob, odds)
    return model_prob * (odds - 1.0) - (1.0 - model_prob)


def value_percentage(model_prob: float, odds: float) -> float:
    """Value% = p*odds - 1."""
    _validate_prob_odds(model_prob, odds)
    return model_prob * odds - 1.0


def _validate_prob(name: str, prob: float) -> None:
    if not (0.0 <= prob <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {prob}")


def overlay(model_prob: float, fair_prob: float) -> float:
    """Overlay = p_model - p_fair."""
    _validate_prob("model_prob", model_prob)
    _validate_prob("fair_prob", fair_prob)
    return model_prob - fair_prob


def probability_edge(model_prob: float, fair_prob: float) -> float:
    """Probability Edge = p_model / p_fair - 1."""
    _validate_prob("model_prob", model_prob)
    _validate_prob("fair_prob", fair_prob)
    if fair_prob <= 0:
        raise ValueError("fair_prob must be positive")
    return model_prob / fair_prob - 1.0


def bootstrap_probability_ci(
    fit_predict_fn: Callable[[np.ndarray], float],
    n_samples: int,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Generic bootstrap CI: `fit_predict_fn(indices)` should refit on a resample
    (given by `indices`, drawn with replacement from range(n_samples)) and return
    a single probability estimate. Returns the (lower, upper) percentile interval."""
    rng = np.random.default_rng(seed)
    alpha = (1.0 - ci) / 2.0
    estimates = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        estimates[b] = fit_predict_fn(idx)
    lower = float(np.quantile(estimates, alpha))
    upper = float(np.quantile(estimates, 1.0 - alpha))
    return lower, upper


def kelly_stake(bankroll: float, prob: float, odds: float, fraction: float = 0.5) -> float:
    """Fractional Kelly: f* = (p*(b-1) - (1-p)) / (b-1) = EV/(b-1). `fraction` applies
    Half (0.5) / Quarter (0.25) Kelly to reduce variance vs. full Kelly."""
    _validate_prob_odds(prob, odds)
    edge = prob * odds - 1.0
    if edge <= 0:
        return 0.0
    f = (prob * (odds - 1) - (1 - prob)) / (odds - 1)
    return bankroll * max(0.0, f * fraction)


def apply_global_limits(
    requested_stake: float,
    bankroll: float,
    max_stake_pct: float = 0.02,
    daily_exposure_so_far: float = 0.0,
    max_daily_exposure_pct: float = 0.10,
) -> float:
    """Caps a single stake at `max_stake_pct` of bankroll and further caps it so
    that today's cumulative exposure does not exceed `max_daily_exposure_pct`."""
    stake = min(requested_stake, bankroll * max_stake_pct)
    remaining_daily_room = max(0.0, bankroll * max_daily_exposure_pct - daily_exposure_so_far)
    return max(0.0, min(stake, remaining_daily_room))


def _validate_prob_odds(prob: float, odds: float) -> None:
    if not (0.0 <= prob <= 1.0):
        raise ValueError(f"prob must be in [0, 1], got {prob}")
    if odds <= 1.0:
        raise ValueError(f"odds must be > 1.0, got {odds}")
