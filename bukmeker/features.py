"""Time-decayed feature engineering utilities (bukmeker.txt §1.6, §2.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LAMBDA = 0.005  # ~138-day half-life, tuned for football via cross-validation


def exponential_weight(days_ago: np.ndarray, lam: float = DEFAULT_LAMBDA) -> np.ndarray:
    """w(t) = e^(-lambda * t). `days_ago` must be >= 0 (no look-ahead)."""
    days_ago = np.asarray(days_ago, dtype=float)
    if np.any(days_ago < 0):
        raise ValueError("days_ago must be non-negative (look-ahead not allowed)")
    return np.exp(-lam * days_ago)


def weighted_rolling_mean(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_col: str,
    as_of_date: pd.Timestamp,
    lam: float = DEFAULT_LAMBDA,
    min_periods: int = 1,
) -> pd.Series:
    """Exponentially-weighted mean of `value_col` per `group_col`, using only rows
    strictly before `as_of_date` to avoid look-ahead bias."""
    history = df[df[date_col] < as_of_date]

    def _agg(group: pd.DataFrame) -> float:
        if len(group) < min_periods:
            return np.nan
        days = (as_of_date - group[date_col]).dt.days.to_numpy()
        weights = exponential_weight(days, lam)
        return float(np.sum(group[value_col].to_numpy() * weights) / np.sum(weights))

    return history.groupby(group_col).apply(_agg, include_groups=False)


def ema(values: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average: EMA_t = alpha*x_t + (1-alpha)*EMA_{t-1}, alpha=2/(N+1)."""
    return pd.Series(values).ewm(span=span, adjust=False).mean().to_numpy()


def home_advantage_score(xg_home_avg: float, xg_away_avg: float) -> float:
    """HA = (avg xG at home / avg xG away) - 1."""
    if xg_away_avg == 0:
        raise ValueError("xg_away_avg must be non-zero")
    return xg_home_avg / xg_away_avg - 1.0
