"""Basketball / high-scoring team sports (ScoringModel.POINT_SPREAD).

Industry-standard approach: instead of modelling every point as a discrete event
(as Poisson does for football), model the final scoring *margin* and *total* as
approximately Normal, since with 80-120+ independent scoring possessions the
Central Limit Theorem makes the Gaussian a good fit (this is the same approach
used for spread/totals markets in the wider sports-betting industry).
"""

from __future__ import annotations

from scipy.stats import norm


def predict_moneyline(mu_margin: float, sigma_margin: float) -> dict:
    """mu_margin = E[home_score - away_score]; sigma_margin = std dev of that margin.
    home_win = P(margin > 0) = Phi(mu/sigma) for a Normal(mu, sigma) margin."""
    if sigma_margin <= 0:
        raise ValueError("sigma_margin must be positive")
    home_win = float(norm.cdf(mu_margin / sigma_margin))
    return {"home_win": home_win, "away_win": 1.0 - home_win}


def predict_spread_cover(mu_margin: float, sigma_margin: float, spread: float) -> dict:
    """P(home covers a given spread), e.g. spread=-5.5 means home must win by >5.5."""
    if sigma_margin <= 0:
        raise ValueError("sigma_margin must be positive")
    z = (mu_margin + spread) / sigma_margin
    home_covers = float(norm.cdf(z))
    return {"home_covers": home_covers, "away_covers": 1.0 - home_covers}


def predict_total(mu_total: float, sigma_total: float, line: float) -> dict:
    """P(total points over/under a given line), total ~ Normal(mu_total, sigma_total)."""
    if sigma_total <= 0:
        raise ValueError("sigma_total must be positive")
    z = (line - mu_total) / sigma_total
    under = float(norm.cdf(z))
    return {"over": 1.0 - under, "under": under}
