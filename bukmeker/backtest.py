"""Validates a fitted rating model against held-out historical matches BEFORE
it's trusted for real predictions -- see bukmeker.txt §3.8: a model that has
never been checked out-of-sample is exactly how "Overfitting" and "Small
Sample Bias" turn into real losses, not just an abstract warning.

Design note: this is a single chronological train/test holdout split, not a
full rolling walk-forward re-fit per test match. Re-running
`scipy.optimize.minimize` once per held-out match doesn't scale, and a single
holdout is still a real, honest out-of-sample check -- just a cheaper one.
Rolling re-fits would be a legitimate future upgrade, not a claim made here.
"""

from __future__ import annotations

import numpy as np

from .calibration import calculate_metrics
from .ratings import PoissonStrength
from .sports.football import predict_1x2


def backtest_poisson_ratings(
    home_ids: np.ndarray,
    away_ids: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    teams: list[str],
    test_fraction: float = 0.2,
) -> tuple[PoissonStrength, dict]:
    """Fits `PoissonStrength` on the earliest `1 - test_fraction` of matches
    (assumes inputs are already ordered chronologically -- e.g. straight from
    `connectors.historical.to_rating_arrays`, which preserves provider order)
    and scores its home-win predictions against the held-out remainder.
    Matches whose teams never appeared in the training split can't be scored
    (the model has no rating for them) and are skipped, not silently guessed.

    Returns `(fitted_model, metrics)`; `metrics` includes `log_loss`, `brier`,
    `roc_auc`, `ece` (see `calibration.calculate_metrics`) plus `n_train`,
    `n_test`, `n_test_skipped_unknown_team`.
    """
    n = len(home_ids)
    split = int(n * (1.0 - test_fraction))
    if split < 10 or (n - split) < 5:
        raise ValueError(
            f"not enough matches for a meaningful backtest (got {n}, "
            f"need at least 10 train + 5 test after the {test_fraction:.0%} split)"
        )

    fitted = PoissonStrength.fit(
        home_ids[:split], away_ids[:split], home_goals[:split], away_goals[:split], teams
    )

    y_true, y_prob_home_win = [], []
    skipped = 0
    for i in range(split, n):
        home, away = home_ids[i], away_ids[i]
        # `fitted.attack`/`defence` have an entry for every name in `teams`
        # regardless of whether it was actually observed while fitting (see
        # PoissonStrength docstring) -- `observed_teams` is the real check.
        if home not in fitted.observed_teams or away not in fitted.observed_teams:
            skipped += 1
            continue
        lam_home, lam_away = fitted.expected_goals(home, away)
        probs = predict_1x2(lam_home, lam_away)
        y_prob_home_win.append(probs["home_win"])
        y_true.append(1 if home_goals[i] > away_goals[i] else 0)

    if len(y_true) < 2 or len(set(y_true)) < 2:
        raise ValueError(
            "held-out test set has too few matches (or only one outcome class) "
            "to compute meaningful metrics -- provide more historical data"
        )

    metrics = calculate_metrics(np.array(y_true), np.array(y_prob_home_win))
    metrics["n_train"] = split
    metrics["n_test"] = len(y_true)
    metrics["n_test_skipped_unknown_team"] = skipped
    return fitted, metrics
