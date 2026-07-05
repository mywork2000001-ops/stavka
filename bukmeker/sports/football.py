"""Football (ScoringModel.POISSON_GOALS): thin convenience wrapper over
`bukmeker.models.goals`, which already implements the Poisson / Dixon-Coles family."""

from __future__ import annotations

from bukmeker.models import dixon_coles_matrix, outcome_probs_from_matrix


def predict_1x2(home_lambda: float, away_lambda: float, rho: float = -0.08, max_goals: int = 10) -> dict:
    """1X2 outcome probabilities from expected goals via Dixon-Coles."""
    matrix = dixon_coles_matrix(home_lambda, away_lambda, rho=rho, max_goals=max_goals)
    return outcome_probs_from_matrix(matrix)
