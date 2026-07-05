"""Team strength ratings (bukmeker.txt §1.7): Elo, Bayesian, Poisson attack/defence."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


def elo_expected_score(rating_home: float, rating_away: float, home_advantage: float = 65.0) -> float:
    """E = 1 / (1 + 10^(-(R_home - R_away + H) / 400))."""
    return 1.0 / (1.0 + 10 ** (-(rating_home - rating_away + home_advantage) / 400.0))


def elo_update(
    rating_home: float,
    rating_away: float,
    actual_home_score: float,
    k: float = 20.0,
    home_advantage: float = 65.0,
) -> tuple[float, float]:
    """R_new = R_old + K*(S - E). `actual_home_score` in {0, 0.5, 1} (loss/draw/win)."""
    expected_home = elo_expected_score(rating_home, rating_away, home_advantage)
    delta = k * (actual_home_score - expected_home)
    return rating_home + delta, rating_away - delta


@dataclass
class BayesianRating:
    """Gaussian belief theta_team ~ N(mu, sigma^2), updated via conjugate Gaussian
    observation model. A lightweight alternative to full TrueSkill/Glicko."""

    mu: float = 0.0
    sigma2: float = 1.0

    def update(self, observation: float, observation_var: float) -> "BayesianRating":
        """Bayesian update given a noisy observation of team strength."""
        posterior_var = 1.0 / (1.0 / self.sigma2 + 1.0 / observation_var)
        posterior_mu = posterior_var * (self.mu / self.sigma2 + observation / observation_var)
        return BayesianRating(mu=posterior_mu, sigma2=posterior_var)


@dataclass
class PoissonStrength:
    """Attack/defence strength ratings fit by maximum likelihood:

        log(lambda_home) = mu + home_adv[i] + attack[i] - defence[j]
        log(lambda_away) = mu + attack[j] - defence[i]

    Fit via MLE on historical (home_id, away_id, home_goals, away_goals) tuples,
    matching the classical Dixon-Coles / Maher parametrisation.
    """

    teams: list[str]
    mu: float = field(default=0.0)
    home_adv: float = field(default=0.0)
    attack: dict = field(default_factory=dict)
    defence: dict = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        home_ids: np.ndarray,
        away_ids: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        teams: list[str],
    ) -> "PoissonStrength":
        n_teams = len(teams)
        idx = {t: i for i, t in enumerate(teams)}
        unknown = (set(home_ids) | set(away_ids)) - set(idx)
        if unknown:
            raise ValueError(
                f"home_ids/away_ids contain team(s) not present in `teams`: {sorted(unknown)}"
            )
        h_idx = np.array([idx[t] for t in home_ids])
        a_idx = np.array([idx[t] for t in away_ids])

        def unpack(params: np.ndarray):
            mu = params[0]
            home_adv = params[1]
            attack = params[2 : 2 + n_teams]
            defence = params[2 + n_teams : 2 + 2 * n_teams]
            return mu, home_adv, attack, defence

        def neg_log_likelihood(params: np.ndarray) -> float:
            mu, home_adv, attack, defence = unpack(params)
            log_lam_home = mu + home_adv + attack[h_idx] - defence[a_idx]
            log_lam_away = mu + attack[a_idx] - defence[h_idx]
            lam_home = np.exp(log_lam_home)
            lam_away = np.exp(log_lam_away)
            ll = np.sum(
                home_goals * log_lam_home - lam_home
                + away_goals * log_lam_away - lam_away
            )
            # Ridge penalty for identifiability (attack/defence centred at 0)
            penalty = 0.01 * (np.sum(attack**2) + np.sum(defence**2))
            return -ll + penalty

        x0 = np.zeros(2 + 2 * n_teams)
        x0[0] = 0.3  # baseline log-goal-rate
        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B")
        mu, home_adv, attack, defence = unpack(result.x)
        return cls(
            teams=teams,
            mu=float(mu),
            home_adv=float(home_adv),
            attack={t: float(attack[i]) for t, i in idx.items()},
            defence={t: float(defence[i]) for t, i in idx.items()},
        )

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        log_lam_home = self.mu + self.home_adv + self.attack[home_team] - self.defence[away_team]
        log_lam_away = self.mu + self.attack[away_team] - self.defence[home_team]
        return float(np.exp(log_lam_home)), float(np.exp(log_lam_away))
