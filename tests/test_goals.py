import numpy as np
import pytest
from scipy.stats import poisson as scipy_poisson

from bukmeker.models import (
    bivariate_poisson_matrix,
    dixon_coles_matrix,
    monte_carlo_outcome_probs,
    negative_binomial_pmf,
    outcome_probs_from_matrix,
    poisson_matrix,
    poisson_pmf,
    skellam_probs,
)


def test_poisson_pmf_matches_scipy():
    assert poisson_pmf(2, 1.5) == pytest.approx(scipy_poisson.pmf(2, 1.5))


def test_poisson_matrix_sums_to_one():
    m = poisson_matrix(1.4, 1.1, max_goals=20)
    assert m.sum() == pytest.approx(1.0, abs=1e-6)


def test_poisson_matrix_independent_marginal_matches_poisson_pmf():
    m = poisson_matrix(1.4, 1.1, max_goals=20)
    home_marginal = m.sum(axis=1)
    assert home_marginal[2] == pytest.approx(scipy_poisson.pmf(2, 1.4), abs=1e-9)


def test_bivariate_poisson_sums_to_one():
    m = bivariate_poisson_matrix(1.3, 1.0, lambda3=0.15, max_goals=15)
    assert m.sum() == pytest.approx(1.0, abs=1e-6)


def test_bivariate_poisson_zero_covariance_matches_independent():
    m_biv = bivariate_poisson_matrix(1.3, 1.0, lambda3=0.0, max_goals=10)
    m_indep = poisson_matrix(1.3, 1.0, max_goals=10)
    assert np.allclose(m_biv, m_indep, atol=1e-8)


def test_dixon_coles_matrix_sums_to_one():
    m = dixon_coles_matrix(1.4, 1.1, rho=-0.1, max_goals=15)
    assert m.sum() == pytest.approx(1.0, abs=1e-6)


def test_dixon_coles_zero_rho_matches_independent_poisson():
    m_dc = dixon_coles_matrix(1.4, 1.1, rho=0.0, max_goals=10)
    m_indep = poisson_matrix(1.4, 1.1, max_goals=10)
    assert np.allclose(m_dc, m_indep, atol=1e-8)


def test_negative_binomial_reduces_to_higher_variance_than_poisson():
    k = np.arange(0, 15)
    mu = 1.5
    nb_probs = negative_binomial_pmf(k, mu=mu, dispersion=2.0)
    nb_var = np.sum(nb_probs * (k - mu) ** 2)
    assert nb_var > mu  # overdispersion: variance > mean (Poisson variance == mean)


def test_skellam_probs_sum_to_one():
    probs = skellam_probs(1.4, 1.1)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)


def test_skellam_favours_home_when_stronger():
    probs = skellam_probs(2.0, 0.8)
    assert probs["home_win"] > probs["away_win"]


def test_monte_carlo_outcome_probs_matches_skellam_within_tolerance():
    mc = monte_carlo_outcome_probs(1.4, 1.1, n_sim=200_000, seed=7)
    exact = skellam_probs(1.4, 1.1)
    for key in ("home_win", "draw", "away_win"):
        assert mc[key] == pytest.approx(exact[key], abs=0.01)


def test_outcome_probs_from_matrix_matches_skellam():
    m = poisson_matrix(1.4, 1.1, max_goals=25)
    from_matrix = outcome_probs_from_matrix(m)
    exact = skellam_probs(1.4, 1.1)
    for key in ("home_win", "draw", "away_win"):
        assert from_matrix[key] == pytest.approx(exact[key], abs=1e-4)


def test_outcome_probs_home_win_when_row_index_exceeds_col():
    # Degenerate matrix: all mass at (2,0) -> home win
    m = np.zeros((3, 3))
    m[2, 0] = 1.0
    result = outcome_probs_from_matrix(m)
    assert result["home_win"] == pytest.approx(1.0)
    assert result["draw"] == pytest.approx(0.0)
    assert result["away_win"] == pytest.approx(0.0)


def test_negative_lambda_rejected_instead_of_silently_producing_nan():
    # Previously a negative lambda propagated as NaN through every downstream
    # probability instead of raising -- e.g. from a buggy upstream rating calc.
    with pytest.raises(ValueError):
        poisson_pmf(2, -1.0)
    with pytest.raises(ValueError):
        poisson_matrix(-1.0, 1.0)
    with pytest.raises(ValueError):
        bivariate_poisson_matrix(1.0, 1.0, lambda3=-0.1)
    with pytest.raises(ValueError):
        dixon_coles_matrix(-1.0, 1.0, rho=0.0)
    with pytest.raises(ValueError):
        skellam_probs(-1.0, 1.0)
    with pytest.raises(ValueError):
        monte_carlo_outcome_probs(-1.0, 1.0, n_sim=10)


def test_zero_lambda_is_a_valid_degenerate_case():
    # lambda=0 is a legitimate (if extreme) Poisson(0) -- must NOT raise.
    m = poisson_matrix(0.0, 1.0, max_goals=20)
    assert m.sum() == pytest.approx(1.0, abs=1e-6)
    assert m[0, :].sum() == pytest.approx(1.0)  # home always scores 0


def test_negative_binomial_rejects_nonpositive_dispersion():
    with pytest.raises(ValueError):
        negative_binomial_pmf(np.arange(5), mu=1.0, dispersion=0.0)
