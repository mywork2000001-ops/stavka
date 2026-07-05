import numpy as np
import pytest

from bukmeker.calibration import (
    IsotonicScaler,
    PlattScaler,
    calculate_metrics,
    expected_calibration_error,
)


def _synthetic_data(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    raw_scores = rng.normal(size=n)
    true_prob = 1 / (1 + np.exp(-1.5 * raw_scores))
    y = rng.binomial(1, true_prob)
    return raw_scores, y


def test_platt_scaler_outputs_valid_probabilities():
    raw_scores, y = _synthetic_data()
    scaler = PlattScaler().fit(raw_scores, y)
    probs = scaler.predict_proba(raw_scores)
    assert np.all((probs >= 0) & (probs <= 1))


def test_isotonic_scaler_outputs_valid_probabilities():
    raw_scores, y = _synthetic_data()
    scaler = IsotonicScaler().fit(raw_scores, y)
    probs = scaler.predict_proba(raw_scores)
    assert np.all((probs >= 0) & (probs <= 1))


def test_expected_calibration_error_zero_for_perfectly_calibrated():
    # deterministic bins: prob predictions equal empirical frequency
    y_prob = np.array([0.1] * 10 + [0.9] * 10)
    y_true = np.array([0] * 9 + [1] + [1] * 9 + [0])
    ece = expected_calibration_error(y_true, y_prob, n_bins=2)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_expected_calibration_error_weights_by_bin_population():
    # Bin A: 90 samples, all y_prob=0.1, all y_true=0 -> |acc-conf| = 0.1
    # Bin B: 10 samples, all y_prob=0.9, 5/10 y_true=1  -> |acc-conf| = 0.4
    # Weighted ECE = 0.9*0.1 + 0.1*0.4 = 0.13, NOT the unweighted mean (0.25).
    y_prob = np.array([0.1] * 90 + [0.9] * 10)
    y_true = np.array([0] * 90 + [1] * 5 + [0] * 5)
    ece = expected_calibration_error(y_true, y_prob, n_bins=2)
    assert ece == pytest.approx(0.13, abs=1e-9)


def test_calculate_metrics_returns_expected_keys():
    raw_scores, y = _synthetic_data()
    probs = 1 / (1 + np.exp(-raw_scores))
    metrics = calculate_metrics(y, probs)
    assert set(metrics) == {"log_loss", "brier", "roc_auc", "ece"}
    assert metrics["roc_auc"] > 0.5


def test_calibration_improves_ece_on_miscalibrated_scores():
    rng = np.random.default_rng(1)
    n = 3000
    true_prob = rng.uniform(0.05, 0.95, size=n)
    y = rng.binomial(1, true_prob)
    miscalibrated = np.clip(true_prob * 0.5, 0, 1)  # deliberately shrunk toward 0

    ece_before = expected_calibration_error(y, miscalibrated)
    calibrated = IsotonicScaler().fit(miscalibrated, y).predict_proba(miscalibrated)
    ece_after = expected_calibration_error(y, calibrated)
    assert ece_after < ece_before
