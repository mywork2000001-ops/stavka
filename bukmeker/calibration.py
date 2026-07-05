"""Probability calibration and evaluation metrics (bukmeker.txt §2.2, §2.3)."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


class PlattScaler:
    """Platt scaling: fits a 1-D logistic regression on raw scores to produce
    calibrated probabilities."""

    def __init__(self) -> None:
        self._model = LogisticRegression()

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> "PlattScaler":
        self._model.fit(np.asarray(raw_scores).reshape(-1, 1), y_true)
        return self

    def predict_proba(self, raw_scores: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(np.asarray(raw_scores).reshape(-1, 1))[:, 1]


class IsotonicScaler:
    """Isotonic regression calibration — non-parametric, monotonic."""

    def __init__(self) -> None:
        self._model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> "IsotonicScaler":
        self._model.fit(raw_scores, y_true)
        return self

    def predict_proba(self, raw_scores: np.ndarray) -> np.ndarray:
        return self._model.predict(raw_scores)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE = (1/M) * sum_m |acc(B_m) - conf(B_m)|."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return float(np.mean(np.abs(prob_true - prob_pred)))


def calculate_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Non-accuracy metric suite: Log Loss, Brier, ROC-AUC, ECE."""
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    metrics = {
        "log_loss": float(log_loss(y_true, y_pred_proba)),
        "brier": float(brier_score_loss(y_true, y_pred_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
    }
    metrics["ece"] = expected_calibration_error(y_true, y_pred_proba)
    return metrics
