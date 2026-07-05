import numpy as np
import pandas as pd
import pytest

from bukmeker.features import ema, exponential_weight, home_advantage_score, weighted_rolling_mean


def test_exponential_weight_zero_days_is_one():
    assert exponential_weight(np.array([0.0]))[0] == pytest.approx(1.0)


def test_exponential_weight_decays_monotonically():
    w = exponential_weight(np.array([0, 10, 50, 200]), lam=0.005)
    assert np.all(np.diff(w) < 0)


def test_exponential_weight_matches_half_life():
    half_life = np.log(2) / 0.005
    w = exponential_weight(np.array([half_life]), lam=0.005)[0]
    assert w == pytest.approx(0.5, rel=1e-6)


def test_exponential_weight_rejects_negative_days():
    with pytest.raises(ValueError):
        exponential_weight(np.array([-1.0]))


def test_weighted_rolling_mean_excludes_future_rows():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-10", "2026-06-01"]),
            "value": [1.0, 2.0, 999.0],
            "team_id": [1, 1, 1],
        }
    )
    result = weighted_rolling_mean(
        df, "date", "value", "team_id", as_of_date=pd.Timestamp("2026-01-15")
    )
    assert result.loc[1] < 3.0  # future row (999.0) must not leak in


def test_ema_reduces_to_constant_for_constant_input():
    values = np.full(20, 5.0)
    result = ema(values, span=5)
    assert np.allclose(result, 5.0)


def test_home_advantage_score_positive_when_stronger_at_home():
    ha = home_advantage_score(xg_home_avg=2.0, xg_away_avg=1.0)
    assert ha == pytest.approx(1.0)
