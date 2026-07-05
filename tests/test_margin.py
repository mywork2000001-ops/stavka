import numpy as np
import pytest

from bukmeker.margin import multiplicative_margin_removal, shin_margin_removal


def test_multiplicative_margin_removal_sums_to_one():
    odds = np.array([2.0, 3.5, 4.0])
    p = multiplicative_margin_removal(odds)
    assert p.sum() == pytest.approx(1.0)


def test_shin_margin_removal_sums_to_one():
    odds = np.array([1.9, 3.6, 4.2])  # overround > 1
    p = shin_margin_removal(odds)
    assert p.sum() == pytest.approx(1.0, abs=1e-6)


def test_shin_margin_removal_preserves_favourite_ordering():
    odds = np.array([1.9, 3.6, 4.2])
    p = shin_margin_removal(odds)
    assert p[0] > p[1] > p[2]


def test_shin_reduces_favourite_longshot_bias_relative_to_multiplicative():
    # Shin should assign a *lower* fair probability to the longshot than naive
    # multiplicative normalisation (correcting the favourite-longshot bias).
    odds = np.array([1.44, 3.8, 13.0])  # overround ~3.4%
    p_shin = shin_margin_removal(odds)
    p_mult = multiplicative_margin_removal(odds)
    assert p_shin[-1] < p_mult[-1]


def test_shin_rejects_odds_with_no_margin():
    odds = np.array([2.0, 2.0])  # sums of implied prob == 1, no margin
    with pytest.raises(ValueError):
        shin_margin_removal(odds)
