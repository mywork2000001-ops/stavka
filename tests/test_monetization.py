import pytest

from bukmeker.monetization import build_coupon_report, format_coupon_report


def _coupon(n_legs=3, joint_odds=5.70, joint_prob=0.18):
    return {"n_legs": n_legs, "joint_odds": joint_odds, "joint_prob": joint_prob}


def test_build_coupon_report_matches_worked_example():
    report = build_coupon_report(_coupon(), stake=100.0, platform_fee_pct=0.05)
    assert report.gross_payout == pytest.approx(570.00)
    assert report.platform_fee == pytest.approx(28.50)
    assert report.net_payout == pytest.approx(541.50)
    assert report.net_profit == pytest.approx(441.50)


def test_build_coupon_report_zero_fee_means_net_equals_gross():
    report = build_coupon_report(_coupon(), stake=100.0, platform_fee_pct=0.0)
    assert report.net_payout == pytest.approx(report.gross_payout)


def test_build_coupon_report_rejects_nonpositive_stake():
    with pytest.raises(ValueError):
        build_coupon_report(_coupon(), stake=0.0)


def test_build_coupon_report_rejects_invalid_fee_pct():
    with pytest.raises(ValueError):
        build_coupon_report(_coupon(), stake=100.0, platform_fee_pct=1.0)


def test_format_coupon_report_contains_key_figures():
    report = build_coupon_report(_coupon(), stake=100.0, platform_fee_pct=0.05)
    text = format_coupon_report(report)
    assert "570.00" in text
    assert "28.50" in text
    assert "541.50" in text
