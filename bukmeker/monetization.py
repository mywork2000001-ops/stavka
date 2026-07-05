"""Coupon monetization: turns a generated coupon into a payout report with a
platform commission. This is a computed report only — no real payment
processing, accounts, or currency movement is implemented here."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CouponReport:
    n_legs: int
    total_odds: float
    joint_prob: float
    stake: float
    gross_payout: float
    platform_fee_pct: float
    platform_fee: float
    net_payout: float
    net_profit: float


def build_coupon_report(coupon: dict, stake: float, platform_fee_pct: float = 0.05) -> CouponReport:
    """`coupon` is one entry from `generate_coupons` (has joint_odds/joint_prob/n_legs).
    `platform_fee_pct` is the platform's commission, taken as a percentage of the
    gross payout (a common "hold" model for coupon/tips marketplaces)."""
    if stake <= 0:
        raise ValueError("stake must be positive")
    if not 0.0 <= platform_fee_pct < 1.0:
        raise ValueError("platform_fee_pct must be in [0, 1)")

    gross_payout = stake * coupon["joint_odds"]
    platform_fee = gross_payout * platform_fee_pct
    net_payout = gross_payout - platform_fee
    return CouponReport(
        n_legs=coupon["n_legs"],
        total_odds=coupon["joint_odds"],
        joint_prob=coupon["joint_prob"],
        stake=stake,
        gross_payout=gross_payout,
        platform_fee_pct=platform_fee_pct,
        platform_fee=platform_fee,
        net_payout=net_payout,
        net_profit=net_payout - stake,
    )


def format_coupon_report(report: CouponReport) -> str:
    return (
        "Coupon Report\n"
        "-------------\n"
        f"legs: {report.n_legs}\n"
        f"total_odds: {report.total_odds:.2f}\n"
        f"stake: {report.stake:.2f}\n"
        f"gross_payout: {report.gross_payout:.2f}\n"
        f"platform_fee ({report.platform_fee_pct:.0%}): {report.platform_fee:.2f}\n"
        f"net_payout: {report.net_payout:.2f}\n"
        f"net_profit: {report.net_profit:.2f}"
    )
