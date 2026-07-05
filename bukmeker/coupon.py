"""Coupon (accumulator) generation with correlation constraints (bukmeker.txt §3.2)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np

from .value_betting import kelly_stake

# Guards against runaway combinatorial blowup (sum_r C(n, r) for r in
# 1..max_events) -- relevant because the dashboard's bet table lets a user
# freely add rows, and an unbounded n here can hang the process for minutes.
_MAX_COMBINATIONS_TO_EVALUATE = 200_000


@dataclass(frozen=True)
class ValueBetCandidate:
    bet_id: int
    match_id: int
    league_id: int
    team_ids: tuple[int, ...]
    prob: float
    odds: float


def pairwise_correlation(a: ValueBetCandidate, b: ValueBetCandidate) -> float:
    """Heuristic correlation proxy between two bets' outcomes:
    same match (mutually dependent outcomes) > shared team > shared league > none."""
    if a.match_id == b.match_id:
        return 1.0
    if set(a.team_ids) & set(b.team_ids):
        return 0.6
    if a.league_id == b.league_id:
        return 0.25
    return 0.0


def combo_is_valid(combo: tuple[ValueBetCandidate, ...], max_corr: float) -> bool:
    for a, b in combinations(combo, 2):
        if pairwise_correlation(a, b) > max_corr:
            return False
    return True


def generate_coupons(
    value_bets: list[ValueBetCandidate],
    bankroll: float,
    max_events: int = 4,
    max_corr: float = 0.3,
    kelly_fraction: float = 0.5,
    top_n: int = 5,
) -> list[dict]:
    """Greedy enumeration of 1..max_events-leg combinations, filtered by the
    correlation constraint, ranked by expected value * recommended stake."""
    n = len(value_bets)
    total_combinations = sum(comb(n, r) for r in range(1, max_events + 1))
    if total_combinations > _MAX_COMBINATIONS_TO_EVALUATE:
        raise ValueError(
            f"{total_combinations:,} combinations to evaluate ({n} candidate bets, "
            f"max_events={max_events}) exceeds the safety limit of "
            f"{_MAX_COMBINATIONS_TO_EVALUATE:,} -- reduce max_events or the number of candidate bets."
        )

    candidates: list[dict] = []
    for r in range(1, max_events + 1):
        for combo in combinations(value_bets, r):
            if not combo_is_valid(combo, max_corr):
                continue
            joint_prob = float(np.prod([b.prob for b in combo]))
            joint_odds = float(np.prod([b.odds for b in combo]))
            ev = joint_prob * (joint_odds - 1.0) - (1.0 - joint_prob)
            if ev <= 0:
                continue
            stake = kelly_stake(bankroll, joint_prob, joint_odds, fraction=kelly_fraction)
            if stake <= 0:
                continue
            candidates.append(
                {
                    "combo": combo,
                    "n_legs": r,
                    "joint_prob": joint_prob,
                    "joint_odds": joint_odds,
                    "ev": ev,
                    "stake": stake,
                }
            )
    return sorted(candidates, key=lambda c: c["ev"] * c["stake"], reverse=True)[:top_n]
