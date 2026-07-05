"""Tennis / individual racket sports decided by a race to N sets (ScoringModel.SET_BASED).

Sets are modelled as iid Bernoulli(p) trials (a standard simplification that
ignores serve alternation and set-score dynamics); the match is then a classic
"race to W wins" problem, solved with the negative-binomial tail formula.
"""

from __future__ import annotations

from math import comb


def race_to_win_prob(p: float, wins_needed: int) -> float:
    """P(player with per-trial win prob `p` reaches `wins_needed` wins first) =
    sum_{k=0}^{W-1} C(W-1+k, k) * p^W * (1-p)^k."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if wins_needed < 1:
        raise ValueError("wins_needed must be >= 1")
    w = wins_needed
    return sum(comb(w - 1 + k, k) * p**w * (1 - p) ** k for k in range(w))


def predict_match_win_prob(p_set: float, best_of: int = 3) -> dict:
    """`p_set` = probability player 1 wins a given set. `best_of` in {3, 5}."""
    if best_of % 2 == 0 or best_of < 1:
        raise ValueError("best_of must be odd (e.g. 3 or 5)")
    wins_needed = best_of // 2 + 1
    p1 = race_to_win_prob(p_set, wins_needed)
    return {"player1_win": p1, "player2_win": 1.0 - p1}
