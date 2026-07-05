"""Command-line entry point: `bukmeker demo`.

End-to-end walkthrough: ratings -> goal expectancy -> Dixon-Coles score matrix ->
1X2 probabilities -> Shin fair odds -> EV / Value% -> Kelly stake -> coupon generation.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from bukmeker.coupon import ValueBetCandidate, generate_coupons
from bukmeker.margin import shin_margin_removal
from bukmeker.models import dixon_coles_matrix, outcome_probs_from_matrix, skellam_probs
from bukmeker.ratings import PoissonStrength
from bukmeker.value_betting import expected_value, kelly_stake, value_percentage


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def run_demo() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # 1) Team ratings (Poisson attack/defence strength), fit on synthetic history
    _section("1) РЕЙТИНГИ КОМАНД (Poisson attack/defence, MLE)")
    rng = np.random.default_rng(42)
    teams = ["Arsenal", "Chelsea", "Liverpool", "ManCity", "Everton", "Fulham"]
    true_attack = {"Arsenal": 0.35, "Chelsea": 0.05, "Liverpool": 0.30,
                   "ManCity": 0.40, "Everton": -0.20, "Fulham": -0.15}
    true_defence = {"Arsenal": 0.15, "Chelsea": 0.00, "Liverpool": 0.20,
                     "ManCity": 0.25, "Everton": -0.10, "Fulham": -0.05}
    home_ids, away_ids, home_goals, away_goals = [], [], [], []
    for _ in range(800):
        h, a = rng.choice(teams, size=2, replace=False)
        lam_h = np.exp(0.25 + 0.25 + true_attack[h] - true_defence[a])
        lam_a = np.exp(0.25 + true_attack[a] - true_defence[h])
        home_ids.append(h)
        away_ids.append(a)
        home_goals.append(rng.poisson(lam_h))
        away_goals.append(rng.poisson(lam_a))

    strength = PoissonStrength.fit(
        np.array(home_ids), np.array(away_ids), np.array(home_goals), np.array(away_goals), teams
    )
    for t in teams:
        print(f"  {t:10s}  attack={strength.attack[t]:+.3f}  defence={strength.defence[t]:+.3f}")

    # 2) Example match: Arsenal (home) vs Everton (away)
    home_team, away_team = "Arsenal", "Everton"
    lam_home, lam_away = strength.expected_goals(home_team, away_team)
    _section(f"2) ОЖИДАЕМЫЕ ГОЛЫ: {home_team} vs {away_team}")
    print(f"  lambda_home = {lam_home:.3f}, lambda_away = {lam_away:.3f}")

    # 3) Dixon-Coles score matrix -> 1X2 probabilities
    _section("3) DIXON-COLES SCORE MATRIX -> 1X2")
    rho = -0.08
    matrix = dixon_coles_matrix(lam_home, lam_away, rho=rho, max_goals=10)
    probs_dc = outcome_probs_from_matrix(matrix)
    probs_skellam = skellam_probs(lam_home, lam_away)  # cross-check without low-score correction
    print(f"  Dixon-Coles (rho={rho}): {probs_dc}")
    print(f"  Skellam (no low-score corr., sanity check): {probs_skellam}")

    model_probs = probs_dc

    # 4) Bookmaker odds (with margin) -> Shin fair probabilities
    _section("4) СНЯТИЕ МАРЖИ БУКМЕКЕРА (Shin)")
    bookmaker_odds = np.array([1.75, 3.60, 4.75])  # home, draw, away
    fair_probs = shin_margin_removal(bookmaker_odds)
    overround = float(np.sum(1.0 / bookmaker_odds) - 1.0)
    print(f"  Bookmaker odds (1X2): {bookmaker_odds.tolist()}  (overround={overround:.2%})")
    print(f"  Shin fair probabilities: home={fair_probs[0]:.4f} draw={fair_probs[1]:.4f} away={fair_probs[2]:.4f}")

    # 5) Value detection
    _section("5) VALUE DETECTION (EV, Value%, Kelly stake)")
    outcomes = ["home_win", "draw", "away_win"]
    bankroll = 10_000.0
    for i, outcome in enumerate(outcomes):
        p_model = model_probs[outcome]
        odds = float(bookmaker_odds[i])
        ev = expected_value(p_model, odds)
        val_pct = value_percentage(p_model, odds)
        stake = kelly_stake(bankroll, p_model, odds, fraction=0.5)
        flag = "  <-- VALUE BET" if ev > 0 else ""
        print(
            f"  {outcome:9s} p_model={p_model:.4f} odds={odds:.2f} "
            f"EV={ev:+.4f} Value%={val_pct:+.2%} HalfKellyStake={stake:8.2f}{flag}"
        )

    # 6) Coupon generation across several independent value bets (different matches)
    _section("6) ГЕНЕРАЦИЯ КУПОНА (несколько независимых матчей)")
    candidates = [
        ValueBetCandidate(bet_id=1, match_id=1001, league_id=1, team_ids=(1, 2), prob=0.55, odds=2.00),
        ValueBetCandidate(bet_id=2, match_id=1002, league_id=1, team_ids=(3, 4), prob=0.40, odds=2.85),
        ValueBetCandidate(bet_id=3, match_id=1003, league_id=2, team_ids=(5, 6), prob=0.30, odds=3.80),
        ValueBetCandidate(bet_id=4, match_id=1001, league_id=1, team_ids=(1, 2), prob=0.20, odds=5.00),  # correlated w/ #1
    ]
    coupons = generate_coupons(candidates, bankroll=bankroll, max_events=3, max_corr=0.3, top_n=3)
    for c in coupons:
        legs = ", ".join(f"#{b.bet_id}" for b in c["combo"])
        print(
            f"  legs=[{legs}] n={c['n_legs']} joint_prob={c['joint_prob']:.4f} "
            f"joint_odds={c['joint_odds']:.2f} EV={c['ev']:+.4f} stake={c['stake']:.2f}"
        )

    _section("ГОТОВО")
    print("Полный путь: рейтинги -> голы -> Dixon-Coles -> снятие маржи -> "
          "value detection -> Kelly -> купон выполнен успешно.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bukmeker", description="Value Betting Math Core CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo", help="Run the end-to-end walkthrough on an example match")

    args = parser.parse_args(argv)
    if args.command == "demo":
        run_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
