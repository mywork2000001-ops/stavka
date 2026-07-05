"""Command-line entry point: `bukmeker demo`.

End-to-end walkthrough: ratings -> goal expectancy -> Dixon-Coles score matrix ->
1X2 probabilities -> Shin fair odds -> EV / Value% -> Kelly stake -> coupon generation.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from bukmeker.coupon import ValueBetCandidate, generate_coupons
from bukmeker.entities import build_seed_registry
from bukmeker.margin import shin_margin_removal
from bukmeker.models import dixon_coles_matrix, outcome_probs_from_matrix, skellam_probs
from bukmeker.monetization import build_coupon_report, format_coupon_report
from bukmeker.ratings import PoissonStrength
from bukmeker.sports import basketball, tennis
from bukmeker.value_betting import expected_value, kelly_stake, value_percentage


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def run_demo() -> None:
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

    # 7) Multi-sport: same value-betting math applied to basketball and tennis
    _section("7) МУЛЬТИСПОРТ (баскетбол, теннис)")
    reg = build_seed_registry()
    lakers = next(c for c in reg.competitors.values() if c.name == "Los Angeles Lakers")
    celtics = next(c for c in reg.competitors.values() if c.name == "Boston Celtics")
    nba_probs = basketball.predict_moneyline(mu_margin=3.5, sigma_margin=12.0)
    print(f"  NBA: {lakers.name} vs {celtics.name}: {nba_probs}")
    totals = basketball.predict_total(mu_total=222.0, sigma_total=14.0, line=228.5)
    print(f"  NBA totals (line=228.5): {totals}")

    djokovic = next(c for c in reg.competitors.values() if c.name == "Novak Djokovic")
    alcaraz = next(c for c in reg.competitors.values() if c.name == "Carlos Alcaraz")
    atp_probs = tennis.predict_match_win_prob(p_set=0.55, best_of=5)
    print(f"  ATP (best of 5): {djokovic.name} vs {alcaraz.name}: {atp_probs}")

    # 8) Coupon monetization: platform commission on top of the top-ranked coupon
    _section("8) МОНЕТИЗАЦИЯ КУПОНА")
    if coupons:
        report = build_coupon_report(coupons[0], stake=100.0, platform_fee_pct=0.05)
        for line in format_coupon_report(report).splitlines():
            print(f"  {line}")
    else:
        print("  Нет купонов с положительным EV для примера монетизации.")

    _section("ГОТОВО")
    print("Полный путь: рейтинги -> голы -> Dixon-Coles -> снятие маржи -> "
          "value detection -> Kelly -> купон -> мультиспорт -> монетизация выполнен успешно.")


def run_connector(args: argparse.Namespace) -> int:
    """Fetch live data from *any* sports-data provider (by its own API key) and
    let the Anthropic API infer how to map its JSON shape onto our canonical
    schema. Requires real network access and real API keys — unlike `demo`,
    this is not a synthetic walkthrough."""
    from bukmeker.connectors import AIDataConnector, ClaudeFieldMapper, RawDataSource

    anthropic_key = args.anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
    source_key = args.source_key or os.environ.get("SOURCE_API_KEY")
    if not anthropic_key or not source_key or not args.source_url:
        print(
            "Usage: bukmeker connector --source-url <base_url> --source-key <key> "
            "--anthropic-key <key> --path <endpoint>\n"
            "(or set SOURCE_API_KEY / ANTHROPIC_API_KEY env vars). "
            "This makes real HTTP + Anthropic API calls -- no synthetic fallback."
        )
        return 1

    source = RawDataSource(
        base_url=args.source_url,
        api_key=source_key,
        key_location=args.key_location,
        key_name=args.key_name,
    )
    mapper = ClaudeFieldMapper(api_key=anthropic_key, model=args.model)
    connector = AIDataConnector(source=source, mapper=mapper)
    matches = connector.fetch_and_normalize(args.path)

    print(f"Normalized {len(matches)} record(s) from {args.source_url}/{args.path}:")
    for m in matches:
        print(f"  {m.home_team!r} vs {m.away_team!r}  odds(H/D/A)="
              f"{m.home_odds}/{m.draw_odds}/{m.away_odds}  start={m.start_time}")

    if args.sync:
        from bukmeker.connectors import sync_registry_from_matches
        from bukmeker.entities import build_seed_registry

        registry = build_seed_registry()
        try:
            sport_id = next(
                s.id for s in registry.sports.values() if s.name.lower() == args.sync_sport.lower()
            )
        except StopIteration:
            known = ", ".join(s.name for s in registry.sports.values())
            print(f"Unknown --sync-sport {args.sync_sport!r}. Known sports: {known}")
            return 1
        try:
            country_id = registry.country_by_alpha3(args.sync_country.upper()).id
        except StopIteration:
            print(f"Unknown --sync-country ISO alpha-3 code {args.sync_country!r}")
            return 1

        report = sync_registry_from_matches(registry, matches, sport_id=sport_id, fallback_country_id=country_id)
        print(
            f"\nSync into entity registry: +{report.leagues_added} league(s), "
            f"+{report.competitors_added} competitor(s) from {report.matches_processed} match(es) "
            f"({report.skipped_incomplete} skipped -- missing team names)."
        )
    return 0


def run_dashboard(args: argparse.Namespace) -> int:
    """Launches the Streamlit web dashboard (bukmeker/dashboard.py) as a
    subprocess -- opens in the browser like a real application, backed by the
    same tested library functions as `demo`."""
    import shutil
    import subprocess
    from pathlib import Path

    if shutil.which("streamlit") is None:
        print(
            'Streamlit is not installed. Install it with: pip install -e ".[dashboard]"\n'
            "Then run: bukmeker dashboard"
        )
        return 1

    dashboard_path = Path(__file__).resolve().parent / "dashboard.py"
    cmd = ["streamlit", "run", str(dashboard_path), "--server.port", str(args.port)]
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="bukmeker", description="Value Betting Math Core CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo", help="Run the end-to-end walkthrough on an example match")

    connector_parser = subparsers.add_parser(
        "connector", help="Fetch + AI-normalize live data from any sports-data API key"
    )
    connector_parser.add_argument("--source-url", help="Base URL of the data provider")
    connector_parser.add_argument("--source-key", help="API key for the data provider")
    connector_parser.add_argument("--anthropic-key", help="Anthropic API key for field-mapping inference")
    connector_parser.add_argument("--path", default="fixtures", help="Endpoint path to fetch")
    connector_parser.add_argument("--key-location", choices=["header", "query"], default="header")
    connector_parser.add_argument("--key-name", default="x-api-key")
    connector_parser.add_argument("--model", default="claude-sonnet-5")
    connector_parser.add_argument(
        "--sync", action="store_true",
        help="Also merge the fetched leagues/teams into the entity registry (bukmeker.entities)",
    )
    connector_parser.add_argument(
        "--sync-sport", default="football",
        help="Which sport (by name: Football/Basketball/Tennis) newly synced leagues belong to",
    )
    connector_parser.add_argument(
        "--sync-country", default="USA",
        help="ISO alpha-3 code to place newly discovered leagues under (matches have no country field)",
    )

    dashboard_parser = subparsers.add_parser(
        "dashboard", help="Launch the interactive web dashboard (Streamlit) in your browser"
    )
    dashboard_parser.add_argument("--port", type=int, default=8501)

    args = parser.parse_args(argv)
    if args.command == "demo":
        run_demo()
        return 0
    if args.command == "connector":
        return run_connector(args)
    if args.command == "dashboard":
        return run_dashboard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
