"""Interactive web dashboard (Streamlit) over the math core.

Run with: `bukmeker dashboard` (spawns `streamlit run` on this module) or
directly via `streamlit run dashboard_app.py` from the project root.

All computation here delegates to the tested library modules (`margin`,
`models`, `value_betting`, `sports`, `coupon`, `monetization`, `entities`) —
this file only wires widgets to those functions and renders the result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from bukmeker.coupon import ValueBetCandidate, generate_coupons
from bukmeker.entities import build_seed_registry
from bukmeker.margin import shin_margin_removal
from bukmeker.models import dixon_coles_matrix, outcome_probs_from_matrix
from bukmeker.monetization import build_coupon_report, format_coupon_report
from bukmeker.sports import basketball, tennis
from bukmeker.value_betting import expected_value, kelly_stake, value_percentage


def _probability_table(probs: dict) -> pd.DataFrame:
    return pd.DataFrame({"outcome": list(probs.keys()), "probability": list(probs.values())})


def _value_row(label: str, prob: float, odds: float, bankroll: float, kelly_fraction: float) -> dict:
    return {
        "outcome": label,
        "p_model": round(prob, 4),
        "odds": odds,
        "EV": round(expected_value(prob, odds), 4),
        "Value %": round(value_percentage(prob, odds) * 100, 2),
        "Kelly stake": round(kelly_stake(bankroll, prob, odds, fraction=kelly_fraction), 2),
    }


def _render_value_section(probs: dict, odds: dict, bankroll: float, kelly_fraction: float) -> None:
    rows = [_value_row(k, probs[k], odds[k], bankroll, kelly_fraction) for k in probs]
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)

    best = max(rows, key=lambda r: r["EV"])
    st.metric(
        label=f"Лучший EV: {best['outcome']}",
        value=f"{best['EV']:+.2%}",
        delta="VALUE BET" if best["EV"] > 0 else "нет преимущества",
        delta_color="normal" if best["EV"] > 0 else "inverse",
    )


def render_football_tab() -> None:
    st.subheader("Футбол — Dixon-Coles")
    col1, col2 = st.columns(2)
    with col1:
        lam_home = st.slider("Ожидаемые голы (дома), λ_home", 0.1, 4.0, 2.47, 0.01)
        rho = st.slider("Dixon-Coles ρ (коррекция низких счётов)", -0.2, 0.2, -0.08, 0.01)
    with col2:
        lam_away = st.slider("Ожидаемые голы (в гостях), λ_away", 0.1, 4.0, 1.02, 0.01)

    matrix = dixon_coles_matrix(lam_home, lam_away, rho=rho, max_goals=10)
    probs = outcome_probs_from_matrix(matrix)

    st.bar_chart(_probability_table(probs), x="outcome", y="probability")

    st.markdown("**Коэффициенты букмекера (1X2)**")
    c1, c2, c3 = st.columns(3)
    odds_home = c1.number_input("Дома", min_value=1.01, value=1.75, step=0.01)
    odds_draw = c2.number_input("Ничья", min_value=1.01, value=3.60, step=0.01)
    odds_away = c3.number_input("В гостях", min_value=1.01, value=4.75, step=0.01)
    odds = {"home_win": odds_home, "draw": odds_draw, "away_win": odds_away}

    overround = 1.0 / odds_home + 1.0 / odds_draw + 1.0 / odds_away - 1.0
    try:
        fair_probs = shin_margin_removal(np.array([odds_home, odds_draw, odds_away], dtype=float))
        st.caption(
            f"Overround: {overround:.2%}. Shin fair probabilities: "
            f"home={fair_probs[0]:.3f} draw={fair_probs[1]:.3f} away={fair_probs[2]:.3f}"
        )
    except ValueError:
        st.warning(f"Overround: {overround:.2%} — коэффициенты не содержат маржи, снимать нечего.")

    bankroll = st.number_input("Банкролл", min_value=1.0, value=10_000.0, step=100.0, key="football_bankroll")
    kelly_fraction = st.slider("Доля Kelly", 0.1, 1.0, 0.5, 0.05, key="football_kelly")
    _render_value_section(probs, odds, bankroll, kelly_fraction)


def render_basketball_tab() -> None:
    st.subheader("Баскетбол — Normal margin/total")
    col1, col2 = st.columns(2)
    with col1:
        mu_margin = st.slider("Ожидаемая разница очков (дома - гости)", -20.0, 20.0, 3.5, 0.5)
        sigma_margin = st.slider("Std разницы очков", 1.0, 20.0, 12.0, 0.5)
    with col2:
        mu_total = st.slider("Ожидаемый суммарный тотал", 150.0, 260.0, 222.0, 1.0)
        sigma_total = st.slider("Std тотала", 1.0, 30.0, 14.0, 0.5)

    ml_probs = basketball.predict_moneyline(mu_margin, sigma_margin)
    st.bar_chart(_probability_table(ml_probs), x="outcome", y="probability")

    c1, c2 = st.columns(2)
    odds_home = c1.number_input("Коэффициент: победа дома", min_value=1.01, value=1.85, step=0.01)
    odds_away = c2.number_input("Коэффициент: победа в гостях", min_value=1.01, value=1.95, step=0.01)
    odds = {"home_win": odds_home, "away_win": odds_away}

    bankroll = st.number_input("Банкролл", min_value=1.0, value=10_000.0, step=100.0, key="bb_bankroll")
    kelly_fraction = st.slider("Доля Kelly", 0.1, 1.0, 0.5, 0.05, key="bb_kelly")
    _render_value_section(ml_probs, odds, bankroll, kelly_fraction)

    st.markdown("**Тотал**")
    line = st.number_input("Линия тотала", value=228.5, step=0.5)
    total_probs = basketball.predict_total(mu_total, sigma_total, line)
    st.dataframe(_probability_table(total_probs), width="stretch", hide_index=True)


def render_tennis_tab() -> None:
    st.subheader("Теннис — race-to-N-sets")
    p_set = st.slider("P(игрок 1 выигрывает сет)", 0.01, 0.99, 0.55, 0.01)
    best_of = st.radio("Формат матча", [3, 5], horizontal=True)

    probs = tennis.predict_match_win_prob(p_set=p_set, best_of=best_of)
    st.bar_chart(_probability_table(probs), x="outcome", y="probability")

    c1, c2 = st.columns(2)
    odds_p1 = c1.number_input("Коэффициент: игрок 1", min_value=1.01, value=1.65, step=0.01)
    odds_p2 = c2.number_input("Коэффициент: игрок 2", min_value=1.01, value=2.20, step=0.01)
    odds = {"player1_win": odds_p1, "player2_win": odds_p2}

    bankroll = st.number_input("Банкролл", min_value=1.0, value=10_000.0, step=100.0, key="tennis_bankroll")
    kelly_fraction = st.slider("Доля Kelly", 0.1, 1.0, 0.5, 0.05, key="tennis_kelly")
    _render_value_section(probs, odds, bankroll, kelly_fraction)


def render_coupon_tab() -> None:
    st.subheader("Купон и монетизация")
    st.caption("Добавьте несколько ставок (разные матчи/лиги), сгенерируйте купон и посчитайте выплату.")

    default_rows = pd.DataFrame(
        [
            {"bet_id": 1, "match_id": 1001, "league_id": 1, "team_ids": "1,2", "prob": 0.55, "odds": 2.00},
            {"bet_id": 2, "match_id": 1002, "league_id": 1, "team_ids": "3,4", "prob": 0.40, "odds": 2.85},
            {"bet_id": 3, "match_id": 1003, "league_id": 2, "team_ids": "5,6", "prob": 0.30, "odds": 3.80},
        ]
    )
    edited = st.data_editor(default_rows, num_rows="dynamic", width="stretch")

    bankroll = st.number_input("Банкролл", min_value=1.0, value=10_000.0, step=100.0, key="coupon_bankroll")
    col1, col2, col3 = st.columns(3)
    max_events = col1.slider("Макс. ног в купоне", 1, 5, 3)
    max_corr = col2.slider("Порог допустимой корреляции", 0.0, 1.0, 0.3, 0.05)
    kelly_fraction = col3.slider("Доля Kelly", 0.1, 1.0, 0.5, 0.05, key="coupon_kelly")

    candidates = []
    for _, row in edited.iterrows():
        try:
            team_ids = tuple(int(x) for x in str(row["team_ids"]).split(",") if x.strip())
            candidates.append(
                ValueBetCandidate(
                    bet_id=int(row["bet_id"]),
                    match_id=int(row["match_id"]),
                    league_id=int(row["league_id"]),
                    team_ids=team_ids,
                    prob=float(row["prob"]),
                    odds=float(row["odds"]),
                )
            )
        except (ValueError, TypeError):
            continue

    if st.button("Сгенерировать купоны", type="primary"):
        coupons = generate_coupons(
            candidates, bankroll=bankroll, max_events=max_events, max_corr=max_corr,
            kelly_fraction=kelly_fraction, top_n=5,
        )
        st.session_state["coupons"] = coupons

    coupons = st.session_state.get("coupons", [])
    if not coupons:
        st.info("Нажмите «Сгенерировать купоны», чтобы увидеть комбинации с положительным EV.")
        return

    coupon_df = pd.DataFrame(
        [
            {
                "legs": ", ".join(f"#{b.bet_id}" for b in c["combo"]),
                "n_legs": c["n_legs"],
                "joint_prob": round(c["joint_prob"], 4),
                "joint_odds": round(c["joint_odds"], 2),
                "EV": round(c["ev"], 4),
                "Kelly stake": round(c["stake"], 2),
            }
            for c in coupons
        ]
    )
    st.dataframe(coupon_df, width="stretch", hide_index=True)

    choice_idx = st.selectbox(
        "Купон для монетизации", options=list(range(len(coupons))),
        format_func=lambda i: coupon_df.iloc[i]["legs"],
    )
    stake = st.number_input("Ставка на купон", min_value=1.0, value=100.0, step=10.0)
    fee_pct = st.slider("Комиссия платформы, %", 0, 30, 5) / 100.0

    report = build_coupon_report(coupons[choice_idx], stake=stake, platform_fee_pct=fee_pct)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Общий коэффициент", f"{report.total_odds:.2f}")
    m2.metric("Валовая выплата", f"{report.gross_payout:.2f}")
    m3.metric("Комиссия", f"{report.platform_fee:.2f}")
    m4.metric("Чистая выплата", f"{report.net_payout:.2f}", delta=f"{report.net_profit:+.2f}")
    st.code(format_coupon_report(report), language=None)


def render_entities_tab() -> None:
    st.subheader("Страны, лиги, участники")
    st.caption(
        "Иллюстративный seed-реестр (не полный список клубов мира) — реальное покрытие "
        "достигается через AI-коннектор данных (см. USAGE.md, раздел 8)."
    )
    reg = build_seed_registry()

    sport_names = {s.id: s.name for s in reg.sports.values()}
    sport_id = st.selectbox("Вид спорта", options=list(sport_names), format_func=lambda i: sport_names[i])

    leagues = reg.leagues_for_sport(sport_id)
    league_names = {lg.id: lg.name for lg in leagues}
    league_id = st.selectbox("Лига", options=list(league_names), format_func=lambda i: league_names[i])

    competitors = reg.competitors_for_league(league_id)
    df = pd.DataFrame([{"Участник": c.name, "Тип": c.kind.value} for c in competitors])
    st.dataframe(df, width="stretch", hide_index=True)


def render_about_tab() -> None:
    st.subheader("О проекте")
    st.markdown(
        """
Это исследовательская математика value-betting, а не готовый продукт для
реальных ставок. Ключевые ограничения:

- Все расчёты на этой странице — **демонстрация**, коэффициенты и вероятности
  вводятся вручную или берутся из синтетических примеров.
- **Монетизация купона — только расчёт** (комиссия и выплата), реальных
  платежей нет.
- **Seed-данные по странам/клубам иллюстративны**, не исчерпывающий список.
- Подключение реальных данных — через AI-коннектор (`bukmeker connector`),
  который требует реальных API-ключей и делает реальные сетевые запросы.

Подробности: `PROMPT.md` (архитектура и решения), `README.md` (обзор),
`USAGE.md` (пошаговая инструкция).
        """
    )


def main() -> None:
    st.set_page_config(page_title="Bukmeker — Value Betting Dashboard", layout="wide")
    st.title("Bukmeker — Value Betting Dashboard")
    st.caption("Мультиспортивный движок оценки value bets, купонов и монетизации.")

    tabs = st.tabs(["Футбол", "Баскетбол", "Теннис", "Купон и монетизация", "Страны/лиги", "О проекте"])
    with tabs[0]:
        render_football_tab()
    with tabs[1]:
        render_basketball_tab()
    with tabs[2]:
        render_tennis_tab()
    with tabs[3]:
        render_coupon_tab()
    with tabs[4]:
        render_entities_tab()
    with tabs[5]:
        render_about_tab()


if __name__ == "__main__":
    main()
