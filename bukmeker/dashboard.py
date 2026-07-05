"""Interactive web dashboard (Streamlit) over the math core.

Run with: `bukmeker dashboard` (spawns `streamlit run` on this module) or
directly via `streamlit run dashboard_app.py` from the project root.

All computation here delegates to the tested library modules (`margin`,
`models`, `value_betting`, `sports`, `coupon`, `monetization`, `entities`) —
this file only wires widgets to those functions and renders the result.

UI note: navigation uses `st.navigation` (a real sidebar menu with icons and
page titles) instead of a flat row of tabs, and every technical input/metric
carries a plain-language `help=` tooltip -- see the "Справка" page for a full
glossary. This directly addresses feedback that the previous flat-tab menu
was not self-explanatory.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from bukmeker.auth import require_password
from bukmeker.coupon import ValueBetCandidate, generate_coupons
from bukmeker.entities import build_seed_registry
from bukmeker.margin import shin_margin_removal
from bukmeker.models import dixon_coles_matrix, outcome_probs_from_matrix
from bukmeker.monetization import build_coupon_report, format_coupon_report
from bukmeker.sports import basketball, tennis
from bukmeker.value_betting import expected_value, kelly_stake, value_percentage

_BANKROLL_HELP = "Условный размер вашего банка — используется только для расчёта размера ставки, реальных денег здесь нет."
_KELLY_HELP = (
    "Доля от полного критерия Келли: 1.0 = ставить 'оптимальную' по формуле долю банка, "
    "0.5 (Half Kelly) — вдвое меньше, но с меньшим риском просадки. Ниже — безопаснее."
)
_ODDS_HELP = "Коэффициент, который предлагает букмекер на этот исход."


def _probability_table(probs: dict, labels: dict[str, str] | None = None) -> pd.DataFrame:
    labels = labels or {}
    return pd.DataFrame(
        {"outcome": [labels.get(k, k) for k in probs], "probability": list(probs.values())}
    )


def _pick_match(reg, sport_name: str, key_prefix: str, role_labels: tuple[str, str], league_label: str = "Лига"):
    """Lets the user pick a league/championship and two named competitors
    from the seed registry, instead of reasoning about anonymous
    lambda/coefficient sliders or raw numeric IDs with no team identity
    attached. Leagues with fewer than 2 seeded competitors are excluded --
    there's no second team to pick as the opponent. Returns
    `(league_id, sport_id, home_competitor, away_competitor)`."""
    sport_id = next(s.id for s in reg.sports.values() if s.name == sport_name)
    leagues = [
        lg for lg in reg.leagues_for_sport(sport_id) if len(reg.competitors_for_league(lg.id)) >= 2
    ]
    league_names = {lg.id: lg.name for lg in leagues}
    league_id = st.selectbox(
        league_label, options=list(league_names), format_func=lambda i: league_names[i],
        key=f"{key_prefix}_league",
    )
    competitors = {c.name: c for c in reg.competitors_for_league(league_id)}
    names = list(competitors)

    c1, c2 = st.columns(2)
    home_name = c1.selectbox(role_labels[0], options=names, index=0, key=f"{key_prefix}_home")
    away_options = [n for n in names if n != home_name]
    away_name = c2.selectbox(role_labels[1], options=away_options, index=0, key=f"{key_prefix}_away")
    return league_id, sport_id, competitors[home_name], competitors[away_name]


def _team_picker(reg, sport_name: str, key_prefix: str, role_labels: tuple[str, str]) -> tuple[str, str]:
    """Thin wrapper over `_pick_match` for pages that only need display names
    (the prediction pages), not competitor IDs / league ID."""
    _, _, home, away = _pick_match(reg, sport_name, key_prefix, role_labels)
    return home.name, away.name


def parse_bet_rows(rows: pd.DataFrame) -> tuple[list[ValueBetCandidate], list[tuple[int, str]]]:
    """Parses the coupon page's editable bet table into `ValueBetCandidate`s.
    Returns `(candidates, skipped)` where `skipped` is `[(row_index, reason), ...]`
    for rows that failed to parse -- callers must surface `skipped` to the
    user rather than silently dropping malformed rows."""
    candidates: list[ValueBetCandidate] = []
    skipped: list[tuple[int, str]] = []
    for idx, row in rows.iterrows():
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
        except (ValueError, TypeError) as exc:
            skipped.append((idx, str(exc)))
    return candidates, skipped


def _value_row(label: str, prob: float, odds: float, bankroll: float, kelly_fraction: float) -> dict:
    return {
        "outcome": label,
        "p_model": round(prob, 4),
        "odds": odds,
        "EV": round(expected_value(prob, odds), 4),
        "Value %": round(value_percentage(prob, odds) * 100, 2),
        "Kelly stake": round(kelly_stake(bankroll, prob, odds, fraction=kelly_fraction), 2),
    }


def _render_value_section(
    probs: dict, odds: dict, bankroll: float, kelly_fraction: float, labels: dict[str, str] | None = None
) -> None:
    labels = labels or {}
    rows = [_value_row(labels.get(k, k), probs[k], odds[k], bankroll, kelly_fraction) for k in probs]
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)

    with st.expander("Что означают эти столбцы?"):
        st.markdown(
            "- **p_model** — вероятность исхода по модели.\n"
            "- **odds** — коэффициент букмекера.\n"
            "- **EV** — ожидаемая прибыль на 1 единицу ставки (>0 значит потенциально выгодно).\n"
            "- **Value %** — то же самое преимущество в процентах (`p_model × odds − 1`).\n"
            "- **Kelly stake** — сколько поставить при выбранной доле Kelly и банкролле."
        )

    best = max(rows, key=lambda r: r["EV"])
    st.metric(
        label=f"Лучший EV: {best['outcome']}",
        value=f"{best['EV']:+.2%}",
        delta="VALUE BET" if best["EV"] > 0 else "нет преимущества",
        delta_color="normal" if best["EV"] > 0 else "inverse",
        help="Исход с наибольшим ожидаемым преимуществом среди показанных выше.",
    )


def render_help_page() -> None:
    st.title("Справка: как пользоваться")
    st.markdown(
        "Это интерактивная витрина математического движка value-betting: вы "
        "задаёте вероятности/коэффициенты через ползунки и поля, страница "
        "сразу пересчитывает результат. Ничего не сохраняется и не отправляется "
        "в интернет — все расчёты происходят локально на этой машине."
    )
    st.subheader("Разделы (меню слева)")
    st.markdown(
        "- **⚽ Футбол / 🏀 Баскетбол / 🎾 Теннис** — вероятности исхода по модели "
        "своего вида спорта + поиск выгодных ставок (value bets) против коэффициентов букмекера.\n"
        "- **🎫 Купон и монетизация** — собрать несколько ставок в один купон и "
        "посчитать итоговую выплату с учётом комиссии платформы.\n"
        "- **🌍 Страны и лиги** — демонстрация структуры данных (какие виды спорта/лиги/клубы есть).\n"
        "- **🔌 AI-коннектор** — подключить реальный источник спортивных данных своим API-ключом "
        "(единственная страница с реальными сетевыми и платными запросами).\n"
        "- **ℹ️ О проекте** — что это такое и чего здесь сознательно нет (реальные деньги, реальные данные)."
    )
    st.subheader("Глоссарий терминов")
    st.markdown(
        "| Термин | Что значит |\n"
        "|---|---|\n"
        "| **p_model** | Вероятность исхода по расчёту модели (0.68 = 68%). |\n"
        "| **odds (коэффициент)** | Во сколько раз букмекер умножит вашу ставку при выигрыше. |\n"
        "| **EV (Expected Value)** | Ожидаемая прибыль на 1 единицу ставки. Больше 0 — потенциально выгодно. |\n"
        "| **Value %** | То же преимущество в процентах: `p_model × odds − 1`. |\n"
        "| **Overround / маржа** | Встроенная прибыль букмекера в его коэффициентах. |\n"
        "| **Fair probability (справедливая вероятность)** | Вероятность после вычитания маржи букмекера. |\n"
        "| **Kelly stake** | Рекомендованный размер ставки по критерию Келли (с учётом выбранной доли). |\n"
        "| **λ (лямбда), ожидаемые голы** | Среднее число голов, которое команда обычно забивает в подобных матчах. |\n"
        "| **ρ (ро), Dixon-Coles** | Небольшая техническая поправка модели для редких низких счётов (0:0, 1:0, 0:1). |\n"
        "| **Overround (баскетбол/теннис аналогично)** | Тот же смысл маржи, что и в футболе. |\n"
        "| **Комиссия платформы** | Процент, который платформа удерживает с выплаты купона (это расчёт, не реальный платёж). |"
    )
    st.info(
        "Наведите курсор на значок (?) рядом с любым полем ввода на других "
        "страницах — там короткая подсказка на этот же случай."
    )


def render_football_page() -> None:
    st.title("⚽ Футбол")
    st.caption("Модель Dixon-Coles: вероятности исхода из ожидаемого числа голов каждой команды.")

    reg = build_seed_registry()
    home_team, away_team = _team_picker(reg, "Football", "football", ("Дома", "В гостях"))
    st.subheader(f"{home_team} — {away_team}")
    st.caption(
        "λ и коэффициенты ниже вводятся вручную — в seed-реестре нет исторических матчей "
        "этих команд, поэтому автоматический расчёт ожидаемых голов недоступен (это дал бы "
        "именно AI-коннектор с реальными данными)."
    )

    col1, col2 = st.columns(2)
    with col1:
        lam_home = st.slider(
            f"Ожидаемые голы, {home_team} (λ_home)", 0.1, 4.0, 2.47, 0.01,
            help="Среднее число голов, которое команда обычно забивает дома в похожих матчах.",
        )
        rho = st.slider(
            "Поправка Dixon-Coles (ρ)", -0.2, 0.2, -0.08, 0.01,
            help="Небольшая техническая поправка для редких низких счётов (0:0, 1:0, 0:1). "
            "Обычно небольшое отрицательное число, можно оставить по умолчанию.",
        )
    with col2:
        lam_away = st.slider(
            f"Ожидаемые голы, {away_team} (λ_away)", 0.1, 4.0, 1.02, 0.01,
            help="Среднее число голов, которое команда обычно забивает в гостях в похожих матчах.",
        )

    matrix = dixon_coles_matrix(lam_home, lam_away, rho=rho, max_goals=10)
    probs = outcome_probs_from_matrix(matrix)
    labels = {"home_win": home_team, "draw": "Ничья", "away_win": away_team}
    st.bar_chart(_probability_table(probs, labels), x="outcome", y="probability")

    st.markdown("**Коэффициенты букмекера (1X2)**")
    c1, c2, c3 = st.columns(3)
    odds_home = c1.number_input(home_team, min_value=1.01, value=1.75, step=0.01, help=_ODDS_HELP)
    odds_draw = c2.number_input("Ничья", min_value=1.01, value=3.60, step=0.01, help=_ODDS_HELP)
    odds_away = c3.number_input(away_team, min_value=1.01, value=4.75, step=0.01, help=_ODDS_HELP)
    odds = {"home_win": odds_home, "draw": odds_draw, "away_win": odds_away}

    overround = 1.0 / odds_home + 1.0 / odds_draw + 1.0 / odds_away - 1.0
    try:
        fair_probs = shin_margin_removal(np.array([odds_home, odds_draw, odds_away], dtype=float))
        st.caption(
            f"Маржа букмекера (overround): {overround:.2%}. Справедливые вероятности после её снятия: "
            f"дома={fair_probs[0]:.3f} ничья={fair_probs[1]:.3f} гости={fair_probs[2]:.3f}"
        )
    except ValueError:
        st.warning(f"Overround: {overround:.2%} — коэффициенты не содержат маржи, снимать нечего.")

    bankroll = st.number_input(
        "Банкролл", min_value=1.0, value=10_000.0, step=100.0, key="football_bankroll", help=_BANKROLL_HELP
    )
    kelly_fraction = st.slider(
        "Доля Kelly", 0.1, 1.0, 0.5, 0.05, key="football_kelly", help=_KELLY_HELP
    )
    _render_value_section(probs, odds, bankroll, kelly_fraction, labels=labels)


def render_basketball_page() -> None:
    st.title("🏀 Баскетбол")
    st.caption("Нормальное распределение для разницы очков и тотала (много независимых владений мячом).")

    reg = build_seed_registry()
    home_team, away_team = _team_picker(reg, "Basketball", "basketball", ("Дома", "В гостях"))
    st.subheader(f"{home_team} — {away_team}")

    col1, col2 = st.columns(2)
    with col1:
        mu_margin = st.slider(
            f"Ожидаемая разница очков ({home_team} − {away_team})", -20.0, 20.0, 3.5, 0.5,
            help="Положительное число — команда дома в среднем сильнее на столько очков.",
        )
        sigma_margin = st.slider(
            "Разброс разницы очков (σ)", 1.0, 20.0, 12.0, 0.5,
            help="Насколько сильно реальный результат обычно отклоняется от ожидаемой разницы.",
        )
    with col2:
        mu_total = st.slider(
            "Ожидаемый суммарный тотал", 150.0, 260.0, 222.0, 1.0,
            help="Среднее суммарное число очков обеих команд в похожих матчах.",
        )
        sigma_total = st.slider(
            "Разброс тотала (σ)", 1.0, 30.0, 14.0, 0.5,
            help="Насколько сильно реальный суммарный счёт обычно отклоняется от ожидаемого тотала.",
        )

    ml_probs = basketball.predict_moneyline(mu_margin, sigma_margin)
    labels = {"home_win": home_team, "away_win": away_team}
    st.bar_chart(_probability_table(ml_probs, labels), x="outcome", y="probability")

    c1, c2 = st.columns(2)
    odds_home = c1.number_input(f"Коэффициент: победа {home_team}", min_value=1.01, value=1.85, step=0.01, help=_ODDS_HELP)
    odds_away = c2.number_input(f"Коэффициент: победа {away_team}", min_value=1.01, value=1.95, step=0.01, help=_ODDS_HELP)
    odds = {"home_win": odds_home, "away_win": odds_away}

    bankroll = st.number_input(
        "Банкролл", min_value=1.0, value=10_000.0, step=100.0, key="bb_bankroll", help=_BANKROLL_HELP
    )
    kelly_fraction = st.slider("Доля Kelly", 0.1, 1.0, 0.5, 0.05, key="bb_kelly", help=_KELLY_HELP)
    _render_value_section(ml_probs, odds, bankroll, kelly_fraction, labels=labels)

    st.markdown("**Тотал (over/under)**")
    line = st.number_input(
        "Линия тотала", value=228.5, step=0.5,
        help="Порог, относительно которого считается 'больше' (over) или 'меньше' (under).",
    )
    total_probs = basketball.predict_total(mu_total, sigma_total, line)
    st.dataframe(_probability_table(total_probs), width="stretch", hide_index=True)


def render_tennis_page() -> None:
    st.title("🎾 Теннис")
    st.caption("Вероятность победы в матче по вероятности выигрыша одного сета (race-to-N-sets).")

    reg = build_seed_registry()
    player1, player2 = _team_picker(reg, "Tennis", "tennis", ("Игрок 1", "Игрок 2"))
    st.subheader(f"{player1} — {player2}")

    p_set = st.slider(
        f"Вероятность выигрыша сета игроком {player1}", 0.01, 0.99, 0.55, 0.01,
        help="Насколько вероятнее игрок 1 выиграет один отдельный сет (0.5 = равные шансы на сет).",
    )
    best_of = st.radio(
        "Формат матча", [3, 5], horizontal=True,
        help="Best of 3 (женский/большинство турниров) или best of 5 (мужской, Большие шлемы).",
    )

    probs = tennis.predict_match_win_prob(p_set=p_set, best_of=best_of)
    labels = {"player1_win": player1, "player2_win": player2}
    st.bar_chart(_probability_table(probs, labels), x="outcome", y="probability")

    c1, c2 = st.columns(2)
    odds_p1 = c1.number_input(f"Коэффициент: {player1}", min_value=1.01, value=1.65, step=0.01, help=_ODDS_HELP)
    odds_p2 = c2.number_input(f"Коэффициент: {player2}", min_value=1.01, value=2.20, step=0.01, help=_ODDS_HELP)
    odds = {"player1_win": odds_p1, "player2_win": odds_p2}

    bankroll = st.number_input(
        "Банкролл", min_value=1.0, value=10_000.0, step=100.0, key="tennis_bankroll", help=_BANKROLL_HELP
    )
    kelly_fraction = st.slider("Доля Kelly", 0.1, 1.0, 0.5, 0.05, key="tennis_kelly", help=_KELLY_HELP)
    _render_value_section(probs, odds, bankroll, kelly_fraction, labels=labels)


_SPORT_DISPLAY_NAMES = {"Football": "⚽ Футбол", "Basketball": "🏀 Баскетбол", "Tennis": "🎾 Теннис"}


def _stable_match_id(league_id: int, competitor_ids: tuple[int, ...]) -> int:
    """Same picked match (league + pair of competitors) must always map to the
    same match_id within a session -- otherwise two outcomes on one real match
    (e.g. home win and draw) would look like independent matches to the
    correlation filter in `generate_coupons`, defeating its whole point."""
    key = (league_id, tuple(sorted(competitor_ids)))
    mapping = st.session_state.setdefault("_coupon_match_id_map", {})
    if key not in mapping:
        mapping[key] = 9000 + len(mapping)
    return mapping[key]


def _render_analysis_leg_builder(reg) -> None:
    st.subheader("Собрать ставку через анализ (рекомендуется)")
    st.caption(
        "Выберите вид спорта, чемпионат и команды/игроков — вероятность считается тем же "
        "движком, что и на страницах Футбол/Баскетбол/Теннис, EV виден сразу."
    )

    sport_name = st.selectbox(
        "Вид спорта", list(_SPORT_DISPLAY_NAMES),
        format_func=lambda s: _SPORT_DISPLAY_NAMES[s], key="coupon_analysis_sport",
    )
    role_labels = ("Игрок 1", "Игрок 2") if sport_name == "Tennis" else ("Дома", "В гостях")
    league_id, _sport_id, home, away = _pick_match(
        reg, sport_name, "coupon_analysis", role_labels, league_label="Чемпионат/лига"
    )
    match_time = st.datetime_input(
        "Дата и время матча", value="now", key="coupon_match_time",
        help="Вводится вручную — в seed-реестре нет реальных расписаний матчей. "
        "Реальное время в реальном времени (live) появляется только через AI-коннектор "
        "с живым источником данных.",
    )

    if sport_name == "Football":
        c1, c2, c3 = st.columns(3)
        lam_home = c1.slider(f"λ {home.name}", 0.1, 4.0, 1.6, 0.01, key="coupon_lam_home")
        lam_away = c2.slider(f"λ {away.name}", 0.1, 4.0, 1.2, 0.01, key="coupon_lam_away")
        rho = c3.slider("ρ (Dixon-Coles)", -0.2, 0.2, -0.08, 0.01, key="coupon_rho")
        matrix = dixon_coles_matrix(lam_home, lam_away, rho=rho, max_goals=10)
        probs = outcome_probs_from_matrix(matrix)
        outcome_labels = {"home_win": home.name, "draw": "Ничья", "away_win": away.name}
        oc = st.columns(3)
        odds = {
            "home_win": oc[0].number_input(home.name, min_value=1.01, value=1.90, step=0.01, key="coupon_odds_home"),
            "draw": oc[1].number_input("Ничья", min_value=1.01, value=3.40, step=0.01, key="coupon_odds_draw"),
            "away_win": oc[2].number_input(away.name, min_value=1.01, value=4.00, step=0.01, key="coupon_odds_away"),
        }
    elif sport_name == "Basketball":
        c1, c2 = st.columns(2)
        mu_margin = c1.slider(
            f"Ожид. разница очков ({home.name} − {away.name})", -20.0, 20.0, 3.0, 0.5, key="coupon_mu_margin"
        )
        sigma_margin = c2.slider("σ разницы очков", 1.0, 20.0, 12.0, 0.5, key="coupon_sigma_margin")
        probs = basketball.predict_moneyline(mu_margin, sigma_margin)
        outcome_labels = {"home_win": home.name, "away_win": away.name}
        oc = st.columns(2)
        odds = {
            "home_win": oc[0].number_input(home.name, min_value=1.01, value=1.85, step=0.01, key="coupon_odds_home_bb"),
            "away_win": oc[1].number_input(away.name, min_value=1.01, value=1.95, step=0.01, key="coupon_odds_away_bb"),
        }
    else:  # Tennis
        p_set = st.slider(f"P({home.name} выигрывает сет)", 0.01, 0.99, 0.55, 0.01, key="coupon_p_set")
        best_of = st.radio("Формат матча", [3, 5], horizontal=True, key="coupon_best_of")
        probs = tennis.predict_match_win_prob(p_set=p_set, best_of=best_of)
        outcome_labels = {"player1_win": home.name, "player2_win": away.name}
        oc = st.columns(2)
        odds = {
            "player1_win": oc[0].number_input(home.name, min_value=1.01, value=1.65, step=0.01, key="coupon_odds_p1"),
            "player2_win": oc[1].number_input(away.name, min_value=1.01, value=2.20, step=0.01, key="coupon_odds_p2"),
        }

    team_ids = (home.id, away.id)
    analysis_rows = [
        {
            "Исход": outcome_labels[k], "p_model": round(probs[k], 4), "odds": odds[k],
            "EV": round(expected_value(probs[k], odds[k]), 4),
        }
        for k in probs
    ]
    st.dataframe(pd.DataFrame(analysis_rows), width="stretch", hide_index=True)

    outcome_choice = st.selectbox(
        "Какой исход добавить в купон:", options=list(probs),
        format_func=lambda k: f"{outcome_labels[k]} (EV {expected_value(probs[k], odds[k]):+.1%})",
        key="coupon_outcome_choice",
    )
    if st.button("➕ Добавить в купон", key="coupon_add_leg"):
        legs = st.session_state.setdefault("coupon_legs", [])
        leg_labels = st.session_state.setdefault("_coupon_leg_labels", {})
        leg_times = st.session_state.setdefault("_coupon_leg_times", {})
        match_id = _stable_match_id(league_id, team_ids)
        bet_id = 1 if not legs else max(leg.bet_id for leg in legs) + 1
        legs.append(
            ValueBetCandidate(
                bet_id=bet_id, match_id=match_id, league_id=league_id,
                team_ids=team_ids, prob=probs[outcome_choice], odds=odds[outcome_choice],
            )
        )
        leg_labels[bet_id] = f"{home.name} — {away.name}: {outcome_labels[outcome_choice]}"
        leg_times[bet_id] = match_time
        st.success(
            f"Добавлено: {home.name} — {away.name}: {outcome_labels[outcome_choice]} "
            f"({match_time:%d.%m.%Y %H:%M})"
        )

    legs = st.session_state.get("coupon_legs", [])
    if legs:
        leg_labels = st.session_state.get("_coupon_leg_labels", {})
        leg_times = st.session_state.get("_coupon_leg_times", {})
        st.markdown("**Собранные ставки:**")
        legs_df = pd.DataFrame(
            [
                {
                    "bet_id": leg.bet_id,
                    "Матч / исход": leg_labels.get(leg.bet_id, f"#{leg.bet_id}"),
                    "Время матча": (
                        leg_times[leg.bet_id].strftime("%d.%m.%Y %H:%M") if leg.bet_id in leg_times else "—"
                    ),
                    "prob": round(leg.prob, 4),
                    "odds": leg.odds,
                }
                for leg in legs
            ]
        )
        st.dataframe(legs_df, width="stretch", hide_index=True)
        if st.button("🗑️ Очистить собранные ставки", key="coupon_clear_legs"):
            st.session_state["coupon_legs"] = []
            st.session_state["_coupon_leg_labels"] = {}
            st.session_state["_coupon_leg_times"] = {}
            st.rerun()


def render_coupon_page() -> None:
    st.title("🎫 Купон и монетизация")
    st.caption(
        "Соберите ставки на разные матчи (через анализ или вручную), сгенерируйте купон "
        "(экспресс) и посчитайте итоговую выплату с комиссией платформы."
    )

    reg = build_seed_registry()
    _render_analysis_leg_builder(reg)

    with st.expander("✍️ Ручной ввод (для опытных / произвольные данные)"):
        st.markdown(
            "- **bet_id** — произвольный номер строки.\n"
            "- **match_id** — номер матча (одинаковый у ставок на один и тот же матч — так купон "
            "поймёт, что их нельзя комбинировать вместе).\n"
            "- **league_id** — номер лиги (для проверки корреляции между ставками одной лиги).\n"
            "- **team_ids** — id команд через запятую, например `1,2`. Соответствие ID → название "
            "команды смотрите в справочнике ниже.\n"
            "- **prob** — ваша оценка вероятности исхода (0–1).\n"
            "- **odds** — коэффициент букмекера на этот исход."
        )
        reference_rows = []
        for c in reg.competitors.values():
            league = reg.leagues[c.league_id]
            reference_rows.append({"ID": c.id, "Название": c.name, "Лига": league.name})
        st.caption("Справочник команд (ID → название):")
        st.dataframe(pd.DataFrame(reference_rows).sort_values("ID"), width="stretch", hide_index=True)

        default_rows = pd.DataFrame(
            [
                {"bet_id": 101, "match_id": 1001, "league_id": 1, "team_ids": "1,2", "prob": 0.55, "odds": 2.00},
                {"bet_id": 102, "match_id": 1002, "league_id": 1, "team_ids": "3,4", "prob": 0.40, "odds": 2.85},
            ]
        )
        edited = st.data_editor(default_rows, num_rows="dynamic", width="stretch", key="coupon_manual_table")

    bankroll = st.number_input(
        "Банкролл", min_value=1.0, value=10_000.0, step=100.0, key="coupon_bankroll", help=_BANKROLL_HELP
    )
    col1, col2, col3 = st.columns(3)
    max_events = col1.slider(
        "Макс. ставок в купоне", 1, 5, 3, key="coupon_max_events",
        help="Ограничение на число ног (ставок) в одном экспрессе.",
    )
    max_corr = col2.slider(
        "Порог допустимой корреляции", 0.0, 1.0, 0.3, 0.05,
        help="Ставки, которые слишком сильно связаны друг с другом (тот же матч/команда/лига), "
        "не объединяются в один купон, если их 'корреляция' выше этого порога.",
    )
    kelly_fraction = col3.slider("Доля Kelly", 0.1, 1.0, 0.5, 0.05, key="coupon_kelly", help=_KELLY_HELP)

    manual_candidates, skipped_rows = parse_bet_rows(edited)
    if skipped_rows:
        details = "; ".join(f"строка {idx + 1}: {msg}" for idx, msg in skipped_rows)
        st.warning(f"Пропущено строк с некорректными данными ({len(skipped_rows)}): {details}")

    candidates = st.session_state.get("coupon_legs", []) + manual_candidates
    if not candidates:
        st.info("Добавьте хотя бы одну ставку через анализ выше или в разделе ручного ввода.")
        return

    if st.button("Сгенерировать купоны", type="primary"):
        try:
            coupons = generate_coupons(
                candidates, bankroll=bankroll, max_events=max_events, max_corr=max_corr,
                kelly_fraction=kelly_fraction, top_n=5,
            )
            st.session_state["coupons"] = coupons
        except ValueError as exc:
            st.error(f"Слишком много комбинаций для перебора: {exc}")

    coupons = st.session_state.get("coupons", [])
    if not coupons:
        st.info("Нажмите «Сгенерировать купоны», чтобы увидеть комбинации с положительным EV.")
        return

    leg_times = st.session_state.get("_coupon_leg_times", {})

    def _earliest_time(combo) -> str:
        times = [leg_times[b.bet_id] for b in combo if b.bet_id in leg_times]
        return min(times).strftime("%d.%m.%Y %H:%M") if times else "—"

    leg_labels = st.session_state.get("_coupon_leg_labels", {})

    def _leg_display_name(leg) -> str:
        # Prefer the "Home — Away: outcome" label recorded by the analysis
        # builder; manually-typed rows have no such label, but their
        # `team_ids` still resolve to real names in the entity registry (the
        # manual table's reference expander uses the same IDs), so club
        # names are shown either way -- never a bare "#3".
        if leg.bet_id in leg_labels:
            return leg_labels[leg.bet_id]
        names = [reg.competitors[tid].name for tid in leg.team_ids if tid in reg.competitors]
        return " / ".join(names) if names else f"#{leg.bet_id}"

    coupon_df = pd.DataFrame(
        [
            {
                "legs": ", ".join(_leg_display_name(b) for b in c["combo"]),
                "n_legs": c["n_legs"],
                "Ближайший матч": _earliest_time(c["combo"]),
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
        help="Выберите одну из сгенерированных комбинаций, чтобы посчитать выплату.",
    )
    stake = st.number_input("Ставка на купон", min_value=1.0, value=100.0, step=10.0)
    fee_pct = st.slider(
        "Комиссия платформы, %", 0, 30, 5,
        help="Какой процент от валовой выплаты платформа удерживает себе — это расчёт, не реальный платёж.",
    ) / 100.0

    report = build_coupon_report(coupons[choice_idx], stake=stake, platform_fee_pct=fee_pct)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Общий коэффициент", f"{report.total_odds:.2f}", help="Произведение коэффициентов всех ног купона.")
    m2.metric("Валовая выплата", f"{report.gross_payout:.2f}", help="Ставка × общий коэффициент, если купон зайдёт.")
    m3.metric("Комиссия", f"{report.platform_fee:.2f}", help="Комиссия платформы от валовой выплаты.")
    m4.metric(
        "Чистая выплата", f"{report.net_payout:.2f}", delta=f"{report.net_profit:+.2f}",
        help="Валовая выплата за вычетом комиссии; дельта — чистая прибыль сверх ставки.",
    )
    st.code(format_coupon_report(report), language=None)

    if st.button("💾 Сохранить купон в историю", key="coupon_save_history"):
        history = st.session_state.setdefault("coupon_history", [])
        combo = coupons[choice_idx]["combo"]
        match_times = [leg_times[b.bet_id] for b in combo if b.bet_id in leg_times]
        history.append(
            {
                "saved_at": datetime.now(),
                "match_time": min(match_times) if match_times else None,
                "legs": coupon_df.iloc[choice_idx]["legs"],
                "n_legs": int(coupon_df.iloc[choice_idx]["n_legs"]),
                "stake": stake,
                "net_profit": report.net_profit,
            }
        )
        st.success("Купон сохранён в историю.")

    _render_coupon_history_section()


def _render_coupon_history_section() -> None:
    st.subheader("📊 История и статистика по купонам")
    st.caption(
        "Локальная история только текущей сессии браузера (не сохраняется между запусками) — "
        "статистика по периодам считается от времени матча (если указано) или от момента сохранения."
    )
    history = st.session_state.get("coupon_history", [])
    if not history:
        st.info("Пока нет сохранённых купонов — нажмите «Сохранить купон в историю» выше.")
        return

    period_labels = {"day": "Сегодня", "month": "Этот месяц", "all": "Всё время"}
    period = st.radio(
        "Период:", list(period_labels), format_func=lambda p: period_labels[p],
        horizontal=True, key="coupon_history_period",
    )
    now = datetime.now()

    def _in_period(record: dict) -> bool:
        moment = record["match_time"] or record["saved_at"]
        if period == "day":
            return moment.date() == now.date()
        if period == "month":
            return (moment.year, moment.month) == (now.year, now.month)
        return True

    filtered = [h for h in history if _in_period(h)]
    if not filtered:
        st.info(f"Нет сохранённых купонов за период «{period_labels[period]}».")
    else:
        total_stake = sum(h["stake"] for h in filtered)
        total_profit = sum(h["net_profit"] for h in filtered)
        roi = total_profit / total_stake if total_stake > 0 else 0.0
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Купонов за период", len(filtered))
        m2.metric("Сумма ставок", f"{total_stake:.2f}")
        m3.metric("Чистая прибыль", f"{total_profit:+.2f}")
        m4.metric("ROI", f"{roi:+.1%}", help="Чистая прибыль / сумма ставок за выбранный период.")

        history_df = pd.DataFrame(
            [
                {
                    "Сохранено": h["saved_at"].strftime("%d.%m.%Y %H:%M"),
                    "Матч": h["match_time"].strftime("%d.%m.%Y %H:%M") if h["match_time"] else "—",
                    "Ставки": h["legs"],
                    "Ставка": h["stake"],
                    "Прибыль": round(h["net_profit"], 2),
                }
                for h in filtered
            ]
        )
        st.dataframe(history_df, width="stretch", hide_index=True)

    if st.button("🗑️ Очистить историю купонов", key="coupon_clear_history"):
        st.session_state["coupon_history"] = []
        st.rerun()


def render_entities_page() -> None:
    st.title("🌍 Страны и лиги")
    reg = build_seed_registry()
    countries_with_data = sum(1 for c in reg.countries.values() if reg.has_league_data(c.id))

    m1, m2 = st.columns(2)
    m1.metric(
        "Стран в системе (реальный список ISO)", len(reg.countries),
        help="Полный официальный список стран/территорий ISO 3166-1 — это все страны мира, не подмножество.",
    )
    m2.metric(
        "Стран с данными по лигам/клубам", countries_with_data,
        help="Курируемая, реальная, но не исчерпывающая подборка — не путать со списком всех стран слева.",
    )
    st.caption(
        "Список стран — полный и настоящий (ISO 3166-1). А вот лиги и клубы — курируемая подборка "
        "известных реальных турниров/команд, а не исчерпывающая база каждого клуба каждой страны — "
        "получить действительно полное покрытие можно через AI-коннектор данных (см. страницу «Справка», "
        "раздел про AI-коннектор, или USAGE.md, раздел 9)."
    )

    mode = st.radio("Смотреть по:", ["Виду спорта", "Стране"], horizontal=True)

    if mode == "Виду спорта":
        sport_names = {s.id: s.name for s in reg.sports.values()}
        sport_id = st.selectbox("Вид спорта", options=list(sport_names), format_func=lambda i: sport_names[i])

        leagues = reg.leagues_for_sport(sport_id)
        league_names = {lg.id: lg.name for lg in leagues}
        league_id = st.selectbox("Лига", options=list(league_names), format_func=lambda i: league_names[i])

        competitors = reg.competitors_for_league(league_id)
        df = pd.DataFrame([{"Участник": c.name, "Тип": c.kind.value} for c in competitors])
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        countries = reg.all_countries_sorted()
        country_names = {c.id: f"{c.name} ({c.iso_code})" for c in countries}
        country_id = st.selectbox(
            "Страна (полный список из 249 стран мира)",
            options=list(country_names), format_func=lambda i: country_names[i],
            key="entities_country_select",
        )

        leagues = reg.leagues_for_country(country_id)
        if not leagues:
            st.info(
                "Для этой страны в seed-реестре пока нет лиг/клубов — это ожидаемо для "
                "большинства из 249 стран (курируемая подборка, не полная база). Реальные "
                "живые данные по любой стране можно получить через AI-коннектор "
                "(`bukmeker connector`)."
            )
        else:
            for league in leagues:
                st.markdown(f"**{league.name}** ({reg.sport_of_league(league.id).name})")
                competitors = reg.competitors_for_league(league.id)
                df = pd.DataFrame([{"Участник": c.name, "Тип": c.kind.value} for c in competitors])
                st.dataframe(df, width="stretch", hide_index=True)


_CLAUDE_MODELS = ["claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5-20251001", "claude-fable-5"]


def _redact(message: str, *secrets: str) -> str:
    """Strips any secret value out of an error message before it's ever
    displayed -- e.g. requests' own exception text includes the full request
    URL, which contains the provider API key verbatim if key_location is
    "query"."""
    for secret in secrets:
        if secret:
            message = message.replace(secret, "***")
    return message


def render_connector_page() -> None:
    st.title("🔌 AI-коннектор данных")
    st.warning(
        "В отличие от всех остальных страниц дашборда, эта страница делает "
        "РЕАЛЬНЫЕ сетевые запросы к указанному провайдеру и РЕАЛЬНЫЙ платный "
        "вызов Anthropic API. Ключи ниже используются только в текущей сессии "
        "браузера, никуда не сохраняются и не логируются."
    )

    st.subheader("Источник спортивных данных")
    source_url = st.text_input(
        "Base URL провайдера", placeholder="https://api.ваш-провайдер.com",
        help="Адрес REST/JSON API вашего источника данных (без конечного эндпоинта).",
    )
    source_key = st.text_input(
        "API-ключ провайдера", type="password",
        help="Ключ, который выдал вам провайдер спортивных данных.",
    )
    c1, c2, c3 = st.columns(3)
    key_location = c1.selectbox(
        "Где ожидается ключ", ["header", "query"],
        help="Смотрите документацию провайдера: ключ передаётся в заголовке запроса или как query-параметр.",
    )
    key_name = c2.text_input(
        "Имя заголовка/параметра", value="x-api-key",
        help="Например: x-apisports-key, apiKey — тоже из документации провайдера.",
    )
    path = c3.text_input("Endpoint (путь запроса)", value="fixtures")

    st.subheader("Anthropic API (версия ИИ для маппинга полей)")
    anthropic_key = st.text_input(
        "Anthropic API-ключ", type="password",
        help="Ваш ключ с console.anthropic.com — используется для сопоставления схемы провайдера.",
    )
    model = st.selectbox(
        "Версия модели", _CLAUDE_MODELS,
        help="Модель Anthropic, которая один раз анализирует пример записи от провайдера и "
        "определяет, как её поля соответствуют нашей канонической схеме.",
    )

    reg = build_seed_registry()
    do_sync = st.checkbox(
        "Синхронизировать результат в реестр сущностей (bukmeker.entities)",
        help="Дополнительно слить полученные лиги/команды в реестр — как флаг --sync у `bukmeker connector`.",
    )
    sync_sport_id, sync_country_id = None, None
    if do_sync:
        sc1, sc2 = st.columns(2)
        sport_names = {s.id: s.name for s in reg.sports.values()}
        sync_sport_id = sc1.selectbox(
            "Вид спорта для новых лиг", options=list(sport_names), format_func=lambda i: sport_names[i]
        )
        countries = reg.all_countries_sorted()
        country_names = {c.id: f"{c.name} ({c.iso_code})" for c in countries}
        sync_country_id = sc2.selectbox(
            "Страна для новых лиг", options=list(country_names), format_func=lambda i: country_names[i],
            help="В ответах провайдеров обычно нет страны лиги — указывается вручную.",
        )

    if st.button("Получить и нормализовать данные", type="primary"):
        if not source_url or not source_key or not anthropic_key:
            st.error("Заполните URL провайдера, ключ провайдера и ключ Anthropic — все три обязательны.")
        else:
            try:
                from bukmeker.connectors import AIDataConnector, ClaudeFieldMapper, RawDataSource

                source = RawDataSource(
                    base_url=source_url, api_key=source_key, key_location=key_location, key_name=key_name
                )
                mapper = ClaudeFieldMapper(api_key=anthropic_key, model=model)
                matches = AIDataConnector(source, mapper).fetch_and_normalize(path)

                st.success(f"Получено и нормализовано записей: {len(matches)}")
                if matches:
                    df = pd.DataFrame(
                        [
                            {
                                "home_team": m.home_team, "away_team": m.away_team,
                                "league": m.league, "home_odds": m.home_odds,
                                "draw_odds": m.draw_odds, "away_odds": m.away_odds,
                            }
                            for m in matches
                        ]
                    )
                    st.dataframe(df, width="stretch", hide_index=True)

                if do_sync:
                    from bukmeker.connectors import sync_registry_from_matches

                    report = sync_registry_from_matches(
                        reg, matches, sport_id=sync_sport_id, fallback_country_id=sync_country_id
                    )
                    st.info(
                        f"Синхронизация: +{report.leagues_added} лиг, +{report.competitors_added} "
                        f"команд ({report.matches_processed} записей обработано, "
                        f"{report.skipped_incomplete} пропущено)."
                    )
            except Exception as exc:  # noqa: BLE001 -- surfaced to the user, secrets redacted first
                st.error(f"Ошибка: {_redact(str(exc), source_key, anthropic_key)}")


def render_about_page() -> None:
    st.title("ℹ️ О проекте")
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


_PAGES_DIR = Path(__file__).resolve().parent / "dashboard_pages"


def main() -> None:
    st.set_page_config(page_title="Bukmeker — Value Betting Dashboard", layout="wide")

    if not require_password():
        return

    # File-based pages (rather than passing the render_* functions directly to
    # st.Page) so that AppTest.switch_page can target each page by file path --
    # callable-based st.Page entries aren't addressable that way. Paths are
    # absolute (resolved from this file's location) so navigation works
    # identically whether Streamlit is launched on this file or on the
    # repo-root `dashboard_app.py` wrapper.
    pages = [
        st.Page(_PAGES_DIR / "help.py", title="Справка", icon="❓", default=True),
        st.Page(_PAGES_DIR / "football.py", title="Футбол", icon="⚽"),
        st.Page(_PAGES_DIR / "basketball.py", title="Баскетбол", icon="🏀"),
        st.Page(_PAGES_DIR / "tennis.py", title="Теннис", icon="🎾"),
        st.Page(_PAGES_DIR / "coupon.py", title="Купон и монетизация", icon="🎫"),
        st.Page(_PAGES_DIR / "entities.py", title="Страны и лиги", icon="🌍"),
        st.Page(_PAGES_DIR / "connector.py", title="AI-коннектор", icon="🔌"),
        st.Page(_PAGES_DIR / "about.py", title="О проекте", icon="ℹ️"),
    ]
    navigation = st.navigation(pages)
    st.sidebar.caption("Мультиспортивный движок оценки value bets, купонов и монетизации.")
    navigation.run()


if __name__ == "__main__":
    main()
