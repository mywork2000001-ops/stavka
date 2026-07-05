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

    col1, col2 = st.columns(2)
    with col1:
        lam_home = st.slider(
            "Ожидаемые голы дома (λ_home)", 0.1, 4.0, 2.47, 0.01,
            help="Среднее число голов, которое команда обычно забивает дома в похожих матчах.",
        )
        rho = st.slider(
            "Поправка Dixon-Coles (ρ)", -0.2, 0.2, -0.08, 0.01,
            help="Небольшая техническая поправка для редких низких счётов (0:0, 1:0, 0:1). "
            "Обычно небольшое отрицательное число, можно оставить по умолчанию.",
        )
    with col2:
        lam_away = st.slider(
            "Ожидаемые голы в гостях (λ_away)", 0.1, 4.0, 1.02, 0.01,
            help="Среднее число голов, которое команда обычно забивает в гостях в похожих матчах.",
        )

    matrix = dixon_coles_matrix(lam_home, lam_away, rho=rho, max_goals=10)
    probs = outcome_probs_from_matrix(matrix)
    st.bar_chart(_probability_table(probs), x="outcome", y="probability")

    st.markdown("**Коэффициенты букмекера (1X2)**")
    c1, c2, c3 = st.columns(3)
    odds_home = c1.number_input("Дома", min_value=1.01, value=1.75, step=0.01, help=_ODDS_HELP)
    odds_draw = c2.number_input("Ничья", min_value=1.01, value=3.60, step=0.01, help=_ODDS_HELP)
    odds_away = c3.number_input("В гостях", min_value=1.01, value=4.75, step=0.01, help=_ODDS_HELP)
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
    _render_value_section(probs, odds, bankroll, kelly_fraction)


def render_basketball_page() -> None:
    st.title("🏀 Баскетбол")
    st.caption("Нормальное распределение для разницы очков и тотала (много независимых владений мячом).")

    col1, col2 = st.columns(2)
    with col1:
        mu_margin = st.slider(
            "Ожидаемая разница очков (дома − гости)", -20.0, 20.0, 3.5, 0.5,
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
    st.bar_chart(_probability_table(ml_probs), x="outcome", y="probability")

    c1, c2 = st.columns(2)
    odds_home = c1.number_input("Коэффициент: победа дома", min_value=1.01, value=1.85, step=0.01, help=_ODDS_HELP)
    odds_away = c2.number_input("Коэффициент: победа в гостях", min_value=1.01, value=1.95, step=0.01, help=_ODDS_HELP)
    odds = {"home_win": odds_home, "away_win": odds_away}

    bankroll = st.number_input(
        "Банкролл", min_value=1.0, value=10_000.0, step=100.0, key="bb_bankroll", help=_BANKROLL_HELP
    )
    kelly_fraction = st.slider("Доля Kelly", 0.1, 1.0, 0.5, 0.05, key="bb_kelly", help=_KELLY_HELP)
    _render_value_section(ml_probs, odds, bankroll, kelly_fraction)

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

    p_set = st.slider(
        "Вероятность выигрыша сета игроком 1", 0.01, 0.99, 0.55, 0.01,
        help="Насколько вероятнее игрок 1 выиграет один отдельный сет (0.5 = равные шансы на сет).",
    )
    best_of = st.radio(
        "Формат матча", [3, 5], horizontal=True,
        help="Best of 3 (женский/большинство турниров) или best of 5 (мужской, Большие шлемы).",
    )

    probs = tennis.predict_match_win_prob(p_set=p_set, best_of=best_of)
    st.bar_chart(_probability_table(probs), x="outcome", y="probability")

    c1, c2 = st.columns(2)
    odds_p1 = c1.number_input("Коэффициент: игрок 1", min_value=1.01, value=1.65, step=0.01, help=_ODDS_HELP)
    odds_p2 = c2.number_input("Коэффициент: игрок 2", min_value=1.01, value=2.20, step=0.01, help=_ODDS_HELP)
    odds = {"player1_win": odds_p1, "player2_win": odds_p2}

    bankroll = st.number_input(
        "Банкролл", min_value=1.0, value=10_000.0, step=100.0, key="tennis_bankroll", help=_BANKROLL_HELP
    )
    kelly_fraction = st.slider("Доля Kelly", 0.1, 1.0, 0.5, 0.05, key="tennis_kelly", help=_KELLY_HELP)
    _render_value_section(probs, odds, bankroll, kelly_fraction)


def render_coupon_page() -> None:
    st.title("🎫 Купон и монетизация")
    st.caption(
        "Добавьте несколько ставок на разные матчи, сгенерируйте купон (экспресс) "
        "и посчитайте итоговую выплату с комиссией платформы."
    )

    with st.expander("Как заполнять таблицу ставок?"):
        st.markdown(
            "- **bet_id** — произвольный номер строки.\n"
            "- **match_id** — номер матча (одинаковый у ставок на один и тот же матч — так купон "
            "поймёт, что их нельзя комбинировать вместе).\n"
            "- **league_id** — номер лиги (для проверки корреляции между ставками одной лиги).\n"
            "- **team_ids** — id команд через запятую, например `1,2`.\n"
            "- **prob** — ваша оценка вероятности исхода (0–1).\n"
            "- **odds** — коэффициент букмекера на этот исход."
        )

    default_rows = pd.DataFrame(
        [
            {"bet_id": 1, "match_id": 1001, "league_id": 1, "team_ids": "1,2", "prob": 0.55, "odds": 2.00},
            {"bet_id": 2, "match_id": 1002, "league_id": 1, "team_ids": "3,4", "prob": 0.40, "odds": 2.85},
            {"bet_id": 3, "match_id": 1003, "league_id": 2, "team_ids": "5,6", "prob": 0.30, "odds": 3.80},
        ]
    )
    edited = st.data_editor(default_rows, num_rows="dynamic", width="stretch")

    bankroll = st.number_input(
        "Банкролл", min_value=1.0, value=10_000.0, step=100.0, key="coupon_bankroll", help=_BANKROLL_HELP
    )
    col1, col2, col3 = st.columns(3)
    max_events = col1.slider(
        "Макс. ставок в купоне", 1, 5, 3, help="Ограничение на число ног (ставок) в одном экспрессе."
    )
    max_corr = col2.slider(
        "Порог допустимой корреляции", 0.0, 1.0, 0.3, 0.05,
        help="Ставки, которые слишком сильно связаны друг с другом (тот же матч/команда/лига), "
        "не объединяются в один купон, если их 'корреляция' выше этого порога.",
    )
    kelly_fraction = col3.slider("Доля Kelly", 0.1, 1.0, 0.5, 0.05, key="coupon_kelly", help=_KELLY_HELP)

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
        st.Page(_PAGES_DIR / "about.py", title="О проекте", icon="ℹ️"),
    ]
    navigation = st.navigation(pages)
    st.sidebar.caption("Мультиспортивный движок оценки value bets, купонов и монетизации.")
    navigation.run()


if __name__ == "__main__":
    main()
