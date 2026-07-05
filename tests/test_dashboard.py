"""Headless checks that the Streamlit dashboard actually boots and its
widgets produce correct output -- using streamlit's official AppTest harness
(no browser, no real server) rather than trusting it "looks right".

Targets `bukmeker/dashboard.py` directly (not the root `dashboard_app.py`
wrapper) because that's exactly what `bukmeker dashboard` runs in production,
and because file-based multipage navigation resolves page paths relative to
whichever script AppTest was constructed from.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_DASHBOARD_PY = str(Path(__file__).resolve().parent.parent / "bukmeker" / "dashboard.py")
_PAGES_DIR = Path(__file__).resolve().parent.parent / "bukmeker" / "dashboard_pages"


def _page_path(name: str) -> str:
    return str(_PAGES_DIR / f"{name}.py")


def _booted(page: str | None = None) -> AppTest:
    at = AppTest.from_file(_DASHBOARD_PY)
    at.run(timeout=30)
    assert not at.exception, f"Dashboard raised on boot: {at.exception}"
    if page is not None:
        at.switch_page(_page_path(page))
        at.run(timeout=30)
        assert not at.exception, f"Dashboard raised switching to {page}: {at.exception}"
    return at


def test_dashboard_boots_to_help_page_by_default():
    at = _booted()
    assert [t.value for t in at.title] == ["Справка: как пользоваться"]


@pytest.mark.parametrize(
    "page,expected_title",
    [
        ("football", "⚽ Футбол"),
        ("basketball", "🏀 Баскетбол"),
        ("tennis", "🎾 Теннис"),
        ("coupon", "🎫 Купон и монетизация"),
        ("entities", "🌍 Страны и лиги"),
        ("about", "ℹ️ О проекте"),
    ],
)
def test_every_page_is_reachable_via_navigation(page, expected_title):
    at = _booted(page)
    assert [t.value for t in at.title] == [expected_title]


def test_football_page_renders_probability_dataframe_and_metric():
    at = _booted("football")
    assert len(at.dataframe) >= 1
    assert len(at.metric) == 1


def test_coupon_page_generates_positive_ev_coupons_on_click():
    at = _booted("coupon")
    at = at.button[0].click().run(timeout=30)
    assert not at.exception

    # AppTest exposes both st.data_editor (the input rows) and st.dataframe
    # (the resulting coupons) under `.dataframe` -- the coupons table is the
    # one with an "EV" column.
    coupons_table = next(df for df in at.dataframe if "EV" in df.value.columns)
    assert all(ev > 0 for ev in coupons_table.value["EV"])


def test_coupon_page_monetization_metrics_match_worked_example():
    at = _booted("coupon")
    at = at.button[0].click().run(timeout=30)

    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Общий коэффициент"] == "5.70"
    assert metrics["Валовая выплата"] == "570.00"
    assert metrics["Комиссия"] == "28.50"
    assert metrics["Чистая выплата"] == "541.50"


def test_entities_page_shows_non_empty_competitor_table_by_default():
    at = _booted("entities")
    assert len(at.dataframe) == 1
    assert len(at.dataframe[0].value) > 0


def test_entities_page_reports_all_249_real_countries():
    at = _booted("entities")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Стран в системе (реальный список ISO)"] == "249"
    # curated real subset, not exhaustive -- must be far fewer than 249
    assert int(metrics["Стран с данными по лигам/клубам"]) < 249


def test_entities_page_country_mode_shows_honest_no_data_message_by_default():
    at = _booted("entities")
    at.radio[0].set_value("Стране").run(timeout=30)
    assert not at.exception
    # alphabetically-first countries (e.g. Afghanistan) have no seeded league data
    assert len(at.info) == 1


def test_entities_page_country_mode_shows_leagues_for_a_seeded_country():
    from bukmeker.entities import build_seed_registry

    usa_id = build_seed_registry().country_by_alpha3("USA").id

    at = _booted("entities")
    at.radio[0].set_value("Стране").run(timeout=30)
    at.session_state["entities_country_select"] = usa_id
    at.run(timeout=30)

    assert not at.exception
    assert len(at.info) == 0
    league_names = " ".join(m.value for m in at.markdown)
    assert "MLS" in league_names


def test_help_page_contains_glossary_terms():
    at = _booted()
    all_text = " ".join(m.value for m in at.markdown)
    for term in ("EV", "Kelly", "Overround", "λ", "ρ"):
        assert term in all_text, f"glossary should mention {term!r}"
