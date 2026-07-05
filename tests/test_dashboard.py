"""Headless checks that the Streamlit dashboard actually boots and its
widgets produce correct output -- using streamlit's official AppTest harness
(no browser, no real server) rather than trusting it "looks right"."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_APP_PATH = str(Path(__file__).resolve().parent.parent / "dashboard_app.py")


def _booted_app() -> AppTest:
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    assert not at.exception, f"Dashboard raised on boot: {at.exception}"
    return at


def test_dashboard_boots_with_all_tabs_and_no_exceptions():
    at = _booted_app()
    assert len(at.tabs) == 6


def test_football_tab_renders_probability_dataframe():
    at = _booted_app()
    football_tab = at.tabs[0]
    assert len(football_tab.dataframe) >= 1
    assert len(football_tab.metric) == 1


def test_coupon_tab_generates_positive_ev_coupons_on_click():
    at = _booted_app()
    coupon_tab = at.tabs[3]
    at = coupon_tab.button[0].click().run(timeout=30)
    assert not at.exception

    # AppTest exposes both st.data_editor (the input rows) and st.dataframe
    # (the resulting coupons) under `.dataframe` -- the coupons table is the
    # one with an "EV" column.
    coupon_dataframes = at.tabs[3].dataframe
    coupons_table = next(df for df in coupon_dataframes if "EV" in df.value.columns)
    assert all(ev > 0 for ev in coupons_table.value["EV"])


def test_coupon_tab_monetization_metrics_match_worked_example():
    at = _booted_app()
    at = at.tabs[3].button[0].click().run(timeout=30)

    metrics = {m.label: m.value for m in at.tabs[3].metric}
    assert metrics["Общий коэффициент"] == "5.70"
    assert metrics["Валовая выплата"] == "570.00"
    assert metrics["Комиссия"] == "28.50"
    assert metrics["Чистая выплата"] == "541.50"


def test_entities_tab_shows_non_empty_competitor_table():
    at = _booted_app()
    entities_tab = at.tabs[4]
    assert len(entities_tab.dataframe) == 1
    assert len(entities_tab.dataframe[0].value) > 0
