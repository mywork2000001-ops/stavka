"""Headless checks that the Streamlit dashboard actually boots and its
widgets produce correct output -- using streamlit's official AppTest harness
(no browser, no real server) rather than trusting it "looks right".

Targets `bukmeker/dashboard.py` directly (not the root `dashboard_app.py`
wrapper) because that's exactly what `bukmeker dashboard` runs in production,
and because file-based multipage navigation resolves page paths relative to
whichever script AppTest was constructed from.
"""

import json
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


def test_football_page_warns_instead_of_crashing_on_margin_free_odds():
    at = _booted("football")
    number_inputs = {ni.label: ni for ni in at.number_input}
    # 1/3 + 1/3 + 1/3 == 1.0 exactly -> no bookmaker margin to remove
    number_inputs["Arsenal"].set_value(3.0)
    number_inputs["Ничья"].set_value(3.0)
    number_inputs["Chelsea"].set_value(3.0)
    at = at.run(timeout=30)

    assert not at.exception
    assert any("не содержат маржи" in w.value for w in at.warning)


def test_football_page_shows_real_team_names_not_bare_sliders():
    at = _booted("football")
    subheaders = [s.value for s in at.subheader]
    assert len(subheaders) == 1
    home, _, away = subheaders[0].partition(" — ")
    assert home and away and home != away
    # the outcome chart/table must be labelled with the actual team names,
    # not generic "home_win"/"away_win" keys
    assert any(home in df.value["outcome"].values for df in at.dataframe)


def test_basketball_page_shows_real_team_names():
    at = _booted("basketball")
    subheaders = [s.value for s in at.subheader]
    assert len(subheaders) == 1
    home, _, away = subheaders[0].partition(" — ")
    assert home and away and home != away


def test_tennis_page_shows_real_player_names():
    at = _booted("tennis")
    subheaders = [s.value for s in at.subheader]
    assert len(subheaders) == 1
    p1, _, p2 = subheaders[0].partition(" — ")
    assert p1 and p2 and p1 != p2


def test_coupon_page_shows_team_id_reference_table():
    at = _booted("coupon")
    reference = next(df for df in at.dataframe if list(df.value.columns) == ["ID", "Название", "Лига"])
    assert "Arsenal" in reference.value["Название"].values


def test_coupon_page_analysis_builder_shows_real_teams_and_computed_ev():
    at = _booted("coupon")
    selectboxes = {sb.label: sb for sb in at.selectbox}
    assert selectboxes["Дома"].value == "Arsenal"
    assert selectboxes["В гостях"].value == "Chelsea"

    analysis_table = next(
        df for df in at.dataframe if list(df.value.columns) == ["Исход", "p_model", "odds", "EV"]
    )
    assert "Arsenal" in analysis_table.value["Исход"].values


def test_coupon_page_analysis_builder_switches_to_basketball():
    at = _booted("coupon")
    at.session_state["coupon_analysis_sport"] = "Basketball"
    at = at.run(timeout=30)
    assert not at.exception

    selectboxes = {sb.label: sb.value for sb in at.selectbox}
    assert selectboxes["Дома"] and selectboxes["В гостях"]
    analysis_table = next(
        df for df in at.dataframe if list(df.value.columns) == ["Исход", "p_model", "odds", "EV"]
    )
    assert len(analysis_table.value) == 2  # home_win/away_win only, no draw


def test_coupon_page_analysis_builder_switches_to_tennis():
    at = _booted("coupon")
    at.session_state["coupon_analysis_sport"] = "Tennis"
    at = at.run(timeout=30)
    assert not at.exception

    selectboxes = {sb.label: sb.value for sb in at.selectbox}
    assert selectboxes["Игрок 1"] and selectboxes["Игрок 2"]
    analysis_table = next(
        df for df in at.dataframe if list(df.value.columns) == ["Исход", "p_model", "odds", "EV"]
    )
    assert selectboxes["Игрок 1"] in analysis_table.value["Исход"].values


def test_coupon_page_add_leg_button_appends_a_real_analyzed_bet():
    at = _booted("coupon")
    at = _click(at, "➕ Добавить в купон")
    assert not at.exception
    assert len(at.success) == 1

    legs = at.session_state["coupon_legs"]
    assert len(legs) == 1
    assert 0.0 <= legs[0].prob <= 1.0
    assert legs[0].team_ids == (1, 2)  # Arsenal, Chelsea


def test_coupon_page_records_match_time_for_added_leg():
    import datetime

    at = _booted("coupon")
    assert len(at.datetime_input) == 1

    at = _click(at, "➕ Добавить в купон")
    assert not at.exception

    leg_times = at.session_state["_coupon_leg_times"]
    assert isinstance(leg_times[1], datetime.datetime)

    legs_table = next(df for df in at.dataframe if "Время матча" in df.value.columns)
    assert legs_table.value["Время матча"].iloc[0] != "—"


def test_coupon_page_clear_legs_button_empties_collected_bets():
    at = _booted("coupon")
    at = _click(at, "➕ Добавить в купон")
    assert len(at.session_state["coupon_legs"]) == 1

    at = _click(at, "🗑️ Очистить собранные ставки")
    assert not at.exception
    assert at.session_state["coupon_legs"] == []
    assert at.session_state["_coupon_leg_labels"] == {}
    assert at.session_state["_coupon_leg_times"] == {}


def test_coupon_page_generate_button_reports_combinatorial_guard_error():
    from bukmeker.coupon import ValueBetCandidate

    at = _booted("coupon")
    # inject 50 synthetic legs directly into session state -- far past the
    # generate_coupons() safety limit for max_events=5 combinations
    at.session_state["coupon_legs"] = [
        ValueBetCandidate(bet_id=i, match_id=100 + i, league_id=i, team_ids=(i,), prob=0.6, odds=2.0)
        for i in range(50)
    ]
    at.session_state["coupon_max_events"] = 5
    at = _click(at, "Сгенерировать купоны")

    assert not at.exception
    assert any("Слишком много комбинаций" in e.value for e in at.error)


def test_coupon_page_generate_includes_analysis_leg_alongside_manual_rows():
    at = _booted("coupon")
    at = _click(at, "➕ Добавить в купон")
    at = _click(at, "Сгенерировать купоны")
    assert not at.exception

    coupons_table = next(df for df in at.dataframe if "joint_odds" in df.value.columns)
    all_legs = ", ".join(coupons_table.value["legs"])
    # the analysis-added leg (Arsenal vs Chelsea) appears in at least one coupon
    assert "Arsenal" in all_legs


def test_coupon_results_table_shows_real_club_names_not_bare_bet_ids():
    at = _booted("coupon")
    at = _click(at, "Сгенерировать купоны")  # uses only the manual table's default rows
    assert not at.exception

    coupons_table = next(df for df in at.dataframe if "joint_odds" in df.value.columns)
    all_legs = ", ".join(coupons_table.value["legs"])
    assert "#" not in all_legs
    assert "Arsenal" in all_legs and "Chelsea" in all_legs


def _click(at, label):
    return {b.label: b for b in at.button}[label].click().run(timeout=30)


def test_coupon_page_generates_positive_ev_coupons_on_click():
    at = _booted("coupon")
    at = _click(at, "Сгенерировать купоны")
    assert not at.exception

    # Both the analysis section's outcome table and the final coupons table
    # have an "EV" column -- match on "joint_odds", unique to the latter.
    coupons_table = next(df for df in at.dataframe if "joint_odds" in df.value.columns)
    assert all(ev > 0 for ev in coupons_table.value["EV"])


def test_coupon_page_monetization_metrics_match_worked_example():
    at = _booted("coupon")
    at = _click(at, "Сгенерировать купоны")

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


class _FakeContentBlock:
    def __init__(self, text):
        self.text = text


class _FakeAnthropicMessage:
    def __init__(self, text):
        self.content = [_FakeContentBlock(text)]


class _FakeAnthropicMessagesResource:
    def __init__(self, response_text):
        self._response_text = response_text

    def create(self, **kwargs):
        return _FakeAnthropicMessage(self._response_text)


class _FakeAnthropicClient:
    def __init__(self, api_key=None):
        self.messages = _FakeAnthropicMessagesResource(
            json.dumps(
                {
                    "match_id": "fixture.id",
                    "sport": None,
                    "league": "league.name",
                    "home_team": "teams.home.name",
                    "away_team": "teams.away.name",
                    "start_time": None,
                    "home_odds": None,
                    "draw_odds": None,
                    "away_odds": None,
                }
            )
        )


def test_connector_page_requires_all_three_keys_before_fetching():
    at = _booted("connector")
    at.button[0].click().run(timeout=30)
    assert not at.exception
    assert len(at.error) == 1
    assert "обязательны" in at.error[0].value


def test_connector_page_fetches_and_normalizes_with_mocked_source_and_ai(monkeypatch):
    def fake_get(url, headers=None, params=None, timeout=None):
        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "response": [
                        {
                            "fixture": {"id": 1},
                            "league": {"name": "Test League"},
                            "teams": {"home": {"name": "Sync FC"}, "away": {"name": "Merge United"}},
                        }
                    ]
                }

        return _Resp()

    monkeypatch.setattr("bukmeker.connectors.raw_source.requests.get", fake_get)
    monkeypatch.setattr("anthropic.Anthropic", _FakeAnthropicClient)

    at = _booted("connector")
    inputs = {ti.label: ti for ti in at.text_input}
    inputs["Base URL провайдера"].set_value("https://provider.example.com")
    inputs["API-ключ провайдера"].set_value("SRC_KEY")
    inputs["Anthropic API-ключ"].set_value("ANTH_KEY")
    at.run(timeout=30)

    at.button[0].click().run(timeout=30)
    assert not at.exception
    assert len(at.success) == 1
    dataframes = at.dataframe
    match_df = next(df for df in dataframes if "home_team" in df.value.columns)
    assert "Sync FC" in match_df.value["home_team"].values


def test_connector_page_redacts_api_key_from_error_message(monkeypatch):
    def fake_get(url, headers=None, params=None, timeout=None):
        raise RuntimeError(f"connection failed for {url}?apiKey=SUPERSECRET123")

    monkeypatch.setattr("bukmeker.connectors.raw_source.requests.get", fake_get)
    monkeypatch.setattr("anthropic.Anthropic", _FakeAnthropicClient)

    at = _booted("connector")
    inputs = {ti.label: ti for ti in at.text_input}
    inputs["Base URL провайдера"].set_value("https://provider.example.com")
    inputs["API-ключ провайдера"].set_value("SUPERSECRET123")
    inputs["Anthropic API-ключ"].set_value("ANTH_KEY")
    at.run(timeout=30)

    at.button[0].click().run(timeout=30)
    assert not at.exception
    assert len(at.error) == 1
    assert "SUPERSECRET123" not in at.error[0].value
    assert "***" in at.error[0].value


def test_connector_page_sync_checkbox_merges_into_registry(monkeypatch):
    provider_payload = {
        "response": [
            {
                "fixture": {"id": 1},
                "league": {"name": "Brand New Sync League"},
                "teams": {"home": {"name": "Sync FC"}, "away": {"name": "Merge United"}},
            }
        ]
    }

    def fake_get(url, headers=None, params=None, timeout=None):
        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return provider_payload

        return _Resp()

    monkeypatch.setattr("bukmeker.connectors.raw_source.requests.get", fake_get)
    monkeypatch.setattr("anthropic.Anthropic", _FakeAnthropicClient)

    at = _booted("connector")
    inputs = {ti.label: ti for ti in at.text_input}
    inputs["Base URL провайдера"].set_value("https://provider.example.com")
    inputs["API-ключ провайдера"].set_value("SRC_KEY")
    inputs["Anthropic API-ключ"].set_value("ANTH_KEY")
    at.checkbox[0].check()
    at.run(timeout=30)

    assert len(at.selectbox) == 4  # +2 for sport/country once sync is enabled

    at.button[0].click().run(timeout=30)
    assert not at.exception
    assert len(at.success) == 1
    assert any("Синхронизация: +1 лиг" in i.value for i in at.info)
