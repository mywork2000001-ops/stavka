"""Integration coverage for the `bukmeker connector` CLI wiring, including
the `--sync` flag added to merge live data into the entity registry. Both the
HTTP call and the Anthropic client construction are monkeypatched so this
test makes no real network/API calls."""

import json

from bukmeker import cli


class _FakeContentBlock:
    def __init__(self, text):
        self.text = text


class _FakeMessage:
    def __init__(self, text):
        self.content = [_FakeContentBlock(text)]


class _FakeMessagesResource:
    def __init__(self, response_text):
        self._response_text = response_text

    def create(self, **kwargs):
        return _FakeMessage(self._response_text)


class _FakeAnthropicClient:
    def __init__(self, api_key=None):
        self.messages = _FakeMessagesResource(
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


class _FakeHttpResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_run_demo_executes_end_to_end_without_error(capsys):
    # This is the primary user-facing command (`bukmeker demo`) -- previously
    # it was only ever exercised by manually running it in a terminal, so a
    # regression here would not have been caught by `pytest tests/ -q` in CI.
    exit_code = cli.main(["demo"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "VALUE BET" in out
    assert "МОНЕТИЗАЦИЯ КУПОНА" in out
    assert "ГОТОВО" in out


def test_run_connector_without_keys_prints_usage_and_returns_1(capsys):
    exit_code = cli.main(["connector"])
    assert exit_code == 1
    assert "Usage: bukmeker connector" in capsys.readouterr().out


def test_run_connector_with_sync_merges_new_league_into_registry(monkeypatch, capsys):
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
        return _FakeHttpResponse(provider_payload)

    monkeypatch.setattr("bukmeker.connectors.raw_source.requests.get", fake_get)
    monkeypatch.setattr("anthropic.Anthropic", _FakeAnthropicClient)

    exit_code = cli.main(
        [
            "connector",
            "--source-url", "https://provider.example.com",
            "--source-key", "SRC_KEY",
            "--anthropic-key", "ANTH_KEY",
            "--path", "fixtures",
            "--sync",
            "--sync-sport", "Football",
            "--sync-country", "USA",
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Sync FC" in out
    assert "Sync into entity registry: +1 league(s), +2 competitor(s)" in out


def test_run_connector_sync_rejects_unknown_sport(monkeypatch, capsys):
    def fake_get(url, headers=None, params=None, timeout=None):
        return _FakeHttpResponse({"response": [{"id": 1, "home": "A", "away": "B"}]})

    monkeypatch.setattr("bukmeker.connectors.raw_source.requests.get", fake_get)
    monkeypatch.setattr("anthropic.Anthropic", _FakeAnthropicClient)

    exit_code = cli.main(
        [
            "connector",
            "--source-url", "https://provider.example.com",
            "--source-key", "SRC_KEY",
            "--anthropic-key", "ANTH_KEY",
            "--sync",
            "--sync-sport", "Curling",
        ]
    )

    assert exit_code == 1
    assert "Unknown --sync-sport" in capsys.readouterr().out


def test_run_dashboard_invokes_streamlit_via_current_python_interpreter(monkeypatch):
    import sys

    captured_cmd = {}

    def fake_call(cmd):
        captured_cmd["cmd"] = cmd
        return 0

    monkeypatch.setattr("subprocess.call", fake_call)

    exit_code = cli.main(["dashboard", "--port", "8600"])

    assert exit_code == 0
    cmd = captured_cmd["cmd"]
    assert cmd[0] == sys.executable
    assert cmd[1:4] == ["-m", "streamlit", "run"]
    assert cmd[-2:] == ["--server.port", "8600"]


def test_run_dashboard_reports_missing_streamlit_without_crashing(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    exit_code = cli.main(["dashboard"])

    assert exit_code == 1


def test_run_connector_sync_rejects_unknown_country(monkeypatch, capsys):
    def fake_get(url, headers=None, params=None, timeout=None):
        return _FakeHttpResponse({"response": [{"id": 1, "home": "A", "away": "B"}]})

    monkeypatch.setattr("bukmeker.connectors.raw_source.requests.get", fake_get)
    monkeypatch.setattr("anthropic.Anthropic", _FakeAnthropicClient)

    exit_code = cli.main(
        [
            "connector",
            "--source-url", "https://provider.example.com",
            "--source-key", "SRC_KEY",
            "--anthropic-key", "ANTH_KEY",
            "--sync",
            "--sync-country", "ZZZ",
        ]
    )

    assert exit_code == 1
    assert "Unknown --sync-country" in capsys.readouterr().out
