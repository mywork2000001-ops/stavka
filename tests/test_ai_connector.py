import json

from bukmeker.connectors.ai_connector import AIDataConnector
from bukmeker.connectors.ai_mapper import ClaudeFieldMapper
from bukmeker.connectors.raw_source import RawDataSource


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
    def __init__(self, response_text):
        self.messages = _FakeMessagesResource(response_text)


class _FakeHttpResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_ai_data_connector_fetches_and_normalizes_unseen_provider_shape(monkeypatch):
    provider_payload = {
        "response": [
            {
                "fixture": {"id": 101, "date": "2026-07-06T18:00:00Z"},
                "teams": {"home": {"name": "Arsenal"}, "away": {"name": "Chelsea"}},
                "odds": {"home": "1.75", "draw": "3.60", "away": "4.75"},
            },
            {
                "fixture": {"id": 102, "date": "2026-07-06T20:00:00Z"},
                "teams": {"home": {"name": "Real Madrid"}, "away": {"name": "Barcelona"}},
                "odds": {"home": "2.10", "draw": "3.40", "away": "3.20"},
            },
        ]
    }

    def fake_get(url, headers=None, params=None, timeout=None):
        return _FakeHttpResponse(provider_payload)

    monkeypatch.setattr("bukmeker.connectors.raw_source.requests.get", fake_get)

    inferred_mapping = json.dumps(
        {
            "match_id": "fixture.id",
            "sport": None,
            "league": None,
            "home_team": "teams.home.name",
            "away_team": "teams.away.name",
            "start_time": "fixture.date",
            "home_odds": "odds.home",
            "draw_odds": "odds.draw",
            "away_odds": "odds.away",
        }
    )
    source = RawDataSource(base_url="https://provider.example.com", api_key="SRC_KEY")
    mapper = ClaudeFieldMapper(client=_FakeAnthropicClient(inferred_mapping))
    connector = AIDataConnector(source=source, mapper=mapper)

    matches = connector.fetch_and_normalize("fixtures")

    assert len(matches) == 2
    assert matches[0].home_team == "Arsenal"
    assert matches[0].away_team == "Chelsea"
    assert matches[0].home_odds == 1.75
    assert matches[1].home_team == "Real Madrid"
    assert matches[1].away_odds == 3.20


def test_ai_data_connector_returns_empty_list_when_no_records_found(monkeypatch):
    def fake_get(url, headers=None, params=None, timeout=None):
        return _FakeHttpResponse({"status": "ok"})

    monkeypatch.setattr("bukmeker.connectors.raw_source.requests.get", fake_get)

    source = RawDataSource(base_url="https://provider.example.com", api_key="SRC_KEY")
    mapper = ClaudeFieldMapper(client=_FakeAnthropicClient("{}"))
    connector = AIDataConnector(source=source, mapper=mapper)

    assert connector.fetch_and_normalize("fixtures") == []
