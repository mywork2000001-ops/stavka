import pytest

from bukmeker.connectors.raw_source import RawDataSource


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_raw_source_injects_key_as_header_by_default(monkeypatch):
    captured = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["params"] = params
        return _FakeResponse({"response": [{"id": 1}]})

    monkeypatch.setattr("bukmeker.connectors.raw_source.requests.get", fake_get)

    source = RawDataSource(base_url="https://api.example.com", api_key="SECRET", key_name="x-api-key")
    result = source.fetch("fixtures")

    assert result == {"response": [{"id": 1}]}
    assert captured["url"] == "https://api.example.com/fixtures"
    assert captured["headers"] == {"x-api-key": "SECRET"}
    assert "x-api-key" not in captured["params"]


def test_raw_source_injects_key_as_query_param(monkeypatch):
    captured = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        captured["params"] = params
        return _FakeResponse({"data": []})

    monkeypatch.setattr("bukmeker.connectors.raw_source.requests.get", fake_get)

    source = RawDataSource(
        base_url="https://api.example.com",
        api_key="SECRET",
        key_location="query",
        key_name="apiKey",
    )
    source.fetch("matches", params={"date": "2026-07-05"})

    assert captured["params"] == {"date": "2026-07-05", "apiKey": "SECRET"}


def test_raw_source_rejects_invalid_key_location():
    with pytest.raises(ValueError):
        RawDataSource(base_url="https://x", api_key="k", key_location="body")


def test_raw_source_strips_slashes_between_base_url_and_path(monkeypatch):
    captured = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        captured["url"] = url
        return _FakeResponse({})

    monkeypatch.setattr("bukmeker.connectors.raw_source.requests.get", fake_get)
    source = RawDataSource(base_url="https://api.example.com/", api_key="k")
    source.fetch("/fixtures")

    assert captured["url"] == "https://api.example.com/fixtures"
