import json

import pytest

from bukmeker.connectors.ai_mapper import ClaudeFieldMapper


class _FakeContentBlock:
    def __init__(self, text):
        self.text = text


class _FakeMessage:
    def __init__(self, text):
        self.content = [_FakeContentBlock(text)] if text is not None else []


class _FakeMessagesResource:
    def __init__(self, response_text):
        self._response_text = response_text
        self.last_call = None

    def create(self, **kwargs):
        self.last_call = kwargs
        return _FakeMessage(self._response_text)


class _FakeAnthropicClient:
    def __init__(self, response_text):
        self.messages = _FakeMessagesResource(response_text)


def test_infer_mapping_parses_clean_json_response():
    mapping_json = json.dumps(
        {
            "match_id": "fixture.id",
            "sport": None,
            "league": "league.name",
            "home_team": "teams.home.name",
            "away_team": "teams.away.name",
            "start_time": "fixture.date",
            "home_odds": "odds.home",
            "draw_odds": "odds.draw",
            "away_odds": "odds.away",
        }
    )
    fake_client = _FakeAnthropicClient(mapping_json)
    mapper = ClaudeFieldMapper(client=fake_client)

    mapping = mapper.infer_mapping({"fixture": {"id": 1}})

    assert mapping.get("home_team") == "teams.home.name"
    assert mapping.get("sport") is None


def test_infer_mapping_parses_response_wrapped_in_markdown_fences():
    mapping_json = "```json\n" + json.dumps({"match_id": "id"}) + "\n```"
    fake_client = _FakeAnthropicClient(mapping_json)
    mapper = ClaudeFieldMapper(client=fake_client)

    mapping = mapper.infer_mapping({"id": 1})
    assert mapping.get("match_id") == "id"


def test_infer_mapping_sends_sample_record_and_system_prompt():
    fake_client = _FakeAnthropicClient(json.dumps({"match_id": "id"}))
    mapper = ClaudeFieldMapper(client=fake_client, model="claude-sonnet-5")

    mapper.infer_mapping({"id": 42})

    call = fake_client.messages.last_call
    assert call["model"] == "claude-sonnet-5"
    assert "match_id" in call["system"]
    assert json.loads(call["messages"][0]["content"]) == {"id": 42}


def test_infer_mapping_raises_on_unparsable_response():
    fake_client = _FakeAnthropicClient("I cannot help with that.")
    mapper = ClaudeFieldMapper(client=fake_client)

    with pytest.raises(ValueError):
        mapper.infer_mapping({"id": 1})


def test_infer_mapping_raises_clear_error_on_empty_content_response():
    # A real (if rare) Anthropic API response shape: no content blocks at all.
    fake_client = _FakeAnthropicClient(None)
    mapper = ClaudeFieldMapper(client=fake_client)

    with pytest.raises(ValueError, match="empty response"):
        mapper.infer_mapping({"id": 1})
