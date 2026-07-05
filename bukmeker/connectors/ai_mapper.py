"""AI-assisted field mapping: given one sample record from a never-before-seen
sports-data provider, asks the Anthropic API to infer where each canonical field
lives in that provider's JSON shape. This is what lets the platform onboard
"any API key" without hand-writing a parser per provider.
"""

from __future__ import annotations

import json
import re

from .schema import CANONICAL_FIELDS, FieldMapping

_SYSTEM_PROMPT = (
    "You are a data-mapping assistant for a sports-betting analytics platform. "
    "You will be given one example JSON record from a sports-data API whose "
    "schema you have never seen before. Map each of the following canonical "
    "fields to a dotted path within that record: " + ", ".join(CANONICAL_FIELDS) + ". "
    "Use dot notation for nested objects and integer indices for list access "
    "(e.g. 'teams.home.name', 'odds.0.price'). If a field genuinely cannot be "
    "found in the record, map it to null. Respond with ONLY a single JSON "
    "object mapping canonical field name to dotted path string or null — no "
    "prose, no markdown code fences."
)


class ClaudeFieldMapper:
    """Real Anthropic API integration. Pass `client` to inject a fake/mock for
    testing without making network calls or spending real API credits."""

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-5", client=None):
        if client is not None:
            self._client = client
        else:
            from anthropic import Anthropic  # optional dependency, imported lazily

            self._client = Anthropic(api_key=api_key)
        self.model = model

    def infer_mapping(self, sample_record: dict) -> FieldMapping:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(sample_record)}],
        )
        text = message.content[0].text
        return FieldMapping(paths=_extract_json_object(text))


def _extract_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find a JSON object in the model response: {text!r}")
    return json.loads(match.group(0))
