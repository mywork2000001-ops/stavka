"""Ties a generic data source to the AI field mapper: fetch raw data from any
provider (by API key), infer its shape once via the LLM, and normalize every
record in the response into the platform's canonical schema."""

from __future__ import annotations

from .ai_mapper import ClaudeFieldMapper
from .raw_source import RawDataSource
from .schema import CanonicalMatch, apply_mapping, find_record_list


class AIDataConnector:
    def __init__(self, source: RawDataSource, mapper: ClaudeFieldMapper):
        self.source = source
        self.mapper = mapper

    def fetch_and_normalize(self, path: str, params: dict | None = None) -> list[CanonicalMatch]:
        raw = self.source.fetch(path, params=params)
        records = find_record_list(raw)
        if not records:
            return []
        mapping = self.mapper.infer_mapping(records[0])
        return [apply_mapping(record, mapping) for record in records]
