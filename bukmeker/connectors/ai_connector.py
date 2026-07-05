"""Ties a generic data source to the AI field mapper: fetch raw data from any
provider (by API key), infer its shape once via the LLM, and normalize every
record in the response into the platform's canonical schema."""

from __future__ import annotations

from typing import Callable

from .ai_mapper import ClaudeFieldMapper
from .raw_source import RawDataSource
from .schema import CanonicalMatch, FieldMapping, apply_mapping, find_record_list


class AIDataConnector:
    """`apply_fn` defaults to the future-fixtures/odds schema (`apply_mapping`)
    -- pass `connectors.historical.apply_historical_mapping` (with a mapper
    built for `HISTORICAL_FIELDS`) to normalize past results instead, without
    duplicating the fetch/mapping-inference plumbing."""

    def __init__(
        self,
        source: RawDataSource,
        mapper: ClaudeFieldMapper,
        apply_fn: Callable[[dict, FieldMapping], object] = apply_mapping,
    ):
        self.source = source
        self.mapper = mapper
        self._apply_fn = apply_fn

    def fetch_and_normalize(self, path: str, params: dict | None = None) -> list[CanonicalMatch]:
        raw = self.source.fetch(path, params=params)
        records = find_record_list(raw)
        if not records:
            return []
        mapping = self.mapper.infer_mapping(records[0])
        return [self._apply_fn(record, mapping) for record in records]
