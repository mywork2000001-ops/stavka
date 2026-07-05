"""Generic HTTP data source for arbitrary sports-data providers.

Makes no assumption about response shape or field names — any provider works as
long as it's a REST/JSON API authenticated by a single API key, because shape
normalisation is delegated to the AI field mapper (see `ai_mapper.py`).
"""

from __future__ import annotations

import requests


class RawDataSource:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        key_location: str = "header",
        key_name: str = "x-api-key",
        timeout: float = 10.0,
    ):
        if key_location not in ("header", "query"):
            raise ValueError("key_location must be 'header' or 'query'")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.key_location = key_location
        self.key_name = key_name
        self.timeout = timeout

    def fetch(self, path: str, params: dict | None = None):
        url = f"{self.base_url}/{path.lstrip('/')}"
        params = dict(params or {})
        headers = {}
        if self.key_location == "header":
            headers[self.key_name] = self.api_key
        else:
            params[self.key_name] = self.api_key

        response = requests.get(url, headers=headers, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
