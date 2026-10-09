"""Test MarketauxAdapter parsing, token detection, and fallback to GDELT."""
from datetime import datetime, timezone
import pytest
from app.ingestion.adapters.marketaux import MarketauxAdapter
from app.ingestion.records import utc_now


class MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json = json_data
        self.text = "mock"
        self.headers = {"Content-Type": "application/json"}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class MockSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append((url, params))
        return self.response


def test_marketaux_collect_success(monkeypatch):
    monkeypatch.setenv("MARKETAUX_API_TOKEN", "mock_test_token_12345")
    mock_payload = {
        "meta": {"found": 1, "returned": 1},
        "data": [
            {
                "uuid": "test-uuid-aapl-1",
                "title": "Apple reveals new AI chip architecture",
                "description": "Apple Inc announced breakthrough silicon efficiency.",
                "url": "https://www.reuters.com/technology/apple-chip-12345",
                "published_at": "2026-10-09T14:30:00.000000Z",
                "source": "reuters.com",
                "language": "en"
            }
        ]
    }
    session = MockSession(MockResponse(200, mock_payload))
    adapter = MarketauxAdapter({"symbols": ["AAPL"]}, timeout=10, session=session)
    items = adapter.collect(utc_now(), limit=5)

    assert len(items) == 1
    item = items[0]
    assert item.source_type == "financial_news"
    assert item.source_name == "reuters.com"
    assert item.source_id == "test-uuid-aapl-1"
    assert "Apple reveals new AI chip" in item.text
    assert item.timestamp_quality == "self_reported"
    assert item.reported_at == datetime(2026, 10, 9, 14, 30, tzinfo=timezone.utc)


def test_marketaux_fallback_when_token_missing(monkeypatch):
    monkeypatch.delenv("MARKETAUX_API_TOKEN", raising=False)
    # When token is missing, MarketauxAdapter delegates to GDELT backup
    gdelt_payload = {
        "articles": [
            {
                "url": "https://www.bloomberg.com/news/apple-earnings",
                "title": "Apple Reports Record Quarterly Revenue",
                "seendate": "20261009T150000Z",
                "language": "English"
            }
        ]
    }
    session = MockSession(MockResponse(200, gdelt_payload))
    adapter = MarketauxAdapter({}, timeout=10, session=session)
    items = adapter.collect(utc_now(), limit=3)
    assert len(items) == 1
    assert items[0].source_type == "financial_news"
    assert "Apple Reports Record Quarterly Revenue" in items[0].text
    assert items[0].timestamp_quality == "gdelt_discovery"
