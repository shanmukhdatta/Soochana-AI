# External Stream Adapters (`backend/app/ingestion/adapters/`)

This package implements modular API client adapters connecting external financial news and social media feeds into the standardized Soochana AI ingestion schema.

---

## 1. Adapter Architecture

```text
  ┌─────────────────────────────────────────────────────────────┐
  │                    EXTERNAL DATA PROVIDERS                  │
  │                                                             │
  │     Marketaux API                 Bluesky (AT Protocol)     │
  │  (Curated Financial News)        (Public Social Firehose)   │
  └─────────────┬───────────────────────────────┬───────────────┘
                │                               │
                ▼                               ▼
  ┌───────────────────────────┐   ┌───────────────────────────┐
  │   `MarketauxAdapter`      │   │     `BlueskyAdapter`      │
  │   (`marketaux.py`)        │   │     (`bluesky.py`)        │
  │ • HTTP client via httpx   │   │ • AT Protocol public API  │
  │ • Entity & ticker mapping │   │ • Query-based term filter │
  │ • ISO 8601 UTC timestamp  │   │ • Social post extraction  │
  └─────────────┬─────────────┘   └─────────────┬─────────────┘
                │                               │
                └───────────────┬───────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ STANDARD INGESTION RECORD CONTRACT (`RawSignalItem`)        │
  │ • text: Cleaned headline or post text                       │
  │ • source: "financial_news" | "social_media"                 │
  │ • published_at: UTC timestamp without timezone ambiguity    │
  │ • external_id: Deterministic source hash                    │
  │ • source_url: Canonical link for auditability               │
  └─────────────────────────────────────────────────────────────┘
```

---

## 2. Adapter Inventory

* [`marketaux.py`](marketaux.py): Connects to the Marketaux Financial News API, extracting entity-tagged financial news, ticker symbols, and publication timestamps.
* [`bluesky.py`](bluesky.py): Connects to Bluesky public endpoints using the AT Protocol, extracting public investor commentary and discussion related to tracked equities.
* [`gdelt.py`](gdelt.py): Optional adapter for GDELT Global Database of Events, Language, and Tone (disabled by default in standard ingestion configuration).
