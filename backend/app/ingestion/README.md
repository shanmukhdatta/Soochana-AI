# Soochana AI — Signal Ingestion & Persistence Pipeline (`app/ingestion/`)

`app/ingestion/` implements the decoupled, multi-source financial signal collection and persistence layer for Soochana AI.

---

## 1. System Architecture Overview

```text
  ┌─────────────────────────────────────────────────────────────┐
  │                   EXTERNAL STREAM PROVIDERS                 │
  │                                                             │
  │   Marketaux Financial News API       Bluesky (AT Protocol)  │
  │      (Curated News Articles)          (Public Social Feeds) │
  └──────────────┬──────────────────────────────┬───────────────┘
                 │                              │
                 ▼                              ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ ADAPTER LAYER (`app/ingestion/adapters/`)                   │
  │ • MarketauxAdapter: Entity-tagged news, published timestamps│
  │ • BlueskyAdapter: Public firehose search via AT Protocol    │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ TICKER MATCHING & DEDUPLICATION (`ticker_matcher.py`)       │
  │ • Multi-pass ticker resolution against configs/tickers.json │
  │ • Exact SHA-256 hash deduplication on normalized text       │
  │ • URL & canonical title deduplication                       │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ SQLITE PERSISTENCE REPOSITORY (`repository.py`)             │
  │ • Local storage: `data/ingestion/riskpulse.db`              │
  │ • Schema: IngestionRecord with status, source metadata,     │
  │   raw text, mapped tickers, and processed model signals     │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  Dispatched to Unified Risk Engine (`app/engine/`) on Demand
```

---

## 2. Component Structure

| File | Purpose |
| :--- | :--- |
| [`adapters/`](adapters/) | Source-specific API clients for Marketaux, Bluesky, and GDELT |
| [`ticker_matcher.py`](ticker_matcher.py) | Resolves company names and stock symbols from raw headlines |
| [`records.py`](records.py) | Internal Pydantic schemas for normalized ingestion records |
| [`repository.py`](repository.py) | Thread-safe SQLite database operations and signal caching |
| [`scheduler.py`](scheduler.py) | Background ingestion job scheduler |
| [`router.py`](router.py) | FastAPI endpoints for triggering collection and inspecting status |
| [`service.py`](service.py) | Coordination between network adapters, repository, and risk engine |

---

## 3. Security & Operational Guardrails

* **Admin Authentication:** Mutation endpoints (`POST /ingestion/poll`) require `X-Admin-Token` matching `INGESTION_ADMIN_TOKEN` (minimum 24 characters) in `.env`.
* **Zero Leakage:** Ingestion timestamps strictly preserve UTC publication times to prevent lookahead bias in downstream quantitative models.
* **Offline Resilience:** The application operates seamlessly from bundled seed signals in SQLite without requiring external network connectivity or paid API credentials.
