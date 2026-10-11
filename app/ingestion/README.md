# RiskPulse AI — Signal Ingestion & Persistence Pipeline

`app/ingestion/` implements the **multi-source financial data collection and persistence pipeline** of RiskPulse AI. It ingests public financial news and social media commentary, normalizes timestamps and URLs, matches corporate tickers, deduplicates syndicated articles, and persists records into SQLite for downstream analysis.

---

## 1. Architectural Scope & Design Principles

* **Decoupled Architecture:** Ingestion is strictly decoupled from machine learning model inference. Scraping and polling do **not** load PyTorch or Hugging Face weights, preventing resource contention and memory leaks.
* **Traceable Provenance:** Every persisted signal retains its original source provider, publication timestamp, external URL, and matched ticker symbol.
* **Safety & Security:** Admin endpoints for manual polling are protected by a minimum 24-character token (`INGESTION_ADMIN_TOKEN`).

---

## 2. Component Structure

| File / Folder | Purpose | Key Classes / Functions |
| :--- | :--- | :--- |
| [`service.py`](service.py) | Ingestion coordinator orchestrating adapters, filtering, and storage | `IngestionService`, `collect_once()` |
| [`repository.py`](repository.py) | SQLite database layer managing signals, dedup hashes, and audit logs | `Repository`, `Repository.insert()`, `Repository.list_signals()` |
| [`ticker_matcher.py`](ticker_matcher.py) | Multi-pass regex & alias matcher resolving company mentions to tickers | `TickerMatcher`, `TickerMatcher.match()` |
| [`records.py`](records.py) | Record normalization schema and UTC timestamp validation | `NormalizedItem`, `normalize()`, `safe_url()` |
| [`scheduler.py`](scheduler.py) | Lightweight background task runner for periodic collection polls | `IngestionScheduler` |
| [`config.py`](config.py) | Configuration reader for sources, rate limits, and aliases | `IngestionConfig` |
| [`router.py`](router.py) | FastAPI endpoints for signal queries and admin triggers | `/signals`, `/signals/{id}`, `/ingestion/poll`, `/ingestion/health` |
| [`adapters/marketaux.py`](adapters/marketaux.py) | API adapter for Marketaux financial news feed | `MarketauxAdapter` |
| [`adapters/bluesky.py`](adapters/bluesky.py) | API adapter for Bluesky AT Protocol public financial posts | `BlueskyAdapter` |
| [`adapters/gdelt.py`](adapters/gdelt.py) | Optional adapter for GDELT Global Knowledge Graph events | `GDELTAdapter` |

---

## 3. Data Sources & Ingestion Adapters

1. **Marketaux (`adapters/marketaux.py`):**
   * Collects curated financial headlines and articles.
   * Requires `MARKETAUX_API_TOKEN` in `.env` for live external fetching.
2. **Bluesky AT Protocol (`adapters/bluesky.py`):**
   * Searches public social posts for financial tickers and cashtags (e.g., `$AAPL`, `$NVDA`).
   * Operates via public AT Protocol endpoints without requiring authenticated login for read access.
3. **GDELT (`adapters/gdelt.py`):**
   * Implements GDELT event query integration. Disabled by default in the primary configuration due to high noise-to-signal ratio and automated phrasing.

---

## 4. Normalization & Deduplication Workflow

```text
External API Response
         │
         ▼
┌──────────────────────────────────────────┐
│      1. Normalization (records.py)       │
│  - UTC timestamp standardizing           │
│  - Language filter (English only)        │
│  - Max age check (≤ 48 hours)            │
│  - Safe URL protocol validation          │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│    2. Ticker Resolution (matcher.py)     │
│  - Cashtags ($MSFT)                      │
│  - Word-boundary uppercase tickers       │
│  - Configured company name aliases       │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│    3. Deduplication & Store (repo.py)    │
│  - SHA-256 hash deduplication            │
│  - Cross-source syndicate suppression    │
│  - SQLite write: riskpulse.db            │
└──────────────────────────────────────────┘
```

---

## 5. Persistence Store & Pre-Bundled Data

* **Database Engine:** SQLite stored at [`data/ingestion/riskpulse.db`](../../data/ingestion/riskpulse.db).
* **Seed Database:** The repository bundles a pre-collected snapshot of **29 verified signals** (2 news + 27 social items) covering key technology, finance, and industrial issuers. This guarantees the dashboard and index rebalancing operate out-of-the-box without requiring active live external API credentials.

---

## 6. API Route Contracts

* `GET /signals`: Public endpoint to query saved signals with pagination (`limit`, `offset`), source filtering (`source_type`), and ticker filtering.
* `GET /signals/{id}`: Retrieves complete metadata and inference outputs for a single signal.
* `GET /ingestion/health`: Returns collector status, configured database path, record counts, and last collection timestamps.
* `POST /ingestion/poll`: Manually triggers a poll cycle across enabled sources (requires `Authorization: Bearer <INGESTION_ADMIN_TOKEN>`).
