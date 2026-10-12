# Soochana AI — Backend Service (`backend/`)

The `backend/` service provides high-performance FastAPI microservices, quantitative financial simulation laboratories, and machine learning inference coordination.

---

## 1. Directory Structure

```text
backend/
├── app/                        # Core application package
│   ├── engine/                 # Unified 3-model risk engine (NLP + Features + Impact)
│   ├── module_a/               # Tactical index rebalancing (15 US equities, SLSQP)
│   ├── module_b/               # Multi-asset balance sheet stress testing (30 positions)
│   ├── ingestion/              # Stream collection adapters (Marketaux, Bluesky) & SQLite
│   ├── assistant/              # Bujji conversational risk assistant (Groq API)
│   ├── services/               # Zero-lookahead market features & price providers
│   ├── config.py               # Application settings & environment schemas
│   ├── main.py                 # FastAPI application factory & lifespan event manager
│   └── schemas.py              # Top-level Pydantic request & response models
├── tests/                      # Automated PyTest test suite (120 test cases)
│   ├── test_api.py             # Route declarations and error contracts
│   ├── test_pipeline.py        # 3-model coordination and fallback behaviors
│   ├── test_module_a_*.py      # SLSQP constraints, backtest, and signals
│   ├── test_module_b_*.py      # Valuation engines, scenarios, and reverse stress
│   └── test_ingestion_*.py     # Providers, normalization, and deduplication
├── Dockerfile                  # Production container definition
├── requirements.txt            # Python dependencies
└── pytest.ini                  # Local test runner configuration
```

---

## 2. Running Locally

From the repository root:
```bash
# Start backend server with hot-reload
uvicorn backend.app.main:app --reload --port 8000
```

Or from inside `backend/`:
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

---

## 3. Testing & Verification

```bash
# From repository root
pytest -q

# Or from inside backend/
cd backend
pytest -q
```
