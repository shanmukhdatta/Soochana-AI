# Backend Test Suite Architecture (`backend/tests/`)

`backend/tests/` provides automated testing coverage (121 test cases) validating every layer of the backend application: API contracts, ML inference coordination, SLSQP optimization constraints, analytical pricing models, stream ingestion, and telemetry.

---

## 1. Test Architecture Matrix

```text
====================================================================================================
                                BACKEND AUTOMATED TEST HIERARCHY
====================================================================================================

  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 1. API & SYSTEM CONTRACTS                                                              │
  │ • test_api.py: Route definitions, batch caps, error boundaries, liveness & metrics     │
  │ • test_configuration.py: Immutable schemas, label orders, and quantiles                │
  │ • test_assistant.py: Bujji prompt assembly, mode adaptation, and fallback handlers    │
  └────────────────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 2. MACHINE LEARNING RISK ENGINE                                                        │
  │ • test_pipeline.py: 3-model coordination, tokenizer isolation, fallback states         │
  │ • test_impact.py: XGBoost feature manifest alignment, decile binning monotonicity      │
  │ • test_prices_and_features.py: Strict anti-lookahead rules, rolling σ_20 calculations  │
  └────────────────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 3. QUANTITATIVE LABORATORIES                                                           │
  │ • test_module_a_api.py: Rebalancing API endpoints and payload serialization          │
  │ • test_module_a_constraints.py: SLSQP position, sector, and turnover bound adherence   │
  │ • test_module_a_signals.py: Exponential half-life decay (τ = 3.0 days) & conviction    │
  │ • test_module_a_backtest.py: 357-session walk-forward backtest verification            │
  │ • test_module_b_api.py: Multi-asset stress API endpoints and scenario execution       │
  │ • test_module_b_pricing.py: Loans (12m ECL), Bonds (DV01/CS01), Swaps, Options (BS)   │
  │ • test_module_b_portfolio.py: 30-position synthetic institutional balance sheet        │
  │ • test_module_b_reverse.py: Deterministic grid search for risk budget breach           │
  │ • test_module_b_all_scenarios.py: Complete scenario library validation                 │
  │ • test_module_b_override_coverage.py: Issuer/sector targeted shock override validation  │
  └────────────────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 4. STREAM INGESTION & DATA PERSISTENCE                                                 │
  │ • test_ingestion_api.py: Admin token authentication and polling trigger routes         │
  │ • test_ingestion_pipeline.py: End-to-end collection, ticker mapping, and persistence   │
  │ • test_ingestion_repository.py: Thread-safe SQLite record insertion and deduplication  │
  │ • test_ingestion_sources.py: Marketaux and Bluesky adapter schema normalization        │
  │ • test_ingestion_tickers.py: Multi-pass entity resolution and alias matching           │
  │ • test_ingestion_scheduler.py: Background polling timer daemon                         │
  │ • test_marketaux.py: Marketaux API client request handling and error policies          │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Test Execution Commands

```bash
# Execute complete backend test suite from repository root
pytest -q

# Run with verbose output and coverage
pytest -v

# Run tests from inside backend/
cd backend
pytest -q
```
