# Soochana AI — Backend Application Architecture (`app/`)

`app/` contains the **FastAPI backend services, machine learning inference pipeline, quantitative financial modules, and contextual assistant** for Soochana AI.

---

## Architecture Overview

```text
                               ┌────────────────────────┐
                               │  app/main.py (FastAPI) │
                               └───────────┬────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
┌─────────────────┐               ┌─────────────────┐               ┌─────────────────┐
│ app/ingestion/  │               │  app/engine/    │               │ app/assistant/  │
│ - Marketaux     │ ────────────> │ - FinBERT Event │ <──────────── │ - Bujji Chat    │
│ - Bluesky       │               │ - FinBERT Senti │               │ - Groq API      │
│ - SQLite Store  │               │ - XGBoost Impact│               │ - Mode Context  │
└─────────────────┘               └────────┬────────┘               └─────────────────┘
                                           │
                                           │ Structured Signals
                                           ▼
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
┌─────────────────────────────────┐               ┌───────────────────────────────────┐
│        app/module_a/            │               │          app/module_b/            │
│  Tactical Index Rebalancing     │               │    Multi-Asset Stress Testing     │
│  - 15 US Equities (SLSQP)       │               │    - 30 Synthetic Positions       │
│  - ±2% Position, ±5% Sector     │               │    - Instantaneous MTM (Bonds/Deriv)│
│  - 10% Max Turnover             │               │    - 12-Month ECL (Loans)         │
│  - 357-Session Backtesting      │               │    - What-If & Reverse Stress     │
└─────────────────────────────────┘               └───────────────────────────────────┘
```

---

## Sub-Module Documentation Index

Each subfolder contains a dedicated, rigorous technical `README.md` documenting its mathematical foundations, file structure, and API contracts:

1. [**`app/module_a/` — Tactical Index Rebalancing Lab**](module_a/README.md)
   * 15-stock reference equity index with quadratic distance minimization under convex constraints ($\pm 2\%$ position deviation, $\pm 5\%$ sector deviation, $\le 10\%$ one-way turnover, 10 bps costs).
   * Deterministic signal deduplication, exponential half-life decay ($\tau = 3.0$ days), and 357-session historical walk-forward backtest.

2. [**`app/module_b/` — Multi-Asset Stress Lab**](module_b/README.md)
   * 30-position synthetic institutional balance sheet (12 loans, 10 bonds, 4 swaps, 4 options).
   * Strict auditable separation between instantaneous Trading Book Mark-to-Market (MTM) and Banking Book 12-Month Expected Credit Loss (ECL).
   * Interactive What-If position scaling ($0.0\times$ to $2.0\times$) and automated Reverse Stress testing.

3. [**`app/engine/` — Unified Risk Engine**](engine/README.md)
   * Three cooperating machine learning models:
     - 5-Class FinBERT Event Classifier (Geopolitical, Macroeconomic, Credit Event, M&A, Product Launch).
     - Continuous FinBERT Sentiment Regressor ($[-1.0, +1.0]$).
     - XGBoost Market Impact Engine (Abnormal reaction severity deciles 1–10).

4. [**`app/ingestion/` — Signal Ingestion & Persistence Pipeline**](ingestion/README.md)
   * Decoupled collection from Marketaux financial news and Bluesky AT Protocol public social commentary.
   * Multi-pass ticker matching, cross-source deduplication, and local SQLite persistence (`data/ingestion/riskpulse.db`).

5. [**`app/assistant/` — Bujji Financial AI Assistant**](assistant/README.md)
   * Context-aware conversational risk assistant powered by Groq API (`qwen/qwen3.8-27b`).
   * Tailored explanations for both Quick View (executive) and Analyst View (quantitative audit), with strict financial accuracy guardrails.

6. [**`app/services/` — Financial Services Layer**](services/README.md)
   * Dual-mode price quote provider (`yfinance` or offline CSV).
   * Step-2 rolling market feature engineering ($\sigma_{20}$, volume ratio, trailing momentum) with strict anti-lookahead guarantees.

---

## Root App Files

* [`main.py`](main.py): FastAPI application factory, lifespan startup model initialization, and global router mounts.
* [`config.py`](config.py): Application settings schema loaded from environment variables (`.env`).
* [`schemas.py`](schemas.py): Top-level Pydantic request and response schemas (`AnalyzeRequest`, `AnalyzeResponse`, `HealthResponse`).
