# Soochana AI — Financial Services Layer

`app/services/` provides the **core market data provider and feature engineering services** for Soochana AI. It connects live market price feeds to the machine learning pipeline, extracting pre-publication market volatility and volume indicators with mathematical parity to the offline training workflow.

---

## 1. Architectural Scope & Purpose

* **Scope:** Market price ingestion (Yahoo Finance / offline CSV archives) and pre-publication feature engineering consumed by `app.engine.pipeline.RiskEngine`.
* **Strict Anti-Lookahead Guarantee:** Financial features must reflect only market conditions known *before* an event occurred. The cutoff date is strictly the UTC calendar date prior to the news publication timestamp. Never joins to future price movements or post-announcement reaction dates.

---

## 2. Component Structure

| File | Purpose | Key Classes / Functions |
| :--- | :--- | :--- |
| [`prices.py`](prices.py) | Dual-mode market quote provider (Yahoo Finance or offline CSV) | `PriceProvider`, `PriceProvider.get_history()`, `MarketDataError` |
| [`features.py`](features.py) | Rolling volatility, volume ratio, and regime feature calculator | `FeatureBuilder`, `FeatureBuilder.build()`, `UnsupportedTicker` |

---

## 3. Market Feature Mechanics (Step-2 Parity)

[`features.py`](features.py) replicates the exact feature engineering pipeline used during the training of the XGBoost impact model:

1. **20-Day Annualized Rolling Volatility ($\sigma_{20}$):**
   Computed from daily adjusted close percentage returns:
   $$\sigma_{20} = \text{std}_{20}(\Delta P / P) \times \sqrt{252}$$
2. **20-Session Volume Ratio:**
   Identifies unusual trading liquidity leading up to the announcement:
   $$\text{Volume Ratio} = \frac{V_t}{\frac{1}{20} \sum_{k=0}^{19} V_{t-k}}$$
3. **Trailing Momentum Returns:**
   * 5-day return: $(P_t / P_{t-5}) - 1$
   * 20-day return: $(P_t / P_{t-20}) - 1$
4. **Market Regime Categorization:**
   Maps raw metrics against thresholds in [`configs/impact_step2_market_regime_thresholds.json`](../../configs/impact_step2_market_regime_thresholds.json) to flag elevated market volatility regimes.

---

## 4. Providers & Configuration

* **`yfinance` Provider (Default):** Fetches live adjusted daily quotes from Yahoo Finance.
* **`csv` Provider (Offline Reproducibility):** Reads static daily quotes from `data/daily_adjusted_prices/` when network access is restricted or deterministic holdout testing is required.
