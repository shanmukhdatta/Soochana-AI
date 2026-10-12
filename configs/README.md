# Soochana AI — System Configuration Repository (`configs/`)

`configs/` contains all centralized system settings, financial taxonomy contracts, model feature manifests, quantile calibration boundaries, and portfolio universe definitions.

---

## 1. Architectural Role & Information Flow

```text
                               ┌────────────────────────────────┐
                               │           `configs/`           │
                               │  System Configurations & Rules │
                               └───────────────┬────────────────┘
                                               │
         ┌────────────────────────┬────────────┴────────────┬────────────────────────┐
         ▼                        ▼                         ▼                        ▼
┌──────────────────┐    ┌──────────────────┐      ┌──────────────────┐     ┌──────────────────┐
│  Risk Engine &   │    │     Module A:    │      │     Module B:    │     │    Ingestion &   │
│   Impact Model   │    │ Tactical Index   │      │ Multi-Asset Lab  │     │ Ticker Resolvers │
├──────────────────┤    ├──────────────────┤      ├──────────────────┤     ├──────────────────┤
│• event_labels    │    │• module_a_config │      │• module_b_config │     │• ingestion_config│
│• feature_manifest│    │• supported_ticker│      │• sector_mapping  │     │• supported_ticker│
│• decile_threshold│    │• sector_mapping  │      │                  │     │• ticker_aliases  │
│• market_regimes  │    │                  │      │                  │     │                  │
└──────────────────┘    └──────────────────┘      └──────────────────┘     └──────────────────┘
```

---

## 2. Configuration Inventory & Manifests

| File | Purpose | Consumers | Contract Invariant |
| :--- | :--- | :--- | :--- |
| [`event_labels.json`](event_labels.json) | 5-class canonical event classification mapping (ID 0 to 4). | `app/engine/nlp.py`, `app/main.py` | Strictly 5 classes: Geopolitical, Macroeconomic, Credit Event, M&A, Product Launch. |
| [`impact_model_feature_manifest.json`](impact_model_feature_manifest.json) | Pre-publication feature specifications for the XGBoost impact predictor. | `app/engine/impact.py`, `app/services/features.py` | 14 numeric features + 3 categorical features; zero future lookahead features. |
| [`impact_decile_thresholds.json`](impact_decile_thresholds.json) | Frozen quantile cutoffs ($p_{10} \dots p_{90}$) mapping continuous abnormal returns to deciles $1 \dots 10$. | `app/engine/impact.py` | Strictly monotonic quantiles derived exclusively from training set. |
| [`impact_step2_market_regime_thresholds.json`](impact_step2_market_regime_thresholds.json) | Benchmark market return and volatility boundaries for regime classification. | `app/services/features.py` | SPY trailing return and 20-day volatility quantiles. |
| [`module_a_config.json`](module_a_config.json) | Tactical index rebalancing optimization limits, baseline weights, and decay half-life. | `app/module_a/config.py`, `app/module_a/strategy.py` | $\pm 2\%$ position limit, $\pm 5\%$ sector limit, $10\%$ max turnover, 10 bps friction. |
| [`module_b_config.json`](module_b_config.json) | Severity thresholds for auto-mapping equity impacts into multi-asset balance sheet shocks. | `app/module_b/event_mapper.py` | Threshold boundaries for Mild, Moderate, and Severe scenario tiers. |
| [`supported_tickers.json`](supported_tickers.json) | 15-stock institutional equity universe. | `app/services/features.py`, `app/ingestion/ticker_matcher.py` | AAPL, MSFT, NVDA, QCOM, GOOGL, CMCSA, AMZN, TSLA, BKNG, SBUX, COST, PEP, GILD, AMGN, CSX. |
| [`sector_mapping.json`](sector_mapping.json) | Industry sector classifications for reference equities. | `app/module_a/`, `app/module_b/`, `app/services/` | Standard GICS sector categorization. |
| [`ticker_aliases.json`](ticker_aliases.json) | Entity name keywords and variations used for news headline entity recognition. | `app/ingestion/ticker_matcher.py` | Maps company names, product brands, and colloquial tags to tickers. |
| [`ingestion_config.json`](ingestion_config.json) | Polling parameters, scheduler intervals, and enabled streaming providers. | `app/ingestion/config.py` | Controls polling frequency, batch caps, and provider activations. |
| [`reference/`](reference/) | Reference validation configurations used in configuration contract tests. | `tests/test_configuration.py` | Model architecture specifications for regression verification. |
