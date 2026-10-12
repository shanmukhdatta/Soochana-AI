# Soochana AI — Data Layer & Persistence Assets (`data/`)

`data/` manages the local persistence tier, holdout financial time series, seed multi-asset portfolios, scenario definitions, and ingestion databases.

---

## 1. Directory Structure & Architecture

```text
data/
├── ingestion/                 # Seed SQLite database for news & social signals
│   └── riskpulse.db           # Pre-collected items with full source metadata
├── module_a/                  # Historical data inputs & backtest outputs for Module A
│   ├── holdout_prices.csv     # 357 sessions of daily OHLCV equity price grids
│   ├── holdout_signals.csv    # Timestamped historical news signals with NLP labels
│   ├── real_example_response.json # Sample production payload for testing
│   └── results/               # Generated backtest evaluation matrices
├── module_b/                  # Balance sheet portfolio & scenario libraries for Module B
│   ├── synthetic_portfolio.json # 30 institutional positions across Loans, Bonds, Swaps, Options
│   ├── scenario_library.json  # Predefined macroeconomic, geopolitical & credit shock templates
│   ├── live_verification_results.json # Verified output snapshot
│   └── results/               # Output JSON files from stress and reverse stress runs
├── impact_step2_report.json   # Training data metrics & feature distributions for XGBoost
└── impact_step2_test_features.parquet # Out-of-time evaluation feature split
```

---

## 2. Sub-Directory Roles

* [**`data/ingestion/`**](ingestion/README.md): SQLite database storing raw news headlines, social posts, entity tags, and model predictions.
* [**`data/module_a/`**](module_a/README.md): 357-session historical walk-forward price and signal matrices used to simulate index rebalancing without lookahead bias.
* [**`data/module_b/`**](module_b/README.md): 30-position synthetic institutional balance sheet, scenario definitions, and verification benchmarks.
