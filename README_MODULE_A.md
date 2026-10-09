# RiskPulse AI — Module A (Source-only integration addon)

**Purpose:** Explainable, event-informed, constrained **simulation** of a 15-stock tactical index. Audited source release v1.1 (preserves Module A v1 API contract). No real trading, broker credentials, or automatic orders.

## Install into your EXISTING Antigravity-verified `c2c/` project

1. **Keep your current project and trained model folders intact.** Back up your repository or commit before integration.
2. Extract the contents of `riskpulse_module_a_addon_v1_1_AUDITED.zip` **directly into your existing `c2c/` directory**, merging `app/`, `configs/`, `scripts/`, `tests/`. It contains **only new Module A files**, not `app/main.py`, not `.env`, not `requirements.txt`, not model weights, and not your corrected Impact files.
3. With `c2c/` as the working directory and your existing Python environment activated:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements_module_a.txt
.\.venv\Scripts\python.exe install_module_a.py
.\.venv\Scripts\python.exe -m pytest -q tests -k module_a
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

`install_module_a.py` appends a **single router include** to your existing `app/main.py`, keeping `app/main.py.before_module_a.bak` as backup. It is idempotent. Your Antigravity fixes remain untouched. Run the installer before starting Uvicorn. If the server already uses port 8000, stop the old instance first.

4. Open `http://127.0.0.1:8000/docs`, find **Module A: Tactical Rebalancing**.

## Try Module A without running your FinBERT models

`POST /module-a/rebalance` accepts a **precomputed risk-engine signal**. Sample request:

```json
{
  "as_of": "2024-05-16T23:00:00Z",
  "strategy": "full_event_aware",
  "signals": [
    {
      "ticker": "NVDA", "published_at": "2024-05-15T12:00:00Z",
      "source": "financial_news", "headline": "Nvidia reports weaker demand and higher costs",
      "sentiment_score": -0.65, "impact_score": 8,
      "event_class": "Macroeconomic", "event_confidence": 0.90
    }
  ]
}
```

This is an **example input**, not an authentic Risk Engine prediction. Each response contains the previous weights, constrained proposed weights, per-stock explanations, turnover, a cost estimate and filtering warnings. Provide `current_weights` for a non-equal-weight simulated portfolio; otherwise Module A starts from equal weights.

## Try real inference → Module A

`POST /module-a/rebalance-from-news` sends headlines through the **already loaded** models in `app.state.pipeline`, then uses the scores:

```json
{
  "as_of":"2024-05-16T23:00:00Z",
  "strategy":"full_event_aware",
  "articles":[
    {"ticker":"NVDA","text":"Nvidia reports weaker quarterly demand outlook",
     "published_at":"2024-05-15T12:00:00Z","source":"financial_news"}
  ]
}
```

This endpoint depends on the original locally installed **real model weights** and historical price access. If the NLP/Impact pipeline is not loaded, it returns **503**. Use representative supported tickers/dates (or local CSV price source). This is a simulated decision, not investment advice.

## Prepare an honest historical backtest

**Do NOT feed `impact_score_1_10` or `realized_impact` from Step-2 into Module A. Those are future realized labels.** Our converter reads the frozen `impact_step2_test_features.parquet` and calls the saved XGBoost model with the verified `base_score`, producing only *predicted* risk signals. This is out-of-sample relative to the Impact model's train + validation fit. IMPORTANT: if you tune Module A rules on these same test dates, the portfolio results stop being an unbiased final evaluation. Treat current defaults as preselected, or establish separate, chronologically earlier development periods and a fresh untouched test.

Requirements: original working `models/impact_engine/{impact_model_final.joblib,impact_preprocessor.joblib}`, original `configs/impact_model_feature_manifest.json` updated by Antigravity with `base_score: 0.021967076942109524` and frozen deciles; Step-2 TEST parquet copied locally from Kaggle.

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_module_a_signals --features "C:\path\to\impact_step2_test_features.parquet" --output data/module_a/holdout_signals.csv
.\.venv\Scripts\python.exe -m scripts.prepare_module_a_prices --start 2022-08-01 --end 2024-01-01 --dataset-id holdout
.\.venv\Scripts\python.exe -m scripts.run_module_a_backtest --dataset-id holdout --start-date 2022-08-19 --end-date 2023-12-26
```

**Prices:** Downloaded using `yfinance` historical **adjusted** closes with an explicit date range. The long-format CSV schema is exactly `date,ticker,adj_close`, one row per stock and market session. If you have a trustworthy adjusted-close export from FNSPID instead, place it at `data/module_a/holdout_prices.csv` using that schema and skip the download command. Price grids must be complete; no price forward-filling or silent ticker substitution. Verify adjusted-close corporate-action and data-source consistency; Yahoo is a third-party research data service, not an exchange-grade price feed.

**Signals:** `data/module_a/holdout_signals.csv` fields: `event_id,ticker,published_at,headline,sentiment_score,impact_score,event_class,event_confidence,source`. Historical `source=financial_news` because FNSPID is a news dataset; **do not claim two-source historical coverage**. Social media ingestion will be added later.

The backtest simulates trading at **weekly final market-session closes** using strictly earlier UTC-publication-date articles. Trades are applied after the close-to-close return for that session, never before. **Research simplification:** the strategy calculates portfolio weights using that close, so fills at exactly the same close are idealized, not a faithfully executable market-on-close order. Do not present the backtest as an execution-grade live-trading simulation; a later version could use next-session execution with slippage. Corporate-action-adjusted daily prices are used. Trading incurs a **10 bps traded-notional cost** (configurable). No leverage/shorts; weights sum to 100%, with a ±2pp stock band, ±5pp sector band and max 10% one-way turnover per scheduled rebalance. Portfolio equity changes due to market returns between rebalances, including possible deviations from initial weights; impossible constraint combinations fail explicitly.

Run all five strategy comparisons over the *same* price range:

1. Equal-weight buy-and-hold.
2. Weekly equal-weight rebalance (with costs).
3. Sentiment-only.
4. Sentiment + predicted Impact Score.
5. Full event-aware/source/recency/confidence strategy.

Outputs: `data/module_a/results/holdout_offline_comparison.json` with actual daily NAV, trades, risk/performance metrics, costs and all five comparisons. This file is created only after a successful run; the package contains **no invented portfolio-performance results**.

## API routes

- `GET /module-a/index` — 15-stock baseline, sectors and policy limits.
- `POST /module-a/rebalance` — deterministic, stateless proposal from scored headlines.
- `POST /module-a/rebalance-from-news` — real pretrained Risk Engine → proposal.
- `POST /module-a/backtest` — compare five strategies from `data/module_a/{dataset_id}_signals.csv` and `{dataset_id}_prices.csv`. Filename IDs are strictly validated; results saved locally under `results/<uuid>.json`.
- `GET /module-a/backtest/{run_id}` — retrieve full saved research comparison.

**Operational caution:** The backtest endpoint runs synchronously; use it locally / with small historical data. Do not expose endpoints to the public without API authentication, rate limiting, request size restrictions and durable job storage. Heavy model-loading workers: keep `--workers 1`. Backtesting 15 stocks requires memory and may need to move to an offline worker for cloud deployment.

## Frozen decisions / methodology

- Index constituents/config and initial equal weights are stored in `configs/module_a_config.json`.
- `source_weights` are transparent source-type **heuristics**, not externally verified credibility scores. The full event-aware method uses a neutral evidence prior (`evidence_prior_weight=0.5`) so source reliability, recency and confidence do not cancel out for a single headline; this prior is not statistically fitted.
- `impact_score` is *magnitude* of estimated market reaction, **not direction**. The strategy's signed input is NLP sentiment; we do **not** assume a positive sentiment predicts a price rise, and results must be validated.
- All signal/portfolio policy values are transparent research baselines, NOT statistically optimized performance claims.
- `published_at` requires a timezone and the date must be **strictly before the execution date**, including exact-time publications, for conservative FNSPID compatibility.
- Duplicated/reposted headlines (same ticker, date, normalized text) are counted once across sources.
- Missing/stale signals do **not** count as negative news.
- Historical backtest *must* use predicted Impact Engine values from held-out records; don't use realized returns as predictive input.
- Model A covers price-derived performance comparison, not a claim of superior returns. It does not yet include automated two-source ingestion, external trading execution, a dashboard, significance testing or full historical transaction liquidity modeling.

## Verification commands and what they prove

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests -k module_a
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.check_models
.\.venv\Scripts\python.exe -m scripts.smoke_inference
```

The module-only tests use synthetic deterministic fixture prices, not your actual model weights. Antigravity must additionally run `POST /module-a/rebalance-from-news` with real files, then prepare actual test signals and prices, run the full offline backtest, verify realistic NAV/costs/weights, and inspect any failures **before** declaring deployment readiness.
