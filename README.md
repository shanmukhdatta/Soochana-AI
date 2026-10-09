# Unified Financial Risk Engine — Source-only FastAPI Backend (v1)

**Purpose:** Given one financial-news/social-media headline, ticker, timestamp, the API runs the **frozen 5-class Event FinBERT**, **frozen FiQA-trained continuous Sentiment FinBERT**, constructs **strictly pre-publication daily market features**, and loads the **frozen XGBoost Impact model** with the **exact saved preprocessing pipeline**. Returns a unified Risk Signal consumed later by Module A (tactical rebalancer) and Module B (portfolio stress testing).

> **This archive does not contain the trained model weights.** All tests using fake/mocked models are software-contract checks, not evidence of real-model accuracy. Production readiness requires installing the original weights and passing the real-model smoke test.

## Requirements

- Python **3.11 or 3.12** recommended; use a clean virtual environment to avoid Kaggle/NumPy binary conflicts.
- Roughly **2+ GB disk** for the two BERT models and dependencies; sufficient RAM for two BERT models and inference. CPU inference is supported but slower.
- Internet only for downloading packages/models, and when using the default yfinance daily-price provider. `PRICE_PROVIDER=csv` permits offline prices if you supply appropriately formatted **historical adjusted-close and volume CSVs**.

## Model files: do NOT upload ~500 MB folders into the GitHub repo

Place the files below, or point the `*_MODEL_DIR` environment variables to existing trusted directories.

```text
models/event_classifier/
    config.json
    model.safetensors          (or pytorch_model.bin)
    tokenizer.json or vocab.txt
    tokenizer_config.json
    special_tokens_map.json    (if exported)

models/sentiment_regressor/
    config.json
    model.safetensors          (or pytorch_model.bin)
    tokenizer.json or vocab.txt
    tokenizer_config.json
    special_tokens_map.json    (if exported)
    modeling_finbert_sentiment_regressor.py (optional reference)

models/impact_engine/
    impact_model_final.joblib
    impact_preprocessor.joblib
    impact_model_final_xgboost.json (optional native backup)
```

The `config_reference/` files are **CHECK-ONLY abbreviated examples**, not deployable model configurations. Use the original model's **full** config.json. Your models use **different tokenizer vocabulary sizes** (event 30,873; sentiment 30,522), so never mix tokenizers.

`impact_preprocessor.joblib` is **mandatory**. The native XGBoost JSON alone cannot replace the fitted sklearn one-hot encoder.

### Optional: host YOUR models on Hugging Face Hub

Upload only the real, trusted model exports to repositories you control. Set:

```bash
export EVENT_MODEL_REPO='your-name/your-event-finbert'
export SENTIMENT_MODEL_REPO='your-name/your-sentiment-finbert'
export IMPACT_MODEL_REPO='your-name/your-impact-artifacts'
# export HF_TOKEN='...'  # only if private repos require it; never commit it
python -m scripts.fetch_models
```

The API itself does **not** silently download model weights on requests. The optional script downloads once to configured directories.

## Run locally

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Optional on CPU-only Linux: install the CPU PyTorch wheel first to save disk.
# pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m scripts.check_models
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Set variables via shell/environment; `.env.example` shows all options. If running from a different working directory, prefer **absolute** `EVENT_MODEL_DIR`, `SENTIMENT_MODEL_DIR`, etc. `SKIP_MODEL_LOAD=1` is intended for unit tests only; in that mode `GET /health` returns HTTP 503 until a real engine is attached.

Visit `http://127.0.0.1:8000/docs` for interactive FastAPI docs, then check `/health`.

### Example API call

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"The company announced a new product launch for business customers.","ticker":"AAPL","published_at":"2026-10-08T08:00:00Z","source":"financial_news"}'
```

`published_at` **must** contain a timezone. Unsupported tickers are deliberately rejected instead of pretending the model was trained on them.

The response contains `sentiment.score`, `event.class_name`, `event.confidence`, `event.probabilities`, `impact.predicted_realized_impact`, `impact.score` (1–10), and `market_context`, including the actual pre-publication stock/SPY feature dates. Missing prices return an explicit error, **not a made-up impact score**.

Batch endpoint: `POST /analyze/batch` accepts `{ "items": [<same input objects>] }`, up to 20 entries (configurable lower via `MAX_BATCH_SIZE`). Batches fail atomically on a provider error.

## Exact Step-2 feature parity

The model uses **14 numerical + 3 categorical features**, in the order stored in `configs/impact_model_feature_manifest.json`:

- Sentiment score/absolute score, five event softmax probabilities, event confidence.
- Trailing stock annualized `20d` volatility, latest-session volume divided by latest 20-session mean volume, stock `5d` and `20d` returns.
- Trailing `20d` SPY return, `20d` annualized SPY volatility.
- `event_predicted_class`, frozen ticker sector, and market regime derived from **frozen training-only thresholds**.

The trained XGBoost model is loaded from `impact_model_final.joblib` alongside the **original** `impact_preprocessor.joblib`. Predicted continuous `realized_impact` is clipped to zero if negative, then converted using the **original train-only nine decile thresholds**. This score measures predicted absolute market-reaction severity, **not a directional price forecast, confidence, or guaranteed price movement**.

### Temporal alignment

We mirror the frozen Step-2 methodology: define `feature_cutoff_date = published_at converted to UTC calendar date`, then use the **last completed session with market_date < feature_cutoff_date** for **both** the company and SPY. Same-calendar-day prices, later prices and reaction-day returns are **never** predictors. This intentionally sacrifices some same-day price context in exchange for auditability, particularly where old source timestamps were date-only.

### Caveat: training/serving source shift

Step 2 used FNSPID adjusted historical stock prices. The default server uses **yfinance**. Even with the same feature formulas, differences in adjustment policy/coverage may cause training-serving skew. Before demonstrating a historical replay, compare features against a sample historical Step-2 parquet row. You may choose CSV files exported from the same historical price archive for exact-source matching.

The trained event classifier heavily favored Product Launch on the FNSPID event subset. The Impact model's holdout performance is **modest** (test RMSE ~0.02133, Spearman ~0.324, exact 1–10 match ~12.94%); do not claim it reliably predicts exact deciles or all real-world events. Scores should be accompanied by model provenance.

## Tests

```bash
pytest -q
```

This suite checks API contracts with fake models, exact decile mapping, manifest integrity, market cutoff leakage, market-feature formulas, price data normalization, and missing-model behavior. Mock tests do **not** establish true 3-model integration; after placing weights, run:

```bash
python -m scripts.smoke_inference
```

This real-model smoke test contacts the configured price provider. If the source is unavailable, fix the provider rather than inventing a fallback score.

## Docker / deployment

```bash
docker build -t unified-risk-engine .
docker run --rm -p 8000:8000 -v /absolute/path/to/models:/service/models:ro unified-risk-engine
```

For CPU-only public hosting, confirm host memory can fit two full BERT encoders, the XGBoost model, and runtime libraries. Free 512 MB instances are too small. The included Dockerfile sets one worker to avoid duplicating model memory. For short demonstrations, run locally and use a suitable temporary tunnel **only if public access is needed**; add access restrictions before exposing a production API. Use provider quotas and don't include secrets in the repo.

## Roadmap (not yet included)

1. Both downstream modules consume the `/analyze` contract (no repeated AI inference inside modules).
2. Two real ingestion adapters (e.g. financial news and public social data) with licensing/rate-limit awareness, deduplication and scheduling.
3. Unified dashboard, persistent event store, observability, API key/auth, retries, and audit trail.
4. Integration/parity test with actual models and sample Stage-2 historical features before production/demo sign-off.

**This backend is a deployable code scaffold once you supply the trained weights; it is not yet an end-to-end production-tested service.**

**Library compatibility:** sklearn/joblib model artifacts should ideally be loaded with the same scikit-learn/XGBoost major versions used to create them in Kaggle. If loading warnings or mismatches occur, check the training notebook environment and install matching versions; never silently retrain the preprocessor in production.

## Optional historical feature-parity check

If you have a **Step-2 historical parquet** and its historical FNSPID adjusted-price CSVs, compare independently computed serving features to the frozen historical feature values:

```bash
export PRICE_PROVIDER=csv
export PRICE_CSV_DIR=/path/to/historical-price-csvs
python -m scripts.compare_feature_parity /path/to/impact_step2_test_features.parquet --max-rows 20
```

A local CSV must be named `<TICKER>.csv`, for example `AAPL.csv` or `SPY.csv`, and contain `Date`, `Adj Close`, and `Volume`. The check prints differences but does not silently change training artifacts or thresholds. **Parity can fail if serving from yfinance while training used FNSPID**; investigate any discrepancies before sign-off.
