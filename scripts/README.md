# Soochana AI — Operational Scripts & CLI Tools (`scripts/`)

`scripts/` provides command-line utilities, model download tools, quantitative simulation runners, live ingestion triggers, and verification harnesses.

---

## 1. Script Categories & Operational Workflows

```text
                               ┌────────────────────────────────┐
                               │           `scripts/`           │
                               │   Operational CLI Utilities    │
                               └───────────────┬────────────────┘
                                               │
         ┌────────────────────────┬────────────┴────────────┬────────────────────────┐
         ▼                        ▼                         ▼                        ▼
┌──────────────────┐    ┌──────────────────┐      ┌──────────────────┐     ┌──────────────────┐
│ Model Deployment │    │ Lab Simulations  │      │ Ingestion Ops    │     │ Audits & Checks  │
├──────────────────┤    ├──────────────────┤      ├──────────────────┤     ├──────────────────┤
│• fetch_models.py │    │• run_module_a_   │      │• run_ingestion_  │     │• smoke_inference │
│• check_models.py │    │  backtest.py     │      │  once.py         │     │• verify_all_live_│
│                  │    │• run_module_b_   │      │• run_ingestion_  │     │  endpoints.py    │
│                  │    │  demo.py         │      │  scheduler.py    │     │• compare_feature_│
│                  │    │                  │      │• check_sources.py│     │  parity.py       │
└──────────────────┘    └──────────────────┘      └──────────────────┘     └──────────────────┘
```

---

## 2. Script Inventory

### 2.1 Model Management & Health
* [`fetch_models.py`](fetch_models.py): Downloads fine-tuned FinBERT and XGBoost weights from Hugging Face Hub into `models/`.
* [`check_models.py`](check_models.py): Inspects file presence, safetensors headers, and configuration integrity across all 3 models.
* [`smoke_inference.py`](smoke_inference.py): Loads the live models into memory and runs a real inference request on AAPL news, outputting structured JSON.

### 2.2 Quantitative Simulation Runners
* [`run_module_a_backtest.py`](run_module_a_backtest.py): Executes the 357-session historical walk-forward backtest for Module A, evaluating 5 strategy variants against the equal-weight buy-and-hold benchmark.
* [`run_module_b_demo.py`](run_module_b_demo.py): Runs credit stress and reverse stress loss calculations on the 30-position synthetic institutional balance sheet.
* [`prepare_module_a_prices.py`](prepare_module_a_prices.py) & [`prepare_module_a_signals.py`](prepare_module_a_signals.py): Generates historical price and signal grids from market data.

### 2.3 Ingestion Pipeline Utilities
* [`run_ingestion_once.py`](run_ingestion_once.py): Triggers a single polling cycle across enabled stream adapters (Marketaux / Bluesky).
* [`run_ingestion_scheduler.py`](run_ingestion_scheduler.py): Starts the background polling daemon using configured interval timers.
* [`check_ingestion_sources.py`](check_ingestion_sources.py): Validates provider network reachability and credentials.

### 2.4 Verification, Audits & Architecture
* [`verify_all_live_endpoints.py`](verify_all_live_endpoints.py): Executes an automated test pass across all 15 FastAPI routes against a live server.
* [`audit_module_b_independent.py`](audit_module_b_independent.py): Re-calculates bond, loan, swap, and option sensitivities independently to verify pricing formulas.
* [`compare_feature_parity.py`](compare_feature_parity.py): Asserts exact numerical parity between runtime feature builders and offline Step-2 training parquets.
* [`architecture_diagram.html`](architecture_diagram.html) & [`render_architecture.mjs`](render_architecture.mjs): High-resolution pipeline visualizer.
