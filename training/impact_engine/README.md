# Impact Engine — Training Code

This directory holds source preserved from the three original Impact Engine notebooks. Notebook outputs are removed; **no training results are bundled**. Matching `.py` files are sequential source exports (not a new training algorithm) with optional path overrides.

## Install

Use a **separate** Python 3.11 environment. Run from the repository root (the folder containing `training/`).

```bash
python -m venv .venv-train
source .venv-train/bin/activate
pip install -r training/impact_engine/requirements.txt
```

For Windows PowerShell, activate with `.venv-train\Scripts\Activate.ps1`.

## Run in order

1. `01_dataset_generation.py` — downloads/streams the original FNSPID news and stock prices (with yfinance fallback), filters dates/tickers, constructs historical abnormal-return targets and training-only deciles, and writes Step-1 Parquet files.
2. `02_feature_preparation.py` — **requires original trained five-class event-model and sentiment-regressor weights and tokenizers**. It loads frozen Step-1 splits, generates NLP probabilities/sentiment, and attaches pre-publication market features; outputs the Step-2 Parquet files and feature manifest.
3. `03_model_training.py` — uses the frozen Step-2 splits to choose between **XGBoost and LightGBM candidates on validation**, refits the selected model, and exports its model/preprocessor files. RiskPulse's deployed inference code expects **an XGBoost model**; do not substitute a selected LightGBM artifact without changing/testing that runtime intentionally.

Run exports on Kaggle or a machine with adequate CPU/RAM and network access. On Kaggle, notebooks can be run in order with their attached input datasets and previously trained checkpoints.

### Local shell (Linux/macOS example)

```bash
export RISKPULSE_IMPACT_WORKDIR="$PWD/impact_work"
export RISKPULSE_IMPACT_INPUT_ROOT="$PWD/impact_work"
mkdir -p "$RISKPULSE_IMPACT_WORKDIR"
python training/impact_engine/scripts/01_dataset_generation.py
# Set these to folders that contain the ACTUAL trained weights, configs and tokenizers:
export RISKPULSE_EVENT_MODEL_DIR="/absolute/path/to/event_classifier"
export RISKPULSE_SENTIMENT_MODEL_DIR="/absolute/path/to/sentiment_regressor"
python training/impact_engine/scripts/02_feature_preparation.py
python training/impact_engine/scripts/03_model_training.py
```

Optional Step-2 overrides: `RISKPULSE_IMPACT_DATA_DIR` points to the Step-1 Parquet folder; `RISKPULSE_PRICE_ARCHIVE` points to an existing FNSPID price ZIP. With path variables unset, exports retain the original notebook discovery behavior.

**External prerequisites:** FNSPID source dataset and price data (or permitted replacements), Internet connectivity where downloading, and the original fine-tuned event and sentiment model weights. Those files are *not* synthesized or bundled here. FNSPID source selection and yfinance fallback can change results across runs; for exact replication, pin original dataset revisions, price snapshots, library versions, and checkpoint hashes.

**Source caveats:** The first two notebooks produce files for the next stage and use chronological splitting/leakage guards. Step 3 saves the selected candidate's artifacts; no training run has been executed as part of this source packaging. The code was syntactically verified, but full end-to-end reproduction requires the external inputs above.
