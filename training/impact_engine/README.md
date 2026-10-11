# RiskPulse AI — Market Impact Engine Training (Original Source)

This directory contains the **source code extracted from the three original Kaggle notebooks** for the RiskPulse AI Market Impact Severity Engine. The workflow implements historical abnormal return calculation, pre-publication feature engineering with NLP embeddings, and competitive candidate training (XGBoost vs. LightGBM).

**Important:** The source code and notebook structures have been preserved, but training was **not rerun** during repository packaging. The raw news archives, intermediate parquet feature matrices, and trained model files are **not bundled in this Git repository** (the trained XGBoost artifacts are published on Hugging Face at [`shanmukhdatta/riskpulse-impact-engine`](https://huggingface.co/shanmukhdatta/riskpulse-impact-engine)). This package documents the authentic training methodology.

---

## 1. Pipeline and Exact Source Files

| Stage | Notebook | Code Export | Key Outputs |
| :--- | :--- | :--- | :--- |
| **1. Dataset Generation** | `notebooks/01_dataset_generation.ipynb` | `scripts/01_dataset_generation.py` | `impact_step1_train.parquet`, `validation.parquet`, `test.parquet`, abnormal-return targets |
| **2. Feature Preparation** | `notebooks/02_feature_preparation.ipynb` | `scripts/02_feature_preparation.py` | `impact_step2_train_features.parquet`, `validation`, `test`, `impact_step2_feature_manifest.json`, `impact_decile_thresholds.json`, `impact_step2_report.json` |
| **3. Model Training & Selection** | `notebooks/03_model_training.ipynb` | `scripts/03_model_training.py` | `impact_model_final.joblib`, `impact_preprocessor.joblib`, evaluation report & confusion matrix |
| **Integrity Audit** | — | `check_source_package.py` | SHA-256 hash verification against `source_manifest.json` |

The `.ipynb` notebooks retain **all original markdown and code cells**, with execution outputs stripped for clean publication. The `scripts/*.py` files are faithful command-line exports with notebook-specific IPython directives commented out.

---

## 2. Problem Formulation & Severity Deciles

The Market Impact Engine models the **historical abnormal market reaction magnitude** following financial news releases:

* **Target Variable (`realized_impact`):**
  Continuous abnormal price return magnitude $|R_{\text{abnormal}}|$ observed in the market session following headline publication:
  $$|R_{\text{abnormal}}| = |R_{\text{stock}} - R_{\text{market}}|$$
* **Calibrated Deciles (`impact_score_1_10`):**
  To provide an interpretable metric for analysts and risk systems, continuous predictions are mapped to empirical deciles $D \in \{1, 2, \dots, 10\}$ based on the training split's quantile thresholds in `configs/impact_decile_thresholds.json`.

> **Domain Caveat:** The impact decile represents the **scale of market volatility / disruption**, not a directional price return forecast. Directional interpretation is separately informed by the continuous sentiment score.

---

## 3. Feature Pipeline & Anti-Leakage Safeguards

Stage 2 attaches pre-publication market context and frozen NLP predictions to every headline:

1. **Frozen NLP Model Outputs:**
   * 5 event class probabilities: $p_{\text{geopolitical}}, p_{\text{macroeconomic}}, p_{\text{credit\_event}}, p_{\text{merger\_acquisition}}, p_{\text{product\_launch}}$
   * Continuous sentiment score $s \in [-1.0, +1.0]$
2. **Pre-Publication Market Features (No Lookahead):**
   * $\sigma_{20}$: 20-day annualized rolling price volatility
   * 20-session volume ratio: $V_t / \bar{V}_{20}$
   * Trailing 5-day and 20-day price returns
3. **Sector Identifiers:**
   * One-hot encoded sector categories derived from `configs/sector_mapping.json`.
4. **Strict Leakage Prevention:**
   * Features strictly use market data from sessions dated **prior to** the publication calendar date (UTC).
   * Target variables (`realized_impact`, future reaction prices) are verified absent from the input feature manifest (`impact_step2_feature_manifest.json`).

---

## 4. Recorded Dataset Statistics

From the official Step-2 training report ([`data/impact_step2_report.json`](../../data/impact_step2_report.json)):

* **Total Cleaned Observations:** **27,460 rows**
  * **Train Split:** 19,084 rows (69.5%)
  * **Validation Split:** 3,446 rows (12.5%)
  * **Test Split:** 4,930 rows (18.0%)
* **Chronological Splitting:**
  Splits preserve chronological boundaries to ensure models are evaluated out-of-time rather than on random in-sample interpolations.

---

## 5. Candidate Evaluation & Model Selection

[`scripts/03_model_training.py`](scripts/03_model_training.py) trains and benchmarks two candidate architectures on the frozen Step-2 validation set:

1. **XGBoost Regressor (`xgboost.XGBRegressor`):**
   * Objective: `reg:squarederror`
   * Early stopping: 100 rounds monitoring validation MSE
2. **LightGBM Regressor (`lightgbm.LGBMRegressor`):**
   * Objective: `regression`
   * Early stopping: 100 rounds monitoring validation MSE

* **Selection & Export:**
  The winning candidate on validation error is refitted and exported via `joblib`:
  * `impact_model_final.joblib`: Trained estimator
  * `impact_preprocessor.joblib`: Fitted ColumnTransformer (handling numeric scaling and one-hot encoding)
  * **Deployed Runtime:** RiskPulse AI's production inference (`app/engine/impact.py`) expects the **XGBoost model artifact**.

---

## 6. How to Reproduce the Training Workflow

### 6.1 Prerequisites
* Python 3.11 with dependencies in `training/impact_engine/requirements.txt`:
  ```bash
  pip install -r training/impact_engine/requirements.txt
  ```
* Trained checkpoints for the **Event Classifier** and **Sentiment Regressor** (required by Stage 2).

### 6.2 Sequential Execution (Local / Kaggle)

```bash
# 1. Setup workspace paths
export RISKPULSE_IMPACT_WORKDIR="$PWD/training_runs/impact"
export RISKPULSE_IMPACT_INPUT_ROOT="$PWD/training_runs/impact"
mkdir -p "$RISKPULSE_IMPACT_WORKDIR"

# 2. Stage 1: Generate dataset from FNSPID / Price history
python training/impact_engine/scripts/01_dataset_generation.py

# 3. Stage 2: Feature preparation (Requires trained NLP models)
export RISKPULSE_EVENT_MODEL_DIR="/path/to/models/event_classifier"
export RISKPULSE_SENTIMENT_MODEL_DIR="/path/to/models/sentiment_regressor"
python training/impact_engine/scripts/02_feature_preparation.py

# 4. Stage 3: Train & select final impact model
python training/impact_engine/scripts/03_model_training.py
```

Outputs will be saved in `impact_step3_final_model/`. Copy `impact_model_final.joblib` and `impact_preprocessor.joblib` into `models/impact_engine/` for deployment.

---

## 7. Model Availability on Hugging Face Hub

To bypass retraining and immediately run full inference, the trained XGBoost model and preprocessor are hosted on Hugging Face:
[`shanmukhdatta/riskpulse-impact-engine`](https://huggingface.co/shanmukhdatta/riskpulse-impact-engine)

They can be downloaded automatically via:
```bash
python -m scripts.fetch_models
```
