# Soochana AI — Market Impact Severity Engine

This module contains the research, data pipelines, model architecture, training notebooks, and inference workflows for the **Soochana AI Market Impact Severity Engine**.

---

## 1. System Architecture Overview

The Market Impact Engine models the **historical abnormal market reaction magnitude** following financial news releases. It combines pre-publication market conditions with frozen NLP features to predict the scale of post-announcement market volatility, mapped into calibrated risk deciles ($1 \dots 10$).

```text
====================================================================================================
                                TRAINING PIPELINE ARCHITECTURE
====================================================================================================

  [ Upstream Archives: FNSPID Financial News & Historical Daily Stock Prices ]
                │
                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Stage 1: Abnormal Return Target Generation                  │
  │ • Target variable: |R_abnormal| = |R_stock - R_market|      │
  │ • Market benchmark: S&P 500 ETF (SPY)                       │
  │ • Measures post-publication 1-day abnormal volatility scale │
  └─────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Stage 2: Feature Engineering & Anti-Leakage Safeguards      │
  │                                                             │
  │ 1. Pre-Publication Market Indicators (Strictly t-1, no leak):│
  │    • σ_20: 20-day annualized rolling return volatility      │
  │    • Volume Ratio: V_t / V̄_20 (20-day average volume)       │
  │    • Momentum: Trailing 5-day return, Trailing 20-day return│
  │ 2. Frozen NLP Multi-Task Model Predictions:                 │
  │    • 5 Event Class Probabilities (p_0 ... p_4)              │
  │    • Continuous Sentiment Score s ∈ [-1.0, +1.0]            │
  │ 3. Asset Metadata:                                          │
  │    • Sector category (One-hot encoded)                      │
  └─────────────────────────────┬───────────────────────────────┘
                                │ (27,460 Cleaned Samples)
                                ▼ Chronological Split
                Train: 19,084 (69.5%) │ Val: 3,446 (12.5%) │ Test: 4,930 (18.0%)
                                │
  ┌─────────────────────────────┴─────────────────────────────────────────────────────────────┐
  │ Stage 3: Candidate Model Benchmarking & Decile Threshold Calibration                      │
  │                                                                                           │
  │   Feature Matrix X ───> [ ColumnTransformer Preprocessor ]                                │
  │                         - StandardScaler (Numeric market features, NLP probs)             │
  │                         - OneHotEncoder (Sector categoricals)                             │
  │                                     │                                                     │
  │                                     ▼                                                     │
  │   Candidate 1: XGBoost Regressor (XGBRegressor)                                           │
  │   Candidate 2: LightGBM Regressor (LGBMRegressor)                                         │
  │   Selection Metric: Validation Mean Squared Error (MSE)                                   │
  │   Winner: XGBoost (Exported to impact_model_final.joblib)                                 │
  │                                     │                                                     │
  │                                     ▼                                                     │
  │   Decile Quantile Calibration on Training Set Predictions:                                │
  │   - Computes quantiles p10, p20, p30, ..., p90 of predicted abnormal impact              │
  │   - Exported to configs/impact_decile_thresholds.json                                     │
  └─────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                        │
                                        ▼ Artifacts Exported
                        [ models/impact_model/ ]
                        • impact_model_final.joblib
                        • impact_preprocessor.joblib
                        • configs/impact_decile_thresholds.json

====================================================================================================
                                INFERENCE PIPELINE ARCHITECTURE
====================================================================================================

  Input News Record + Pre-Publication Market Context
                │
                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Feature Assembly (`app/services/features.py`)               │
  │ • Fetch rolling prices via PriceProvider (offline or API)   │
  │ • Compute σ_20, volume ratio, 5d return, 20d return         │
  │ • Inject event probabilities (5 dims) & sentiment score     │
  │ • Validate against manifest: ensure no future leakage       │
  └─────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Scikit-Learn Preprocessing (`ImpactPredictor`)              │
  │ • Load fitted ColumnTransformer (`impact_preprocessor`)     │
  │ • Numeric standard scaling & one-hot categorical expansion  │
  │ • Output shape: (1, n_features)                             │
  └─────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ XGBoost Booster Prediction                                  │
  │ • Model evaluation with restored base_score parameter       │
  │ • Predicted continuous abnormal reaction: ŷ ≥ 0.0           │
  └─────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Decile Binning Calibration (`np.searchsorted`)              │
  │   score = searchsorted(frozen_deciles, ŷ, side='right') + 1 │
  │   Maps ŷ into integer risk severity scale: [1, 10]          │
  └─────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
  Output JSON:
  {
    "predicted_realized_impact": 0.0384,
    "score": 7
  }
```

---

## 2. Problem Formulation & Severity Deciles

The Market Impact Engine models the **historical abnormal market reaction magnitude** following financial news releases:

* **Target Variable (`realized_impact`):**
  Continuous abnormal price return magnitude $|R_{\text{abnormal}}|$ observed in the market session following headline publication:
  $$|R_{\text{abnormal}}| = |R_{\text{stock}} - R_{\text{market}}|$$
* **Calibrated Deciles (`impact_score_1_10`):**
  To provide an interpretable metric for analysts and risk systems, continuous predictions are mapped to empirical deciles $D \in \{1, 2, \dots, 10\}$ based on the training split's quantile thresholds in [`configs/impact_decile_thresholds.json`](../../configs/impact_decile_thresholds.json).

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
  * **Deployed Runtime:** Soochana AI's production inference (`app/engine/impact.py`) expects the **XGBoost model artifact**.

---

## 6. How to Reproduce & Run Locally

### Option A: Download Trained Weights from Hugging Face Hub (Production Path)

The trained XGBoost and preprocessor artifacts are published on Hugging Face Hub:
```bash
python -m scripts.fetch_models
python -m scripts.check_models
```

### Option B: Retrain from Notebooks

```bash
# 1. Install dependencies
pip install -r training/impact_engine/requirements.txt

# 2. Run feature generation and model training
python training/impact_engine/scripts/01_dataset_generation.py
python training/impact_engine/scripts/02_feature_preparation.py
python training/impact_engine/scripts/03_model_training.py
```
