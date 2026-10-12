# Deployed Market Impact Engine Weights (`models/impact_engine/`)

This directory contains the production models for the **Soochana AI Market Impact Severity Engine**.

---

## 1. Artifact Manifest

* `impact_model_final.joblib`: Trained `xgboost.XGBRegressor` estimator predicting abnormal price reaction magnitude.
* `impact_preprocessor.joblib`: Pre-fitted scikit-learn `ColumnTransformer` (standardizing numeric indicators and one-hot encoding sector categories).

---

## 2. Runtime Loading Contract

Loaded by [`backend/app/engine/impact.py`](../../backend/app/engine/impact.py). Restores the trained `base_score` ($0.021967$) and maps predictions into calibrated deciles $1 \dots 10$ using [`configs/impact_decile_thresholds.json`](../../configs/impact_decile_thresholds.json).
