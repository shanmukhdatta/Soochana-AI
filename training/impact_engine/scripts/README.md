# Impact Engine Training Scripts (`training/impact_engine/scripts/`)

This directory contains standalone Python CLI exports of the Market Impact Engine research notebooks.

---

## 1. Script Inventory

* [`01_dataset_generation.py`](01_dataset_generation.py): Generates abnormal-return target parquets.
* [`02_feature_preparation.py`](02_feature_preparation.py): Computes pre-publication indicators and outputs `impact_step2_feature_manifest.json` and `impact_decile_thresholds.json`.
* [`03_model_training.py`](03_model_training.py): Fits the winning XGBoost Regressor estimator and exports `impact_model_final.joblib` and `impact_preprocessor.joblib`.
