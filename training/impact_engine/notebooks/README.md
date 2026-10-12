# Impact Engine Research Notebooks (`training/impact_engine/notebooks/`)

This directory houses the sequential research notebooks for abnormal return calculation, pre-publication feature engineering, and candidate model benchmarking.

---

## 1. Notebook Pipeline

* [`01_dataset_generation.ipynb`](01_dataset_generation.ipynb): Computes realized abnormal return targets $|R_{\text{abnormal}}| = |R_{\text{stock}} - R_{\text{market}}|$ from historical prices and financial news archives.
* [`02_feature_preparation.ipynb`](02_feature_preparation.ipynb): Integrates rolling pre-publication volatility ($\sigma_{20}$), volume ratios, and frozen NLP multi-task model outputs.
* [`03_model_training.ipynb`](03_model_training.ipynb): Benchmarks XGBoost Regressor against LightGBM and evaluates decile thresholds.
