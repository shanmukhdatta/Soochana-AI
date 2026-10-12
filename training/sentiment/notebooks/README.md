# Sentiment Regressor Research Notebooks (`training/sentiment/notebooks/`)

This directory houses the sequential Jupyter notebooks documenting data sanitization and continuous FinBERT fine-tuning on the FiQA financial sentiment challenge corpus.

---

## 1. Notebook Pipeline

* [`01_fiqa_data_preparation.ipynb`](01_fiqa_data_preparation.ipynb): Downloads and cleans the FiQA dataset, validates numerical bounds $[-1.0, +1.0]$, resolves duplicates via SHA-256 hashes, and exports partitioned Parquet splits.
* [`02_finbert_regression_training.ipynb`](02_finbert_regression_training.ipynb): Implements supervised regression fine-tuning of `ProsusAI/finbert` with early stopping and MSE loss optimization.
