# Sentiment Regressor Training Scripts (`training/sentiment/scripts/`)

This directory contains standalone Python CLI exports of the sentiment regression research notebooks.

---

## 1. Script Inventory

* [`01_fiqa_data_preparation.py`](01_fiqa_data_preparation.py): Performs data sanitization, duplicate resolution, and Parquet split creation.
* [`02_finbert_regression_training.py`](02_finbert_regression_training.py): Executes the PyTorch fine-tuning loop for the continuous sentiment regressor.
