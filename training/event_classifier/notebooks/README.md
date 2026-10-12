# Event Classifier Research Notebooks (`training/event_classifier/notebooks/`)

This directory houses the four sequential Jupyter notebooks documenting data collection, source-sensitive preprocessing, synthetic minority augmentation, and 5-class FinBERT fine-tuning.

---

## 1. Notebook Pipeline

| Stage | Notebook File | Objective |
| :--- | :--- | :--- |
| **Stage 1** | [`01_data_gathering.ipynb`](01_data_gathering.ipynb) | Aggregates 26,172 candidate rows across 6 Hugging Face financial corpora. |
| **Stage 2** | [`02_preprocessing.ipynb`](02_preprocessing.ipynb) | Deduplication, text normalization, and 80/10/10 stratified split creation. |
| **Stage 3** | [`03_synthetic_generation_optional.ipynb`](03_synthetic_generation_optional.ipynb) | Optional LLM synthetic candidate generation for minority classes (kept separate from real splits). |
| **Stage 4** | [`04_finbert_training_real_only.ipynb`](04_finbert_training_real_only.ipynb) | Supervised fine-tuning of `nickmuchi/finbert-tone` backbone using PyTorch and Hugging Face Trainer. |
