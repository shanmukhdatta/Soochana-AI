# Event Classifier Training Scripts (`training/event_classifier/scripts/`)

This directory contains standalone Python CLI exports of the research notebooks for automated headless execution and diff inspection.

---

## 1. Script Inventory

* [`01_data_gathering.py`](01_data_gathering.py): CLI runner downloading and concatenating upstream Hugging Face financial datasets.
* [`02_preprocessing.py`](02_preprocessing.py): CLI runner performing SHA-256 deduplication and Parquet split generation.
* [`03_synthetic_generation_optional.py`](03_synthetic_generation_optional.py): Optional synthetic text generator using Groq API.
* [`04_finbert_training_real_only.py`](04_finbert_training_real_only.py): PyTorch training loop fine-tuning the 5-class FinBERT model.
