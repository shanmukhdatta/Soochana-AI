# Deployed Event Classifier Weights (`models/event_classifier/`)

This directory contains the production checkpoint for the **Soochana AI Five-Class Financial Event Classifier**.

---

## 1. Artifact Manifest

* `model.safetensors`: Safetensors binary checkpoint of the fine-tuned 12-layer FinBERT transformer model (~439 MB).
* `config.json`: Declares `model_type: "bert"`, `num_labels: 5`, and canonical `id2label` mapping:
  `{0: "Geopolitical", 1: "Macroeconomic", 2: "Credit Event", 3: "Merger / Acquisition", 4: "Product Launch"}`.
* `tokenizer.json` & `vocab.txt`: WordPiece vocabulary of 30,873 tokens.
* `tokenizer_config.json` & `special_tokens_map.json`: Fast tokenizer configuration.

---

## 2. Runtime Loading Contract

Loaded by [`backend/app/engine/nlp.py`](../../backend/app/engine/nlp.py) via Hugging Face `AutoTokenizer` and `AutoModelForSequenceClassification`. Validates `num_labels == 5`, head presence, and vocabulary bounds before accepting requests.
