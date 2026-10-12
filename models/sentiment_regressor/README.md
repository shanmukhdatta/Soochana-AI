# Deployed Sentiment Regressor Weights (`models/sentiment_regressor/`)

This directory contains the production checkpoint for the **Soochana AI Continuous Financial Sentiment Regressor**.

---

## 1. Artifact Manifest

* `model.safetensors`: Safetensors binary checkpoint of the fine-tuned FinBERT regression model (~438 MB).
* `config.json`: Configuration declaring `problem_type: "regression"`, `num_labels: 1`, and `sentiment_head: "Dropout -> Linear(hidden_size,1) -> tanh"`.
* `modeling_finbert_sentiment_regressor.py`: PyTorch model implementation (`FinBertSentimentRegressor`).
* `tokenizer.json` & `vocab.txt`: WordPiece vocabulary of 30,522 tokens.

---

## 2. Runtime Loading Contract

Loaded by [`backend/app/engine/nlp.py`](../../backend/app/engine/nlp.py) and [`backend/app/engine/sentiment_model.py`](../../backend/app/engine/sentiment_model.py). Verifies regression head metadata and continuous output bounds in $[-1.0, +1.0]$.
