# Soochana AI — Model Artifacts & Weight Checkpoints (`models/`)

This directory houses the deployed machine learning model weights, tokenizer vocabularies, configuration schemas, and serialized scikit-learn/XGBoost pipelines used in real-time inference.

---

## 1. Directory Structure

```text
models/
├── event_classifier/          # Fine-Tuned 5-Class FinBERT Event Classifier
│   ├── model.safetensors      # Trained PyTorch transformer weights (~439 MB)
│   ├── config.json            # Model architecture & id2label mapping
│   ├── tokenizer.json         # Fast WordPiece tokenizer
│   └── vocab.txt              # 30,873 token vocabulary
├── sentiment_regressor/       # Continuous Financial Sentiment Regressor
│   ├── model.safetensors      # Trained PyTorch transformer weights (~438 MB)
│   ├── config.json            # Regression head architecture configuration
│   ├── modeling_finbert_sentiment_regressor.py # PyTorch drop-in module
│   ├── tokenizer.json         # WordPiece tokenizer
│   └── vocab.txt              # 30,522 token vocabulary
└── impact_engine/             # Market Reaction Severity Predictor
    ├── impact_model_final.joblib   # Trained XGBoost Regressor estimator
    └── impact_preprocessor.joblib  # Fitted ColumnTransformer (scaling & encoding)
```

---

## 2. Hugging Face Hub Hosting & Automated Fetching

Trained checkpoints are mirrored on Hugging Face Hub for automated continuous deployment:

```bash
# Automated download into models/
python -m scripts.fetch_models

# Integrity and file verification
python -m scripts.check_models
```

| Sub-Model | Hugging Face Repository | Architecture |
| :--- | :--- | :--- |
| **Event Classifier** | [`shanmukhdatta/riskpulse-event-classifier`](https://huggingface.co/shanmukhdatta/riskpulse-event-classifier) | FinBERT 12-Layer + 5-Way Classification Head |
| **Sentiment Regressor** | [`shanmukhdatta/riskpulse-sentiment-regressor`](https://huggingface.co/shanmukhdatta/riskpulse-sentiment-regressor) | ProsusAI/finbert + Linear Tanh Regression Head |
| **Impact Engine** | [`shanmukhdatta/riskpulse-impact-engine`](https://huggingface.co/shanmukhdatta/riskpulse-impact-engine) | Scikit-Learn ColumnTransformer + XGBoost Booster |
