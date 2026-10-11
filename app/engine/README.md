# RiskPulse AI — Core Risk Engine

`app/engine/` is the **core machine learning intelligence engine** of RiskPulse AI. It orchestrates three cooperating AI models to convert unstructured financial text (news headlines, corporate filings, social commentary) and historical market data into structured, explainable financial risk signals.

---

## 1. Architectural Scope & Pipeline

The Risk Engine operates as a sequential three-stage pipeline consumed by both Module A (Index Lab) and Module B (Stress Lab):

```text
Raw Text & Ticker
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                       Stage 1 & 2: NLP                      │
│                                                             │
│  FinBERT Event Classifier        FinBERT Sentiment          │
│  (5-Class Softmax Probabilities) (Continuous Score [-1, +1])│
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               └──────────────┬───────────────┘
                              │ + Pre-publication Market Context
                              ▼   (σ₂₀, Volume Ratio, Trailing Returns)
┌─────────────────────────────────────────────────────────────┐
│                  Stage 3: Market Impact                     │
│                                                             │
│             XGBoost Impact Severity Predictor               │
│             (Abnormal Reaction Decile 1–10)                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
                   Structured Risk Signal
                   (JSON Response Schema)
```

---

## 2. Component Structure

| File | Purpose | Key Classes / Functions |
| :--- | :--- | :--- |
| [`pipeline.py`](pipeline.py) | Top-level facade orchestrating NLP, features, and impact models | `RiskEngine`, `RiskEngine.from_settings()`, `RiskEngine.analyze()` |
| [`nlp.py`](nlp.py) | Hugging Face Transformers runtime for event & sentiment models | `NLPModels`, `NLPModels.analyze()`, `Softmax` |
| [`sentiment_model.py`](sentiment_model.py) | Custom FinBERT PyTorch module with continuous regression head | `FinBertSentimentRegressor` |
| [`impact.py`](impact.py) | XGBoost inference engine and decile calibration mapper | `ImpactPredictor`, `ImpactPredictor.predict()` |

---

## 3. The Three Cooperating AI Models

### 3.1 Model 1: Five-Class Event Classifier
* **Architecture:** FinBERT (`yiyanghkust/finbert-tone` / `BertForSequenceClassification`) fine-tuned for single-label financial event detection.
* **Weights:** `models/event_classifier/model.safetensors` (~439 MB), hosted at [`shanmukhdatta/riskpulse-event-classifier`](https://huggingface.co/shanmukhdatta/riskpulse-event-classifier).
* **Taxonomy (Immutable Contract):**
  ```json
  {
    "0": "Geopolitical",
    "1": "Macroeconomic",
    "2": "Credit Event",
    "3": "Merger / Acquisition",
    "4": "Product Launch"
  }
  ```
* **Output:** Full 5-class softmax probability distribution $\mathbf{p} = [p_0, p_1, p_2, p_3, p_4]$ such that $\sum p_i = 1.0$, plus top class and confidence score.

### 3.2 Model 2: Continuous Financial Sentiment Regressor
* **Architecture:** [`FinBertSentimentRegressor`](sentiment_model.py) — FinBERT transformer backbone with a custom linear regression head (`Linear(hidden_size, 1)` followed by `Tanh` activation).
* **Weights:** `models/sentiment_regressor/model.safetensors` (~438 MB), hosted at [`shanmukhdatta/riskpulse-sentiment-regressor`](https://huggingface.co/shanmukhdatta/riskpulse-sentiment-regressor).
* **Output:** Continuous sentiment scalar $s \in [-1.0, +1.0]$, avoiding discrete loss of nuance from standard 3-class positive/neutral/negative classifiers.
  * Thresholds: Positive ($s > +0.15$), Neutral ($-0.15 \le s \le +0.15$), Negative ($s < -0.15$).

### 3.3 Model 3: XGBoost Market Impact Engine
* **Architecture:** Scikit-Learn pipeline wrapping an XGBoost regressor trained on historical market reactions (FNSPID financial news + market data).
* **Artifacts:** `models/impact_engine/impact_model_final.joblib` and `impact_preprocessor.joblib`, hosted at [`shanmukhdatta/riskpulse-impact-engine`](https://huggingface.co/shanmukhdatta/riskpulse-impact-engine).
* **Input Features (Feature Vector):**
  * 5 event probabilities: $p_{\text{geopolitical}}, \dots, p_{\text{product\_launch}}$
  * Continuous sentiment score $s$
  * Pre-publication market indicators (via `app.services.features.FeatureBuilder`):
    * $\sigma_{20}$: 20-day annualized rolling price volatility
    * Volume ratio: last trading volume divided by 20-session rolling mean
    * Trailing 5-day and 20-day stock returns
* **Output:** Continuous abnormal reaction magnitude mapped to an empirical **severity decile (1 to 10)** based on training-set cutoffs in `configs/impact_decile_thresholds.json`.

---

## 4. Methodological & Financial Disclaimers

1. **Reaction Magnitude, Not Directional Return:**
   The impact decile (1–10) measures the estimated **scale of abnormal market reaction / disruption**, not whether the stock price will rise or fall. Directional bias is informed separately by the continuous sentiment score.
2. **Strict Pre-Publication Cutoff:**
   Features passed to the impact engine use market data strictly preceding the news publication date to prevent lookahead contamination.
3. **Model Weight Loading:**
   When `SKIP_MODEL_LOAD=true` in `.env`, the engine reports `503 Service Unavailable` for `/analyze` rather than generating fabricated or mock predictions. Full inference requires genuine weights in `models/`.
