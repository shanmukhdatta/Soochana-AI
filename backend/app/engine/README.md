# Soochana AI — Unified Risk Engine (`app/engine/`)

The **Unified Risk Engine** coordinates three machine learning models and market feature extraction services to convert unstructured financial text into structured quantitative risk intelligence.

---

## 1. High-Level Engine Architecture

```text
                               ┌───────────────────────────────────────────────┐
                               │           AnalyzeRequest (Input Payload)      │
                               │  - text: "Fed raises benchmark rate by 50bps" │
                               │  - ticker: "JPM"                              │
                               │  - published_at: "2026-03-15T14:30:00Z"       │
                               │  - source: "financial_news"                   │
                               └───────────────────────┬───────────────────────┘
                                                       │
                                                       ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                      PART 1: NLP FOUNDATION MODELS                                     │
  │                                          (`app/engine/nlp.py`)                                         │
  │                                                                                                        │
  │      ┌────────────────────────────────────────┐      ┌────────────────────────────────────────┐        │
  │      │     1A: FinBERT Event Classifier       │      │   1B: FinBERT Sentiment Regressor      │        │
  │      │   (Backbone: 12 Layers, 768-Dim)       │      │    (Backbone: ProsusAI/finbert)        │        │
  │      │   Linear Head: 768 ──> 5 Logits        │      │    Reg Head: 768 ──> 1 ──> tanh(·)     │        │
  │      │   Softmax ──> 5-Class Probabilities    │      │    Output: s ∈ [-1.0, +1.0]            │        │
  │      └───────────────────┬────────────────────┘      └───────────────────┬────────────────────┘        │
  │                          │                                               │                             │
  │                          │ p_event = [p_0 ... p_4]                       │ sentiment score s           │
  └──────────────────────────┼───────────────────────────────────────────────┼─────────────────────────────┘
                             │                                               │
                             └───────────────────────┬───────────────────────┘
                                                     │
                                                     ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                PART 2: ZERO-LOOKAHEAD FEATURE BUILDER                                  │
  │                            (`app/services/features.py`, `prices.py`)                                  │
  │                                                                                                        │
  │  Historical Market Data Lookup (Strictly prior sessions t-1, zero future leakage):                     │
  │  • PriceProvider fetches historical OHLCV grid from offline CSV or live yfinance                       │
  │  • 20-Day Annualized Volatility: σ_20 = std(returns_20) × √252                                         │
  │  • Volume Ratio: V_t / V̄_20                                                                           │
  │  • Momentum Indicators: trailing 5-day return, trailing 20-day return                                  │
  │  • One-Hot Sector Encoding: Sector mapped from configs/sector_mapping.json                             │
  │  • Unified Feature Vector: [σ_20, vol_ratio, ret_5d, ret_20d, p_0..p_4, s, sector_onehot]             │
  └──────────────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                   PART 3: MARKET IMPACT PREDICTOR                                      │
  │                                        (`app/engine/impact.py`)                                        │
  │                                                                                                        │
  │  • Scikit-Learn Preprocessor: ColumnTransformer (impact_preprocessor.joblib)                           │
  │    Standardizes numerical indicators, encodes categoricals                                             │
  │  • XGBoost Regressor (impact_model_final.joblib):                                                      │
  │    Evaluates fitted booster with restored base_score (fit mean = 0.021967)                             │
  │    Continuous Abnormal Volatility Return: ŷ = max(pred, 0.0)                                           │
  │  • Decile Calibration (configs/impact_decile_thresholds.json):                                         │
  │    score = np.searchsorted(deciles, ŷ, side='right') + 1 ──> Decile ∈ [1, 10]                          │
  └──────────────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
                               ┌───────────────────────────────────────────────┐
                               │          AnalyzeResponse (Output Payload)     │
                               │  - sentiment: {score: -0.62, label: "negative"}│
                               │  - event: {class_name: "Macroeconomic", ...}  │
                               │  - impact: {score: 8, predicted_impact: 0.041}│
                               │  - market_context: {volatility_20d: 0.24, ...}│
                               └───────────────────────────────────────────────┘
```

---

## 2. Component Deep Dive: The 3 Engine Parts

### Part 1: NLP Foundation Models (`nlp.py`, `sentiment_model.py`)

* **Dual Tokenizer Isolation:** Because the Event Classifier and Sentiment Regressor utilize distinct domain tokenizers with differing vocabulary hashes, separate tokenizers are loaded and cached:
  ```python
  self.event_tok = AutoTokenizer.from_pretrained(event_dir, local_files_only=True)
  self.sent_tok = AutoTokenizer.from_pretrained(sentiment_dir, local_files_only=True)
  ```
* **Strict Integrity Handshake:** On startup, the engine validates:
  1. `num_labels == 5` on the event classifier with exact canonical label dictionary matching.
  2. Classification head weights are fully populated (no missing head keys).
  3. Sentiment model configuration declares `problem_type = "regression"`, `num_labels = 1`, and regression head metadata `Dropout -> Linear(hidden_size,1) -> tanh`.
* **Execution Mode:** Inference runs under `torch.inference_mode()` with head truncation at `max_length = 128` (configurable up to 512).

### Part 2: Feature Engineering & Pre-Publication Market Context (`features.py`)

* **Anti-Lookahead Safeguard:** The feature pipeline guarantees that only market data strictly preceding the news publication timestamp is incorporated:
  $$\text{Cutoff Date} = \text{Date}(\text{published\_at}) - 1 \text{ session}$$
* **Engineered Signals:**
  * **20-Day Annualized Volatility:** $\sigma_{20} = \sqrt{252} \cdot \sqrt{\frac{1}{19}\sum_{i=1}^{20} (r_i - \bar{r})^2}$
  * **20-Session Volume Ratio:** $\nu = \frac{V_t}{\frac{1}{20}\sum_{i=1}^{20} V_{t-i}}$
  * **Short & Medium Horizon Momentum:** $R_{5d} = \frac{P_t}{P_{t-5}} - 1$, $R_{20d} = \frac{P_t}{P_{t-20}} - 1$
  * **NLP Vector Injection:** Direct integration of the 5 event probabilities and continuous sentiment score.

### Part 3: Market Impact Severity Engine (`impact.py`)

* **Preprocessor Transformation:** Takes raw dictionary features and applies a pre-fitted scikit-learn `ColumnTransformer` (`StandardScaler` on continuous features, `OneHotEncoder` on sector keys).
* **XGBoost Scoring:** Evaluates trained decision trees to output the predicted abnormal return magnitude:
  $$\hat{y} = \max(\hat{R}_{\text{abnormal}}, 0.0)$$
* **Decile Quantile Mapping:** Maps continuous predictions to calibrated deciles $1 \dots 10$ using frozen quantile thresholds from [`configs/impact_decile_thresholds.json`](../../configs/impact_decile_thresholds.json):
  $$\text{Decile} = \min\left(10, \; \arg\max_k \{q_k \le \hat{y}\} + 1\right)$$

---

## 3. Input & Output Contracts

### Input: `AnalyzeRequest`
```json
{
  "text": "Tesla announces global price cuts across Model 3 and Model Y lineup amid inventory expansion.",
  "ticker": "TSLA",
  "published_at": "2026-04-10T11:00:00Z",
  "source": "financial_news"
}
```

### Output: `AnalyzeResponse`
```json
{
  "ticker": "TSLA",
  "published_at": "2026-04-10T11:00:00Z",
  "source": "financial_news",
  "sentiment": {
    "score": -0.2841,
    "label": "negative"
  },
  "event": {
    "class_name": "Product Launch",
    "confidence": 0.9412,
    "probabilities": {
      "Geopolitical": 0.0021,
      "Macroeconomic": 0.0114,
      "Credit Event": 0.0083,
      "Merger / Acquisition": 0.0370,
      "Product Launch": 0.9412
    }
  },
  "impact": {
    "score": 6,
    "predicted_realized_impact": 0.0248
  },
  "market_context": {
    "volatility_20d": 0.482,
    "volume_ratio_20d": 1.28,
    "trailing_return_5d": -0.031,
    "trailing_return_20d": -0.084,
    "sector": "Consumer Discretionary"
  },
  "warnings": []
}
```
