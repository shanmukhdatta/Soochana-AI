# Soochana AI — Continuous Financial Sentiment Regressor

This module contains the research, data pipelines, model architecture, training notebooks, and inference workflows for the **Soochana AI Continuous Financial Sentiment Regressor**.

---

## 1. System Architecture Overview

Unlike standard 3-class classification models (positive/negative/neutral), Soochana AI models sentiment as a continuous scalar regression task:

$$\hat{s} \in [-1.0, +1.0]$$

where $-1.0$ represents severe distress or pessimistic market outlook, $0.0$ neutral sentiment, and $+1.0$ strong expansionary optimism.

```text
====================================================================================================
                                TRAINING PIPELINE ARCHITECTURE
====================================================================================================

  [ Upstream Corpus: FiQA Financial Opinion Mining ]
  - TheFinAI / fiqa-sentiment-classification
  - ChanceFocus / fiqa-sentiment-classification
  - vikrantburman / fiqa-sentiment-classification
                │
                ▼ (Raw Financial Sentiment Records)
  ┌─────────────────────────────────────────────────────────────┐
  │ Stage 1: Data Sanitization & Cleaning Rules                 │
  │ • Removal of null, non-string, or empty text records        │
  │ • Strict boundary validation: enforce s ∈ [-1.0, +1.0]      │
  │ • Purge non-finite values (±∞, NaN)                         │
  │ • SHA-256 hash generation of normalized lowercase text      │
  │ • Cross-split duplicate resolution (no headline leakage)    │
  └─────────────────────────────┬───────────────────────────────┘
                                │
                                ▼ Partitioned Parquet Splits
                         Train (70%) │ Val (15%) │ Test (15%)
                                     │
  ┌──────────────────────────────────┴────────────────────────────────────────────────────────┐
  │ Stage 2: FinBERT Regression Head Architecture & Fine-Tuning                               │
  │                                                                                           │
  │   Input Text ───> [ WordPiece Tokenizer (ProsusAI/finbert, Max Length: 512) ]             │
  │                                     │                                                     │
  │                                     ▼                                                     │
  │                    [ Input IDs, Attention Masks ]                                         │
  │                                     │                                                     │
  │                                     ▼                                                     │
  │             [ Pre-Trained FinBERT Transformer Backbone (12 Layers, 768-Dim) ]             │
  │                                     │                                                     │
  │                                     ▼                                                     │
  │                        [CLS] Pooled Hidden State h_CLS ∈ ℝ^768                            │
  │                                     │                                                     │
  │                                     ▼                                                     │
  │                       Dropout Layer (p = 0.10)                                            │
  │                                     │                                                     │
  │                                     ▼                                                     │
  │                 Linear Dense Projection (Linear: 768 ──> 1)                               │
  │                                     │                                                     │
  │                                     ▼                                                     │
  │                    Hyperbolic Tangent Activation: tanh(·)                                 │
  │                         ŝ = tanh(w^T · h_CLS + b) ∈ [-1.0, +1.0]                          │
  │                                     │                                                     │
  │                                     ▼                                                     │
  │                        Mean Squared Error Loss (MSE):                                     │
  │                          L_MSE = 1/N ∑ (s_i - ŝ_i)^2                                      │
  │                                                                                           │
  │   Optimizer: AdamW (lr = 2e-5, wd = 0.01, linear warmup = 10%)                            │
  │   Early Stopping: Patience = 3 epochs on Validation MSE                                   │
  └─────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                        │
                                        ▼ Best Checkpoint Saved
                        [ finbert_sentiment_regression/ ]
                        (model.safetensors, config.json, tokenizer)

====================================================================================================
                                INFERENCE PIPELINE ARCHITECTURE
====================================================================================================

  Raw Input Headline / Market Commentary
                │
                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Tokenization & Device Loading (`app/engine/nlp.py`)         │
  │ • Tokenizer: AutoTokenizer.from_pretrained(sentiment_dir)   │
  │ • Truncation: max_length = 128 / 512                        │
  │ • Attention tensor dispatched to torch.device('cuda'/'cpu') │
  └─────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Forward Pass (`FinBertSentimentRegressor`)                  │
  │ • CLS pooled embedding extracted                            │
  │ • Linear projection + Tanh evaluated in inference mode      │
  │ • Strict verification: finite check, bounded in [-1.0, +1.0]│
  └─────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Categorical Polarity Thresholding                           │
  │                                                             │
  │   polarity = "positive" if score > +0.15                    │
  │              "negative" if score < -0.15                    │
  │              "neutral"  otherwise                           │
  └─────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
  Output JSON:
  {
    "score": -0.6842,
    "label": "negative"
  }
```

---

## 2. Mathematical Formulation & Architecture

The regression head is explicitly implemented in [`architecture/modeling_finbert_sentiment_regressor.py`](architecture/modeling_finbert_sentiment_regressor.py) as `FinBertSentimentRegressor`:

```python
class FinBertSentimentRegressor(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.regressor = nn.Sequential(
            nn.Dropout(p=getattr(config, "classifier_dropout", 0.1)),
            nn.Linear(config.hidden_size, 1),
            nn.Tanh()
        )
```

1. **Contextual Token Representation:**
   $$\mathbf{H} = \text{FinBERT}(\mathbf{x}) \in \mathbb{R}^{L \times 768}$$
2. **First Token Extraction:**
   $$\mathbf{h}_{\text{CLS}} = \mathbf{H}[0] \in \mathbb{R}^{768}$$
3. **Dropout Regularization:**
   $$\mathbf{h}_{\text{drop}} = \text{Dropout}(\mathbf{h}_{\text{CLS}}, p=0.10)$$
4. **Linear Mapping:**
   $$z = \mathbf{w}^T \mathbf{h}_{\text{drop}} + b, \quad \mathbf{w} \in \mathbb{R}^{768}, \; b \in \mathbb{R}$$
5. **Bounded Tanh Activation:**
   $$\hat{s} = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \in (-1.0, +1.0)$$

---

## 3. Training Hyperparameters

Extracted from `notebooks/02_finbert_regression_training.ipynb`:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Base Checkpoint** | `ProsusAI/finbert` | Pretrained on corporate financial filings and news |
| **Loss Function** | Mean Squared Error (MSE) | Direct continuous gradient penalty |
| **Learning Rate** | $2 \times 10^{-5}$ | AdamW optimizer with $\beta_1 = 0.9, \beta_2 = 0.999$ |
| **Warmup Ratio** | 10% of total steps | Linear learning rate warmup |
| **Weight Decay** | 0.01 | $L_2$ regularization on non-bias parameters |
| **Batch Size** | 16 (train), 32 (eval) | Dynamically padded batches |
| **Max Epochs** | 12 | Early stopping patience = 3 epochs on Validation MSE |
| **Grad Clipping** | Max norm 1.0 | Prevents exploding gradient spikes |

---

## 4. Evaluation Metrics & Performance

The model is evaluated using both regression error and rank correlation metrics:

* **Mean Squared Error (MSE):** Measures magnitude of squared prediction error.
* **Root Mean Squared Error (RMSE):** Error in natural sentiment units.
* **Mean Absolute Error (MAE):** Average absolute deviation from ground truth.
* **Pearson Correlation ($r$):** Linear alignment between predicted and true polarity.
* **Spearman Rank Correlation ($\rho$):** Monotonic ranking alignment of financial headlines.

---

## 5. How to Reproduce & Run Locally

### Option A: Download Trained Weights from Hugging Face Hub (Production Path)

The trained regression checkpoint is published on Hugging Face Hub:
```bash
python -m scripts.fetch_models
python -m scripts.check_models
```

### Option B: Retrain from Notebooks

```bash
# 1. Install dependencies
pip install -r training/sentiment/requirements.txt

# 2. Run data preparation and training
python training/run.py sentiment-prepare
python training/run.py sentiment-train
```
