# RiskPulse AI — Continuous Sentiment Regressor Training (Original Source)

This directory contains the **source code extracted from the original Kaggle notebooks** for the RiskPulse AI continuous financial sentiment regressor. The workflow implements end-to-end data preparation, cleaning, deduplication, fine-tuning, and evaluation on the FiQA financial sentiment corpus.

**Important:** The source code and notebook structures have been preserved, but training was **not rerun** during repository packaging. The raw upstream cache, intermediate Parquet splits, and fine-tuned checkpoints are **not bundled in this Git repository** (weights are published on Hugging Face at [`shanmukhdatta/riskpulse-sentiment-regressor`](https://huggingface.co/shanmukhdatta/riskpulse-sentiment-regressor)). This package documents the authentic training methodology.

---

## 1. Pipeline and Exact Source Files

| Stage | Notebook | Code Export | Key Outputs |
| :--- | :--- | :--- | :--- |
| **1. FiQA Data Preparation** | `notebooks/01_fiqa_data_preparation.ipynb` | `scripts/01_fiqa_data_preparation.py` | `fiqa_train.parquet`, `fiqa_validation.parquet`, `fiqa_test.parquet`, `technical_report.json` |
| **2. FinBERT Regression Training** | `notebooks/02_finbert_regression_training.ipynb` | `scripts/02_finbert_regression_training.py` | `finbert_sentiment_regression/` (safetensors checkpoint), training curves, metrics report |
| **Architecture Reference** | — | `architecture/modeling_finbert_sentiment_regressor.py` | Drop-in PyTorch regression head module (`FinBertSentimentRegressor`) |

The `.ipynb` notebooks retain **all original markdown and code cells**, with execution outputs stripped for clean publication. The `scripts/*.py` files are faithful command-line exports with notebook-specific IPython directives commented out.

---

## 2. Model Architecture & Continuous Formulation

Unlike conventional 3-class sentiment classifiers (positive, neutral, negative), RiskPulse AI formulates financial sentiment as a **continuous scalar regression task**:

$$s \in [-1.0, +1.0]$$

* **Base Foundation:** `ProsusAI/finbert` (domain-specific financial BERT pre-trained on corporate communications).
* **Regression Head (`architecture/modeling_finbert_sentiment_regressor.py`):**
  * Takes the BERT `[CLS]` token representation $\mathbf{h}_{\text{CLS}} \in \mathbb{R}^{768}$.
  * Applies dropout ($p = 0.10$).
  * Passes through a dense linear layer $W \in \mathbb{R}^{768 \times 1}$ with bias $b$.
  * Applies a `Tanh` activation function to strictly bound outputs in $[-1.0, +1.0]$:
    $$\hat{s} = \tanh(\mathbf{w}^T \mathbf{h}_{\text{CLS}} + b)$$
* **Loss Function:** Mean Squared Error (MSE) against continuous ground-truth sentiment labels:
  $$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (s_i - \hat{s}_i)^2$$

---

## 3. Training Data & Preprocessing

The preparation pipeline consumes the **FiQA (Financial Opinion Mining and Question Answering)** sentiment challenge corpus:

* **Hugging Face Candidates:**
  * `TheFinAI/fiqa-sentiment-classification`
  * `ChanceFocus/fiqa-sentiment-classification`
  * `vikrantburman/fiqa-sentiment-classification`
* **Data Sanitization & Cleaning Rules:**
  1. Removal of null, non-string, and empty text rows.
  2. Coercion and validation of sentiment scores to strict numerical bounds $[-1.0, 1.0]$. Non-finite values ($\pm\infty$, NaN) are purged.
  3. SHA-256 hash generation of normalized lowercase text for exact and near-duplicate detection.
  4. Cross-split duplicate resolution: prevents identical headlines from co-occurring in both train and test splits.
  5. Exported to partitioned Parquet files (`fiqa_train.parquet`, `fiqa_validation.parquet`, `fiqa_test.parquet`).

---

## 4. How to Reproduce the Training Workflow

The training workflow was developed for **Kaggle GPU environments** (Tesla P100 / T4). To reproduce without altering methodology:

1. **Environment Setup:**
   Install dependencies from `training/sentiment/requirements.txt`:
   ```bash
   pip install -r training/sentiment/requirements.txt
   ```
2. **Execute Stage 1 (Data Preparation):**
   Open and execute `notebooks/01_fiqa_data_preparation.ipynb` (or run `scripts/01_fiqa_data_preparation.py`).
   Verify generated splits in `/kaggle/working/` or the configured `RISK_PULSE_WORKDIR`.
3. **Execute Stage 2 (FinBERT Fine-Tuning):**
   Open and execute `notebooks/02_finbert_regression_training.ipynb` (or run `scripts/02_finbert_regression_training.py`).
   The script loads preprocessed splits, initializes `ProsusAI/finbert`, trains the regression head, evaluates each epoch, and saves the best model checkpoint based on validation MSE.
4. **Export Trained Artifacts:**
   Export the final model directory containing `model.safetensors`, `config.json`, `modeling_finbert_sentiment_regressor.py`, and tokenizer files into `models/sentiment_regressor/`.

---

## 5. Recorded Training Hyperparameters

From `notebooks/02_finbert_regression_training.ipynb`:

* **Base Model Initializer:** `ProsusAI/finbert`
* **Max Token Sequence Length:** 512 tokens (head truncation, dynamic padding)
* **Maximum Epochs:** 12
* **Early Stopping:** Patience of 3 epochs monitoring validation MSE
* **Batch Sizes:** 16 (training), 32 (validation / evaluation)
* **Gradient Accumulation:** 1
* **Optimizer & Learning Rate:** AdamW with $\text{lr} = 2 \times 10^{-5}$
* **Weight Decay:** 0.01
* **Learning Rate Schedule:** Linear warmup with `warmup_ratio = 0.10`
* **Gradient Clipping:** Maximum gradient norm $1.0$
* **Head Dropout:** 0.10
* **Evaluation Metrics:**
  * Mean Squared Error (MSE)
  * Root Mean Squared Error (RMSE)
  * Mean Absolute Error (MAE)
  * Pearson Correlation Coefficient ($r$)
  * Spearman Rank Correlation ($\rho$)

---

## 6. Critical Integrity & Reproduction Notes

1. **Continuous Score Interpretation:**
   A score of $+1.0$ represents maximum positive financial tone, $0.0$ neutral market commentary, and $-1.0$ severe negative distress. Downstream components (Module A & B) map this score with event category multipliers and XGBoost impact deciles.
2. **Hugging Face Model Availability:**
   To bypass retraining, the trained checkpoint weights are available on Hugging Face Hub:
   [`shanmukhdatta/riskpulse-sentiment-regressor`](https://huggingface.co/shanmukhdatta/riskpulse-sentiment-regressor)
   and can be fetched automatically via:
   ```bash
   python -m scripts.fetch_models
   ```
3. **Publishing Safety:**
   Never commit private API tokens, raw external datasets, or local cache checkpoints directly to Git. Model checkpoints are excluded via `.gitignore`.
