# Soochana AI — Five-Class Financial Event Classifier

This module contains the research, data pipelines, model architecture, training notebooks, and inference workflows for the **Soochana AI Five-Class Financial Event Classifier**.

---

## 1. System Architecture Overview

The Event Classifier transforms unstructured financial text (news headlines, company filings, press releases, social market discourse) into a calibrated 5-class probability distribution across institutional risk categories.

```text
====================================================================================================
                                TRAINING PIPELINE ARCHITECTURE
====================================================================================================

  [ 6 Upstream Data Sources ]
  - Twitter Financial News Topic
  - DescribeEvents / GDELT News
  - GillesJacobs / SentiVent
  - ZipLime / SEC 8-K Items 1.03 & 2.04
  - AgungPambudi / Corporate Events
  - MatthewChung74 / Catalyst Events
                │
                ▼ (26,172 Raw Candidate Records)
  ┌─────────────────────────────────────────────────────────────┐
  │ Stage 1: Data Gathering & Source Normalization              │
  │ • Schema harmonization (id, text, candidate_label, source)  │
  │ • Non-ASCII cleanup, HTML entity unescaping                 │
  └─────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Stage 2: Deduplication & Split Generation                   │
  │ • Exact SHA-256 deduplication on normalized lowercase text   │
  │ • Near-duplicate token Jaccard filtering (MinHash)          │
  │ • Source-skew & label ambiguity rejection                   │
  │ • Stratified 80 / 10 / 10 split (Seed = 42)                 │
  └─────────────────────────────┬───────────────────────────────┘
                                │ (16,576 Cleaned Records)
                                ├────────────────────────┬────────────────────────┐
                                ▼                        ▼                        ▼
                         Train (13,260)              Val (1,658)              Test (1,658)
                                │                        │                        │
  ┌─────────────────────────────┴────────────────────────┴────────────────────────┴───┐
  │ Stage 3: Neural Model Architecture & Supervised Fine-Tuning                       │
  │                                                                                   │
  │   Input Text ───> [ FinBERT Tokenizer (Vocab: 30,522, Max Length: 512) ]          │
  │                                     │                                             │
  │                                     ▼                                             │
  │                    [ Input IDs, Attention Masks ]                                 │
  │                                     │                                             │
  │                                     ▼                                             │
  │             [ Pre-Trained FinBERT Transformer Encoder (12 Layers, 768-Dim) ]       │
  │                                     │                                             │
  │                                     ▼                                             │
  │                        [CLS] Pooled Hidden State h_CLS ∈ ℝ^768                    │
  │                                     │                                             │
  │                                     ▼                                             │
  │                       Dropout Layer (p = 0.10)                                    │
  │                                     │                                             │
  │                                     ▼                                             │
  │                 Dense Classification Head (Linear: 768 ──> 5)                     │
  │                                     │                                             │
  │                                     ▼                                             │
  │                           Raw Logits z ∈ ℝ^5                                      │
  │                                     │                                             │
  │                                     ▼                                             │
  │               Weighted Cross-Entropy Loss / WeightedRandomSampler                 │
  │                      Optimizer: AdamW (lr = 2e-5, wd = 0.01)                      │
  │                       Metric Monitor: Validation Macro-F1                         │
  └─────────────────────────────────────┬─────────────────────────────────────────────┘
                                        │
                                        ▼ (Best Checkpoint: Step 415, Epoch 3.0)
                        [ finbert_event_classifier_5class/ ]
                        (safetensors, config.json, tokenizer)

====================================================================================================
                                INFERENCE PIPELINE ARCHITECTURE
====================================================================================================

  Raw News / Post String: "Fed signals 50 bps interest rate hike amid persistent inflation"
                │
                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Tokenization & Tensor Assembly (`app/engine/nlp.py`)        │
  │ • Head truncation (max_length = 128 / 512)                  │
  │ • Dynamic batch padding & attention mask creation           │
  │ • Device allocation: torch.device('cuda' or 'cpu')          │
  └─────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ PyTorch Inference Mode (`torch.inference_mode()`)           │
  │ • Forward pass through FinBERT backbone                     │
  │ • Linear classification head projection: ℝ^768 ──> ℝ^5      │
  └─────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Softmax Calibration & Output Structuring                    │
  │                 exp(z_k)                                    │
  │   P(Event = k) = ──────────                                 │
  │                 ∑_j exp(z_j)                                │
  │                                                             │
  │ • Selected Class: argmax_k P(Event = k)                     │
  │ • Confidence Score: max_k P(Event = k)                      │
  │ • Strict verification: ∑ P(k) = 1.0, all values finite      │
  └─────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
  Output JSON:
  {
    "class_name": "Macroeconomic",
    "confidence": 0.9984,
    "probabilities": {
      "Geopolitical": 0.0004,
      "Macroeconomic": 0.9984,
      "Credit Event": 0.0005,
      "Merger / Acquisition": 0.0003,
      "Product Launch": 0.0004
    }
  }
```

---

## 2. Five-Class Taxonomy Definition

The classifier enforces an immutable contract defined in [`configs/event_labels.json`](../../configs/event_labels.json):

| Class ID | Canonical Label | Definition & Scope | Example Headlines |
| :---: | :--- | :--- | :--- |
| **0** | **Geopolitical** | International conflicts, trade sanctions, tariffs, regulatory embargoes, cross-border diplomatic tensions. | *"US expands tech export restrictions and semiconductor tariffs on Asian suppliers"* |
| **1** | **Macroeconomic** | Central bank rate decisions, CPI/PPI inflation prints, GDP reports, employment data, yield curve shifts. | *"Federal Reserve holds benchmark rate steady at 5.25%-5.50% following inflation cooling"* |
| **2** | **Credit Event** | Corporate debt defaults, SEC 8-K distress filings, credit rating downgrades, liquidity crunches, debt restructuring. | *"Moody's downgrades commercial real estate REIT issuer to B3 citing debt covenant breaches"* |
| **3** | **Merger / Acquisition** | Corporate buyouts, hostile takeovers, divestitures, joint ventures, FTC antitrust merger reviews. | *"Broadcom secures European antitrust approval for $61B acquisition of VMware"* |
| **4** | **Product Launch** | New commercial product announcements, software platform releases, FDA drug approvals, patent grants. | *"Apple unveils M3-powered MacBook Pro family and next-generation neural silicon"* |

---

## 3. Pipeline Stages & Source Code Artifacts

The workflow consists of four modular stages located in `notebooks/` and exported to `scripts/`:

| Stage | Notebook File | Script Export | Inputs | Primary Outputs |
| :--- | :--- | :--- | :--- | :--- |
| **1. Gathering** | `notebooks/01_data_gathering.ipynb` | `scripts/01_data_gathering.py` | 6 Hugging Face repositories | `final_raw_event_dataset.csv`, `extraction_report.json` |
| **2. Preprocessing** | `notebooks/02_preprocessing.ipynb` | `scripts/02_preprocessing.py` | Raw dataset | `event_train.parquet`, `event_validation.parquet`, `event_test.parquet` |
| **3. Synthetic (Opt)**| `notebooks/03_synthetic_generation_optional.ipynb` | `scripts/03_synthetic_generation_optional.py` | Training split + Groq API | Optional minority class augmentation (kept separate) |
| **4. FinBERT Training**| `notebooks/04_finbert_training_real_only.ipynb` | `scripts/04_finbert_training_real_only.py` | Parquet train/val/test splits | Fine-tuned FinBERT checkpoint, confusion matrices, evaluation metrics |

---

## 4. Model Training Configuration & Hyperparameters

Extracted from `notebooks/04_finbert_training_real_only.ipynb`:

* **Backbone Initializer:** `nickmuchi/finbert-tone-finetuned-finance-topic-classification` (20-class finance BERT adapted with `ignore_mismatched_sizes=True` to initialize a 5-class head).
* **Tokenization:** WordPiece vocabulary (30,522 tokens), dynamic padding, max length 512 tokens with head truncation.
* **Batch Size:** 16 per device (train), 32 per device (evaluation), gradient accumulation steps = 1.
* **Optimizer:** AdamW (`betas=(0.9, 0.999)`, `eps=1e-8`).
* **Learning Rate:** $2 \times 10^{-5}$ with linear warmup over the first 10% of training steps.
* **Weight Decay:** $0.01$, gradient clipping norm $1.0$.
* **Minority Class Handling:** PyTorch `WeightedRandomSampler` utilizing inverse class frequency with power exponent $\alpha = 0.5$:
  $$w_c = \left(\frac{1}{N_c}\right)^{0.5}$$
* **Early Stopping:** Monitored on validation Macro-F1 with patience of 3 epochs. Best checkpoint selected at step 415 (Epoch 3.0).

---

## 5. Recorded Experimental Results

*Data recorded in `recorded_results/notebook_evidence.json`:*

| Metric | Training Split | Validation Split | Test Split (Held-Out) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 99.82% | 99.34% | **99.46%** |
| **Macro-F1** | 0.9941 | 0.9880 | **0.9798** |
| **Weighted-F1** | 0.9981 | 0.9935 | **0.9946** |

### Per-Class Test Performance (1,658 Test Samples)

| Class | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Geopolitical** | 1,160 | 0.998 | 0.997 | 0.998 |
| **Macroeconomic** | 249 | 0.988 | 0.984 | 0.986 |
| **Credit Event** | 12 | 1.000 | 0.917 | 0.957 |
| **Merger / Acquisition** | 75 | 0.974 | 0.987 | 0.980 |
| **Product Launch** | 162 | 0.988 | 0.994 | 0.991 |

---

## 6. How to Reproduce & Run Locally

### Option A: Download Trained Weights from Hugging Face Hub (Production Path)

The trained weights are published on Hugging Face Hub:
```bash
python -m scripts.fetch_models
python -m scripts.check_models
```

### Option B: Retrain from Notebooks

```bash
# 1. Install dependencies
pip install -r training/event_classifier/requirements.txt

# 2. Run sequential pipeline stages via training runner
python training/run.py check
python training/run.py event-gather
python training/run.py event-preprocess
python training/run.py event-train
```
