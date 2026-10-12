# Soochana AI — Model Training & Research Architecture

This directory houses the end-to-end machine learning research, data processing pipelines, and training implementations for the three cooperating AI models in **Soochana AI**.

---

## 1. Multi-Model Architecture Overview

Soochana AI decomposes unstructured financial text intelligence into three specialized cooperating models:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 UNSTRUCTURED FINANCIAL TEXT                            │
│                  (News Headlines, SEC 8-K Disclosures, Social Market Feeds)            │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
       ┌───────────────────────────┐                 ┌───────────────────────────┐
       │   1. EVENT CLASSIFIER     │                 │  2. SENTIMENT REGRESSOR   │
       │    FinBERT 5-Way Softmax  │                 │    FinBERT Tanh Regressor │
       │  ─────────────────────────│                 │  ─────────────────────────│
       │  • Geopolitical           │                 │  • Continuous Polarity:   │
       │  • Macroeconomic          │                 │    s ∈ [-1.0, +1.0]       │
       │  • Credit Event           │                 │  • Tanh-activated Linear  │
       │  • Merger / Acquisition   │                 │    head on CLS embedding  │
       │  • Product Launch         │                 │  • Supervised on FiQA     │
       └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                     │                                             │
                     │  p(Event) Vector                            │  Sentiment s
                     └──────────────────────┬──────────────────────┘
                                            │
                                            ▼
                             ┌─────────────────────────────┐
                             │  PRE-PUBLICATION MARKET CONTEXT│
                             │  • 20-Day Rolling Vol (σ_20)│
                             │  • Volume Ratio (V / V̄_20)  │
                             │  • Trailing 5d & 20d Returns│
                             │  • One-Hot Sector Encoding  │
                             └──────────────┬──────────────┘
                                            │
                                            ▼
                             ┌─────────────────────────────┐
                             │   3. MARKET IMPACT ENGINE   │
                             │      XGBoost Regressor      │
                             │  ───────────────────────────│
                             │  • Continuous |R_abnormal|  │
                             │  • Calibrated Deciles (1-10)│
                             └──────────────┬──────────────┘
                                            │
                                            ▼
                             ┌─────────────────────────────┐
                             │    STRUCTURED RISK SIGNAL   │
                             │ (P_event, s, Impact Decile) │
                             └─────────────────────────────┘
```

---

## 2. Specialized Model Deep Dives

Each model contains its own dedicated technical specification, training scripts, notebook source, and reproducible execution flow:

1. [**`training/event_classifier/` — 5-Class Financial Event Classifier**](event_classifier/README.md)
   * **Backbone:** Domain FinBERT initialized from `nickmuchi/finbert-tone-finetuned-finance-topic-classification`.
   * **Corpus:** 26,172 raw candidate records cleaned to 16,576 real financial samples across 6 institutional sources.
   * **Taxonomy:** Geopolitical, Macroeconomic, Credit Event, Merger / Acquisition, Product Launch.
   * **Performance:** 99.46% test accuracy, 0.9798 test Macro-F1.

2. [**`training/sentiment/` — Continuous Financial Sentiment Regressor**](sentiment/README.md)
   * **Backbone:** `ProsusAI/finbert` with custom linear projection and hyperbolic tangent activation.
   * **Formulation:** Continuous scalar regression $s \in [-1.0, +1.0]$ trained on the FiQA financial sentiment challenge dataset.
   * **Objective:** Direct MSE loss optimization with AdamW and linear warmup.

3. [**`training/impact_engine/` — Market Impact Severity Engine**](impact_engine/README.md)
   * **Backbone:** Gradient boosted decision trees (`xgboost.XGBRegressor`) evaluated against LightGBM.
   * **Corpus:** 27,460 observations with pre-publication market features and frozen NLP outputs.
   * **Calibration:** Quantile-thresholded decile bins ($1 \dots 10$) representing abnormal volatility reaction scale.

---

## 3. Quickstart & Local Pipeline Execution

### Model Weights on Hugging Face Hub

Pretrained weights are hosted on Hugging Face Hub and can be downloaded into `models/`:
```bash
python -m scripts.fetch_models
python -m scripts.check_models
```

### Reproducing Training from Scratch

```bash
# Install environment dependencies
pip install -r training/event_classifier/requirements.txt
pip install -r training/sentiment/requirements.txt
pip install -r training/impact_engine/requirements.txt

# Run Event Classifier Pipeline
python training/run.py check
python training/run.py event-gather
python training/run.py event-preprocess
python training/run.py event-train

# Run Sentiment Regressor Pipeline
python training/run.py sentiment-prepare
python training/run.py sentiment-train
```
