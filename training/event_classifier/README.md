# RiskPulse AI — Five-Class Event Classifier Training (Original Source)

This folder contains the **source code extracted from four supplied Kaggle notebooks** for the RiskPulse AI event classifier. The notebooks form a documented collection, preprocessing, optional synthetic augmentation, fine-tuning and evaluation workflow.

**Important:** The source has been preserved, but the training was **not rerun** when preparing this package. The raw datasets, preprocessed Parquet files, model checkpoints and fine-tuned weights are **not included**. Consequently, this is a source-code reproduction package, **not yet a fully self-contained reproducible trained-model release**.

## Pipeline and exact source files

| Stage | Notebook | Code export | Key output |
| --- | --- | --- | --- |
| 1. Real data gathering | `notebooks/01_data_gathering.ipynb` | `scripts/01_data_gathering.py` | `final_raw_event_dataset.csv` / `.parquet`, `extraction_report.json` |
| 2. Source-aware preprocessing | `notebooks/02_preprocessing.ipynb` | `scripts/02_preprocessing.py` | `event_train`, `event_validation`, `event_test` (CSV + Parquet) |
| 3. Synthetic generation **optional** | `notebooks/03_synthetic_generation_optional.ipynb` | `scripts/03_synthetic_generation_optional.py` | `event_train_augmented.parquet`, checkpoint and reports |
| 4. Five-class FinBERT training | `notebooks/04_finbert_training_real_only.ipynb` | `scripts/04_finbert_training_real_only.py` | `finbert_event_classifier_5class/`, test predictions, metrics, confusion matrices |

The `.ipynb` copies retain **all original markdown and source code cells**, with execution outputs and nonessential metadata stripped for safer publication. `scripts/*.py` are faithful code exports whose notebook-style `%pip` and `!pip` installation cells were commented out. **The notebooks are the recommended execution format.** Check `source_manifest.json` for hashes proving that the source-cell text was preserved.

## Class taxonomy — immutable contract

The original notebooks and RiskPulse runtime both expect:

```json
{
  "0": "Geopolitical",
  "1": "Macroeconomic",
  "2": "Credit Event",
  "3": "Merger / Acquisition",
  "4": "Product Launch"
}
```

Do **not** change label indices to make a new model look compatible. Production inference in `app/engine/nlp.py` separately checks `num_labels=5` and exact label names.

## Original real-data sources

The gathering notebook requests six public dataset collections on Hugging Face:

- `zeroshot/twitter-financial-news-topic`
- `DescribeEvents/gdelt_news_events`
- `GillesJacobs/sentivent`
- `ZipLime/sec-8k-events` (SEC 8-K credit-related items 1.03 and 2.04)
- `agungpambudi/corporate-event-detection`
- `matthewchung74/catalyst` (restricted by the notebook to suitable previously labeled entries)

Source names are from the original notebooks. **Review each upstream dataset's access conditions, licensing, version and availability before redistributing it.** Dataset revisions were not pinned in the supplied notebooks; future downloads may yield different data. Gathering can continue after individual source failures, so always inspect `extraction_report.json` before interpreting counts.

The recorded gathering run produced **26,172 raw candidate rows**. Preprocessing used normalization only for duplicate matching, exact and approximate deduplication, source-sensitive label checks, candidate rejection/manual review, and stratified **80/10/10 split, seed 42**. The recorded real-only cleaned total was **16,576 rows**: 13,260 train, 1,658 validation, 1,658 test.

## How to reproduce the original workflow

The original workflow was written for **Kaggle notebooks**. To reproduce without altering the method:

1. Use a Python-enabled Kaggle notebook environment with internet access, and install the libraries in `requirements.txt`. Record your **actual package versions** and GPU; the dependency list here is not a pinned lockfile.
2. Run `notebooks/01_data_gathering.ipynb` to download and aggregate permitted datasets. Save its output artifacts from `/kaggle/working/`. Inspect source-extraction failures and licensing before proceeding.
3. Attach those outputs as an input dataset and run `notebooks/02_preprocessing.ipynb`. Inspect the manual review, near-duplicate, source-skew and split-audit reports. Save `event_train.parquet`, `event_validation.parquet`, and `event_test.parquet`.
4. **Reproduce the recorded trained checkpoint:** attach the three **real-only** split files and run `notebooks/04_finbert_training_real_only.ipynb`. It automatically discovers `event_train`, `event_validation`, and `event_test` under `/kaggle/input`. It does **not** load augmented training data.
5. Optionally run `notebooks/03_synthetic_generation_optional.ipynb` after preprocessing, with your own Groq credentials supplied via Kaggle Secrets / environment. This generates additional **training-only** candidates, and keeps real validation and test splits frozen. **It is not part of the recorded classifier training run.** Do not describe that run as synthetic-augmented.
6. For real inference, copy the **correct saved fine-tuned checkpoint** and tokenizer files into the configured event-model location and run `python -m scripts.check_models` from the repository root. Verify checkpoint provenance and label mapping before use.

The exports in `scripts/` are suitable for inspecting, diffing and syntax checking, but may still require Kaggle inputs and notebook-specific display helpers. They have **not** been validated as independent CLI entry points.

## Recorded training configuration

From `04_finbert_training_real_only.ipynb`:

- Training base ID: `nickmuchi/finbert-tone-finetuned-finance-topic-classification` (20-class checkpoint loaded with a newly initialized **five-way** classification head using `ignore_mismatched_sizes=True`).
- Task: five-class single-label classification, softmax probabilities.
- Token length: 512; head truncation; dynamic padding.
- Seed: 42; 5 requested epochs; training selected `checkpoint-415` and output reports epoch **3.0**.
- Batch sizes: 16 train, 32 evaluation; gradient accumulation 1.
- Learning rate: `2e-5`, weight decay `0.01`, warmup ratio `0.10`, max gradient norm `1.0`.
- Minority sampling: `WeightedRandomSampler` with inverse-class-frequency power `0.5`; ordinary cross-entropy loss, **not** extra heavy loss weights.
- Selection criterion: validation macro-F1, saved per epoch, best model loaded; early-stopping callback configured (the notebook also shows a metric-name warning during a later validation call—verify exact effective stopping behavior when rerunning).

**Recorded test results (not reproduced during packaging):** test accuracy **0.99457**, test macro-F1 **0.97976**, validation macro-F1 **0.98796**. Test classes had supports of 1,160 Geopolitical, 249 Macroeconomic, **12 Credit Event**, 75 M&A, and 162 Product Launch. The test split has **only 12 Credit Event examples**, which makes that per-class estimate unstable.

See `recorded_results/notebook_evidence.json` for measured notebook-output values and limitations.

## Critical integrity and provenance caveats

1. **Checkpoint provenance mismatch:** The uploaded fine-tuning notebook starts from `nickmuchi/finbert-tone-finetuned-finance-topic-classification`, whereas the current GitHub runtime event model config references `yiyanghkust/finbert-tone`. This may be caused by separate artifact generation/configuration changes. **Do not assert that this notebook produced the deployed classifier** without checking the original exported `config.json`, saved model weights, training report, and hashes. A config match alone is not enough.
2. **Recorded training was real-only:** `04_finbert_training_real_only.ipynb` resolves `event_train`, not `event_train_augmented`. The optional synthetic run accepted only 45 Credit Event examples before stopping due to a Groq request/token-limit error; its declared target counts were not met. That run did **not** produce the documented fine-tuned checkpoint.
3. **Potential source-style shortcut:** Recorded preprocessing flagged that Geopolitical examples were entirely from machine-generated GDELT styles in the clean set and Credit Event was heavily concentrated in SEC filings. Very high in-distribution test performance does not establish out-of-source robustness. Evaluate separately on new sources and live financial headlines.
4. **Dataset license and version:** The original source datasets are not bundled or revision-pinned here. Upstream revisions or unavailable sources can change reproduction results. Original generated social/news text is not automatically safe to redistribute without checking rights.
5. **No complete artifact supply chain:** To claim full trained-model reproducibility, additionally supply immutable dataset revisions or hashes, exported split files if redistributable, exact package versions, the correct fine-tuned model artifact and checksum, and independent execution evidence.

## Publishing safety

- Never upload Groq/Hugging Face credentials, `.env` files, Kaggle secrets or private model repositories.
- Only notebook **source** is included; execution outputs (which can contain provider error details, raw texts, or user environment data) were omitted.
- Model checkpoints, data copies and generated CSV/Parquet files are excluded through `.gitignore` by default; publish permitted datasets/artifacts separately with a documented manifest when legally and technically appropriate.
- The application requires its original fine-tuned model weights for real inference. `SKIP_MODEL_LOAD=true` disables real ML inference; it does not reproduce trained models.

## Status

**Prepared and syntax-checked:** source-only notebooks and 4 Python exports, dataset/task manifest, recorded-run report and dependencies.

**Not yet established:** reproducibility of the exact deployed checkpoint, training data download equivalence, notebook rerun, and verified end-to-end retraining. Do not claim these items as completed.
