"""Faithful code-cell export of event-classification-finberttraining.ipynb.
Execution order follows original notebook. Notebook/Kaggle inputs are required.
This file is a SOURCE EXPORT, not a verified local standalone retraining command.
Do not claim this script was executed during packaging.
"""



# ===== NOTEBOOK CELL 2 =====
# ORIGINAL NOTEBOOK SETUP (install via requirements.txt): %pip -q install -U "transformers>=4.46,<5" "datasets>=3,<4" "accelerate>=1,<2" "scikit-learn>=1.4" pyarrow safetensors


# ===== NOTEBOOK CELL 4 =====
import os
import gc
import json
import math
import random
import shutil
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print("Device:", DEVICE)
print("Hardware:", GPU_NAME)
print("PyTorch:", torch.__version__)


# ===== NOTEBOOK CELL 5 =====
# Hugging Face login from Kaggle Secret

from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")

login(token=HF_TOKEN)

print("Hugging Face login successful.")


# ===== NOTEBOOK CELL 7 =====
# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

MODEL_NAME = "nickmuchi/finbert-tone-finetuned-finance-topic-classification"

LABEL2ID = {
    "Geopolitical": 0,
    "Macroeconomic": 1,
    "Credit Event": 2,
    "Merger / Acquisition": 3,
    "Product Launch": 4,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

MAX_LENGTH = 512

# Training settings
NUM_EPOCHS = 5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0

# Imbalance handling:
# 1.0 = full inverse-frequency sampling (very aggressive)
# 0.5 = inverse-sqrt sampling (recommended baseline)
# 0.0 = natural distribution
SAMPLER_POWER = 0.50

# Early stopping protects against overfitting, especially Credit Event.
EARLY_STOPPING_PATIENCE = 2

# Trainer will pick the best checkpoint by Macro-F1.
BEST_MODEL_METRIC = "macro_f1"

WORKDIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
INPUT_ROOT = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path.cwd()

MODEL_OUT = WORKDIR / "finbert_event_classifier_5class"
CHECKPOINT_OUT = WORKDIR / "finbert_training_checkpoints"

print("Model:", MODEL_NAME)
print("Max length:", MAX_LENGTH)
print("Epochs:", NUM_EPOCHS)
print("Sampler power:", SAMPLER_POWER)


# ===== NOTEBOOK CELL 9 =====
def find_file(root: Path, exact_names):
    hits = []
    if root.exists():
        for name in exact_names:
            hits.extend(root.rglob(name))
    hits = sorted(set(hits), key=lambda p: (p.suffix.lower() != ".parquet", str(p)))
    return hits

def resolve_split(stem):
    hits = find_file(INPUT_ROOT, [f"{stem}.parquet", f"{stem}.csv"])
    print(f"\nCandidates for {stem}:")
    for h in hits:
        print(" ", h)
    if not hits:
        raise FileNotFoundError(
            f"Could not find {stem}.parquet or {stem}.csv under {INPUT_ROOT}. "
            "Attach the complete output of the preprocessing notebook as a Kaggle Input."
        )
    chosen = hits[0]
    print("Selected:", chosen)
    return chosen

TRAIN_PATH = resolve_split("event_train")
VAL_PATH = resolve_split("event_validation")
TEST_PATH = resolve_split("event_test")

def load_table(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

train_df = load_table(TRAIN_PATH)
val_df = load_table(VAL_PATH)
test_df = load_table(TEST_PATH)

print("\nShapes")
print("train     :", train_df.shape)
print("validation:", val_df.shape)
print("test      :", test_df.shape)


# ===== NOTEBOOK CELL 11 =====
REQUIRED = {"text", "label", "label_id"}

def validate_split(df, split_name):
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{split_name} missing required columns: {sorted(missing)}")

    if df["text"].isna().any():
        raise ValueError(f"{split_name} contains null text.")

    empty = df["text"].astype(str).str.strip().eq("")
    if empty.any():
        raise ValueError(f"{split_name} contains {int(empty.sum())} empty texts.")

    bad_labels = set(df["label"].astype(str).unique()) - set(LABEL2ID)
    if bad_labels:
        raise ValueError(f"{split_name} contains unknown labels: {bad_labels}")

    pairs = df[["label", "label_id"]].drop_duplicates()
    for r in pairs.itertuples(index=False):
        expected = LABEL2ID[str(r.label)]
        if int(r.label_id) != expected:
            raise ValueError(
                f"{split_name}: label mapping mismatch for {r.label}: "
                f"got {r.label_id}, expected {expected}"
            )

validate_split(train_df, "train")
validate_split(val_df, "validation")
validate_split(test_df, "test")

def norm_text(x):
    return " ".join(str(x).lower().split())

train_norm = set(train_df["text"].astype(str).map(norm_text))
val_norm = set(val_df["text"].astype(str).map(norm_text))
test_norm = set(test_df["text"].astype(str).map(norm_text))

assert train_norm.isdisjoint(val_norm), "Exact text leakage between train and validation."
assert train_norm.isdisjoint(test_norm), "Exact text leakage between train and test."
assert val_norm.isdisjoint(test_norm), "Exact text leakage between validation and test."

print("Class counts:")
counts_table = pd.DataFrame({
    "train": train_df["label"].value_counts(),
    "validation": val_df["label"].value_counts(),
    "test": test_df["label"].value_counts(),
}).reindex(LABEL2ID.keys()).fillna(0).astype(int)

display(counts_table)
print("\nDataset validation: PASSED")


# ===== NOTEBOOK CELL 13 =====
ax = train_df["label"].value_counts().reindex(LABEL2ID.keys(), fill_value=0).plot(
    kind="bar", figsize=(9, 4)
)
ax.set_title("REAL training class distribution")
ax.set_xlabel("")
ax.set_ylabel("Rows")
plt.tight_layout()
plt.savefig(WORKDIR / "plot_train_class_distribution.png", dpi=160, bbox_inches="tight")
plt.show()


# ===== NOTEBOOK CELL 15 =====
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    token=HF_TOKEN
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=5,
    label2id=LABEL2ID,
    id2label=ID2LABEL,
    ignore_mismatched_sizes=True,
    token=HF_TOKEN
)

model.config.problem_type = "single_label_classification"

print("Model architecture loaded.")
print("num_labels:", model.config.num_labels)
print("id2label:", model.config.id2label)


# ===== NOTEBOOK CELL 17 =====
def to_hf_dataset(df):
    tmp = df[["text", "label_id"]].copy()
    tmp = tmp.rename(columns={"label_id": "labels"})
    tmp["labels"] = tmp["labels"].astype(int)
    return Dataset.from_pandas(tmp, preserve_index=False)

train_ds = to_hf_dataset(train_df)
val_ds = to_hf_dataset(val_df)
test_ds = to_hf_dataset(test_df)

print(train_ds)
print(val_ds)
print(test_ds)


# ===== NOTEBOOK CELL 19 =====
def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

train_tok = train_ds.map(tokenize_batch, batched=True, desc="Tokenizing train")
val_tok = val_ds.map(tokenize_batch, batched=True, desc="Tokenizing validation")
test_tok = test_ds.map(tokenize_batch, batched=True, desc="Tokenizing test")

# Remove raw text from tensors sent to the model.
train_tok = train_tok.remove_columns(["text"])
val_tok = val_tok.remove_columns(["text"])
test_tok = test_tok.remove_columns(["text"])

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="longest",
    return_tensors="pt",
)

print(train_tok)


# ===== NOTEBOOK CELL 21 =====
train_labels = train_df["label_id"].astype(int).to_numpy()
class_counts = np.bincount(train_labels, minlength=5)

class_sampling_weights = np.power(
    1.0 / np.maximum(class_counts, 1),
    SAMPLER_POWER
)

sample_weights = class_sampling_weights[train_labels]

sampler_audit = pd.DataFrame({
    "label_id": range(5),
    "label": [ID2LABEL[i] for i in range(5)],
    "train_count": class_counts,
    "sampling_weight": class_sampling_weights,
})

display(sampler_audit)

print(
    "\nImportant: ordinary cross-entropy will be used. "
    "We are NOT combining this sampler with the old 27x Credit loss weight."
)


# ===== NOTEBOOK CELL 23 =====
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
    }


# ===== NOTEBOOK CELL 25 =====
class ImbalanceAwareTrainer(Trainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires train_dataset.")

        generator = torch.Generator()
        generator.manual_seed(SEED)

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


# ===== NOTEBOOK CELL 27 =====
use_fp16 = bool(torch.cuda.is_available())
use_bf16 = False

# Prefer BF16 on newer GPUs when supported.
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    use_bf16 = True
    use_fp16 = False

training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_OUT),

    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    max_grad_norm=MAX_GRAD_NORM,

    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,

    load_best_model_at_end=True,
    metric_for_best_model=BEST_MODEL_METRIC,
    greater_is_better=True,
    save_total_limit=2,

    fp16=use_fp16,
    bf16=use_bf16,

    dataloader_num_workers=2,
    dataloader_pin_memory=torch.cuda.is_available(),

    report_to="none",
    seed=SEED,
    data_seed=SEED,

    remove_unused_columns=True,
)

print("fp16:", use_fp16)
print("bf16:", use_bf16)
print(training_args)


# ===== NOTEBOOK CELL 29 =====
trainer = ImbalanceAwareTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE
        )
    ],
)

print("Trainer ready.")


# ===== NOTEBOOK CELL 31 =====
train_result = trainer.train()

print("\nTraining complete.")
print("Best checkpoint:", trainer.state.best_model_checkpoint)
print("Best validation metric:", trainer.state.best_metric)

train_metrics = train_result.metrics
train_metrics


# ===== NOTEBOOK CELL 33 =====
val_metrics = trainer.evaluate(val_tok, metric_key_prefix="validation")
print(json.dumps(val_metrics, indent=2))

val_pred_output = trainer.predict(val_tok, metric_key_prefix="validation_pred")
val_preds = np.argmax(val_pred_output.predictions, axis=-1)
val_true = val_pred_output.label_ids

print("\nVALIDATION CLASSIFICATION REPORT")
print(
    classification_report(
        val_true,
        val_preds,
        labels=list(range(5)),
        target_names=[ID2LABEL[i] for i in range(5)],
        digits=4,
        zero_division=0,
    )
)


# ===== NOTEBOOK CELL 35 =====
test_output = trainer.predict(test_tok, metric_key_prefix="test")

test_logits = test_output.predictions
test_true = test_output.label_ids
test_preds = np.argmax(test_logits, axis=-1)

test_metrics = compute_metrics((test_logits, test_true))

print("TEST SUMMARY")
print(json.dumps(test_metrics, indent=2))

report_dict = classification_report(
    test_true,
    test_preds,
    labels=list(range(5)),
    target_names=[ID2LABEL[i] for i in range(5)],
    output_dict=True,
    digits=6,
    zero_division=0,
)

report_text = classification_report(
    test_true,
    test_preds,
    labels=list(range(5)),
    target_names=[ID2LABEL[i] for i in range(5)],
    digits=4,
    zero_division=0,
)

print("\nTEST CLASSIFICATION REPORT")
print(report_text)


# ===== NOTEBOOK CELL 37 =====
labels_order = list(range(5))
display_labels = [ID2LABEL[i] for i in labels_order]

cm = confusion_matrix(test_true, test_preds, labels=labels_order)
cm_norm = confusion_matrix(test_true, test_preds, labels=labels_order, normalize="true")

fig, ax = plt.subplots(figsize=(9, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(ax=ax, xticks_rotation=35, values_format="d")
ax.set_title("Test Confusion Matrix — Counts")
plt.tight_layout()
plt.savefig(WORKDIR / "confusion_matrix_test_counts.png", dpi=180, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(9, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=display_labels)
disp.plot(ax=ax, xticks_rotation=35, values_format=".2f")
ax.set_title("Test Confusion Matrix — Row Normalized")
plt.tight_layout()
plt.savefig(WORKDIR / "confusion_matrix_test_normalized.png", dpi=180, bbox_inches="tight")
plt.show()


# ===== NOTEBOOK CELL 39 =====
per_class_rows = []

for label_name in LABEL2ID:
    r = report_dict[label_name]
    per_class_rows.append({
        "label": label_name,
        "label_id": LABEL2ID[label_name],
        "precision": r["precision"],
        "recall": r["recall"],
        "f1": r["f1-score"],
        "support": int(r["support"]),
    })

per_class_df = pd.DataFrame(per_class_rows)
display(per_class_df)

per_class_df.to_csv(WORKDIR / "test_per_class_metrics.csv", index=False)


# ===== NOTEBOOK CELL 41 =====
# Stable softmax probabilities.
probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()

predictions_df = test_df.copy().reset_index(drop=True)
predictions_df["true_label_id"] = test_true
predictions_df["pred_label_id"] = test_preds
predictions_df["pred_label"] = [ID2LABEL[int(i)] for i in test_preds]
predictions_df["correct"] = predictions_df["true_label_id"] == predictions_df["pred_label_id"]

for i in range(5):
    safe_name = ID2LABEL[i].lower().replace(" ", "_").replace("/", "or")
    predictions_df[f"prob_{i}_{safe_name}"] = probs[:, i]

predictions_df.to_csv(WORKDIR / "test_predictions_with_probabilities.csv", index=False)
predictions_df.to_parquet(WORKDIR / "test_predictions_with_probabilities.parquet", index=False)

display(predictions_df.head())
print("Test accuracy from saved predictions:", predictions_df["correct"].mean())


# ===== NOTEBOOK CELL 43 =====
MODEL_OUT.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(MODEL_OUT))
tokenizer.save_pretrained(str(MODEL_OUT))

print("Saved model to:", MODEL_OUT)
print("Files:")
for p in sorted(MODEL_OUT.iterdir()):
    print(" ", p.name)


# ===== NOTEBOOK CELL 45 =====
final_report = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "base_model": MODEL_NAME,
    "task": "5-class financial event classification",
    "label_mapping": LABEL2ID,
    "dataset": {
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_class_counts": {
            str(k): int(v) for k, v in train_df["label"].value_counts().to_dict().items()
        },
        "validation_class_counts": {
            str(k): int(v) for k, v in val_df["label"].value_counts().to_dict().items()
        },
        "test_class_counts": {
            str(k): int(v) for k, v in test_df["label"].value_counts().to_dict().items()
        },
    },
    "training": {
        "max_length": MAX_LENGTH,
        "num_epochs_requested": NUM_EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "sampler": "WeightedRandomSampler",
        "sampler_power": SAMPLER_POWER,
        "loss": "ordinary cross-entropy from AutoModelForSequenceClassification",
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "best_model_metric": BEST_MODEL_METRIC,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_metric": trainer.state.best_metric,
        "hardware": GPU_NAME,
    },
    "validation_metrics": {
        str(k): (float(v) if isinstance(v, (int, float, np.floating)) else v)
        for k, v in val_metrics.items()
    },
    "test_summary_metrics": {
        str(k): float(v) for k, v in test_metrics.items()
    },
    "test_classification_report": report_dict,
    "known_limitations": [
        "Credit Event has few real validation/test examples; its per-class metric is statistically unstable.",
        "Geopolitical is dominated by GDELT-style text, creating possible source-style shortcut learning.",
        "Macroeconomic is dominated by Twitter Financial News Topic text.",
        "Long documents are head-truncated to 512 tokens in this baseline.",
        "Synthetic augmentation is intentionally deferred until after the full architecture works."
    ]
}

with open(WORKDIR / "finbert_event_classifier_report.json", "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=2)

with open(WORKDIR / "test_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print(json.dumps(final_report["test_summary_metrics"], indent=2))


# ===== NOTEBOOK CELL 47 =====
# Reloading is unnecessary here because Trainer already holds the best model.
inference_model = trainer.model
inference_model.eval()

def predict_event(text):
    encoded = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    encoded = {k: v.to(inference_model.device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = inference_model(**encoded).logits
        p = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

    pred_id = int(np.argmax(p))

    return {
        "label_id": pred_id,
        "label": ID2LABEL[pred_id],
        "confidence": float(p[pred_id]),
        "class_probabilities": {
            ID2LABEL[i]: float(p[i]) for i in range(5)
        },
    }

# Example — replace with any financial event text.
example = "The company filed for Chapter 11 protection after missing a scheduled bond payment."
print(json.dumps(predict_event(example), indent=2))


# ===== NOTEBOOK CELL 49 =====
archive_base = WORKDIR / "finbert_event_classifier_5class"
archive_path = shutil.make_archive(
    str(archive_base),
    "zip",
    root_dir=str(MODEL_OUT)
)

print("Model archive:", archive_path)
