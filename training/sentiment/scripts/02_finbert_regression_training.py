try:
    from IPython.display import display
except ImportError:
    display = print

# Original Kaggle notebook source export — code cells in original sequence.
# Notebook/IPython and Kaggle environment are required for successful execution.
# Preferred reproduction path: use notebooks/ files as described in README.md.


# --- Notebook cell 2 ---

# IPython magic / shell: %pip -q install -U "transformers>=4.46,<5" "datasets>=3,<4" "accelerate>=1,<2" "scikit-learn>=1.4" "scipy>=1.11" pyarrow safetensors huggingface_hub



# --- Notebook cell 4 ---

import os
import gc
import json
import math
import random
import shutil
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from datasets import Dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from transformers import (
    AutoTokenizer,
    AutoConfig,
    BertModel,
    BertPreTrainedModel,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HARDWARE = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

print("Device:", DEVICE)
print("Hardware:", HARDWARE)
print("PyTorch:", torch.__version__)



# --- Notebook cell 6 ---

HF_TOKEN = None

try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login

    user_secrets = UserSecretsClient()
    HF_TOKEN = user_secrets.get_secret("HF_TOKEN")

    if HF_TOKEN:
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("Hugging Face login successful.")
    else:
        print("HF_TOKEN is empty. Continuing anonymously.")
except Exception as e:
    print("No usable Kaggle HF_TOKEN secret. Continuing anonymously.")
    print("Reason:", type(e).__name__)



# --- Notebook cell 8 ---

MODEL_NAME = "ProsusAI/finbert"

MAX_LENGTH = 512

NUM_EPOCHS = 12
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0

EARLY_STOPPING_PATIENCE = 3
DROPOUT = 0.10

WORKDIR = Path(os.environ.get("RISK_PULSE_WORKDIR", "/kaggle/working" if Path("/kaggle/working").exists() else "./training_runs/sentiment")).resolve()
INPUT_ROOT = Path(os.environ.get("RISK_PULSE_INPUT_DIR", "/kaggle/input" if Path("/kaggle/input").exists() else str(WORKDIR))).resolve()

CHECKPOINT_DIR = WORKDIR / "finbert_sentiment_regression_checkpoints"
MODEL_DIR = WORKDIR / "finbert_sentiment_regression"

print("Base model:", MODEL_NAME)
print("Max length:", MAX_LENGTH)
print("Maximum epochs:", NUM_EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Early-stopping patience:", EARLY_STOPPING_PATIENCE)



# --- Notebook cell 10 ---

def find_candidates(root, names):
    hits = []
    if root.exists():
        for name in names:
            hits.extend(root.rglob(name))
    return sorted(set(hits), key=lambda p: (p.suffix.lower() != ".parquet", str(p)))

def resolve_split(stem):
    hits = find_candidates(
        INPUT_ROOT,
        [f"{stem}.parquet", f"{stem}.csv"]
    )

    print(f"\nCandidates for {stem}:")
    for h in hits:
        print(" ", h)

    if not hits:
        raise FileNotFoundError(
            f"Could not find {stem}.parquet or {stem}.csv under {INPUT_ROOT}. "
            "Attach the complete output of the sentiment preprocessing notebook as a Kaggle Input."
        )

    selected = hits[0]
    print("Selected:", selected)
    return selected

TRAIN_PATH = resolve_split("sentiment_train")
VAL_PATH = resolve_split("sentiment_validation")
TEST_PATH = resolve_split("sentiment_test")

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



# --- Notebook cell 12 ---

REQUIRED_COLUMNS = {"text", "sentiment_score"}

def validate_frame(df, split_name):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{split_name} missing columns: {sorted(missing)}")

    if df["text"].isna().any():
        raise ValueError(f"{split_name} contains null text.")

    if df["text"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{split_name} contains empty text.")

    scores = pd.to_numeric(df["sentiment_score"], errors="coerce")

    if scores.isna().any():
        raise ValueError(f"{split_name} contains non-numeric/missing sentiment scores.")

    if not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError(f"{split_name} contains non-finite sentiment scores.")

    if not scores.between(-1.0, 1.0, inclusive="both").all():
        raise ValueError(f"{split_name} contains sentiment scores outside [-1,+1].")

    df["sentiment_score"] = scores.astype(np.float32)

for name, df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df),
]:
    validate_frame(df, name)

if all("text_hash" in df.columns for df in [train_df, val_df, test_df]):
    train_hashes = set(train_df["text_hash"].astype(str))
    val_hashes = set(val_df["text_hash"].astype(str))
    test_hashes = set(test_df["text_hash"].astype(str))

    assert train_hashes.isdisjoint(val_hashes), "Exact leakage: train vs validation"
    assert train_hashes.isdisjoint(test_hashes), "Exact leakage: train vs test"
    assert val_hashes.isdisjoint(test_hashes), "Exact leakage: validation vs test"

    print("Exact hash leakage check: PASSED")
else:
    print("text_hash column not found in every split; skipping hash leakage re-check.")

print("\nScore ranges")
for name, df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df),
]:
    print(
        f"{name:>10}: n={len(df):>4} "
        f"min={df.sentiment_score.min(): .4f} "
        f"mean={df.sentiment_score.mean(): .4f} "
        f"max={df.sentiment_score.max(): .4f}"
    )

print("\nDataset validation: PASSED")



# --- Notebook cell 14 ---

for split_name, df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df),
]:
    plt.figure(figsize=(8, 4.5))
    plt.hist(df["sentiment_score"], bins=25)
    plt.axvline(0.0, linewidth=1)
    plt.title(f"Continuous sentiment target — {split_name}")
    plt.xlabel("sentiment_score")
    plt.ylabel("Examples")
    plt.tight_layout()

    path = WORKDIR / f"training_input_distribution_{split_name}.png"
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()



# --- Notebook cell 16 ---

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    token=HF_TOKEN,
)

def token_length_stats(df, split_name):
    lengths = []
    for text in df["text"].astype(str).tolist():
        ids = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        lengths.append(len(ids))

    arr = np.asarray(lengths)
    report = {
        "split": split_name,
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": int(arr.max()),
        "over_max_length": int((arr > MAX_LENGTH).sum()),
        "pct_over_max_length": float((arr > MAX_LENGTH).mean() * 100),
    }
    return report

token_length_report = {
    "train": token_length_stats(train_df, "train"),
    "validation": token_length_stats(val_df, "validation"),
    "test": token_length_stats(test_df, "test"),
}

print(json.dumps(token_length_report, indent=2))



# --- Notebook cell 18 ---

class FinBertSentimentRegressor(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)

        dropout_prob = getattr(
            config,
            "classifier_dropout",
            None
        )
        if dropout_prob is None:
            dropout_prob = DROPOUT

        self.dropout = nn.Dropout(dropout_prob)
        self.regressor = nn.Linear(config.hidden_size, 1)
        self.tanh = nn.Tanh()
        self.loss_fn = nn.MSELoss()

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)

        raw_score = self.regressor(pooled)
        bounded_score = self.tanh(raw_score)

        loss = None
        if labels is not None:
            labels = labels.float().view(-1, 1)
            loss = self.loss_fn(bounded_score, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=bounded_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

config = AutoConfig.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
)

config.num_labels = 1
config.problem_type = "regression"
config.sentiment_output_min = -1.0
config.sentiment_output_max = 1.0
config.sentiment_head = "Dropout -> Linear(hidden_size,1) -> tanh"
config.sentiment_loss = "MSE"

model = FinBertSentimentRegressor.from_pretrained(
    MODEL_NAME,
    config=config,
    token=HF_TOKEN,
)

print("Custom FinBERT regressor loaded.")
print("Hidden size:", config.hidden_size)
print("Output dimension:", model.regressor.out_features)



# --- Notebook cell 20 ---

model.to(DEVICE)
model.eval()

sanity_texts = [
    "The company reported record profit and raised its outlook.",
    "The issuer missed an interest payment and entered default.",
    "Markets were little changed in afternoon trading.",
]

sanity_inputs = tokenizer(
    sanity_texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
)

sanity_inputs = {
    k: v.to(DEVICE)
    for k, v in sanity_inputs.items()
}

with torch.no_grad():
    sanity_scores = model(**sanity_inputs).logits.squeeze(-1).cpu().numpy()

print("Initial bounded scores:", sanity_scores)

assert np.all(sanity_scores >= -1.0)
assert np.all(sanity_scores <= 1.0)

print("tanh output-range assertion: PASSED")



# --- Notebook cell 22 ---

def to_hf_dataset(df):
    tmp = pd.DataFrame({
        "text": df["text"].astype(str),
        "labels": df["sentiment_score"].astype(np.float32),
    })
    return Dataset.from_pandas(tmp, preserve_index=False)

train_ds = to_hf_dataset(train_df)
val_ds = to_hf_dataset(val_df)
test_ds = to_hf_dataset(test_df)

print(train_ds)
print(val_ds)
print(test_ds)



# --- Notebook cell 24 ---

def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

train_tok = train_ds.map(
    tokenize_batch,
    batched=True,
    desc="Tokenizing train",
)
val_tok = val_ds.map(
    tokenize_batch,
    batched=True,
    desc="Tokenizing validation",
)
test_tok = test_ds.map(
    tokenize_batch,
    batched=True,
    desc="Tokenizing test",
)

train_tok = train_tok.remove_columns(["text"])
val_tok = val_tok.remove_columns(["text"])
test_tok = test_tok.remove_columns(["text"])

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="longest",
    return_tensors="pt",
)

print(train_tok)



# --- Notebook cell 26 ---

def safe_pearson(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(pearsonr(y_true, y_pred).statistic)

def safe_spearman(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(spearmanr(y_true, y_pred).statistic)

def regression_metrics_from_arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson": safe_pearson(y_true, y_pred),
        "spearman": safe_spearman(y_true, y_pred),
    }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    return regression_metrics_from_arrays(
        labels,
        predictions,
    )



# --- Notebook cell 28 ---

train_mean = float(train_df["sentiment_score"].mean())

val_mean_predictions = np.full(
    len(val_df),
    train_mean,
    dtype=float,
)

validation_mean_baseline = regression_metrics_from_arrays(
    val_df["sentiment_score"].to_numpy(),
    val_mean_predictions,
)

print("Training-set mean sentiment:", train_mean)
print("Validation mean-predictor baseline:")
print(json.dumps(validation_mean_baseline, indent=2))



# --- Notebook cell 30 ---

use_fp16 = bool(torch.cuda.is_available())
use_bf16 = False

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    use_bf16 = True
    use_fp16 = False

training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),

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
    logging_steps=10,

    load_best_model_at_end=True,
    metric_for_best_model="rmse",
    greater_is_better=False,
    save_total_limit=3,

    fp16=use_fp16,
    bf16=use_bf16,

    dataloader_num_workers=2,
    dataloader_pin_memory=torch.cuda.is_available(),

    report_to="none",
    seed=SEED,
    data_seed=SEED,
)

print("fp16:", use_fp16)
print("bf16:", use_bf16)



# --- Notebook cell 32 ---

trainer = Trainer(
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



# --- Notebook cell 34 ---

train_result = trainer.train()

print("\nTraining complete.")
print("Best checkpoint:", trainer.state.best_model_checkpoint)
print("Best validation RMSE:", trainer.state.best_metric)

train_result.metrics



# --- Notebook cell 36 ---

history_df = pd.DataFrame(trainer.state.log_history)
history_path = WORKDIR / "sentiment_training_history.csv"
history_df.to_csv(history_path, index=False)

display(history_df.tail(20))

eval_history = history_df[
    history_df["eval_rmse"].notna()
].copy() if "eval_rmse" in history_df.columns else pd.DataFrame()

if len(eval_history):
    plt.figure(figsize=(7, 4.5))
    plt.plot(
        eval_history["epoch"],
        eval_history["eval_rmse"],
        marker="o",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Validation RMSE")
    plt.title("Validation RMSE by epoch")
    plt.tight_layout()
    plt.savefig(
        WORKDIR / "validation_rmse_by_epoch.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.show()

print("Saved:", history_path)



# --- Notebook cell 38 ---

validation_metrics = trainer.evaluate(
    val_tok,
    metric_key_prefix="validation",
)

print(json.dumps(validation_metrics, indent=2))

val_output = trainer.predict(
    val_tok,
    metric_key_prefix="validation_prediction",
)

val_predictions = np.asarray(
    val_output.predictions
).reshape(-1)

assert np.all(val_predictions >= -1.000001)
assert np.all(val_predictions <= 1.000001)

print("Validation prediction range:",
      float(val_predictions.min()),
      float(val_predictions.max()))



# --- Notebook cell 40 ---

test_output = trainer.predict(
    test_tok,
    metric_key_prefix="test",
)

test_predictions = np.asarray(
    test_output.predictions
).reshape(-1)

test_true = np.asarray(
    test_output.label_ids
).reshape(-1)

assert np.all(test_predictions >= -1.000001)
assert np.all(test_predictions <= 1.000001)

test_metrics = regression_metrics_from_arrays(
    test_true,
    test_predictions,
)

print("FINAL TEST METRICS")
print(json.dumps(test_metrics, indent=2))

print(
    "\nPrediction range:",
    float(test_predictions.min()),
    "to",
    float(test_predictions.max()),
)



# --- Notebook cell 42 ---

test_mean_predictions = np.full(
    len(test_df),
    train_mean,
    dtype=float,
)

test_mean_baseline = regression_metrics_from_arrays(
    test_true,
    test_mean_predictions,
)

comparison_df = pd.DataFrame([
    {
        "model": "train-mean baseline",
        **test_mean_baseline,
    },
    {
        "model": "FinBERT continuous regressor",
        **test_metrics,
    },
])

display(comparison_df)

comparison_df.to_csv(
    WORKDIR / "sentiment_test_model_comparison.csv",
    index=False,
)



# --- Notebook cell 44 ---

def display_sentiment_label(score):
    if score < -0.15:
        return "negative"
    if score > 0.15:
        return "positive"
    return "neutral"

test_predictions_df = test_df.copy().reset_index(drop=True)

test_predictions_df["true_sentiment_score"] = test_true
test_predictions_df["predicted_sentiment_score"] = test_predictions

test_predictions_df["residual"] = (
    test_predictions_df["predicted_sentiment_score"]
    - test_predictions_df["true_sentiment_score"]
)

test_predictions_df["absolute_error"] = (
    test_predictions_df["residual"].abs()
)

test_predictions_df["true_display_label"] = (
    test_predictions_df["true_sentiment_score"]
    .map(display_sentiment_label)
)

test_predictions_df["predicted_display_label"] = (
    test_predictions_df["predicted_sentiment_score"]
    .map(display_sentiment_label)
)

csv_path = WORKDIR / "sentiment_test_predictions.csv"
parquet_path = WORKDIR / "sentiment_test_predictions.parquet"

test_predictions_df.to_csv(csv_path, index=False)
test_predictions_df.to_parquet(parquet_path, index=False)

display(
    test_predictions_df[
        [
            "text",
            "true_sentiment_score",
            "predicted_sentiment_score",
            "absolute_error",
        ]
    ].head(10)
)

print("Saved:", csv_path)
print("Saved:", parquet_path)



# --- Notebook cell 46 ---

plt.figure(figsize=(6.5, 6.5))
plt.scatter(
    test_true,
    test_predictions,
    alpha=0.7,
)

line_min = min(float(test_true.min()), float(test_predictions.min()))
line_max = max(float(test_true.max()), float(test_predictions.max()))

plt.plot(
    [line_min, line_max],
    [line_min, line_max],
)

plt.xlabel("True FiQA sentiment score")
plt.ylabel("Predicted sentiment score")
plt.title("FinBERT sentiment regression — test set")
plt.tight_layout()

scatter_path = WORKDIR / "sentiment_true_vs_predicted.png"
plt.savefig(scatter_path, dpi=180, bbox_inches="tight")
plt.show()

print("Saved:", scatter_path)



# --- Notebook cell 48 ---

residuals = test_predictions - test_true

plt.figure(figsize=(8, 4.5))
plt.hist(residuals, bins=25)
plt.axvline(0.0, linewidth=1)
plt.xlabel("Prediction - true sentiment")
plt.ylabel("Examples")
plt.title("Sentiment regression residual distribution")
plt.tight_layout()

residual_path = WORKDIR / "sentiment_residual_distribution.png"
plt.savefig(residual_path, dpi=180, bbox_inches="tight")
plt.show()

print("Mean residual:", float(np.mean(residuals)))
print("Saved:", residual_path)



# --- Notebook cell 50 ---

largest_errors = (
    test_predictions_df
    .sort_values("absolute_error", ascending=False)
    .head(25)
)

largest_errors.to_csv(
    WORKDIR / "sentiment_largest_test_errors.csv",
    index=False,
)

display(
    largest_errors[
        [
            "text",
            "true_sentiment_score",
            "predicted_sentiment_score",
            "absolute_error",
        ]
    ].head(15)
)



# --- Notebook cell 52 ---

MODEL_DIR.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))

print("Saved final model to:", MODEL_DIR)
print("\nModel files:")
for p in sorted(MODEL_DIR.iterdir()):
    print(" ", p.name)



# --- Notebook cell 54 ---

modeling_source = r"""
import torch
import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class FinBertSentimentRegressor(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)

        dropout_prob = getattr(config, "classifier_dropout", None)
        if dropout_prob is None:
            dropout_prob = 0.10

        self.dropout = nn.Dropout(dropout_prob)
        self.regressor = nn.Linear(config.hidden_size, 1)
        self.tanh = nn.Tanh()
        self.loss_fn = nn.MSELoss()

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        pooled = self.dropout(outputs.pooler_output)
        bounded_score = self.tanh(self.regressor(pooled))

        loss = None
        if labels is not None:
            labels = labels.float().view(-1, 1)
            loss = self.loss_fn(bounded_score, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=bounded_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""

modeling_path = MODEL_DIR / "modeling_finbert_sentiment_regressor.py"
modeling_path.write_text(modeling_source, encoding="utf-8")

print("Saved:", modeling_path)



# --- Notebook cell 56 ---

# Free some GPU memory before reload.
trainer_model_device = next(trainer.model.parameters()).device
trainer.model.to("cpu")
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

reloaded_config = AutoConfig.from_pretrained(
    MODEL_DIR
)

reloaded_model = FinBertSentimentRegressor.from_pretrained(
    MODEL_DIR,
    config=reloaded_config,
)

reloaded_model.to(DEVICE)
reloaded_model.eval()

reload_text = "The issuer defaulted after failing to make an interest payment."

reload_inputs = tokenizer(
    reload_text,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_LENGTH,
)

reload_inputs = {
    k: v.to(DEVICE)
    for k, v in reload_inputs.items()
}

with torch.no_grad():
    reload_score = float(
        reloaded_model(**reload_inputs)
        .logits.squeeze()
        .cpu()
        .item()
    )

print("Reloaded-model score:", reload_score)

assert -1.0 <= reload_score <= 1.0

print("Saved-model reload test: PASSED")



# --- Notebook cell 58 ---

def predict_sentiment(text):
    encoded = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )

    encoded = {
        k: v.to(reloaded_model.device)
        for k, v in encoded.items()
    }

    with torch.no_grad():
        score = float(
            reloaded_model(**encoded)
            .logits.squeeze()
            .detach()
            .cpu()
            .item()
        )

    # Numerical safety only; tanh already guarantees this range.
    score = float(np.clip(score, -1.0, 1.0))

    return {
        "sentiment_score": score,
        "sentiment_label": display_sentiment_label(score),
    }

examples = [
    "The company raised its full-year profit outlook after record revenue.",
    "The issuer missed a bond payment and entered default.",
    "Shares were little changed after the announcement.",
]

for text in examples:
    print("\nTEXT:", text)
    print(json.dumps(predict_sentiment(text), indent=2))



# --- Notebook cell 60 ---

report = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "task": "continuous financial sentiment regression",
    "base_model": MODEL_NAME,

    "architecture": {
        "encoder": "ProsusAI/finbert BERT encoder",
        "head": "Dropout -> Linear(hidden_size,1) -> tanh",
        "output_range": [-1.0, 1.0],
        "training_loss": "MSE",
    },

    "data": {
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_score_min": float(train_df.sentiment_score.min()),
        "train_score_max": float(train_df.sentiment_score.max()),
        "validation_score_min": float(val_df.sentiment_score.min()),
        "validation_score_max": float(val_df.sentiment_score.max()),
        "test_score_min": float(test_df.sentiment_score.min()),
        "test_score_max": float(test_df.sentiment_score.max()),
    },

    "token_length_audit": token_length_report,

    "training": {
        "max_length": MAX_LENGTH,
        "max_epochs_requested": NUM_EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "max_grad_norm": MAX_GRAD_NORM,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "best_model_metric": "validation RMSE",
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_rmse": (
            float(trainer.state.best_metric)
            if trainer.state.best_metric is not None
            else None
        ),
        "hardware": HARDWARE,
    },

    "validation_metrics": {
        str(k): (
            float(v)
            if isinstance(v, (int, float, np.integer, np.floating))
            else v
        )
        for k, v in validation_metrics.items()
    },

    "test_metrics": test_metrics,

    "mean_predictor_baselines": {
        "train_mean_sentiment": train_mean,
        "validation": validation_mean_baseline,
        "test": test_mean_baseline,
    },

    "display_only_thresholds": {
        "negative": "score < -0.15",
        "neutral": "-0.15 <= score <= 0.15",
        "positive": "score > 0.15",
    },

    "known_limitations": [
        "FiQA is a small supervised dataset, so model variance across random seeds may be non-trivial.",
        "The regression model is text-only and does not explicitly condition on FiQA target/entity metadata.",
        "Display labels use fixed +/-0.15 thresholds; the core model output remains continuous.",
        "This baseline fine-tunes the full FinBERT encoder; additional data or seed-averaged training can be tested later if necessary.",
    ],
}

report_path = WORKDIR / "finbert_sentiment_regression_report.json"

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report["test_metrics"], indent=2))
print("\nSaved report:", report_path)



# --- Notebook cell 62 ---

archive_base = WORKDIR / "finbert_sentiment_regression"
archive_path = shutil.make_archive(
    str(archive_base),
    "zip",
    root_dir=str(MODEL_DIR),
)

print("Created:", archive_path)

