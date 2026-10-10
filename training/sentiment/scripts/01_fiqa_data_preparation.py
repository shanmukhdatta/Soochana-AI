# Original Kaggle notebook source export — code cells in original sequence.
# Notebook/IPython and Kaggle environment are required for successful execution.
# Preferred reproduction path: use notebooks/ files as described in README.md.


# --- Notebook cell 2 ---

# IPython magic / shell: %pip -q install -U "datasets>=3,<4" "huggingface_hub>=0.27" pyarrow pandas matplotlib



# --- Notebook cell 4 ---

import os
import json
import hashlib
import random
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

WORKDIR = Path(os.environ.get("RISK_PULSE_WORKDIR", "/kaggle/working" if Path("/kaggle/working").exists() else "./training_runs/sentiment")).resolve()
WORKDIR.mkdir(parents=True, exist_ok=True)

print("Working directory:", WORKDIR)



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
    print("No usable HF_TOKEN secret. Continuing anonymously.")
    print("Reason:", type(e).__name__)



# --- Notebook cell 8 ---

DATASET_CANDIDATES = [
    "TheFinAI/fiqa-sentiment-classification",
    "ChanceFocus/fiqa-sentiment-classification",
    "vikrantburman/fiqa-sentiment-classification",
]

dataset = None
DATASET_USED = None
load_errors = {}

for repo in DATASET_CANDIDATES:
    try:
        print("Trying:", repo)
        dataset = load_dataset(repo, token=HF_TOKEN)
        DATASET_USED = repo
        print("Loaded:", repo)
        break
    except Exception as e:
        load_errors[repo] = f"{type(e).__name__}: {str(e)[:300]}"
        print("  failed:", load_errors[repo])

if dataset is None:
    raise RuntimeError(
        "Could not load a FiQA sentiment source. "
        + json.dumps(load_errors, indent=2)
    )

print("\nAvailable splits:", list(dataset.keys()))
for name, ds in dataset.items():
    print(f"{name:>12}: {len(ds):>5} rows | {ds.column_names}")



# --- Notebook cell 10 ---

available = set(dataset.keys())

def choose_split(candidates, role):
    for name in candidates:
        if name in available:
            print(f"{role}: source split '{name}'")
            return name
    raise ValueError(f"No {role} split found. Available: {sorted(available)}")

SOURCE_TRAIN = choose_split(["train"], "train")
SOURCE_VALID = choose_split(["validation", "valid", "val", "dev"], "validation")
SOURCE_TEST = choose_split(["test"], "test")

SPLIT_MAP = {
    "train": SOURCE_TRAIN,
    "validation": SOURCE_VALID,
    "test": SOURCE_TEST,
}

print(SPLIT_MAP)



# --- Notebook cell 12 ---

raw_frames = {}

for final_name, source_name in SPLIT_MAP.items():
    df = dataset[source_name].to_pandas()
    raw_frames[final_name] = df

    print(f"\n--- {final_name.upper()} ---")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    display(df.head(3))



# --- Notebook cell 14 ---

common_columns = set.intersection(*(set(df.columns) for df in raw_frames.values()))

TEXT_COL = next((c for c in ["sentence", "text", "context"] if c in common_columns), None)
SCORE_COL = next((c for c in ["score", "sentiment_score"] if c in common_columns), None)

if TEXT_COL is None:
    raise ValueError(f"No supported text column found. Common columns: {sorted(common_columns)}")

if SCORE_COL is None:
    raise ValueError(f"No continuous score column found. Common columns: {sorted(common_columns)}")

print("Text column :", TEXT_COL)
print("Score column:", SCORE_COL)



# --- Notebook cell 16 ---

META_CANDIDATES = [
    "_id", "id", "target", "aspect", "aspects",
    "type", "format", "snippets", "label"
]

def standardize(df, split_name):
    out = pd.DataFrame({
        "text": df[TEXT_COL],
        "sentiment_score": df[SCORE_COL],
    })

    for c in META_CANDIDATES:
        if c in df.columns and c not in out.columns:
            out[c] = df[c]

    out["source_dataset"] = DATASET_USED
    out["source_split"] = split_name
    out["source_row_index"] = np.arange(len(out), dtype=int)
    return out

frames = {k: standardize(v, k) for k, v in raw_frames.items()}

for split, df in frames.items():
    print(split, df.shape)



# --- Notebook cell 18 ---

technical_report = {}
technical_clean = {}

for split, df in frames.items():
    x = df.copy()
    r = Counter()
    r["rows_in"] = len(x)

    bad_text = x["text"].isna() | ~x["text"].map(lambda z: isinstance(z, str))
    r["missing_or_non_string_text"] = int(bad_text.sum())
    x = x.loc[~bad_text].copy()

    x["text"] = x["text"].astype(str).str.strip()
    empty = x["text"].eq("")
    r["empty_text"] = int(empty.sum())
    x = x.loc[~empty].copy()

    numeric = pd.to_numeric(x["sentiment_score"], errors="coerce")
    bad_score = numeric.isna()
    r["missing_or_non_numeric_score"] = int(bad_score.sum())
    x = x.loc[~bad_score].copy()
    numeric = numeric.loc[~bad_score]

    finite_mask = np.isfinite(numeric.to_numpy(dtype=float))
    r["non_finite_score"] = int((~finite_mask).sum())
    x = x.loc[finite_mask].copy()
    numeric = numeric.loc[finite_mask]

    x["sentiment_score"] = numeric.astype(float).to_numpy()

    out_of_range = ~x["sentiment_score"].between(-1.0, 1.0, inclusive="both")
    r["score_outside_minus1_plus1"] = int(out_of_range.sum())
    x = x.loc[~out_of_range].copy()

    r["rows_out"] = len(x)
    technical_report[split] = dict(r)
    technical_clean[split] = x.reset_index(drop=True)

print(json.dumps(technical_report, indent=2))
print("\nTechnical validation complete.")



# --- Notebook cell 20 ---

def normalize_text(text):
    return " ".join(str(text).lower().split())

def make_hash(text):
    return hashlib.sha256(
        normalize_text(text).encode("utf-8")
    ).hexdigest()

for split in technical_clean:
    technical_clean[split] = technical_clean[split].copy()
    technical_clean[split]["text_hash"] = technical_clean[split]["text"].map(make_hash)

print("Text hashes created.")



# --- Notebook cell 22 ---

SCORE_TOL = 1e-9

within_summary = {}
conflict_frames = []
deduped = {}

for split, df in technical_clean.items():
    x = df.copy()
    dup_mask = x.duplicated("text_hash", keep=False)
    groups = x.loc[dup_mask].groupby("text_hash", sort=False)

    conflict_hashes = []

    for h, g in groups:
        scores = g["sentiment_score"].astype(float).to_numpy()
        if np.max(scores) - np.min(scores) > SCORE_TOL:
            conflict_hashes.append(h)
            conflict_frames.append(g.assign(conflict_split=split))

    before = len(x)
    x = x.loc[~x["text_hash"].isin(conflict_hashes)].copy()
    x = x.drop_duplicates("text_hash", keep="first").reset_index(drop=True)

    within_summary[split] = {
        "rows_before": int(before),
        "exact_duplicate_text_groups": int(len(groups)),
        "conflicting_duplicate_groups": int(len(conflict_hashes)),
        "rows_after": int(len(x)),
    }
    deduped[split] = x

within_conflicts = (
    pd.concat(conflict_frames, ignore_index=True)
    if conflict_frames else pd.DataFrame()
)

if len(within_conflicts):
    path = WORKDIR / "sentiment_within_split_conflicts.csv"
    within_conflicts.to_csv(path, index=False)
    print("Saved:", path)

print(json.dumps(within_summary, indent=2))



# --- Notebook cell 24 ---

train_df = deduped["train"].copy()
val_df = deduped["validation"].copy()
test_df = deduped["test"].copy()

test_hashes = set(test_df["text_hash"])

val_in_test = val_df["text_hash"].isin(test_hashes)
val_removed = val_df.loc[val_in_test].assign(leakage_pair="validation->test")
val_df = val_df.loc[~val_in_test].copy().reset_index(drop=True)

val_hashes = set(val_df["text_hash"])

train_in_test = train_df["text_hash"].isin(test_hashes)
train_in_val = train_df["text_hash"].isin(val_hashes)

train_removed_test = train_df.loc[train_in_test].assign(leakage_pair="train->test")
train_removed_val = train_df.loc[
    train_in_val & ~train_in_test
].assign(leakage_pair="train->validation")

train_df = train_df.loc[
    ~(train_in_test | train_in_val)
].copy().reset_index(drop=True)

leakage_removed = pd.concat(
    [val_removed, train_removed_test, train_removed_val],
    ignore_index=True
)

cross_split_summary = {
    "validation_removed_due_to_test_overlap": int(len(val_removed)),
    "train_removed_due_to_test_overlap": int(len(train_removed_test)),
    "train_removed_due_to_validation_overlap": int(len(train_removed_val)),
}

if len(leakage_removed):
    path = WORKDIR / "sentiment_cross_split_leakage_removed.csv"
    leakage_removed.to_csv(path, index=False)
    print("Saved:", path)

print(json.dumps(cross_split_summary, indent=2))

assert set(train_df["text_hash"]).isdisjoint(set(val_df["text_hash"]))
assert set(train_df["text_hash"]).isdisjoint(set(test_df["text_hash"]))
assert set(val_df["text_hash"]).isdisjoint(set(test_df["text_hash"]))

print("Cross-split exact leakage check: PASSED")



# --- Notebook cell 26 ---

def display_label(score):
    if score < -0.15:
        return "negative"
    if score > 0.15:
        return "positive"
    return "neutral"

final_frames = {
    "train": train_df,
    "validation": val_df,
    "test": test_df,
}

for split, df in final_frames.items():
    df["display_sentiment_label"] = df["sentiment_score"].map(display_label)
    print(f"\n{split.upper()}")
    print(df["display_sentiment_label"].value_counts().to_string())



# --- Notebook cell 28 ---

score_stats = {}

for split, df in final_frames.items():
    s = df["sentiment_score"].astype(float)
    score_stats[split] = {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "min": float(s.min()),
        "q05": float(s.quantile(0.05)),
        "q25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "q75": float(s.quantile(0.75)),
        "q95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }

print(json.dumps(score_stats, indent=2))



# --- Notebook cell 30 ---

for split, df in final_frames.items():
    plt.figure(figsize=(8, 4.5))
    plt.hist(df["sentiment_score"], bins=25)
    plt.axvline(0.0, linewidth=1)
    plt.title(f"FiQA continuous sentiment distribution — {split}")
    plt.xlabel("sentiment_score")
    plt.ylabel("examples")
    plt.tight_layout()

    path = WORKDIR / f"sentiment_score_distribution_{split}.png"
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()

    print("Saved:", path)



# --- Notebook cell 32 ---

FORMAT_COL = None

for candidate in ["type", "format"]:
    if candidate in final_frames["train"].columns:
        FORMAT_COL = candidate
        break

if FORMAT_COL is None:
    print("No type/format field available.")
else:
    for split, df in final_frames.items():
        print(f"\n{split.upper()} — {FORMAT_COL}")
        print(df[FORMAT_COL].value_counts(dropna=False).to_string())



# --- Notebook cell 34 ---

for split, df in final_frames.items():
    assert len(df) > 0, f"{split} is empty"
    assert df["text"].notna().all()
    assert df["text"].astype(str).str.strip().ne("").all()
    assert pd.api.types.is_numeric_dtype(df["sentiment_score"])
    assert np.isfinite(df["sentiment_score"].astype(float)).all()
    assert df["sentiment_score"].between(-1, 1).all()

assert set(final_frames["train"]["text_hash"]).isdisjoint(
    set(final_frames["validation"]["text_hash"])
)
assert set(final_frames["train"]["text_hash"]).isdisjoint(
    set(final_frames["test"]["text_hash"])
)
assert set(final_frames["validation"]["text_hash"]).isdisjoint(
    set(final_frames["test"]["text_hash"])
)

print("ALL HARD ASSERTIONS PASSED.")



# --- Notebook cell 36 ---

output_files = {}

for split, df in final_frames.items():
    parquet_path = WORKDIR / f"sentiment_{split}.parquet"
    csv_path = WORKDIR / f"sentiment_{split}.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    output_files[f"{split}_parquet"] = str(parquet_path)
    output_files[f"{split}_csv"] = str(csv_path)

    print(f"{split}: {len(df)} rows")
    print(" ", parquet_path)
    print(" ", csv_path)



# --- Notebook cell 38 ---

report = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "task": "continuous financial sentiment regression",
    "dataset_used": DATASET_USED,
    "source_split_map": SPLIT_MAP,
    "raw_rows": {
        split: int(len(df))
        for split, df in raw_frames.items()
    },
    "technical_validation": technical_report,
    "within_split_duplicate_audit": within_summary,
    "cross_split_leakage_audit": cross_split_summary,
    "clean_rows": {
        split: int(len(df))
        for split, df in final_frames.items()
    },
    "score_statistics": score_stats,
    "display_only_thresholds": {
        "negative": "score < -0.15",
        "neutral": "-0.15 <= score <= 0.15",
        "positive": "score > 0.15",
        "note": "These are not regression training labels."
    },
    "display_category_counts": {
        split: {
            str(k): int(v)
            for k, v in df["display_sentiment_label"].value_counts().to_dict().items()
        }
        for split, df in final_frames.items()
    },
    "format_column": FORMAT_COL,
    "output_files": output_files,
    "next_model_plan": {
        "base_model": "ProsusAI/finbert",
        "head": "single-output regression head",
        "output_constraint": "tanh -> [-1,+1]",
        "training_loss": "MSE",
        "evaluation_metrics": ["MAE", "RMSE", "Pearson", "Spearman"]
    }
}

report_path = WORKDIR / "sentiment_preprocessing_report.json"

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
print("\nSaved:", report_path)

