#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Faithful Python export of original notebook: impact-engine-training-final.ipynb.

Kaggle-friendly original behavior; Python export adds environment variable overrides
for input/output locations and removes notebook-only pip directives. It does not
fabricate pretrained weights, datasets, or experiment results.
"""
import os
from IPython.display import display


# %% Code cell 2
# Notebook-only %pip installation omitted; see ../requirements.txt

# %% Code cell 3
import json, zipfile, warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix
import joblib
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
np.random.seed(SEED)

INPUT_ROOT = Path(os.environ.get('RISKPULSE_IMPACT_INPUT_ROOT', str(Path('/kaggle/input') if Path('/kaggle/input').exists() else Path.cwd()))).expanduser().resolve()
WORKDIR = Path(os.environ.get('RISKPULSE_IMPACT_WORKDIR', str(Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()))).expanduser().resolve()
OUTPUT_DIR = WORKDIR / "impact_step3_final_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "realized_impact"
TRUE_SCORE = "impact_score_1_10"
EARLY_STOP = 100

print("xgboost:", xgb.__version__)
print("lightgbm:", lgb.__version__)
print("OUTPUT_DIR:", OUTPUT_DIR)

# %% Code cell 5
NAMES = {
    "train": "impact_step2_train_features.parquet",
    "validation": "impact_step2_validation_features.parquet",
    "test": "impact_step2_test_features.parquet",
    "manifest": "impact_step2_feature_manifest.json",
    "deciles": "impact_decile_thresholds.json",
    "report": "impact_step2_report.json",
}

def find_one(name, required=True):
    hits = sorted(INPUT_ROOT.rglob(name), key=lambda p: (len(str(p)), str(p)))
    if hits:
        return hits[0]
    if required:
        raise FileNotFoundError(f"Missing required input: {name}")
    return None

PATHS = {k: find_one(v, required=(k != "report")) for k, v in NAMES.items()}

for k, p in PATHS.items():
    print(f"{k:>10}:", p)

# %% Code cell 7
with open(PATHS["manifest"], "r", encoding="utf-8") as f:
    manifest = json.load(f)

with open(PATHS["deciles"], "r", encoding="utf-8") as f:
    raw_deciles = json.load(f)

NUMERIC = list(manifest["numeric_features"])
CATEGORICAL = list(manifest["categorical_features"])
FEATURES = list(manifest["model_features"])
FORBIDDEN = set(manifest.get("forbidden_future_features", []))

if TARGET in FEATURES:
    raise RuntimeError("Target leakage: realized_impact is in model features.")

bad = sorted(set(FEATURES) & FORBIDDEN)
if bad:
    raise RuntimeError(f"Forbidden future features found: {bad}")

if len(FEATURES) != len(set(FEATURES)):
    raise RuntimeError("Duplicate feature names in manifest.")

def get_deciles(obj):
    if isinstance(obj, dict):
        if "quantiles" in obj and isinstance(obj["quantiles"], dict):
            obj = obj["quantiles"]
        keys = [f"p{i}" for i in range(10, 100, 10)]
        if all(k in obj for k in keys):
            return np.array([float(obj[k]) for k in keys], dtype=float)
        for v in obj.values():
            if isinstance(v, dict):
                try:
                    return get_deciles(v)
                except ValueError:
                    pass
    raise ValueError("Could not find p10...p90 in decile JSON.")

DECILES = get_deciles(raw_deciles)

if len(DECILES) != 9 or not np.all(np.diff(DECILES) > 0):
    raise RuntimeError("Frozen impact deciles are invalid.")

print("Numeric:", len(NUMERIC))
print("Categorical:", len(CATEGORICAL))
print("Total raw model features:", len(FEATURES))
print("Frozen deciles:", DECILES.tolist())

# %% Code cell 9
train = pd.read_parquet(PATHS["train"]).copy()
val = pd.read_parquet(PATHS["validation"]).copy()
test = pd.read_parquet(PATHS["test"]).copy()

required = set(FEATURES) | {
    TARGET, TRUE_SCORE, "reaction_date", "ticker", "headline", "split"
}

def audit(df, split_name):
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"{split_name} missing: {sorted(miss)}")
    if not df["split"].astype(str).eq(split_name).all():
        raise ValueError(f"{split_name}: split label mismatch")
    df["reaction_date"] = pd.to_datetime(
        df["reaction_date"], errors="coerce"
    ).dt.tz_localize(None).dt.normalize()
    if df["reaction_date"].isna().any():
        raise ValueError(f"{split_name}: invalid reaction dates")
    if df[TARGET].isna().any() or (df[TARGET] < 0).any():
        raise ValueError(f"{split_name}: invalid target")
    if not df[TRUE_SCORE].between(1, 10).all():
        raise ValueError(f"{split_name}: invalid impact score")

for d, n in [(train, "train"), (val, "validation"), (test, "test")]:
    audit(d, n)

assert train.reaction_date.max() < val.reaction_date.min()
assert val.reaction_date.max() < test.reaction_date.min()

print("Rows:", len(train), len(val), len(test), "total", len(train)+len(val)+len(test))
print("Train:", train.reaction_date.min(), "to", train.reaction_date.max())
print("Validation:", val.reaction_date.min(), "to", val.reaction_date.max())
print("Test:", test.reaction_date.min(), "to", test.reaction_date.max())
print("Frozen chronological split audit: PASSED")

# %% Code cell 10
if PATHS["report"] is not None:
    with open(PATHS["report"], "r", encoding="utf-8") as f:
        step2_report = json.load(f)
    checks = step2_report.get("anti_leakage_assertions", {})
    failed = [k for k, v in checks.items() if v is not True]
    if failed:
        raise RuntimeError(f"Step-2 anti-leakage checks failed: {failed}")
    print("Step-2 anti-leakage report: PASSED")
else:
    print("Step-2 report absent; local schema/split audits passed.")

# %% Code cell 12
def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )

prep_select = make_preprocessor()

Xtr = prep_select.fit_transform(train[FEATURES])
Xv = prep_select.transform(val[FEATURES])
ytr = train[TARGET].astype(float).to_numpy()
yv = val[TARGET].astype(float).to_numpy()

print("Train matrix:", Xtr.shape)
print("Validation matrix:", Xv.shape)
print("Encoded feature count:", len(prep_select.get_feature_names_out()))

# %% Code cell 14
def clip0(x):
    return np.maximum(np.asarray(x, dtype=float), 0.0)

def to_score(x):
    return np.searchsorted(
        DECILES,
        clip0(x),
        side="right",
    ).astype(int) + 1

def corr_safe(fn, a, b):
    try:
        v = fn(a, b)[0]
        return float(v) if np.isfinite(v) else 0.0
    except Exception:
        return 0.0

def metrics(y, pred, true_score=None):
    pred = clip0(pred)
    out = {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "Pearson": corr_safe(pearsonr, y, pred),
        "Spearman": corr_safe(spearmanr, y, pred),
    }
    if true_score is not None:
        ps = to_score(pred)
        ts = np.asarray(true_score, dtype=int)
        out.update({
            "Score_MAE": float(mean_absolute_error(ts, ps)),
            "Score_Exact_Accuracy": float(np.mean(ps == ts)),
            "Score_Within1_Accuracy": float(np.mean(np.abs(ps - ts) <= 1)),
        })
    return out

# %% Code cell 16
train_mean = float(ytr.mean())
train_median = float(np.median(ytr))

baseline_val = pd.DataFrame([
    {
        "model": "mean",
        **metrics(yv, np.full_like(yv, train_mean), val[TRUE_SCORE]),
    },
    {
        "model": "median",
        **metrics(yv, np.full_like(yv, train_median), val[TRUE_SCORE]),
    },
])

display(baseline_val)

# %% Code cell 19
XGB_CFGS = [
    dict(name="xgb_A", max_depth=4, learning_rate=0.03, min_child_weight=8,  subsample=0.85, colsample_bytree=0.85, reg_alpha=0.00, reg_lambda=2.0),
    dict(name="xgb_B", max_depth=5, learning_rate=0.025,min_child_weight=12, subsample=0.85, colsample_bytree=0.80, reg_alpha=0.05, reg_lambda=3.0),
    dict(name="xgb_C", max_depth=3, learning_rate=0.04, min_child_weight=6,  subsample=0.90, colsample_bytree=0.90, reg_alpha=0.00, reg_lambda=2.0),
    dict(name="xgb_D", max_depth=6, learning_rate=0.02, min_child_weight=15, subsample=0.80, colsample_bytree=0.80, reg_alpha=0.10, reg_lambda=4.0),
]

records = []

for cfg in XGB_CFGS:
    name = cfg["name"]
    params = {k:v for k,v in cfg.items() if k != "name"}

    model = xgb.XGBRegressor(
        n_estimators=4000,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        early_stopping_rounds=EARLY_STOP,
        **params,
    )

    model.fit(
        Xtr, ytr,
        eval_set=[(Xv, yv)],
        verbose=False,
    )

    pv = clip0(model.predict(Xv))
    m = metrics(yv, pv, val[TRUE_SCORE])

    best_it = getattr(model, "best_iteration", None)
    best_it = int(best_it if best_it is not None else 3999)

    records.append({
        "family": "XGBoost",
        "model": name,
        "best_iteration": best_it,
        "config": cfg,
        **m,
    })

    print(name, {k:round(v,6) for k,v in m.items()}, "best_it", best_it)

# %% Code cell 21
LGB_CFGS = [
    dict(name="lgb_A", num_leaves=31, learning_rate=0.03,  min_child_samples=30, subsample=0.85, subsample_freq=1, colsample_bytree=0.85, reg_alpha=0.00, reg_lambda=2.0),
    dict(name="lgb_B", num_leaves=20, learning_rate=0.035, min_child_samples=40, subsample=0.90, subsample_freq=1, colsample_bytree=0.90, reg_alpha=0.05, reg_lambda=3.0),
    dict(name="lgb_C", num_leaves=45, learning_rate=0.02,  min_child_samples=35, subsample=0.80, subsample_freq=1, colsample_bytree=0.80, reg_alpha=0.10, reg_lambda=4.0),
    dict(name="lgb_D", num_leaves=15, learning_rate=0.04,  min_child_samples=25, subsample=0.90, subsample_freq=1, colsample_bytree=0.85, reg_alpha=0.00, reg_lambda=2.5),
]

for cfg in LGB_CFGS:
    name = cfg["name"]
    params = {k:v for k,v in cfg.items() if k != "name"}

    model = lgb.LGBMRegressor(
        n_estimators=4000,
        objective="regression",
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
        **params,
    )

    model.fit(
        Xtr, ytr,
        eval_set=[(Xv, yv)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )

    pv = clip0(model.predict(Xv))
    m = metrics(yv, pv, val[TRUE_SCORE])

    best_it = getattr(model, "best_iteration_", None)
    best_it = int(best_it if best_it and best_it > 0 else 4000)

    records.append({
        "family": "LightGBM",
        "model": name,
        "best_iteration": best_it,
        "config": cfg,
        **m,
    })

    print(name, {k:round(v,6) for k,v in m.items()}, "best_it", best_it)

# %% Code cell 23
selection = pd.DataFrame([
    {k:v for k,v in r.items() if k != "config"}
    for r in records
]).sort_values(["RMSE","MAE"]).reset_index(drop=True)

display(selection)

winner_name = str(selection.iloc[0]["model"])
winner_family = str(selection.iloc[0]["family"])
winner = next(r for r in records if r["model"] == winner_name)

MODEL_SELECTION_COMPLETE = True

selection_summary = {
    "selection_metric": "validation RMSE",
    "test_used_for_selection": False,
    "winner_model": winner_name,
    "winner_family": winner_family,
    "winner_best_iteration": int(winner["best_iteration"]),
    "winner_config": winner["config"],
    "winner_validation_metrics": {
        k: float(winner[k])
        for k in [
            "MAE","RMSE","Pearson","Spearman",
            "Score_MAE","Score_Exact_Accuracy","Score_Within1_Accuracy"
        ]
    },
}

print(json.dumps(selection_summary, indent=2))

# %% Code cell 26
assert MODEL_SELECTION_COMPLETE

fit_df = pd.concat([train, val], ignore_index=True, sort=False)

prep_final = make_preprocessor()
Xfit = prep_final.fit_transform(fit_df[FEATURES])
Xtest_final = prep_final.transform(test[FEATURES])

yfit = fit_df[TARGET].astype(float).to_numpy()
ytest_final = test[TARGET].astype(float).to_numpy()

encoded_names = prep_final.get_feature_names_out().tolist()

cfg = {k:v for k,v in winner["config"].items() if k != "name"}

if winner_family == "XGBoost":
    ntrees = int(winner["best_iteration"]) + 1
    final_model = xgb.XGBRegressor(
        n_estimators=ntrees,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        **cfg,
    )
elif winner_family == "LightGBM":
    ntrees = max(1, int(winner["best_iteration"]))
    final_model = lgb.LGBMRegressor(
        n_estimators=ntrees,
        objective="regression",
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
        **cfg,
    )
else:
    raise RuntimeError(f"Unknown winner family: {winner_family}")

final_model.fit(Xfit, yfit)

print("Final family:", winner_family)
print("Final trees:", ntrees)
print("Final fitting rows:", len(fit_df))

# %% Code cell 28
pred_test = clip0(final_model.predict(Xtest_final))
true_score_test = test[TRUE_SCORE].astype(int).to_numpy()
pred_score_test = to_score(pred_test)

TEST_METRICS = metrics(
    ytest_final,
    pred_test,
    true_score_test,
)

print("FINAL TEST METRICS")
print(json.dumps({k:round(v,6) for k,v in TEST_METRICS.items()}, indent=2))

# %% Code cell 30
fit_mean = float(yfit.mean())
fit_median = float(np.median(yfit))

test_baselines = pd.DataFrame([
    {
        "model":"mean",
        **metrics(
            ytest_final,
            np.full_like(ytest_final, fit_mean),
            true_score_test,
        ),
    },
    {
        "model":"median",
        **metrics(
            ytest_final,
            np.full_like(ytest_final, fit_median),
            true_score_test,
        ),
    },
])

display(test_baselines)

best_rmse = float(test_baselines["RMSE"].min())
best_mae = float(test_baselines["MAE"].min())

improvement = {
    "RMSE_reduction_fraction": float(
        (best_rmse - TEST_METRICS["RMSE"]) / best_rmse
    ),
    "MAE_reduction_fraction": float(
        (best_mae - TEST_METRICS["MAE"]) / best_mae
    ),
}

print(json.dumps(improvement, indent=2))

# %% Code cell 32
print("True score counts:")
print(pd.Series(true_score_test).value_counts().sort_index())

print("\nPredicted score counts:")
print(pd.Series(pred_score_test).value_counts().sort_index())

cm = confusion_matrix(
    true_score_test,
    pred_score_test,
    labels=list(range(1,11)),
)

cm_df = pd.DataFrame(
    cm,
    index=[f"true_{i}" for i in range(1,11)],
    columns=[f"pred_{i}" for i in range(1,11)],
)

display(cm_df)

# %% Code cell 34
importance = np.asarray(final_model.feature_importances_, dtype=float)

if len(importance) != len(encoded_names):
    raise RuntimeError("Feature importance/name mismatch.")

importance_df = pd.DataFrame({
    "feature": encoded_names,
    "importance": importance,
}).sort_values("importance", ascending=False).reset_index(drop=True)

pred_df = pd.DataFrame({
    "reaction_date": test["reaction_date"].to_numpy(),
    "ticker": test["ticker"].astype(str).to_numpy(),
    "headline": test["headline"].astype(str).to_numpy(),
    "true_realized_impact": ytest_final,
    "predicted_realized_impact": pred_test,
    "true_impact_score": true_score_test,
    "predicted_impact_score": pred_score_test,
})

pred_df["absolute_error"] = np.abs(
    pred_df["true_realized_impact"]
    - pred_df["predicted_realized_impact"]
)

display(importance_df.head(25))
display(pred_df.sort_values("absolute_error", ascending=False).head(15))

# %% Code cell 35
fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(
    pred_df["true_realized_impact"],
    pred_df["predicted_realized_impact"],
    alpha=0.35,
    s=14,
)
mx = max(
    float(pred_df["true_realized_impact"].max()),
    float(pred_df["predicted_realized_impact"].max()),
)
ax.plot([0,mx],[0,mx], linestyle="--")
ax.set_xlabel("True realized impact")
ax.set_ylabel("Predicted realized impact")
ax.set_title("Impact Engine — Test Predictions")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "test_true_vs_predicted.png", dpi=160, bbox_inches="tight")
plt.show()

# %% Code cell 36
top = importance_df.head(25).sort_values("importance")

fig, ax = plt.subplots(figsize=(8,8))
ax.barh(top["feature"], top["importance"])
ax.set_xlabel("Feature importance")
ax.set_title("Impact Engine — Top 25 Features")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "feature_importance_top25.png", dpi=160, bbox_inches="tight")
plt.show()

# %% Code cell 38
MODEL_PATH = OUTPUT_DIR / "impact_model_final.joblib"
PREP_PATH = OUTPUT_DIR / "impact_preprocessor.joblib"

joblib.dump(final_model, MODEL_PATH)
joblib.dump(prep_final, PREP_PATH)

if winner_family == "XGBoost":
    NATIVE_PATH = OUTPUT_DIR / "impact_model_final_xgboost.json"
    final_model.save_model(NATIVE_PATH)
else:
    NATIVE_PATH = OUTPUT_DIR / "impact_model_final_lightgbm.txt"
    final_model.booster_.save_model(str(NATIVE_PATH))

importance_df.to_csv(
    OUTPUT_DIR / "impact_feature_importance.csv",
    index=False,
)
pred_df.to_csv(
    OUTPUT_DIR / "impact_test_predictions.csv",
    index=False,
)
cm_df.to_csv(
    OUTPUT_DIR / "impact_score_confusion_matrix.csv"
)
selection.to_csv(
    OUTPUT_DIR / "impact_validation_model_selection.csv",
    index=False,
)
test_baselines.to_csv(
    OUTPUT_DIR / "impact_test_baselines.csv",
    index=False,
)

print("Saved:", MODEL_PATH)
print("Saved:", PREP_PATH)
print("Saved:", NATIVE_PATH)

# %% Code cell 39
report = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "stage": "Impact Engine Step 3 - final model training",
    "target": {
        "continuous_target": TARGET,
        "impact_score_conversion": "Frozen Step-1 training-only deciles -> 1..10",
        "decile_thresholds": DECILES.tolist(),
    },
    "data": {
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "final_fit_rows": int(len(fit_df)),
        "test_start": str(test["reaction_date"].min().date()),
        "test_end": str(test["reaction_date"].max().date()),
    },
    "selection": selection_summary,
    "final_model": {
        "family": winner_family,
        "name": winner_name,
        "final_n_estimators": int(ntrees),
    },
    "test_metrics": {
        k: float(v)
        for k,v in TEST_METRICS.items()
    },
    "test_baselines": test_baselines.to_dict(orient="records"),
    "improvement_vs_best_constant_baseline": improvement,
    "features": {
        "raw_features": FEATURES,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "encoded_feature_count": int(len(encoded_names)),
        "forbidden_future_features": sorted(FORBIDDEN),
    },
    "artifacts": {
        "model": str(MODEL_PATH),
        "preprocessor": str(PREP_PATH),
        "native_model": str(NATIVE_PATH),
        "feature_importance": str(
            OUTPUT_DIR / "impact_feature_importance.csv"
        ),
        "test_predictions": str(
            OUTPUT_DIR / "impact_test_predictions.csv"
        ),
    },
}

with open(
    OUTPUT_DIR / "impact_model_metrics.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(report, f, indent=2)

with open(
    OUTPUT_DIR / "impact_model_feature_manifest.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(manifest, f, indent=2)

with open(
    OUTPUT_DIR / "impact_decile_thresholds.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(raw_deciles, f, indent=2)

with open(
    OUTPUT_DIR / "impact_model_selected_config.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(selection_summary, f, indent=2)

print("Final metrics/manifests saved.")

# %% Code cell 41
helper = '''import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_FEATURES = __FEATURES__
DECILES = np.array(__DECILES__, dtype=float)

def load_engine(bundle_dir):
    p = Path(bundle_dir)
    model = joblib.load(p / "impact_model_final.joblib")
    prep = joblib.load(p / "impact_preprocessor.joblib")
    return model, prep

def predict_impact(df, model, prep):
    missing = sorted(set(MODEL_FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"Missing features: {missing}")
    X = prep.transform(df[MODEL_FEATURES])
    pred = np.maximum(np.asarray(model.predict(X), dtype=float), 0.0)
    score = np.searchsorted(DECILES, pred, side="right").astype(int) + 1
    return pd.DataFrame({
        "predicted_realized_impact": pred,
        "impact_score_1_10": score,
    }, index=df.index)
'''

helper = helper.replace(
    "__FEATURES__",
    repr(FEATURES),
).replace(
    "__DECILES__",
    repr(DECILES.tolist()),
)

(OUTPUT_DIR / "impact_inference.py").write_text(
    helper,
    encoding="utf-8",
)

print("Saved impact_inference.py")

# %% Code cell 43
m2 = joblib.load(MODEL_PATH)
p2 = joblib.load(PREP_PATH)

reload_pred = clip0(
    m2.predict(
        p2.transform(test[FEATURES].head(10))
    )
)

if not np.allclose(
    reload_pred,
    pred_test[:10],
    rtol=1e-6,
    atol=1e-8,
):
    raise RuntimeError("Reload smoke test failed.")

print("Saved-model reload smoke test: PASSED")

# %% Code cell 45
quality = {
    "finite_predictions": bool(np.isfinite(pred_test).all()),
    "nonnegative_predictions": bool((pred_test >= 0).all()),
    "prediction_variance_positive": bool(np.std(pred_test) > 0),
    "test_pearson_positive": bool(TEST_METRICS["Pearson"] > 0),
    "test_spearman_positive": bool(TEST_METRICS["Spearman"] > 0),
    "beats_constant_baseline_rmse": bool(TEST_METRICS["RMSE"] < best_rmse),
    "beats_constant_baseline_mae": bool(TEST_METRICS["MAE"] < best_mae),
}

print(json.dumps(quality, indent=2))

mechanical = [
    "finite_predictions",
    "nonnegative_predictions",
    "prediction_variance_positive",
]

failed = [k for k in mechanical if not quality[k]]
if failed:
    raise RuntimeError(f"Mechanical quality gates failed: {failed}")

with open(
    OUTPUT_DIR / "impact_model_quality_flags.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(quality, f, indent=2)

print("Mechanical final quality gates: PASSED")

# %% Code cell 47
bundle = WORKDIR / "impact_engine_final_bundle.zip"

if bundle.exists():
    bundle.unlink()

with zipfile.ZipFile(
    bundle,
    "w",
    compression=zipfile.ZIP_DEFLATED,
) as zf:
    for p in sorted(OUTPUT_DIR.rglob("*")):
        if p.is_file():
            zf.write(
                p,
                arcname=p.relative_to(OUTPUT_DIR),
            )

print("Final bundle:", bundle)
print("Size MB:", round(bundle.stat().st_size / 1024**2, 2))

# %% Code cell 49
print("=" * 78)
print("IMPACT ENGINE STEP 3 — FINAL MODEL TRAINING COMPLETE")
print("=" * 78)
print("Selected model :", winner_name, "/", winner_family)
print("Final trees    :", ntrees)
print("Test MAE       :", round(TEST_METRICS["MAE"], 6))
print("Test RMSE      :", round(TEST_METRICS["RMSE"], 6))
print("Test Pearson   :", round(TEST_METRICS["Pearson"], 6))
print("Test Spearman  :", round(TEST_METRICS["Spearman"], 6))
print("Score MAE      :", round(TEST_METRICS["Score_MAE"], 4))
print("Score exact    :", round(TEST_METRICS["Score_Exact_Accuracy"], 4))
print("Score within±1 :", round(TEST_METRICS["Score_Within1_Accuracy"], 4))
print("Bundle         :", bundle)

print("\nSEND BACK:")
print(" - impact_model_metrics.json")
print(" - impact_validation_model_selection.csv")
print(" - impact_feature_importance.csv")
print(" - impact_test_predictions.csv")
