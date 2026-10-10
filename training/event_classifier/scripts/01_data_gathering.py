"""Faithful code-cell export of event-classification-data-gathering(1).ipynb.
Execution order follows original notebook. Notebook/Kaggle inputs are required.
This file is a SOURCE EXPORT, not a verified local standalone retraining command.
Do not claim this script was executed during packaging.
"""



# ===== NOTEBOOK CELL 1 =====
# ORIGINAL NOTEBOOK SETUP (install via requirements.txt): !pip install -q -U "datasets>=3.2.0" huggingface_hub
# ORIGINAL NOTEBOOK SETUP (install via requirements.txt): !pip install -q pyarrow tqdm


# ===== NOTEBOOK CELL 3 =====
import os, re, json, sys, traceback, warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow
from tqdm.auto import tqdm

import datasets
import huggingface_hub
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
from huggingface_hub import hf_hub_download, list_repo_files, login, HfApi

try:
    from IPython.display import display
except Exception:  # plain python
    display = print

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_colwidth", 140)
pd.set_option("display.width", 220)

print("datasets", datasets.__version__, "| huggingface_hub", huggingface_hub.__version__,
      "| pandas", pd.__version__, "| pyarrow", pyarrow.__version__)

WORK_DIR = Path("/kaggle/working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV          = WORK_DIR / "final_raw_event_dataset.csv"
OUT_PARQUET      = WORK_DIR / "final_raw_event_dataset.parquet"
OUT_SUMMARY_CSV  = WORK_DIR / "dataset_source_summary.csv"
OUT_REPORT_JSON  = WORK_DIR / "extraction_report.json"
OUT_CONFLICTS    = WORK_DIR / "sentivent_multiclass_conflicts.csv"
OUT_SEC_META     = WORK_DIR / "sec_8k_credit_items_metadata.csv"   # Item 1.03 + 2.04 sidecar

# ---- Hugging Face auth (token is never hard-coded and never printed) ----
HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    print("Found Kaggle secret HF_TOKEN.")
except Exception as e:
    print(f"HF_TOKEN not available ({type(e).__name__}); will use anonymous public access.")

if HF_TOKEN:
    try:
        who = HfApi().whoami(token=HF_TOKEN)
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("Authenticated to Hugging Face as:", who.get("name"))
    except Exception as e:
        print(f"HF_TOKEN was rejected ({type(e).__name__}: {str(e)[:150]}); falling back to anonymous access.")
        HF_TOKEN = None

# ---- Target classes / schema ----
LABEL_TO_ID = {
    "Geopolitical": 0,
    "Macroeconomic": 1,
    "Credit Event": 2,
    "Merger / Acquisition": 3,
    "Product Launch": 4,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
SCHEMA_COLS = ["text", "label", "label_id", "source_dataset", "source_label",
               "source_subtype", "source_split", "source_id", "mapping_quality"]
ALLOWED_SOURCES = {"twitter_financial_news_topic", "gdelt_news_events", "sentivent",
                   "sec_8k_events", "corporate_event_detection", "catalyst_financial_news"}
ALLOWED_QUALITY = {"very_strong_candidate", "strong_candidate", "broad_candidate"}

# Added-source controls
CATALYST_REPO = "matthewchung74/catalyst"
CATALYST_MIN_CONFIDENCE = 0.80
SEC_CREDIT_ITEMS = {
    "1.03": ("1.03 Bankruptcy or Receivership", "very_strong_candidate"),
    "2.04": ("2.04 Triggering Events That Accelerate or Increase a Direct Financial Obligation", "strong_candidate"),
}

# Collected as each extractor runs, saved as extraction_report.json at the end
REPORT = {"sources": {}, "final": {}, "files_written": []}


# ===== NOTEBOOK CELL 5 =====
def banner(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)

def short(obj, n=400):
    s = repr(obj)
    return s if len(s) <= n else s[:n] + "...<truncated>"

def is_blank(x):
    """True for None / NaN / non-string / empty-after-strip."""
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if not isinstance(x, str):
        return True
    return x.strip() == ""

def as_list(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def first_present(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def make_row(text, label, source_dataset, source_label, source_subtype,
             source_split, source_id, mapping_quality):
    if isinstance(text, str):   # UTF-8 safety: lone surrogates become U+FFFD instead of crashing later
        text = re.sub(r"[\ud800-\udfff]", "\ufffd", text)
    return {
        "text": text, "label": label, "label_id": LABEL_TO_ID[label],
        "source_dataset": source_dataset, "source_label": source_label,
        "source_subtype": source_subtype, "source_split": source_split,
        "source_id": source_id, "mapping_quality": mapping_quality,
    }

def make_df(rows):
    return pd.DataFrame(rows, columns=SCHEMA_COLS)

def safe_load_dataset(path, name=None, **kw):
    """load_dataset with diagnostics and two documented fallbacks. Never swallows the final error."""
    try:
        return load_dataset(path, name, token=HF_TOKEN, **kw)
    except Exception as e1:
        print(f"  [load_dataset] plain attempt failed: {type(e1).__name__}: {str(e1)[:300]}")
        first_err = e1
    if "trust_remote_code" in str(first_err):
        try:
            print("  [load_dataset] retrying with trust_remote_code=True")
            return load_dataset(path, name, token=HF_TOKEN, trust_remote_code=True, **kw)
        except Exception as e2:
            print(f"  [load_dataset] trust_remote_code attempt failed: {type(e2).__name__}: {str(e2)[:300]}")
    try:
        print("  [load_dataset] retrying with revision='refs/convert/parquet'")
        return load_dataset(path, name, token=HF_TOKEN, revision="refs/convert/parquet", **kw)
    except Exception as e3:
        print(f"  [load_dataset] parquet-conversion attempt failed: {type(e3).__name__}: {str(e3)[:300]}")
    raise first_err

def print_configs(repo):
    try:
        cfgs = get_dataset_config_names(repo, token=HF_TOKEN)
        print(f"  configs for {repo}: {cfgs}")
        return cfgs
    except Exception as e:
        print(f"  (could not list configs for {repo}: {type(e).__name__}: {str(e)[:200]})")
        return None

def describe_dataset(ds, example_chars=600):
    print("  splits:", list(ds.keys()))
    for sp in ds.keys():
        d = ds[sp]
        print(f"  [{sp}] rows={getattr(d, 'num_rows', 'n/a')}")
        print(f"  [{sp}] columns={getattr(d, 'column_names', None)}")
        print(f"  [{sp}] features={d.features}")
        try:
            print(f"  [{sp}] example={short(next(iter(d)), example_chars)}")
        except Exception as e:
            print(f"  [{sp}] could not fetch an example: {type(e).__name__}: {e}")


# ===== NOTEBOOK CELL 7 =====
TWITTER_REPO = "zeroshot/twitter-financial-news-topic"
# label id -> (expected source label name, our label, mapping_quality)
TWITTER_MAP = {
    16: ("Politics",               "Geopolitical",         "broad_candidate"),
    14: ("Macro",                  "Macroeconomic",        "strong_candidate"),
    1:  ("Fed | Central Banks",    "Macroeconomic",        "strong_candidate"),
    13: ("M&A | Investments",      "Merger / Acquisition", "broad_candidate"),
    2:  ("Company | Product News", "Product Launch",       "broad_candidate"),
}

def _twitter_usable_map(names):
    norm = lambda s: re.sub(r"\s+", "", str(s).lower())
    usable = {}
    for lid, (exp_name, label, q) in TWITTER_MAP.items():
        if names is None:
            print(f"  !! WARNING: 'label' has no ClassLabel names; id {lid} ('{exp_name}') cannot be verified "
                  f"- using the documented id as instructed.")
            usable[lid] = (exp_name, label, q)
        elif lid < len(names) and norm(names[lid]) == norm(exp_name):
            print(f"  OK  label {lid:>2} == '{names[lid]}' -> {label} ({q})")
            usable[lid] = (names[lid], label, q)
        else:
            actual = names[lid] if lid < len(names) else "<out of range>"
            print(f"  !!  label {lid} is '{actual}' but expected '{exp_name}'. NOT extracting this label.")
    return usable

def extract_twitter():
    banner("SOURCE 1/5 - Twitter Financial News Topic")
    src = "twitter_financial_news_topic"
    rep = REPORT["sources"].setdefault(src, {})
    print_configs(TWITTER_REPO)
    ds = safe_load_dataset(TWITTER_REPO)
    describe_dataset(ds)

    rows, rep["per_split"], rep["missing_text"] = [], {}, 0
    for split in ds.keys():
        d = ds[split]
        missing_cols = [c for c in ("text", "label") if c not in d.column_names]
        if missing_cols:
            print(f"  !! split '{split}' lacks expected columns {missing_cols}; available: {d.column_names}. Skipping split.")
            continue
        names = getattr(d.features["label"], "names", None)
        usable = _twitter_usable_map(names)
        all_counts = Counter(d["label"])
        print(f"  [{split}] label counts (all 20 labels): {dict(sorted(all_counts.items()))}")
        n_kept = 0
        for idx, (text, lab) in enumerate(zip(d["text"], d["label"])):
            if lab not in usable:
                continue
            if is_blank(text):
                rep["missing_text"] += 1
                continue
            src_label, our_label, q = usable[lab]
            rows.append(make_row(text.strip(), our_label, src, src_label, "", split, f"{split}_{idx}", q))
            n_kept += 1
        rep["per_split"][split] = {"rows_in_split": len(d), "usable_rows": n_kept}
        print(f"  [{split}] extracted {n_kept} rows")
    df = make_df(rows)
    rep["usable_rows"] = len(df)
    return df


# ===== NOTEBOOK CELL 9 =====
GDELT_REPO = "DescribeEvents/gdelt_news_events"

def extract_gdelt():
    banner("SOURCE 2/5 - GDELT news events")
    src = "gdelt_news_events"
    rep = REPORT["sources"].setdefault(src, {})
    print_configs(GDELT_REPO)
    ds = safe_load_dataset(GDELT_REPO)
    describe_dataset(ds)

    id_candidates = ["id", "sequence_id", "seq_id", "sample_id", "doc_id", "document_id", "uid", "url"]
    rows = []
    n_seq = n_pairs = n_mismatch = n_missing_text = 0
    type_counts, logged = Counter(), 0
    for split in ds.keys():
        d = ds[split]
        need = ["type_event", "type_text"]
        missing_cols = [c for c in need if c not in d.column_names]
        if missing_cols:
            print(f"  !! split '{split}' lacks {missing_cols}; available: {d.column_names}. Skipping split.")
            continue
        id_col = first_present(d.column_names, id_candidates)
        print(f"  [{split}] id column: {id_col!r} (falls back to row index if None)")
        sub = d.select_columns(need + ([id_col] if id_col else []))
        for idx, ex in enumerate(tqdm(sub, desc=f"gdelt/{split}", unit="seq")):
            n_seq += 1
            te, tt = as_list(ex["type_event"]), as_list(ex["type_text"])
            if len(te) != len(tt):
                n_mismatch += 1
                if logged < 5:
                    print(f"  !! [{split}] sequence {idx}: len(type_event)={len(te)} != len(type_text)={len(tt)} -> SKIPPED")
                    logged += 1
                continue
            seq_id = str(ex[id_col]) if id_col and ex.get(id_col) is not None else f"{split}_{idx}"
            for j, (ev, txt) in enumerate(zip(te, tt)):
                n_pairs += 1
                if is_blank(txt):
                    n_missing_text += 1
                    continue
                type_counts[str(ev)] += 1
                rows.append(make_row(txt.strip(), "Geopolitical", src, str(ev), "", split,
                                     f"{seq_id}#{j}", "strong_candidate"))
    print(f"  sequences={n_seq}  event/text pairs={n_pairs}  length-mismatch sequences skipped={n_mismatch}  "
          f"blank event texts skipped={n_missing_text}")
    print(f"  type_event distribution: {dict(type_counts.most_common())}")
    rep.update({"n_sequences": n_seq, "n_exploded_event_pairs": n_pairs, "n_length_mismatch_sequences": n_mismatch,
                "missing_text": n_missing_text, "type_event_counts": dict(type_counts), "usable_rows": len(rows)})
    return make_df(rows)


# ===== NOTEBOOK CELL 11 =====
SENTIVENT_REPO = "GillesJacobs/sentivent"
SENTIVENT_CONFIG = "sentivent_unified_sentence"
# expected event_type -> (our label, mapping_quality)
SENTIVENT_MAP = {
    "Macroeconomics":    ("Macroeconomic",        "strong_candidate"),
    "Merger/Acquisition": ("Merger / Acquisition", "strong_candidate"),
    "Product/Service":   ("Product Launch",       "broad_candidate"),
    "Rating":            ("Credit Event",         "broad_candidate"),
}

def normalize_annotations(ann):
    """Return a list of annotation dicts whether HF gave list-of-dicts, dict-of-lists, JSON string, or one dict."""
    if ann is None:
        return []
    if isinstance(ann, str):
        try:
            ann = json.loads(ann)
        except Exception:
            return []
    if isinstance(ann, dict):
        vals = list(ann.values())
        if vals and all(isinstance(v, (list, tuple, np.ndarray)) for v in vals):
            lens = {len(v) for v in vals}
            if len(lens) == 1:
                n = lens.pop()
                return [{k: ann[k][i] for k in ann} for i in range(n)]
        return [ann]
    if isinstance(ann, (list, tuple, np.ndarray)):
        return [a for a in ann if isinstance(a, dict)]
    return []

_warned_no_kind = False
def is_event_annotation(a):
    global _warned_no_kind
    if "kind" in a:
        return a["kind"] == "event"
    if not _warned_no_kind:
        print("  !! annotations have no 'kind' key; treating annotations that carry 'event_type' as events.")
        _warned_no_kind = True
    return a.get("event_type") is not None

def _txt(x):
    return None if x is None else str(x)

def inspect_sentivent(ds):
    kinds, keys = Counter(), Counter()
    type_counts = Counter()
    subtypes = defaultdict(Counter)
    for sp in ds.keys():
        for ex in ds[sp]:
            for a in normalize_annotations(ex.get("annotations")):
                keys.update(a.keys())
                kinds[a.get("kind")] += 1
                if is_event_annotation(a):
                    et = _txt(a.get("event_type"))
                    type_counts[et] += 1
                    subtypes[et][_txt(a.get("event_subtype"))] += 1
    print("\n  --- SENTiVENT annotation inspection (BEFORE mapping) ---")
    print(f"  annotation keys seen: {dict(keys)}")
    print(f"  annotation 'kind' values: {dict(kinds)}")
    print(f"  event counts by event_type ({len(type_counts)} unique):")
    for et, n in type_counts.most_common():
        print(f"     {et!s:<28} {n}")
    print("  unique event_subtype values grouped by event_type:")
    for et, _ in type_counts.most_common():
        print(f"     {et}:")
        for st, n in subtypes[et].most_common():
            print(f"        - {st}  ({n})")
    return type_counts, subtypes

def resolve_event_types(discovered):
    """Map expected event_type strings onto what actually exists; only accept exact or case/punctuation-insensitive matches."""
    norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
    disc_norm = {norm(k): k for k in discovered if k is not None}
    resolved = {}
    for expected, (label, q) in SENTIVENT_MAP.items():
        if expected in discovered:
            resolved[expected] = (label, q)
        elif norm(expected) in disc_norm:
            real = disc_norm[norm(expected)]
            print(f"  note: expected event_type '{expected}' found as '{real}' (case/punctuation-only difference) - using it.")
            resolved[real] = (label, q)
        else:
            print(f"  !! expected event_type '{expected}' NOT found among discovered types; NOT extracting it.")
    return resolved

def extract_sentivent():
    banner("SOURCE 3/5 - SENTiVENT (sentivent_unified_sentence)")
    src = "sentivent"
    rep = REPORT["sources"].setdefault(src, {})
    cfgs = print_configs(SENTIVENT_REPO)
    if cfgs is not None and SENTIVENT_CONFIG not in cfgs:
        print(f"  !! config '{SENTIVENT_CONFIG}' not in available configs; stopping this source.")
        rep["status_detail"] = "config missing"
        return make_df([])
    ds = safe_load_dataset(SENTIVENT_REPO, SENTIVENT_CONFIG)
    describe_dataset(ds)
    for sp in ds.keys():
        miss = [c for c in ("text", "annotations") if c not in ds[sp].column_names]
        if miss:
            print(f"  !! split '{sp}' lacks {miss}; available: {ds[sp].column_names}. Stopping this source.")
            rep["status_detail"] = f"missing columns {miss}"
            return make_df([])

    type_counts, subtypes = inspect_sentivent(ds)
    resolved = resolve_event_types(set(type_counts))
    rep["event_type_counts_all"] = {str(k): v for k, v in type_counts.items()}
    rep["event_subtypes_by_type"] = {str(k): {str(s): n for s, n in c.items()} for k, c in subtypes.items()}
    rep["resolved_event_types"] = {k: v[0] for k, v in resolved.items()}
    if not resolved:
        print("  !! none of the target event types exist; stopping this source.")
        return make_df([])

    id_cols_all = ["sentence_id", "sent_id", "id", "uid", "doc_id", "document_id", "doc"]
    rows, conflicts = [], []
    n_sent = n_no_target = n_missing_text = 0
    per_label = Counter()
    for split in ds.keys():
        d = ds[split]
        id_cols = [c for c in id_cols_all if c in d.column_names]
        for idx, ex in enumerate(d):
            n_sent += 1
            matches = []  # (event_type, subtype, our_label, quality)
            for a in normalize_annotations(ex.get("annotations")):
                if not is_event_annotation(a):
                    continue
                et = _txt(a.get("event_type"))
                if et in resolved:
                    matches.append((et, _txt(a.get("event_subtype")) or "", resolved[et][0], resolved[et][1]))
            if not matches:
                n_no_target += 1
                continue
            text = ex.get("text")
            if is_blank(text):
                n_missing_text += 1
                continue
            ids = [str(ex[c]) for c in id_cols if ex.get(c) is not None]
            sid = f"{split}_{idx}" + (f"|{':'.join(ids)}" if ids else "")
            labels = {m[2] for m in matches}
            ev_types = sorted({m[0] for m in matches})
            subs = sorted({m[1] for m in matches if m[1]})
            if len(labels) == 1:
                label = next(iter(labels))
                q = matches[0][3]
                rows.append(make_row(text.strip(), label, src, "; ".join(ev_types), "; ".join(subs), split, sid, q))
                per_label[label] += 1
            else:
                conflicts.append({
                    "text": text.strip(), "source_split": split, "source_id": sid,
                    "candidate_labels": "; ".join(sorted(labels)),
                    "event_types": "; ".join(ev_types), "event_subtypes": "; ".join(subs),
                    "matches_json": json.dumps([[m[0], m[1], m[2]] for m in matches], ensure_ascii=False),
                })
    conf_df = pd.DataFrame(conflicts, columns=["text", "source_split", "source_id", "candidate_labels",
                                               "event_types", "event_subtypes", "matches_json"])
    conf_df.to_csv(OUT_CONFLICTS, index=False, encoding="utf-8")
    REPORT["files_written"].append(str(OUT_CONFLICTS))
    print(f"  sentences scanned={n_sent}  without target events={n_no_target}  kept={len(rows)}  "
          f"multi-class CONFLICTS={len(conf_df)} (saved to {OUT_CONFLICTS})")
    print(f"  kept per label: {dict(per_label)}")
    rep.update({"n_sentences_scanned": n_sent, "n_sentences_without_target_event": n_no_target,
                "missing_text": n_missing_text, "multiclass_conflicts": len(conf_df),
                "usable_rows": len(rows), "kept_per_label": dict(per_label)})
    return make_df(rows)


# ===== NOTEBOOK CELL 13 =====
SEC_REPO = "ZipLime/sec-8k-events"
SEC_META_CANDIDATES = [
    "accession_no", "entity_id", "cik", "knowledge_date", "source_url",
    "filing_date", "report_date", "ticker", "company_name", "form_type",
    "item_title", "item_name"
]
SEC_MAX_ROWS_SCANNED = None   # only affects the full-scan fallback

def extract_sec_8k():
    """
    Extract two SEC 8-K items that directly support our Credit Event class:
      1.03 Bankruptcy or Receivership               -> very_strong_candidate
      2.04 Triggering Events That Accelerate or
           Increase a Direct Financial Obligation   -> strong_candidate

    We keep ONLY rows with non-empty item_text. Metadata-only rows are written
    to the sidecar report but are not added to classifier text data.
    """
    banner("SOURCE 4/6 - SEC 8-K credit events (Items 1.03 + 2.04)")
    src = "sec_8k_events"
    rep = REPORT["sources"].setdefault(src, {})
    configs = print_configs(SEC_REPO)
    rep["available_configs"] = configs

    if configs is not None and "items" not in configs:
        print("  !! config 'items' not found; stopping this source.")
        rep["status_detail"] = "config 'items' missing"
        return make_df([])

    try:
        splits = get_dataset_split_names(SEC_REPO, "items", token=HF_TOKEN)
    except Exception as e:
        print(f"  (could not list splits: {type(e).__name__}: {str(e)[:200]}); assuming ['train']")
        splits = ["train"]

    print("  splits of 'items':", splits)
    rows, meta_rows = [], []
    totals = Counter()
    rep["per_split"] = {}

    for split in splits:
        try:
            first = next(iter(load_dataset(
                SEC_REPO, "items", split=split, streaming=True, token=HF_TOKEN
            )))
        except Exception as e:
            print(f"  !! could not open split '{split}': {type(e).__name__}: {str(e)[:300]}")
            continue

        cols = list(first.keys())
        print(f"  [{split}] columns: {cols}")
        print(f"  [{split}] example: " + short(
            {k: (short(v, 150) if k == "item_text" else v) for k, v in first.items()}, 900
        ))

        missing_cols = [c for c in ("item_code", "item_text") if c not in cols]
        if missing_cols:
            print(f"  !! [{split}] lacks expected columns {missing_cols}; skipping.")
            continue

        meta_cols = [c for c in SEC_META_CANDIDATES if c in cols]
        split_report = {}

        # Preferred route: run one small parquet-pushdown stream per target item.
        # If pushdown is unsupported, fall back to one full streaming scan.
        pushdown_succeeded_for_all = True
        local_rows, local_meta = [], []

        for target_code, (source_label, quality) in SEC_CREDIT_ITEMS.items():
            c = Counter()
            try:
                stream = load_dataset(
                    SEC_REPO, "items", split=split, streaming=True, token=HF_TOKEN,
                    columns=["item_code", "item_text"] + meta_cols,
                    filters=[("item_code", "==", target_code)],
                )
                for rec in tqdm(stream, desc=f"SEC {target_code}/{split} [pushdown]", unit="rows"):
                    c["returned"] += 1
                    if str(rec.get("item_code")).strip() != target_code:
                        c["wrong_code_returned"] += 1
                        continue

                    c["target_total"] += 1
                    text = rec.get("item_text")
                    if is_blank(text):
                        c["missing_text"] += 1
                        continue

                    c["with_text"] += 1
                    acc = rec.get("accession_no")
                    acc_s = "" if acc is None else str(acc).strip()
                    sid = f"{acc_s}:{target_code}" if acc_s else f"{split}:{target_code}:{c['returned']}"

                    local_rows.append(make_row(
                        text.strip(), "Credit Event", src, source_label, target_code,
                        split, sid, quality
                    ))
                    local_meta.append({
                        "source_id": sid,
                        "item_code": target_code,
                        "source_split": split,
                        **{m: (None if rec.get(m) is None else str(rec.get(m))) for m in meta_cols}
                    })

                split_report[target_code] = {"mode": "parquet_filter_pushdown", **dict(c)}
                print(
                    f"  [{split}] Item {target_code}: total={c['target_total']} "
                    f"with_text={c['with_text']} missing_text={c['missing_text']}"
                )

            except Exception as e:
                pushdown_succeeded_for_all = False
                print(
                    f"  !! pushdown for Item {target_code} failed: "
                    f"{type(e).__name__}: {str(e)[:300]}"
                )
                break

        if pushdown_succeeded_for_all:
            rows.extend(local_rows)
            meta_rows.extend(local_meta)
            for code, stats in split_report.items():
                totals[f"{code}_total"] += stats.get("target_total", 0)
                totals[f"{code}_with_text"] += stats.get("with_text", 0)
                totals[f"{code}_missing_text"] += stats.get("missing_text", 0)
            rep["per_split"][split] = split_report
            continue

        # Fallback: one full streaming scan for both codes.
        print(f"  [{split}] falling back to one full streaming scan for Items {list(SEC_CREDIT_ITEMS)}")
        c = Counter()
        code_stats = {code: Counter() for code in SEC_CREDIT_ITEMS}
        try:
            stream = load_dataset(
                SEC_REPO, "items", split=split, streaming=True, token=HF_TOKEN
            )
            for rec in tqdm(stream, desc=f"SEC items/{split} [full scan]", unit="rows"):
                c["scanned"] += 1
                if SEC_MAX_ROWS_SCANNED and c["scanned"] >= SEC_MAX_ROWS_SCANNED:
                    print(f"  (stopped at SEC_MAX_ROWS_SCANNED={SEC_MAX_ROWS_SCANNED})")
                    break

                code = str(rec.get("item_code")).strip()
                if code not in SEC_CREDIT_ITEMS:
                    continue

                source_label, quality = SEC_CREDIT_ITEMS[code]
                code_stats[code]["target_total"] += 1
                text = rec.get("item_text")
                if is_blank(text):
                    code_stats[code]["missing_text"] += 1
                    continue

                code_stats[code]["with_text"] += 1
                acc = rec.get("accession_no")
                acc_s = "" if acc is None else str(acc).strip()
                sid = f"{acc_s}:{code}" if acc_s else f"{split}:{code}:{c['scanned']}"

                rows.append(make_row(
                    text.strip(), "Credit Event", src, source_label, code,
                    split, sid, quality
                ))
                meta_rows.append({
                    "source_id": sid,
                    "item_code": code,
                    "source_split": split,
                    **{m: (None if rec.get(m) is None else str(rec.get(m))) for m in meta_cols}
                })

            rep["per_split"][split] = {
                "mode": "full_streaming_scan",
                "rows_scanned": c["scanned"],
                **{code: dict(stats) for code, stats in code_stats.items()}
            }
            for code, stats in code_stats.items():
                totals[f"{code}_total"] += stats["target_total"]
                totals[f"{code}_with_text"] += stats["with_text"]
                totals[f"{code}_missing_text"] += stats["missing_text"]

        except Exception as e:
            print(f"  !! full scan failed: {type(e).__name__}: {str(e)[:300]}")

    if meta_rows:
        pd.DataFrame(meta_rows).to_csv(OUT_SEC_META, index=False, encoding="utf-8")
        REPORT["files_written"].append(str(OUT_SEC_META))

    item_report = {}
    for code, (source_label, quality) in SEC_CREDIT_ITEMS.items():
        item_report[code] = {
            "source_label": source_label,
            "mapping_quality": quality,
            "total_records": int(totals[f"{code}_total"]),
            "usable_with_item_text": int(totals[f"{code}_with_text"]),
            "missing_item_text": int(totals[f"{code}_missing_text"]),
        }

    rep.update({
        "credit_item_report": item_report,
        "missing_text": int(sum(totals[f"{c}_missing_text"] for c in SEC_CREDIT_ITEMS)),
        "usable_rows": len(rows),
    })

    print("\n  SEC credit-item summary:")
    for code, stats in item_report.items():
        print(f"    Item {code}: {stats}")
    print(f"  TOTAL usable SEC Credit Event text rows: {len(rows)}")

    return make_df(rows)


# ===== NOTEBOOK CELL 15 =====
EDT_REPO = "agungpambudi/corporate-event-detection"
EDT_FILES = [("train", "datasets/event-detection/train.txt"),
             ("dev",   "datasets/event-detection/dev.txt")]
DETOKENIZE_EDT = True      # light spacing repair when re-joining tokens (set False for plain ' '.join)
EDT_MAX_TAG_VOCAB = 500    # a real tag column is small; a token column would have thousands of values

def edt_norm_tag(tag):
    return re.sub(r"^[BIESLU][-_:]", "", tag.strip())

def edt_is_acquisition(tag):
    n = edt_norm_tag(tag)
    return n == "A" or n.lower() == "acquisition"

def detok(tokens):
    s = " ".join(tokens)
    if DETOKENIZE_EDT:
        s = re.sub(r"\s+([,.;:!?%)\]}])", r"\1", s)
        s = re.sub(r"([(\[{$])\s+", r"\1", s)
        s = re.sub(r"\s+('s|n't|'re|'ve|'ll|'d|'m)\b", r"\1", s)
    return s.strip()

def _edt_split(line, delim):
    return [x.strip() for x in line.split("\t")] if delim else line.split()

def edt_inspect(path, split):
    """Inspect the REAL file format. Returns (ok, info) - ok=False means 'do not parse'."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [l.rstrip("\r") for l in f.read().split("\n")]
    print(f"\n  [{split}] {path}\n  [{split}] total lines: {len(lines)}")
    print(f"  [{split}] first 25 raw lines (repr):")
    for l in lines[:25]:
        print("     ", repr(l))
    body = [l for l in lines if l.strip() and not l.startswith("-DOCSTART-")]
    info = {"split": split, "total_lines": len(lines), "token_lines": len(body)}
    reasons = []
    if not body:
        return False, {**info, "reasons": ["file has no token lines"]}
    if body[0].lstrip().startswith(("{", "[")):
        return False, {**info, "reasons": ["file looks like JSON, not token-per-line text; no parser written for it"]}

    sample = body[:2000]
    tab_frac = float(np.mean(["\t" in l for l in sample]))
    delim = "\t" if tab_frac >= 0.9 else None
    info["delimiter"] = "TAB" if delim else "whitespace"
    ncols = Counter(len(_edt_split(l, delim)) for l in sample)
    info["columns_per_line_sample"] = dict(ncols)
    print(f"  [{split}] delimiter: {info['delimiter']} | fields-per-line (first 2000 lines): {dict(ncols)}")
    card = defaultdict(set)
    for l in body[:20000]:
        for i, v in enumerate(_edt_split(l, delim)):
            if len(card[i]) <= 5000:
                card[i].add(v)
    print(f"  [{split}] distinct values per column index (first 20k lines): "
          f"{ {i: len(v) for i, v in sorted(card.items())} }")

    vocab, malformed, bad_examples = Counter(), 0, []
    for l in body:
        f_ = _edt_split(l, delim)
        if len(f_) < 2:
            malformed += 1
            if len(bad_examples) < 5:
                bad_examples.append(l)
            continue
        vocab[f_[-1]] += 1
    info.update({"malformed_lines": malformed, "tag_vocab_size": len(vocab)})
    shown = vocab.most_common(100)
    print(f"  [{split}] tag column = LAST field. Unique tags: {len(vocab)} (showing up to 100): {shown}")
    if bad_examples:
        print(f"  [{split}] malformed (<2 fields) examples: {bad_examples}")

    acq_tags = sorted(t for t in vocab if edt_is_acquisition(t))
    info["acquisition_tags"] = acq_tags
    print(f"  [{split}] tags interpreted as Acquisition (BIO prefix stripped, == 'A'/'Acquisition'): {acq_tags}")

    if malformed / max(len(body), 1) > 0.01:
        reasons.append(f"{malformed} malformed lines (>1%)")
    if not (2 <= len(vocab) <= EDT_MAX_TAG_VOCAB):
        reasons.append(f"last column has {len(vocab)} distinct values - does not look like a tag column")
    if not acq_tags:
        reasons.append("no tag in the file normalises to 'A' / 'Acquisition'")
    if reasons:
        info["reasons"] = reasons
        return False, {**info, "delim": delim}
    return True, {**info, "delim": delim, "acq_tags": set(acq_tags)}

def edt_parse(path, split, delim, acq_tags):
    docs, toks, tags = [], [], []
    def flush():
        if toks:
            docs.append((list(toks), list(tags)))
        toks.clear(); tags.clear()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if not line.strip() or line.startswith("-DOCSTART-"):
                flush(); continue
            f_ = _edt_split(line, delim)
            if len(f_) < 2:
                continue   # counted as malformed during inspection
            toks.append(f_[0]); tags.append(f_[-1])
    flush()
    out = []
    for i, (t, g) in enumerate(docs):
        hit = sorted({x for x in g if x in acq_tags})
        out.append((i, t, hit))
    return out

def extract_corporate_events():
    banner("SOURCE 5/5 - Corporate Event Detection (event-detection/train.txt + dev.txt)")
    src = "corporate_event_detection"
    rep = REPORT["sources"].setdefault(src, {})
    rep["per_file"] = {}
    repo_files = list_repo_files(EDT_REPO, repo_type="dataset", token=HF_TOKEN)
    ed_files = [f for f in repo_files if f.startswith("datasets/event-detection/")]
    print(f"  files under datasets/event-detection/: {ed_files}")

    rows = []
    n_docs = n_docs_acq = 0
    for split, fname in EDT_FILES:
        if fname not in repo_files:
            print(f"  !! '{fname}' not found in repo; skipping this file.")
            rep["per_file"][split] = {"status": "file not found"}
            continue
        path = hf_hub_download(repo_id=EDT_REPO, filename=fname, repo_type="dataset", token=HF_TOKEN)
        ok, info = edt_inspect(path, split)
        if not ok:
            print(f"\n  !! [{split}] FORMAT NOT SAFELY INTERPRETABLE -> not extracting. Reasons: {info.get('reasons')}")
            rep["per_file"][split] = {"status": "unsafe format", **{k: v for k, v in info.items() if k != 'acq_tags'}}
            continue
        parsed = edt_parse(path, split, info["delim"], info["acq_tags"])
        f_docs, f_acq = len(parsed), 0
        for i, toks, hit in parsed:
            if not hit:
                continue
            text = detok(toks)
            if is_blank(text):
                continue
            f_acq += 1
            rows.append(make_row(text, "Merger / Acquisition", src, "Acquisition", "; ".join(hit), split,
                                 f"{split}_doc{i}", "very_strong_candidate"))
        n_docs += f_docs; n_docs_acq += f_acq
        rep["per_file"][split] = {"status": "ok", "documents_parsed": f_docs, "documents_with_acquisition": f_acq,
                                  "delimiter": info["delimiter"], "acquisition_tags": sorted(info["acq_tags"]),
                                  "malformed_lines": info["malformed_lines"]}
        print(f"  [{split}] documents parsed={f_docs}  containing Acquisition={f_acq}")
    rep.update({"documents_parsed": n_docs, "documents_with_acquisition": n_docs_acq,
                "missing_text": 0, "usable_rows": len(rows)})
    return make_df(rows)


# ===== NOTEBOOK CELL 17 =====
CATALYST_MAP = {
    "MERGER_ANNOUNCEMENT": ("Merger / Acquisition", "strong_candidate"),
    "PRODUCT_LAUNCHES": ("Product Launch", "strong_candidate"),
}

def extract_catalyst_labeled():
    """
    Adds only manually labeled Catalyst rows.
    Synthetic rows in the source dataset are explicitly excluded.

    We also require confidence >= CATALYST_MIN_CONFIDENCE to keep this source
    conservative for the first real-data extension.
    """
    banner("SOURCE 6/6 - Catalyst financial news (labeled-only M&A + Product Launch)")
    src = "catalyst_financial_news"
    rep = REPORT["sources"].setdefault(src, {})

    print_configs(CATALYST_REPO)
    ds = safe_load_dataset(CATALYST_REPO)
    describe_dataset(ds)

    rows = []
    rep["per_split"] = {}
    totals = Counter()

    required = {"catalyst_type", "source"}
    text_candidates = ["text", "headline", "summary"]
    id_candidates = ["id", "uuid", "source_id"]

    for split in ds.keys():
        d = ds[split]
        missing = required - set(d.column_names)
        if missing:
            print(f"  !! [{split}] missing required columns {sorted(missing)}; skipping.")
            continue

        if not any(c in d.column_names for c in text_candidates):
            print(f"  !! [{split}] has no usable text/headline/summary column; skipping.")
            continue

        id_col = first_present(d.column_names, id_candidates)
        c = Counter()

        keep_cols = list(dict.fromkeys(
            [x for x in ["id", "text", "headline", "summary", "catalyst_type",
                         "source", "confidence", "symbol"] if x in d.column_names]
        ))
        sub = d.select_columns(keep_cols)

        for idx, ex in enumerate(tqdm(sub, desc=f"catalyst/{split}", unit="rows")):
            c["seen"] += 1

            source_kind = str(ex.get("source", "")).strip().lower()
            if source_kind != "labeled":
                c["excluded_non_labeled"] += 1
                continue

            cat = str(ex.get("catalyst_type", "")).strip()
            if cat not in CATALYST_MAP:
                c["excluded_other_catalyst"] += 1
                continue

            conf = ex.get("confidence")
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = np.nan

            if np.isfinite(conf_f) and conf_f < CATALYST_MIN_CONFIDENCE:
                c["excluded_low_confidence"] += 1
                continue

            text = ex.get("text")
            if is_blank(text):
                headline = "" if is_blank(ex.get("headline")) else str(ex.get("headline")).strip()
                summary = "" if is_blank(ex.get("summary")) else str(ex.get("summary")).strip()
                text = (headline + " " + summary).strip()

            if is_blank(text):
                c["missing_text"] += 1
                continue

            label, quality = CATALYST_MAP[cat]
            sid_raw = ex.get(id_col) if id_col else None
            sid = str(sid_raw).strip() if sid_raw is not None else f"{split}_{idx}"

            subtype_parts = ["source=labeled"]
            if np.isfinite(conf_f):
                subtype_parts.append(f"confidence={conf_f:.3f}")
            symbol = ex.get("symbol")
            if symbol is not None and str(symbol).strip():
                subtype_parts.append(f"symbol={str(symbol).strip()}")

            rows.append(make_row(
                str(text).strip(),
                label,
                src,
                cat,
                ";".join(subtype_parts),
                split,
                sid,
                quality,
            ))
            c[f"accepted_{cat}"] += 1

        rep["per_split"][split] = dict(c)
        totals.update(c)
        print(f"  [{split}] {dict(c)}")

    rep.update({
        "source_policy": "source == labeled only",
        "minimum_confidence": CATALYST_MIN_CONFIDENCE,
        "synthetic_rows_explicitly_excluded": int(totals["excluded_non_labeled"]),
        "missing_text": int(totals["missing_text"]),
        "usable_rows": len(rows),
        "accepted_merger_announcement": int(totals["accepted_MERGER_ANNOUNCEMENT"]),
        "accepted_product_launches": int(totals["accepted_PRODUCT_LAUNCHES"]),
    })

    print(
        f"  Accepted labeled Catalyst rows: {len(rows)} | "
        f"M&A={totals['accepted_MERGER_ANNOUNCEMENT']} | "
        f"Product={totals['accepted_PRODUCT_LAUNCHES']} | "
        f"excluded source!=labeled={totals['excluded_non_labeled']}"
    )
    return make_df(rows)


# ===== NOTEBOOK CELL 19 =====
EXTRACTORS = [
    ("twitter_financial_news_topic", extract_twitter),
    ("gdelt_news_events",            extract_gdelt),
    ("sentivent",                    extract_sentivent),
    ("sec_8k_events",                extract_sec_8k),
    ("corporate_event_detection",    extract_corporate_events),
    ("catalyst_financial_news",      extract_catalyst_labeled),
]

frames = {}
for name, fn in EXTRACTORS:
    try:
        df_src = fn()
        REPORT["sources"].setdefault(name, {})["status"] = "ok" if len(df_src) else "ok_but_empty"
        frames[name] = df_src
        print(f"\n>>> {name}: {len(df_src)} rows extracted")
    except Exception as e:
        REPORT["sources"].setdefault(name, {})["status"] = f"FAILED: {type(e).__name__}: {str(e)[:300]}"
        print(f"\n!!! {name} FAILED - full traceback below; continuing with remaining sources.")
        traceback.print_exc()

print("\nSource status:")
for name, _ in EXTRACTORS:
    print(f"  {name:<32} {REPORT['sources'].get(name, {}).get('status')}  rows={len(frames.get(name, []))}")


# ===== NOTEBOOK CELL 21 =====
def validate_and_standardize(df):
    stats = {}
    df = df.copy()
    for c in SCHEMA_COLS:
        if c not in df.columns:
            raise ValueError(f"missing schema column: {c}")
    df = df[SCHEMA_COLS]
    stats["rows_in"] = len(df)

    null_text = df["text"].isna() | ~df["text"].map(lambda x: isinstance(x, str))
    stats["null_or_non_string_text_dropped"] = int(null_text.sum())
    df = df[~null_text].copy()

    df["text"] = df["text"].str.strip()                       # whitespace only - nothing semantic
    empty = df["text"] == ""
    stats["empty_after_strip_dropped"] = int(empty.sum())
    df = df[~empty].copy()

    fixed = df["text"].map(lambda s: re.sub(r"[\ud800-\udfff]", "\ufffd", s))
    stats["utf8_repaired_rows"] = int((fixed != df["text"]).sum())
    df["text"] = fixed

    for c in ["source_label", "source_subtype", "source_id", "source_split"]:
        df[c] = df[c].fillna("").astype(str)

    bad_label = ~df["label"].isin(LABEL_TO_ID)
    if bad_label.any():
        raise ValueError(f"unexpected labels: {df.loc[bad_label, 'label'].unique()}")
    if (df["label"].map(LABEL_TO_ID) != df["label_id"]).any():
        raise ValueError("label_id disagrees with label for some rows")
    df["label_id"] = df["label_id"].astype(int)
    if not set(df["source_dataset"]) <= ALLOWED_SOURCES:
        raise ValueError(f"unexpected source_dataset: {set(df['source_dataset']) - ALLOWED_SOURCES}")
    if not set(df["mapping_quality"]) <= ALLOWED_QUALITY:
        raise ValueError(f"unexpected mapping_quality: {set(df['mapping_quality']) - ALLOWED_QUALITY}")

    stats["rows_out"] = len(df)
    return df.reset_index(drop=True), stats

non_empty = [frames[n] for n, _ in EXTRACTORS if n in frames and len(frames[n])]
if not non_empty:
    raise RuntimeError("No source produced any rows - see the diagnostics printed above.")

df_all = pd.concat(non_empty, ignore_index=True)       # source order preserved, no shuffling
df, val_stats = validate_and_standardize(df_all)
REPORT["final"]["validation"] = val_stats
print("Validation:", val_stats)

df.to_csv(OUT_CSV, index=False, encoding="utf-8")
df.to_parquet(OUT_PARQUET, index=False)
REPORT["files_written"] += [str(OUT_CSV), str(OUT_PARQUET)]
print(f"Saved {OUT_CSV}\nSaved {OUT_PARQUET}")


# ===== NOTEBOOK CELL 23 =====
by_class   = df.groupby(["label_id", "label"]).size()
by_source  = df["source_dataset"].value_counts()
by_quality = df["mapping_quality"].value_counts()
cross      = pd.crosstab(df["label"], df["source_dataset"])

S = REPORT["sources"]
REPORT["final"].update({
    "A_total_rows": int(len(df)),
    "B_rows_per_class": {f"{i}:{l}": int(n) for (i, l), n in by_class.items()},
    "C_rows_per_source": {k: int(v) for k, v in by_source.items()},
    "D_rows_per_class_and_source": {l: {s: int(cross.loc[l, s]) for s in cross.columns} for l in cross.index},
    "E_rows_per_mapping_quality": {k: int(v) for k, v in by_quality.items()},
    "F_missing_text_encountered": {n: int(S.get(n, {}).get("missing_text", 0)) for n, _ in EXTRACTORS},
    "G_sentivent_multiclass_conflicts": int(S.get("sentivent", {}).get("multiclass_conflicts", 0)),
    "H_sec_credit_items": S.get("sec_8k_events", {}).get("credit_item_report", {}),
    "H2_catalyst_labeled": {
        "usable_rows": S.get("catalyst_financial_news", {}).get("usable_rows"),
        "accepted_merger_announcement": S.get("catalyst_financial_news", {}).get("accepted_merger_announcement"),
        "accepted_product_launches": S.get("catalyst_financial_news", {}).get("accepted_product_launches"),
        "synthetic_rows_excluded": S.get("catalyst_financial_news", {}).get("synthetic_rows_explicitly_excluded"),
        "minimum_confidence": S.get("catalyst_financial_news", {}).get("minimum_confidence"),
    },
    "I_corporate_event_detection": {
        "documents_parsed": S.get("corporate_event_detection", {}).get("documents_parsed"),
        "documents_with_acquisition": S.get("corporate_event_detection", {}).get("documents_with_acquisition"),
    },
    "J_gdelt": {
        "n_sequences": S.get("gdelt_news_events", {}).get("n_sequences"),
        "n_exploded_individual_events": S.get("gdelt_news_events", {}).get("usable_rows"),
        "n_event_pairs_seen": S.get("gdelt_news_events", {}).get("n_exploded_event_pairs"),
    },
})

summary = (df.groupby(["source_dataset", "label_id", "label", "mapping_quality"])
             .size().reset_index(name="n_rows")
             .sort_values(["source_dataset", "label_id", "mapping_quality"]).reset_index(drop=True))
summary.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8")
REPORT["files_written"].append(str(OUT_SUMMARY_CSV))

def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (set, tuple)): return list(o)
    return str(o)

REPORT["files_written"].append(str(OUT_REPORT_JSON))
with open(OUT_REPORT_JSON, "w", encoding="utf-8") as f:
    json.dump(REPORT, f, indent=2, ensure_ascii=False, default=_json_default)

banner("EXTRACTION REPORT")
print(f"A. Total rows: {len(df)}\n")
print("B. Rows per class:");               print(by_class.to_string(), "\n")
print("C. Rows per source dataset:");      print(by_source.to_string(), "\n")
print("D. Class x source:");               print(cross.to_string(), "\n")
print("E. Rows per mapping_quality:");     print(by_quality.to_string(), "\n")
print("F. Missing-text counts encountered:", REPORT["final"]["F_missing_text_encountered"], "\n")
print("G. SENTiVENT multiclass conflicts:", REPORT["final"]["G_sentivent_multiclass_conflicts"], "\n")
print("H. SEC Credit Items:", REPORT["final"]["H_sec_credit_items"], "\n")
print("H2. Catalyst labeled:", REPORT["final"]["H2_catalyst_labeled"], "\n")
print("I. Corporate Event Detection:", REPORT["final"]["I_corporate_event_detection"], "\n")
print("J. GDELT:", REPORT["final"]["J_gdelt"], "\n")
print("Files written:"); [print("  ", p) for p in REPORT["files_written"]]


# ===== NOTEBOOK CELL 25 =====
EXPECTED_IDS = {"Geopolitical": 0, "Macroeconomic": 1, "Credit Event": 2,
                "Merger / Acquisition": 3, "Product Launch": 4}

print("1. shape:", df.shape)
print("\n2. head(20):");  display(df.head(20))
print("\n3. random sample:"); display(df.sample(min(20, len(df)), random_state=42))
print("\n4. class distribution:");   print(df["label"].value_counts().to_string())
print("\n5. source distribution:");  print(df["source_dataset"].value_counts().to_string())
print("\n6. class x source:");       display(pd.crosstab(df["label"], df["source_dataset"], margins=True))
print("\n7. mapping-quality distribution:"); print(df["mapping_quality"].value_counts().to_string())

assert set(df["label"]) <= set(EXPECTED_IDS), f"unexpected labels: {set(df['label']) - set(EXPECTED_IDS)}"
for lab, lid in EXPECTED_IDS.items():
    sub = df[df["label"] == lab]
    assert (sub["label_id"] == lid).all(), f"label_id mismatch for {lab}"
assert df["text"].map(lambda s: isinstance(s, str) and s.strip() != "").all()
assert list(df.columns) == SCHEMA_COLS
missing_classes = set(EXPECTED_IDS) - set(df["label"])
if missing_classes:
    print(f"\nWARNING: no rows for classes {missing_classes} - check the source diagnostics above.")
print("\nAll sanity assertions passed.")

print("\n13. targeted-extension counts:")
if "catalyst_financial_news" in frames:
    print("Catalyst labeled rows:", len(frames["catalyst_financial_news"]))
if "sec_8k_events" in frames:
    print("SEC credit-item rows:", len(frames["sec_8k_events"]))

print("\nNOTE: this is still a RAW collection dataset. Run the downstream preprocessing notebook before model training.")

