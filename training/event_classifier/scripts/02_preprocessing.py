"""Faithful code-cell export of event-classification-preprocessing(1).ipynb.
Execution order follows original notebook. Notebook/Kaggle inputs are required.
This file is a SOURCE EXPORT, not a verified local standalone retraining command.
Do not claim this script was executed during packaging.
"""



# ===== NOTEBOOK CELL 1 =====
# ============================== SETUP ==============================
import os, re, json, glob, html, unicodedata, warnings, random
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60)
pd.set_option("display.max_colwidth", 140); pd.set_option("display.max_rows", 200)

LABELS   = {0: "Geopolitical", 1: "Macroeconomic", 2: "Credit Event", 3: "Merger / Acquisition", 4: "Product Launch"}
LABEL2ID = {v: k for k, v in LABELS.items()}
EXPECTED_RAW = {"Geopolitical": 15013, "Macroeconomic": 3909, "Credit Event": 241,
                "Merger / Acquisition": 931, "Product Launch": 4965}   # sanity reference only
EXPECTED_COLS = ["text", "label", "label_id", "source_dataset", "source_label",
                 "source_subtype", "source_split", "source_id", "mapping_quality"]
QUALITY_RANK = {"very_strong_candidate": 0, "strong_candidate": 1, "broad_candidate": 2}

NEAR_DUP_THRESHOLD = 0.92     # cosine on word 1-2gram TF-IDF (conservative)
NEAR_DUP_MAX_CHARS = 1500     # only the first N chars are compared (bounds cost for ~3k-char documents)
MIN_REAL_PER_CLASS = 500      # warn below this many clean REAL examples

def find_input(name, required=True):
    cands = (glob.glob(f"/kaggle/input/**/{name}", recursive=True)
             + [f"/kaggle/working/{name}", f"/mnt/user-data/uploads/{name}", f"./{name}"])
    for c in cands:
        if os.path.isfile(c):
            return c
    if required:
        raise FileNotFoundError(f"Required input not found: {name}. Add it to the notebook inputs or /kaggle/working.")
    return None

WORK_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("./working")
os.makedirs(WORK_DIR, exist_ok=True)
def out(name): return os.path.join(WORK_DIR, name)

FLAGS = []      # every uncertainty/anomaly is collected here and written to the report
def flag(msg):
    print("FLAG: " + msg); FLAGS.append(msg)

STATS = {}      # numbers for the final report
print("work dir:", WORK_DIR)


# ===== NOTEBOOK CELL 2 =====
# ============================== STEP 1 — INSPECT INPUT ==============================
RAW_PATH = find_input("final_raw_event_dataset.csv")
print("reading", RAW_PATH)
# keep_default_na=False so literal strings such as "NA"/"None" inside texts are not turned into NaN
raw = pd.read_csv(RAW_PATH, keep_default_na=False, na_values=[""])
raw.insert(0, "raw_row_id", np.arange(len(raw)))      # stable row id for provenance

print("shape:", raw.shape)
print("\ncolumns:", list(raw.columns))
print("\ndtypes:\n", raw.dtypes)
print("\nmissing values:\n", raw.isna().sum())

missing_cols = [c for c in EXPECTED_COLS if c not in raw.columns]
extra_cols = [c for c in raw.columns if c not in EXPECTED_COLS + ["raw_row_id"]]
if missing_cols: flag(f"Expected columns missing from input: {missing_cols}")
if extra_cols:   flag(f"Unexpected extra columns present (kept untouched): {extra_cols}")
assert "text" in raw.columns and "label" in raw.columns and "label_id" in raw.columns, "text/label/label_id are required"
for c in EXPECTED_COLS:
    if c not in raw.columns: raw[c] = np.nan   # keep schema stable downstream (flagged above)

print("\n--- label counts ---");           print(raw.label.value_counts(dropna=False))
print("\n--- label_id counts ---");        print(raw.label_id.value_counts(dropna=False).sort_index())
print("\n--- source_dataset counts ---");  print(raw.source_dataset.value_counts(dropna=False))
print("\n--- mapping_quality counts ---"); print(raw.mapping_quality.value_counts(dropna=False))
print("\n--- class x source_dataset ---"); print(pd.crosstab(raw.label, raw.source_dataset, margins=True))
print("\n--- class x mapping_quality ---");print(pd.crosstab(raw.label, raw.mapping_quality, margins=True))
print("\n--- source_dataset x source_label ---"); print(raw.groupby(["source_dataset", "source_label"]).size())

# ---- label <-> label_id verification (flag, never silently fix) ----
bad_label = ~raw.label.isin(LABEL2ID)
bad_id    = ~raw.label_id.isin(list(LABELS))
mismatch  = (raw.label.map(LABEL2ID) != raw.label_id) & ~bad_label & ~bad_id
raw["_label_problem"] = np.where(bad_label, "unknown_label", np.where(bad_id, "unknown_label_id", np.where(mismatch, "label_label_id_mismatch", "")))
if (raw._label_problem != "").any():
    flag(f"{(raw._label_problem != '').sum()} rows have invalid/inconsistent label or label_id -> moved to technical_failures.csv (NOT auto-fixed). Breakdown: {raw._label_problem[raw._label_problem!=''].value_counts().to_dict()}")
else:
    print("\nOK: every label string agrees with label_id (0..4 mapping as specified).")
print("observed mapping:", raw.groupby("label").label_id.unique().to_dict())

# ---- compare with expected counts ----
actual = raw.label.value_counts().to_dict()
cmp_tbl = pd.DataFrame({"expected": EXPECTED_RAW, "actual": {k: actual.get(k, 0) for k in EXPECTED_RAW}})
cmp_tbl["diff"] = cmp_tbl.actual - cmp_tbl.expected
print("\n--- actual vs expected raw counts ---"); print(cmp_tbl, "\nTOTAL actual:", len(raw), "expected:", sum(EXPECTED_RAW.values()))
if (cmp_tbl["diff"] != 0).any():
    flag(f"Raw class counts differ from the expected reference: {cmp_tbl['diff'].to_dict()}  (no data was modified to match)")
STATS["original_rows"] = int(len(raw))
STATS["raw_class_counts"] = {k: int(v) for k, v in raw.label.value_counts().items()}
STATS["raw_class_vs_expected_diff"] = {k: int(v) for k, v in cmp_tbl["diff"].items()}

# ---- cross-check against extraction_report.json / dataset_source_summary.csv ----
rep_path = find_input("extraction_report.json", required=False)
sum_path = find_input("dataset_source_summary.csv", required=False)
EXTRACTION = None
if rep_path:
    EXTRACTION = json.load(open(rep_path))
    rep_cls = EXTRACTION["final"]["B_rows_per_class"]
    rep_cls = {k.split(":", 1)[1]: v for k, v in rep_cls.items()}
    if rep_cls != {k: int(v) for k, v in raw.label.value_counts().items()}:
        flag("extraction_report.json class counts differ from the uploaded CSV.")
    else:
        print("OK: class counts match extraction_report.json")
    rep_src = EXTRACTION["final"]["C_rows_per_source"]
    if rep_src != {k: int(v) for k, v in raw.source_dataset.value_counts().items()}:
        flag("extraction_report.json source counts differ from the uploaded CSV.")
    sec = EXTRACTION["sources"].get("sec_8k_events", {})
    if sec.get("item_103_missing_text", 0) > 0:
        flag(f"SEC 8-K Item 1.03: only {sec.get('item_103_with_text')} of {sec.get('item_103_total_records')} bankruptcy records have text "
             f"({sec.get('item_103_missing_text')} had NO item text). This is the largest recoverable source of Credit Event examples.")
else:
    flag("extraction_report.json not found - cross-check skipped.")
if sum_path:
    print("\ndataset_source_summary.csv:\n", pd.read_csv(sum_path))
else:
    flag("dataset_source_summary.csv not found - cross-check skipped.")

conf_path = find_input("sentivent_multiclass_conflicts.csv", required=False)
SENTI_CONFLICT_TEXTS = set()
if conf_path:
    _c = pd.read_csv(conf_path, keep_default_na=False, na_values=[""])
    print("\nsentivent_multiclass_conflicts.csv columns:", list(_c.columns), "rows:", len(_c))
    tcol = next((c for c in _c.columns if c.lower() in ("text", "sentence", "raw_text")), None)
    if tcol is None:
        flag("sentivent_multiclass_conflicts.csv has no recognizable text column - not used.")
    else:
        SENTI_CONFLICT_TEXTS = set(_c[tcol].astype(str))
else:
    flag("sentivent_multiclass_conflicts.csv was NOT provided (the 124 SENTiVENT multi-class sentences are per the report already excluded from the raw CSV). Overlap check skipped.")

# ---- provenance / structure checks ----
for c in ["source_dataset", "source_label", "source_id", "mapping_quality", "source_split"]:
    n = int(raw[c].isna().sum())
    if n: flag(f"{n} rows have missing provenance column '{c}'.")
dup_ids = raw.duplicated(["source_dataset", "source_id"]).sum()
if dup_ids: flag(f"{dup_ids} rows share the same (source_dataset, source_id) - possible parser duplication.")
unknown_q = set(raw.mapping_quality.dropna()) - set(QUALITY_RANK)
if unknown_q: flag(f"Unknown mapping_quality values: {unknown_q}")

# ---- structural confounds worth knowing about ----
share = raw.groupby("label").source_dataset.agg(lambda s: s.value_counts(normalize=True).iloc[0]).round(3)
dom = raw.groupby("label").source_dataset.agg(lambda s: s.value_counts().index[0])
print("\ndominant source per class (share):"); print(pd.DataFrame({"dominant_source": dom, "share": share}))
raw["_nchar"] = raw.text.astype(str).str.len()
print("\ntext length (chars) by source:"); print(raw.groupby("source_dataset")._nchar.describe()[["count", "mean", "50%", "max"]].round(0))


# ===== NOTEBOOK CELL 4 =====
# ============================== STEP 2 — NORMALIZATION ==============================
URL_RE     = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"(?<![\w])@\w{1,30}")
ZW_RE      = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
QUOTE_MAP  = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                            "\u2013": "-", "\u2014": "-", "\u00a0": " "})
CTRL_RE    = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

def normalize_text(t):
    # lowercase / placeholders apply ONLY to this helper column. Numbers, punctuation, symbols are kept.
    t = unicodedata.normalize("NFKC", t)
    t = html.unescape(t)
    t = ZW_RE.sub("", t)
    t = t.translate(QUOTE_MAP)
    t = t.lower()
    t = URL_RE.sub(" <url> ", t)
    t = MENTION_RE.sub(" <user> ", t)
    t = re.sub(r"(?:<url>\s*){2,}", "<url> ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tech_failure(orig, norm):
    if not isinstance(orig, str):
        return "null_or_non_string_text"
    if orig.strip() == "":
        return "empty_text"
    if orig.count("\ufffd") >= 3 or orig.count("\ufffd") / len(orig) > 0.02:
        return "corrupted_replacement_characters"
    if CTRL_RE.search(orig):
        return "control_characters"
    letters = sum(ch.isalpha() for ch in norm.replace("<url>", "").replace("<user>", ""))
    if letters < 5:
        return "no_textual_content"
    return None

raw["normalized_text"] = [normalize_text(t) if isinstance(t, str) else "" for t in raw.text]
reasons = [tech_failure(t, n) for t, n in zip(raw.text, raw.normalized_text)]
raw["_tech_reason"] = [r if r else (lp if lp else "") for r, lp in zip(reasons, raw._label_problem)]

tech_fail = raw[raw._tech_reason != ""].copy()
tech_fail.rename(columns={"_tech_reason": "failure_reason"}).drop(columns=["_label_problem", "_nchar"]).to_csv(out("technical_failures.csv"), index=False)
print("technical failures:", len(tech_fail)); print(tech_fail._tech_reason.value_counts())

work = raw[raw._tech_reason == ""].drop(columns=["_tech_reason", "_label_problem", "_nchar"]).copy()
STATS["technical_failures_removed"] = int(len(tech_fail))
STATS["technical_failure_reasons"] = {k: int(v) for k, v in tech_fail._tech_reason.value_counts().items()}
print("rows kept after technical cleaning:", len(work))

print("\nnormalization examples:")
for t in work.sample(4, random_state=SEED).text:
    print("  ORIG:", repr(t[:140])); print("  NORM:", repr(normalize_text(t)[:140]))

if SENTI_CONFLICT_TEXTS:
    sc_norm = {normalize_text(t) for t in SENTI_CONFLICT_TEXTS}
    ov = work.normalized_text.isin(sc_norm).sum()
    if ov: flag(f"{ov} raw rows match texts listed in sentivent_multiclass_conflicts.csv (multi-class sentences should have been excluded).")
    else: print("OK: no raw row overlaps with sentivent_multiclass_conflicts.csv")


# ===== NOTEBOOK CELL 6 =====
# ============================== STEP 3 — EXACT DUPLICATES ==============================
w = work.copy()
w["_qrank"] = w.mapping_quality.map(QUALITY_RANK).fillna(9)
g = w.groupby("normalized_text", sort=False)
w["_grp_size"]    = g.raw_row_id.transform("size")
w["_grp_nlabels"] = g.label_id.transform("nunique")

# ---- Case B: cross-label conflicts ----
conf_mask = w._grp_nlabels > 1
exact_conf = w[conf_mask].copy()
exact_conf["exact_conflict_group_id"] = exact_conf.groupby("normalized_text", sort=False).ngroup()
cols_conf = ["exact_conflict_group_id", "text", "label", "label_id", "source_dataset", "source_label", "source_subtype",
             "source_split", "source_id", "mapping_quality", "raw_row_id", "normalized_text"]
exact_conf = exact_conf.sort_values(["exact_conflict_group_id", "raw_row_id"])[cols_conf]
exact_conf.to_csv(out("exact_label_conflicts.csv"), index=False)      # written even if empty (header only)

# ---- Case A / C: same label ----
ok = w[~conf_mask].sort_values(["_qrank", "raw_row_id"])
def _u(s): return "|".join(sorted(set(map(str, s))))
agg = ok.groupby("normalized_text", sort=False).agg(
    duplicate_count=("raw_row_id", "size"),
    all_source_datasets=("source_dataset", _u),
    all_source_labels=("source_label", _u),
    all_mapping_qualities=("mapping_quality", _u))
reps = ok.drop_duplicates("normalized_text", keep="first").merge(agg, left_on="normalized_text", right_index=True, how="left")
reps = reps.sort_values("raw_row_id").drop(columns=["_qrank", "_grp_size", "_grp_nlabels"]).reset_index(drop=True)

n_dup_groups = int((agg.duplicate_count > 1).sum())
n_removed_same = int(len(ok) - len(reps))
n_conf_groups = int(exact_conf.exact_conflict_group_id.nunique()) if len(exact_conf) else 0
print(f"exact duplicate groups (same label, size>1) : {n_dup_groups}")
print(f"same-label duplicate rows removed            : {n_removed_same}")
print(f"cross-label exact conflict groups / rows     : {n_conf_groups} / {len(exact_conf)}  -> exact_label_conflicts.csv")
print(f"representatives kept                         : {len(reps)}")

removed_by_src = ok[ok.duplicated("normalized_text", keep="first")].source_dataset.value_counts()
print("\nsame-label duplicates removed, by source of the removed row:"); print(removed_by_src)
multi = reps[reps.all_source_datasets.str.contains(r"\|")]
print("\ngroups whose duplicates span >1 source dataset:", len(multi))
print("largest duplicate groups:"); print(reps.nlargest(5, "duplicate_count")[["text", "label", "source_dataset", "duplicate_count"]])
for src, n in removed_by_src.items():
    tot = (work.source_dataset == src).sum()
    if n / tot > 0.10: flag(f"{src}: {n} of {tot} rows ({n/tot:.1%}) were exact duplicates - strong sign of templated/repeated text in this source.")

STATS.update(exact_duplicate_groups=n_dup_groups, exact_duplicates_removed=n_removed_same,
             exact_conflict_groups=n_conf_groups, exact_conflict_rows=int(len(exact_conf)),
             exact_duplicates_removed_by_source={k: int(v) for k, v in removed_by_src.items()})
assert len(work) == len(reps) + n_removed_same + len(exact_conf), "exact-dup accounting mismatch"


# ===== NOTEBOOK CELL 8 =====
# ============================== STEP 4 — NEAR DUPLICATES ==============================
dd = reps.reset_index(drop=True)
n = len(dd)
cmp_text = (dd.normalized_text.str.replace("<url>", " ", regex=False).str.replace("<user>", " ", regex=False)
            .str.slice(0, NEAR_DUP_MAX_CHARS))
vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b",
                      sublinear_tf=True, min_df=1, dtype=np.float32)
X = vec.fit_transform(cmp_text).tocsr()
XT = X.T.tocsc()
print("tf-idf matrix:", X.shape, "nnz:", X.nnz)

NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
nums = [frozenset(NUM_RE.findall(t)) for t in cmp_text]
is_long = (cmp_text.str.len() > 600).values
def numbers_compatible(i, j):
    a, b = nums[i], nums[j]
    if a == b: return True
    if is_long[i] and is_long[j]:                 # long documents: allow small numeric differences
        return len(a & b) / max(1, len(a | b)) >= 0.8
    return False

edge_i, edge_j, edge_s = [], [], []
n_blocked_by_numbers = 0
CHUNK = 500
for s in range(0, n, CHUNK):
    S = (X[s:s + CHUNK] @ XT).tocoo()
    m = (S.data >= NEAR_DUP_THRESHOLD - 1e-6) & (S.col > S.row + s)       # upper triangle only
    for r, c, v in zip(S.row[m] + s, S.col[m], S.data[m]):
        if numbers_compatible(r, c):
            edge_i.append(r); edge_j.append(c); edge_s.append(float(v))
        else:
            n_blocked_by_numbers += 1
print(f"near-duplicate edges (>= {NEAR_DUP_THRESHOLD}): {len(edge_i)}   blocked by number-mismatch guard: {n_blocked_by_numbers}")

if edge_i:
    A = sparse.csr_matrix((np.ones(len(edge_i), dtype=np.int8), (edge_i, edge_j)), shape=(n, n))
    _, comp = connected_components(A, directed=False)
else:
    comp = np.arange(n)
dd["near_dup_cluster_id"] = comp
csize = dd.groupby("near_dup_cluster_id").raw_row_id.transform("size")
cnl   = dd.groupby("near_dup_cluster_id").label_id.transform("nunique")
dd["_csize"] = csize
print("cluster size distribution (size>1):", dd[dd._csize > 1].groupby("near_dup_cluster_id").size().value_counts().sort_index().to_dict())
if csize.max() > 50: flag(f"Largest near-duplicate cluster has {csize.max()} members - possible chaining; inspect near_duplicates_removed.csv.")

# ---- cross-label clusters ----
nd_conf = dd[cnl > 1].copy()
cols_nd = ["near_dup_cluster_id", "text", "label", "label_id", "source_dataset", "source_label", "source_subtype",
           "source_split", "source_id", "mapping_quality", "raw_row_id", "duplicate_count", "normalized_text"]
nd_conf.sort_values(["near_dup_cluster_id", "raw_row_id"])[cols_nd].to_csv(out("near_duplicate_label_conflicts.csv"), index=False)
n_nd_conf_clusters = int(nd_conf.near_dup_cluster_id.nunique()) if len(nd_conf) else 0
print(f"near-duplicate CROSS-LABEL clusters / rows: {n_nd_conf_clusters} / {len(nd_conf)} -> near_duplicate_label_conflicts.csv")

# ---- same-label clusters: keep one representative, merge provenance ----
same = dd[cnl == 1].copy()
same["_qrank"] = same.mapping_quality.map(QUALITY_RANK).fillna(9)
same = same.sort_values(["_qrank", "raw_row_id"])
def _union(s):
    return "|".join(sorted({p for v in s for p in str(v).split("|") if p and p != "nan"}))
nd_agg = same.groupby("near_dup_cluster_id").agg(
    near_dup_count=("raw_row_id", "size"),
    duplicate_count=("duplicate_count", "sum"),
    all_source_datasets=("all_source_datasets", _union),
    all_source_labels=("all_source_labels", _union),
    all_mapping_qualities=("all_mapping_qualities", _union))
keep_nd = same.drop_duplicates("near_dup_cluster_id", keep="first").drop(
    columns=["duplicate_count", "all_source_datasets", "all_source_labels", "all_mapping_qualities", "_qrank", "_csize"])
keep_nd = keep_nd.merge(nd_agg, left_on="near_dup_cluster_id", right_index=True, how="left").sort_values("raw_row_id").reset_index(drop=True)
removed_nd = same[same.duplicated("near_dup_cluster_id", keep="first")]
n_removed_nd = len(removed_nd)

rep_of = same.drop_duplicates("near_dup_cluster_id", keep="first").set_index("near_dup_cluster_id")[["raw_row_id", "text"]]
audit = removed_nd[["near_dup_cluster_id", "raw_row_id", "text", "label", "source_dataset", "mapping_quality"]].copy()
audit["kept_raw_row_id"] = audit.near_dup_cluster_id.map(rep_of.raw_row_id)
audit["kept_text"] = audit.near_dup_cluster_id.map(rep_of.text)
audit.to_csv(out("near_duplicates_removed.csv"), index=False)

print(f"near-duplicate rows removed (same label)   : {n_removed_nd}")
print("removed by source:", removed_nd.source_dataset.value_counts().to_dict())
for src, k in removed_nd.source_dataset.value_counts().items():
    tot = (dd.source_dataset == src).sum()
    if k / tot > 0.10: flag(f"{src}: {k} of {tot} rows ({k/tot:.1%}) removed as near-duplicates - review near_duplicates_removed.csv.")
print("\nsample merged pairs (removed -> kept):")
for _, r in audit.sample(min(8, len(audit)), random_state=SEED).iterrows():
    print("  X", r.text[:110].replace("\n", " ")); print("  K", r.kept_text[:110].replace("\n", " ")); print()
if n_nd_conf_clusters:
    print("sample cross-label clusters:")
    for cid, grp in list(nd_conf.groupby("near_dup_cluster_id"))[:5]:
        for _, r in grp.iterrows(): print(f"  [{r.label}] {r.text[:110]}")
        print()

STATS.update(near_duplicate_edges=len(edge_i), near_duplicates_removed=int(n_removed_nd),
             near_duplicate_conflict_clusters=n_nd_conf_clusters, near_duplicate_conflict_rows=int(len(nd_conf)),
             near_duplicate_pairs_blocked_by_number_guard=int(n_blocked_by_numbers), near_dup_threshold=NEAR_DUP_THRESHOLD)
assert len(dd) == len(keep_nd) + n_removed_nd + len(nd_conf), "near-dup accounting mismatch"
cand = keep_nd.copy()      # candidates for semantic validation
print("\ncandidates entering semantic validation:", len(cand))


# ===== NOTEBOOK CELL 10 =====
# ============================== STEPS 5-11 — SEMANTIC RULES ==============================
def rx(p): return re.compile(p, re.I)
def hits(pattern, text, k=6):
    found = []
    for m in pattern.finditer(text):
        t = m.group(0).lower().strip()
        if t not in found: found.append(t)
        if len(found) >= k: break
    return found

# ---------- Geopolitical (Twitter Politics) ----------
GEO_STRONG = rx(r"\b(sanction(?:s|ed|ing)?|export controls?|embargo(?:es|ed)?|tariffs?|trade war|trade (?:restrictions?|ban|dispute|tensions?|curbs?)|"
                r"wars?|invasion|invad(?:e|es|ed|ing)|military|missiles?|troops?|nato|cease-?fire|air ?strikes?|nuclear\w*|weaponry|armed conflict|conflicts?|"
                r"annex(?:ed|ation|es)?|territorial|border (?:clash(?:es)?|dispute|crisis|conflict|tensions?)|embass(?:y|ies)|ambassadors?|"
                r"treaty|geopolitic\w*|blockade|hostages?|espionage|proxy war|arms (?:sales|deal|embargo)|peace (?:talks|deal)|summit|"
                r"diplomat(?:s|ic)?|foreign (?:minister|policy|ministry|affairs))\b")
GEO_COUNTRY = rx(r"\b(china|chinese|beijing|xi jinping|taiwan\w*|russia\w*|moscow|putin|kremlin|ukrain\w*|kyiv|kiev|iran\w*|tehran|north korea\w*|"
                 r"south korea\w*|israel\w*|gaza|palestin\w*|hamas|hezbollah|saudi\w*|opec|pakistan|syria\w*|yemen|venezuela\w*|cuba\w*|afghan\w*|"
                 r"myanmar|belarus\w*|european union|eu|united nations|un|g7|g20|brexit)\b")
def eval_geo_twitter(t):
    s, c = hits(GEO_STRONG, t), hits(GEO_COUNTRY, t)
    score = len(s) + len(c)
    if len(s) >= 1 and (len(c) >= 1 or len(s) >= 2):
        return "accept", "geopolitical event terms + country/organisation marker", score, s + c
    if len(s) == 0 and len(c) == 0:
        return "reject", "Politics tweet without any geopolitical/international evidence (likely domestic politics)", 0, []
    return "review", "weak geopolitical evidence (single strong term without country, or country without event term)", score, s + c

# ---------- Merger / Acquisition (Twitter M&A | Investments, SENTiVENT M&A) ----------
MA_CORE = rx(r"\b(acqui(?:re|res|red|ring|sition|sitions)|merg(?:er|ers|es|ed|ing)|takeover|take over|buy-?out|buy out|tender offer|"
             r"business combination|take-private|go(?:es|ing)? private|takes? private|bid for|hostile bid|spin-?off|divest\w*)\b")
MA_SOFT = rx(r"\b(?:agree[sd]? to (?:buy|purchase|sell)|to (?:buy|purchase)|plans? to (?:buy|sell)|in talks to (?:buy|sell)|buys|bought|"
             r"(?:sale|sell|sells|sold|selling) (?:of )?(?:its |the |a |an )?[\w\s,&'-]{0,40}?(?:unit|business|division|subsidiary|arm|brand|operations)|"
             r"takes? control|controlling stake|majority stake|deal to (?:buy|sell)|deal for|(?:potential |proposed |pending )?purchase of|completes? (?:the )?(?:sale|purchase)|"
             r"talks to (?:buy|acquire)|closing the deal|deal (?:completion|terminated|closed))\b")
MA_NEG  = rx(r"\b(invests? in|invested in|investment in|funding round|series [a-f]\b|seed round|venture|private placement|raises? \$|raised \$|"
             r"\bipo\b|stock split|price target|buy rating|upgrade[sd]?|downgrade[sd]?|earnings|dividend|bullish|bearish)\b")
MA_PARTIAL = rx(r"\b(?:minority stake|acqui\w* (?:a|an|\d+(?:\.\d+)?%) (?:\w+ )?stake|stake in)\b")
MA_FULLCTRL = rx(r"\b(majority|controlling|remaining|full control|100%|all of)\b")
def eval_ma(t, ctx):
    core, soft, neg = hits(MA_CORE, t), hits(MA_SOFT, t), hits(MA_NEG, t)
    partial = bool(MA_PARTIAL.search(t)) and not MA_FULLCTRL.search(t)
    score = len(core) + len(soft) - len(neg)
    if partial:
        return "review", "stake purchase without clear control (M&A vs ordinary investment)", score, core + soft + neg
    if core and not neg:
        return "accept", "clear M&A language (" + ", ".join(core[:3]) + ")", score, core + soft
    if core and neg:
        return "accept" if len(core) >= 2 else "review", "M&A terms together with investment/finance terms", score, core + soft + neg
    if soft and not neg:
        return "accept", "deal/purchase/sale-of-business language", score, soft
    if neg and not (core or soft):
        return "reject", "investment/funding/analyst language without M&A evidence", score, neg
    if soft and neg:
        return "review", "conflicting M&A and investment signals", score, soft + neg
    return "review", "no M&A evidence in text (source label only)", 0, []

# ---------- Product Launch (Twitter Company|Product News, SENTiVENT Product/Service) ----------
PL_STRONG = rx(r"\b(launch(?:es|ed|ing)?|unveil(?:s|ed|ing)?|introduc(?:es|ed|ing)|releas(?:es|ed|ing)|rolls? out|rolled out|rolling out|rollout|"
               r"debut(?:s|ed|ing)?|now available|goes on sale|went on sale|begins? (?:selling|shipping)|starts? (?:selling|shipping)|"
               r"will (?:start|begin) selling|available (?:today|now|for (?:purchase|order|download|pre-?order))|pre-?orders? (?:open|begin|start))\b")
PL_WEAK   = rx(r"\b(announces? (?:the )?(?:new|availability)|new (?:product|service|app|platform|solution|software|device|model|version|lineup|offering)|"
               r"now (?:offers?|supports?)|offers? new|expands? (?:its )?(?:product|service)s?|available (?:in|on|at|through))\b")
PL_NEG    = rx(r"\b(partner(?:s|ed|ship|ships)?\b|teams? up|collaborat\w+|earnings|revenues?\b|sales (?:rose|fell|jump\w*|drop\w*|figures)|quarterly|"
               r"price (?:cut|hike|increase|rise)|raises? prices|recalls?|recalled|review(?:s|ed)?\b|lawsuits?|sues|layoffs?|lays off|stock split|dividend|"
               r"named\b|recogni[sz]ed|awards?\b|appoints?|"
               r"launch(?:es|ed|ing)? (?:an? )?(?:investigation|probe|inquiry|lawsuit|attack|offensive|strike|review|bid|offering|ipo|campaign)|"
               r"launch (?:customer|partner|date|pad|window)|satellites?|rockets?|"
               r"launch(?:es|ed|ing)? (?:of )?(?:its |the |a |an |new |first )*(?:[\w-]+ ){0,3}(?:study|trial|board|council|program|programme|initiative|fund|foundation|website|site|marketplace|manufacturing|"
               r"investigation|process|auction|tender|consultation|survey|challenge|competition|scholarship|webinar|podcast|series)|"
               r"releas(?:es|ed|ing) (?:its |the |a |an |new |latest |first |second |third |fourth |q[1-4] |annual |quarterly |full[- ]year )*(?:[\w&-]+ ){0,4}"
               r"(?:report|results|earnings|data|statement|guidance|findings|study|whitepaper|survey|financials?|presentation|schedule|dates?|update|letter|filing)|"
               r"introduc(?:es|ed|ing) (?:[\w-]+ ){0,4}(?:ceo|cfo|cto|chief|president|director|officer|chairman|head of|board)|"
               r"clinical (?:study|trial|hold)|public offering|offering of|(?:after|since|before|ahead of|following) (?:its |the |their )?launch|shutter\w*|discontinu\w+|postpone\w*)\b")
def pl_signal(t):
    return hits(PL_STRONG, t), hits(PL_WEAK, t), hits(PL_NEG, t)
def eval_pl_twitter(t):
    s, wk, ng = pl_signal(t)
    score = len(s) + len(wk) - len(ng)
    if s and not ng:  return "accept", "explicit launch/unveil/release language (" + ", ".join(s[:3]) + ")", score, s
    if s and ng:      return "review", "launch language together with non-launch signals (partnership/earnings/etc.)", score, s + ng
    if ng:            return "reject", "non-launch company news (" + ", ".join(ng[:3]) + ")", score, ng
    if wk:            return "review", "only weak product-availability language", score, wk
    return "reject", "general company/product news without launch language", 0, []
def eval_pl_senti(t, sub):
    """Source-aware SENTiVENT Product/Service rule."""
    toks = {x.strip() for x in str(sub).split(";")} if isinstance(sub, str) else set()
    s, wk, ng = pl_signal(t)
    score = len(s) + len(wk) - len(ng)

    if "Launch" in toks:
        hard_neg = [x for x in ng if any(k in x for k in [
            "recall", "earnings", "revenue", "investigation", "probe",
            "lawsuit", "ipo", "public offering"
        ])]
        if hard_neg:
            return "review", "SENTiVENT Launch annotation but strong non-launch signal is present", score, sorted(toks) + hard_neg
        return "accept", "SENTiVENT Product/Service subtype Launch (trusted annotation)", score + 2, sorted(toks) + s + wk

    if "Cancellation/Recall" in toks:
        planned_launch = bool(re.search(r"\b(launch|release|rollout|debut|availability|go(?:es|ing)? on sale)\b", t))
        cancelled = bool(re.search(r"\b(cancel(?:led|s|ing|lation)?|postpon(?:e|ed|es|ing)|delay(?:ed|s|ing)?)\b", t))
        if planned_launch and cancelled:
            return "accept", "SENTiVENT cancellation subtype describing cancelled/delayed launch", score + 1, sorted(toks) + s
        return "reject", "SENTiVENT Cancellation/Recall without clear cancelled/delayed launch", score, sorted(toks) + ng

    if "Trial" in toks:
        return "reject", "SENTiVENT Product/Service Trial is not a commercial product launch", score, sorted(toks)

    if s and not ng:
        return "review", "generic Product/Service annotation with explicit launch wording but no Launch subtype", score, s
    return "reject", "generic product/service mention without Launch subtype", score, ng

# ---------- Credit Event
# ---------- Credit Event (SENTiVENT Rating, SEC 8-K Items 1.03 / 2.04) ----------
CR_AGENCY   = rx(r"\b(moody'?s|fitch|s&p|standard & poor'?s|credit rating|credit grade|credit ratings|issuer default rating|\bidr\b|credit watch|credit outlook)\b")
CR_DISTRESS = rx(r"\b(downgrad\w*|cut|lowered|negative (?:outlook|watch)|outlook (?:is |remains )?negative|default\w*|bankrupt\w*|insolven\w*|missed (?:a |its )?(?:payment|coupon)|covenant|distress\w*|deteriorat\w*|junk|liquidity|non-?payment|payment default|debt service|refinancing risk|repayment risk)\b")
CR_ANALYST  = rx(r"\b(price target|price objective|target price|overweight|underweight|equal weight|outperform|underperform|buy rating|sell rating|neutral|initiate coverage|equity analysts?|brokerage)\b")
CR_BENIGN   = rx(r"\b(upgrad\w*|raised its? (?:credit )?rating|outlook (?:is |remains )?(?:positive|stable)|positive outlook|stable outlook)\b")

def eval_credit_senti(t, sub):
    ag, ds, an, bn = hits(CR_AGENCY, t), hits(CR_DISTRESS, t), hits(CR_ANALYST, t), hits(CR_BENIGN, t)
    toks = {x.strip() for x in str(sub).split(";")} if isinstance(sub, str) else set()
    score = len(ag) + len(ds) - len(an) - len(bn)

    if an and not ds:
        return "reject", "equity-analyst rating/price-target sentence, not a credit event", score, sorted(toks) + an

    if "Credit/Debt" in toks:
        if bn and not ds:
            return "reject", "Credit/Debt subtype but benign/improving credit action", score, sorted(toks) + bn
        return "accept", "SENTiVENT Rating subtype Credit/Debt (trusted annotation)", score + 2, sorted(toks) + ag + ds

    if "Downgrade" in toks and (ag or ds) and not an:
        return "accept", "SENTiVENT downgrade with credit/deterioration evidence", score + 1, sorted(toks) + ag + ds

    strong_agency = bool(re.search(
        r"moody|fitch|s&p|standard & poor|issuer default|credit rating|credit grade|credit watch|credit outlook", t
    ))
    if strong_agency and ds and not bn:
        return "accept", "credit-agency action with deterioration/distress evidence", score, ag + ds
    if strong_agency and bn and not ds:
        return "reject", "positive/benign credit-rating action", score, ag + bn

    if toks.intersection({"BuyOutperform","Hold","Upgrade","PriceTarget","Maintain","SellUnderperform"}):
        return "reject", "non-credit SENTiVENT Rating subtype", score, sorted(toks) + an

    if strong_agency or ds:
        return "review", "possible credit deterioration but evidence is incomplete", score, ag + ds
    return "reject", "no credit-event evidence", score, sorted(toks)

BANKR = rx(r"\b(chapter\s*(?:7|11|15)|bankruptcy|petitions?|receiver(?:ship)?|insolven\w*|assignment for the benefit of creditors|wind(?:ing)?[- ]up|liquidat\w+|ccaa|debtors?|creditors?|reorganization)\b")
SEC_204 = rx(r"\b(accelerat(?:e|ed|es|ing|ion)|event of default|default\w*|breach(?:ed|es|ing)?|covenant|immediate(?:ly)? due|due and payable|obligation|financial obligation|indebtedness|loan agreement|credit agreement|principal|interest payment|missed payment|non-?payment|repayment|maturity|waiver|forbearance|lenders?|creditors?)\b")
SEC_REFERENCE_ONLY = rx(r"\b(incorporated by reference|see exhibit|filed as exhibit|attached hereto|reference is made to)\b")

def eval_sec(t, source_label="", source_subtype=""):
    meta = f"{source_label} {source_subtype}".lower()

    if re.search(r"\[(?:company name|filing date)", t):
        return "review", "SEC template placeholder text", 0, []

    if "1.03" in meta or "bankruptcy or receivership" in meta:
        h = hits(BANKR, t)
        if h:
            return "accept", "SEC Item 1.03 with bankruptcy/insolvency evidence", len(h) + 2, h
        if SEC_REFERENCE_ONLY.search(t) and len(t.split()) < 80:
            return "review", "SEC Item 1.03 reference-only fragment", 0, hits(SEC_REFERENCE_ONLY, t)
        return "accept", "SEC Item 1.03 Bankruptcy/Receivership (trusted filing item code)", 2, []

    if "2.04" in meta or "accelerate" in meta or "financial obligation" in meta:
        h = hits(SEC_204, t)
        if len(h) >= 2:
            return "accept", "SEC Item 2.04 with obligation/default/acceleration evidence", len(h) + 1, h
        if len(h) == 1:
            return "review", "SEC Item 2.04 with only one weak obligation/trigger signal", 1, h
        return "review", "SEC Item 2.04 text lacks clear debt-trigger/acceleration evidence", 0, []

    h = hits(BANKR, t)
    if h:
        return "accept", "SEC filing with bankruptcy/insolvency evidence", len(h), h
    h2 = hits(SEC_204, t)
    if len(h2) >= 2:
        return "accept", "SEC filing with obligation/default evidence", len(h2), h2
    return "review", "SEC credit candidate without enough semantic evidence", 0, h + h2

# ---------- Macroeconomic
# ---------- Macroeconomic ----------
MACRO = rx(r"\b(inflation|deflation|disinflation|interest rates?|rate (?:hike|hikes|cut|cuts|decision)|central bank\w*|federal reserve|fed|fomc|ecb|boj|boe|"
           r"bank of (?:england|japan|canada|korea|israel)|gdp|economic growth|economy|economies|economic|recession|unemployment|jobless|employment|payrolls?|"
           r"monetary|fiscal|stimulus|treasury yields?|bond yields?|trade (?:deficit|balance|war|agreement)|nafta|currency|currencies|exchange rate|dollar|euro|yuan|"
           r"consumer (?:confidence|spending|prices?|sentiment)|cpi|ppi|pmi|housing (?:market|starts|prices)|minimum wage|budget deficit|debt ceiling|tax (?:cut|cuts|reform|plan)|"
           r"oil prices?|commodit\w+|quantitative (?:easing|tightening)|wages?|labor market|stagflation|growth forecast|retail sales|jobs report|trade data|"
           r"exports|imports|rba|rbi|pboc|snb|imf|world bank|tic flows|real pay|hike|hikes)\b")
MACRO_NONMACRO = rx(r"\b(market size|market research|cagr|featuring .{0,80}among others|industry (?:to|report|forecast|analysis|outlook)\s*(?:\d{4}|to \d{4})|forecast(?:s)? to 20\d\d|worldwide .{0,60} industry)\b")
def eval_macro_twitter(t):
    if MACRO_NONMACRO.search(t):
        return "review", "market-research/press-release style text under Macro label", 0, hits(MACRO_NONMACRO, t)
    h = hits(MACRO, t)
    return "accept", "Twitter Fed/Macro topic label" + (" + macro terms" if h else " (no macro lexicon hit; label trusted)"), len(h), h
def eval_macro_senti(t):
    h = hits(MACRO, t)
    if h: return "accept", "SENTiVENT Macroeconomics + macro terms", len(h), h
    return "review", "SENTiVENT 'Macroeconomics' sentence without macro terms (often industry/market commentary)", 0, []

# ---------- Strong-source sanity ----------
ACQ_DOC = rx(r"\b(acqui\w+|merger|merge[sd]?|takeover|buy(?:s|ing)?|purchas\w+|business combination|tender offer)\b")
def eval_corp_doc(t):
    h = hits(ACQ_DOC, t[:1500])
    if h: return "accept", "press-release/document with acquisition language in the opening", len(h), h
    return "review", "corporate_event_detection document without acquisition wording in its opening", 0, []

# ---------- Catalyst manually labeled financial news ----------
CAT_MA_NEG = rx(r"\b(partnership|partners? with|funding round|venture funding|minority investment|supplier agreement|distribution agreement)\b")

def eval_catalyst(t, lab, source_label):
    if lab == "Merger / Acquisition":
        core = hits(MA_CORE, t) + hits(MA_SOFT, t)
        neg = hits(CAT_MA_NEG, t)
        if core:
            return "accept", "Catalyst manually labeled MERGER_ANNOUNCEMENT + transaction evidence", len(core) - len(neg) + 2, core + neg
        if neg:
            return "review", "Catalyst M&A label but text looks like investment/partnership rather than control transaction", -len(neg), neg
        return "accept", "Catalyst manually labeled MERGER_ANNOUNCEMENT (trusted labeled source)", 2, []

    if lab == "Product Launch":
        s, wk, ng = pl_signal(t)
        hard_neg = [x for x in ng if any(k in x for k in [
            "recall", "earnings", "investigation", "probe", "lawsuit", "ipo", "public offering"
        ])]
        if hard_neg and not s:
            return "review", "Catalyst Product Launch label but text has a strong non-launch signal", -len(hard_neg), hard_neg
        return "accept", "Catalyst manually labeled PRODUCT_LAUNCHES (trusted labeled source)", len(s) + len(wk) + 2, s + wk

    return "review", f"Catalyst row has unexpected target label {lab}", 0, []

def evaluate(r):
    t, src, lab, sub, q = r.normalized_text, r.source_dataset, r.label, r.source_subtype, r.mapping_quality
    if src == "gdelt_news_events":
        return "accept", "GDELT event-pair sentence (strong source mapping; templated text - see flags)", 1, []
    if src == "sec_8k_events":                      return eval_sec(t, r.source_label, r.source_subtype)
    if src == "corporate_event_detection":          return eval_corp_doc(t)
    if src == "catalyst_financial_news":            return eval_catalyst(t, lab, r.source_label)
    if src == "twitter_financial_news_topic":
        if lab == "Geopolitical":                   return eval_geo_twitter(t)
        if lab == "Merger / Acquisition":           return eval_ma(t, "twitter")
        if lab == "Product Launch":                 return eval_pl_twitter(t)
        if lab == "Macroeconomic":                  return eval_macro_twitter(t)
    if src == "sentivent":
        if lab == "Credit Event":                   return eval_credit_senti(t, sub)
        if lab == "Product Launch":                 return eval_pl_senti(t, sub)
        if lab == "Merger / Acquisition":
            d, why, sc, m = eval_ma(t, "sentivent")
            return d, "SENTiVENT M&A: " + why, sc, m
        if lab == "Macroeconomic":                  return eval_macro_senti(t)
    return "review", f"no validation rule for source={src}, label={lab}", 0, []
print("rules defined")


# ===== NOTEBOOK CELL 11 =====
# ============================== APPLY RULES ==============================
res = [evaluate(r) for r in cand.itertuples(index=False)]
cand["filter_decision"] = [r[0] for r in res]
cand["filter_reason"]   = [r[1] for r in res]
cand["filter_score"]    = [r[2] for r in res]
cand["matched_terms"]   = ["; ".join(r[3]) for r in res]

unruled = cand[cand.filter_reason.str.startswith("no validation rule")]
if len(unruled): flag(f"{len(unruled)} rows come from a source/label combination with no validation rule -> sent to manual review: {unruled.groupby(['source_dataset','label']).size().to_dict()}")

print("=== decision by source x label ===")
dec = cand.groupby(["source_dataset", "label", "mapping_quality", "filter_decision"]).size().unstack(fill_value=0)
for c in ["accept", "reject", "review"]:
    if c not in dec: dec[c] = 0
dec["total"] = dec[["accept", "reject", "review"]].sum(axis=1)
dec["accept_%"] = (100 * dec.accept / dec.total).round(1)
print(dec[["accept", "reject", "review", "total", "accept_%"]])

# ---- warn when a filter removes an unexpectedly large share ----
for (src, lab), grp in cand.groupby(["source_dataset", "label"]):
    lost = (grp.filter_decision != "accept").mean()
    if lost > 0.50:
        flag(f"Validation removes {lost:.0%} of {src} / {lab} ({(grp.filter_decision!='accept').sum()} of {len(grp)}). Check filter_reason before trusting this class.")


print("\n=== targeted source-aware retention audit ===")
for src_name in ["catalyst_financial_news", "sec_8k_events", "sentivent", "twitter_financial_news_topic"]:
    sub = cand[cand.source_dataset == src_name]
    if len(sub):
        print(f"\n[{src_name}]")
        print(pd.crosstab([sub.label, sub.source_label], sub.filter_decision, margins=True))

print("\n=== top reasons ===")
print(cand.groupby(["label", "filter_decision", "filter_reason"]).size().sort_values(ascending=False).head(30))

print("\n=== random ACCEPTED / REJECTED / REVIEW examples for broad sources ===")
for lab in ["Geopolitical", "Merger / Acquisition", "Product Launch", "Credit Event", "Macroeconomic"]:
    sub = cand[(cand.label == lab) & (cand.mapping_quality == "broad_candidate")]
    for d in ["accept", "reject", "review"]:
        s = sub[sub.filter_decision == d]
        print(f"\n--- {lab} | {d} ({len(s)}) ---")
        for _, r in s.sample(min(4, len(s)), random_state=SEED).iterrows():
            print("  •", r.text[:170].replace("\n", " "), "||", r.filter_reason[:70])


# ===== NOTEBOOK CELL 13 =====
# ============================== STEP 12 — MANUAL REVIEW ==============================
REVIEW_COLS = ["text", "current_label", "label_id", "source_dataset", "source_label", "source_subtype", "mapping_quality",
               "review_reason", "filter_score", "matched_terms", "source_id", "source_split", "raw_row_id",
               "duplicate_count", "near_dup_cluster_id", "reviewer_label"]
rv = cand[cand.filter_decision == "review"].copy()
rv["current_label"] = rv.label; rv["review_reason"] = rv.filter_reason
nd = nd_conf.copy()
nd["current_label"] = nd.label
nd["review_reason"] = "near-duplicate cluster contains different labels (cluster " + nd.near_dup_cluster_id.astype(str) + ")"
nd["filter_score"] = np.nan; nd["matched_terms"] = ""
manual = pd.concat([rv, nd], ignore_index=True)
manual["reviewer_label"] = ""        # fill in by hand if you recover rows
for c in REVIEW_COLS:
    if c not in manual: manual[c] = np.nan
manual = manual[REVIEW_COLS].sort_values(["current_label", "review_reason"])
manual.to_csv(out("manual_review_candidates.csv"), index=False)
print("manual_review_candidates.csv rows:", len(manual))
print(manual.groupby(["current_label", "source_dataset"]).size())

rej = cand[cand.filter_decision == "reject"].copy()
rej[["raw_row_id", "text", "label", "label_id", "source_dataset", "source_label", "source_subtype", "mapping_quality",
     "filter_reason", "filter_score", "matched_terms"]].to_csv(out("rejected_candidates.csv"), index=False)
print("rejected_candidates.csv (audit) rows:", len(rej))

# ============================== STEP 13 — CLEAN DATASET ==============================
CLEAN_COLS = ["text", "label", "label_id", "source_dataset", "source_label", "source_subtype", "source_split", "source_id",
              "mapping_quality", "duplicate_count", "all_source_datasets", "all_source_labels", "all_mapping_qualities",
              "near_dup_cluster_id", "near_dup_count", "filter_reason", "filter_score", "raw_row_id"]
clean = cand[cand.filter_decision == "accept"].copy()
clean["near_dup_count"] = clean.near_dup_count.fillna(1).astype(int)
clean = clean[CLEAN_COLS].reset_index(drop=True)
clean["label_id"] = clean.label_id.astype(int)
clean.to_csv(out("event_dataset_clean.csv"), index=False)

# accounting identities
n_after_exact = len(reps)
assert n_after_exact == n_removed_nd + len(nd_conf) + (cand.filter_decision == "reject").sum() + (cand.filter_decision == "review").sum() + len(clean), "final accounting mismatch"
assert STATS["original_rows"] == STATS["technical_failures_removed"] + STATS["exact_duplicates_removed"] + STATS["exact_conflict_rows"] + n_after_exact

flow = pd.DataFrame([
    ("raw rows", STATS["original_rows"]),
    ("- technical failures", -STATS["technical_failures_removed"]),
    ("- exact same-label duplicates collapsed", -STATS["exact_duplicates_removed"]),
    ("- exact cross-label conflict rows", -STATS["exact_conflict_rows"]),
    ("- near-duplicates collapsed", -int(n_removed_nd)),
    ("- near-duplicate cross-label rows", -int(len(nd_conf))),
    ("- rejected by validation", -int((cand.filter_decision == "reject").sum())),
    ("- sent to manual review", -int((cand.filter_decision == "review").sum())),
    ("= CLEAN rows", len(clean))], columns=["step", "rows"])
print("\n=== row flow ==="); print(flow.to_string(index=False))

print("\n=== CLEAN class counts (natural, NOT balanced) ===")
cc = clean.label.value_counts().reindex([LABELS[i] for i in range(5)]).fillna(0).astype(int)
print(pd.DataFrame({"label_id": range(5), "clean": cc.values, "raw": [STATS["raw_class_counts"].get(l, 0) for l in cc.index],
                    "retained_%": [round(100 * c / max(1, STATS["raw_class_counts"].get(l, 0)), 1) for l, c in cc.items()]}, index=cc.index))
print("\nclean by source:"); print(pd.crosstab(clean.label, clean.source_dataset, margins=True))
print("\nclean by mapping_quality:"); print(pd.crosstab(clean.label, clean.mapping_quality, margins=True))

# broad candidate fate (after dedup)
broad = cand[cand.mapping_quality == "broad_candidate"]
STATS["broad_candidates_after_dedup"] = int(len(broad))
STATS["broad_accepted"] = int((broad.filter_decision == "accept").sum())
STATS["broad_rejected"] = int((broad.filter_decision == "reject").sum())
STATS["broad_manual_review"] = int((broad.filter_decision == "review").sum())
print("\nbroad candidates -> accepted/rejected/review:", STATS["broad_accepted"], STATS["broad_rejected"], STATS["broad_manual_review"])
STATS["broad_fate_by_source_label"] = {f"{a} | {b}": c.to_dict() for (a, b), c in broad.groupby(["source_dataset", "label"]).filter_decision.value_counts().unstack(fill_value=0).iterrows()}

# ============================== STEP 14 — CLASS-SIZE PROBLEMS ==============================
WEAK = {}
for i in range(5):
    lab = LABELS[i]
    k = int(cc.get(lab, 0))
    if k < MIN_REAL_PER_CLASS:
        WEAK[lab] = k
        print("\n" + "!" * 78)
        print(f"WARNING: '{lab}' has only {k} clean REAL examples (< {MIN_REAL_PER_CLASS}).")
        print("THIS CLASS IS STILL SMALL. BASELINE TRAINING MAY PROCEED, BUT FINAL METRICS SHOULD BE INTERPRETED CAUTIOUSLY.")
        print("!" * 78)
        flag(f"Weak class '{lab}': {k} clean real examples (< {MIN_REAL_PER_CLASS}).")
if not WEAK: print("\nAll classes have >= %d clean real examples." % MIN_REAL_PER_CLASS)

# ---- source/style confounds in the clean data ----
print("\n--- source confound check (clean) ---")
for i in range(5):
    lab = LABELS[i]; s = clean[clean.label == lab].source_dataset.value_counts(normalize=True)
    print(f"{lab:22s} dominant source: {s.index[0]} ({s.iloc[0]:.0%})")
    if s.iloc[0] > 0.80:
        flag(f"'{lab}' is {s.iloc[0]:.0%} from a single source ({s.index[0]}). The classifier may learn source style (length, templates, vocabulary) instead of event semantics; evaluate per source.")
clean["_words"] = clean.text.str.split().str.len()
print("\nmedian words per class/source:"); print(clean.groupby(["label", "source_dataset"])._words.median().unstack())
long_share = (clean._words * 1.3 > 512).groupby(clean.source_dataset).mean()
for src, v in long_share.items():
    if v > 0.20: flag(f"{src}: {v:.0%} of clean texts are likely longer than 512 tokens -> decide truncation (head) or chunking before transformer training.")
gd = clean[clean.source_dataset == "gdelt_news_events"]
if len(gd):
    tmpl = gd.text.str.replace(r"\(.*?\)", "", regex=True).str.split().str.len().describe()
    flag(f"GDELT contributes {len(gd)} rows ({len(gd)/ (clean.label=='Geopolitical').sum():.0%} of clean Geopolitical). Texts are machine-generated templates ('X criticized Y.'), not news prose; mapping is 'strong' per source metadata and was kept, but this is a domain/style mismatch with the other classes.")
clean = clean.drop(columns=["_words"])
STATS["weak_classes"] = WEAK
STATS["clean_rows"] = int(len(clean))
STATS["clean_class_counts"] = {k: int(v) for k, v in cc.items()}
STATS["clean_by_source"] = {k: int(v) for k, v in clean.source_dataset.value_counts().items()}
STATS["clean_by_mapping_quality"] = {k: int(v) for k, v in clean.mapping_quality.value_counts().items()}
STATS["clean_class_by_source"] = {l: {k: int(v) for k, v in clean[clean.label == l].source_dataset.value_counts().items()} for l in cc.index}


# ===== NOTEBOOK CELL 15 =====
# ============================== STEP 15 — SPLIT ==============================
assert clean.near_dup_cluster_id.is_unique, "a near-duplicate cluster has >1 representative - cannot split safely"
min_cls = clean.label_id.value_counts().min()
if min_cls < 10: flag(f"Smallest class has only {min_cls} rows; validation/test will contain ~{max(1, round(min_cls*0.1))} examples of it - metrics for that class will be very noisy.")
idx = np.arange(len(clean))
tr_idx, tmp_idx = train_test_split(idx, test_size=0.20, stratify=clean.label_id, random_state=SEED)
va_idx, te_idx  = train_test_split(tmp_idx, test_size=0.50, stratify=clean.label_id.values[tmp_idx], random_state=SEED)
train = clean.iloc[tr_idx].sort_values("raw_row_id").reset_index(drop=True)
val   = clean.iloc[va_idx].sort_values("raw_row_id").reset_index(drop=True)
test  = clean.iloc[te_idx].sort_values("raw_row_id").reset_index(drop=True)
print("split sizes:", len(train), len(val), len(test), "| fractions:", [round(x / len(clean), 3) for x in (len(train), len(val), len(test))])

# ============================== STEP 16 — OUTPUT FILES ==============================
for name, d in [("event_train", train), ("event_validation", val), ("event_test", test)]:
    d.to_csv(out(name + ".csv"), index=False)
clean.to_csv(out("event_dataset_clean.csv"), index=False)
try:
    for name, d in [("event_dataset_clean", clean), ("event_train", train), ("event_validation", val), ("event_test", test)]:
        d.to_parquet(out(name + ".parquet"), index=False)
    print("parquet files written")
except Exception as e:
    print("parquet skipped:", e)

# ============================== STEP 17 — CLASS WEIGHTS (train only) ==============================
classes = np.arange(5)
assert set(train.label_id.unique()) == set(classes), "a class is missing from the training split"
cw = compute_class_weight(class_weight="balanced", classes=classes, y=train.label_id.values)
class_weights = {str(int(c)): float(w_) for c, w_ in zip(classes, cw)}
json.dump(class_weights, open(out("class_weights.json"), "w"), indent=2)
print("class weights (train only, sklearn 'balanced'):", class_weights)

def dist(d):
    return {LABELS[i]: int((d.label_id == i).sum()) for i in range(5)}
STATS.update(train_class_counts=dist(train), validation_class_counts=dist(val), test_class_counts=dist(test), class_weights=class_weights)
print(pd.DataFrame({"train": dist(train), "validation": dist(val), "test": dist(test)}))
for lab, k in dist(val).items():
    if k < 20: flag(f"Validation split has only {k} '{lab}' examples; too few for reliable per-class metrics.")
    if dist(test)[lab] < 20: flag(f"Test split has only {dist(test)[lab]} '{lab}' examples; too few for reliable per-class metrics.")


# ===== NOTEBOOK CELL 17 =====
# ============================== STEP 18 — REPORTS ==============================
STATS["manual_review_rows_total"] = int(len(manual))
STATS["rejected_by_validation_total"] = int((cand.filter_decision == "reject").sum())
STATS["sent_to_manual_review_by_validation"] = int((cand.filter_decision == "review").sum())
STATS["row_flow"] = {r.step: int(r.rows) for r in flow.itertuples()}
STATS["filter_decision_by_source_label"] = {f"{a} | {b}": c.to_dict() for (a, b), c in cand.groupby(["source_dataset", "label"]).filter_decision.value_counts().unstack(fill_value=0).iterrows()}
STATS["parameters"] = dict(seed=SEED, near_dup_threshold=NEAR_DUP_THRESHOLD, near_dup_max_chars=NEAR_DUP_MAX_CHARS,
                           split="80/10/10 stratified by label_id", min_real_per_class=MIN_REAL_PER_CLASS,
                           semantic_filter="source-aware metadata + regex rules; trusted Catalyst labeled rows, SEC item-specific rules, relaxed SENTiVENT subtype trust (no ML/NLI model, no API)")
STATS["flags"] = FLAGS
json.dump(STATS, open(out("preprocessing_report.json"), "w"), indent=2, default=str)

rows = []
def add(section, key, value): rows.append({"section": section, "key": key, "value": value})
for k in ["original_rows", "technical_failures_removed", "exact_duplicate_groups", "exact_duplicates_removed", "exact_conflict_groups",
          "exact_conflict_rows", "near_duplicates_removed", "near_duplicate_conflict_clusters", "near_duplicate_conflict_rows",
          "broad_candidates_after_dedup", "broad_accepted", "broad_rejected", "broad_manual_review", "rejected_by_validation_total",
          "sent_to_manual_review_by_validation", "manual_review_rows_total", "clean_rows"]:
    add("overview", k, STATS[k])
for sec in ["raw_class_counts", "clean_class_counts", "train_class_counts", "validation_class_counts", "test_class_counts",
            "clean_by_source", "clean_by_mapping_quality", "class_weights"]:
    for k, v in STATS[sec].items(): add(sec, k, v)
for lab, k in WEAK.items(): add("weak_class_warning", lab, k)
for i, f in enumerate(FLAGS): add("flag", i + 1, f)
pd.DataFrame(rows).to_csv(out("preprocessing_summary.csv"), index=False)
print("report written. flags collected:", len(FLAGS))


# ===== NOTEBOOK CELL 19 =====
# ============================== STEP 19 — PLOTS ==============================
order = [LABELS[i] for i in range(5)]
def bar(ax, d, title):
    vals = [d.get(l, 0) for l in order]
    ax.bar(range(5), vals); ax.set_xticks(range(5)); ax.set_xticklabels([l.replace(" / ", "/\n") for l in order], fontsize=8)
    ax.set_title(title)
    for i, v in enumerate(vals): ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
bar(ax[0], STATS["raw_class_counts"], "Original (raw) class distribution")
bar(ax[1], STATS["clean_class_counts"], "Cleaned class distribution")
plt.tight_layout(); plt.savefig(out("plot_class_distribution.png"), dpi=110); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
raw.source_dataset.value_counts().plot.bar(ax=ax[0], title="Raw rows by source"); ax[0].tick_params(axis="x", labelsize=7)
clean.source_dataset.value_counts().plot.bar(ax=ax[1], title="Clean rows by source"); ax[1].tick_params(axis="x", labelsize=7)
plt.tight_layout(); plt.savefig(out("plot_source_distribution.png"), dpi=110); plt.show()

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, (nm, d) in zip(ax, [("Train", train), ("Validation", val), ("Test", test)]):
    bar(a, dist(d), f"{nm} (n={len(d)})")
plt.tight_layout(); plt.savefig(out("plot_split_distribution.png"), dpi=110); plt.show()


# ===== NOTEBOOK CELL 21 =====
# ============================== STEP 20 — SANITY CHECKS ==============================
files = {n: pd.read_csv(out(n + ".csv"), keep_default_na=False, na_values=[""]) for n in ["event_dataset_clean", "event_train", "event_validation", "event_test"]}
allowed = set(LABELS.values())
for n, d in files.items():
    assert set(d.label.unique()) <= allowed, f"{n}: invalid label"
    assert d.label.map(LABEL2ID).eq(d.label_id).all(), f"{n}: label and label_id disagree"
    assert d.text.notna().all(), f"{n}: null text"
    assert d.text.astype(str).str.strip().ne("").all(), f"{n}: empty text"
    assert set(d.label_id.unique()) == set(range(5)), f"{n}: not all 5 classes present"
assert {l: i for i, l in LABELS.items()} == {"Geopolitical": 0, "Macroeconomic": 1, "Credit Event": 2, "Merger / Acquisition": 3, "Product Launch": 4}

cl, tr, va, te = (files[k] for k in ["event_dataset_clean", "event_train", "event_validation", "event_test"])
assert len(tr) + len(va) + len(te) == len(cl)
for d in (cl, tr, va, te): d["_n"] = d.text.map(normalize_text)
s_tr, s_va, s_te = set(tr._n), set(va._n), set(te._n)
assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "exact duplicate text across splits"
assert cl.groupby("_n").label_id.nunique().max() == 1 and cl._n.is_unique, "duplicate/conflicting text remains in clean data"
cid = {k: set(d.near_dup_cluster_id) for k, d in [("train", tr), ("val", va), ("test", te)]}
assert not (cid["train"] & cid["val"]) and not (cid["train"] & cid["test"]) and not (cid["val"] & cid["test"]), "near-dup cluster crosses splits"
assert (set(cl.raw_row_id) & set(exact_conf.raw_row_id)) == set(), "exact-conflict row leaked into clean data"
assert (set(cl.raw_row_id) & set(manual.raw_row_id)) == set(), "manual-review row leaked into clean data"
print("ALL HARD ASSERTIONS PASSED")

# ---- residual cross-split similarity (informational): near-misses just below/at threshold ----
def cross_sim(a, b, thr):
    A_ = vec.transform(a._n.str.replace("<url>", " ", regex=False).str.replace("<user>", " ", regex=False).str.slice(0, NEAR_DUP_MAX_CHARS))
    B_ = vec.transform(b._n.str.replace("<url>", " ", regex=False).str.replace("<user>", " ", regex=False).str.slice(0, NEAR_DUP_MAX_CHARS)).T.tocsc()
    cnt = 0
    for s in range(0, A_.shape[0], 500):
        M = (A_[s:s + 500] @ B_).tocoo()
        cnt += len(set(M.row[M.data >= thr]))        # distinct query rows with at least one neighbour >= thr
    return cnt
for thr in (0.92, 0.85):
    print(f"validation rows with a train neighbour >= {thr}: {cross_sim(va, tr, thr)}   | test rows: {cross_sim(te, tr, thr)}")
print("(rows >= 0.92 can only exist where the number-mismatch guard kept two near-identical texts apart; 0.85-0.92 are paraphrase-like neighbours.)")

# ---- random inspection ----
print("\n" + "=" * 100 + "\nRANDOM CLEAN EXAMPLES PER CLASS (inspect manually)\n" + "=" * 100)
for i in range(5):
    sub = cl[cl.label_id == i]
    print(f"\n######## {LABELS[i]}  (n={len(sub)}) ########")
    for _, r in sub.sample(min(12, len(sub)), random_state=SEED).iterrows():
        print(f"  [{r.source_dataset[:12]:12s}|{str(r.source_subtype)[:14]:14s}] {r.text[:200].replace(chr(10), ' ')}")

print("\n" + "=" * 100 + f"\nFLAGS COLLECTED ({len(FLAGS)})\n" + "=" * 100)
for f in FLAGS: print("-", f)
print("\nFiles in", WORK_DIR, ":", sorted(os.listdir(WORK_DIR)))
