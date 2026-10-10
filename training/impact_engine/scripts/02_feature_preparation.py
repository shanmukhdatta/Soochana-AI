#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Faithful Python export of original notebook: impact-engine-training-data-prepration.ipynb.

Kaggle-friendly original behavior; Python export adds environment variable overrides
for input/output locations and removes notebook-only pip directives. It does not
fabricate pretrained weights, datasets, or experiment results.
"""
import os
from IPython.display import display


# %% Code cell 2
# Notebook-only %pip installation omitted; see ../requirements.txt

# %% Code cell 3
import os, re, gc, json, math, random, zipfile, hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

INPUT_ROOT=Path(os.environ.get('RISKPULSE_IMPACT_INPUT_ROOT', str(Path('/kaggle/input') if Path('/kaggle/input').exists() else Path.cwd()))).expanduser().resolve()
WORKDIR=Path(os.environ.get('RISKPULSE_IMPACT_WORKDIR', str(Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()))).expanduser().resolve()
WORKDIR.mkdir(parents=True,exist_ok=True)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('INPUT_ROOT:',INPUT_ROOT)
print('WORKDIR:',WORKDIR)
print('DEVICE:',DEVICE)
if torch.cuda.is_available(): print('GPU:',torch.cuda.get_device_name(0))

# %% Code cell 5
EVENT_MODEL_PATH_OVERRIDE=os.getenv('RISKPULSE_EVENT_MODEL_DIR') or None
SENTIMENT_MODEL_PATH_OVERRIDE=os.getenv('RISKPULSE_SENTIMENT_MODEL_DIR') or None
IMPACT_DATA_DIR_OVERRIDE=os.getenv('RISKPULSE_IMPACT_DATA_DIR') or None
PRICE_ARCHIVE_PATH_OVERRIDE=os.getenv('RISKPULSE_PRICE_ARCHIVE') or None

NLP_MAX_LENGTH=128
EVENT_BATCH_SIZE=64
SENTIMENT_BATCH_SIZE=64

PRICE_REPO='Zihan1004/FNSPID'
PRICE_ARCHIVE_FILE='Stock_price/full_history.zip'
BENCHMARK='SPY'

EXPECTED_EVENT_LABELS={
    0:'Geopolitical',
    1:'Macroeconomic',
    2:'Credit Event',
    3:'Merger / Acquisition',
    4:'Product Launch',
}

TICKER_SECTOR={
'AAPL':'Technology','MSFT':'Technology','GOOGL':'Communication Services','META':'Communication Services',
'AMZN':'Consumer Discretionary','TSLA':'Consumer Discretionary','NVDA':'Technology','AMD':'Technology',
'INTC':'Technology','QCOM':'Technology','CSCO':'Technology','ADBE':'Technology','NFLX':'Communication Services',
'PEP':'Consumer Staples','COST':'Consumer Staples','SBUX':'Consumer Discretionary','BKNG':'Consumer Discretionary',
'AMGN':'Health Care','GILD':'Health Care','REGN':'Health Care','HON':'Industrials','CSX':'Industrials',
'ADP':'Industrials','MDLZ':'Consumer Staples','CMCSA':'Communication Services'}

FORBIDDEN_FUTURE_FEATURES={
    'stock_return_1d','benchmark_return_1d','abnormal_return_1d','car_3d',
    'realized_impact','impact_score_1_10','session_news_count'
}
TARGET_COLUMN='realized_impact'
print('Event labels:',EXPECTED_EVENT_LABELS)

# %% Code cell 7
HF_TOKEN=None
try:
    from kaggle_secrets import UserSecretsClient
    try: HF_TOKEN=UserSecretsClient().get_secret('HF_TOKEN')
    except Exception: HF_TOKEN=None
except Exception:
    HF_TOKEN=os.environ.get('HF_TOKEN')
print('HF token available:',bool(HF_TOKEN))

# %% Code cell 9
MODEL_EXTRACT_ROOT=WORKDIR/'impact_step2_extracted_models'; MODEL_EXTRACT_ROOT.mkdir(parents=True,exist_ok=True)
ZIP_HINTS=('event','classifier','5class','5-class','sentiment','regress','finbert','model')
for z in INPUT_ROOT.rglob('*.zip'):
    low=str(z).lower()
    if 'full_history' in low or 'stock_price' in low: continue
    if not any(h in z.name.lower() for h in ZIP_HINTS): continue
    dest=MODEL_EXTRACT_ROOT/f'{z.stem}_{hashlib.sha1(str(z).encode()).hexdigest()[:10]}'
    if not dest.exists():
        try:
            dest.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(z,'r') as zf: zf.extractall(dest)
            print('Extracted:',z,'->',dest)
        except Exception as e:
            print('WARNING could not extract',z,':',e)

def has_weights(p):
    return any((p/f).exists() for f in ['model.safetensors','pytorch_model.bin']) or any(p.glob('model-*.safetensors'))

def collect_model_dirs():
    out=[]; seen=set()
    for root in [INPUT_ROOT,MODEL_EXTRACT_ROOT]:
        if not root.exists(): continue
        for cfg in root.rglob('config.json'):
            p=cfg.parent
            if has_weights(p):
                key=str(p.resolve())
                if key not in seen: seen.add(key); out.append(p)
    return out
MODEL_DIR_CANDIDATES=collect_model_dirs()
print('Model directories found:',len(MODEL_DIR_CANDIDATES))
for p in MODEL_DIR_CANDIDATES[:30]: print(' ',p)

# %% Code cell 10
def cfg_json(p):
    try: return json.loads((p/'config.json').read_text(encoding='utf-8'))
    except Exception: return {}

def id2label_norm(cfg):
    out={}
    for k,v in (cfg.get('id2label') or {}).items():
        try: out[int(k)]=re.sub(r'[^a-z]+','',str(v).lower())
        except Exception: pass
    return out
EXPECTED_NORM={i:re.sub(r'[^a-z]+','',v.lower()) for i,v in EXPECTED_EVENT_LABELS.items()}

def event_score(p):
    cfg=cfg_json(p); name=str(p).lower(); s=0
    if int(cfg.get('num_labels',-1) or -1)==5: s+=50
    if id2label_norm(cfg)==EXPECTED_NORM: s+=100
    for hint,pts in [('event',25),('classifier',20),('5class',20),('5-class',20),('finbert',5),('checkpoint',-5),('sentiment',-50),('regress',-50)]:
        if hint in name: s+=pts
    return s

def sentiment_score_path(p):
    cfg=cfg_json(p); name=str(p).lower(); s=0
    if int(cfg.get('num_labels',-1) or -1)==1: s+=60
    if str(cfg.get('problem_type','')).lower()=='regression': s+=25
    if 'sentiment_head' in cfg: s+=30
    if (p/'modeling_finbert_sentiment_regressor.py').exists(): s+=80
    for hint,pts in [('sentiment',35),('regress',30),('finbert',5),('checkpoint',-5),('event',-40),('classifier',-20)]:
        if hint in name: s+=pts
    return s

def resolve_model(override,score_fn,role):
    if override:
        p=Path(override)
        if not p.exists(): raise FileNotFoundError(p)
        return p
    ranked=sorted([(score_fn(p),p) for p in MODEL_DIR_CANDIDATES],key=lambda x:(x[0],str(x[1])),reverse=True)
    print(); print(role,'candidates:')
    for s,p in ranked[:10]: print(f'  score={s:>4}  {p}')
    if not ranked or ranked[0][0]<=0: raise RuntimeError(f'Could not auto-discover {role}; set path override.')
    return ranked[0][1]

EVENT_MODEL_PATH=resolve_model(EVENT_MODEL_PATH_OVERRIDE,event_score,'Event classifier')
SENTIMENT_MODEL_PATH=resolve_model(SENTIMENT_MODEL_PATH_OVERRIDE,sentiment_score_path,'Sentiment regressor')
if EVENT_MODEL_PATH.resolve()==SENTIMENT_MODEL_PATH.resolve(): raise RuntimeError('Same folder selected for both models; set overrides.')
print(); print('Selected event model:',EVENT_MODEL_PATH); print('Selected sentiment model:',SENTIMENT_MODEL_PATH)

# %% Code cell 12
REQ=['impact_train.parquet','impact_validation.parquet','impact_test.parquet','impact_decile_thresholds.json']
def exact_matches(name): return sorted(INPUT_ROOT.rglob(name),key=lambda p:len(str(p)))
if IMPACT_DATA_DIR_OVERRIDE:
    IMPACT_DATA_DIR=Path(IMPACT_DATA_DIR_OVERRIDE)
else:
    parents=Counter()
    for name in REQ:
        for p in exact_matches(name): parents[p.parent]+=1
    if not parents: raise FileNotFoundError('No frozen Impact Step-1 files found under /kaggle/input.')
    IMPACT_DATA_DIR=parents.most_common(1)[0][0]

def resolve_impact(name):
    p=IMPACT_DATA_DIR/name
    if p.exists(): return p
    m=exact_matches(name)
    if not m: raise FileNotFoundError(name)
    return m[0]
IMPACT_PATHS={'train':resolve_impact('impact_train.parquet'),'validation':resolve_impact('impact_validation.parquet'),'test':resolve_impact('impact_test.parquet'),'deciles':resolve_impact('impact_decile_thresholds.json')}
for k,p in IMPACT_PATHS.items(): print(k,':',p)

# %% Code cell 14
train_df = pd.read_parquet(IMPACT_PATHS['train']).copy(); train_df['split'] = 'train'
validation_df = pd.read_parquet(IMPACT_PATHS['validation']).copy(); validation_df['split'] = 'validation'
test_df = pd.read_parquet(IMPACT_PATHS['test']).copy(); test_df['split'] = 'test'

with open(IMPACT_PATHS['deciles'], 'r', encoding='utf-8') as f:
    impact_deciles = json.load(f)

# publication_time_utc + timestamp_precision are required because Step 2
# must construct market predictors using only information available
# strictly before the news publication calendar date.
REQUIRED = {
    'headline',
    'ticker',
    'publication_time_utc',
    'timestamp_precision',
    'reaction_date',
    'sector',
    'realized_impact',
    'impact_score_1_10',
    'split',
}

def validate(df, name):
    miss = REQUIRED - set(df.columns)
    if miss:
        raise ValueError(f'{name} missing {sorted(miss)}')

    if df['headline'].isna().any() or df['headline'].astype(str).str.strip().eq('').any():
        raise ValueError(f'{name}: empty headline')

    if df['ticker'].isna().any() or df['ticker'].astype(str).str.strip().eq('').any():
        raise ValueError(f'{name}: empty ticker')

    df['reaction_date'] = (
        pd.to_datetime(df['reaction_date'], errors='coerce')
        .dt.tz_localize(None)
        .dt.normalize()
    )
    if df['reaction_date'].isna().any():
        raise ValueError(f'{name}: invalid reaction date')

    df['publication_time_utc'] = pd.to_datetime(
        df['publication_time_utc'],
        errors='coerce',
        utc=True,
    )
    if df['publication_time_utc'].isna().any():
        raise ValueError(f'{name}: invalid publication_time_utc')

    allowed_precision = {'date_only', 'exact_time'}
    bad_precision = set(df['timestamp_precision'].dropna().astype(str).unique()) - allowed_precision
    if bad_precision:
        raise ValueError(f'{name}: unexpected timestamp_precision values: {sorted(bad_precision)}')

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f'{name}: missing target')

for d, n in [
    (train_df, 'train'),
    (validation_df, 'validation'),
    (test_df, 'test'),
]:
    validate(d, n)

assert train_df.reaction_date.max() < validation_df.reaction_date.min()
assert validation_df.reaction_date.max() < test_df.reaction_date.min()

print(
    'Rows:',
    len(train_df),
    len(validation_df),
    len(test_df),
    'total=',
    len(train_df) + len(validation_df) + len(test_df),
)
print('Chronological split audit: PASSED')

# %% Code cell 15
all_df = pd.concat(
    [train_df, validation_df, test_df],
    ignore_index=True,
    sort=False,
).reset_index(drop=True)

all_df['event_uid'] = np.arange(len(all_df), dtype=np.int64)

ORIGINAL_ROW_COUNTS = {
    s: int((all_df['split'] == s).sum())
    for s in ['train', 'validation', 'test']
}
ORIGINAL_ROW_COUNTS['all'] = int(len(all_df))

# Conservative leakage-safe market-information cutoff:
# use only completed trading sessions STRICTLY BEFORE the news
# publication calendar date.
#
# This is especially important for FNSPID date-only timestamps. A record
# such as 2021-06-15 00:00 UTC does not tell us whether the article was
# published before or after the June 15 market close, so June 15 price/
# volume data must not be used as an input feature.
all_df['publication_date_utc'] = (
    all_df['publication_time_utc']
    .dt.tz_convert('UTC')
    .dt.tz_localize(None)
    .dt.normalize()
)

all_df['feature_cutoff_date'] = all_df['publication_date_utc']

if (all_df['feature_cutoff_date'] > all_df['reaction_date']).any():
    bad = int((all_df['feature_cutoff_date'] > all_df['reaction_date']).sum())
    raise RuntimeError(
        f'{bad} rows have feature_cutoff_date after reaction_date.'
    )

print('Frozen rows:', ORIGINAL_ROW_COUNTS)
print('Tickers:', all_df.ticker.nunique())
print(
    'Date range:',
    all_df.reaction_date.min(),
    'to',
    all_df.reaction_date.max(),
)
print(
    'Timestamp precision:',
    all_df.timestamp_precision.value_counts(dropna=False).to_dict(),
)
print(
    'Market predictors will use the latest completed trading session '
    'strictly BEFORE feature_cutoff_date (= publication calendar date).'
)

# %% Code cell 18
event_tokenizer = AutoTokenizer.from_pretrained(
    EVENT_MODEL_PATH,
    local_files_only=True,
    use_fast=True,
)

event_model = AutoModelForSequenceClassification.from_pretrained(
    EVENT_MODEL_PATH,
    local_files_only=True,
)

if int(event_model.config.num_labels) != 5:
    raise RuntimeError(
        f'Event model num_labels={event_model.config.num_labels}, expected 5.'
    )

def _normalize_event_id2label(mapping):
    out = {}
    for k, v in dict(mapping).items():
        try:
            key = int(k)
        except Exception:
            continue
        out[key] = re.sub(r'[^a-z]+', '', str(v).lower())
    return out

expected_event_mapping_norm = {
    i: re.sub(r'[^a-z]+', '', label.lower())
    for i, label in EXPECTED_EVENT_LABELS.items()
}

actual_event_mapping_norm = _normalize_event_id2label(
    event_model.config.id2label
)

if actual_event_mapping_norm != expected_event_mapping_norm:
    raise RuntimeError(
        'Event-model label mapping does not match the frozen project mapping.\n'
        f'Expected: {EXPECTED_EVENT_LABELS}\n'
        f'Found: {event_model.config.id2label}\n'
        'Upload the final 5-class event-classifier export, not a different '
        'FinBERT model/checkpoint.'
    )

event_model.to(DEVICE)
event_model.eval()

print('Event model loaded.')
print('Verified id2label:', event_model.config.id2label)
print('Event-label mapping audit: PASSED')

# %% Code cell 20
class FinBertSentimentRegressor(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        p = getattr(config, 'classifier_dropout', None)
        if p is None:
            p = 0.10

        self.dropout = nn.Dropout(p)
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
        o = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        y = self.tanh(
            self.regressor(
                self.dropout(o.pooler_output)
            )
        )

        loss = None
        if labels is not None:
            loss = self.loss_fn(
                y,
                labels.float().view(-1, 1),
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=y,
            hidden_states=o.hidden_states,
            attentions=o.attentions,
        )

sentiment_config = AutoConfig.from_pretrained(
    SENTIMENT_MODEL_PATH,
    local_files_only=True,
)

if int(getattr(sentiment_config, 'num_labels', -1)) != 1:
    raise RuntimeError(
        'Sentiment model is not the frozen one-output regression model. '
        f'num_labels={getattr(sentiment_config, "num_labels", None)}'
    )

if str(getattr(sentiment_config, 'problem_type', '')).lower() != 'regression':
    raise RuntimeError(
        'Sentiment config problem_type is not regression. '
        'Upload the final sentiment-regression export.'
    )

sentiment_tokenizer = AutoTokenizer.from_pretrained(
    SENTIMENT_MODEL_PATH,
    local_files_only=True,
    use_fast=True,
)

sentiment_model = FinBertSentimentRegressor.from_pretrained(
    SENTIMENT_MODEL_PATH,
    config=sentiment_config,
    local_files_only=True,
)

sentiment_model.to(DEVICE)
sentiment_model.eval()

if sentiment_model.regressor.out_features != 1:
    raise RuntimeError('Sentiment output dimension is not 1.')

print('Sentiment regressor loaded.')
print('hidden_size:', sentiment_model.config.hidden_size)
print('problem_type:', sentiment_model.config.problem_type)
print('Sentiment-model architecture audit: PASSED')

# %% Code cell 22
def run_event(texts,bs):
    out=[]
    for start in tqdm(range(0,len(texts),bs),desc='Event classifier',unit='batch'):
        enc=event_tokenizer(texts[start:start+bs],return_tensors='pt',padding=True,truncation=True,max_length=NLP_MAX_LENGTH)
        enc={k:v.to(DEVICE) for k,v in enc.items()}
        with torch.inference_mode(): out.append(torch.softmax(event_model(**enc).logits,dim=-1).cpu().numpy())
    return np.vstack(out)

def run_sentiment(texts,bs):
    out=[]
    for start in tqdm(range(0,len(texts),bs),desc='Sentiment regressor',unit='batch'):
        enc=sentiment_tokenizer(texts[start:start+bs],return_tensors='pt',padding=True,truncation=True,max_length=NLP_MAX_LENGTH)
        enc={k:v.to(DEVICE) for k,v in enc.items()}
        with torch.inference_mode(): out.append(sentiment_model(**enc).logits.view(-1).cpu().numpy())
    return np.concatenate(out)

def oom_retry(fn,texts,initial,name):
    bs=int(initial)
    while True:
        try:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return fn(texts,bs),bs
        except torch.cuda.OutOfMemoryError:
            if bs<=1: raise
            bs=max(1,bs//2); torch.cuda.empty_cache(); gc.collect(); print(name,'OOM -> retry batch',bs)

# %% Code cell 24
texts=all_df.headline.astype(str).tolist()
event_probs,actual_event_bs=oom_retry(run_event,texts,EVENT_BATCH_SIZE,'Event classifier')
if event_probs.shape!=(len(all_df),5): raise RuntimeError(f'Unexpected event shape {event_probs.shape}')
EVENT_PROB_COLUMNS=['event_prob_geopolitical','event_prob_macroeconomic','event_prob_credit_event','event_prob_merger_acquisition','event_prob_product_launch']
for i,c in enumerate(EVENT_PROB_COLUMNS): all_df[c]=event_probs[:,i].astype(np.float32)
pred=event_probs.argmax(1); conf=event_probs.max(1)
all_df['event_predicted_class_id']=pred.astype(np.int8)
all_df['event_predicted_class']=[EXPECTED_EVENT_LABELS[int(i)] for i in pred]
all_df['event_confidence']=conf.astype(np.float32)
assert np.allclose(all_df[EVENT_PROB_COLUMNS].sum(axis=1).to_numpy(),1.0,atol=1e-5)
print('Event inference complete. batch=',actual_event_bs)
print(all_df.event_predicted_class.value_counts())

# %% Code cell 26
sent,actual_sent_bs=oom_retry(run_sentiment,texts,SENTIMENT_BATCH_SIZE,'Sentiment')
sent=np.clip(sent.astype(np.float32),-1.0,1.0)
all_df['sentiment_score']=sent; all_df['abs_sentiment']=np.abs(sent).astype(np.float32)
def display_sent(x):
    if x>0.15: return 'positive'
    if x<-0.15: return 'negative'
    return 'neutral'
all_df['sentiment_display_label']=[display_sent(float(x)) for x in sent]
assert all_df.sentiment_score.between(-1,1).all()
print('Sentiment inference complete. batch=',actual_sent_bs)
print(all_df.sentiment_score.describe())
print(all_df.sentiment_display_label.value_counts())

# %% Code cell 27
display(all_df[['split','reaction_date','ticker','headline','event_predicted_class','event_confidence',*EVENT_PROB_COLUMNS,'sentiment_score','abs_sentiment']].head(20))

# %% Code cell 30
def find_price_zip():
    if PRICE_ARCHIVE_PATH_OVERRIDE:
        p=Path(PRICE_ARCHIVE_PATH_OVERRIDE)
        if not p.exists(): raise FileNotFoundError(p)
        return p
    cand=[p for p in INPUT_ROOT.rglob('full_history.zip') if ('fnspid' in str(p).lower() or 'stock' in str(p).lower())]
    return sorted(cand,key=lambda p:len(str(p)))[0] if cand else None
PRICE_ARCHIVE_PATH=find_price_zip(); price_archive_download_error=None
if PRICE_ARCHIVE_PATH is None:
    try: PRICE_ARCHIVE_PATH=Path(hf_hub_download(repo_id=PRICE_REPO,filename=PRICE_ARCHIVE_FILE,repo_type='dataset',token=HF_TOKEN))
    except Exception as e: price_archive_download_error=f'{type(e).__name__}: {e}'; PRICE_ARCHIVE_PATH=None
print('Price archive:',PRICE_ARCHIVE_PATH)
if PRICE_ARCHIVE_PATH: print('Size MB:',round(PRICE_ARCHIVE_PATH.stat().st_size/1024**2,1))
else: print('FNSPID archive unavailable; yfinance fallback only. Error:',price_archive_download_error)

# %% Code cell 32
def canon(c): return re.sub(r'[^a-z0-9]+','_',str(c).strip().lower()).strip('_')
def normalize_price_frame(raw,ticker,source):
    x=raw.copy()
    if isinstance(x.columns,pd.MultiIndex): x.columns=['_'.join(str(v) for v in t if str(v) not in {'','None',ticker}) for t in x.columns.to_flat_index()]
    cmap={canon(c):c for c in x.columns}
    dc=next((cmap[k] for k in ['date','datetime','timestamp'] if k in cmap),None)
    if dc is None and not isinstance(x.index,pd.RangeIndex):
        x=x.reset_index(); cmap={canon(c):c for c in x.columns}; dc=next((cmap[k] for k in ['date','datetime','timestamp','index'] if k in cmap),None)
    if dc is None: raise ValueError(f'{ticker}: date column missing')
    pc=None; adjusted=False
    for k in ['adj_close','adjusted_close','adjclose']:
        if k in cmap: pc=cmap[k]; adjusted=True; break
    if pc is None and 'close' in cmap: pc=cmap['close']
    if pc is None: raise ValueError(f'{ticker}: close missing')
    vc=cmap.get('volume'); dates=pd.to_datetime(x[dc],errors='coerce',utc=True)
    out=pd.DataFrame({'date':dates.dt.tz_convert('UTC').dt.tz_localize(None).dt.normalize(),'adj_close':pd.to_numeric(x[pc],errors='coerce')})
    out['volume']=pd.to_numeric(x[vc],errors='coerce') if vc is not None else np.nan
    out=out.dropna(subset=['date','adj_close']); out=out[out.adj_close>0].drop_duplicates('date',keep='last').sort_values('date').reset_index(drop=True)
    out['ticker']=ticker; out['price_source']=source; out['adjusted_price_used']=bool(adjusted)
    return out

zip_member_map={}
if PRICE_ARCHIVE_PATH:
    with zipfile.ZipFile(PRICE_ARCHIVE_PATH,'r') as zf: members=[m for m in zf.namelist() if m.lower().endswith('.csv') and not m.endswith('/')]
    for m in members: zip_member_map.setdefault(Path(m).stem.strip().upper(),m)
    print('FNSPID CSV members:',len(members))
required_tickers=sorted(all_df.ticker.astype(str).str.upper().unique()); symbols=required_tickers+[BENCHMARK]
print('Required stocks:',len(required_tickers),required_tickers)

# %% Code cell 33
import yfinance as yf
PSTART=(all_df.reaction_date.min()-pd.Timedelta(days=120)).strftime('%Y-%m-%d')
PEND=(all_df.reaction_date.max()+pd.Timedelta(days=10)).strftime('%Y-%m-%d')
def from_fnspid(sym):
    if PRICE_ARCHIVE_PATH is None: raise FileNotFoundError('archive unavailable')
    member=zip_member_map.get(sym.upper())
    if member is None: raise FileNotFoundError(f'{sym} missing in ZIP')
    with zipfile.ZipFile(PRICE_ARCHIVE_PATH,'r') as zf:
        with zf.open(member) as f: raw=pd.read_csv(f)
    return normalize_price_frame(raw,sym,'FNSPID_full_history')
def from_yf(sym):
    raw=yf.download(sym,start=PSTART,end=PEND,auto_adjust=False,progress=False,actions=False,threads=False)
    if raw is None or len(raw)==0: raise RuntimeError('no yfinance rows')
    return normalize_price_frame(raw.reset_index(),sym,'yfinance_fallback')
price_frames={}; price_load_report={}
for sym in tqdm(symbols,desc='Loading prices'):
    errs=[]
    try:
        pf=from_fnspid(sym)
        if len(pf)<40: raise ValueError(f'only {len(pf)} rows')
    except Exception as e:
        errs.append(f'FNSPID: {type(e).__name__}: {str(e)[:220]}')
        try: pf=from_yf(sym)
        except Exception as e2: errs.append(f'yfinance: {type(e2).__name__}: {str(e2)[:220]}'); price_load_report[sym]={'status':'failed','errors':errs}; continue
    price_frames[sym]=pf; price_load_report[sym]={'status':'ok','source':str(pf.price_source.iloc[0]),'rows':int(len(pf)),'adjusted_price_used':bool(pf.adjusted_price_used.iloc[0]),'volume_non_null':int(pf.volume.notna().sum()),'errors_before_success':errs}
missing=sorted(set(symbols)-set(price_frames))
if missing: raise RuntimeError('Missing required price histories: '+', '.join(missing))
print(json.dumps(price_load_report,indent=2))

# %% Code cell 35
def stock_features(pf):
    x = (
        pf[['date', 'adj_close', 'volume']]
        .copy()
        .sort_values('date')
        .reset_index(drop=True)
    )

    r = x.adj_close.pct_change()

    x['stock_volatility_20d_ann'] = (
        r.rolling(20, min_periods=20).std()
        * np.sqrt(252.0)
    )

    vm = x.volume.rolling(20, min_periods=20).mean()
    x['volume_ratio_20d'] = x.volume / vm

    x['stock_return_5d'] = x.adj_close.pct_change(5)
    x['stock_return_20d'] = x.adj_close.pct_change(20)

    x = x.replace([np.inf, -np.inf], np.nan)

    return x[[
        'date',
        'stock_volatility_20d_ann',
        'volume_ratio_20d',
        'stock_return_5d',
        'stock_return_20d',
    ]]

stock_tables = {
    t: stock_features(price_frames[t])
    for t in required_tickers
}

parts = []

for ticker, g in tqdm(
    all_df.groupby('ticker', sort=False),
    desc='Join pre-publication stock features',
):
    left = g.sort_values('feature_cutoff_date')

    right = (
        stock_tables[ticker]
        .rename(columns={'date': 'stock_feature_date'})
        .sort_values('stock_feature_date')
    )

    parts.append(
        pd.merge_asof(
            left,
            right,
            left_on='feature_cutoff_date',
            right_on='stock_feature_date',
            direction='backward',
            allow_exact_matches=False,
        )
    )

all_df = (
    pd.concat(parts, ignore_index=True)
    .sort_values('event_uid')
    .reset_index(drop=True)
)

mask = all_df.stock_feature_date.notna()

if not (
    all_df.loc[mask, 'stock_feature_date']
    <
    all_df.loc[mask, 'feature_cutoff_date']
).all():
    raise RuntimeError(
        'Stock-feature leakage: feature date is not strictly before '
        'publication calendar date.'
    )

all_df['stock_feature_gap_days'] = (
    all_df.feature_cutoff_date
    - all_df.stock_feature_date
).dt.days

print('Stock pre-publication strict-date audit: PASSED')
print(all_df.stock_feature_gap_days.describe())

# %% Code cell 37
spy = (
    price_frames[BENCHMARK][['date', 'adj_close']]
    .copy()
    .sort_values('date')
    .reset_index(drop=True)
)

sr = spy.adj_close.pct_change()

spy['spy_return_20d'] = spy.adj_close.pct_change(20)
spy['spy_volatility_20d_ann'] = (
    sr.rolling(20, min_periods=20).std()
    * np.sqrt(252.0)
)

spy_feat = (
    spy[[
        'date',
        'spy_return_20d',
        'spy_volatility_20d_ann',
    ]]
    .rename(columns={'date': 'spy_feature_date'})
)

all_df = (
    pd.merge_asof(
        all_df.sort_values('feature_cutoff_date'),
        spy_feat.sort_values('spy_feature_date'),
        left_on='feature_cutoff_date',
        right_on='spy_feature_date',
        direction='backward',
        allow_exact_matches=False,
    )
    .sort_values('event_uid')
    .reset_index(drop=True)
)

mask = all_df.spy_feature_date.notna()

if not (
    all_df.loc[mask, 'spy_feature_date']
    <
    all_df.loc[mask, 'feature_cutoff_date']
).all():
    raise RuntimeError(
        'SPY-feature leakage: feature date is not strictly before '
        'publication calendar date.'
    )

all_df['spy_feature_gap_days'] = (
    all_df.feature_cutoff_date
    - all_df.spy_feature_date
).dt.days

print('SPY pre-publication strict-date audit: PASSED')
print(all_df.spy_feature_gap_days.describe())

# %% Code cell 39
train_end=train_df.reaction_date.max(); hist=spy[spy.date<train_end][['spy_return_20d','spy_volatility_20d_ann']].dropna()
if len(hist)<100: raise RuntimeError('Insufficient SPY training history')
ret_q33=float(hist.spy_return_20d.quantile(1/3)); ret_q67=float(hist.spy_return_20d.quantile(2/3)); vol_q75=float(hist.spy_volatility_20d_ann.quantile(.75))
MARKET_REGIME_THRESHOLDS={'calibration_source':'SPY history before frozen training end','training_end_date':str(pd.Timestamp(train_end).date()),'spy_return_20d_q33':ret_q33,'spy_return_20d_q67':ret_q67,'spy_volatility_20d_ann_q75':vol_q75}
def dreg(x):
    if pd.isna(x): return 'unknown'
    if x<ret_q33: return 'bearish'
    if x>ret_q67: return 'bullish'
    return 'neutral'
def vreg(x):
    if pd.isna(x): return 'unknown'
    return 'high_vol' if x>vol_q75 else 'normal_vol'
all_df['market_direction_regime']=all_df.spy_return_20d.map(dreg); all_df['market_volatility_regime']=all_df.spy_volatility_20d_ann.map(vreg); all_df['market_regime']=all_df.market_direction_regime+'_'+all_df.market_volatility_regime
print(json.dumps(MARKET_REGIME_THRESHOLDS,indent=2)); print(all_df.market_regime.value_counts())

# %% Code cell 41
all_df['sector']=all_df.sector.replace('',np.nan).fillna(all_df.ticker.map(TICKER_SECTOR))
if all_df.sector.isna().any(): print('WARNING missing sector tickers:',all_df.loc[all_df.sector.isna(),'ticker'].value_counts().to_dict())
else: print('Sector coverage: complete')

# %% Code cell 43
MODEL_NUMERIC_FEATURES=[
'sentiment_score','abs_sentiment',
'event_prob_geopolitical','event_prob_macroeconomic','event_prob_credit_event','event_prob_merger_acquisition','event_prob_product_launch','event_confidence',
'stock_volatility_20d_ann','volume_ratio_20d','stock_return_5d','stock_return_20d','spy_return_20d','spy_volatility_20d_ann']
MODEL_CATEGORICAL_FEATURES=['event_predicted_class','sector','market_regime']
MODEL_FEATURES=MODEL_NUMERIC_FEATURES+MODEL_CATEGORICAL_FEATURES
print('Numeric:',len(MODEL_NUMERIC_FEATURES),'Categorical:',len(MODEL_CATEGORICAL_FEATURES),'Total:',len(MODEL_FEATURES))
for x in MODEL_FEATURES: print(' ',x)

# %% Code cell 44
bad=sorted(set(MODEL_FEATURES)&FORBIDDEN_FUTURE_FEATURES)
if bad: raise RuntimeError('Forbidden future predictors: '+', '.join(bad))
if TARGET_COLUMN in MODEL_FEATURES: raise RuntimeError('Target leaked into predictor list')
patterns=['realized_impact','impact_score','abnormal_return','car_3d','stock_return_1d','benchmark_return_1d','session_news_count']
susp=[f for f in MODEL_FEATURES if any(p in f.lower() for p in patterns)]
if susp: raise RuntimeError('Suspicious predictor names: '+', '.join(susp))
print('Predictor-name leakage audit: PASSED')

# %% Code cell 45
NLP_REQUIRED=['sentiment_score','abs_sentiment',*EVENT_PROB_COLUMNS,'event_confidence','event_predicted_class']
if all_df[NLP_REQUIRED].isna().any().any(): raise RuntimeError('Missing NLP outputs')
assert all_df.sentiment_score.between(-1,1).all(); assert all_df.event_confidence.between(0,1).all(); assert np.allclose(all_df[EVENT_PROB_COLUMNS].sum(axis=1).to_numpy(),1,atol=1e-5)
missing_report=all_df[MODEL_FEATURES].isna().sum().sort_values(ascending=False).to_frame('missing_rows'); missing_report['missing_rate']=missing_report.missing_rows/len(all_df)
display(missing_report)

# %% Code cell 46
CURRENT_ROW_COUNTS = {
    s: int((all_df.split == s).sum())
    for s in ['train', 'validation', 'test']
}
CURRENT_ROW_COUNTS['all'] = int(len(all_df))

if CURRENT_ROW_COUNTS != ORIGINAL_ROW_COUNTS:
    raise RuntimeError(
        f'Row counts changed: before={ORIGINAL_ROW_COUNTS}, '
        f'after={CURRENT_ROW_COUNTS}'
    )

assert (
    all_df.loc[
        all_df.split == 'train',
        'reaction_date',
    ].max()
    <
    all_df.loc[
        all_df.split == 'validation',
        'reaction_date',
    ].min()
)

assert (
    all_df.loc[
        all_df.split == 'validation',
        'reaction_date',
    ].max()
    <
    all_df.loc[
        all_df.split == 'test',
        'reaction_date',
    ].min()
)

stock_viol = int(
    (
        all_df.stock_feature_date.notna()
        &
        (
            all_df.stock_feature_date
            >= all_df.feature_cutoff_date
        )
    ).sum()
)

spy_viol = int(
    (
        all_df.spy_feature_date.notna()
        &
        (
            all_df.spy_feature_date
            >= all_df.feature_cutoff_date
        )
    ).sum()
)

cutoff_reaction_viol = int(
    (
        all_df.feature_cutoff_date
        > all_df.reaction_date
    ).sum()
)

if stock_viol or spy_viol or cutoff_reaction_viol:
    raise RuntimeError(
        'Temporal leakage/cutoff audit failed: '
        f'stock={stock_viol}, SPY={spy_viol}, '
        f'cutoff_after_reaction={cutoff_reaction_viol}'
    )

print('Rows preserved:', CURRENT_ROW_COUNTS)
print(
    'Temporal violations: stock=',
    stock_viol,
    'SPY=',
    spy_viol,
    'cutoff_after_reaction=',
    cutoff_reaction_viol,
)
print('Split + strict pre-publication temporal audits: PASSED')

# %% Code cell 48
PREVIEW_COLS = [
    'split',
    'publication_time_utc',
    'timestamp_precision',
    'publication_date_utc',
    'feature_cutoff_date',
    'reaction_date',
    'ticker',
    'sector',
    'headline',
    'event_predicted_class',
    'event_confidence',
    'sentiment_score',
    'abs_sentiment',
    'stock_feature_date',
    'stock_volatility_20d_ann',
    'volume_ratio_20d',
    'stock_return_5d',
    'stock_return_20d',
    'spy_feature_date',
    'spy_return_20d',
    'spy_volatility_20d_ann',
    'market_regime',
    'realized_impact',
    'impact_score_1_10',
]

display(all_df[PREVIEW_COLS].head(20))

# %% Code cell 50
OUTPUT_DIR=WORKDIR/'impact_step2_features'; OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
def split_sorted(name): return all_df[all_df.split==name].sort_values(['reaction_date','ticker','event_uid']).reset_index(drop=True)
train_features=split_sorted('train'); validation_features=split_sorted('validation'); test_features=split_sorted('test'); all_features=all_df.sort_values(['reaction_date','ticker','event_uid']).reset_index(drop=True)
OUTPUT_PATHS={'all':OUTPUT_DIR/'impact_step2_features_all.parquet','train':OUTPUT_DIR/'impact_step2_train_features.parquet','validation':OUTPUT_DIR/'impact_step2_validation_features.parquet','test':OUTPUT_DIR/'impact_step2_test_features.parquet'}
all_features.to_parquet(OUTPUT_PATHS['all'],index=False); train_features.to_parquet(OUTPUT_PATHS['train'],index=False); validation_features.to_parquet(OUTPUT_PATHS['validation'],index=False); test_features.to_parquet(OUTPUT_PATHS['test'],index=False)
for k,p in OUTPUT_PATHS.items(): print(k,'->',p)

# %% Code cell 51
feature_manifest = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'stage': 'Impact Engine Step 2 - leakage-safe feature enrichment',
    'target_column': TARGET_COLUMN,
    'numeric_features': MODEL_NUMERIC_FEATURES,
    'categorical_features': MODEL_CATEGORICAL_FEATURES,
    'model_features': MODEL_FEATURES,
    'forbidden_future_features': sorted(FORBIDDEN_FUTURE_FEATURES),
    'notes': {
        'sentiment_score': (
            'continuous frozen FinBERT regression in [-1,+1]'
        ),
        'event_probabilities': (
            'frozen 5-class event softmax; label order hard-validated'
        ),
        'market_features': (
            'latest completed market session strictly before the '
            'news publication calendar date'
        ),
        'date_only_safety': (
            'same-calendar-day market data is never used for date-only '
            'FNSPID news, because exact publication time is unknown'
        ),
        'market_regime': (
            'thresholds calibrated from training-period SPY history only'
        ),
        'text_embeddings': (
            'deferred until after baseline Impact model works'
        ),
    },
}

with open(
    OUTPUT_DIR / 'impact_step2_feature_manifest.json',
    'w',
    encoding='utf-8',
) as f:
    json.dump(feature_manifest, f, indent=2)

with open(
    OUTPUT_DIR / 'impact_step2_market_regime_thresholds.json',
    'w',
    encoding='utf-8',
) as f:
    json.dump(MARKET_REGIME_THRESHOLDS, f, indent=2)

with open(
    OUTPUT_DIR / 'impact_decile_thresholds.json',
    'w',
    encoding='utf-8',
) as f:
    json.dump(impact_deciles, f, indent=2)

missing_payload = {
    c: {
        'missing_rows': int(all_df[c].isna().sum()),
        'missing_rate': float(all_df[c].isna().mean()),
    }
    for c in MODEL_FEATURES
}

report = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'stage': 'Impact Engine Step 2 - leakage-safe feature enrichment',
    'inputs': {
        'event_model_path': str(EVENT_MODEL_PATH),
        'sentiment_model_path': str(SENTIMENT_MODEL_PATH),
        'impact_step1_paths': {
            k: str(v)
            for k, v in IMPACT_PATHS.items()
        },
        'price_archive_path': (
            None
            if PRICE_ARCHIVE_PATH is None
            else str(PRICE_ARCHIVE_PATH)
        ),
        'price_archive_download_error': price_archive_download_error,
    },
    'row_counts': CURRENT_ROW_COUNTS,
    'ticker_count': int(all_df.ticker.nunique()),
    'date_range': {
        'start': str(
            pd.Timestamp(
                all_df.reaction_date.min()
            ).date()
        ),
        'end': str(
            pd.Timestamp(
                all_df.reaction_date.max()
            ).date()
        ),
    },
    'timestamp_precision_counts': {
        str(k): int(v)
        for k, v in all_df.timestamp_precision.value_counts(
            dropna=False
        ).to_dict().items()
    },
    'nlp': {
        'max_length': NLP_MAX_LENGTH,
        'event_batch_size_used': int(actual_event_bs),
        'sentiment_batch_size_used': int(actual_sent_bs),
        'event_label_mapping_verified': True,
        'event_predicted_class_counts': {
            str(k): int(v)
            for k, v in all_df.event_predicted_class.value_counts().to_dict().items()
        },
        'sentiment_summary': {
            'min': float(all_df.sentiment_score.min()),
            'mean': float(all_df.sentiment_score.mean()),
            'median': float(all_df.sentiment_score.median()),
            'max': float(all_df.sentiment_score.max()),
        },
    },
    'prices': {
        'price_load_report': price_load_report,
        'stock_feature_date_violations': stock_viol,
        'spy_feature_date_violations': spy_viol,
        'feature_cutoff_after_reaction_violations': cutoff_reaction_viol,
        'feature_cutoff_rule': (
            'latest completed market date strictly before '
            'publication calendar date'
        ),
    },
    'market_regime_thresholds': MARKET_REGIME_THRESHOLDS,
    'market_regime_counts': {
        str(k): int(v)
        for k, v in all_df.market_regime.value_counts(
            dropna=False
        ).to_dict().items()
    },
    'feature_manifest': feature_manifest,
    'missing_features': missing_payload,
    'anti_leakage_assertions': {
        'frozen_split_preserved': True,
        'all_original_rows_preserved': True,
        'market_feature_dates_strictly_before_publication_date': True,
        'same_day_market_data_blocked_for_date_only_news': True,
        'impact_deciles_not_recomputed': True,
        'future_target_fields_excluded_from_model_features': True,
        'market_regime_calibrated_without_validation_or_test': True,
    },
    'outputs': {
        k: str(v)
        for k, v in OUTPUT_PATHS.items()
    },
}

with open(
    OUTPUT_DIR / 'impact_step2_report.json',
    'w',
    encoding='utf-8',
) as f:
    json.dump(report, f, indent=2)

preview = all_features[PREVIEW_COLS].copy()

if len(preview) > 600:
    preview = preview.iloc[
        np.linspace(
            0,
            len(preview) - 1,
            600,
            dtype=int,
        )
    ]

preview.to_csv(
    OUTPUT_DIR / 'impact_step2_preview.csv',
    index=False,
)

missing_report.to_csv(
    OUTPUT_DIR / 'impact_step2_missing_feature_report.csv'
)

print('Saved reports/manifests to', OUTPUT_DIR)

# %% Code cell 52
bundle=WORKDIR/'impact_step2_feature_bundle.zip'
if bundle.exists(): bundle.unlink()
with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(OUTPUT_DIR.rglob('*')):
        if p.is_file(): zf.write(p,arcname=p.relative_to(OUTPUT_DIR))
print('Bundle:',bundle,'size MB=',round(bundle.stat().st_size/1024**2,2))

# %% Code cell 54
print('='*72); print('IMPACT ENGINE STEP 2 — COMPLETE'); print('='*72)
print('Rows preserved:',CURRENT_ROW_COUNTS); print('Tickers:',all_df.ticker.nunique()); print('Model features:',len(MODEL_FEATURES)); print('Temporal leakage violations: stock=',stock_viol,'SPY=',spy_viol)
print('Output directory:',OUTPUT_DIR); print('Bundle:',bundle)
print(); print('NEXT INPUTS FOR IMPACT STEP 3:')
for name in ['impact_step2_train_features.parquet','impact_step2_validation_features.parquet','impact_step2_test_features.parquet','impact_step2_feature_manifest.json','impact_decile_thresholds.json','impact_step2_report.json']: print(' -',name)
