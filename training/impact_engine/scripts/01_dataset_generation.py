#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Faithful Python export of original notebook: impact-engine-dataset-generation (1).ipynb.

Kaggle-friendly original behavior; Python export adds environment variable overrides
for input/output locations and removes notebook-only pip directives. It does not
fabricate pretrained weights, datasets, or experiment results.
"""
import os
from IPython.display import display


# %% Code cell 2
# Notebook-only %pip installation omitted; see ../requirements.txt

# %% Code cell 4
import os, re, json, math, random, zipfile
from pathlib import Path
from datetime import datetime, timezone, time
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download

SEED=42
random.seed(SEED); np.random.seed(SEED)
WORKDIR = Path(os.environ.get('RISKPULSE_IMPACT_WORKDIR', str(Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()))).expanduser().resolve()
WORKDIR.mkdir(parents=True, exist_ok=True)
print('WORKDIR:', WORKDIR)

# %% Code cell 6
HF_TOKEN=None
try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login
    HF_TOKEN=UserSecretsClient().get_secret('HF_TOKEN')
    if HF_TOKEN:
        login(token=HF_TOKEN, add_to_git_credential=False)
        print('Hugging Face login successful.')
    else:
        print('HF_TOKEN empty; continuing anonymously.')
except Exception as e:
    print('No usable HF_TOKEN; continuing anonymously:', type(e).__name__)

# %% Code cell 8
TICKER_SECTOR={
'AAPL':'Technology','MSFT':'Technology','GOOGL':'Communication Services','META':'Communication Services',
'AMZN':'Consumer Discretionary','TSLA':'Consumer Discretionary','NVDA':'Technology','AMD':'Technology',
'INTC':'Technology','QCOM':'Technology','CSCO':'Technology','ADBE':'Technology','NFLX':'Communication Services',
'PEP':'Consumer Staples','COST':'Consumer Staples','SBUX':'Consumer Discretionary','BKNG':'Consumer Discretionary',
'AMGN':'Health Care','GILD':'Health Care','REGN':'Health Care','HON':'Industrials','CSX':'Industrials',
'ADP':'Industrials','MDLZ':'Consumer Staples','CMCSA':'Communication Services'}
SELECTED_TICKERS=list(TICKER_SECTOR)
DATE_START=pd.Timestamp('2015-01-01')
DATE_END=pd.Timestamp('2023-12-31 23:59:59')
NEWS_DATASET_CANDIDATES=[
    'benstaf/FNSPID-filtered-nasdaq-100',
    'benstaf/FNSPID-nasdaq-100-post2019-1newsperrow',
    'sabareesh88/FNSPID_external',
]
PRICE_REPO='Zihan1004/FNSPID'
PRICE_ARCHIVE_FILE='Stock_price/full_history.zip'
BENCHMARK='SPY'
MIN_HEADLINE_CHARS=12
MIN_HEADLINE_WORDS=3
MIN_FINAL_EVENTS_PER_TICKER=30
TRAIN_FRACTION=0.70
VALIDATION_FRACTION=0.15
TEST_FRACTION=0.15
MARKET_CLOSE_ET=time(16,0)
assert abs(TRAIN_FRACTION+VALIDATION_FRACTION+TEST_FRACTION-1)<1e-9
print('Tickers:',len(SELECTED_TICKERS))
print('Requested range:',DATE_START.date(),'to',DATE_END.date())

# %% Code cell 10
PRICE_ACTION_PATTERNS=[
 r'\bshares? (?:are |is )?trading (?:higher|lower|up|down)\b',
 r'\bstocks? (?:are )?moving\b', r'\bbiggest movers?\b', r'\bmarket movers?\b',
 r'\btop (?:gainers?|losers?)\b', r'\b52[- ]week (?:highs?|lows?)\b',
 r'\bunusual (?:trading )?volume\b', r'\bmost active stocks?\b',
 r'\bstocks? making (?:the )?biggest moves\b',
 r'\bwhy .* shares? (?:are |is )?(?:up|down|higher|lower|moving)\b']
PRICE_ACTION_RX=re.compile('|'.join(f'(?:{p})' for p in PRICE_ACTION_PATTERNS),re.I)
def is_price_action_headline(x): return bool(PRICE_ACTION_RX.search(str(x)))
print('Leakage filter ready.')

# %% Code cell 12
NEWS_COLUMNS=['Date','Article_title','Stock_symbol','Url','Publisher']
news_stream=None; NEWS_DATASET_USED=None; news_load_errors={}
for repo in NEWS_DATASET_CANDIDATES:
    try:
        print('Trying:',repo)
        ds=load_dataset(repo, split='train', streaming=True, token=HF_TOKEN)
        missing=set(NEWS_COLUMNS)-set(ds.column_names)
        if missing: raise ValueError(f'missing columns {sorted(missing)}')
        ds=ds.select_columns(NEWS_COLUMNS)
        print('Example:',next(iter(ds.take(1))))
        news_stream=ds; NEWS_DATASET_USED=repo
        print('Using:',repo)
        break
    except Exception as e:
        news_load_errors[repo]=f'{type(e).__name__}: {str(e)[:350]}'
        print(' failed:',news_load_errors[repo])
if news_stream is None: raise RuntimeError(json.dumps(news_load_errors,indent=2))

# %% Code cell 14
rows=[]; scan=Counter(); ticker_set=set(SELECTED_TICKERS)
for rec in tqdm(news_stream, desc='Scanning FNSPID news', unit='rows'):
    scan['rows_scanned']+=1
    ticker=str(rec.get('Stock_symbol') or '').strip().upper()
    if ticker not in ticker_set:
        scan['not_selected_ticker']+=1; continue
    raw_date=rec.get('Date'); headline=rec.get('Article_title')
    if raw_date is None or headline is None:
        scan['missing_date_or_headline']+=1; continue
    ts=pd.to_datetime(raw_date, utc=True, errors='coerce')
    if pd.isna(ts): scan['unparseable_timestamp']+=1; continue
    naive=ts.tz_convert('UTC').tz_localize(None)
    if naive<DATE_START or naive>DATE_END:
        scan['outside_date_range']+=1; continue
    headline=str(headline).strip()
    if len(headline)<MIN_HEADLINE_CHARS or len(headline.split())<MIN_HEADLINE_WORDS:
        scan['too_short']+=1; continue
    if is_price_action_headline(headline):
        scan['price_action_leakage']+=1; continue
    rows.append({
        'publication_time_utc':ts,'publication_time_raw':str(raw_date),'headline':headline,
        'ticker':ticker,'url':None if rec.get('Url') is None else str(rec.get('Url')),
        'publisher':None if rec.get('Publisher') is None else str(rec.get('Publisher')),
        'source_dataset':NEWS_DATASET_USED})
    scan['selected_before_dedup']+=1
print(json.dumps(dict(scan),indent=2))
if not rows: raise RuntimeError('No selected news rows survived.')
news_df=pd.DataFrame(rows)
print('Selected rows:',len(news_df)); display(news_df.head())

# %% Code cell 16
def normalize_headline(x):
    x=str(x).lower(); x=re.sub(r'https?://\S+',' ',x); x=re.sub(r'[^a-z0-9$%+\- ]+',' ',x); return re.sub(r'\s+',' ',x).strip()
news_df['normalized_headline']=news_df['headline'].map(normalize_headline)
news_df['publication_utc_date']=news_df['publication_time_utc'].dt.tz_convert('UTC').dt.date
dup=news_df.duplicated(['ticker','publication_utc_date','normalized_headline'],keep='first')
exact_duplicates_removed=int(dup.sum())
if dup.any(): news_df.loc[dup].to_csv(WORKDIR/'impact_exact_duplicate_headlines_removed.csv',index=False)
news_df=news_df.loc[~dup].sort_values(['ticker','publication_time_utc']).reset_index(drop=True)
print('Exact duplicates removed:',exact_duplicates_removed)
print('Rows after dedup:',len(news_df))

# %% Code cell 18
PRICE_ARCHIVE_PATH=None; price_archive_error=None
try:
    PRICE_ARCHIVE_PATH=Path(hf_hub_download(repo_id=PRICE_REPO,filename=PRICE_ARCHIVE_FILE,repo_type='dataset',token=HF_TOKEN))
    print('Archive:',PRICE_ARCHIVE_PATH)
    print('Size MB:',round(PRICE_ARCHIVE_PATH.stat().st_size/1024**2,1))
except Exception as e:
    price_archive_error=f'{type(e).__name__}: {str(e)[:500]}'
    print('FNSPID archive unavailable; yfinance fallback will be used:',price_archive_error)

# %% Code cell 20
zip_member_map={}; zip_member_sample=[]
if PRICE_ARCHIVE_PATH is not None:
    with zipfile.ZipFile(PRICE_ARCHIVE_PATH,'r') as zf:
        members=[m for m in zf.namelist() if m.lower().endswith('.csv') and not m.endswith('/')]
    zip_member_sample=members[:20]
    for m in members: zip_member_map.setdefault(Path(m).stem.strip().upper(),m)
    print('CSV members:',len(members)); print('Sample:',zip_member_sample[:10])
    found=[s for s in SELECTED_TICKERS+[BENCHMARK] if s in zip_member_map]
    print('Requested symbols found:',len(found),found)
else:
    print('No ZIP available.')

# %% Code cell 22
def canon(c): return re.sub(r'[^a-z0-9]+','_',str(c).strip().lower()).strip('_')
def normalize_price_frame(raw,ticker,source_name):
    x=raw.copy()
    if isinstance(x.columns,pd.MultiIndex):
        x.columns=['_'.join(str(v) for v in tup if str(v) not in {'','None',ticker}) for tup in x.columns.to_flat_index()]
    cmap={canon(c):c for c in x.columns}
    date_col=next((cmap[k] for k in ['date','datetime','timestamp'] if k in cmap),None)
    if date_col is None and not isinstance(x.index,pd.RangeIndex):
        x=x.reset_index(); cmap={canon(c):c for c in x.columns}; date_col=next((cmap[k] for k in ['date','datetime','timestamp','index'] if k in cmap),None)
    if date_col is None: raise ValueError(f'{ticker}: date column missing: {list(x.columns)}')
    price_col=None; adjusted=False
    for k in ['adj_close','adjusted_close','adjclose']:
        if k in cmap: price_col=cmap[k]; adjusted=True; break
    if price_col is None and 'close' in cmap: price_col=cmap['close']
    if price_col is None: raise ValueError(f'{ticker}: close column missing')
    volume_col=cmap.get('volume')
    dates=pd.to_datetime(x[date_col],errors='coerce',utc=True)
    out=pd.DataFrame({'date':dates.dt.tz_convert('UTC').dt.tz_localize(None).dt.normalize(),
                      'adj_close':pd.to_numeric(x[price_col],errors='coerce')})
    out['volume']=pd.to_numeric(x[volume_col],errors='coerce') if volume_col is not None else np.nan
    out=out.dropna(subset=['date','adj_close']); out=out[out['adj_close']>0]
    out=out.drop_duplicates('date',keep='last').sort_values('date').reset_index(drop=True)
    out['ticker']=ticker; out['price_source']=source_name; out['adjusted_price_used']=bool(adjusted)
    return out

# %% Code cell 24
import yfinance as yf
PSTART=(DATE_START-pd.Timedelta(days=60)).strftime('%Y-%m-%d')
PEND=(DATE_END+pd.Timedelta(days=20)).strftime('%Y-%m-%d')

def from_fnspid(symbol):
    if PRICE_ARCHIVE_PATH is None: raise FileNotFoundError('archive unavailable')
    member=zip_member_map.get(symbol.upper())
    if member is None: raise FileNotFoundError(f'{symbol} missing in ZIP')
    with zipfile.ZipFile(PRICE_ARCHIVE_PATH,'r') as zf:
        with zf.open(member) as f: raw=pd.read_csv(f)
    return normalize_price_frame(raw,symbol,'FNSPID_full_history')

def from_yfinance(symbol):
    raw=yf.download(symbol,start=PSTART,end=PEND,auto_adjust=False,progress=False,actions=False,threads=False)
    if raw is None or len(raw)==0: raise RuntimeError('no yfinance rows')
    raw=raw.reset_index()
    if isinstance(raw.columns,pd.MultiIndex):
        raw.columns=['_'.join(str(v) for v in tup if str(v) not in {'','None',symbol}) for tup in raw.columns.to_flat_index()]
    return normalize_price_frame(raw,symbol,'yfinance_fallback')

price_frames={}; price_load_report={}
for symbol in tqdm(SELECTED_TICKERS+[BENCHMARK],desc='Loading prices'):
    errs=[]
    try:
        pf=from_fnspid(symbol)
        if len(pf)<50: raise ValueError(f'only {len(pf)} rows')
    except Exception as e:
        errs.append(f'FNSPID: {type(e).__name__}: {str(e)[:220]}')
        try: pf=from_yfinance(symbol)
        except Exception as e2:
            errs.append(f'yfinance: {type(e2).__name__}: {str(e2)[:220]}')
            price_load_report[symbol]={'status':'failed','errors':errs}; continue
    price_frames[symbol]=pf
    price_load_report[symbol]={'status':'ok','source':str(pf.price_source.iloc[0]),'rows':int(len(pf)),
                               'adjusted_price_used':bool(pf.adjusted_price_used.iloc[0]),'errors_before_success':errs}
print(json.dumps(price_load_report,indent=2))
if BENCHMARK not in price_frames: raise RuntimeError('SPY benchmark could not be loaded.')

# %% Code cell 26
usable=set(price_frames)-{BENCHMARK}; missing_price=sorted(set(SELECTED_TICKERS)-usable)
print('Usable stock tickers:',len(usable)); print('Missing:',missing_price)
news_df=news_df[news_df.ticker.isin(usable)].copy().reset_index(drop=True)
print('News rows with prices:',len(news_df))

# %% Code cell 28
def ts_precision(ts):
    t=pd.Timestamp(ts).tz_convert('UTC')
    return 'date_only' if (t.hour,t.minute,t.second)==(0,0,0) else 'exact_time'
news_df['timestamp_precision']=news_df.publication_time_utc.map(ts_precision)
print(news_df.timestamp_precision.value_counts())

# %% Code cell 30
trading_dates={t:pf.date.to_numpy(dtype='datetime64[ns]') for t,pf in price_frames.items() if t!=BENCHMARK}
def first_on_or_after(arr,d):
    d=np.datetime64(pd.Timestamp(d).normalize(),'ns'); pos=np.searchsorted(arr,d,'left')
    return pd.NaT if pos>=len(arr) else pd.Timestamp(arr[pos])
def first_after(arr,d):
    d=np.datetime64(pd.Timestamp(d).normalize(),'ns'); pos=np.searchsorted(arr,d,'right')
    return pd.NaT if pos>=len(arr) else pd.Timestamp(arr[pos])
def reaction_date(row):
    arr=trading_dates[row.ticker]; ts=pd.Timestamp(row.publication_time_utc)
    if row.timestamp_precision=='date_only':
        return first_after(arr,ts.tz_convert('UTC').tz_localize(None).normalize())
    et=ts.tz_convert('America/New_York'); local=pd.Timestamp(et.date())
    pos=np.searchsorted(arr,np.datetime64(local,'ns'),'left')
    if pos<len(arr) and pd.Timestamp(arr[pos])==local and et.time()<MARKET_CLOSE_ET: return local
    return first_after(arr,local)

tqdm.pandas(desc='Mapping reaction sessions')
news_df['reaction_date']=news_df.progress_apply(reaction_date,axis=1)
reaction_mapping_failures=int(news_df.reaction_date.isna().sum())
if reaction_mapping_failures:
    news_df[news_df.reaction_date.isna()].to_csv(WORKDIR/'impact_reaction_date_mapping_failures.csv',index=False)
news_df=news_df.dropna(subset=['reaction_date']).sort_values(['ticker','reaction_date','publication_time_utc']).reset_index(drop=True)
print('Mapping failures:',reaction_mapping_failures); print('Mapped rows:',len(news_df))

# %% Code cell 32
counts=news_df.groupby(['ticker','reaction_date']).size().rename('session_news_count').reset_index()
news_df=news_df.merge(counts,on=['ticker','reaction_date'],how='left')
before_session=len(news_df)
events_df=(news_df.sort_values(['ticker','reaction_date','publication_time_utc'])
           .drop_duplicates(['ticker','reaction_date'],keep='first').reset_index(drop=True))
session_rows_removed=before_session-len(events_df)
print('Rows before:',before_session,'removed:',session_rows_removed,'events:',len(events_df))

# %% Code cell 34
benchmark_df=price_frames[BENCHMARK][['date','adj_close']].copy().rename(columns={'adj_close':'benchmark_close'}).sort_values('date')
benchmark_df['benchmark_return_1d']=benchmark_df.benchmark_close.pct_change()
print('SPY range:',benchmark_df.date.min(),'to',benchmark_df.date.max(),'rows=',len(benchmark_df))

# %% Code cell 36
target_tables={}
for ticker in sorted(usable):
    stock=price_frames[ticker][['date','adj_close','volume','price_source','adjusted_price_used']].copy().sort_values('date')
    stock['stock_return_1d']=stock.adj_close.pct_change()
    m=stock.merge(benchmark_df[['date','benchmark_close','benchmark_return_1d']],on='date',how='inner')
    m['abnormal_return_1d']=m.stock_return_1d-m.benchmark_return_1d
    m['car_3d']=m.abnormal_return_1d+m.abnormal_return_1d.shift(-1)+m.abnormal_return_1d.shift(-2)
    m['realized_impact']=np.maximum(m.abnormal_return_1d.abs(),m.car_3d.abs())
    target_tables[ticker]=m
print('Target tables:',len(target_tables))

# %% Code cell 38
frames=[]; target_join_report={}
for ticker,g in events_df.groupby('ticker'):
    t=target_tables[ticker][['date','adj_close','benchmark_close','stock_return_1d','benchmark_return_1d',
                             'abnormal_return_1d','car_3d','realized_impact','price_source','adjusted_price_used']].rename(columns={'date':'reaction_date'})
    m=g.merge(t,on='reaction_date',how='left')
    miss=m[['abnormal_return_1d','car_3d','realized_impact']].isna().any(axis=1)
    target_join_report[ticker]={'events_before_target_join':int(len(m)),'missing_complete_target':int(miss.sum()),'usable_events':int((~miss).sum())}
    frames.append(m.loc[~miss].copy())
impact_df=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
if impact_df.empty: raise RuntimeError('No events received complete targets.')
impact_df['sector']=impact_df.ticker.map(TICKER_SECTOR)
impact_df['extreme_target_flag']=impact_df.realized_impact>0.50
if impact_df.extreme_target_flag.any():
    impact_df[impact_df.extreme_target_flag].sort_values('realized_impact',ascending=False).to_csv(WORKDIR/'impact_extreme_target_review.csv',index=False)
print('Events with targets:',len(impact_df),'extreme flags:',int(impact_df.extreme_target_flag.sum()))

# %% Code cell 40
pre_coverage=impact_df.ticker.value_counts().sort_index()
eligible=pre_coverage[pre_coverage>=MIN_FINAL_EVENTS_PER_TICKER].index.tolist()
dropped=pre_coverage[pre_coverage<MIN_FINAL_EVENTS_PER_TICKER].to_dict()
impact_df=impact_df[impact_df.ticker.isin(eligible)].sort_values(['reaction_date','ticker']).reset_index(drop=True)
print('Eligible tickers:',len(eligible),eligible); print('Dropped:',dropped); print('Events:',len(impact_df))
if len(eligible)<5: raise RuntimeError('Too few tickers survived. Inspect the report before proceeding.')

# %% Code cell 42
assert not impact_df.duplicated(['ticker','reaction_date']).any()
assert impact_df.realized_impact.notna().all() and np.isfinite(impact_df.realized_impact).all()
assert (impact_df.realized_impact>=0).all()
print('Target sanity checks PASSED')
print(impact_df.realized_impact.describe(percentiles=[.5,.75,.9,.95,.99]))

# %% Code cell 44
# Normalize every reaction date to one consistent representation.
impact_df["reaction_date"] = (
    pd.to_datetime(
        impact_df["reaction_date"],
        errors="coerce",
        utc=True,
    )
    .dt.tz_convert(None)
    .dt.normalize()
)

if impact_df["reaction_date"].isna().any():
    bad = int(impact_df["reaction_date"].isna().sum())
    raise RuntimeError(f"{bad} reaction dates became NaT before splitting.")

# Sort unique market dates.
unique_dates = pd.DatetimeIndex(
    sorted(impact_df["reaction_date"].unique())
)

n_dates = len(unique_dates)

if n_dates < 30:
    raise RuntimeError(
        f"Only {n_dates} unique reaction dates are available."
    )

i1 = max(1, int(np.floor(n_dates * TRAIN_FRACTION)))

i2 = max(
    i1 + 1,
    int(
        np.floor(
            n_dates * (TRAIN_FRACTION + VALIDATION_FRACTION)
        )
    )
)

if i2 >= n_dates:
    i2 = n_dates - 1

train_last_date = pd.Timestamp(unique_dates[i1 - 1])
validation_last_date = pd.Timestamp(unique_dates[i2 - 1])

# Vectorized split assignment.
impact_df["split"] = np.select(
    [
        impact_df["reaction_date"] <= train_last_date,
        impact_df["reaction_date"] <= validation_last_date,
    ],
    [
        "train",
        "validation",
    ],
    default="test",
)

train_df = impact_df[
    impact_df["split"] == "train"
].copy()

validation_df = impact_df[
    impact_df["split"] == "validation"
].copy()

test_df = impact_df[
    impact_df["split"] == "test"
].copy()

for name, df in [
    ("train", train_df),
    ("validation", validation_df),
    ("test", test_df),
]:
    print(
        f"{name:>10}: rows={len(df):>6} | "
        f"{df['reaction_date'].min()} "
        f"to {df['reaction_date'].max()} | "
        f"unique_dates={df['reaction_date'].nunique()}"
    )

assert len(train_df) > 0
assert len(validation_df) > 0
assert len(test_df) > 0

assert (
    train_df["reaction_date"].max()
    < validation_df["reaction_date"].min()
)

assert (
    validation_df["reaction_date"].max()
    < test_df["reaction_date"].min()
)

print("\nChronological split assertions: PASSED")
print("Train last date:", train_last_date.date())
print(
    "Validation last date:",
    validation_last_date.date()
)

# %% Code cell 46
qs=np.arange(.10,1.00,.10)
edges=np.quantile(train_df.realized_impact.to_numpy(float),qs)
def score10(v): return int(np.clip(np.searchsorted(edges,float(v),side='right')+1,1,10))
impact_df['impact_score_1_10']=impact_df.realized_impact.map(score10)
train_df=impact_df[impact_df.split=='train'].copy(); validation_df=impact_df[impact_df.split=='validation'].copy(); test_df=impact_df[impact_df.split=='test'].copy()
decile_payload={'calibration_source':'training split only','target':'realized_impact = max(abs(AR_1D), abs(CAR_3D))',
                'quantiles':{f'p{int(q*100)}':float(e) for q,e in zip(qs,edges)}}
with open(WORKDIR/'impact_decile_thresholds.json','w') as f: json.dump(decile_payload,f,indent=2)
print(json.dumps(decile_payload,indent=2))

# %% Code cell 48
for n,d in [('train',train_df),('validation',validation_df),('test',test_df)]:
    print('\n'+n.upper()); print(d.impact_score_1_10.value_counts().sort_index().to_string())

# %% Code cell 50
plt.figure(figsize=(8,4.5)); plt.hist(train_df.realized_impact,bins=50); plt.xlabel('realized_impact'); plt.ylabel('Training events'); plt.title('Training realized abnormal impact'); plt.tight_layout(); plt.savefig(WORKDIR/'impact_realized_impact_train_distribution.png',dpi=170,bbox_inches='tight'); plt.show()
for n,d in [('train',train_df),('validation',validation_df),('test',test_df)]:
    c=d.impact_score_1_10.value_counts().reindex(range(1,11),fill_value=0).sort_index(); plt.figure(figsize=(8,4.5)); c.plot(kind='bar'); plt.xlabel('Impact Score'); plt.ylabel('Events'); plt.title(f'Impact Score 1–10 — {n}'); plt.tight_layout(); plt.savefig(WORKDIR/f'impact_score_distribution_{n}.png',dpi=170,bbox_inches='tight'); plt.show()

# %% Code cell 52
ticker_summary=(impact_df.groupby('ticker').agg(sector=('sector','first'),events=('headline','size'),first_event=('reaction_date','min'),last_event=('reaction_date','max'),median_realized_impact=('realized_impact','median'),mean_realized_impact=('realized_impact','mean'),max_realized_impact=('realized_impact','max'),adjusted_price_used=('adjusted_price_used','first'),price_source=('price_source','first')).reset_index().sort_values('events',ascending=False))
ticker_summary.to_csv(WORKDIR/'impact_ticker_summary.csv',index=False); display(ticker_summary)

# %% Code cell 54
FORBIDDEN_FUTURE_FEATURES=['stock_return_1d','benchmark_return_1d','abnormal_return_1d','car_3d','realized_impact','impact_score_1_10','session_news_count']
print('\n'.join(' - '+x for x in FORBIDDEN_FUTURE_FEATURES))

# %% Code cell 56
assert not impact_df.duplicated(['ticker','reaction_date']).any()
assert train_df.reaction_date.max()<validation_df.reaction_date.min()<test_df.reaction_date.min()
def keys(d): return set(zip(d.ticker.astype(str),pd.to_datetime(d.reaction_date).astype(str)))
a,b,c=keys(train_df),keys(validation_df),keys(test_df)
assert a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c)
for n,d in [('train',train_df),('validation',validation_df),('test',test_df)]:
    assert len(d)>0 and np.isfinite(d.realized_impact).all() and (d.realized_impact>=0).all() and d.impact_score_1_10.between(1,10).all()
assert np.allclose(edges,np.quantile(train_df.realized_impact.to_numpy(float),qs))
print('ALL HARD ASSERTIONS PASSED.')

# %% Code cell 58
impact_df=impact_df.sort_values(['reaction_date','ticker']).reset_index(drop=True)
train_df=impact_df[impact_df.split=='train'].reset_index(drop=True); validation_df=impact_df[impact_df.split=='validation'].reset_index(drop=True); test_df=impact_df[impact_df.split=='test'].reset_index(drop=True)
paths={'all':WORKDIR/'impact_events_all.parquet','train':WORKDIR/'impact_train.parquet','validation':WORKDIR/'impact_validation.parquet','test':WORKDIR/'impact_test.parquet'}
impact_df.to_parquet(paths['all'],index=False); train_df.to_parquet(paths['train'],index=False); validation_df.to_parquet(paths['validation'],index=False); test_df.to_parquet(paths['test'],index=False)
train_df.to_csv(WORKDIR/'impact_train.csv',index=False); validation_df.to_csv(WORKDIR/'impact_validation.csv',index=False); test_df.to_csv(WORKDIR/'impact_test.csv',index=False)
for k,v in paths.items(): print(k,'->',v)

# %% Code cell 60
cols=['publication_time_utc','timestamp_precision','reaction_date','ticker','sector','headline','publisher','abnormal_return_1d','car_3d','realized_impact','impact_score_1_10','split']
preview=impact_df[cols].copy()
if len(preview)>500: preview=preview.iloc[np.linspace(0,len(preview)-1,500,dtype=int)]
preview_path=WORKDIR/'impact_event_target_preview.csv'; preview.to_csv(preview_path,index=False); display(preview.head(15))

# %% Code cell 62
def stats(s):
    s=pd.Series(s).dropna().astype(float)
    return {'count':int(len(s)),'mean':float(s.mean()),'std':float(s.std(ddof=1)) if len(s)>1 else 0.0,'min':float(s.min()),'p10':float(s.quantile(.1)),'p25':float(s.quantile(.25)),'median':float(s.median()),'p75':float(s.quantile(.75)),'p90':float(s.quantile(.9)),'p95':float(s.quantile(.95)),'p99':float(s.quantile(.99)),'max':float(s.max())}
report={
 'created_utc':datetime.now(timezone.utc).isoformat(),
 'stage':'Impact Engine Step 1 - historical target construction',
 'target_definition':{'benchmark':BENCHMARK,'abnormal_return_1d':'stock daily return - benchmark daily return on reaction session','car_3d':'sum abnormal returns over reaction session + next 2 trading sessions','realized_impact':'max(abs(abnormal_return_1d), abs(car_3d))','primary_future_training_target':'realized_impact','impact_score':'training-only realized_impact deciles -> 1..10'},
 'configuration':{'requested_start':str(DATE_START.date()),'requested_end':str(DATE_END.date()),'selected_tickers':SELECTED_TICKERS,'min_final_events_per_ticker':MIN_FINAL_EVENTS_PER_TICKER,'split_fractions':{'train':TRAIN_FRACTION,'validation':VALIDATION_FRACTION,'test':TEST_FRACTION}},
 'news':{'dataset_used':NEWS_DATASET_USED,'dataset_candidates':NEWS_DATASET_CANDIDATES,'load_errors':news_load_errors,'scan_counts':dict(scan),'exact_duplicates_removed':exact_duplicates_removed,'rows_before_ticker_session_collapse':int(before_session),'ticker_session_rows_removed':int(session_rows_removed),'timestamp_precision_counts':{str(k):int(v) for k,v in news_df.timestamp_precision.value_counts().to_dict().items()},'reaction_mapping_failures':reaction_mapping_failures},
 'prices':{'primary_repo':PRICE_REPO,'archive_file':PRICE_ARCHIVE_FILE,'archive_download_error':price_archive_error,'benchmark':BENCHMARK,'per_symbol':price_load_report,'missing_stock_price_tickers':missing_price},
 'target_join_report':target_join_report,
 'coverage':{'events_before_minimum_ticker_filter':{str(k):int(v) for k,v in pre_coverage.to_dict().items()},'eligible_tickers':eligible,'dropped_low_coverage_tickers':{str(k):int(v) for k,v in dropped.items()},'final_ticker_count':int(impact_df.ticker.nunique()),'final_event_count':int(len(impact_df)),'extreme_target_flags_gt_50pct':int(impact_df.extreme_target_flag.sum())},
 'realized_impact_statistics':{'all':stats(impact_df.realized_impact),'train':stats(train_df.realized_impact),'validation':stats(validation_df.realized_impact),'test':stats(test_df.realized_impact)},
 'splits':{n:{'rows':int(len(d)),'unique_dates':int(d.reaction_date.nunique()),'start':str(pd.Timestamp(d.reaction_date.min()).date()),'end':str(pd.Timestamp(d.reaction_date.max()).date()),'tickers':int(d.ticker.nunique())} for n,d in [('train',train_df),('validation',validation_df),('test',test_df)]},
 'impact_score_calibration':decile_payload,
 'impact_score_counts':{n:{str(k):int(v) for k,v in d.impact_score_1_10.value_counts().sort_index().to_dict().items()} for n,d in [('train',train_df),('validation',validation_df),('test',test_df)]},
 'forbidden_future_features':FORBIDDEN_FUTURE_FEATURES,
 'anti_leakage_rules':['obvious market-price-action headlines removed','one earliest eligible headline per ticker/reaction session','publication mapped to first market close after availability','00:00 UTC treated as date-only metadata','chronological split by unique reaction dates','impact deciles from train only','future returns are target/audit fields only','session_news_count is audit-only'],
 'outputs':{'impact_events_all':str(paths['all']),'impact_train':str(paths['train']),'impact_validation':str(paths['validation']),'impact_test':str(paths['test']),'decile_thresholds':str(WORKDIR/'impact_decile_thresholds.json'),'ticker_summary':str(WORKDIR/'impact_ticker_summary.csv'),'event_preview':str(preview_path)}
}
report_path=WORKDIR/'impact_dataset_report.json'
with open(report_path,'w') as f: json.dump(report,f,indent=2)
print('Saved:',report_path); print('Final events/tickers:',len(impact_df),impact_df.ticker.nunique())
