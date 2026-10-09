"""Convert frozen Step-2 TEST parquet into leakage-safe Module-A PREDICTIONS.

CRITICAL: realized_impact and impact_score_1_10 in Step-2 are *future targets*.
Never copy them into module A. Predictions come from frozen fitted XGBoost.
Requires: pip install pyarrow; local trusted .joblib model artifacts.
"""
import argparse
import json
import hashlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',required=True,help='impact_step2_test_features.parquet')
    p.add_argument('--output',default='data/module_a/holdout_signals.csv')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    df=pd.read_parquet(args.features).copy()
    if 'split' not in df.columns or not df['split'].eq('test').all():
        raise ValueError('Refusing non-test rows: only the frozen Step-2 TEST partition is permitted')
    config=json.loads((root/'configs/impact_model_feature_manifest.json').read_text())
    names=config['model_features'];numeric=config['numeric_features'];categorical=config['categorical_features']
    required=set(names)|{'headline','ticker','publication_time_utc','timestamp_precision'}
    if not required.issubset(df):raise ValueError(f'Missing parquet columns: {sorted(required-set(df))}')
    if 'base_score' not in config:
        raise ValueError('Impact feature manifest requires corrected base_score from verified Antigravity fix')
    baseline=float(config['base_score'])
    if not 0<baseline<0.1:raise ValueError('Unexpected Impact model training base score')
    from app.module_a.config import ModuleAConfig
    cfg=ModuleAConfig.load()
    df=df.loc[df.ticker.isin(cfg.tickers)].copy()
    if df.empty:raise ValueError('No supported index ticker rows in test data')
    X=df[names].copy()
    for col in numeric:X[col]=pd.to_numeric(X[col],errors='coerce')
    for col in categorical:X[col]=X[col].astype(str)
    prep=joblib.load(root/'models/impact_engine/impact_preprocessor.joblib')
    model=joblib.load(root/'models/impact_engine/impact_model_final.joblib')
    model.get_booster().set_param({'base_score':baseline})
    pred=np.maximum(model.predict(prep.transform(X)),0.0)
    if not np.isfinite(pred).all() or not 0.001<np.median(pred)<0.1:
        raise ValueError('Predictions outside the expected v1 scale; verify serialization/base score')
    cuts=json.loads((root/'configs/impact_decile_thresholds.json').read_text())['quantiles']
    q=np.array([cuts[f'p{i}'] for i in range(10,100,10)],dtype=float)
    score=np.searchsorted(q,pred,side='right')+1
    pub=pd.to_datetime(df['publication_time_utc'],utc=True,errors='raise')
    # Distinct event IDs do not encode realized targets.
    ids=[hashlib.sha256(f'{t}|{d.isoformat()}|{str(h).strip().lower()}'.encode()).hexdigest()[:20]
         for t,d,h in zip(df.ticker,pub,df.headline)]
    out=pd.DataFrame({'event_id':ids,'ticker':df.ticker.to_numpy(),
        'published_at':pub.dt.strftime('%Y-%m-%dT%H:%M:%SZ').to_numpy(),
        'headline':df.headline.astype(str).to_numpy(),
        'sentiment_score':df.sentiment_score.to_numpy(),
        'impact_score':score,'event_class':df.event_predicted_class.astype(str).to_numpy(),
        'event_confidence':df.event_confidence.to_numpy(),'source':'financial_news'})
    from app.module_a.schemas import Signal
    for record in out.to_dict(orient='records'):Signal.model_validate(record)
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    out.sort_values('published_at').to_csv(target,index=False)
    print(f'Generated {len(out)} test-set PREDICTED events in {target}')
    print(f'Impact prediction mean={np.mean(pred):.6f}, median={np.median(pred):.6f}, range=[{min(pred):.6f},{max(pred):.6f}]')
    print('No realized_impact or impact_score_1_10 ground-truth columns exported.')

if __name__=='__main__':main()
