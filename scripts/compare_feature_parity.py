"""Optional regression check: compare live-computed market features to Step-2 historical rows.

Important: the best parity match uses PRICE_PROVIDER=csv with CSVs drawn from
FNSPID stock-price history (adjusted-close+volume) and a Step-2 parquet row.
Yfinance vs FNSPID differences should be measured, not concealed.

Example:
    PRICE_PROVIDER=csv PRICE_CSV_DIR=/path/to/price_csvs \
        python -m scripts.compare_feature_parity /path/to/impact_step2_test_features.parquet --max-rows 20
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import Settings
from app.services.prices import PriceProvider, MarketDataError
from app.services.features import FeatureBuilder


COMPARE_NUMERIC = [
    'stock_volatility_20d_ann','volume_ratio_20d','stock_return_5d',
    'stock_return_20d','spy_return_20d','spy_volatility_20d_ann',
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('parquet',type=Path,help='Frozen Step-2 train/validation/test features parquet')
    p.add_argument('--max-rows',type=int,default=20)
    args=p.parse_args()
    settings=Settings.from_env()
    provider=PriceProvider(settings.price_provider,settings.price_csv_dir)
    fb=FeatureBuilder(provider,settings.config_dir)
    df=pd.read_parquet(args.parquet)
    required = set(COMPARE_NUMERIC) | {
        'ticker','publication_time_utc','event_predicted_class',
        'event_confidence','sentiment_score','sector','market_regime',
        'stock_feature_date','spy_feature_date',
        'event_prob_geopolitical','event_prob_macroeconomic','event_prob_credit_event',
        'event_prob_merger_acquisition','event_prob_product_launch',
    }
    missing=sorted(required-set(df.columns))
    if missing:
        raise ValueError(f'Provided parquet lacks parity columns: {missing}')
    results=[]
    subset=df.head(args.max_rows)
    for i,row in subset.iterrows():
        event={'class_name':row['event_predicted_class'],
               'confidence':float(row['event_confidence']),
               'probability_vector':[float(row[k]) for k in [
                   'event_prob_geopolitical','event_prob_macroeconomic',
                   'event_prob_credit_event','event_prob_merger_acquisition','event_prob_product_launch'
               ]]}
        try:
            features,ctx,_=fb.build(
                str(row['ticker']), pd.Timestamp(row['publication_time_utc']).to_pydatetime(),
                event,float(row['sentiment_score'])
            )
        except (MarketDataError, ValueError) as e:
            results.append({'row':str(i),'ticker':str(row['ticker']),'status':f'unavailable: {e}'})
            continue
        delta={k:abs(float(row[k])-float(features[k])) for k in COMPARE_NUMERIC}
        date_match=(
            ctx['stock_feature_date']==pd.Timestamp(row['stock_feature_date']).date().isoformat()
            and ctx['spy_feature_date']==pd.Timestamp(row['spy_feature_date']).date().isoformat()
        )
        result={
            'row':str(i),'ticker':str(row['ticker']),'status':'compared',
            'as_of_date_match':bool(date_match),
            'regime_match':bool(features['market_regime']==row['market_regime']),
            'sector_match':bool(features['sector']==row['sector']),
            'max_abs_feature_delta':float(max(delta.values())),
            **{f'delta_{k}':float(v) for k,v in delta.items()},
        }
        results.append(result)
    table=pd.DataFrame(results)
    print(table.to_string(index=False,max_cols=12))
    if len(table)==0 or (table['status']=='compared').sum()==0:
        raise SystemExit('No rows could be compared: check the chosen price source/data coverage.')
    problems=table.loc[
        (table.status=='compared')
        & ((table.as_of_date_match==False) | (table.regime_match==False))
    ]
    if len(problems):
        print('\nWARNING: Feature parity differed. Check historical price-provider adjustment policies and dates.')
    else:
        print('\nDates/regimes match for all compared rows. Inspect numeric deltas before production sign-off.')


if __name__ == '__main__':
    main()
