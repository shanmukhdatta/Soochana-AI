"""Download adjusted-close historical prices into one long-format Module A CSV.

Research/demo utility; dependent on yfinance availability and terms. Compare
against a trusted licensed data source before drawing investment conclusions.
"""
import argparse
from datetime import date,timedelta
from pathlib import Path
import pandas as pd
from app.module_a.config import ModuleAConfig
from app.services.prices import normalize_prices


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--start',default='2022-08-01')
    ap.add_argument('--end',default='2024-01-01',help='exclusive date')
    ap.add_argument('--dataset-id',default='holdout')
    args=ap.parse_args()
    if not args.dataset_id.replace('_','').replace('-','').isalnum():raise ValueError('Invalid dataset ID')
    start=date.fromisoformat(args.start);end=date.fromisoformat(args.end)
    if start>=end:raise ValueError('Invalid date range')
    import yfinance as yf
    cfg=ModuleAConfig.load()
    parts=[]
    for ticker in cfg.tickers:
        print('Fetching',ticker,flush=True)
        raw=yf.download(ticker,start=start.isoformat(),end=end.isoformat(),auto_adjust=False,
                        progress=False,actions=False,threads=False)
        normalized=normalize_prices(raw,ticker,end)
        df=normalized.loc[(normalized.date>=pd.Timestamp(start)) & (normalized.date<pd.Timestamp(end)),
                          ['date','adj_close']].copy()
        df['ticker']=ticker
        parts.append(df)
    result=pd.concat(parts,ignore_index=True)
    root=Path(__file__).resolve().parents[1]
    out=root/'data'/'module_a'/f'{args.dataset_id}_prices.csv'
    out.parent.mkdir(parents=True,exist_ok=True)
    result[['date','ticker','adj_close']].to_csv(out,index=False)
    print('Saved',out,'rows=',len(result))
    from app.module_a.backtester import read_prices
    read_prices(out,cfg)
    print('Verified complete adjusted-close price grid')

if __name__=='__main__':main()
