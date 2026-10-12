"""Leakage-safe daily-close index simulation.

Trades at weekly session closes. Only signals from strictly earlier publication
calendar dates can be used. Closing prices at the trade date set execution
prices; P&L begins AFTER the trade. No same-day signal/return leakage.
"""
from collections import defaultdict
from datetime import datetime, date, timezone, time
from pathlib import Path
import numpy as np
import pandas as pd
from .config import ModuleAConfig
from .schemas import Signal
from .strategy import rebalance
from .metrics import performance

class BacktestDataError(ValueError):pass


def read_prices(path:Path|str,cfg:ModuleAConfig)->pd.DataFrame:
    path=Path(path)
    if not path.is_file():raise BacktestDataError(f'Missing prices file: {path}')
    df=pd.read_csv(path)
    required={'date','ticker','adj_close'}
    if not required.issubset(df):raise BacktestDataError(f'Prices missing {sorted(required-set(df))}')
    df=df[['date','ticker','adj_close']].copy()
    df['date']=pd.to_datetime(df.date,errors='coerce',utc=True).dt.tz_convert(None).dt.normalize()
    df['ticker']=df.ticker.astype(str).str.strip().str.upper()
    df['adj_close']=pd.to_numeric(df.adj_close,errors='coerce')
    if df.isna().any().any() or (df.adj_close<=0).any():raise BacktestDataError('Invalid date/ticker/adjusted close')
    if df.duplicated(['date','ticker']).any():raise BacktestDataError('Duplicate ticker/date prices')
    df=df[df.ticker.isin(cfg.tickers)]
    if set(df.ticker)!=set(cfg.tickers):
        raise BacktestDataError(f'Price data missing index tickers: {sorted(set(cfg.tickers)-set(df.ticker))}')
    mat=df.pivot(index='date',columns='ticker',values='adj_close').sort_index()
    if mat.isna().any().any():raise BacktestDataError('Incomplete price grid; refusing silent forward-fill')
    mat=mat.loc[:,list(cfg.tickers)]
    if len(mat)<12:raise BacktestDataError('At least 12 market sessions are required')
    if (mat.index.to_series().diff().dt.days.dropna()>10).any():
        raise BacktestDataError('Price calendar gap over 10 days; check missing data')
    return mat


def read_signals(path:Path|str,cfg:ModuleAConfig)->list[Signal]:
    path=Path(path)
    if not path.is_file():raise BacktestDataError(f'Missing signal file: {path}')
    df=pd.read_csv(path)
    need={'ticker','published_at','headline','sentiment_score','impact_score','event_class','event_confidence','source'}
    if not need.issubset(df):raise BacktestDataError(f'Missing signal fields {sorted(need-set(df))}')
    # Hard-stop on files that include ground-truth future targets; prevents accidental
    # use of impact_score_1_10 as though it were a prediction.
    forbidden={'realized_impact','impact_score_1_10','car_3d','abnormal_return_1d',
               'benchmark_return_1d','stock_return_1d','session_news_count'}
    if forbidden.intersection(df.columns):
        raise BacktestDataError(f'Future target column(s) in signal file: {sorted(forbidden.intersection(df.columns))}')
    if len(df)>100000:raise BacktestDataError('Refusing excessive signal rows')
    signals=[]
    for record in df.to_dict(orient='records'):
        item={k:v for k,v in record.items() if k in Signal.model_fields}
        if pd.isna(item.get('event_id')):item.pop('event_id',None)
        signals.append(Signal.model_validate(item))
    bad=sorted({x.ticker for x in signals}-set(cfg.tickers))
    if bad:raise BacktestDataError(f'Unsupported signal ticker(s): {bad}')
    return signals


def weekly_sessions(index:pd.DatetimeIndex)->set[pd.Timestamp]:
    """Observed last trading session per completed ISO week.

    Don't pretend an incomplete final week is a scheduled rebalance if a
    historical slice happens to stop midweek. A Friday itself is sufficient.
    """
    output=set()
    for i,day in enumerate(index):
        iso=day.isocalendar()
        known_next_week=(i+1<len(index) and index[i+1].isocalendar()[:2]!=iso[:2])
        if known_next_week or day.weekday()==4:
            output.add(day)
    return output


def _simulate(cfg:ModuleAConfig,mat:pd.DataFrame,signals:list[Signal],mode:str)->dict:
    dates=list(mat.index)
    n=len(cfg.tickers)
    start=mat.iloc[0].to_numpy(dtype=float)
    shares=(1/n)/start
    nav=[1.0]
    turnovers=[];costs=[];trades=[]
    sched=weekly_sessions(mat.index)
    for i in range(1,len(dates)):
        prices=mat.iloc[i].to_numpy(dtype=float)
        value=float(shares@prices)
        if value<=0 or not np.isfinite(value):raise BacktestDataError('Invalid portfolio valuation')
        day=dates[i]
        if mode!='equal_weight_buyhold' and day in sched:
            current={t:float((shares[j]*prices[j])/value) for j,t in enumerate(cfg.tickers)}
            if mode=='equal_weight_weekly':
                target=cfg.baseline
            else:
                decision_at=datetime.combine(day.date(),time(23,59),tzinfo=timezone.utc)
                window=[s for s in signals if s.published_at.date()<day.date()
                        and (decision_at-s.published_at).total_seconds()/86400<=cfg.news_lookback_days]
                # Backtest decisions must honor the same batch caps as online.
                if len(window)>cfg.max_signals_per_request:
                    raise BacktestDataError(f'{day.date()}: {len(window)} items exceed configured signal cap')
                r=rebalance(cfg,window,decision_at,current,mode)
                target=r.target_weights
            target_vec=np.array([target[t] for t in cfg.tickers])
            # Transaction costs reduce investable equity. Solve the cost/target
            # circularity iteratively, so we do not create negative cash/margin.
            after_cost=value
            rate=cfg.transaction_cost_bps/10000
            for _ in range(50):
                new_shares=(target_vec*after_cost)/prices
                traded_dollars=float(np.abs((new_shares-shares)*prices).sum())
                revised=value-rate*traded_dollars
                if abs(revised-after_cost)<1e-12:break
                after_cost=revised
            if after_cost<=0:raise BacktestDataError('Transaction costs exhausted portfolio')
            new_shares=(target_vec*after_cost)/prices
            traded_dollars=float(np.abs((new_shares-shares)*prices).sum())
            charge=rate*traded_dollars
            # Recompute exact self-financing target using fixed-point iteration tolerance.
            if abs(value-charge-float(new_shares@prices))>1e-8:
                raise BacktestDataError('Trade simulation failed self-financing check')
            one_way=0.5*traded_dollars/value
            if one_way>cfg.one_way_turnover_limit+1e-5 and mode!='equal_weight_weekly':
                raise BacktestDataError('Actual turnover exceeded limit')
            if one_way>1e-8:
                trades.append({'date':str(day.date()),'one_way_turnover':one_way,
                               'cost_nav_units':charge,'strategy':mode})
            shares=new_shares
            turnovers.append(one_way)
            costs.append(charge)
            value=float(shares@prices)
        nav.append(value)
    summary=performance(nav,turnovers,costs,len(trades))
    return {'mode':mode,'metrics':summary,
            'daily_nav':[{'date':str(d.date()),'nav':float(100*v)} for d,v in zip(dates,nav)],
            'trades':trades}


def run_backtest(cfg:ModuleAConfig,prices:pd.DataFrame,signals:list[Signal],
                 start_date:date|None=None,end_date:date|None=None,selected:str='full_event_aware')->dict:
    if start_date:prices=prices.loc[prices.index.date>=start_date]
    if end_date:prices=prices.loc[prices.index.date<=end_date]
    if len(prices)<12:raise BacktestDataError('Requested range contains fewer than 12 trading sessions')
    # Evaluate all strategies over identical dates, universe, and transaction costs.
    modes=['equal_weight_buyhold','equal_weight_weekly','sentiment_only','sentiment_impact','full_event_aware']
    reports={m:_simulate(cfg,prices,signals,m) for m in modes}
    baseline=reports['equal_weight_weekly']['daily_nav']
    result={'status':'completed','ticker_universe':list(cfg.tickers),'start_date':str(prices.index[0].date()),
            'end_date':str(prices.index[-1].date()),'session_count':len(prices),
            'selected_strategy':selected,'evaluations':reports,
            'limitations':['Simulation only; model signal is not a validated direction-of-return forecast.',
                'All reported results are conditional on the provided historical data and cost assumptions.',
                'Same-day articles excluded; rebalancing at the weekly session close.',
                'Closing-price target sizing is idealized and cannot guarantee executable pre-close fills; no intraday price information is modeled.',
                'Data supplied to this backtest must contain out-of-sample PREDICTED impact, not ground truth.',
                'FNSPID historical news is one source; social data requires separate provenance.']}
    return result
