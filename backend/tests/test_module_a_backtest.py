from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from app.module_a.config import ModuleAConfig
from app.module_a.schemas import Signal
from app.module_a.backtester import read_prices,read_signals,run_backtest,BacktestDataError,weekly_sessions


def config():
    c=ModuleAConfig.load()
    return replace(c,tickers=('AAPL','MSFT','NVDA'),sector_map={'AAPL':'Technology','MSFT':'Technology','NVDA':'Technology'})


def synthetic_prices(tmp_path, cfg):
    dates=pd.bdate_range('2023-09-01',periods=60)
    rates={'AAPL':0.001,'MSFT':0.0002,'NVDA':-0.0008}
    rows=[dict(date=str(d.date()),ticker=t,adj_close=100*(1+rates[t])**i)
          for i,d in enumerate(dates) for t in cfg.tickers]
    path=tmp_path/'prices.csv';pd.DataFrame(rows).to_csv(path,index=False)
    return read_prices(path,cfg)


def signal(when='2023-09-14T12:00:00Z'):
    return Signal(ticker='NVDA',published_at=when,headline='Nvidia reports weak demand and rising operational losses',
                  sentiment_score=-.9,impact_score=8,event_class='Macroeconomic',event_confidence=.9)


def test_backtest_runs_five_strategies_and_true_returns(tmp_path):
    cfg=config();price=synthetic_prices(tmp_path,cfg)
    result=run_backtest(cfg,price,[signal()])
    assert set(result['evaluations'])=={'equal_weight_buyhold','equal_weight_weekly','sentiment_only','sentiment_impact','full_event_aware'}
    assert result['session_count']==60
    for run in result['evaluations'].values():
        assert len(run['daily_nav'])==60
        assert run['daily_nav'][0]['nav']==100
        assert np.isfinite(run['metrics']['total_return'])
    assert result['evaluations']['equal_weight_buyhold']['metrics']['trade_count']==0
    assert result['evaluations']['full_event_aware']['metrics']['total_transaction_costs_nav_units']>=0


def test_no_using_same_day_news_and_costs(tmp_path):
    cfg=config();prices=synthetic_prices(tmp_path,cfg)
    weekly=sorted(weekly_sessions(prices.index))
    # Inject news on a scheduled weekly close; it cannot be traded that day.
    day=weekly[3]
    same_day=signal(f'{day.date()}T12:00:00Z')
    end=weekly[3]
    partial=prices.loc[:end]
    result=run_backtest(cfg,partial,[same_day])
    assert result['evaluations']['full_event_aware']['metrics']['trade_count']==0
    # By contrast news on previous calendar day is eligible to change weights.
    prior=day-pd.Timedelta(days=1)
    prior_event=signal(f'{prior.date()}T12:00:00Z')
    result2=run_backtest(cfg,prices,[prior_event])
    assert result2['evaluations']['full_event_aware']['metrics']['trade_count']>0


def test_reject_realized_future_targets(tmp_path):
    cfg=config()
    df=pd.DataFrame([dict(ticker='NVDA',published_at='2023-09-14T12:00:00Z',headline='Major market change',
        sentiment_score=-.5,impact_score=8,event_class='Credit Event',event_confidence=.9,source='financial_news',
        realized_impact=.04)])
    path=tmp_path/'leaking.csv';df.to_csv(path,index=False)
    with pytest.raises(BacktestDataError,match='Future target'):read_signals(path,cfg)


def test_reject_incomplete_price_grid(tmp_path):
    cfg=config()
    rows=[dict(date='2023-09-01',ticker='AAPL',adj_close=100)]
    path=tmp_path/'incomplete.csv';pd.DataFrame(rows).to_csv(path,index=False)
    with pytest.raises(BacktestDataError):read_prices(path,cfg)
