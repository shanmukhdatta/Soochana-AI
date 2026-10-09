from dataclasses import replace
from datetime import datetime,timezone
import numpy as np
import pytest
from app.module_a.config import ModuleAConfig
from app.module_a.constraints import project,InfeasiblePortfolio,validate_weights
from app.module_a.strategy import rebalance
from app.module_a.schemas import Signal


def config():return ModuleAConfig.load()


def test_fifteen_stock_sector_position_turnover_bounds():
    cfg=config();n=len(cfg.tickers)
    current=np.full(n,1/n)
    desired=np.array([0.15 if i<4 else 0.01 for i in range(n)])
    weights=project(desired,current,cfg)
    assert abs(weights.sum()-1)<1e-7
    assert np.max(abs(weights-current))<=cfg.position_deviation_limit+1e-6
    assert .5*sum(abs(weights-current))<=cfg.one_way_turnover_limit+1e-6
    for s,b in cfg.sector_targets.items():
        assert abs(sum(weights[i] for i,t in enumerate(cfg.tickers) if cfg.sector_map[t]==s)-b) <=cfg.sector_deviation_limit+1e-6


def test_negative_signal_reduces_weight():
    cfg=config()
    sig=Signal(ticker='NVDA',published_at='2024-05-15T10:00:00Z',headline='Nvidia reports major loss after a serious disruption',
         sentiment_score=-.95,impact_score=9,event_class='Credit Event',event_confidence=.9)
    r=rebalance(cfg,[sig],datetime(2024,5,16,23,0,tzinfo=timezone.utc))
    assert r.target_weights['NVDA']<cfg.baseline['NVDA']
    assert sum(r.target_weights.values())==pytest.approx(1,abs=1e-7)
    assert r.one_way_turnover<=cfg.one_way_turnover_limit+1e-6


def test_no_news_holds_current():
    cfg=config()
    r=rebalance(cfg,[],datetime(2024,5,16,23,0,tzinfo=timezone.utc))
    assert r.one_way_turnover==pytest.approx(0,abs=1e-8)
    assert r.target_weights==pytest.approx(cfg.baseline)


def test_bad_current_weights_and_infeasible_turnover():
    cfg=config()
    with pytest.raises(ValueError):validate_weights({'NVDA':1.0},cfg)
    current=np.zeros(len(cfg.tickers));current[0]=1
    with pytest.raises(InfeasiblePortfolio):project(np.full(len(cfg.tickers),1/len(cfg.tickers)),current,cfg)
