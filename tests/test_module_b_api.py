from contextlib import asynccontextmanager
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.module_b.router import router
from app.module_b.scenarios import template
from app.module_b.schemas import Shocks, Scenario
from app.schemas import AnalyzeResponse


class FakePipeline:
    def analyze(self, req):
        return AnalyzeResponse(ticker=req.ticker,published_at=req.published_at,source=req.source,
            sentiment={'score':-.4,'label':'negative'},
            event={'class_name':'Credit Event','confidence':.92,'probabilities':{'Credit Event':.92,'Geopolitical':.02,'Macroeconomic':.02,'Merger / Acquisition':.02,'Product Launch':.02}},
            impact={'predicted_realized_impact':.025,'score':8},
            market_context={'sector':'Technology','market_regime':'neutral_normal_vol','stock_feature_date':'2024-05-14','spy_feature_date':'2024-05-14','provider':'fixture'})


def make_client(real=False):
    @asynccontextmanager
    async def lifespan(app):
        app.state.pipeline=FakePipeline() if real else None
        yield
    app=FastAPI(lifespan=lifespan)
    app.include_router(router)
    return TestClient(app)

def news_payload(**kwargs):
    return {'news':{'ticker':'AAPL','text':'Apple faces a major credit downgrade','published_at':'2024-05-15T14:00:00Z','source':'financial_news'},**kwargs}

def test_get_portfolio_and_scenarios():
    with make_client() as client:
        r=client.get('/module-b/portfolio');assert r.status_code==200
        assert r.json()['summary']['position_count']==30
        s=client.get('/module-b/scenarios');assert s.status_code==200
        assert len(s.json()['scenario_library'])==5

def test_run_stress_valid():
    with make_client() as c:
        r=c.post('/module-b/stress',json={'scenario':template('Credit Event','moderate','AAPL').model_dump()})
        assert r.status_code==200,r.text
        assert len(r.json()['positions'])==30
        assert r.json()['totals']['incremental_12m_expected_credit_loss_usd']>=0

def test_stress_zero_shocks():
    with make_client() as c:
        r=c.post('/module-b/stress',json={'scenario':Scenario(name='Zero',shocks=Shocks()).model_dump()})
        assert r.status_code==200
        assert abs(r.json()['totals']['illustrative_mixed_horizon_net_effect_usd'])<1e-7

def test_news_without_pipeline_returns_503():
    with make_client() as c:
        assert c.post('/module-b/stress-from-news',json=news_payload()).status_code==503

def test_news_suggest_only_and_approval_gate():
    with make_client(real=True) as c:
        r=c.post('/module-b/stress-from-news',json=news_payload());assert r.status_code==200,r.text
        j=r.json();assert j['executed'] is False and 'stress_result' not in j
        assert 'AAPL' in j['suggested_scenario']['issuer_shocks']
        r2=c.post('/module-b/stress-from-news',json=news_payload(run_stress=True));assert r2.status_code==200,r2.text
        assert len(r2.json()['stress_result']['positions'])==30
        assert r2.json()['suggested_scenario']['severity']=='severe'

def test_news_bad_time_returns_422():
    x=news_payload();x['news']['published_at']='2024-05-15T10:00:00'
    with make_client(real=True) as c:
        assert c.post('/module-b/stress-from-news',json=x).status_code==422

def test_invalid_shocks_returns_422():
    with make_client() as c:
        r=c.post('/module-b/stress',json={'scenario':{'name':'Bad','shocks':{'equity_pct':-120}}})
        assert r.status_code==422

def test_what_if_and_reverse():
    s=Scenario(name='Credit',shocks=Shocks(pd_multiplier=2)).model_dump()
    with make_client() as c:
        a=c.post('/module-b/what-if',json={'scenario':s,'changes':[{'position_id':'L01','scale':0.5}]})
        assert a.status_code==200,a.text
        assert a.json()['before']['portfolio']['position_count']==30
        b=c.post('/module-b/reverse-stress',json={'scenario':s,'loss_budget_usd':1000,'steps':11,'max_multiplier':2})
        assert b.status_code==200,b.text
        assert len(b.json()['trace'])==11

def test_bad_override_rejected():
    with make_client() as c:
        a=c.post('/module-b/stress',json={'scenario':{'name':'Bad combo', 'shocks':{'rate_bps':2000},'sector_shocks':{'Technology':{'rate_bps':1000}}}})
        assert a.status_code==422
