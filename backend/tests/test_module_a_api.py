from datetime import datetime,timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.module_a.router import router

app=FastAPI()
app.include_router(router)


def test_index_endpoint_and_weights():
    c=TestClient(app)
    r=c.get('/module-a/index')
    assert r.status_code==200
    payload=r.json()
    assert len(payload['tickers'])==15
    assert abs(sum(payload['baseline_weights'].values())-1)<1e-10


def test_stateless_rebalance_endpoint():
    c=TestClient(app)
    data={'as_of':'2024-05-16T23:00:00Z','signals':[{'ticker':'NVDA','published_at':'2024-05-15T12:00:00Z',
        'source':'financial_news','headline':'Nvidia announced an unexpected product recall',
        'sentiment_score':-.6,'impact_score':8,'event_class':'Product Launch','event_confidence':.95}]}
    r=c.post('/module-a/rebalance',json=data)
    assert r.status_code==200,r.text
    p=r.json()
    assert abs(sum(p['target_weights'].values())-1)<1e-8
    assert p['target_weights']['NVDA']<1/15


def test_bad_date_ticker_and_missing_models():
    c=TestClient(app)
    assert c.post('/module-a/rebalance',json={'as_of':'2024-05-16','signals':[]}).status_code==422
    assert c.post('/module-a/rebalance',json={'as_of':'2024-05-16T23:00:00Z',
        'signals':[{'ticker':'INVALID','published_at':'2024-05-15T12:00:00Z','headline':'Negative company news now',
                    'sentiment_score':-.6,'impact_score':6,'event_class':'Credit Event','event_confidence':.7}]}).status_code==422
    payload={'as_of':'2024-05-16T23:00:00Z','articles':[{'ticker':'AAPL','text':'Apple announced a new device',
                                                             'published_at':'2024-05-15T12:00:00Z'}]}
    assert c.post('/module-a/rebalance-from-news',json=payload).status_code==503


def test_safe_backtest_request_paths():
    c=TestClient(app)
    assert c.post('/module-a/backtest',json={'dataset_id':'../secret'}).status_code==422
    assert c.post('/module-a/backtest',json={'dataset_id':'nonexistent'}).status_code==422
    assert c.get('/module-a/backtest/not-a-uuid').status_code==422


def test_end_to_end_route_with_stubbed_risk_engine():
    """Integration wiring test; not a claim to have run the real model weights."""
    from types import SimpleNamespace
    class FakeRiskEngine:
        def analyze(self, req):
            return SimpleNamespace(ticker=req.ticker,published_at=req.published_at,source=req.source,
                sentiment=SimpleNamespace(score=-.7),
                impact=SimpleNamespace(score=8),
                event=SimpleNamespace(class_name='Product Launch', confidence=.9))
    app.state.pipeline=FakeRiskEngine()
    try:
        c=TestClient(app)
        response=c.post('/module-a/rebalance-from-news',json={
            'as_of':'2024-05-16T23:00:00Z',
            'articles':[{'ticker':'NVDA','published_at':'2024-05-15T14:00:00Z',
                'text':'Nvidia reports major product recall', 'source':'financial_news'}]})
        assert response.status_code==200,response.text
        data=response.json()
        assert data['target_weights']['NVDA']<1/15
        assert data['signal_counts']['NVDA']==1
    finally:
        del app.state.pipeline


def test_backtest_api_saves_and_reads_results(tmp_path,monkeypatch):
    import importlib
    import pandas as pd
    module=importlib.import_module('app.module_a.router')
    monkeypatch.setattr(module,'DATA_ROOT',tmp_path)
    cfg=module.cfg()
    dates=pd.bdate_range('2023-09-01',periods=20)
    prices=[{'date':str(d.date()),'ticker':t,'adj_close':100*(1+(i%5)*0.0001)**j}
        for j,d in enumerate(dates) for i,t in enumerate(cfg.tickers)]
    pd.DataFrame(prices).to_csv(tmp_path/'toy_prices.csv',index=False)
    headers=['event_id','ticker','published_at','headline','sentiment_score',
             'impact_score','event_class','event_confidence','source']
    pd.DataFrame(columns=headers).to_csv(tmp_path/'toy_signals.csv',index=False)
    c=TestClient(app)
    response=c.post('/module-a/backtest',json={'dataset_id':'toy'})
    assert response.status_code==200,response.text
    report=response.json()
    assert set(report['metrics'])=={'equal_weight_buyhold','equal_weight_weekly',
                                    'sentiment_only','sentiment_impact','full_event_aware'}
    retrieved=c.get(report['result_url'])
    assert retrieved.status_code==200
    assert retrieved.json()['session_count']==20
