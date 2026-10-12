import os
from datetime import datetime,timezone
from fastapi.testclient import TestClient
from app.main import app
from app.schemas import AnalyzeResponse


class FakeRiskEngine:
    def analyze(self, req):
        return AnalyzeResponse(
            ticker=req.ticker,
            published_at=req.published_at,
            source=req.source,
            sentiment={'score':-.4,'label':'negative'},
            event={'class_name':'Credit Event','confidence':.81,
                   'probabilities':{'Geopolitical':.02,'Macroeconomic':.05,'Credit Event':.81,
                                    'Merger / Acquisition':.04,'Product Launch':.08}},
            impact={'predicted_realized_impact':.038,'score':8},
            market_context={'sector':'Technology','market_regime':'neutral_normal_vol',
                            'stock_feature_date':'2026-10-06','spy_feature_date':'2026-10-06',
                            'provider':'csv'},
        )


def sample():
    return {'text':'Company misses debt obligations for the quarter.',
            'ticker':'aapl','published_at':'2026-10-07T14:30:00Z','source':'financial_news'}


def test_health_not_ready_with_no_weights(monkeypatch):
    monkeypatch.setenv('SKIP_MODEL_LOAD','1')
    with TestClient(app) as client:
        response=client.get('/health')
        assert response.status_code==503
        assert 'not_ready' in str(response.json())


def test_analyze_and_batch_contract(monkeypatch):
    monkeypatch.setenv('SKIP_MODEL_LOAD','1')
    with TestClient(app) as client:
        app.state.pipeline=FakeRiskEngine()
        assert client.get('/health').json()['status']=='ready'
        r=client.post('/analyze',json=sample())
        assert r.status_code==200,r.text
        j=r.json()
        assert j['ticker']=='AAPL'
        assert j['sentiment']['score']==-.4
        assert j['event']['class_name']=='Credit Event'
        assert j['impact']['score']==8
        assert j['market_context']['stock_feature_date']=='2026-10-06'
        b=client.post('/analyze/batch',json={'items':[sample(),sample()]})
        assert b.status_code==200,b.text
        assert b.json()['total']==2
        assert len(b.json()['results'])==2


def test_timestamp_timezone_required(monkeypatch):
    monkeypatch.setenv('SKIP_MODEL_LOAD','1')
    with TestClient(app) as client:
        obj=sample()
        obj['published_at']='2026-10-07T14:30:00'
        assert client.post('/analyze',json=obj).status_code==422


def test_unavailable_model_is_503(monkeypatch):
    monkeypatch.setenv('SKIP_MODEL_LOAD','1')
    with TestClient(app) as client:
        assert client.post('/analyze',json=sample()).status_code==503


def test_observability_endpoints_and_headers():
    with TestClient(app) as client:
        res = client.get('/health/live')
        assert res.status_code == 200
        assert res.json()['status'] == 'healthy'
        assert 'X-Request-ID' in res.headers
        assert 'X-Process-Time' in res.headers

        m = client.get('/metrics')
        assert m.status_code == 200
        data = m.json()
        assert data['service'] == 'soochana-backend'
        assert 'uptime_seconds' in data
        assert data['total_requests_served'] >= 1
