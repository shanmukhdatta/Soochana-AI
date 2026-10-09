from datetime import datetime,timedelta,timezone
import json
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.ingestion.records import RawItem
from app.ingestion.router import get_service


def fake_prediction(ticker='AAPL'):
    return {
      'ticker':ticker,'published_at':'2026-10-08T15:00:00Z','source':'financial_news',
      'sentiment':{'score':0.62,'label':'positive'},
      'event':{'class_name':'Product Launch','confidence':0.89,
        'probabilities':{'Geopolitical':0.01,'Macroeconomic':0.02,'Credit Event':0.02,
                         'Merger / Acquisition':0.06,'Product Launch':0.89}},
      'impact':{'score':7,'predicted_realized_impact':0.023},
      'market_context':{'sector':'Technology','market_regime':'neutral_normal_vol',
                        'stock_feature_date':'2026-10-07','spy_feature_date':'2026-10-07','provider':'synthetic-test-only'},
      'warnings':['test fixture'], 'model_version':'risk-engine-v1'
    }


@pytest.fixture
def client(monkeypatch,tmp_path):
    monkeypatch.setenv('SKIP_MODEL_LOAD','1')
    monkeypatch.setenv('INGESTION_DB_PATH',str(tmp_path/'injection_test.db'))
    monkeypatch.delenv('INGESTION_ADMIN_TOKEN',raising=False)
    get_service.cache_clear()
    with TestClient(app) as test:
        yield test
    get_service.cache_clear()


def test_health_and_protected_mutation_when_token_unset(client):
    r=client.get('/ingestion/health')
    assert r.status_code==200 and r.json()['live_provider_connectivity']=='not_tested_by_this_endpoint'
    assert client.post('/ingestion/collect-once').status_code==503
    assert client.post('/ingestion/process-once').status_code==503
    assert client.get('/signals/summary').status_code==200
    assert client.get('/health').status_code==503  # SKIP_MODEL_LOAD is explicit


def test_token_required_and_unloaded_engine_preserves_queue(client,monkeypatch):
    monkeypatch.setenv('INGESTION_ADMIN_TOKEN','set-before-loading-service')
    get_service.cache_clear()
    assert client.post('/ingestion/process-once',headers={'X-Ingestion-Token':'wrong'}).status_code==403
    assert client.post('/ingestion/process-once',headers={'X-Ingestion-Token':'set-before-loading-service'}).status_code==503


def test_saved_signals_feed_module_a_and_module_b_suggestions(client):
    repo=get_service().repository
    now=datetime.now(timezone.utc)-timedelta(hours=10)
    item=RawItem('financial_news','Source','article-id-1','https://example.com/news',
                 'Apple launches next-generation processors',now,language='en')
    assert repo.insert(item,'AAPL','company_alias')
    row=repo.claim(1,600,3)[0]
    repo.success(row['id'],fake_prediction('AAPL'))
    module_a=client.post('/ingestion/module-a/recommend')
    assert module_a.status_code==200, module_a.text
    assert module_a.json()['saved_signals_considered']==1
    assert abs(sum(module_a.json()['recommendation']['target_weights'].values())-1)<1e-5
    module_b=client.get(f"/signals/{row['id']}/module-b-suggestion")
    assert module_b.status_code==200, module_b.text
    assert module_b.json()['executed'] is False
    assert module_b.json()['requires_manual_review'] is True
    assert module_b.json()['proposal']['suggested_scenario']['name']
    assert client.get('/signals').json()['total']==1
    assert client.get(f"/signals/{row['id']}").json()['inference']['impact']['score']==7


def test_bogus_filters_and_unfinished_signal(client):
    assert client.get('/signals?limit=99999').status_code==422
    assert client.get('/signals?source_type=bogus').status_code==422
    assert client.get('/signals/not-a-valid-id').status_code==422
    assert client.get('/signals/'+'0'*64).status_code==404
