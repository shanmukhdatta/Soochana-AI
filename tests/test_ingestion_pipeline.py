from dataclasses import dataclass
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import pytest
from app.ingestion.records import RawItem
from app.ingestion.config import IngestionConfig
from app.ingestion.service import IngestionService
from app.ingestion.repository import Repository

NOW=datetime.now(timezone.utc)

class Adapter:
    def __init__(self,items=None,exc=None):self.items=items or [];self.exc=exc
    def collect(self,observed_at,limit):
        if self.exc:raise self.exc
        # use freshly observed timestamp rather than the static test NOW
        return [RawItem(*item[:5],first_seen_at=observed_at,language='en') for item in self.items][:limit]

class FakeResult:
    def model_dump(self,mode='json'):
        return {'sentiment':{'score':0.6,'label':'positive'},
                'event':{'class_name':'Product Launch','confidence':0.9},
                'impact':{'score':6,'predicted_realized_impact':0.017},
                'ticker':'NVDA','model_version':'test-only'}

class Engine:
    def __init__(self,fail=False):self.calls=[];self.fail=fail
    def analyze(self,request):
        self.calls.append(request)
        if self.fail:raise RuntimeError('market data unavailable')
        return FakeResult()


def config(tmp_path):
    return IngestionConfig({'enabled_sources':['gdelt','bluesky'],
      'max_rows_per_source_per_poll':20,'max_inference_per_poll':5,'lease_seconds':600,
      'max_attempts':3,'retry_base_seconds':0,'max_item_age_hours':48,
      'allow_unknown_language':False}, {'NVDA':['Nvidia'],'AAPL':['Apple']},tmp_path/'db.sqlite','secret')


def test_collect_process_and_dedupe_then_summary(tmp_path):
    cfg=config(tmp_path)
    a=Adapter([('financial_news','News','id1','https://news.example/x','Nvidia launches new GPU')])
    b=Adapter([('social_media','Bluesky','id2','https://bsky.app/x','$NVDA launches product')])
    svc=IngestionService(cfg,{'gdelt':a,'bluesky':b})
    report=svc.collect_once()
    assert report['total_inserted']==2
    assert svc.collect_once()['total_inserted']==0
    engine=Engine()
    outcome=svc.process_once(engine)
    assert outcome['completed']==2 and len(engine.calls)==2
    assert all(x.published_at.tzinfo is not None for x in engine.calls)
    assert svc.repository.list(status='completed')['total']==2
    assert len(svc.repository.summary()['source_counts'])==2


def test_source_failure_does_not_cancel_other_source(tmp_path):
    svc=IngestionService(config(tmp_path),{
       'gdelt':Adapter(exc=RuntimeError('HTTP 429')),
       'bluesky':Adapter([('social_media','Bluesky','abc','https://bsky.app/abc','$NVDA news')])})
    out=svc.collect_once()
    assert out['sources']['gdelt']['status']=='error'
    assert out['sources']['bluesky']['status']=='success'
    assert out['total_inserted']==1


def test_model_failure_stays_retryable_and_no_fabricated_result(tmp_path):
    svc=IngestionService(config(tmp_path),{'gdelt':Adapter([('financial_news','News','id1','https://x.test/1','Nvidia good news')]),'bluesky':Adapter()})
    svc.collect_once()
    outcome=svc.process_once(Engine(fail=True))
    assert outcome['retry']==1
    assert svc.repository.list(status='completed')['total']==0
    assert svc.repository.list(status='retry')['total']==1
    with pytest.raises(RuntimeError):svc.process_once(None)


def test_unknown_ticker_not_forwarded_to_model(tmp_path):
    svc=IngestionService(config(tmp_path),{'gdelt':Adapter([('financial_news','News','id1','https://x.test/1','No specific company')]),'bluesky':Adapter()})
    assert svc.collect_once()['total_inserted']==0
    assert svc.process_once(Engine())['claimed']==0
