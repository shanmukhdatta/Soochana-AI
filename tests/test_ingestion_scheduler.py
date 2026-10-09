from app.ingestion.scheduler import run_one_cycle
from types import SimpleNamespace


class Cfg:
    admin_token='test-token-that-is-long-enough-123'

class Service:
    config=Cfg()
    def collect_once(self):return {'total_inserted':2}


def test_scheduler_calls_local_api_and_reuses_existing_model(monkeypatch):
    seen={}
    class Resp:
        def raise_for_status(self):pass
        def json(self):return {'claimed':2,'completed':2}
    def fake_post(url,headers,timeout):
        seen.update(url=url,headers=headers,timeout=timeout)
        return Resp()
    monkeypatch.setattr('app.ingestion.scheduler.requests.post',fake_post)
    out=run_one_cycle(Service())
    assert out['inference']['completed']==2
    assert seen['url']=='http://127.0.0.1:8000/ingestion/process-once'
    assert seen['headers']['X-Ingestion-Token']==Cfg.admin_token


def test_scheduler_rejects_non_loopback_destinations():
    import pytest
    with pytest.raises(ValueError):run_one_cycle(Service(),'http://evil.example:9999')


def test_scheduler_without_secret_does_not_process():
    class NoSecret:
        admin_token=''
    svc=Service();svc.config=NoSecret()
    assert run_one_cycle(svc)['inference']=='not_dispatched'
