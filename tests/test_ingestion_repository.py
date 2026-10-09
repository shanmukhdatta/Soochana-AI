from datetime import datetime,timezone
from app.ingestion.records import RawItem
from app.ingestion.repository import Repository

NOW=datetime(2026,10,9,16,0,tzinfo=timezone.utc)

def sample():
    return RawItem('financial_news','News','id','https://example.com/1',
                   'Nvidia launches new chip',NOW,language='en')


def test_repository_dedupe_atomic_claim_and_completed(tmp_path):
    repo=Repository(tmp_path/'new.db')
    assert repo.insert(sample(),'NVDA','company_alias')
    assert not repo.insert(sample(),'NVDA','company_alias')
    rows=repo.claim(5,600,3)
    assert len(rows)==1 and len(repo.claim(5,600,3))==0
    repo.success(rows[0]['id'],{'impact':{'score':7}})
    result=repo.one(rows[0]['id'])
    assert result['inference']['impact']['score']==7
    assert repo.summary()['status_counts'][0]['count']==1


def test_repository_failure_retry_and_dead(tmp_path):
    repo=Repository(tmp_path/'news.db')
    repo.insert(sample(),'NVDA','company_alias')
    row=repo.claim(1,600,2)[0]
    repo.failure(row['id'],'network',1,2,0)
    again=repo.claim(1,600,2)[0]
    repo.failure(again['id'],'network',2,2,0)
    assert repo.one(row['id'])['status']=='dead'
    assert repo.claim(2,600,2)==[]


def test_repo_two_tickers_same_item(tmp_path):
    repo=Repository(tmp_path/'events.db')
    repo.insert(sample(),'NVDA','company_alias')
    repo.insert(sample(),'MSFT','company_alias')
    assert repo.list()['total']==2
    assert repo.list(ticker='NVDA')['total']==1
