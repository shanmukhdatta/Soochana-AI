from datetime import datetime,timezone
import pytest
from app.ingestion.ticker_matcher import TickerMatcher
from app.ingestion.records import RawItem,normalize,canonical_url

ALIASES={'AAPL':['Apple','Apple Inc'],'MSFT':['Microsoft'],'NVDA':['Nvidia'],'COST':['Costco']}
NOW=datetime(2026,10,9,16,0,tzinfo=timezone.utc)

def test_cashtag_and_alias_and_multiple_tickers():
    matcher=TickerMatcher(ALIASES)
    assert matcher.match('Microsoft and $NVDA announce a deal') == {'MSFT':'company_alias','NVDA':'cashtag'}
    assert matcher.match('cost increases unexpectedly') == {}
    assert matcher.match('Costco announces more stores') == {'COST':'company_alias'}


def test_provider_query_not_used_as_ticker_evidence():
    assert TickerMatcher(ALIASES).match('Investors see higher bond yields')=={}


def test_canonical_removes_only_trackers():
    assert canonical_url('https://EXAMPLE.org/item?id=10&utm_source=test&fbclid=123')=='https://example.org/item?id=10'
    with pytest.raises(ValueError):canonical_url('file:///tmp/secret')


def test_normalization_keeps_discovery_separate_from_actual_publication():
    from datetime import timedelta
    n=normalize(RawItem('financial_news','GDELT','id','https://example.org/news',
          ' Nvidia   launches new chip ',NOW,NOW+timedelta(hours=1),'gdelt_discovery','en'))
    assert n.text=='Nvidia launches new chip' and n.reported_at is None
    assert n.first_seen_at==NOW
