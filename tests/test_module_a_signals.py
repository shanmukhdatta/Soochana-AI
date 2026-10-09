from datetime import datetime,timezone,timedelta
from dataclasses import replace
from app.module_a.config import ModuleAConfig
from app.module_a.schemas import Signal
from app.module_a.signals import aggregate


def small():
    c=ModuleAConfig.load()
    return replace(c,tickers=('AAPL','MSFT','NVDA'),sector_map={t:'Technology' for t in ('AAPL','MSFT','NVDA')})


def item(**kw):
    fields=dict(ticker='NVDA',published_at='2024-05-15T12:00:00Z',source='financial_news',
                headline='Nvidia reports weaker quarterly demand outlook',
                sentiment_score=-0.8,impact_score=8,event_class='Macroeconomic',event_confidence=0.9)
    fields.update(kw)
    return Signal(**fields)


def test_deduplicate_cross_source_reposts():
    cfg=small()
    first=item()
    repost=item(source='social_media',headline='Nvidia reports weaker quarterly demand outlook!')
    values,counts,warnings=aggregate([first,repost],datetime(2024,5,16,tzinfo=timezone.utc),cfg)
    assert counts['NVDA']==1
    assert values['NVDA']<0
    assert any('Duplicate' in w for w in warnings)


def test_same_calendar_day_and_future_excluded():
    cfg=small()
    signals=[item(published_at='2024-05-16T10:00:00Z'),item(published_at='2024-05-17T00:00:00Z')]
    values,counts,warnings=aggregate(signals,datetime(2024,5,16,23,59,tzinfo=timezone.utc),cfg)
    assert counts['NVDA']==0 and values['NVDA']==0
    assert '2 signal(s) excluded' in warnings[0]


def test_validation_rejects_outside_bounds():
    import pytest
    with pytest.raises(ValueError):item(sentiment_score=2)
    with pytest.raises(ValueError):item(published_at='2024-05-15T12:00:00')


def test_full_event_aware_source_recency_and_confidence_do_not_cancel():
    from app.module_a.config import ModuleAConfig
    from app.module_a.schemas import Signal
    from app.module_a.signals import aggregate
    from datetime import datetime, timezone
    cfg = ModuleAConfig.load()
    as_of = datetime(2024,5,17,23,tzinfo=timezone.utc)
    def strength(source='financial_news', when='2024-05-16T12:00:00Z', confidence=.95):
        s = Signal(ticker='NVDA', published_at=when, source=source,
            headline='Nvidia reports a major unexpected product recall',
            sentiment_score=-.8, impact_score=8, event_class='Product Launch',
            event_confidence=confidence)
        return abs(aggregate([s],as_of,cfg,'full_event_aware')[0]['NVDA'])
    assert strength() > strength(source='social_media')
    assert strength() > strength(when='2024-05-12T12:00:00Z')
    assert strength() > strength(confidence=.1)

def test_ablation_sentiment_only_remains_normalized_without_prior():
    from app.module_a.config import ModuleAConfig
    from app.module_a.schemas import Signal
    from app.module_a.signals import aggregate
    from datetime import datetime, timezone
    c=ModuleAConfig.load();as_of=datetime(2024,5,17,23,tzinfo=timezone.utc)
    sig=Signal(ticker='NVDA',published_at='2024-05-16T12:00:00Z',
        headline='Nvidia reports a major unexpected product recall',
        sentiment_score=-.8,impact_score=8,event_class='Product Launch',event_confidence=.1)
    assert abs(aggregate([sig],as_of,c,'sentiment_only')[0]['NVDA']+.8)<1e-9
