from datetime import date,datetime,timezone
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from app.config import ROOT
from app.services.prices import normalize_prices, MarketDataError
from app.services.features import FeatureBuilder, UnsupportedTicker


class FakePrices:
    provider='csv'
    def __init__(self, prices):
        self.prices=prices
        self.calls=[]
    def get(self, ticker, cutoff):
        self.calls.append((ticker,cutoff))
        return normalize_prices(self.prices[ticker],ticker,cutoff)


def sample_daily(start='2024-01-02', n=55, scale=1):
    return pd.DataFrame({
        'Date':pd.bdate_range(start,periods=n),
        'Adj Close':(100+np.arange(n)*.6+np.sin(np.arange(n)/7))*scale,
        'Volume':10000+np.arange(n)*10,
    })


def event():
    return {
        'class_name':'Product Launch', 'confidence':.6,
        'probability_vector':[.1,.1,.1,.1,.6],
    }


def test_strict_publication_date_and_feature_math():
    prices=FakePrices({'AAPL':sample_daily(),'SPY':sample_daily(scale=2)})
    fb=FeatureBuilder(prices,ROOT/'configs')
    published=datetime(2024,3,4,20,0,tzinfo=timezone.utc)
    features,context,warnings=fb.build('AAPL',published,event(),.4)
    cutoff=date(2024,3,4)
    assert prices.calls == [('AAPL',cutoff),('SPY',cutoff)]
    assert context['stock_feature_date'] < cutoff.isoformat()
    assert context['spy_feature_date'] < cutoff.isoformat()
    assert features['event_predicted_class']=='Product Launch'
    assert features['event_prob_product_launch']==.6
    assert features['sentiment_score']==.4
    assert features['abs_sentiment']==.4
    assert features['sector']=='Technology'
    assert features['market_regime'] in {
        'bearish_high_vol','bearish_normal_vol','neutral_high_vol','neutral_normal_vol',
        'bullish_high_vol','bullish_normal_vol',
    }
    expected = normalize_prices(sample_daily(),'AAPL',cutoff)
    assert abs(features['stock_return_5d']-(expected.adj_close.iloc[-1]/expected.adj_close.iloc[-6]-1))<1e-12
    assert len(warnings)>=1


def test_bars_on_same_date_excluded():
    raw=sample_daily(start='2024-02-01',n=32)
    cutoff=raw['Date'].iloc[-1].date()
    norm=normalize_prices(raw,'AAPL',cutoff)
    assert norm['date'].max().date()<cutoff
    assert len(norm)==31


def test_yfinance_multiindex_ticker_suffix():
    raw=sample_daily(start='2024-01-02',n=45).set_index('Date')
    raw.columns=pd.MultiIndex.from_tuples([(col,'AAPL') for col in raw.columns])
    norm=normalize_prices(raw,'AAPL',date(2024,3,5))
    assert len(norm)==45
    assert {'date','adj_close','volume'}.issubset(norm.columns)


def test_unknown_ticker_rejected():
    prices=FakePrices({'AAPL':sample_daily(),'SPY':sample_daily()})
    fb=FeatureBuilder(prices,ROOT/'configs')
    with pytest.raises(UnsupportedTicker):
        fb.build('FAKE',datetime(2024,3,4,tzinfo=timezone.utc),event(),.2)


def test_missing_adjusted_prices_not_silently_used():
    raw=sample_daily().rename(columns={'Adj Close':'Close'})
    with pytest.raises(MarketDataError,match='adjusted-close'):
        normalize_prices(raw,'AAPL',date(2024,4,1))


def test_old_prices_fail_instead_of_made_up_feature():
    with pytest.raises(MarketDataError,match='stale'):
        normalize_prices(sample_daily(start='2019-01-01',n=40),'AAPL',date(2024,4,1))
