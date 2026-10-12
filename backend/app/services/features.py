"""Parity with the exact Step-2 rolling market predictors and regime cuts.

Critical: cutoff is the UTC publication calendar date, and only the latest
completed market date STRICTLY BEFORE it is usable. Never join to reaction_date.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.prices import PriceProvider, MarketDataError


EVENT_PROB_FEATURES = (
    'event_prob_geopolitical',
    'event_prob_macroeconomic',
    'event_prob_credit_event',
    'event_prob_merger_acquisition',
    'event_prob_product_launch',
)


class UnsupportedTicker(ValueError):
    pass


class FeatureBuilder:
    def __init__(self, prices: PriceProvider, config_dir: Path):
        self.prices = prices
        config_dir = Path(config_dir)
        def read(name):
            with (config_dir/name).open('r', encoding='utf-8') as f:
                return json.load(f)
        self.sectors: dict[str, str] = read('sector_mapping.json')
        self.supported: set[str] = set(read('supported_tickers.json'))
        self.thresholds: dict[str, float] = read('impact_step2_market_regime_thresholds.json')
        if not self.supported.issubset(self.sectors):
            raise ValueError('Supported ticker has no sector mapping')

    @staticmethod
    def _stock_features(df: pd.DataFrame) -> dict:
        # Mirrors Step 2 exactly: pct_change, 20-day std(ddof=1)*sqrt(252),
        # volume divided by its 20-session rolling mean (including last day),
        # and trailing 5/20-session returns.
        close = df['adj_close']
        volume = df['volume']
        daily_returns = close.pct_change()
        vol20 = daily_returns.rolling(20, min_periods=20).std().iloc[-1] * np.sqrt(252.0)
        mean_vol20 = volume.rolling(20, min_periods=20).mean().iloc[-1]
        volume_ratio = float(volume.iloc[-1] / mean_vol20) if pd.notna(mean_vol20) and mean_vol20 > 0 else np.nan
        return {
            'stock_volatility_20d_ann': float(vol20),
            'volume_ratio_20d': volume_ratio,
            'stock_return_5d': float(close.pct_change(5).iloc[-1]),
            'stock_return_20d': float(close.pct_change(20).iloc[-1]),
        }

    @staticmethod
    def _spy_features(df: pd.DataFrame) -> dict:
        close = df['adj_close']
        sr = close.pct_change()
        return {
            'spy_return_20d': float(close.pct_change(20).iloc[-1]),
            'spy_volatility_20d_ann': float(sr.rolling(20,min_periods=20).std().iloc[-1] * np.sqrt(252.0)),
        }

    def _regime(self, ret: float, vol: float) -> str:
        t = self.thresholds
        if pd.isna(ret):
            direction = 'unknown'
        elif ret < t['spy_return_20d_q33']:
            direction = 'bearish'
        elif ret > t['spy_return_20d_q67']:
            direction = 'bullish'
        else:
            direction = 'neutral'
        if pd.isna(vol):
            volatility = 'unknown'
        else:
            volatility = 'high_vol' if vol > t['spy_volatility_20d_ann_q75'] else 'normal_vol'
        return f'{direction}_{volatility}'

    def build(self, ticker: str, published_at: datetime, event: dict,
              sentiment: float) -> tuple[dict, dict, list[str]]:
        if ticker not in self.supported:
            raise UnsupportedTicker(
                f'Ticker {ticker!r} is outside the {len(self.supported)} stocks in the training universe; '
                'unsupported inference would be out-of-distribution'
            )
        if published_at.tzinfo is None or published_at.utcoffset() is None:
            raise ValueError('Timezone-aware published_at required')
        # Matching FNSPID notebook: normalize timestamp to UTC publication date.
        cutoff = published_at.astimezone(timezone.utc).date()
        stock_prices = self.prices.get(ticker, cutoff)
        spy_prices = self.prices.get('SPY', cutoff)
        if not (stock_prices['date'].iloc[-1].date() < cutoff and spy_prices['date'].iloc[-1].date() < cutoff):
            raise MarketDataError('Temporal leak: same-day/future price entered predictors')
        stock = self._stock_features(stock_prices)
        spy = self._spy_features(spy_prices)
        assert len(event['probability_vector']) == 5
        features: dict = {
            'sentiment_score': float(sentiment),
            'abs_sentiment': abs(float(sentiment)),
            **dict(zip(EVENT_PROB_FEATURES, event['probability_vector'])),
            'event_confidence': float(event['confidence']),
            **stock,
            **spy,
            'event_predicted_class': event['class_name'],
            'sector': self.sectors[ticker],
            'market_regime': self._regime(spy['spy_return_20d'],spy['spy_volatility_20d_ann']),
        }
        # Reject infinite values but allow missing numeric values (tree-based training did).
        for key, v in features.items():
            if isinstance(v, (float,np.floating)) and np.isinf(v):
                features[key] = np.nan
        warnings = [
            'Impact Score is an estimated historical market-reaction severity, not a guaranteed price prediction.',
            'Daily adjusted-price provider may differ from the FNSPID training data; verify parity before production use.',
        ]
        context = {
            'sector': features['sector'],
            'market_regime': features['market_regime'],
            'stock_feature_date': stock_prices['date'].iloc[-1].date().isoformat(),
            'spy_feature_date': spy_prices['date'].iloc[-1].date().isoformat(),
            'provider': self.prices.provider,
        }
        return features, context, warnings
