"""Price providers used solely for PRE-publication daily market context.

Users must choose either yfinance (external provider) or local adjusted-price CSV files.
No future/current-day price rows are allowed to enter the model features.
"""
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
import re
import pandas as pd


class MarketDataError(RuntimeError):
    pass


def _key(col) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(col).strip().lower()).strip('_')


def normalize_prices(raw: pd.DataFrame, symbol: str, cutoff_date: date) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise MarketDataError(f'No price history for {symbol}')
    data = raw.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            '_'.join(str(x) for x in tup if str(x).lower() not in ('', 'none', symbol.lower()))
            for tup in data.columns.to_flat_index()
        ]
    cols = {_key(c): c for c in data.columns}
    if 'date' not in cols and not isinstance(data.index, pd.RangeIndex):
        data = data.reset_index()
        cols = {_key(c): c for c in data.columns}
    dcol = next((cols[c] for c in ['date', 'datetime', 'timestamp', 'index'] if c in cols), None)
    close_col = next((cols[c] for c in ['adj_close', 'adjusted_close', 'adjclose'] if c in cols), None)
    volume_col = cols.get('volume')
    if dcol is None or close_col is None:
        raise MarketDataError(
            f'{symbol}: date and adjusted-close columns required; unadjusted Close is not a safe replacement'
        )
    d = pd.to_datetime(data[dcol], errors='coerce', utc=True)
    frame = pd.DataFrame({
        'date': d.dt.tz_convert('UTC').dt.tz_localize(None).dt.normalize(),
        'adj_close': pd.to_numeric(data[close_col], errors='coerce'),
        'volume': pd.to_numeric(data[volume_col], errors='coerce') if volume_col is not None else float('nan'),
    })
    frame = frame.dropna(subset=['date','adj_close'])
    frame = frame[frame['adj_close'] > 0]
    frame = frame.drop_duplicates('date', keep='last').sort_values('date')
    # Hard exclude same-day/future entries, even if a provider gives them.
    frame = frame[frame['date'] < pd.Timestamp(cutoff_date)].reset_index(drop=True)
    if len(frame) < 21:
        raise MarketDataError(f'{symbol}: fewer than 21 completed market sessions strictly before {cutoff_date}')
    last_day = frame['date'].iloc[-1].date()
    if (cutoff_date - last_day).days > 10:
        raise MarketDataError(
            f'{symbol}: historical price data stale: latest {last_day}, requested cutoff {cutoff_date}'
        )
    return frame


class PriceProvider:
    def __init__(self, provider: str, price_csv_dir: Path | None=None):
        if provider not in {'csv','yfinance'}:
            raise ValueError('Unknown price provider')
        self.provider = provider
        self.price_csv_dir = price_csv_dir

    @lru_cache(maxsize=512)
    def get(self, ticker: str, cutoff: date) -> pd.DataFrame:
        if self.provider == 'csv':
            path = Path(self.price_csv_dir) / f'{ticker}.csv'
            if not path.is_file():
                raise MarketDataError(f'Missing local price CSV: {path}')
            raw = pd.read_csv(path)
        else:
            import yfinance as yf
            start = cutoff - timedelta(days=210)
            try:
                raw = yf.download(
                    ticker,
                    start=start.isoformat(),
                    end=cutoff.isoformat(),  # yfinance end is exclusive
                    auto_adjust=False,
                    progress=False,
                    actions=False,
                    threads=False,
                )
            except Exception as e:
                raise MarketDataError(f'{ticker}: price provider request failed: {e}') from e
        return normalize_prices(raw, ticker, cutoff)
