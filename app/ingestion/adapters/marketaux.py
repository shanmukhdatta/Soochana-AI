"""Marketaux financial news adapter with automatic GDELT fallback."""
from __future__ import annotations
import os
import logging
from datetime import datetime
from urllib.parse import urlsplit
import requests
from ..records import RawItem, canonical_url, parse_utc, sha

log = logging.getLogger('riskpulse.ingestion.marketaux')


class MarketauxAdapter:
    name = 'marketaux'

    def __init__(self, config: dict, timeout: int = 15, session = None):
        self.config = config or {}
        self.timeout = timeout
        self.session = session or requests.Session()

    def collect(self, observed_at: datetime, limit: int) -> list[RawItem]:
        api_token = os.environ.get('MARKETAUX_API_TOKEN', '').strip() or self.config.get('api_token', '').strip()
        if not api_token:
            log.info("MARKETAUX_API_TOKEN not set; falling back to GDELT backup adapter.")
            from .gdelt import GDELTAdapter
            gdelt_cfg = self.config.get('gdelt_fallback') or self.config.get('gdelt') or {
                'base_url': 'https://api.gdeltproject.org/api/v2/doc/doc',
                'timespan': '',
                'max_records_per_query': 25,
                'queries': ['Microsoft OR Apple OR Nvidia OR Tesla']
            }
            gdelt_adapter = GDELTAdapter(gdelt_cfg, self.timeout, self.session)
            return gdelt_adapter.collect(observed_at, limit)

        base_url = self.config.get('base_url', 'https://api.marketaux.com/v1/news/all')
        symbols = self.config.get('symbols', ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'QCOM', 'META'])
        if isinstance(symbols, list):
            symbols_str = ','.join(symbols)
        else:
            symbols_str = str(symbols)

        params = {
            'api_token': api_token,
            'symbols': symbols_str,
            'filter_entities': 'true',
            'language': 'en',
            'limit': min(limit, int(self.config.get('max_records_per_query', 10)))
        }

        try:
            res = self.session.get(base_url, params=params, timeout=self.timeout,
                                   headers={'User-Agent': 'RiskPulseAI-Ingestion/1.0'})
            res.raise_for_status()
            payload = res.json()
            if not isinstance(payload, dict) or 'data' not in payload or not isinstance(payload['data'], list):
                raise ValueError('Unexpected Marketaux API response structure')

            items: list[RawItem] = []
            seen = set()
            for art in payload['data']:
                if not isinstance(art, dict):
                    continue
                url = art.get('url', '')
                title = art.get('title', '')
                if not url or not title:
                    continue
                try:
                    canonical = canonical_url(url)
                except ValueError:
                    continue
                if canonical in seen:
                    continue
                seen.add(canonical)
                host = art.get('source') or urlsplit(canonical).hostname or 'Marketaux'
                uuid = art.get('uuid') or sha(canonical)
                published = parse_utc(art.get('published_at'))

                text = title
                desc = art.get('description') or art.get('snippet')
                if desc and desc not in text:
                    text = f"{title}. {desc}"

                items.append(RawItem(
                    'financial_news',
                    host,
                    uuid,
                    canonical,
                    text,
                    observed_at,
                    published,
                    'self_reported',
                    art.get('language', 'en')
                ))
                if len(items) >= limit:
                    return items
            return items
        except Exception as exc:
            log.warning("Marketaux request failed (%s); falling back to GDELT backup: %s", type(exc).__name__, exc)
            from .gdelt import GDELTAdapter
            gdelt_cfg = self.config.get('gdelt_fallback') or self.config.get('gdelt') or {
                'base_url': 'https://api.gdeltproject.org/api/v2/doc/doc',
                'timespan': '',
                'max_records_per_query': 25,
                'queries': ['Microsoft OR Apple OR Nvidia OR Tesla']
            }
            gdelt_adapter = GDELTAdapter(gdelt_cfg, self.timeout, self.session)
            return gdelt_adapter.collect(observed_at, limit)
