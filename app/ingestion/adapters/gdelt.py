"""GDELT DOC 2.0 ArticleList. It provides headlines and discoverability, NOT verified publication timestamps."""
from __future__ import annotations
from datetime import datetime
from urllib.parse import urlsplit
import requests
from ..records import RawItem, canonical_url, parse_utc, sha


def parse_seen(value):
    if not isinstance(value, str):
        return None
    if len(value) == 14 and value.isdigit():
        from datetime import timezone
        return datetime.strptime(value,'%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
    return parse_utc(value)


class GDELTAdapter:
    name = 'gdelt'
    def __init__(self, config:dict, timeout:int=12, session=None):
        self.config=config
        self.timeout=timeout
        self.session=session or requests.Session()

    def collect(self, observed_at:datetime, limit:int) -> list[RawItem]:
        import time
        items=[]
        seen=set()
        for i, query in enumerate(self.config['queries']):
            if i > 0:
                time.sleep(7.0)
            params={'query':query,'mode':'artlist','format':'json','sort':'datedesc',
                    'maxrecords':int(self.config['max_records_per_query'])}
            if self.config.get('timespan'):
                params['timespan'] = self.config['timespan']
            res = None
            url = self.config['base_url']
            for attempt in range(3):
                try:
                    res = self.session.get(url, params=params, timeout=self.timeout,
                                           headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
                    if getattr(res, 'status_code', 200) == 429:
                        time.sleep(8.0)
                        continue
                    break
                except requests.RequestException:
                    if attempt < 2:
                        time.sleep(6.0)
                        continue
                    raise
            if res is None:
                continue
            if getattr(res, 'status_code', 200) == 429:
                if items:
                    return items
                time.sleep(7.5)
                continue
            res.raise_for_status()
            text = getattr(res, 'text', None)
            if text is not None and not text.strip():
                payload = {'articles': []}
            else:
                if 'json' not in res.headers.get('Content-Type','').lower() and 'articles' not in (text or ''):
                    raise ValueError('GDELT returned non-JSON response; check API query/rate limit')
                payload=res.json()
            if isinstance(payload, list):
                payload = {'articles': payload}
            if not isinstance(payload,dict) or not isinstance(payload.get('articles',[]),list):
                raise ValueError('Unexpected GDELT response structure')
            for a in payload.get('articles',[]):
                if not isinstance(a,dict):continue
                if str(a.get('language','English')).lower() not in ('english','en'):continue
                url=a.get('url','')
                title=a.get('title','')
                if not url or not title:continue
                try:
                    canonical=canonical_url(url)
                except ValueError:continue
                if canonical in seen:continue
                seen.add(canonical)
                host=urlsplit(canonical).hostname or 'Unknown'
                items.append(RawItem('financial_news',host,sha(canonical),canonical,title,
                                     observed_at,parse_seen(a.get('seendate')),
                                     'gdelt_discovery','en'))
        if not items:
            import os
            token = os.environ.get('MARKETAUX_API_TOKEN', '').strip()
            if token:
                try:
                    from .marketaux import MarketauxAdapter
                    m_adapter = MarketauxAdapter({}, self.timeout, self.session)
                    return m_adapter.collect(observed_at, limit)
                except Exception:
                    pass
        return items
