"""Bluesky public searchPosts. Posts may be unavailable/require auth by provider."""
from __future__ import annotations
import requests
from datetime import datetime
from ..records import RawItem,parse_utc


class BlueskyAdapter:
    name='bluesky'
    def __init__(self, config:dict, timeout:int=12, session=None):
        self.config=config
        self.timeout=timeout
        self.session=session or requests.Session()

    def collect(self, observed_at:datetime, limit:int) -> list[RawItem]:
        items=[]
        seen=set()
        base_url = self.config['base_url']
        for query in self.config['queries']:
            params={'q':query,'sort':'latest','limit':int(self.config['max_records_per_query'])}
            res=self.session.get(base_url,params=params,timeout=self.timeout,
                                 headers={'User-Agent':'RiskPulseAI-Ingestion/1.0 (research prototype)'})
            if getattr(res, 'status_code', 200) == 403 and 'public.api.bsky.app' in base_url:
                base_url = base_url.replace('public.api.bsky.app', 'api.bsky.app')
                res=self.session.get(base_url,params=params,timeout=self.timeout,
                                     headers={'User-Agent':'RiskPulseAI-Ingestion/1.0 (research prototype)'})
            res.raise_for_status()
            payload=res.json()
            if not isinstance(payload,dict) or not isinstance(payload.get('posts',[]),list):
                raise ValueError('Unexpected Bluesky searchPosts response')
            for p in payload.get('posts',[]):
                if not isinstance(p,dict):continue
                record=p.get('record') or {}
                if not isinstance(record,dict):continue
                text=record.get('text')
                uri=p.get('uri')
                if not isinstance(text,str) or not isinstance(uri,str) or not uri.startswith('at://'):continue
                if uri in seen:continue
                langs=record.get('langs') or []
                if langs and 'en' not in langs:continue
                author=p.get('author') or {}
                did=author.get('did','')
                if not did or '/' not in uri:continue
                rkey=uri.rsplit('/',1)[-1]
                if not rkey or not did.startswith('did:'):continue
                url=f'https://bsky.app/profile/{did}/post/{rkey}'
                seen.add(uri)
                items.append(RawItem('social_media','Bluesky',uri,url,text,observed_at,
                                     parse_utc(record.get('createdAt')),
                                     'self_reported','en' if langs else None))
                if len(items)>=limit:return items
        return items
