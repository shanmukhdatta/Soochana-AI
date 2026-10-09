"""Normalized provider metadata. Never claim provider discovery time = publication time."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
import hashlib
import re


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(t: datetime) -> str:
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError('Timestamp must be timezone-aware')
    return t.astimezone(timezone.utc).isoformat().replace('+00:00','Z')


def parse_utc(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        t = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if t.tzinfo is None or t.utcoffset() is None:
            return None
        return t.astimezone(timezone.utc)
    except ValueError:
        return None


def canonical_url(url: str) -> str:
    """Strip common tracking arguments; retain article-specific query arguments."""
    if not isinstance(url, str):
        raise ValueError('URL must be a string')
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in ('http','https') or not parts.hostname:
        raise ValueError('Source URL must be HTTP(S)')
    if parts.username or parts.password:
        raise ValueError('Source URL may not contain credentials')
    blocked = {'fbclid','gclid','igshid','mc_cid','mc_eid','ref_src','ref_url'}
    args = [(k,v) for k,v in parse_qsl(parts.query,keep_blank_values=True)
            if not k.lower().startswith('utm_') and k.lower() not in blocked]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(),parts.path or '/', urlencode(args), ''))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class RawItem:
    source_type: str
    source_name: str
    source_id: str
    source_url: str
    text: str
    first_seen_at: datetime
    reported_at: datetime | None = None
    timestamp_quality: str = 'unknown'
    language: str | None = None


def normalize(item: RawItem) -> RawItem:
    if item.source_type not in ('financial_news','social_media'):
        raise ValueError('Unsupported source type')
    if item.timestamp_quality not in ('gdelt_discovery','self_reported','unknown'):
        raise ValueError('Unexpected timestamp quality')
    url = canonical_url(item.source_url)
    text = re.sub(r'\s+', ' ', item.text or '').strip()
    if not 5 <= len(text) <= 10000:
        raise ValueError('Text length outside allowed range')
    if len(item.source_id) > 500 or not item.source_id.strip():
        raise ValueError('Invalid source ID')
    first_seen = parse_utc(iso(item.first_seen_at))
    claimed = parse_utc(iso(item.reported_at)) if item.reported_at else None
    if claimed and claimed > first_seen:
        # Provider times cannot be trusted to move our market cutoff forward.
        claimed = None
    return RawItem(item.source_type,item.source_name.strip()[:100],item.source_id,url,
                   text,first_seen,claimed,item.timestamp_quality,item.language)
