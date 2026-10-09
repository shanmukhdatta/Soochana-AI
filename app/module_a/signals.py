"""Deterministic signal aggregation with identity-based news deduplication."""
from datetime import datetime, timezone
import hashlib
import math
import re
from .schemas import Signal
from .config import ModuleAConfig


def normalized_headline(s: str)->str:
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]+', ' ', s.casefold())).strip()


def identity(s:Signal)->str:
    # Same article/syndicated headline for the same ticker and publication day is one event.
    # Source is deliberately excluded: reposts across outlets cannot multiply influence.
    text=normalized_headline(s.headline)
    raw=f'{s.ticker}|{s.published_at.date().isoformat()}|{text or s.event_id}'
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def aggregate(signals:list[Signal], as_of:datetime, config:ModuleAConfig,
              strategy:str='full_event_aware')->tuple[dict[str,float],dict[str,int],list[str]]:
    if as_of.tzinfo is None:
        raise ValueError('as_of must be timezone-aware')
    now=as_of.astimezone(timezone.utc)
    if len(signals)>config.max_signals_per_request:
        raise ValueError('Too many signals')
    selected={}
    ignored_future=0
    out_of_window=0
    for s in signals:
        if s.ticker not in config.tickers:
            raise ValueError(f'Unsupported constituent: {s.ticker}')
        # Conservative rule: at session-close, any publication from the same UTC
        # calendar day is excluded even if the timestamp predates market close.
        if s.published_at >= now or s.published_at.date()>=now.date():
            ignored_future+=1
            continue
        age=(now-s.published_at).total_seconds()/86400
        if age > config.news_lookback_days:
            out_of_window+=1
            continue
        key=identity(s)
        old=selected.get(key)
        if old is None or config.source_weights[s.source]>config.source_weights[old.source]:
            selected[key]=s
    per_ticker={t:[] for t in config.tickers}
    for s in selected.values():
        age=(now-s.published_at).total_seconds()/86400
        impact=.5+s.impact_score/10 if strategy!='sentiment_only' else 1.0
        if strategy=='full_event_aware':
            importance=(config.source_weights[s.source] * 2**(-age/config.signal_half_life_days)
                        *max(.05,s.event_confidence))
            impact*=config.event_multipliers[s.event_class]
        else:
            importance=1.0
        per_ticker[s.ticker].append((importance,s.sentiment_score*impact))
    result={}
    count={}
    for t,rows in per_ticker.items():
        count[t]=len(rows)
        evidence_mass=sum(w for w,_ in rows)
        # Full event-aware signals shrink toward zero when evidence is weak.
        # Without a neutral prior, a single headline's source, age and model
        # confidence all cancel out of a normalized weighted mean.
        denom=(evidence_mass + config.evidence_prior_weight
               if strategy=='full_event_aware' else evidence_mass)
        result[t]=float(max(-1.65,min(1.65,sum(w*value for w,value in rows)/denom))) if denom>0 else 0.0
    warnings=[]
    if ignored_future:warnings.append(f'{ignored_future} signal(s) excluded because publication is not strictly before the session date')
    if out_of_window:warnings.append(f'{out_of_window} signal(s) excluded outside lookback window')
    if len(selected)<len(signals)-ignored_future-out_of_window:
        warnings.append('Duplicate headlines deduplicated across sources')
    return result,count,warnings
