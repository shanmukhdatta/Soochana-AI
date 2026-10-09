"""Collection and inference are deliberately separate; no model weights loaded twice."""
from __future__ import annotations
from datetime import timedelta
import logging
from .config import IngestionConfig
from .records import normalize,utc_now,iso
from .ticker_matcher import TickerMatcher
from .repository import Repository
from .adapters.gdelt import GDELTAdapter
from .adapters.bluesky import BlueskyAdapter
from .adapters.marketaux import MarketauxAdapter
from app.schemas import AnalyzeRequest
from app.services.features import UnsupportedTicker
from app.services.prices import MarketDataError

log=logging.getLogger('riskpulse.ingestion')


class IngestionService:
    def __init__(self,config:IngestionConfig,adapters=None,repository=None):
        self.config=config
        self.repository=repository or Repository(config.db_path)
        self.matcher=TickerMatcher(config.aliases)
        self.adapters=adapters or {
            'gdelt':GDELTAdapter(config.raw.get('gdelt',{}),config.raw['http_timeout_seconds']),
            'bluesky':BlueskyAdapter(config.raw.get('bluesky',{}),config.raw['http_timeout_seconds']),
            'marketaux':MarketauxAdapter(config.raw.get('marketaux',{}),config.raw['http_timeout_seconds'])
        }

    def collect_once(self):
        outcomes={}; total_inserted=0
        for name in self.config.raw['enabled_sources']:
            fetched=inserted=rejected=0
            status='success';detail=''
            try:
                observed=utc_now()
                batch=self.adapters[name].collect(observed,self.config.raw['max_rows_per_source_per_poll'])
                fetched=len(batch)
                for item in batch:
                    try:
                        n=normalize(item)
                        if n.language not in ('en',None):
                            rejected+=1;continue
                        if n.language is None and not self.config.raw.get('allow_unknown_language',False):
                            rejected+=1;continue
                        if n.first_seen_at < utc_now()-timedelta(hours=self.config.raw['max_item_age_hours']):
                            rejected+=1;continue
                        matches=self.matcher.match(n.text)
                        if not matches:
                            rejected+=1;continue
                        for ticker,method in matches.items():
                            if self.repository.insert(n,ticker,method):inserted+=1
                    except (ValueError,KeyError,TypeError) as exc:
                        log.debug('Dropped malformed record from %s: %s',name,exc)
                        rejected+=1
            except Exception as exc:
                status='error';detail=f'{type(exc).__name__}: {exc}'[:500]
                log.warning('Collector %s failed: %s',name,detail)
            self.repository.record_collection(name,status,fetched,inserted,rejected,detail)
            outcomes[name]={'status':status,'fetched':fetched,'inserted':inserted,'rejected':rejected,
                            'detail':detail or None}
            total_inserted+=inserted
        return {'sources':outcomes,'total_inserted':total_inserted,
                'note':'Collection alone does not mean real-model inference completed.'}

    def process_once(self,engine):
        if engine is None:raise RuntimeError('Unified Risk Engine is not loaded')
        conf=self.config.raw
        claimed=self.repository.claim(conf['max_inference_per_poll'],conf['lease_seconds'],conf['max_attempts'])
        counts={'claimed':len(claimed),'completed':0,'retry':0,'dead':0}
        for row in claimed:
            try:
                # Observed-at convention is conservative. Never backdate market features
                # to a self-reported social creation time or GDELT discovery timestamp.
                from .records import parse_utc
                req=AnalyzeRequest(text=row['text'],ticker=row['ticker'],
                     published_at=parse_utc(row['analysis_as_of']),source=row['source_type'])
                result=engine.analyze(req)
                payload=result.model_dump(mode='json') if hasattr(result,'model_dump') else dict(result)
                self.repository.success(row['id'],payload)
                counts['completed']+=1
            except Exception as exc:
                permanent=isinstance(exc,(UnsupportedTicker,ValueError))
                attempts=int(row['attempts'])+1
                self.repository.failure(row['id'],f'{type(exc).__name__}: {exc}',attempts,
                   conf['max_attempts'],conf['retry_base_seconds'],permanent)
                counts['dead' if permanent or attempts>=conf['max_attempts'] else 'retry']+=1
        return counts
