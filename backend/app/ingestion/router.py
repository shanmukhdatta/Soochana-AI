"""Dashboard-read API plus protected manual collection; one controlled model worker."""
from __future__ import annotations
from functools import lru_cache
from fastapi import APIRouter, Header, HTTPException, Query, Request
from .config import IngestionConfig
from .service import IngestionService

router=APIRouter(tags=['Ingestion: Financial News and Social'])


@lru_cache(maxsize=1)
def get_service():
    return IngestionService(IngestionConfig.load())


def authorized(token:str|None):
    cfg=get_service().config
    if not cfg.admin_token or len(cfg.admin_token) < 24:
        raise HTTPException(503,'Set INGESTION_ADMIN_TOKEN to a random value of at least 24 characters')
    import secrets
    if token is None or not secrets.compare_digest(token,cfg.admin_token):
        raise HTTPException(403,'Invalid or missing ingestion admin token')


@router.get('/ingestion/health')
def ingestion_health():
    service=get_service()
    return {'status':'configured','database_available':True,
            'enabled_sources':service.config.raw['enabled_sources'],
            'live_provider_connectivity':'not_tested_by_this_endpoint',
            'scheduler':'external_process_only',
            'metrics':service.repository.summary()['recent_collection_runs']}


@router.post('/ingestion/collect-once')
def collect_once(x_ingestion_token:str|None=Header(None)):
    authorized(x_ingestion_token)
    return get_service().collect_once()


@router.post('/ingestion/process-once')
def process_once(request:Request,x_ingestion_token:str|None=Header(None)):
    authorized(x_ingestion_token)
    engine=getattr(request.app.state,'pipeline',None)
    if engine is None:raise HTTPException(503,'Risk Engine not loaded; queued signals remain pending')
    return get_service().process_once(engine)


@router.get('/signals/summary')
def summary():
    return get_service().repository.summary()


@router.get('/signals')
def signals(limit:int=Query(30,ge=1,le=100),offset:int=Query(0,ge=0),
            ticker:str|None=None,status:str|None=None,source_type:str|None=None):
    if status and status not in {'pending','processing','completed','retry','dead'}:
        raise HTTPException(422,'Invalid status')
    if source_type and source_type not in {'financial_news','social_media'}:
        raise HTTPException(422,'Invalid source_type')
    if ticker:
        ticker=ticker.upper()
        if ticker not in get_service().config.aliases:raise HTTPException(422,'Unsupported ticker filter')
    return get_service().repository.list(limit,offset,ticker,status,source_type)


@router.get('/signals/{event_id}')
def signal_detail(event_id:str):
    if len(event_id)!=64 or any(c not in '0123456789abcdef' for c in event_id):
        raise HTTPException(422,'Invalid event identifier')
    record=get_service().repository.one(event_id)
    if record is None:raise HTTPException(404,'Event not found')
    return record


@router.post('/ingestion/module-a/recommend')
def module_a_from_saved_signals(request:Request, limit:int=Query(100,ge=1,le=300)):
    """Rebalance against saved *real* predictions. Does not run ML again."""
    try:
        from app.module_a.config import ModuleAConfig
        from app.module_a.schemas import Signal
        from app.module_a.strategy import rebalance
        from app.ingestion.records import utc_now,parse_utc
        from app.module_a.constraints import InfeasiblePortfolio
    except ImportError as exc:
        raise HTTPException(503,'Module A not installed') from exc
    cfg=ModuleAConfig.load()
    records=get_service().repository.list(limit=limit,status='completed')['items']
    items=[]
    for row in records:
        if row['ticker'] not in cfg.tickers or not row['inference']:
            continue
        x=row['inference']
        items.append(Signal(ticker=row['ticker'],source=row['source_type'],headline=row['text'],
           published_at=parse_utc(row['analysis_as_of']),sentiment_score=x['sentiment']['score'],
           impact_score=x['impact']['score'],event_class=x['event']['class_name'],
           event_confidence=x['event']['confidence']))
    try:
        response=rebalance(cfg,items,utc_now())
    except (ValueError,InfeasiblePortfolio) as exc:
        raise HTTPException(422,str(exc)) from exc
    return {'recommendation':response,'saved_signals_considered':len(items),
            'note':'Simulated weights only. Does not place trades; no news is not negative news.'}


@router.get('/signals/{event_id}/module-b-suggestion')
def module_b_from_saved_signal(event_id:str):
    """Propose a reviewable scenario; NEVER executes a stress without opt-in."""
    row=get_service().repository.one(event_id)
    if row is None:raise HTTPException(404,'Event not found')
    if row['status']!='completed' or row['inference'] is None:
        raise HTTPException(409,'Event has no completed inference')
    try:
        from app.module_b.event_mapper import propose_scenario
        from app.schemas import AnalyzeResponse
    except ImportError as exc:
        raise HTTPException(503,'Module B not installed') from exc
    result=propose_scenario(AnalyzeResponse.model_validate(row['inference']), 'auto')
    return {'event_id':event_id, 'proposal':{
        **result,'suggested_scenario':result['suggested_scenario'].model_dump(mode='json')},
        'executed':False,'requires_manual_review':True}
