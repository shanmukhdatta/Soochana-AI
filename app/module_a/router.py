"""Self-contained module router; no changes to existing /analyze endpoints."""
from datetime import date
from pathlib import Path
import json
import uuid
from fastapi import APIRouter, HTTPException, Request
from .config import ModuleAConfig
from .schemas import RebalanceRequest, RebalanceFromNewsRequest, BacktestRequest, Signal
from .strategy import rebalance
from .constraints import InfeasiblePortfolio
from .backtester import read_prices, read_signals,run_backtest,BacktestDataError
from app.services.features import UnsupportedTicker
from app.services.prices import MarketDataError

router=APIRouter(prefix='/module-a',tags=['Module A: Tactical Rebalancing'])
ROOT=Path(__file__).resolve().parents[2]
DATA_ROOT=ROOT/'data'/'module_a'


def cfg():return ModuleAConfig.load()

@router.get('/index')
def get_index():
    c=cfg()
    return {'version':'module-a-v1','tickers':list(c.tickers),'baseline_weights':c.baseline,
            'sector_targets':c.sector_targets,'constraints':{
            'max_position_deviation':c.position_deviation_limit,
            'max_sector_deviation':c.sector_deviation_limit,
            'max_one_way_turnover':c.one_way_turnover_limit},
            'mode':'simulation_only','note':'Equal-weight baseline, not live account holdings'}


def safe_rebalance(*args,**kwargs):
    try:return rebalance(*args,**kwargs)
    except (ValueError,InfeasiblePortfolio) as exc:raise HTTPException(422,str(exc)) from exc

@router.post('/rebalance')
def rebalance_scores(req:RebalanceRequest):
    return safe_rebalance(cfg(),req.signals,req.as_of,req.current_weights,req.strategy)

@router.post('/rebalance-from-news')
def rebalance_from_news(req:RebalanceFromNewsRequest,request:Request):
    engine=getattr(request.app.state,'pipeline',None)
    if engine is None:
        raise HTTPException(status_code=503,detail='Risk Engine weights are not loaded')
    c=cfg()
    items=[]
    for news in req.articles:
        try:
            analyzed=engine.analyze(news)
        except (UnsupportedTicker,ValueError) as e:
            raise HTTPException(status_code=422,detail=str(e)) from e
        except MarketDataError as e:
            raise HTTPException(status_code=503,detail=f'Market data unavailable: {e}') from e
        items.append(Signal(ticker=analyzed.ticker,published_at=analyzed.published_at,
            source=analyzed.source,headline=news.text,sentiment_score=analyzed.sentiment.score,
            impact_score=analyzed.impact.score,event_class=analyzed.event.class_name,
            event_confidence=analyzed.event.confidence))
    return safe_rebalance(c,items,req.as_of,req.current_weights,req.strategy)

@router.post('/backtest')
def backtest(req:BacktestRequest):
    # Dataset IDs may only refer to files inside data/module_a/, never arbitrary paths.
    c=cfg()
    try:
        prices=read_prices(DATA_ROOT/f'{req.dataset_id}_prices.csv',c)
        signals=read_signals(DATA_ROOT/f'{req.dataset_id}_signals.csv',c)
        result=run_backtest(c,prices,signals,
            date.fromisoformat(req.start_date) if req.start_date else None,
            date.fromisoformat(req.end_date) if req.end_date else None,req.strategy)
        run_id=str(uuid.uuid4())
        output=DATA_ROOT/'results'
        output.mkdir(parents=True,exist_ok=True)
        path=output/f'{run_id}.json'
        path.write_text(json.dumps(result,allow_nan=False,indent=2),encoding='utf-8')
        return {'run_id':run_id,'status':'completed','selected_strategy':req.strategy,
                'metrics':{k:v['metrics'] for k,v in result['evaluations'].items()},
                'session_count':result['session_count'],'result_url':f'/module-a/backtest/{run_id}'}
    except (BacktestDataError,ValueError) as e:
        raise HTTPException(status_code=422,detail=str(e)) from e

@router.get('/backtest/{run_id}')
def backtest_result(run_id:str):
    try:canonical=str(uuid.UUID(run_id))
    except ValueError as e:raise HTTPException(422,'Invalid run ID') from e
    path=DATA_ROOT/'results'/f'{canonical}.json'
    if not path.is_file():raise HTTPException(404,'No such run')
    return json.loads(path.read_text(encoding='utf-8'))
