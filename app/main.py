"""FastAPI server for one event/batches; no duplicated AI logic in modules."""
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, status

from app.config import Settings
from app.schemas import (
    AnalyzeRequest, AnalyzeResponse, BatchRequest, BatchResponse, HealthResponse,
)
from app.engine.pipeline import RiskEngine
from app.services.prices import MarketDataError
from app.services.features import UnsupportedTicker

log = logging.getLogger('risk-engine')


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = None
    app.state.load_error = None
    try:
        settings = Settings.from_env()
        app.state.settings = settings
        if not settings.skip_model_load:
            app.state.pipeline = RiskEngine.from_settings(settings)
            log.info('Risk Engine models loaded and ready')
        else:
            app.state.load_error = 'SKIP_MODEL_LOAD=1: mock/testing mode; real inference unavailable'
    except Exception as exc:
        app.state.load_error = f'{type(exc).__name__}: {exc}'
        log.exception('Risk Engine failed to initialize')
    yield
    app.state.pipeline = None


app = FastAPI(
    title='Unified Financial Risk Engine',
    version='1.0.0',
    description='Unified sentiment, event and estimated market-impact signals. Uses trained weights; not investment advice.',
    lifespan=lifespan,
)


def ready_engine() -> RiskEngine:
    engine = getattr(app.state,'pipeline',None)
    if engine is None:
        msg = getattr(app.state,'load_error','Engine not initialized')
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=msg)
    return engine


@app.get('/health', response_model=HealthResponse)
def health():
    if getattr(app.state,'pipeline',None) is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={'status':'not_ready','reason':getattr(app.state,'load_error','Model weights not loaded')},
        )
    return HealthResponse(status='ready')


def _analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    engine = ready_engine()
    try:
        return engine.analyze(req)
    except UnsupportedTicker as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except MarketDataError as exc:
        raise HTTPException(status_code=503, detail=f'Market price data unavailable: {exc}') from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post('/analyze', response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    return _analyze(req)


@app.post('/analyze/batch', response_model=BatchResponse)
def analyze_batch(req: BatchRequest):
    from app.config import Settings
    settings = getattr(app.state,'settings',None)
    max_items = settings.max_batch if settings is not None else 20
    if len(req.items) > max_items:
        raise HTTPException(status_code=422,detail=f'Maximum {max_items} items per batch')
    # Atomic result by design. For unavailable data on any item, the whole
    # batch fails with an explicit error instead of returning invented signals.
    items = [_analyze(item) for item in req.items]
    return BatchResponse(results=items,total=len(items))


# RiskPulse Module A; original Risk Engine endpoints remain unchanged.
from app.module_a.router import router as module_a_router
app.include_router(module_a_router)
