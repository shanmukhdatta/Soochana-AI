"""FastAPI server for Soochana AI — production-grade, scalable, and observable."""
from contextlib import asynccontextmanager
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings
from app.schemas import (
    AnalyzeRequest, AnalyzeResponse, BatchRequest, BatchResponse, HealthResponse,
)
from app.engine.pipeline import RiskEngine
from app.services.prices import MarketDataError
from app.services.features import UnsupportedTicker

log = logging.getLogger('soochana-backend')
START_TIME = time.time()
REQUEST_COUNTER = 0


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
    title='Soochana AI — Institutional Financial Risk Intelligence API',
    version='1.0.0',
    description='Scalable ML inference pipeline, tactical index rebalancing, and multi-asset stress testing.',
    lifespan=lifespan,
)

# CORS middleware for secure cross-origin dashboard communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.middleware('http')
async def observability_middleware(request: Request, call_next):
    global REQUEST_COUNTER
    REQUEST_COUNTER += 1
    req_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
    t0 = time.perf_counter()
    response: Response = await call_next(request)
    elapsed = time.perf_counter() - t0
    response.headers['X-Request-ID'] = req_id
    response.headers['X-Process-Time'] = f'{elapsed:.4f}s'
    return response


def ready_engine() -> RiskEngine:
    engine = getattr(app.state, 'pipeline', None)
    if engine is None:
        msg = getattr(app.state, 'load_error', 'Engine not initialized')
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=msg)
    return engine


@app.get('/health', response_model=HealthResponse)
def health():
    if getattr(app.state, 'pipeline', None) is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={'status': 'not_ready', 'reason': getattr(app.state, 'load_error', 'Model weights not loaded')},
        )
    return HealthResponse(status='ready')


@app.get('/health/live')
def liveness():
    return {'status': 'healthy', 'service': 'soochana-backend'}


@app.get('/metrics')
def metrics():
    uptime = time.time() - START_TIME
    is_ready = getattr(app.state, 'pipeline', None) is not None
    settings = getattr(app.state, 'settings', None)
    return {
        'service': 'soochana-backend',
        'version': '1.0.0',
        'uptime_seconds': round(uptime, 2),
        'total_requests_served': REQUEST_COUNTER,
        'model_pipeline_ready': is_ready,
        'device': getattr(settings, 'model_device', 'cpu') if settings else 'cpu',
        'modules': ['engine', 'module_a', 'module_b', 'ingestion', 'assistant', 'portfolio'],
    }


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
    settings = getattr(app.state, 'settings', None)
    max_items = settings.max_batch if settings is not None else 20
    if len(req.items) > max_items:
        raise HTTPException(status_code=422, detail=f'Maximum {max_items} items per batch')
    items = [_analyze(item) for item in req.items]
    return BatchResponse(results=items, total=len(items))


# Module A: Tactical Index Rebalancing
from app.module_a.router import router as module_a_router
app.include_router(module_a_router)

# Module B: Multi-Asset Stress Testing
from app.module_b.router import router as module_b_router
app.include_router(module_b_router)

# Ingestion Pipeline
from app.ingestion.router import router as ingestion_router
app.include_router(ingestion_router)

# Bujji Financial Copilot Assistant
from app.assistant.router import router as assistant_router
app.include_router(assistant_router)

# Custom Portfolio Studio & Retail Investor Shield
from app.portfolio.router import router as portfolio_router
app.include_router(portfolio_router)

