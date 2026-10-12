"""Dedicated collector loop; calls EXISTING FastAPI for inference to avoid second BERT copy."""
from __future__ import annotations
import os
import time
import logging
import requests
from .config import IngestionConfig
from .service import IngestionService

log=logging.getLogger('riskpulse.scheduler')


def run_one_cycle(service:IngestionService, api_url='http://127.0.0.1:8000'):
    result=service.collect_once()
    cfg=service.config
    # Do not open a publicly configurable arbitrary URL (avoids SSRF).
    from urllib.parse import urlsplit
    parts=urlsplit(api_url)
    if parts.scheme!='http' or parts.hostname not in {'localhost','127.0.0.1'}:
        raise ValueError('Scheduler may only dispatch to loopback FastAPI')
    if not cfg.admin_token or len(cfg.admin_token)<24:
        return {'collection':result,'inference':'not_dispatched',
                'reason':'Set INGESTION_ADMIN_TOKEN and start Uvicorn to run inference'}
    try:
        r=requests.post(api_url.rstrip('/')+'/ingestion/process-once',
           headers={'X-Ingestion-Token':cfg.admin_token},timeout=180)
        r.raise_for_status()
        result_inference=r.json()
    except (requests.RequestException, ValueError) as exc:
        log.warning('Existing FastAPI inference unavailable; pending rows remain queued: %s',exc)
        return {'collection':result,'inference':'not_dispatched',
                'reason':f'{type(exc).__name__}: {str(exc)[:120]}'}
    return {'collection':result,'inference':result_inference}


def run_forever():
    config=IngestionConfig.load()
    service=IngestionService(config)
    lock=config.db_path.with_suffix('.scheduler.lock')
    # Prevent accidental multiple collectors. Manually remove stale lock only
    # after verifying the old process has exited.
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError as e:
        raise RuntimeError('Scheduler already running or stale lock exists') from e
    try:
        with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
        while True:
            try:log.info('Collection/inference cycle: %s',run_one_cycle(service))
            except Exception:log.exception('Ingestion iteration failed; retry next interval')
            time.sleep(config.raw['poll_interval_seconds'])
    finally:
        try:lock.unlink()
        except FileNotFoundError:pass
