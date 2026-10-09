"""Opt-in REAL provider connectivity check, not part of offline test suite."""
from app.ingestion.config import IngestionConfig
from app.ingestion.service import IngestionService
from app.ingestion.records import utc_now


def main():
    cfg=IngestionConfig.load()
    svc=IngestionService(cfg)
    good=0
    for name in cfg.raw['enabled_sources']:
        try:
            batch=svc.adapters[name].collect(utc_now(),3)
            print(f'{name}: REQUEST SUCCEEDED; {len(batch)} results (zero can be valid)', flush=True)
            for entry in batch[:2]:
                safe_text = entry.text[:90].encode('ascii', errors='replace').decode('ascii')
                print(f'  {entry.source_type}: {safe_text}', flush=True)
            good+=1
        except Exception as exc:
            print(f'{name}: REQUEST FAILED: {type(exc).__name__}: {exc}', flush=True)
    if good!=len(cfg.raw['enabled_sources']):
        raise SystemExit('One or more providers unavailable. Do NOT claim two-source live ingestion.')
    print('Both provider HTTP calls succeeded; downstream acceptance/inference still requires separate checks.', flush=True)

if __name__=='__main__':main()
