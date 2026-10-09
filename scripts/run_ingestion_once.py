"""Controlled local test that collects then loads existing real ML models once."""
import argparse
import json
from app.ingestion.config import IngestionConfig
from app.ingestion.service import IngestionService


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--collect-only',action='store_true',help='Do not load trained models')
    p.add_argument('--process-only',action='store_true',help='Process previously stored events')
    args=p.parse_args()
    if args.collect_only and args.process_only:p.error('Choose one mode')
    svc=IngestionService(IngestionConfig.load())
    if not args.process_only:
        print('COLLECT',json.dumps(svc.collect_once(),indent=2))
    if not args.collect_only:
        from app.config import Settings
        from app.engine.pipeline import RiskEngine
        engine=RiskEngine.from_settings(Settings.from_env())
        print('INFERENCE',json.dumps(svc.process_once(engine),indent=2))

if __name__=='__main__':main()
