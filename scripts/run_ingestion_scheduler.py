"""Separate polling process: run ONLY one on the machine."""
import logging
from app.ingestion.scheduler import run_forever

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s')
    run_forever()
