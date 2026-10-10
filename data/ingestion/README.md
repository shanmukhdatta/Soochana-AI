Runtime SQLite databases belong in this folder; **do not commit** `riskpulse.db`, `-wal`, `-shm`, or scheduler lock files.

The repository does not ship fabricated news/social-media records. Offline tests use clearly labeled synthetic fixtures. To demonstrate genuine two-source ingestion, run `python -m scripts.check_ingestion_sources` then `python -m scripts.run_ingestion_once` and show stored source records with real URLs and timestamps.
