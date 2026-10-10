# RiskPulse AI — Ingestion Storage & Repository

This directory contains storage configurations and demonstration records for the RiskPulse AI multi-source ingestion pipeline:

1. **Seed Demonstration Database (`riskpulse.db`):**
   - A seeded demonstration SQLite database containing 29 verified public news and social media signals with real URLs and timestamps is bundled so reviewers can inspect live signals, model inferences, and downstream modules immediately offline.
   - The dataset contains only public news headlines/snippets (Marketaux) and public AT-Protocol social posts (Bluesky) referencing supported equity tickers. No private or proprietary records are included.

2. **Runtime Files:**
   - Active database write-ahead logs (`*.db-wal`), shared memory (`*.db-shm`), and scheduler lock files (`*.scheduler.lock`) are transient runtime artifacts and remain git-ignored.

3. **Live Ingestion Testing:**
   - To test genuine live two-source polling, provide valid provider configuration in `.env` and execute:
     ```bash
     python -m scripts.check_ingestion_sources
     python -m scripts.run_ingestion_once
     ```
