# Antigravity — RiskPulse AI Signal Atlas Frontend Integration & Audit

You are integrating a **separate, pre-authored React + TypeScript frontend** into the existing, working `c2c` FastAPI repository. Preserve all model assets, Risk Engine, ingestion providers, security, Module A, Module B, local SQLite database, and the working git history. Do not retrain models or reconfigure GDELT/Marketaux/Bluesky as part of this task.

## Setup

1. Inspect the current repository and create a safe checkpoint commit. Never blindly overwrite existing `frontend/` or package files; ask for a safe merge if a frontend exists.
2. Extract this add-on into `c2c/frontend/`.
3. Ensure Node.js 20+ and npm are installed; run `npm install`, `npm test`, `npm run check:contracts`, `npm run build`.
4. Start the existing FastAPI backend (unchanged) and the React dev server `npm run dev`. Use Vite's `/api` proxy; don't expose admin tokens to frontend.
5. Inspect `http://127.0.0.1:8000/openapi.json` and actual response samples for all endpoints. Correct frontend adapters for any actual drift; don't silently change the API to fit the UI.

## Required acceptance tests — genuine data only

- `GET /health` ready and independent `/ingestion/health` configured; do **not** treat configured as verified live provider connectivity.
- Overview loads `/signals/summary` and `/signals` with real collected items, correct counts, source provenance, signal filtering and item selection. Verify both genuine Marketaux/GDELT news AND Bluesky social data in stored records. No invented numbers or fake records.
- Event Intelligence supports URL provenance, published vs first-seen timestamps, score ranges, all five event probabilities, status handling, errors, empty states, real user-entered manual inference via `/analyze`. Confirm the event classifier's known Product Launch skew is not hidden by design.
- Module A `/module-a/index` correctly supplies all 15 tickers and baseline weights. Use `POST /ingestion/module-a/recommend` for real saved-signal recommendations and verify weights sum to 1 and constraints; no automatic trade claims. Test user-triggered holdout `/module-a/backtest` with dataset available. If missing, visibly show failure, not static synthetic performance.
- Module B loads synthetic `portfolio`/scenario library; issuer placeholders must be resolved, and explicit manual confirmation is required. Verify stress, what-if, reverse-stress endpoints using genuine synthetic-model calculations, accurate horizon labels, separate loan 12-month ECL from instantaneous MTM, reference derivative notional != NAV.
- Verify selected saved-signal Module B proposal is reviewable (not auto-executed). Fix any stale tabs or scenario selection bugs.
- Check all four pages at widths 1600, 1440, 1024, 768, 390; screenshot and inspect clipping, contrast, chart/table legibility, keyboard access, and reduced-motion. Ensure text is readable on projected screen.
- Disconnect FastAPI intentionally and confirm UI displays error/empty states with no mocked successes.
- Figma reference: `https://www.figma.com/design/MayTa8LELjBHHpPNKUkNCc`. The account may be rate-limited; source mockups were visually inspected before writing code. Do not block frontend work on Figma access.
- Security: no `VITE_` provider keys, collector admin tokens, or secrets in JS bundles; source links use noopener/noreferrer; no remote image tracking; API request timeout and error handling working. For deployment enable HTTPS, auth as appropriate, rate controls, and explicit CORS/reverse proxy.
- End-to-end check: actual **news + social → saved models → Risk Journey → Module A suggestion → reviewed Module B stress**; collect sanitized logs/screenshots, never keys.
- Report `FRONTEND_FINAL_VERIFICATION_REPORT.md` with commands, counts, source URLs where publicly safe, screenshots, PASS/FAIL per feature and explicit limitations. Use `READY FOR DEPLOYMENT: YES` only after `npm build`, browser tests, genuine backend integration, and responsive audit pass. Otherwise say `NO` with blockers and recommended fixes.

Be accurate: the upstream research models and scenarios have measured limitations. UI polish does not validate financial predictions. Our product is a simulated decision-support demonstrator, not trading automation or regulatory risk infrastructure.
