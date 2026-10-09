# RiskPulse AI — Signal Atlas React Dashboard

An original 4-screen **Obsidian Iris** frontend mapped to the editable Figma design: https://www.figma.com/design/MayTa8LELjBHHpPNKUkNCc

**Scope:** Separate Vite + React + TypeScript frontend package for integration with the existing, working Windows `c2c` FastAPI repository. Does not replace backend, retrain ML, ingest providers in the browser, or execute trades.

## Quick start (Windows PowerShell)

1. Back up your `c2c` repository and make a checkpoint commit.
2. Extract this ZIP into `c2c/frontend/` (ensure `frontend/package.json` exists).
3. With FastAPI running at `http://127.0.0.1:8000`, open a separate terminal:

```powershell
cd C:\Users\lokes\datta\c2c\frontend
Copy-Item .env.example .env
npm install
npm test
npm run check:contracts
npm run build
npm run dev
```

4. Open http://127.0.0.1:5173 . Vite proxies `/api` to local FastAPI. No CORS change is needed for local development.
5. External deployment: do not use localhost proxy as the production API. Set `VITE_API_BASE_URL` to the public HTTPS backend origin/proxy route; configure allowed origins in FastAPI or use a same-origin reverse proxy. Never put provider or ingestion admin tokens in `VITE_*` variables or browser code.

## Screen-to-endpoint mapping

| Screen | Backend routes |
|---|---|
| Command Center | `GET /health`, `GET /ingestion/health`, `GET /signals/summary`, `GET /signals` |
| Event Intelligence | `GET /signals`, `GET /signals/{id}`, `POST /analyze`, `GET /signals/{id}/module-b-suggestion` |
| Module A | `GET /module-a/index`, `POST /ingestion/module-a/recommend`, `POST /module-a/backtest` |
| Module B | `GET /module-b/portfolio`, `GET /module-b/scenarios`, `POST /module-b/stress`, `POST /module-b/what-if`, `POST /module-b/reverse-stress` |

**Important behavior**
- `GET /ingestion/health` only reports configuration/database status, NOT verified real-time provider connectivity. Actual source successes are shown from `recent_collection_runs`.
- Scores are never populated with synthetic placeholders. Empty or offline sections show empty/error states.
- `GET /signals` is capped at the latest 100 records in this MVP. Event metrics calculated from loaded completed records say so explicitly; they are **not** global historical averages.
- The 15-stock allocations are loaded from `/module-a/index`; recommendations are generated on demand from saved analyzed signals. No trades occur.
- Historical backtest returns are requested from the actual backend `POST /module-a/backtest` only on a manual click; no static fake performance figures.
- Module B scenarios are hypothetical. Issuer placeholders must be explicitly resolved, shock fields are editable, and an acknowledgment is required before stress/what-if/reverse-stress actions.
- Loans: projected incremental 12-month expected credit loss. Bonds, swaps, options: instantaneous mark-to-market. Derivative notional is NOT additive NAV.
- UI uses no admin token and does not expose mutating collector endpoints.
- Current CSS imports Inter via Google Fonts when available and falls back to system fonts when blocked. For fully offline demo, self-host a legally obtained web font or use system fonts.

## API compatibility / integration gate

The source ZIPs used for contract review were versions of Risk Engine, Module A, Module B, and Ingestion add-ons preserved in this conversation. The user's running Antigravity-integrated backend may have subsequent modifications (e.g., Marketaux). **Verify against the actual current `http://127.0.0.1:8000/openapi.json` and live responses before declaring integrated.** The dashboard does not make provider requests directly; all requests go to FastAPI.

### Testing

`npm test`: pure Node unit tests for source safety, numeric display, data counts, issuer placeholder handling.
`npm run check:contracts`: validates our required frontend API route strings and UI design markers.
`npm run build`: TypeScript + Vite build after `npm install`.

These static checks **do not** prove browser rendering, live model inference, deployed API connectivity, or external provider access. Conduct manual browser acceptance tests with the actual backend and a legitimate saved news/social signal.

## Design principles

Obsidian Iris Figma theme: deep charcoal `#10121C`, panels `#191D2B`, violet `#9B8CFF`, amber `#F0B56B`. Distinct sources, auditable model outputs, constrained index analytics and separated risk measurement horizons. Responsive layouts with reduced-motion support. No asset screenshots pasted as fake components.

## Known current limitations

- The frontend is a fully sourced implementation package but is **not certified live** until installed and run with your latest `c2c` checkout.
- The source-provider scheduler runs as a separate backend process; launching the dashboard does not start ingestion.
- Browser-visible history is capped to latest 100 records; add server-side pagination/aggregations if the data grows.
- Reverse-stress and what-if metrics are shown in compact text rather than full interactive plots; chart enhancement can follow real-data acceptance.
- No authentication system is added; the front end assumes the backend or gateway handles access policy for a public deployment.
- The Figma Starter account hit its design-context MCP call limit; the implementation follows the four visually verified editable Figma screens rather than an automated export.
