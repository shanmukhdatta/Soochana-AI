# Reviewer demo deployment — Vercel

This folder supports two **separate** builds; the original full FastAPI-backed application has not been replaced.

| Build | Command | Behavior |
|---|---|---|
| Original application | `npm run build` | Existing API-powered UI. Requires a reachable FastAPI backend for live modules. |
| Static reviewer preview | `npm run build:demo` | Read-only, evidence-backed dashboard. Uses `src/demo-fixtures.json` derived from repository-recorded JSON. No backend, model weights, API keys or inference required. |

## Link Vercel with GitHub

1. Sign into Vercel and choose **Add New > Project**; authorize the Vercel GitHub integration for `shanmukhdatta/nitj-shanmukhdatta-hackthon` if prompted.
2. Import that repository. Set **Root Directory** to `frontend` (important); **Framework Preset** to `Vite`. This folder's `vercel.json` sets `npm run build:demo` and `dist`.
3. Use `master` as the production branch. Deploy and wait for a successful build.
4. Open the generated HTTPS deployment in an incognito browser. Verify Overview, Event Intelligence, Index Lab, Stress Lab, Quick/Analyst toggles, the source links, mobile layout, and all read-only/unavailable indicators.
5. When the URL exists and works, put that **real URL** in the top-level README's currently blank Live Demo Site Link field.

## Integrity and limitations

The preview intentionally does **not** claim live financial news, instant market prices, AI inference, trade execution, new rebalancing calculations, or on-demand stress pricing. It only renders **recorded** observations from:
- `data/module_a/real_example_response.json`
- `data/module_b/live_verification_results.json`
- `data/module_b/synthetic_portfolio.json`
- `data/module_b/scenario_library.json`
- `configs/module_a_config.json`

The exact source-derived snapshot is `frontend/src/demo-fixtures.json`. If a reference JSON changes, update the fixture deliberately, document the new provenance and re-test. Do not call cached numbers “live results.”

The normal application's homepage no longer silently fabricates scores when model inference fails. The original FastAPI backend still needs genuine NLP checkpoints for real inference.

## Local validation

```bash
cd frontend
npm ci
npm test
npm run check:contracts
npm run build
npm run build:demo
npm run dev -- --mode demo
```

Open `http://127.0.0.1:5173`. The Vercel deploy is not complete until all checks pass and the hosted URL has been tested. A green static build does **not** validate the original model runtime.
