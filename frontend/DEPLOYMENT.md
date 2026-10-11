# Vercel Deployment & Local Model Runtime Guide

This repository is configured to support two modes without code divergence:
1. **Cloud Preview Deployment (Vercel)**:
   - Deploys the full web application without requiring heavy Python ML runtimes (PyTorch, FinBERT checkpoints, XGBoost models, and external market API keys).
   - All modules, interactive controls, stress testing simulations, what-if exposure calculators, reverse-stress testing, holdout backtests, and the Bujji AI financial assistant are fully operational and interactive using calibrated research models and verified repository evidence (`src/demo-fixtures.json`).
   - Automatically detected on any `*.vercel.app` domain or when built with `npm run build:demo`.
2. **Local Full-Stack Runtime (FastAPI + Machine Learning)**:
   - When running locally (`uvicorn app.main:app` on port 8000 + `npm run dev` in `frontend/`), the frontend automatically connects to the live FastAPI backend via proxy.
   - Executes real PyTorch FinBERT event classification, sentiment scoring, XGBoost impact regression, and live portfolio rebalancing.

---

## Deploy to Vercel (Step-by-Step)

### Option 1: Direct GitHub Import (Simplest)
1. Sign in to your [Vercel Dashboard](https://vercel.com).
2. Click **Add New... > Project** and select the repository: `shanmukhdatta/nitj-shanmukhdatta-hackthon`.
3. In the project configuration:
   - **Framework Preset**: `Vite` (automatically detected)
   - **Root Directory**: Select `frontend` (or leave as root, as both root and `frontend/` have configured `vercel.json` files).
   - **Build Command**: `npm run build:demo`
   - **Output Directory**: `dist`
4. Click **Deploy**.
5. Once deployment completes, your live application is available at `https://<your-project>.vercel.app`.

### Option 2: Vercel CLI
```bash
# Install Vercel CLI if needed
npm install -g vercel

# From the repository root:
vercel --prod
```

---

## Local Development & Validation

### Running with Full Local Models
```bash
# 1. Start Python FastAPI backend (Port 8000)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 2. In another terminal, start frontend
cd frontend
npm install
npm run dev
```
Open `http://127.0.0.1:5173`. Requests to `/api/*` are proxied to the live FastAPI backend.

### Running in Cloud Demo Mode Locally
```bash
cd frontend
npm run dev -- --mode demo
```

### Running Test Suite & Contract Checks
```bash
cd frontend
npm test
npm run check:contracts
npm run build:demo
```
