# RiskPulse AI — Signal Atlas React Dashboard

**RiskPulse AI Dashboard** is a clean, modern, white financial risk intelligence workspace built with Vite, React 18, and TypeScript. It features a dual-mode interface designed for both executive screening (**Quick View**) and quantitative audit (**Analyst View**), as well as the **Bujji Financial AI Assistant**.

**Scope:** Dedicated frontend client communicating with the RiskPulse AI FastAPI backend. It does not replace the backend, retrain machine learning models, ingest external providers directly in the browser, or execute live exchange transactions.

---

## Quick Start (Windows PowerShell)

With FastAPI running at `http://127.0.0.1:8000`, run:

```powershell
cd nitj-shanmukhdatta-hackthon\frontend
Copy-Item .env.example .env
npm ci
npm test
npm run check:contracts
npm run build
npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` directly to local FastAPI (`http://127.0.0.1:8000`). No CORS modifications are needed for local development.

### External / Production Deployment
For cloud deployment, configure `VITE_API_BASE_URL` to point to the public HTTPS backend origin or deploy behind a reverse proxy. Never expose provider tokens or ingestion secrets in frontend environment variables.

---

## Screen-to-Endpoint Architecture

| Section | Key Endpoints | Description |
|---|---|---|
| **Home Landing Page** | — | Interactive walkthrough, live model inference demo, visual architecture map, and view mode guide. |
| **AI Signals & Ingestion** | `GET /signals`<br>`GET /signals/{id}`<br>`POST /analyze`<br>`GET /ingestion/health` | Multi-source feed (financial news & social media), 5-class event taxonomy distribution, continuous sentiment score, impact decile, and manual text analysis. |
| **Module A: Index Lab** | `GET /module-a/index`<br>`POST /ingestion/module-a/recommend`<br>`POST /module-a/backtest` | 15-stock constrained index optimization with quadratic distance minimization, ±2% stock deviation, ±5% sector deviation, 10% maximum one-way turnover, and 357-session historical backtesting. |
| **Module B: Stress Lab** | `GET /module-b/portfolio`<br>`GET /module-b/scenarios`<br>`POST /module-b/stress`<br>`POST /module-b/what-if`<br>`POST /module-b/reverse-stress` | 30-position synthetic institutional portfolio (12 loans, 10 bonds, 4 swaps, 4 options). Instantaneous MTM vs. 12-month incremental ECL separation, interactive What-If sliders, and automated Reverse Stress testing. |
| **Bujji Financial Assistant** | Embedded floating assistant | Context-aware risk assistant explaining financial terms, model outputs, and portfolio risk metrics. |

---

## Design System & Interface Modes

### Dual Audience Modes
- **Quick View Mode:** Executive and portfolio manager perspective. Streamlines cards, highlights high-level summary totals, provides 1-click optimization recommendations and macro scenario executions, and tucks away granular probability arrays.
- **Analyst View Mode:** Quantitative analyst and auditor perspective. Unveils complete 5-class softmax probability distributions, continuous sentiment intensities $[-1.0, +1.0]$, 20-day rolling price volatility ($\sigma_{20}$), mathematical constraint checks, interactive What-If position sliders ($0.0\times$ to $2.0\times$), and automated Reverse Stress bisection search.

### Modern Clean Aesthetics
- **Palette:** Minimalist bright white canvas (`#ffffff` / `#f8fafc`), neutral slate typography (`#0f172a`, `#475569`), subtle cool borders (`#e2e8f0`), and crisp risk badge accents (green for positive, amber for moderate, crimson for severe).
- **Responsive Layout:** Adaptive breakpoints at 1600px, 1440px, 1024px, 768px, and 390px, with `prefers-reduced-motion` accessibility support.
- **Auditable Financial Distinctions:** Distinct visual badges for Trading Book instantaneous MTM vs. Banking Book 12-month ECL. Clear disclaimers that derivative notional is not additive Net Asset Value (NAV).

---

## Automated Verification Suite

Run the verified test and contract checks:
```bash
# Unit and logic tests (14 tests)
npm test

# API contract and route declaration checks (15 routes verified)
npm run check:contracts

# Production bundle packaging (TypeScript + Vite)
npm run build
```
