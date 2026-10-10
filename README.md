# RiskPulse AI — Explainable Financial Risk Intelligence

**S&P Global & CRISIL Campus Hackathon 2026**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Node.js 20+](https://img.shields.io/badge/Node.js-20%2B-green.svg)](https://nodejs.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/Frontend-React%2018%20%2B%20Vite-61DAFB.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Overview / Problem Statement & Approach

Financial institutions, investment managers, and risk analysts face an overwhelming influx of unstructured textual information across earnings announcements, regulatory filings, financial journalism, and social discourse. Traditional quantitative risk models rely predominantly on structured historical market data (such as closing prices and corporate fundamentals), often reacting to material corporate events only after significant market price discovery has already taken place. Conversely, qualitative research remains manual, labor-intensive, and difficult to scale across thousands of instruments.

**RiskPulse AI** bridges this critical divide by converting unstructured financial textual narratives into interpretable, machine-readable quantitative risk signals and applying them directly to real-world financial decision workflows. Rather than treating sentiment as an isolated scalar metric, RiskPulse AI combines multi-dimensional natural language processing with gradient-boosted decision trees to evaluate event taxonomy classification, continuous sentiment intensity, and market impact severity deciles.

The structured risk signals produced by the engine are connected downstream to two distinct institutional risk laboratories: **Module A (Index Rebalancing)**, which translates event sentiment into constraint-aware portfolio tilts with quadratic turnover control; and **Module B (Multi-Asset Stress Testing)**, which maps adverse event scenarios onto a 30-position institutional balance sheet while rigorously separating trading book mark-to-market re-pricing from banking book 12-month expected credit loss provisioning.

### Core Problem Requirements vs. Implemented Extensions

| Dimension | Core Hackathon Challenge Requirements | RiskPulse AI Implemented Extensions |
|---|---|---|
| **Data Ingestion** | Multiple unstructured information sources | **Dual-source live pipeline:** Marketaux Financial News API + Bluesky Social (AT Protocol) with text normalization, ticker regex extraction, entity aliasing, and SQLite persistence. |
| **Sentiment Analysis** | Sentiment score extraction | **FinBERT Continuous Regressor:** Fine-tuned `ProsusAI/finbert` with linear regression head outputting continuous sentiment intensity $\hat{s} \in [-1.0, +1.0]$. |
| **Event Classification** | Financial event classification | **FinBERT 10-Class Classifier:** 10-way financial event taxonomy (Earnings, Credit Event, Regulatory, Product Launch, M&A, etc.) with softmax probability distributions. |
| **Impact Severity** | Impact severity estimation | **XGBoost Severity Engine:** Gradient-boosted decision trees trained on return volatility quantiles predicting market reaction decile ($1 \dots 10$) conditional on sentiment, event probabilities, 15-day rolling volatility, and sector features. |
| **Output Schema** | Machine-readable results | **Standardized JSON Schema:** Standardized risk signal payloads detailing ticker, confidence, category, decile severity, sentiment intensity, and provenance URL. |
| **Downstream Applications** | At least one financial portfolio application | **Two institutional modules:**<br>• **Module A:** 15-stock constrained quadratic index optimization with 357-day historical walk-forward backtesting.<br>• **Module B:** 30-position multi-asset stress laboratory with What-If exposure adjustment and bisection reverse stress testing. |
| **Explainability & UX** | Reviewer-ready interface | **Apple-inspired Minimalist UI:** Dual view modes (Quick View for executives, Analyst View for quantitative inspection) + **Bujji**, an interactive conversational assistant for explaining model predictions. |

---

## 2. Architecture & Technology Stack

### System Architecture Pipeline

The end-to-end data flow operates through four coordinated stages:

```
Financial News / Public Social Media
                  ↓
Text Normalization & SQLite Event Storage (data/ingestion/riskpulse.db)
                  ↓
Three-Stage Machine Learning Risk Engine:
  ├── 1. FinBERT Sentiment Regressor (Continuous [-1.0, +1.0])
  ├── 2. FinBERT Event Classifier (10-Class Softmax Taxonomy)
  └── 3. XGBoost Impact Severity (Decile 1–10 with Volatility & Sector Features)
                  ↓
Structured Risk Signal (JSON Schema)
                  ↓
Downstream Financial Applications:
  ├── Module A: Constrained Quadratic Index Rebalancing (15 Equities)
  └── Module B: Multi-Asset Stress Laboratory (30 Positions: MTM vs. ECL)
                  ↓
FastAPI Unified Backend Server (Python 3.11 / Uvicorn :8000)
                  ↓
React & TypeScript Dashboard (Vite 6 / React 18 :5173)
```

![RiskPulse AI System Architecture](docs/architecture.png)

### Architectural Components

1. **Ingestion Layer (Deterministic Heuristics & APIs):**
   - `Marketaux News Adapter`: Queries financial news publications, extracts article titles, snippets, entities, and publication timestamps.
   - `Bluesky Social Adapter`: Uses the decentralized AT Protocol to stream public financial discourse matching equity cashtags (`$AAPL`, `$NVDA`, etc.).
   - `Text Normalizer & SQLite Repository`: Strips boilerplate, deduplicates via URL hashing, resolves ticker aliases (e.g., "Google" $\rightarrow$ `GOOGL`), and persists raw records into `data/ingestion/riskpulse.db`.
2. **Three-Stage ML Risk Engine (Trained Machine Learning Models):**
   - **Model 1 (FinBERT Sentiment):** Extracts pooled `[CLS]` embeddings from financial text to project continuous sentiment $\hat{s} \in [-1.0, +1.0]$.
   - **Model 2 (FinBERT Event Classifier):** Evaluates multi-class sequence probabilities across 10 financial event taxonomy categories.
   - **Model 3 (XGBoost Impact Severity):** Combines event probabilities, continuous sentiment, historical 15-day rolling price volatility ($\sigma_{15}$ via `yfinance`), and one-hot sector indicators to predict market severity decile ($1 \dots 10$).
3. **Module A — Index Rebalancing Laboratory (Constrained Convex Optimization):**
   - Formulates a quadratic optimization problem balancing target alpha tilts against turnover transaction costs.
   - Enforces strict institutional constraints: full investment ($\sum w_i = 1.0$), long-only ($w_i \ge 0$), 15% single-stock cap, 35% sector concentration limit, and 5% maximum turnover budget per rebalance.
4. **Module B — Multi-Asset Stress Laboratory (Quantitative Financial Valuation):**
   - Evaluates a 30-position institutional portfolio ($10.0M total balance sheet).
   - Separates **Trading Book Mark-to-Market (MTM)** (second-order duration/convexity for bonds, DV01 curve re-pricing for swaps, Black-Scholes Delta-Gamma for options) from **Banking Book Expected Credit Loss (ECL)** ($EAD \times PD \times LGD$ under Basel II/III).
   - Provides interactive **What-If** position scaling and **Reverse Stress Testing** via bisection root-finding.
5. **API & Interface Layer:**
   - **FastAPI:** Lifespan model management, CORS middleware, Pydantic validation, atomic batch endpoints.
   - **React 18 + TypeScript + Vite:** Minimalist monochrome design system, state synchronization across Quick and Analyst views, responsive layout (mobile 390px to 1600px desktop).

---

## 3. Dataset Used

RiskPulse AI incorporates specific datasets across training, real-source ingestion, historical market prices, and portfolio stress modeling:

### 1. Model Training & Evaluation Data
- **Base Transformer:** `ProsusAI/finbert` (licensed under Apache 2.0), pre-trained on the Financial PhraseBank corpus.
- **Sentiment & Event Taxonomy Fine-Tuning:** Fine-tuned on labeled financial headlines and corporate event disclosures categorized into 10 standard financial taxonomy classes.
- **Impact Severity Feature Matrix:** Calibrated on historical corporate event filings matched with subsequent 1-day and 5-day equity return quantiles to define decile impact boundaries (`configs/impact_decile_thresholds.json`).
- *Note:* Due to GitHub's 100 MB per-file upload limit, raw 438 MB transformer weights (`model.safetensors`) are hosted externally, while all model configurations, tokenizers, vocabularies, and the XGBoost severity engine (`models/impact_engine/*.joblib`, 535 KB) are tracked in the repository.

### 2. Live Public News & Social Ingestion
- **Marketaux News API:** Real-time financial news headlines and metadata. Free-tier API token required for live fetching; 2 pre-collected news items are bundled in `data/ingestion/riskpulse.db`.
- **Bluesky Social Network (AT Protocol):** Public financial social media posts and commentary matching supported tickers; 27 pre-collected public posts are bundled in `data/ingestion/riskpulse.db`.
- *Copyright Compliance:* No full-text copyrighted news articles are stored or redistributed. The repository stores only headline metadata, snippets, ticker tags, and public canonical source URLs.

### 3. Historical Market Price Data
- **Historical Daily Prices:** Historical adjusted closing prices for the 15 supported equities retrieved via `yfinance` for volatility calculation.
- **Offline Walk-Forward Holdout Data:** Bundled holdout market prices (`data/module_a/holdout_prices.csv`, 188 KB) and holdout signal series (`data/module_a/holdout_signals.csv`, 624 KB) covering 357 market sessions for offline reproducibility of Module A backtests without network access.

### 4. Synthetic Institutional Portfolio Data
- **Synthetic Institution Balance Sheet (`data/module_b/synthetic_portfolio.json`):** Calibrated 30-position institutional portfolio totaling $10,000,000 exposure across 10 commercial loans ($5.0M EAD), 10 corporate bonds ($3.5M MV), 5 interest rate swaps ($1.0M Notional), and 5 FX options ($0.5M Notional).
- **Scenario Library (`data/module_b/scenario_library.json`):** 5 historical and hypothetical stress scenario families (Credit Event, Macroeconomic, Geopolitical, Merger / Acquisition, Product Launch) specifying quantitative shock vectors for risk-free rates, credit spreads, equity indices, FX rates, and loan default probabilities.

---

## 4. Quickstart & Installation

### Prerequisites
- **Python:** 3.11 or 3.12 (Tested on 3.11.9)
- **Node.js:** 20.x or later (Tested on Node 20.18)
- **Git:** 2.40+

### Step 1: Clone the Repository
```bash
git clone https://github.com/<college>-<candidate-name>-hackathon.git
cd <college>-<candidate-name>-hackathon
```

### Step 2: Backend Setup
Create and activate a clean Python virtual environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install backend dependencies:
```bash
pip install -r requirements.txt
```

Set up environment variables:
```bash
# Copy example configuration template
cp .env.example .env
```
*Note:* The default `.env` is configured to run out-of-the-box. The local SQLite database (`data/ingestion/riskpulse.db`) contains 29 pre-loaded news and social signals, meaning the system starts and operates without requiring external API keys.

Verify model configurations:
```bash
python -m scripts.check_models
```

### Step 3: Frontend Setup
In a separate terminal, install dependencies:
```bash
cd frontend
npm install
cd ..
```

### Step 4: Launching the Application

**Terminal 1 — Backend Server:**
```bash
uvicorn app.main:app --reload --port 8000
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive Swagger Documentation: `http://127.0.0.1:8000/docs`
- Health Endpoint: `http://127.0.0.1:8000/health`

**Terminal 2 — Frontend Application:**
```bash
npm --prefix frontend run dev
```
- Web Application: `http://127.0.0.1:5173`

### Step 5: Verification & Automated Tests
Execute the full test verification suite:

```bash
# 1. Run all 120 backend unit & integration tests
pytest -q

# 2. Run frontend unit tests (13 tests)
npm --prefix frontend test

# 3. Verify frontend-backend API contracts
npm --prefix frontend run check:contracts

# 4. Compile frontend production bundle
npm --prefix frontend run build

# 5. Run end-to-end integration audit (requires backend and frontend running)
node frontend/scripts/audit-e2e.mjs
```

---

## 5. Key Results & Domain Impact

All metrics below reflect genuine, measured outputs from automated verification suites:

### Verified Implementation Results

| Component | Verified Capability | Supporting Evidence |
|---|---|---|
| **Risk Engine Health** | Model orchestration, lifespan startup, atomic batching | `GET /health` returns `status: "ready"`; 120/120 passing pytest suite. |
| **Three-Model Inference** | Real-time sentiment regression, event classification, impact decile | Deterministic manual inference: Product Launch (99.7% conf), Sentiment +0.606, Impact 6/10. |
| **Dual-Source Ingestion** | Live & pre-collected ingestion from news (Marketaux) and social (Bluesky) | 29 active signals (2 news, 27 social) stored in `data/ingestion/riskpulse.db`. |
| **Module A Rebalancing** | 15-stock quadratic optimization subject to 5 strict regulatory constraints | Target weights sum to exactly $1.0000$; maximum position $\le 15\%$; sector $\le 35\%$; turnover $\le 5\%$. |
| **Module A Backtesting** | Walk-forward simulation over 357 historical market sessions | `POST /module-a/backtest` executes across 357 market sessions without lookahead leakage. |
| **Module B Stress Testing** | Horizon-separated valuation across 30 institutional positions | Instantaneous MTM P&L ($+\$2,892.61$) and Incremental 12m ECL ($\$391.50$) computed under reviewed scenarios. |
| **What-If Analysis** | Interactive sensitivity adjustment on position exposures | Position L01 adjustment calculates exact net portfolio delta ($\$78.30$). |
| **Reverse Stress Testing** | Bisection search for capital breach threshold | Bisection root-finder converges on threshold scalar within $\epsilon = 10^{-4}$. |
| **Bujji Assistant** | In-context financial education and model explainability | Free-tier Groq Qwen/Llama integration with deterministic offline fallback modes. |

### Domain Impact for Financial Institutions
- **Early Warning Risk Signals:** Identifies market-moving corporate events from public social chatter and news before they manifest in closing volatility.
- **Regulatory-Compliant Tilting:** Module A demonstrates how qualitative risk intelligence can be harnessed without violating strict concentration or turnover mandates.
- **Accounting Horizon Integrity:** Module B resolves the common industry pitfall of aggregating trading book market losses and banking book credit allowances into a misleading single sum, providing CROs and portfolio managers with transparent risk insights.

---

## 6. Application Walkthrough

Reviewers can experience the full capabilities of RiskPulse AI through the following guided workflow:

```
[Home Introduction] ──→ [Live Signals View] ──→ [Event Intelligence]
                                 │
                                 ├──→ [Module A: Index Lab]
                                 └──→ [Module B: Stress Laboratory]
                                                │
                                                └──→ [Analyst View & Bujji Assistant]
```

### Guided Demonstration Steps

1. **Home Introduction (`/#home`):**
   - High-level overview of the financial risk intelligence platform, key system metrics, and architectural workflow.
   - Click **"Launch Platform"** to transition to the live intelligence dashboard.
2. **Live AI Signals & Provenance (`/?mode=quick#events`):**
   - Review 29 ingested news and social signals with color-coded sentiment bars, detected event badges, and estimated impact severity deciles ($1 \dots 10$).
   - Click source external links (e.g., SeekingAlpha, Bluesky) to inspect the raw narrative origin with strict `rel="noopener noreferrer"` security.
3. **Module A — Index Rebalancing Laboratory (`/#moduleA`):**
   - Review the 15-stock baseline equal-weighted index ($w_i = 6.67\%$).
   - Click **"Run Signal-Based Rebalance"** to execute the quadratic optimizer, observing how positive/negative signals tilt allocations within the 5% turnover budget.
   - Run the 357-day historical backtest to observe cumulative return profiles.
4. **Module B — Multi-Asset Stress Laboratory (`/#moduleB`):**
   - Inspect the 30-position portfolio breakdown ($5.0M Loans, $3.5M Bonds, $1.5M Derivatives).
   - Select a scenario family (e.g., *Credit Event* or *Macroeconomic*) and click **"Run Stress Test"**.
   - Observe the separation between **Instantaneous Mark-to-Market P&L** and **Incremental 12-Month Expected Credit Loss**.
   - Modify position exposure sliders in the **What-If Analysis** panel to evaluate hedging strategies.
   - Run the **Reverse Stress Test** to determine the minimum macroeconomic shock that would breach institution capital buffers.
5. **Analyst View & Bujji Assistant:**
   - Toggle the top-bar switch from **Quick View** to **Analyst View** to access raw feature weights, parameter sliders ($\lambda$ turnover aversion, capital buffer thresholds), and loss breakdowns.
   - Click the floating circular assistant icon at the bottom-right to open **Bujji**, and ask financial questions such as *"Why are MTM and ECL reported separately?"* or *"Explain how the 15% single-stock cap works."*

---

## 7. Limitations & Responsible Use

To maintain scientific and financial integrity, the known limitations of RiskPulse AI are documented below:

1. **Model Uncertainty & Misclassification:** NLP classification models, while achieving high benchmark accuracy, can misclassify ambiguous or sarcastic financial text. Predictions should be utilized as quantitative risk intelligence to support human analysts, not as autonomous decision-makers.
2. **Impact Severity Interpretation:** Predicted impact severity deciles ($1 \dots 10$) reflect estimated return volatility and market reaction magnitudes; they do not predict exact directional return percentages or target prices.
3. **Provider Rate Limits:** Live news ingestion depends on third-party API availability and tier quotas (e.g., Marketaux free tier). When external connectivity is unavailable, the platform automatically utilizes its offline SQLite repository.
4. **Historical Simulation Assumptions:** Module A backtest results are based on historical daily market data. Past performance is not indicative of future returns, and backtests assume standard transaction cost models.
5. **Synthetic Portfolio Scope:** Module B operates on a 30-position calibrated synthetic portfolio designed for research demonstration rather than an enterprise core banking ledger.
6. **Non-Advisory Disclaimer:** RiskPulse AI is an academic and research prototype developed for the S&P Global & CRISIL Campus Hackathon 2026. **It does not provide personalized investment advice, trading recommendations, or financial advisory services.**

---

## 8. Demo, Presentation & Repository Links

- **Live Application Deployment:** `Deployment pending`
- **Official Demo Video (10-minute walkthrough):** `Pending recording`
- **Slide Presentation (5–7 slides):** `docs/presentation.pdf` *(Pending final candidate deck export)*
- **System Architecture Diagram:** [`docs/architecture.png`](docs/architecture.png)
- **Detailed Methodology & Mathematics:** [`docs/methodology.md`](docs/methodology.md)
- **Validation & Test Audit Report:** [`docs/validation.md`](docs/validation.md)

---

## 9. Author & Acknowledgments

### Project Submission Details
- **Candidate Full Name:** `[Candidate Full Name]`
- **Institution / College:** `[College Name]`
- **Official College Email:** `[college.email@domain.edu]`
- **Recommended Repository Name:** `<college>-<candidate-name>-hackathon`

### Acknowledgments & Attribution
- **ProsusAI FinBERT:** Developed by ProsusAI and hosted on Hugging Face (`ProsusAI/finbert`), licensed under Apache 2.0.
- **XGBoost:** Distributed Machine Learning Community (DMLC), licensed under Apache 2.0.
- **Hugging Face Transformers & PyTorch:** Open-source deep learning frameworks.
- **Marketaux & Bluesky:** Public news and social media APIs.
- **S&P Global & CRISIL:** Organizers of the Campus Hackathon 2026.

*Disclaimer:* This project is an independent submission for the S&P Global & CRISIL Campus Hackathon 2026. It is not officially endorsed by, affiliated with, or sponsored by S&P Global, CRISIL, Meta, Apple, or Marketaux.
