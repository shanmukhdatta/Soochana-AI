# Soochana AI — Institutional Financial Risk Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%2018%20%2B%20TypeScript-61DAFB.svg?logo=react)](https://react.dev/)
[![Docker](https://img.shields.io/badge/DevOps-Docker%20%26%20Compose-2496ED.svg?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Models-Hugging%20Face%20Hub-yellow.svg)](https://huggingface.co/shanmukhdatta)

> **From Market Information to Financial Decisions.**  
> An institutional-grade machine learning platform combining fine-tuned transformer NLP, gradient-boosted volatility forecasting, convex portfolio index rebalancing, and horizon-separated multi-asset balance sheet stress testing.

**Author:** [Shanmukh Datta](https://github.com/shanmukhdatta) (National Institute of Technology Jalandhar)  
**Repository:** [https://github.com/shanmukhdatta/Soochana-AI](https://github.com/shanmukhdatta/Soochana-AI)  
**Live Application:** [https://nitj-shanmukhdatta-hackthon-lime.vercel.app/](https://nitj-shanmukhdatta-hackthon-lime.vercel.app/)

---

## 1. Executive Summary & Problem Formulation

### The Financial Challenge
Institutional portfolio managers and quantitative risk officers face an overwhelming flood of unstructured data — regulatory disclosures, financial news headlines, and public market discussions. Traditional market systems present price-only telemetry without underlying narrative attribution, while manual qualitative review introduces latency and cognitive bias. Furthermore, converting qualitative event intelligence into disciplined, constraint-aware portfolio allocation and balance sheet stress scenarios requires rigorous mathematical formulation.

### The Soochana AI Solution
Soochana AI bridges the gap between unstructured textual information and quantitative decision-making:
1. **Multi-Source Signal Ingestion:** Collects and normalizes live feeds from **Marketaux** financial news and **Bluesky (AT Protocol)** public social discourse with ticker extraction and deduplication into a local SQLite repository.
2. **Three-Stage Machine Learning Risk Engine:**
   - **FinBERT Event Classifier:** Categorizes text into 5 institutional event classes (*Geopolitical, Macroeconomic, Credit Event, Merger / Acquisition, Product Launch*).
   - **FinBERT Sentiment Regressor:** Predicts continuous polarity on $[-1.0, +1.0]$ via a hyperbolic tangent regression head.
   - **XGBoost Market Impact Engine:** Evaluates abnormal market volatility magnitude deciles ($1 \dots 10$) using zero-lookahead pre-publication market indicators ($\sigma_{20}$, volume ratio, momentum) and frozen NLP outputs.
3. **Module A — Tactical Index Rebalancing Laboratory:** Translates structured event signals into convex quadratic distance minimization rebalancing for a 15-stock US equity index subject to constituent, sector, turnover, and transaction cost constraints across 357 walk-forward backtest sessions.
4. **Module B — Multi-Asset Stress Testing Laboratory:** Evaluates a 30-position synthetic institutional portfolio across 4 asset classes (Loans, Bonds, Swaps, Options) with auditable separation between instantaneous Trading Book Mark-to-Market (MTM) and Banking Book 12-Month Expected Credit Loss (ECL).
5. **Bujji — Embedded Conversational Risk Copilot:** An intelligent assistant powered by Groq API (`qwen/qwen3.8-27b`) dynamically tailored for both executive Quick View and quantitative Analyst View.
6. **Custom Portfolio Studio & Retail Investor Shield:** Empowers individual investors to input their personal stock holdings ($ / ₹), stream live ticker news, filter emotional clickbait chatter from genuine balance sheet threats, receive convex quadratic rebalancing recommendations, and simulate everyday macroeconomic crisis stress scenarios with an institutional-grade 0–100 Portfolio Health Score.

---

## 2. End-to-End System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    TIER 1: INGESTION & DATA LAYER                                │
│                                                                                                 │
│    Marketaux Financial News API                     Bluesky Public Firehose (AT Protocol)        │
│                │                                                       │                        │
│                └───────────────────────────┬───────────────────────────┘                        │
│                                            ▼                                                    │
│                        Ticker Resolution & Exact SHA-256 Deduplication                          │
│                                            │                                                    │
│                                            ▼                                                    │
│                           SQLite Ingestion Store (`riskpulse.db`)                               │
└────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                             │ Unstructured Text Payload
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 TIER 2: UNIFIED AI RISK ENGINE (`app/engine/`)                  │
│                                                                                                 │
│  ┌───────────────────────────────────┐               ┌───────────────────────────────────────┐  │
│  │   FinBERT Event Classifier        │               │   FinBERT Sentiment Regressor         │  │
│  │   • 5-way softmax probabilities   │               │   • Continuous polarity s ∈ [-1, +1]  │  │
│  │   • 99.46% test accuracy          │               │   • Tanh regression head on CLS       │  │
│  └─────────────────┬─────────────────┘               └───────────────────┬───────────────────┘  │
│                    │                                                     │                      │
│                    └──────────────────────┬──────────────────────────────┘                      │
│                                           ▼                                                     │
│                        Pre-Publication Historical Market Features                               │
│                        • σ_20 annualized volatility (no lookahead)                              │
│                        • 20-session volume ratio (V_t / V̄_20)                                   │
│                        • Trailing 5d & 20d momentum returns                                     │
│                                           │                                                     │
│                                           ▼                                                     │
│                        XGBoost Return Impact Severity Engine                                    │
│                        • Trained abnormal reaction magnitude |R_abnormal|                       │
│                        • Calibrated Empirical Quantile Deciles: [1 ... 10]                      │
└───────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                            │ Structured Risk Signal Record
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             TIER 3: QUANTITATIVE FINANCIAL LABORATORIES                          │
│                                                                                                 │
│  ┌─────────────────────────────────────────────────┐ ┌────────────────────────────────────────┐ │
│  │        MODULE A: TACTICAL INDEX REBALANCING     │ │       MODULE B: MULTI-ASSET STRESS LAB │ │
│  │  • 15 US Equities (Equal-weight baseline)       │ │ • 30 Synthetic Institutional Positions │ │
│  │  • Temporal signal decay: τ = 3.0 days          │ │ • Banking Book (12-Month Loan ECL)     │ │
│  │  • SLSQP Quadratic Distance Minimization:       │ │ • Trading Book (Bonds, Swaps, Options) │ │
│  │    min 1/2 ||w - w_desired||^2                  │ │ • Strict Horizon Separation (MTM vs ECL│ │
│  │    subject to:                                  │ │ • Interactive What-If Position Scaling │ │
│  │    - Full Investment: ∑ w_i = 1.0               │ │ • Reverse Stress Breach Search on Grid │ │
│  │    - Position deviation: |Δw| ≤ 2.0%            │ └────────────────────────────────────────┘ │
│  │    - Sector deviation: |ΔS_k| ≤ 5.0%            │                                            │
│  │    - One-way turnover: T ≤ 10.0%                │                                            │
│  │    - Transaction cost: 10 bps friction          │                                            │
│  │  • 357-session historical walk-forward backtest │                                            │
│  └─────────────────────────────────────────────────┘                                            │
└───────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                            │ API Responses (FastAPI)
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              TIER 4: PRODUCTION PRESENTATION & COPILOT                          │
│                                                                                                 │
│    React 18 + TypeScript Dashboard (Vite 6)          Bujji AI Assistant (Groq Cloud API)        │
│    • Quick View: Executive summaries & key metrics   • Context-aware risk copilot               │
│    • Analyst View: Probability vectors & audit       • Dual audience mode adapting prompts      │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Machine Learning Models & Weights

Trained model checkpoints are hosted on Hugging Face Hub:

| Model | Task & Output | Foundation Backbone | Training Data | Hugging Face Checkpoint |
| :--- | :--- | :--- | :--- | :--- |
| **Event Classifier** | 5-Class Categorical Softmax | `nickmuchi/finbert-tone-finetuned-finance-topic-classification` | 16,576 cleaned records across 6 datasets | [`shanmukhdatta/riskpulse-event-classifier`](https://huggingface.co/shanmukhdatta/riskpulse-event-classifier) |
| **Sentiment Regressor** | Continuous Scalar Polarity $[-1.0, +1.0]$ | `ProsusAI/finbert` + custom Tanh head | FiQA Financial Sentiment Challenge | [`shanmukhdatta/riskpulse-sentiment-regressor`](https://huggingface.co/shanmukhdatta/riskpulse-sentiment-regressor) |
| **Impact Engine** | Volatility Severity Decile $[1 \dots 10]$ | `xgboost.XGBRegressor` + ColumnTransformer | 27,460 pre-publication market feature rows | [`shanmukhdatta/riskpulse-impact-engine`](https://huggingface.co/shanmukhdatta/riskpulse-impact-engine) |

Detailed model architecture specifications and reproduction guides:
- [Event Classifier Architecture & Training Guide](training/event_classifier/README.md)
- [Sentiment Regressor Architecture & Training Guide](training/sentiment/README.md)
- [Market Impact Engine Architecture & Training Guide](training/impact_engine/README.md)
- [Unified Risk Engine Runtime Specification](backend/app/engine/README.md)

---

## 4. Quantitative Laboratories

### Module A: Tactical Index Rebalancing
- **Universe:** 15 large-cap US equities (`AAPL`, `MSFT`, `NVDA`, `QCOM`, `GOOGL`, `CMCSA`, `AMZN`, `TSLA`, `BKNG`, `SBUX`, `COST`, `PEP`, `GILD`, `AMGN`, `CSX`).
- **Optimization:** SciPy Sequential Least Squares Programming (SLSQP).
- **Constraints:**
  - Full investment ($\sum w_i = 1$) and long-only ($w_i \ge 0$).
  - Constituent deviation cap: $\pm 2.0\%$ from equal-weight baseline.
  - Sector deviation cap: $\pm 5.0\%$ from sector baseline.
  - One-way turnover limit: $\le 10.0\%$ per rebalance.
  - Transaction friction: 10 basis points linear deduction.
- **Deep Dive:** [Module A Architectural Specification](backend/app/module_a/README.md)

### Module B: Multi-Asset Stress Testing
- **Portfolio Composition:** 30 synthetic institutional positions:
  - **12 Commercial Loans:** Banking Book 12-Month Expected Credit Loss ($\Delta \text{ECL} = \text{EAD} \times \text{LGD} \times \Delta\text{PD}$).
  - **10 Corporate Bonds:** Trading Book MTM via modified duration and spread convexity ($\Delta P \approx -D_{\text{mod}} \Delta y + \frac{1}{2} C (\Delta y)^2$).
  - **4 Interest Rate Swaps:** Trading Book MTM via linear DV01 curve shifts ($\Delta \text{MTM} \approx \text{DV01} \cdot \Delta r$).
  - **4 Equity Options:** Trading Book MTM via European Black-Scholes analytical revaluation under price, volatility, and rate shifts.
- **Horizon Separation:** Strict segregation between instantaneous Trading Book Mark-to-Market P&L and Banking Book 12-Month Expected Credit Loss.
- **Deep Dive:** [Module B Architectural Specification](backend/app/module_b/README.md)

### Bujji: Conversational Financial AI Copilot
- Context-aware risk assistant powered by Groq API (`qwen/qwen3.8-27b`).
- Dynamically adapts tone and mathematical complexity between Quick View and Analyst View.
- Incorporates strict financial accuracy and anti-hallucination guardrails.
- **Deep Dive:** [Assistant Architectural Specification](backend/app/assistant/README.md)

---

## 5. Repository Structure

```text
.
├── backend/                    # Production FastAPI backend service
│   ├── app/                    # Application package
│   │   ├── engine/             # Unified 3-model risk engine (NLP + Features + Impact)
│   │   ├── module_a/           # Tactical index rebalancing & SLSQP optimizer
│   │   ├── module_b/           # Multi-asset stress testing & analytical pricing
│   │   ├── ingestion/          # Multi-source adapters (Marketaux, Bluesky) & SQLite
│   │   ├── assistant/          # Bujji conversational risk assistant (Groq API)
│   │   ├── services/           # Zero-lookahead feature builder & price providers
│   │   ├── main.py             # FastAPI application factory & lifecycle manager
│   │   ├── config.py           # Pydantic Settings schema
│   │   └── schemas.py          # Global API request/response models
│   ├── tests/                  # Automated PyTest test suite (120 test cases)
│   ├── Dockerfile              # Production backend container build
│   ├── requirements.txt        # Backend Python dependencies
│   ├── pytest.ini              # Backend test runner configuration
│   └── README.md               # Backend service documentation
├── frontend/                   # React 18 + TypeScript + Vite 6 Dashboard
│   ├── src/
│   │   ├── components/         # Quick View, Analyst View, Bujji Chat, Lab Views
│   │   ├── demo-data.ts        # Seed demonstration state & client-side fallbacks
│   │   └── App.tsx             # Root React dashboard shell
│   ├── nginx.conf              # Production Nginx reverse proxy configuration
│   ├── Dockerfile              # Multi-stage production container build
│   └── README.md               # Frontend client documentation
├── training/                   # Model training sources, notebooks & reproduction
│   ├── event_classifier/       # 5-class FinBERT event classifier training
│   ├── sentiment/              # Continuous FinBERT sentiment regression training
│   ├── impact_engine/          # XGBoost market impact training
│   └── README.md               # Model training documentation
├── models/                     # Tokenizer assets & downloaded trained model weights
├── configs/                    # Sector taxonomies, label maps, decile thresholds, reference
├── data/                       # Bundled demonstration datasets (SQLite, holdouts)
├── scripts/                    # CLI backtest runners, verification, model fetcher
├── .github/workflows/          # Continuous Integration CI pipelines
├── docker-compose.yml          # Full-stack container orchestration
├── requirements.txt            # Top-level Python dependencies
└── README.md                   # Project documentation
```

---

## 6. Quickstart & Installation

### Option 1: Docker Compose (Recommended Production Setup)

Run the entire platform (FastAPI backend + Nginx React frontend) with a single command:

```bash
# 1. Clone repository
git clone https://github.com/shanmukhdatta/Soochana-AI.git
cd Soochana-AI

# 2. Configure environment
cp .env.example .env

# 3. Launch multi-container stack
docker compose up --build
```

- **Frontend Dashboard:** [http://localhost:3000](http://localhost:3000)
- **Backend API & Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Option 2: Local Development Setup

#### Prerequisites
- Python 3.11 or 3.12
- Node.js 20+ and npm
- Git

#### 1. Backend Setup
```bash
# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
cp .env.example .env
```

#### 2. Model Checkpoint Download
Download the trained transformer and impact models from Hugging Face Hub:
```bash
python -m scripts.fetch_models
python -m scripts.check_models
```

*(Note: If operating in lightweight demonstration mode without GPU/model downloads, set `SKIP_MODEL_LOAD=true` in `.env` to run portfolio labs independently).*

#### 3. Frontend Setup
```bash
cd frontend
npm ci
npm run dev
```

#### 4. Start Backend Server
In a separate terminal (with virtual environment active):
```bash
python -m uvicorn app.main:app --reload --port 8000
```

- Local Dashboard: [http://127.0.0.1:5173](http://127.0.0.1:5173)
- API Interactive Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 7. Automated Verification & Testing

The platform enforces extensive verification across backend unit tests, frontend logic audits, and API contract bounds:

```bash
# Run backend test suite (120 automated tests)
pytest -q

# Run frontend test suite
npm --prefix frontend test

# Run API contract validation
npm --prefix frontend run check:contracts

# Compile TypeScript and build production frontend bundle
npm --prefix frontend run build
```

*For comprehensive test coverage reports and audit logs, see [`docs/validation.md`](docs/validation.md).*

---

## 8. Continuous Integration & DevOps

The repository includes automated GitHub Actions workflows ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) validating every pull request:
1. **Backend Matrix:** Installs dependencies and executes all 120 PyTest tests under Python 3.11.
2. **Frontend Matrix:** Validates Node 20 runtime, runs node test runner, audits API contracts, and compiles TypeScript bundles.
3. **Container Linting:** Verifies Dockerfile specifications and validates `docker compose config`.

---

## 9. Research & Responsible Use Notice

**Educational & Research Disclaimer:** Soochana AI is an independent academic research platform developed by Shanmukh Datta (NIT Jalandhar). The risk signals, index rebalancing weights, and stress testing models are mathematical prototypes designed for risk analytics exploration. They do not constitute personalized investment advice, brokerage trading instructions, or certified regulatory capital calculations.

---

## 10. License

This project is licensed under the terms of the [MIT License](LICENSE).
