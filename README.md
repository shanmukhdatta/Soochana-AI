# RiskPulse AI — S&P Global & CRISIL Campus Hackathon 2026

**Project:** Explainable Financial Risk Intelligence  
**Candidate Name:** Shanmukh Datta  
**College Email ID:** bodasd.ic.24@nitj.ac.in  
**College / Campus:** National Institute of Technology Jalandhar (NIT Jalandhar)  
**Demo Video Link:**  
**Slide Deck Link:** [Repository PDF (docs/Presentation_.pdf)](docs/Presentation_.pdf) · [Google Drive Link](https://drive.google.com/file/d/1ySZRghGGWxB0SQZsvHdQ40yYgSAaQkuJ/view?usp=sharing)  
**Live Demo Site Link:** https://nitj-shanmukhdatta-hackthon-lime.vercel.app/#how-it-works  

**Repository:** https://github.com/shanmukhdatta/nitj-shanmukhdatta-hackthon

RiskPulse AI is an academic prototype for financial event intelligence, constraint-aware index rebalancing, and synthetic multi-asset stress analysis. It is not an investment-advice product or a regulatory risk system.

---

## 1. Project Overview / Problem Statement & Approach

**The problem.** Financial analysts must interpret high volumes of financial news, company announcements, and public market discussion. These unstructured sources can be difficult to connect consistently with quantitative market context and portfolio-level risk analysis. Price-only dashboards do not explain the underlying event narrative, while qualitative review can be time-consuming.

**Our approach.** RiskPulse AI integrates optional live collection from **Marketaux** financial news and **Bluesky** public posts with a three-stage analytics pipeline: (1) a fine-tuned FinBERT-style regressor produces continuous financial sentiment in `[-1, 1]`; (2) a five-class event classifier assigns probabilities to **Geopolitical, Macroeconomic, Credit Event, Merger / Acquisition,** and **Product Launch**; and (3) an **XGBoost** model combines those outputs with historical market features to estimate an impact *severity decile* (1–10). Event source metadata and results are represented in structured records, and the React dashboard exposes both Quick View and Analyst View.

**Business applications.** **Module A** converts event signals into proposed rebalancing weights for a simulated **15-stock index**, with position, sector, and turnover constraints and a historical walk-forward backtest. **Module B** evaluates a synthetic **30-position portfolio** of loans, bonds, swaps, and options under hypothetical shocks; it separates instantaneous trading-book mark-to-market (MTM) effects from incremental **12-month expected credit loss (ECL)** for loans. A contextual assistant, **Bujji**, can explain the workflow when its optional Groq API is configured. The prototype supports analyst exploration rather than autonomous trades, measured early-warning superiority, or certified capital calculations.

> **Model weights hosted on Hugging Face:** Trained transformer weight checkpoints and the impact model are officially hosted on Hugging Face Hub:
> - **Event Classifier:** [`shanmukhdatta/riskpulse-event-classifier`](https://huggingface.co/shanmukhdatta/riskpulse-event-classifier)
> - **Sentiment Regressor:** [`shanmukhdatta/riskpulse-sentiment-regressor`](https://huggingface.co/shanmukhdatta/riskpulse-sentiment-regressor)
> - **Impact Engine:** [`shanmukhdatta/riskpulse-impact-engine`](https://huggingface.co/shanmukhdatta/riskpulse-impact-engine)
> 
> A new clone can download all model weights in one command with `python -m scripts.fetch_models`. See [Model setup](#model-setup-and-run-modes).

## 2. Architecture & Tech Stack

![RiskPulse AI system architecture](docs/architecture.png)

```text
Marketaux news / Bluesky public posts     Historical prices / features
                    |                               |
            Normalize, match tickers, deduplicate   |
                    |                               |
            SQLite ingestion repository             |
                    |                               |
            FinBERT sentiment regression             |
            Five-class event classification          |
                    |                               |
                    +-----------+-------------------+
                                |
                    XGBoost impact severity
                                |
                      Structured risk signal
                                |
                    +-----------+-----------+
                    |                       |
            Module A: index lab       Module B: stress lab
            - 15 equity weights      - 30 synthetic positions
            - constrained optimizer  - bonds, swaps, options MTM
            - 357-session backtest   - loan 12-month ECL
                                     - What-If and reverse stress
                    |                       |
                    +-----------+-----------+
                                |
                   FastAPI + React dashboard
                      Quick / Analyst views
                         Bujji assistant
```

| Component | Implementation |
|---|---|
| Backend & API | Python 3.11/3.12, FastAPI, Pydantic, Uvicorn |
| Frontend | React **18**, TypeScript, Vite **6**, responsive Quick/Analyst views |
| NLP | Hugging Face Transformers, PyTorch, custom sentiment-regression head, five-class event head |
| Market-impact model | XGBoost, scikit-learn, pandas/NumPy, historical market features |
| Ingestion & storage | Marketaux, Bluesky AT Protocol, optional GDELT adapter (disabled by default), SQLite |
| Portfolio computation | SciPy SLSQP constrained optimization; historical backtesting; bond, swap, loan, and option sensitivity/pricing implementations |
| Optional assistant | Groq API integration; unavailable without configured API key |

**Implementation details.** The deployed event classifier must match the exact five-category label mapping in [`configs/event_labels.json`](configs/event_labels.json). The impact model uses event probabilities, sentiment and *pre-publication* historical price/volume indicators. The resulting decile represents estimated historical abnormal-reaction **magnitude**, not a directional stock-return forecast. The event-to-stress shock mapping is a separate **illustrative heuristic**, not learned default-probability or spread forecasting.

**Module A constraints.** A 15-equity, equal-weight reference index is optimized subject to full investment, long-only holdings, a **±2 percentage-point** constituent deviation, **±5 percentage-point** sector deviation, and **10%** maximum one-way turnover. The transaction-cost assumption is **10 basis points**. The optimizer uses **SLSQP** for the constrained squared-distance objective. This is a simulated portfolio, not a brokerage execution system.

**Module B risk horizons.** The portfolio includes **12 loans, 10 bonds, 4 interest-rate swaps, and 4 options**. Bonds, swaps and options contribute instantaneous MTM changes; loans contribute incremental 12-month ECL. Loan exposure at default, bond market value, and derivative reference notional are **not additive NAV**. Reverse stress uses a **deterministic scenario grid**, reports the first *sampled* budget breach, and does not claim an exact minimum or monotonic losses.

**Source layout (existing paths intentionally preserved):**

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example
├── app/                    # FastAPI, ML runtime, ingestion, modules A/B, assistant
├── frontend/               # React + Vite dashboard and frontend tests
├── configs/                # Tickers, labels, model features, scenarios and controls
├── data/
│   ├── ingestion/          # Seed SQLite demo database
│   ├── module_a/           # Bundled holdout prices/signals and example response
│   ├── module_b/           # Synthetic portfolio, scenarios, verification response
│   └── impact_step2_report.json
├── models/                 # Model configs/tokenizers; impact joblib artifacts
│                           # NLP transformer trained weights NOT committed
├── training/               # Source notebooks, exports, methodology and results evidence
├── scripts/                # Independent demos, model checks, diagnostics
├── tests/                  # Backend tests
└── docs/
    ├── Presentation_.pdf
    ├── architecture.png
    ├── methodology.md
    ├── validation.md
    └── screenshots/
```

Further technical detail: [Methodology](docs/methodology.md) · [Validation evidence](docs/validation.md) · [Training sources](training/README.md).

## 3. Dataset Used

Only publicly sourced or synthetic **research/demo** data is used; no confidential client data is required. The table distinguishes datasets actually bundled in GitHub from external datasets/checkpoints needed to repeat training.

| Data / provenance | Nature and use | Availability in repository |
|---|---|---|
| Event-classification research corpus: financial topic posts, GDELT-derived events, SentiVent, SEC 8-K samples, corporate-event and catalyst datasets | Publicly sourced labeled financial text used to construct a five-class event dataset. Recorded source run: **26,172 raw candidate rows**, **16,576 cleaned real rows**; **13,260 train / 1,658 validation / 1,658 test** (stratified, seed 42). | **Not bundled** as raw/processed splits. Original source notebooks, dataset identifiers and recorded counts: [event classifier training](training/event_classifier/README.md). |
| Continuous sentiment: **FiQA** financial sentiment text and continuous scores | Supervised sentiment-regression training; continuous score, not a three-class sentiment-label target. | Training notebooks and source exports under [`training/sentiment/`](training/sentiment/). Underlying processed data and fine-tuned weights not bundled. |
| Impact-model research: **FNSPID** financial news and historical stock-price sources | Build historical abnormal-return targets and train XGBoost with lagged news/market features. Original Step-2 report records **19,084 train / 3,446 validation / 4,930 test** rows; this is distinct from the event-classification corpus. | Training scripts and feature/calibration configs plus an impact model joblib are committed. Original news/price archives and intermediate parquet feature matrices are **not bundled**. See [impact training](training/impact_engine/README.md) and [Step-2 report](data/impact_step2_report.json). |
| Marketaux and Bluesky records | Public news metadata and public social posts for event ingestion, with original source links. External providers require network/API access for *new* records. | [`data/ingestion/riskpulse.db`](data/ingestion/riskpulse.db) seed demonstration database; recorded snapshot described as **2 news + 27 social entries** (29 total). This is **pre-collected**, not proof of live connectivity at review time. |
| Module A offline price and signal data | Historical 15-stock daily price grid and timestamped signals for **357-session** example backtesting. | [holdout_prices.csv](data/module_a/holdout_prices.csv) and [holdout_signals.csv](data/module_a/holdout_signals.csv). |
| Module B synthetic portfolio | Artificial institution-style balance-sheet exposures: 12 loans, 10 bonds, 4 swaps, 4 options; illustrative shocks, not real client positions. | [synthetic_portfolio.json](data/module_b/synthetic_portfolio.json) and [scenario_library.json](data/module_b/scenario_library.json). |

**Data assumptions and safeguards.** Labels, source styles, headline deduplication, source-specific distributions, simulated portfolio exposures, and hypothetical scenario shock sizes affect interpretation. Different upstream dataset revisions or refreshed prices may change training results. Impact-calibration cutoffs are declared training-derived; timestamp-sensitive features must use only earlier market information. An example result or offline backtest is not evidence of profitable live trading. Before redistributing additional public dataset files or original model checkpoints, verify each source's licensing, access requirements and any applicable platform terms.

**Availability distinction:** The GitHub repository contains the **demo inputs needed for independent portfolio simulations** plus model-training source, but not *all* original ML training inputs or the trained NLP checkpoints. A full end-to-end trained-model reproduction consequently needs additional authentic external assets; see the next section.

## 4. Quickstart & Installation

**Runtime:** Python 3.11 or 3.12; Node.js 20+; Git. The prior project documentation reports a Windows/Python 3.11 development configuration; this README update has **not rerun** a clean installation on Windows, macOS or Linux. Commands below are provided for both Windows PowerShell and POSIX shells.

### Clone and install

```bash
git clone https://github.com/shanmukhdatta/nitj-shanmukhdatta-hackthon.git
cd nitj-shanmukhdatta-hackthon
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell (if script execution is restricted: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
```

```bash
# Linux / macOS (instead of Windows commands above)
source .venv/bin/activate
cp .env.example .env
```

Install dependencies **from the repository root**:

```bash
python -m pip install -r requirements.txt
npm --prefix frontend ci
```

### Model setup and run modes

**Full real inference (from Hugging Face Hub):** The trained event classifier, sentiment regressor, and impact engine weights are hosted on Hugging Face. Download them automatically into `models/`:

```bash
python -m scripts.fetch_models
python -m scripts.check_models
```

Once downloaded, `check_models` will verify file presence. Keep `SKIP_MODEL_LOAD=false` in `.env` for complete real inference.

**Portfolio-only demonstration (no transformer checkpoints required):** Edit `.env` to set `SKIP_MODEL_LOAD=true`. This allows the backend to start with the risk engine unavailable: `/analyze`, full news-to-portfolio inference, and `/health` will return **503**, rather than fabricated model predictions. You can still inspect saved records and use independent Module A/Module B routes or demos:

```bash
python -m scripts.run_module_a_backtest
python -m scripts.run_module_b_demo
```

Those scripts write outputs under `data/module_a/results/` and `data/module_b/results/`. Neither requires the FinBERT weight files. The GitHub README does **not** label this limited configuration as complete three-model inference.

### Start the backend and frontend

From the repository root, after activation and dependency installation:

```bash
python -m uvicorn app.main:app --reload --port 8000
# or: uvicorn app.main:app --reload --port 8000
```

In **another terminal**, from the same repository root:

```bash
npm --prefix frontend run dev
```

- **Local dashboard:** http://127.0.0.1:5173
- **API documentation:** http://127.0.0.1:8000/docs
- **API health:** http://127.0.0.1:8000/health (200 only when all inference models load)
- **Ingestion health/configuration:** http://127.0.0.1:8000/ingestion/health (does not assert live provider connectivity)
- **Saved signal feed:** http://127.0.0.1:8000/signals
- **Simulated index:** http://127.0.0.1:8000/module-a/index
- **Synthetic stress portfolio:** http://127.0.0.1:8000/module-b/portfolio

The Vite development server forwards the browser's `/api` requests to the local FastAPI backend. To fetch **new live news**, configure the optional `MARKETAUX_API_TOKEN` in `.env`; Bluesky collection needs external connectivity. Collection and model processing are distinct stages. The protected ingestion endpoints require an `INGESTION_ADMIN_TOKEN` of at least 24 characters; do not commit any real tokens. **Bujji** requires an optional `GROQ_API_KEY` and explicitly reports unavailability when that key is absent.

### Verification

Run from the repository root:

```bash
pytest -q
npm --prefix frontend test
npm --prefix frontend run check:contracts
npm --prefix frontend run build
```

The [validation report](docs/validation.md) records a previous **120/120 backend** and **14/14 frontend** test run plus build/API checks. These were **not rerun** while preparing this README. Unit tests and production bundling do not establish model checkpoint provenance or trained-model correctness. End-to-end testing against live servers is separately documented in `frontend/scripts/audit-e2e.mjs`.

Training is documented separately under [training/README.md](training/README.md), [event classifier](training/event_classifier/README.md), and [impact engine](training/impact_engine/README.md). Training sources include external data downloads and model checkpoints and should not be mistaken for a fully offline training release.

## 5. Key Results & Domain Impact

### Recorded evidence and demonstrated functionality

These figures are taken from files already committed in the repository. They are **recorded evidence**, not results freshly reproduced as part of this documentation update.

| Component | Recorded result / implemented capability | Evidence and interpretation |
|---|---|---|
| Event-classifier research experiment | **99.46%** recorded held-out test accuracy; **0.9798 macro-F1** on the real-only dataset | [Recorded notebook evidence](training/event_classifier/recorded_results/notebook_evidence.json). The Credit Event test subset contains only **12** examples. The notebook's base-checkpoint ID differs from the current runtime event configuration, so deployed equivalence is unverified. |
| Sentiment analysis | Continuous **[-1, +1]** regression model design using FiQA | [Source training code](training/sentiment/notebooks/02_finbert_regression_training.ipynb). Do not infer new test metrics without the original training report. |
| Market impact engine | Predicted severity deciles **1–10**, using historical reaction-magnitude targets and past-only market features | [Feature manifest](configs/impact_model_feature_manifest.json), [quantile boundaries](configs/impact_decile_thresholds.json). Severity does not predict guaranteed movement direction. |
| Ingestion | **29** previously collected source-linked items described in bundled SQLite demo snapshot: 2 news, 27 public social | [Seed database](data/ingestion/riskpulse.db). Snapshot counts do not prove live polling remains operational. |
| Module A | **15-stock** constrained rebalance, with bundled **357-session** walk-forward holdout inputs | [Backtest runner](scripts/run_module_a_backtest.py), [data](data/module_a/holdout_prices.csv), [example rebalance](data/module_a/real_example_response.json). Financial results should be rerun from the bundled data before claiming performance improvements. |
| Module B | **30-position** synthetic multi-asset stress calculator with separately reported MTM and incremental 12-month ECL | [Portfolio](data/module_b/synthetic_portfolio.json), [scenarios](data/module_b/scenario_library.json), [captured API verification](data/module_b/live_verification_results.json). Stress shocks and outputs are scenario assumptions, not live losses. |
| What-If / Reverse Stress | Configurable position scaling and **first sampled** loss-budget breach on a grid | [Demo script](scripts/run_module_b_demo.py), [grid implementation](app/module_b/reverse_stress.py). No exact bisection-root or globally minimum threshold is claimed. |
| Automated implementation tests | Previously logged **120/120** backend and **14/14** frontend tests | [Validation report](docs/validation.md). This is software test evidence, not an out-of-distribution model benchmark. |

### Financial-domain relevance

**Risk triage:** Combining event category, sentiment, estimated impact magnitude and source provenance can help analysts systematically organize financial narratives for further investigation. RiskPulse does not establish that its alerts precede or outperform real market reactions.

**Constraint-aware allocation analysis:** The index lab demonstrates how qualitative event scores can be translated into proposed weights while enforcing specified sector, constituent and trading-cost constraints. It does not execute orders or certify regulatory compliance.

**Cross-asset stress exploration:** The stress lab demonstrates how hypothetical events might affect different instrument types while making the risk-horizon distinction explicit. Interactive exposure changes and grid-based breach exploration help communicate assumptions and sensitivities; they are not production capital calculations.

### Known limitations and reproducibility

- The repository does **not** contain the two full trained NLP checkpoints; real inference requires genuine matching external model artifacts.
- The event classifier's training initializer is `nickmuchi/finbert-tone-finetuned-finance-topic-classification`, while the committed runtime config refers to `yiyanghkust/finbert-tone`. Checkpoint identity must be resolved with original exports and hashes before claiming the notebook results correspond to the deployed model.
- Event-source imbalance and low support for some classes make high in-distribution accuracy an incomplete measure of real-news generalization.
- Original training datasets and price snapshots are not fully bundled; upstream access, licensing, revisions and computation affect repeatability.
- Module A is a constrained historical simulation, not proof of improved real-money performance. Module B uses **synthetic** exposures and assumed scenarios; mixed MTM/ECL horizons should not be interpreted as regulatory NAV or capital.
- Live news and assistant functionality require external provider availability, appropriate API credentials and permissions.

**Responsible use:** Educational demonstration only. No personalized investment advice, autonomous trading, guaranteed market forecasts, confidential client data, or representation of endorsement by S&P Global or CRISIL.

---

### Demo and supporting documentation

**YouTube Demo Video:**  
**Presentation Deck:** [Repository PDF (docs/Presentation_.pdf)](docs/Presentation_.pdf) · [Google Drive Link](https://drive.google.com/file/d/1ySZRghGGWxB0SQZsvHdQ40yYgSAaQkuJ/view?usp=sharing)  
**Live Demo Site:** https://nitj-shanmukhdatta-hackthon-lime.vercel.app/#how-it-works  

- [System architecture](docs/architecture.png)
- [Quantitative methodology](docs/methodology.md)
- [Recorded validation evidence](docs/validation.md)
- [Frontend instructions](frontend/README.md)
- [Training source instructions](training/README.md)
- [MIT license and third-party notices](LICENSE)

### Acknowledgments

This independent academic prototype uses third-party libraries and publicly sourced data under their respective terms, including Hugging Face Transformers, PyTorch, FinBERT-related foundations, XGBoost, SciPy, Marketaux, Bluesky and yfinance. Third-party assets and licenses must be verified before redistributing datasets or checkpoints. AI-assisted documentation and code review do not replace original implementation, testing and honest attribution.
