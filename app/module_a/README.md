# RiskPulse AI — Module A: Tactical Index Rebalancing Lab

`app/module_a/` implements the **tactical index rebalancing engine and walk-forward historical backtester** of RiskPulse AI. It translates unstructured event signals (continuous financial sentiment, event taxonomy probabilities, and estimated market impact severity) into disciplined, mathematically constrained equity index rebalancing recommendations.

---

## 1. Architectural Scope & Purpose

* **Scope:** Quantitative simulation and backtesting module. It consumes structured event records from the Risk Engine (`app.engine.pipeline.RiskEngine`) or offline historical signals, solves a convex constrained portfolio rebalancing problem, and computes performance attribution.
* **Non-Goals:** Does **not** execute live exchange orders, integrate with brokerage APIs, manage custodial cash balances, or guarantee alpha. It is a research and audit simulator designed for financial analysts.

---

## 2. Component Structure

| File | Purpose | Key Classes / Functions |
| :--- | :--- | :--- |
| [`config.py`](config.py) | Immutable dataclass schema & validation for index parameters | `ModuleAConfig`, `ModuleAConfig.load()` |
| [`constraints.py`](constraints.py) | SciPy SLSQP quadratic projection solver with auxiliary trade variables | `project()`, `validate_weights()`, `InfeasiblePortfolio` |
| [`strategy.py`](strategy.py) | Event-informed tactical weight recommendation generation | `rebalance()` |
| [`signals.py`](signals.py) | Identity-based headline deduplication, half-life decay, and Bayesian shrinkage | `aggregate()`, `identity()`, `normalized_headline()` |
| [`backtester.py`](backtester.py) | 357-session historical walk-forward backtesting simulator | `run_backtest()`, `read_prices()`, `read_signals()` |
| [`metrics.py`](metrics.py) | Portfolio performance analytics (Sharpe, max drawdown, turnover) | `evaluate_performance()`, `annualized_sharpe()` |
| [`schemas.py`](schemas.py) | Pydantic request/response models | `RebalanceRequest`, `RebalanceResponse`, `BacktestRequest` |
| [`router.py`](router.py) | FastAPI route declarations for Module A | `/module-a/index`, `/module-a/rebalance`, `/module-a/backtest` |

---

## 3. Reference Index & Mathematical Optimization

### 3.1 Constituent Universe (15 Equities)
The reference portfolio represents a 15-stock equal-weighted US equity index across 6 sectors (baseline weight $w_{i,0} = \frac{1}{15} \approx 6.667\%$):

* **Technology (4):** `AAPL`, `MSFT`, `NVDA`, `QCOM`
* **Communication Services (2):** `GOOGL`, `CMCSA`
* **Consumer Discretionary (4):** `AMZN`, `TSLA`, `BKNG`, `SBUX`
* **Consumer Staples (2):** `COST`, `PEP`
* **Health Care (2):** `GILD`, `AMGN`
* **Industrials (1):** `CSX`

### 3.2 Quadratic Optimization Formulation
Given raw desired weights $\mathbf{w}_{\text{desired}}$ derived from signal tilts, the target portfolio $\mathbf{w}^*$ is projected onto the convex feasible set by solving:

$$\min_{\mathbf{w}, \mathbf{u}} \sum_{i=1}^{n} (w_i - w_{\text{desired}, i})^2$$

Subject to the following operational constraints:
1. **Full Investment & Long-Only:**
   $$\sum_{i=1}^{n} w_i = 1.0, \quad w_i \ge 0 \quad \forall i$$
2. **Constituent Position Deviation Limit ($\pm 2.0\%$):**
   $$\max(0, w_{i,0} - 0.02) \le w_i \le \min(1, w_{i,0} + 0.02) \quad \implies \quad 4.667\% \le w_i \le 8.667\%$$
3. **Sector Exposure Limit ($\pm 5.0\%$):**
   $$\sum_{i \in S_k} w_i \in \left[\max(0, T_k - 0.05), \min(1, T_k + 0.05)\right] \quad \forall \text{ sector } k$$
4. **One-Way Turnover Limit ($\le 10.0\%$):**
   $$\text{Turnover} = \frac{1}{2} \sum_{i=1}^{n} |w_i - w_{i, \text{current}}| \le 0.10$$
   *Linearized via auxiliary non-negative trade slack variables $u_i \ge \pm(w_i - w_{i, \text{current}})$ such that $\sum u_i \le 2 \times 0.10$.*
5. **Friction & Rebalancing Deadband:**
   * Trades below `minimum_trade_turnover` ($0.1\%$) are suppressed if the current portfolio remains within feasible bands.
   * Transaction costs are penalized at **10 basis points** ($0.10\%$) per unit of two-way turnover ($2 \times \text{turnover} \times 0.0010$).

---

## 4. Signal Processing & Aggregation Pipeline

[`signals.py`](signals.py) aggregates discrete event records into a continuous composite scalar score $S_i \in [-1.65, +1.65]$ per ticker:

1. **Strict No-Lookahead Window:**
   * Signals dated on or after the rebalancing session calendar date (UTC) are strictly excluded to avoid lookahead bias.
   * Maximum lookback window is **7 calendar days**.
2. **Deterministic Cross-Source Deduplication:**
   * Articles are identified by SHA-256 hash of `ticker | publication_date | normalized_headline`.
   * Cross-outlet syndication and reposts cannot multiply risk scores; the record with the highest source authority weight is retained.
3. **Source Authority & Exponential Half-Life Decay:**
   * Source weights: Financial News ($1.0$), Social Media ($0.55$), Manual ($0.50$), Other ($0.30$).
   * Time decay: $2^{-\text{age} / \tau}$ with half-life $\tau = 3.0$ days.
4. **Event Category Multipliers:**
   * Geopolitical ($1.1\times$), Credit Event ($1.1\times$), Macroeconomic ($1.0\times$), Merger/Acquisition ($1.0\times$), Product Launch ($0.9\times$).
5. **Bayesian Shrinkage (Evidence Prior Weight):**
   * To prevent a single isolated headline from driving maximum allowable portfolio tilt, the denominator adds an evidence prior mass $\kappa = 0.50$:
   $$S_i = \text{clip}\left(\frac{\sum_j w_j \cdot (\text{sentiment}_j \times \text{impact}_j)}{\sum_j w_j + 0.50}, -1.65, +1.65\right)$$

---

## 5. Walk-Forward Backtester

[`backtester.py`](backtester.py) evaluates strategy performance across **357 historical trading sessions** using pre-publication market quotes (`data/module_a/holdout_prices.csv`) and event signals (`data/module_a/holdout_signals.csv`):

* **Evaluated Strategies:**
  1. `benchmark`: Static equal-weighted buy-and-hold index.
  2. `sentiment_only`: Tilts informed strictly by sentiment $[-1, 1]$ without event taxonomy or impact severity weighting.
  3. `full_event_aware`: Combined FinBERT sentiment $\times$ event class multiplier $\times$ XGBoost impact severity decile.
* **Execution Convention:** Rebalancing trades execute at the official session market close; price drift between sessions is dynamically tracked.

---

## 6. API Route Contracts

* `GET /module-a/index`: Returns the 15-ticker index constituent metadata, sector targets, baseline weights, and optimization constraints.
* `POST /module-a/rebalance`: Runs the SLSQP projection given pre-computed signal inputs and returns before/after weights, delta percentage points, one-way turnover, and transaction costs.
* `POST /module-a/rebalance-from-news`: Accepts raw text articles, routes them through the full Risk Engine NLP & impact pipeline, and rebalances the index.
* `POST /module-a/backtest`: Triggers a 357-session historical walk-forward simulation and stores run results under `data/module_a/results/<uuid>.json`.
* `GET /module-a/backtest/{run_id}`: Retrieves completed backtest analytics and cumulative performance series.
