# Soochana AI — Tactical Index Rebalancing Laboratory (`app/module_a/`)

`app/module_a/` implements the **Tactical Index Rebalancing Laboratory**, translating qualitative event risk signals into mathematically constrained portfolio weight allocations across an equal-weight US equity universe.

---

## 1. System Architecture & Internal Flow

```text
====================================================================================================
                            MODULE A: REBALANCING ENGINE PIPELINE
====================================================================================================

  Input News Stream / Saved Signals (`RebalanceRequest` / `RebalanceFromNewsRequest`)
                               │
                               ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 1. Signal Ingestion & Temporal Decay (`signals.py`)                                    │
  │ • Deduplication by headline hash per ticker                                            │
  │ • Exponential Half-Life Temporal Decay:                                                │
  │     decay(Δt) = exp(-Δt / τ),   where τ = 3.0 days                                     │
  │ • Event Conviction Weighting:                                                          │
  │     signal_score_i = ∑_k [ sentiment_k × (impact_k / 10.0) × confidence_k × decay_k ]  │
  │ • Score centering: s_centered = s_vec - mean(s_vec)                                    │
  └────────────────────────────┬───────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 2. Desired Tilt Generation (`strategy.py`)                                             │
  │ • Baseline allocation: Equal weight w_base = 1 / 15 ≈ 6.67%                            │
  │ • Desired unconstrained target:                                                        │
  │     w_desired = w_base + γ · s_centered,   where γ = 0.02 (signal_tilt_coefficient)   │
  └────────────────────────────┬───────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 3. Constrained Convex Quadratic Optimization (`constraints.py`)                        │
  │                                                                                        │
  │   Solve: min_w  1/2 ||w - w_desired||_2^2                                              │
  │                                                                                        │
  │   Subject to:                                                                          │
  │   1. Full Investment:      ∑_{i=1}^{15} w_i = 1.0                                      │
  │   2. Long-Only:            w_i ≥ 0.0  ∀ i                                              │
  │   3. Position Band:        |w_i - w_{base, i}| ≤ 0.02  (±2 percentage points)          │
  │   4. Sector Band:          |∑_{j∈S_k} w_j - ∑_{j∈S_k} w_{base, j}| ≤ 0.05  (±5 pp)     │
  │   5. Turnover Cap:         1/2 ∑_{i=1}^{15} |w_i - w_{current, i}| ≤ 0.10  (10% max)   │
  │                                                                                        │
  │   Optimizer: SciPy SLSQP (Sequential Least Squares Programming)                        │
  └────────────────────────────┬───────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 4. Transaction Friction & Execution Accounting                                         │
  │ • Realized one-way turnover: T = 1/2 ∑ |w_target - w_current|                          │
  │ • Estimated Transaction Cost: Cost = 2 · T · (10 bps / 10,000)                         │
  │ • WeightChange breakdown with per-constituent signal explanation                       │
  └────────────────────────────┬───────────────────────────────────────────────────────────┘
                               │
                               ▼
                         `RebalanceResponse`
```

---

## 2. Mathematical Formulation

### Objective Function
To avoid extreme turnover and maintain index tracking fidelity, the optimizer minimizes the squared Euclidean distance between the target portfolio weights $\mathbf{w}$ and the event-desired weights $\mathbf{w}_{\text{desired}}$:

$$\min_{\mathbf{w}} \frac{1}{2} \sum_{i=1}^{N} \left(w_i - w_{\text{desired}, i}\right)^2$$

### Portfolio Constraints
1. **Full Investment:**
   $$\sum_{i=1}^{N} w_i = 1.0$$
2. **No Short Sales (Long-Only):**
   $$w_i \ge 0, \quad \forall i \in \{1, \dots, N\}$$
3. **Constituent Position Deviation Limit:**
   $$w_{\text{base}, i} - \delta_{\text{pos}} \le w_i \le w_{\text{base}, i} + \delta_{\text{pos}}, \quad \text{with } \delta_{\text{pos}} = 0.02 \; (\pm 2\%)$$
4. **Sector Deviation Limit:**
   For each sector $S_k$:
   $$\left|\sum_{j \in S_k} w_j - \sum_{j \in S_k} w_{\text{base}, j}\right| \le \delta_{\text{sec}}, \quad \text{with } \delta_{\text{sec}} = 0.05 \; (\pm 5\%)$$
5. **One-Way Turnover Limit:**
   $$\frac{1}{2} \sum_{i=1}^{N} |w_i - w_{\text{current}, i}| \le \tau_{\max}, \quad \text{with } \tau_{\max} = 0.10 \; (10\%)$$

---

## 3. Reference Universe & Sector Breakdown

The 15-stock universe is defined in [`configs/module_a_index.json`](../../configs/module_a_index.json):

| Sector | Tickers | Baseline Weight | Maximum Permitted Sector Weight |
| :--- | :--- | :---: | :---: |
| **Technology** | AAPL, MSFT, NVDA, QCOM, GOOGL | 33.33% | 38.33% |
| **Communication Services** | CMCSA | 6.67% | 11.67% |
| **Consumer Discretionary** | AMZN, TSLA, BKNG, SBUX | 26.67% | 31.67% |
| **Consumer Staples** | COST, PEP | 13.33% | 18.33% |
| **Healthcare** | GILD, AMGN | 13.33% | 18.33% |
| **Industrials** | CSX | 6.67% | 11.67% |

---

## 4. Historical Walk-Forward Backtester (`backtester.py`)

* **Dataset:** 357 daily sessions of bundled holdout prices ([`data/module_a/holdout_prices.csv`](../../data/module_a/holdout_prices.csv)) and timestamped holdout signals ([`data/module_a/holdout_signals.csv`](../../data/module_a/holdout_signals.csv)).
* **Execution Realism:**
  * Strict anti-lookahead rules: news signals dated on session $t$ are executed at the close of session $t$ or open of session $t+1$.
  * Transaction cost deduction: 10 bps linear friction applied to all portfolio turnover.
  * Benchmark comparisons: Equal-weight benchmark vs tactical event-driven strategy.

---

## 5. API Endpoints

* `GET /module-a/index`: Retrieve universe constituents, baseline weights, and constraint parameters.
* `POST /module-a/rebalance`: Rebalance portfolio given an explicit array of pre-computed `Signal` objects.
* `POST /module-a/rebalance-from-news`: End-to-end endpoint accepting raw article texts, running real-time NLP inference via the Risk Engine, and executing the rebalancing optimization.
