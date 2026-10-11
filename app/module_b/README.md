# RiskPulse AI — Module B: Multi-Asset Stress Lab

`app/module_b/` implements the **synthetic institutional portfolio stress-testing engine** of RiskPulse AI. It evaluates balance-sheet risk across multi-asset holdings under macro, sector, and issuer-specific event shocks, with an auditable separation between instantaneous trading-book mark-to-market (MTM) P&L and 12-month incremental expected credit loss (ECL).

---

## 1. Architectural Scope & Purpose

* **Scope:** Financial stress testing, sensitivity analysis, interactive What-If position scaling, and automated Reverse Stress testing on a synthetic multi-asset portfolio.
* **Non-Goals:** Does **not** compute certified Basel III/IV regulatory capital ratios, internal ratings-based (IRB) formulas, comprehensive credit migration matrices, full yield-curve term structure boots, or live counterparty margin calls. It is designed for scenario analysis and transparent risk audit.

---

## 2. Component Structure

| File / Folder | Purpose | Key Classes / Functions |
| :--- | :--- | :--- |
| [`portfolio.py`](portfolio.py) | 30-position synthetic portfolio loader & asset aggregation | `load_portfolio()`, `portfolio_summary()` |
| [`scenarios.py`](scenarios.py) | Scenario library manager and effective shock resolution | `load_scenarios()`, `effective_shocks()`, `Scenario` |
| [`stress_engine.py`](stress_engine.py) | Core multi-asset stress evaluator separating MTM & ECL | `run_stress()` |
| [`what_if.py`](what_if.py) | Position-level exposure scaling ($0.0\times$ to $2.0\times$) | `apply_what_if()` |
| [`reverse_stress.py`](reverse_stress.py) | Grid search for budget-breaching scenario multipliers | `run_reverse_stress()` |
| [`event_mapper.py`](event_mapper.py) | Heuristic mapping of NLP event signals into scenario shocks | `signal_to_shocks()` |
| [`pricing/loans.py`](pricing/loans.py) | IFRS 9 / CECL style 12-month Expected Credit Loss model | `loan_effect()` |
| [`pricing/bonds.py`](pricing/bonds.py) | Modified duration & convexity bond repricing | `bond_effect()` |
| [`pricing/swaps.py`](pricing/swaps.py) | Linear DV01 interest-rate swap sensitivity | `swap_effect()` |
| [`pricing/options.py`](pricing/options.py) | European Black-Scholes option repricing (Delta, Gamma, Vega) | `option_effect()` |
| [`schemas.py`](schemas.py) | Pydantic data models for positions, shocks, and stress results | `Portfolio`, `Position`, `StressResult` |
| [`router.py`](router.py) | FastAPI endpoints for Module B | `/module-b/portfolio`, `/module-b/stress`, etc. |

---

## 3. Synthetic Portfolio Composition (30 Positions)

The portfolio represents an illustrative institutional balance sheet configured in [`data/module_b/synthetic_portfolio.json`](../../data/module_b/synthetic_portfolio.json):

1. **Banking Book — Loans (12 Positions):**
   * Commercial loans, term debt, and revolving credit facilities across Technology, Energy, Industrials, Real Estate, and Financials.
   * Total Exposure at Default (EAD): ~$24.2M.
2. **Trading Book — Corporate & Sovereign Bonds (10 Positions):**
   * Fixed-coupon senior debt with durations ranging from 2.1 to 8.4 years.
   * Total Market Value: ~$18.7M.
3. **Trading Book — Derivatives:**
   * **Interest Rate Swaps (4 Positions):** Pay-fixed and receive-fixed IRS with reference notionals of $5.0M–$15.0M and distinct DV01 sensitivities.
   * **Equity Options (4 Positions):** S&P index and single-stock calls/puts modeled via Black-Scholes.

> **Important Balance Sheet Disclaimer:** Loan Exposure at Default (EAD), Bond Market Value, and Derivative Reference Notional are **not additive Net Asset Value (NAV)**. Derivative notionals indicate contract sizing, not capital at risk.

---

## 4. Multi-Asset Valuation Mechanics

```text
                            ┌──────────────────────────────────────────────┐
                            │              Multi-Asset Shocks              │
                            │  Rates (bps) • Spreads (bps) • Equity • Vol  │
                            └──────────────────────┬───────────────────────┘
                                                   │
                  ┌────────────────────────────────┴──────────────────────────────┐
                  ▼                                                               ▼
       Banking Book (Loans)                                           Trading Book (Bonds, Swaps, Options)
    ┌───────────────────────────┐                                  ┌────────────────────────────────────────┐
    │       12-Month ECL        │                                  │           Instantaneous MTM            │
    │  EAD × PD(stressed) × LGD │                                  │  Duration/Convexity, DV01, Blk-Scholes │
    └───────────────────────────┘                                  └────────────────────────────────────────┘
```

### 4.1 Loans (Banking Book — 12-Month Incremental ECL)
Loans are evaluated on an expected credit loss horizon:

$$\text{ECL}_{\text{base}} = \text{EAD} \times \text{PD}_{\text{base}} \times \text{LGD}$$

$$\text{ECL}_{\text{stressed}} = \text{EAD} \times \min(1.0, \text{PD}_{\text{base}} \times \text{Multiplier}_{\text{stress}}) \times \text{LGD}$$

$$\Delta \text{ECL}_{12\text{m}} = \text{ECL}_{\text{stressed}} - \text{ECL}_{\text{base}}$$

### 4.2 Corporate Bonds (Trading Book — MTM Repricing)
Bonds reprice instantaneously based on modified duration $D_{\text{mod}}$, convexity $C$, benchmark rate shock $\Delta y$, and credit spread widening $\Delta s$:

$$\Delta P \approx -P \cdot D_{\text{mod}} \cdot (\Delta y + \Delta s) + \frac{1}{2} P \cdot C \cdot (\Delta y + \Delta s)^2$$

### 4.3 Interest Rate Swaps (Trading Book — Curve Shifts)
Swaps contribute linear mark-to-market changes via position dollar value of a basis point ($\text{DV01}$):

$$\Delta \text{MTM}_{\text{swap}} = \text{DV01} \times \Delta y_{\text{bps}} \times (\pm 1 \text{ direction})$$

### 4.4 Equity Options (Trading Book — Black-Scholes Repricing)
Options reprice under simultaneous underlying equity price shifts $\Delta S_{\%}$, implied volatility shifts $\Delta \sigma$, and risk-free rate moves $\Delta r$:

$$C = S \Phi(d_1) - K e^{-r T} \Phi(d_2), \quad P = K e^{-r T} \Phi(-d_2) - S \Phi(-d_1)$$

---

## 5. Advanced Analytical Capabilities

### 5.1 Dual-Horizon Reporting
[`stress_engine.py`](stress_engine.py) maintains strict accounting integrity by reporting separate line items:
* `instantaneous_mtm_pnl_usd`: Trading book immediate mark-to-market effect.
* `incremental_12m_expected_credit_loss_usd`: Banking book 1-year cumulative credit loss projection.
* `illustrative_mixed_horizon_loss_usd`: An explicit, labeled combination ($-\text{MTM} + \Delta\text{ECL}_{12\text{m}}$) for executive screening, accompanied by mandatory disclaimers.

### 5.2 What-If Sensitivity Adjustments
[`what_if.py`](what_if.py) enables interactive position scaling:
* Analysts can adjust position sizing via multipliers $\alpha_i \in [0.0\times, 2.0\times]$ in the dashboard.
* Re-evaluates portfolio stress without mutating original portfolio files.

### 5.3 Automated Reverse Stress Testing
[`reverse_stress.py`](reverse_stress.py) searches for the threshold multiplier $\lambda^*$ that triggers a targeted dollar loss budget:
* Evaluates a deterministic scenario multiplier grid $\lambda \in [0.1, 3.0]$ with step $0.05$.
* Reports the first sampled parameter combination that breaches the loss threshold.

---

## 6. API Route Contracts

* `GET /module-b/portfolio`: Returns complete 30-position inventory, asset breakdown, and base ECL figures.
* `GET /module-b/scenarios`: Returns predefined macro and credit stress scenario definitions.
* `POST /module-b/stress`: Executes stress testing for a selected scenario across the portfolio.
* `POST /module-b/what-if`: Evaluates stress under custom position-level exposure multipliers.
* `POST /module-b/reverse-stress`: Executes automated reverse stress testing to find the first budget breach.
