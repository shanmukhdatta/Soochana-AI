# Soochana AI — Multi-Asset Stress Testing Laboratory (`app/module_b/`)

`app/module_b/` implements the **Multi-Asset Stress Testing Laboratory**, evaluating the vulnerability of an institutional balance sheet under hypothetical macroeconomic, geopolitical, and credit stress scenarios.

---

## 1. System Architecture & Valuation Flow

```text
====================================================================================================
                        MODULE B: MULTI-ASSET STRESS ENGINE PIPELINE
====================================================================================================

  Scenario Specification (`Scenario` with Macro Shocks, Sector Shocks, Issuer Overrides)
                                │
                                ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 1. Shock Resolution (`scenarios.py`)                                                   │
  │ • Macro base shocks: Interest rate Δr, Equity return ΔS, Volatility Δσ, PD multiplier  │
  │ • Sector-level shock overlays (e.g., Technology equity shock vs Energy spread shock)   │
  │ • Issuer-specific targeted overrides (e.g., AAPL rating downgrade, credit spread spike)│
  │ • Validation: ensure all targeted overrides match held portfolio positions             │
  └────────────────────────────┬───────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 2. Horizon-Separated Instrument Pricing Engines (`pricing/`)                           │
  │                                                                                        │
  │  ┌───────────────────────────────────────────────────────────────────────────────┐     │
  │  │ BANKING BOOK: 12-Month Expected Credit Loss (ECL)                             │     │
  │  │ • 12 Commercial Loans (`pricing/loans.py`)                                    │     │
  │  │   PD_stressed = min(1.0, PD_base × pd_multiplier)                             │     │
  │  │   Incremental ECL_12m = EAD × LGD × (PD_stressed - PD_base)                   │     │
  │  └───────────────────────────────────────────────────────────────────────────────┘     │
  │                                                                                        │
  │  ┌───────────────────────────────────────────────────────────────────────────────┐     │
  │  │ TRADING BOOK: Instantaneous Mark-to-Market (MTM) P&L                          │     │
  │  │ • 10 Corporate Bonds (`pricing/bonds.py`):                                    │     │
  │  │   ΔP ≈ -D_mod × (Δr + Δspread) + 1/2 × Convexity × (Δr + Δspread)^2           │     │
  │  │ • 4 Interest Rate Swaps (`pricing/swaps.py`):                                 │     │
  │  │   ΔMTM ≈ DV01 × Δr  (Payer vs Receiver swap conventions)                      │     │
  │  │ • 4 Equity Options (`pricing/options.py`):                                    │     │
  │  │   Black-Scholes revaluation under (S + ΔS, σ + Δσ, r + Δr)                    │     │
  │  └───────────────────────────────────────────────────────────────────────────────┘     │
  └────────────────────────────┬───────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 3. Loss Contribution & Risk Aggregation (`stress_engine.py`)                           │
  │ • Total MTM P&L: ∑ MTM_pnl (instantaneous trading book effect)                         │
  │ • Total Incremental ECL: ∑ Incremental_ECL_12m (12-month forward credit deterioration)  │
  │ • Breakdown by Instrument Type, Sector, and Issuer Ticker                              │
  │ • Top-10 Loss Contributors identification                                              │
  └────────────────────────────┬───────────────────────────────────────────────────────────┘
                               │
                               ▼
                         `run_stress()` Output Payload
```

---

## 2. Mathematical Valuation Formulations

### 2.1 Commercial Loans: 12-Month Expected Credit Loss (`loans.py`)
Loans reside in the banking book and are evaluated under an IFRS 9 / CECL-aligned credit risk framework:

$$\text{ECL}_{\text{base}} = \text{EAD} \times \text{PD}_{\text{base}} \times \text{LGD}$$
$$\text{PD}_{\text{stressed}} = \min\left(1.0, \; \text{PD}_{\text{base}} \times M_{\text{PD}}\right)$$
$$\Delta \text{ECL}_{12m} = \text{EAD} \times \text{LGD} \times \left(\text{PD}_{\text{stressed}} - \text{PD}_{\text{base}}\right)$$

* $\text{EAD}$: Exposure at Default ($USD$)
* $\text{PD}$: 12-Month Probability of Default
* $\text{LGD}$: Loss Given Default ($1 - \text{Recovery Rate}$)
* $M_{\text{PD}}$: Scenario-driven PD shock multiplier

### 2.2 Corporate Bonds: Duration & Convexity MTM (`bonds.py`)
Bond mark-to-market changes account for both benchmark yield shifts and credit spread widening:

$$\Delta y = \Delta r_{\text{benchmark}} + \Delta s_{\text{credit}}$$
$$\frac{\Delta P}{P} \approx -D_{\text{mod}} \cdot \Delta y + \frac{1}{2} C \cdot (\Delta y)^2$$
$$\Delta \text{MTM}_{\text{bond}} = \text{Notional} \times \frac{P_{\text{base}}}{100} \times \left(\frac{\Delta P}{P}\right)$$

### 2.3 Interest Rate Swaps: DV01 Curve Sensitivity (`swaps.py`)
Swaps model net fixed-for-floating exposure revalued via dollar-value-of-a-basis-point (DV01):

$$\Delta \text{MTM}_{\text{swap}} = \text{DV01} \times \Delta r_{\text{bps}} \times \text{Direction}$$
where $\text{Direction} = +1$ for fixed-rate payers (floating receivers) and $-1$ for fixed-rate receivers.

### 2.4 Equity Options: Black-Scholes Revaluation (`options.py`)
Equity options are priced using the European Black-Scholes analytical formulation:

$$d_1 = \frac{\ln(S / K) + \left(r + \frac{\sigma^2}{2}\right)T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}$$
$$C(S, K, T, r, \sigma) = S \Phi(d_1) - K e^{-rT} \Phi(d_2)$$
$$P(S, K, T, r, \sigma) = K e^{-rT} \Phi(-d_2) - S \Phi(-d_1)$$

Under stress, the option is re-evaluated at $(S \cdot (1 + \Delta S), \; \sigma + \Delta \sigma, \; r + \Delta r)$, and $\Delta \text{MTM} = \text{Notional} \times (V_{\text{stressed}} - V_{\text{base}})$.

---

## 3. Strict Horizon & Accounting Boundary Rules

> [!IMPORTANT]
> **Audit Distinction:** Instantaneous Trading Book Mark-to-Market (MTM) P&L and Banking Book 12-Month Expected Credit Loss (ECL) represent fundamentally different accounting horizons.
> - **MTM** reflects immediate market liquidity liquidation value.
> - **ECL** reflects projected loan default provisions over a 1-year horizon.
> - Soochana AI strictly segregates these figures in the API responses and warns against treating their sum as an economic Net Asset Value (NAV) or certified regulatory capital measure.

---

## 4. Interactive What-If & Reverse Stress Capabilities

1. **What-If Position Scaling (`what_if.py`):**
   * Analysts can scale individual position notionals between $0.0\times$ (full divestment/hedge) and $2.0\times$ (doubling exposure) to test risk-mitigation strategies.
2. **Reverse Stress Testing (`reverse_stress.py`):**
   * Given a maximum tolerated loss budget (e.g., $\$5,000,000$), the engine evaluates a multi-dimensional parameter shock grid to identify the first scenario parameter threshold that breaches the risk tolerance.

---

## 5. API Endpoints

* `GET /module-b/portfolio`: Returns the 30-position synthetic portfolio and baseline asset valuations.
* `GET /module-b/scenarios`: Returns predefined historical and hypothetical stress scenarios.
* `POST /module-b/stress`: Executes stress test for a given scenario definition.
* `POST /module-b/what-if`: Re-runs stress testing with custom position-level scaling multipliers.
* `POST /module-b/reverse-stress`: Searches parameter space for the minimum breach threshold.
