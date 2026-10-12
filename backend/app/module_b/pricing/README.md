# Analytical Instrument Pricing Models (`backend/app/module_b/pricing/`)

This package implements the financial instrument pricing and sensitivity models for the **Multi-Asset Stress Testing Laboratory**.

---

## 1. Instrument Pricing & Sensitivity Architecture

```text
====================================================================================================
                        INSTRUMENT PRICING & SENSITIVITY FRAMEWORK
====================================================================================================

  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 1. BANKING BOOK: COMMERCIAL LOANS (`loans.py`)                                         │
  │ • Accounting Standard: IFRS 9 / CECL 12-Month Expected Credit Loss (ECL)                │
  │ • Formula:                                                                             │
  │     ECL_base = EAD × PD_base × LGD                                                     │
  │     PD_stressed = min(1.0, PD_base × M_pd)                                             │
  │     ΔECL_12m = EAD × LGD × (PD_stressed - PD_base)                                     │
  │ • Horizon: 12-month forward credit deterioration, not instantaneous liquidation P&L    │
  └────────────────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 2. TRADING BOOK: CORPORATE BONDS (`bonds.py`)                                          │
  │ • Sensitivity: Modified Duration (D_mod) and Convexity (C)                             │
  │ • Yield Shift: Δy = Δr_benchmark + Δs_credit                                           │
  │ • Formula:                                                                             │
  │     ΔP/P ≈ -D_mod · Δy + 1/2 · C · (Δy)^2                                              │
  │     ΔMTM_bond = Notional × (P_base / 100) × (ΔP / P)                                   │
  │ • Metrics: Reports dollar DV01 and CS01 (per basis point risk)                         │
  └────────────────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 3. TRADING BOOK: INTEREST RATE SWAPS (`swaps.py`)                                      │
  │ • Sensitivity: Dollar Value of a Basis Point (DV01)                                    │
  │ • Formula:                                                                             │
  │     ΔMTM_swap = DV01 × Δr_bps × Direction                                              │
  │     Direction = +1 (Fixed Payer / Float Receiver), -1 (Fixed Receiver / Float Payer)   │
  └────────────────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 4. TRADING BOOK: EQUITY OPTIONS (`options.py`)                                         │
  │ • Valuation Engine: European Black-Scholes Analytical Formula                          │
  │ • Stressed Parameters: S_stressed = S × (1 + ΔS), σ_stressed = σ + Δσ, r_stressed = r + Δr│
  │ • Formulas:                                                                            │
  │     d_1 = [ln(S/K) + (r + 1/2 σ^2)T] / [σ √T],   d_2 = d_1 - σ √T                      │
  │     Call = S Φ(d_1) - K e^{-rT} Φ(d_2)                                                 │
  │     Put  = K e^{-rT} Φ(-d_2) - S Φ(-d_1)                                               │
  │     ΔMTM_option = Notional × (Price_stressed - Price_base)                             │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. File Inventory

* [`loans.py`](loans.py): Calculates base and stressed 12-month Expected Credit Loss for commercial loans.
* [`bonds.py`](bonds.py): Evaluates mark-to-market P&L on fixed-rate bonds using modified duration and credit spread sensitivity.
* [`swaps.py`](swaps.py): Evaluates interest rate swap mark-to-market changes under parallel yield curve shifts.
* [`options.py`](options.py): Evaluates Black-Scholes option revaluations under joint underlying price, implied volatility, and risk-free rate shocks.
