# Module B Portfolio & Scenarios (`data/module_b/`)

This directory contains the synthetic institutional balance sheet and adverse scenario definitions for the **Multi-Asset Stress Testing Laboratory**.

---

## 1. File Inventory

* [`synthetic_portfolio.json`](synthetic_portfolio.json): 30 institutional positions across 4 asset classes:
  - 12 Commercial Loans ($EAD$, $PD$, $LGD$, Sector)
  - 10 Corporate Bonds (Notional, Duration, Spread, Yield)
  - 4 Interest Rate Swaps (Notional, DV01, Direction, Tenor)
  - 4 Equity Options (Strike, Maturity, Volatility, Type)
* [`scenario_library.json`](scenario_library.json): Adverse scenario families:
  - Macroeconomic rate spikes and inflation shifts
  - Geopolitical supply chain disruption
  - Credit distress and rating downgrade events
  - Tech selloff and volatility expansion
* [`live_verification_results.json`](live_verification_results.json): Audit record verifying portfolio calculation outputs.
* [`results/`](results/): Stores demo and batch scenario execution runs.
