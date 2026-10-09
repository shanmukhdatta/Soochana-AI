# Module B v1.1 — Audit notes (independent second-pass cross-check)

## Validated by package author

- Module B unit tests cover known Black–Scholes call/put benchmark, put–call parity, bond rate and spread bp arithmetic, signed swap DV01, PD cap, incremental 12-month loan ECL, zero-shock neutrality, source units, issuer scoped overrides, portfolio structure, loss reconciliation, what-if nonmutation, reverse stress threshold and invalid shocks.
- API tests in isolation use a deterministic **FakePipeline**. They prove API contracts and the explicit scenario approval gate, **NOT that your real FinBERT/XGBoost is working in this sandbox**.
- Tests are also executed after physically merging source-only initial Unified Risk Engine + Module A v1.1 add-on + Module B into an isolated test directory. No model weights are supplied in this ZIP.
- Offline scenario demo executes with synthetic portfolio and outputs JSON at `data/module_b/results/`; reported results are **assumption-driven outputs**, not predictions or financial returns.
- `install_module_b.py` is idempotent and backs up existing `app/main.py` before appending router registration. Source bundle never contains `app/main.py`, `models/` or existing module files.

## Major caveats before public demonstrations

1. No real loan/bond/swap/option portfolio market quotes or empirically calibrated shocks. Every scenario is synthetic and labelled accordingly.
2. Instantaneous MTM and incremental 12-month loan expected credit losses have mismatched risk horizons. They are reported separately; mixed indicator is an *illustrative combined screening signal*, not real economic P&L or regulatory capital.
3. Simplified rate/spread shocks may be economically inconsistent; no correlated macroeconomic scenario calibration is claimed.
4. Event class and Impact Score 1–10 are **not calibrated** to default probability, spread shifts, interest-rate changes or credit loss. Severity tier mapping is explicitly a configurable heuristic and user review is expected.
5. Options use European exercise Black–Scholes; swaps signed linear DV01; bonds duration/convexity. No full contract-term repricing or regulatory validation.
6. Actual legacy Risk Engine weights and live public market data are available only on the user's computer. Run `POST /module-b/stress-from-news` there and preserve actual HTTP responses and timestamps.
7. No live public-cloud deployment, HTTPS, authentication, rate limiting, or monitoring included. Local API demonstration only.

## Readiness criteria

- Offline software pass: full test suite passes with working base app and Module A; valid 30-position portfolio; 15 scenario templates execute; price arithmetic checked; app router registers once; all previous routes retained.
- End-to-end pass: **requires Antigravity on the user's PC** to load actual models and test the news-to-scenario bridge, including invalid input paths and CPU performance.
- Financial validation: deliberately limited to numerical consistency and synthetic demonstrations; no claims of regulatory use, real predictive loss or profitable strategy.

## Independent second-pass audit (October 9, 2026)

A fresh merge of the source-only Unified Risk Engine, Module A v1.1, and Module B was executed locally. **91 total tests passed** after the following additional changes:

- Issuer override keys such as `aapl` are canonicalized to `AAPL`, and duplicate keys after canonicalization fail validation.
- Issuer or sector overrides matching zero current holdings are **reported**, not silently ignored, in `override_coverage` and human-readable warnings. A nonheld issuer may legitimately have no direct exposure; therefore a warning is more appropriate than fabricating an effect.
- Reverse-stress output now retains the override-coverage result as well.
- Bond sensitivity scenarios generating a negative stressed market value fail with a clear ValueError rather than reporting an economically impossible price.
- Regression tests cover each edge case. All 15 scenario-library templates still run and reconcile.
- Merged FastAPI checks under `SKIP_MODEL_LOAD=1` confirm `/module-a/index`, `/module-b/portfolio`, `/module-b/scenarios`, and `/module-b/stress` are available; model-required `/health` and `/module-b/stress-from-news` correctly return HTTP 503.

**Not verified in this container:** the user's two FinBERT weights and corrected XGBoost model actually loading in their Antigravity environment; live market-data access; CPU/Oracle Cloud performance or authentication. These are required follow-up checks.
