# RiskPulse StressLab — Module B v1.1 AUDITED (Synthetic Portfolio Stress Testing)

**Add-on for an EXISTING `c2c` Unified Risk Engine + Module A installation.** Does not retrain any model or change existing Risk Engine / Module A logic. Run locally on CPU. Nothing in this package contains real model weights or user financial data.

## What is included

- **30 synthetic positions:** 12 corporate loans, 10 fixed-rate bonds, 4 simplified USD interest-rate swaps, 4 European equity options.
- **Five scenario families** matching the event classifier, with mild/moderate/severe templates; issuer-specific adjustments for Credit Event, M&A and Product Launch.
- **Four pricing/risk mechanisms:** 12-month loan expected-credit-loss increment; bond duration+rate convexity/spread-duration instantaneous MTM approximation; swap signed DV01 MTM; Black–Scholes European option repricing.
- **Risk attribution:** position, sector, ticker, and instrument class with negative loss contributions for gains/hedges. Issuer override keys are case-normalized; unmatched issuer/sector overrides are surfaced in `override_coverage` and `warnings` instead of silently ignored.
- **What-if:** scale existing positions and compare identical scenario before/after.
- **Reverse stress:** sample a deterministic severity-multiplier grid to find first budget breach; no monotonicity assumption, explicit truncation if scenarios become physically impossible.
- **News integration:** reuses `request.app.state.pipeline` from existing backend (one copy of FinBERT/XGBoost); shows suggested shock template **without executing it unless `run_stress=true`**.
- **Tests** and offline demo (no model downloads needed).

## VERY IMPORTANT: valuation basis and limitations

The illustrative **$10M headline of exposure** represents $5M loan EAD, $3.5M bond current market value and $1.5M derivative *reference notional* (swap notional plus option underlying reference notional). This is **NOT** a $10M portfolio NAV: loan EAD, bond MV, and derivative notional are not directly additive. The API returns each separately. Option market values are modelled; swap initial market values are not modelled and are assumed 0 for stress deltas.

The output **separates** instantaneous changes in bond/swap/option market values from changes in **12-month expected loan credit losses**. The optional `illustrative_mixed_horizon_net_effect_usd` is a *scenario screening indicator*, never realized P&L, portfolio market value, regulatory capital, Value-at-Risk, real economic loss or a forecast. The scenario templates, loss-given-default, baseline PDs and instrument parameters are synthetic. Do not double-count these sensitivities with external mark-to-market loss models.

Extreme custom rate/spread inputs that make the approximate stressed bond value negative are rejected rather than reported as plausible bond prices. No stochastic credit default simulation, dependence/correlation, credit migration, yield-curve bootstrapping, collateral, CSA/netting, liquidity, funding, basis, transaction costs, option early exercise or regulatory stress testing is implemented. Swap DV01 and bond duration are **linear/second-order approximations**, not full revaluations. For options we hold time to maturity unchanged during instantaneous stress and assume European exercise and continuous dividend yield. **The Impact Score predicts the magnitude of a historical equity abnormal return, not loan PDs, yields, spreads or actual multi-asset loss.** The impact-to-severity mapping is an unvalidated UI heuristic and must be labelled as such.

## Where to put files on Windows

**Back up your working `c2c` first.** Extract this ZIP into `c2c/module_b_files/` for staging. Use the following mapping (copy into existing folders; do not replace directory trees):

| From `c2c/module_b_files/` | To `c2c/` |
|---|---|
| `app/module_b/` | `app/module_b/` |
| `configs/module_b_config.json` | `configs/module_b_config.json` |
| `data/module_b/synthetic_portfolio.json` | `data/module_b/synthetic_portfolio.json` |
| `data/module_b/scenario_library.json` | `data/module_b/scenario_library.json` |
| `scripts/validate_module_b_portfolio.py` | `scripts/validate_module_b_portfolio.py` |
| `scripts/run_module_b_demo.py` | `scripts/run_module_b_demo.py` |
| `scripts/verify_module_b_scenarios.py` | `scripts/verify_module_b_scenarios.py` |
| `tests/test_module_b_*.py` | `tests/` |
| `install_module_b.py` | `install_module_b.py` |
| `requirements_module_b.txt` | `requirements_module_b.txt` |
| `README_MODULE_B.md`, `AUDIT_NOTES_MODULE_B.md`, `ANTIGRAVITY_MODULE_B_PROMPT.md` | same-named files in `c2c/` |

Keep `models/`, `app/engine/`, `app/module_a/`, existing `app/services/`, `configs/impact_model_feature_manifest.json`, `.env`, `requirements.txt` and original data untouched.

Exact minimal PowerShell commands **after merging** into root `c2c/`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements_module_b.txt
.\.venv\Scripts\python.exe install_module_b.py
.\.venv\Scripts\python.exe install_module_b.py
.\.venv\Scripts\python.exe -m scripts.validate_module_b_portfolio
.\.venv\Scripts\python.exe -m pytest -q tests -k module_b
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.verify_module_b_scenarios
.\.venv\Scripts\python.exe -m scripts.run_module_b_demo
.\.venv\Scripts\python.exe -m scripts.smoke_inference
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

The `install_module_b.py` script is idempotent and makes `app/main.py.before_module_b.bak` before adding a single import and router inclusion at the end of `app/main.py`. Review the diff afterward. It does **not** modify model configurations or existing registered routers. Do not start a second server if port 8000 is already in use; instead restart the existing one.

## Six API endpoints

- `GET /module-b/portfolio` — synthetic holdings, separate exposure summaries
- `GET /module-b/scenarios` — 15 assumptions templates; **`__ISSUER__` is a template placeholder** in some configurations
- `POST /module-b/stress` — run explicitly specified scenario
- `POST /module-b/stress-from-news` — propose a scenario from a real `/analyze` result; **requires real loaded models even in suggest-only mode**; executes stress only with `run_stress=true`
- `POST /module-b/what-if` — compare before/after scaling selected positions
- `POST /module-b/reverse-stress` — grid search for first loss-budget breach (may be truncated by physical model bounds)

See `/docs` after server starts. **Check `override_coverage` in every result**: a typo in a sector or an issuer not present in the portfolio is reported in the response and its targeted shock is not applied to any holding. The reverse-stress result also includes this coverage information. **Use issuer-substituted scenarios**, e.g. the `suggested_scenario` returned from `/stress-from-news`, or replace `__ISSUER__` with the actual supported ticker manually before `POST /stress`. The `__ISSUER__` marker from GET `/scenarios` is a placeholder, not an asset.

### Example: deterministic stress, with no NLP model required

`POST http://127.0.0.1:8000/module-b/stress`

```json
{
  "scenario": {
    "name": "Custom hypothetical macro shock",
    "event_class": "Macroeconomic",
    "severity": "custom",
    "shocks": {"rate_bps":100, "spread_bps":150,"equity_pct":-15,"volatility_points":0.05,"pd_multiplier":2}
  }
}
```

### Example: genuine real-model news bridge

`POST http://127.0.0.1:8000/module-b/stress-from-news`

```json
{
  "news": {
    "text": "Apple bonds face downgrade following liquidity concerns.",
    "ticker": "AAPL",
    "published_at": "2024-05-15T14:30:00Z",
    "source": "financial_news"
  },
  "severity": "auto",
  "run_stress": false
}
```

First review the `suggested_scenario` and event quality flags. Repost with `run_stress=true` only after explicitly accepting the assumptions; results remain research-only even if executed.

### Example: reverse stress threshold

```json
{
  "scenario": {
    "name": "Illustrative credit deterioration",
    "event_class": "Credit Event",
    "shocks": {"rate_bps":25,"spread_bps":100,"equity_pct":-10,"volatility_points":0.05,"pd_multiplier":1.8}
  },
  "loss_budget_usd":100000,
  "max_multiplier":2,
  "steps":81
}
```

## Acceptance checks for Antigravity

1. Confirm merge adds only `app/module_b/`, its own config/data/scripts/tests/docs and a small backed-up `app/main.py` router registration.
2. Confirm `GET /health`, `/analyze`, `/analyze/batch`, Module A endpoints still work with real models; no second model instance.
3. Confirm all module B and original tests pass; all 15 templates run; zero shocks give no incremental effect; 30-position totals and position attribution reconcile.
4. Confirm original XGBoost `base_score = 0.021967076942109524` and frozen model parameters untouched.
5. Confirm *real* `/module-b/stress-from-news` call produces a genuine model output and proposed scenario; explicit approved version produces actual stress result.
6. Report separate instantaneous MTM, incremental 12-month ECL, and indicative mixed-horizon screening results. Do not describe the last as actual NAV/P&L.
7. CPU benchmark initial `/module-b/stress` latency; real model `/stress-from-news` latency separately; document limitations.
8. Validate lowercase issuer override normalization, duplicate ticker-key rejection, unmatched sector/issuer override warnings and extreme bond-price rejection. Report clearly whether Module B local software is ready; **do not claim Oracle/public deployment or financial calibration are verified** without direct evidence.
