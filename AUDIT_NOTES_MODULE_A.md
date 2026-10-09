# Module A v1.1 — independent source-code cross-check

## Verified using available source and deterministic tests

- Audited the actual original Module A ZIP and the original unified Risk Engine source ZIP together.
- Reproduced 17 passing Module A tests and 16 passing original backend tests before the corrections.
- Fixed a genuine logic defect: in `full_event_aware`, a single article's source weight, age/recency decay, and event confidence cancelled out from a normalized weighted mean. The revised strategy uses `evidence_prior_weight = 0.5` to shrink weak evidence toward neutral. This is a **policy heuristic fixed before backtest**, not a fitted parameter.
- Added two regression tests for this defect; 35 full tests pass on the combined available source.
- Expanded hard-stop future-data column filtering in historical signal CSVs.
- Made the idealized same-session closing-price fill assumption explicit in the backtest limitations. This is a research simulation, not a directly executable market-on-close strategy.
- Tested 250 deterministic diverse portfolio weight proposals: no optimizer errors, portfolio weights sum to one and the turnover cap was respected.
- Tested `install_module_a.py` twice: it registered the router once and left the original main file backed up; mocked app integration returned `/module-a/index` 200 and `/docs` 200. `/health` 503 is expected in the no-weights testing environment.

## Not verified in this environment

- Real trained model weights and their corrected XGBoost `base_score` are NOT in this ZIP. No claims of live model-inference validation.
- No original Kaggle Step-2 test parquet or live Yahoo price feed was available for a real historical backtest. Test fixtures were synthetic.
- Historical event stream is FNSPID financial news only. The hackathon's separate social data source is NOT part of this add-on.
- The existing sentiment/sign heuristics have not been shown to outperform the baseline out of sample. Do not tune strategies repeatedly on the frozen holdout and still present that holdout as untouched.
- Deployment hardening (auth/rate limits, job storage, latency tests) and dashboard are not included.

## Installation

**Prefer this revised ZIP over the older v1 ZIP.** Extract on top of your **backed-up** `c2c/` project, preserving `models/`, `app/main.py`, and the existing Impact Engine config/model corrections. Follow `README_MODULE_A.md` and have Antigravity execute a real model + price data end-to-end test.
