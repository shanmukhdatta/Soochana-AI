# Soochana AI — System Documentation & Specifications (`docs/`)

`docs/` provides comprehensive architectural diagrams, quantitative financial methodologies, test matrices, and user interface previews.

---

## 1. Documentation Index

* [**`architecture.png`**](architecture.png): End-to-end multi-tier system architecture diagram illustrating data flows from streaming feeds to models, quantitative labs, and presentation.
* [**`methodology.md`**](methodology.md): Rigorous mathematical and econometric specifications:
  - 3-stage ML formulation (FinBERT continuous sentiment, 5-class event taxonomy, XGBoost abnormal volatility deciles).
  - Module A constrained quadratic distance optimization under convex position, sector, and turnover bounds.
  - Module B horizon-separated balance sheet pricing (Trading Book instantaneous MTM vs. Banking Book 12-Month ECL).
* [**`validation.md`**](validation.md): Complete test matrix, validation logs, benchmark evidence, and out-of-distribution evaluation results.
* [**`screenshots/`**](screenshots/README.md): High-resolution interface captures across Quick View and Analyst View on desktop and mobile viewports.
