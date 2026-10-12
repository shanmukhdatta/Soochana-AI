# Custom Portfolio Studio & Retail Investor Shield (`backend/app/portfolio/`)

This directory houses the dynamic custom portfolio rebalancer, market covariance harvester, live news scoring pipeline, and retail investor protection engine for **Soochana AI**.

---

## 1. Architectural Role & Purpose

While Module A operates over a fixed 15-stock research index and Module B evaluates a synthetic 30-position institutional ledger, the **Custom Portfolio Studio** empowers retail investors and portfolio managers to:
1. **Define Arbitrary Stock Portfolios:** Input any $N$ equity tickers (e.g. `AAPL`, `NVDA`, `MSFT`, `JPM`, `KO`) and capital weights.
2. **Harvest Live Market Covariance:** Pulls 6-month to 1-year historical adjusted close returns dynamically via `yfinance` to compute empirical covariance ($\Sigma$) and volatility matrices.
3. **Stream & Score Ticker News:** Ingests live financial headlines for the exact portfolio constituents via real-time feeds and scores them through the local **FinBERT Event Classifier**, **FinBERT Sentiment Regressor**, and **XGBoost Impact Engine**.
4. **Filter Panic & Sensationalized Clickbait (FUD):** Distinguishes between low-impact media noise (`PANIC_NOISE` — preventing emotional panic selling) and high-severity balance sheet events (`GENUINE_RISK`).
5. **Solve Convex Quadratic Rebalancing:** Optimizes target weights subject to budget, turnover, and concentration constraints.
6. **Generate Everyday Plain-English Metrics:** Computes a 0–100 **Portfolio Health Score**, dollar-denominated trade execution orders ($ / ₹), and relatable **Macro Crisis Stress Simulations** (Recession, AI Regulatory Clampdown, Oil Shock).

---

## 2. Directory Structure

```text
backend/app/portfolio/
├── __init__.py           # Package marker
├── custom_engine.py      # Core quantitative optimizer, covariance builder, and scoring engine
├── router.py             # FastAPI REST endpoints (/api/portfolio/custom/...)
└── README.md             # Architecture and module documentation
```

---

## 3. Mathematical & Algorithmic Formulation

### A. Dynamic Covariance Matrix $\Sigma$
Given daily log-returns $R_{i,t} = \ln(P_{i,t} / P_{i,t-1})$ over lookback window $T$:
$$\Sigma_{ij} = \frac{252}{T-1} \sum_{t=1}^{T} (R_{i,t} - \bar{R}_i)(R_{j,t} - \bar{R}_j)$$

### B. Alpha / Risk Signal Vector $\alpha$
For headline $k \in \mathcal{H}_i$ belonging to ticker $i$:
$$\alpha_i = \frac{1}{|\mathcal{H}_i|} \sum_{k} s_k \times \left(0.5 + 0.1 \times d_k\right)$$
where $s_k \in [-1.0, +1.0]$ is FinBERT sentiment polarity and $d_k \in [1 \dots 10]$ is XGBoost abnormal volatility decile.

### C. Convex Quadratic Rebalancer
$$\min_{w} \ (w - w_0)^T \Sigma (w - w_0) - \lambda (\alpha^T w) + \gamma \sum |w_i - w_{0,i}|$$
*Subject to:*
$$\sum_{i=1}^n w_i = 1.0, \quad 0.02 \le w_i \le \max\left(0.40, \frac{2}{n}\right)$$

### D. 0–100 Portfolio Health Score
$$\text{Health Score} = \text{Diversification Points (0–35)} + \text{Volatility Shield (0–35)} + \text{News Sentiment (0–30)}$$

---

## 4. API Endpoints

* `GET /api/portfolio/custom/presets`: Returns starter portfolios (Mag 7 AI, All-Weather Balanced, Defensive Dividend, Global Tech).
* `POST /api/portfolio/custom/analyze`: Executes full live news ingestion, NLP scoring, quadratic rebalancing, stress simulations, and plain-English recommendations.
