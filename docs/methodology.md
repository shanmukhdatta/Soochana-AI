# RiskPulse AI — Quantitative Methodology & Modeling Specifications

**Project:** RiskPulse AI — Explainable Financial Risk Intelligence  
**Competition:** S&P Global & CRISIL Campus Hackathon 2026  
**Document:** Mathematical Specifications, Model Architecture & Risk Methodology  

---

## 1. System Overview

RiskPulse AI bridges the gap between unstructured financial data (news publications and social discourse) and quantitative financial portfolio risk management. The architecture is organized into four interconnected tiers:

1. **Multi-Source Ingestion & Provenance:** Real-time collection, normalization, entity mapping, and deduplication across financial news and public market discussion.
2. **Three-Stage Machine Learning Risk Engine:**
   - FinBERT Continuous Sentiment Regressor ($\hat{s} \in [-1.0, +1.0]$ via Tanh activation)
   - FinBERT Event Taxonomy Classifier ($k \in \{1 \dots 5\}$ over calibrated financial categories)
   - XGBoost Return Impact Severity Engine (Decile scale $1 \dots 10$)
3. **Module A — Index Rebalancing Laboratory:** Signal-informed portfolio optimization with quadratic distance minimization subject to long-only, full investment, position deviation ($\pm 2\%$), sector deviation ($\pm 5\%$), and turnover limit ($10\%$).
4. **Module B — Multi-Asset Stress Testing Laboratory:** Horizon-separated scenario evaluation across trading book instruments (instantaneous Mark-to-Market for bonds, swaps, and options) and banking book assets (incremental 12-month Expected Credit Loss for commercial loans).

---

## 2. Three-Stage Machine Learning Risk Engine

### 2.1 Model 1: Sentiment Regressor
- **Architecture:** BERT-based sequence regressor (`FinBertSentimentRegressor`).
- **Regression Head:** Pooled contextual representation $\mathbf{h}_{\text{pooled}} \in \mathbb{R}^{768}$ passed through a dropout layer and linear projection with a hyperbolic tangent activation:
  $$\hat{s} = \tanh\left(\mathbf{w}_s^T \mathbf{h}_{\text{pooled}} + b_s\right), \quad \hat{s} \in [-1.0, +1.0]$$
- **Loss Function & Training:** Trained with Mean Squared Error (MSE) loss on continuous polarity labels.
- **Output:** Continuous sentiment intensity where $-1.0$ represents severe distress/pessimism, $0.0$ neutral market sentiment, and $+1.0$ strong expansionary optimism.

### 2.2 Model 2: Financial Event Classifier
- **Model identity caveat:** Runtime configuration references `yiyanghkust/finbert-tone`, while the original classifier training notebook initializes `nickmuchi/finbert-tone-finetuned-finance-topic-classification`. The original trained weights are not bundled; these sources must not be treated as the same verified checkpoint. The five-class runtime label contract is `num_labels = 5`.
- **Classification Head:** 5-way softmax distribution over verified financial event categories:
  $$P(\text{Event} = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{h}_{\text{pooled}} + b_k)}{\sum_{j=1}^{5} \exp(\mathbf{w}_j^T \mathbf{h}_{\text{pooled}} + b_j)}, \quad k \in \{1 \dots 5\}$$
- **Supported 5-Class Taxonomy (`configs/event_labels.json`):**
  1. `Geopolitical` (ID 0): Trade tariffs, international sanctions, cross-border military tensions, policy sanctions.
  2. `Macroeconomic` (ID 1): Central bank interest rates, inflation indicators, CPI/PPI releases, GDP reports, employment data.
  3. `Credit Event` (ID 2): Debt defaults, rating downgrades, debt restructuring, credit spread widening.
  4. `Merger / Acquisition` (ID 3): Corporate takeovers, asset divestitures, buyout bids, regulatory antitrust reviews.
  5. `Product Launch` (ID 4): New product announcements, technology rollouts, architecture upgrades, commercial releases.

### 2.3 Model 3: XGBoost Impact Severity Engine
- **Objective:** Estimate market impact severity decile ($1 \dots 10$) conditional on detected event probabilities, sentiment intensity, and market context.
- **Feature Vector Formulation (`configs/impact_model_feature_manifest.json`):**
  $$\mathbf{z} = \left[ \hat{s}, \; |\hat{s}|, \; P_0, \dots, P_4, \; \max_k P_k, \; \sigma_{20\text{d}}, \; v_{20\text{d}}, \; r_{5\text{d}}, \; r_{20\text{d}}, \; r_{\text{SPY}}, \; \sigma_{\text{SPY}, 20\text{d}}, \; c_{\text{event}}, \; \mathbf{e}_{\text{sector}}, \; \mathbf{r}_{\text{regime}} \right]$$
  where:
  - $\hat{s}, |\hat{s}|$: Continuous sentiment score and absolute sentiment intensity.
  - $P_0 \dots P_4$: Softmax probability vector across the 5 event classes.
  - $\max_k P_k$: Primary event classification confidence.
  - $\sigma_{20\text{d}}$: 20-day annualized rolling price return volatility of the target equity.
  - $v_{20\text{d}}$: 20-day rolling trading volume ratio.
  - $r_{5\text{d}}, r_{20\text{d}}$: Historical 5-day and 20-day cumulative equity returns.
  - $r_{\text{SPY}}, \sigma_{\text{SPY}, 20\text{d}}$: Market benchmark return and 20-day volatility for broad market context.
  - $c_{\text{event}}, \mathbf{e}_{\text{sector}}, \mathbf{r}_{\text{regime}}$: Event class index, sector one-hot encoding, and binary market regime indicators.
- **Target Formulation & Decile Mapping:**
  - Training target: continuous historical price movement $\max(|AR_{1D}|, |CAR_{3D}|)$.
  - Decile mapping: Mapped into integer severity score $1 \dots 10$ using calibrated quantile cutoffs $p_{10} \dots p_{90}$ (`configs/impact_decile_thresholds.json`).
  - **Important Interpretation:** Severity is an empirical decile indicator of market turbulence and attention; it is *not* a directional return forecast, guaranteed price change, or probability of default.

---

## 3. Module A: Constrained Index Rebalancing

### 3.1 Universe & Benchmark
- **Index Universe (`configs/module_a_config.json`):** 15 large-cap US equities:
  `AAPL`, `MSFT`, `NVDA`, `QCOM`, `GOOGL`, `CMCSA`, `AMZN`, `TSLA`, `BKNG`, `SBUX`, `COST`, `PEP`, `GILD`, `AMGN`, `CSX`.
- **Baseline Allocation:** Equal-weighted benchmark with baseline weights $w_{i, \text{base}} = \frac{1}{15} \approx 6.6667\%$.

### 3.2 Optimization Objective Function
The optimal portfolio weight vector $\mathbf{w}^* = [w_1^*, \dots, w_{15}^*]^T$ minimizes the squared Euclidean distance from the signal-informed target allocation:
$$\min_{\mathbf{w}} \quad \frac{1}{2} \sum_{i=1}^{15} \left( w_i - w_{i, \text{target}} \right)^2$$
where:
$$w_{i,\text{target}} = \frac{1}{15} + \beta \left(a_i-\frac{1}{15}\sum_{j=1}^{15}a_j\right), \quad \beta=0.025.$$
Here $a_i$ is the ticker's deduplicated, age-weighted sentiment/event/impact aggregate from `app/module_a/signals.py`. The strategy creates centered target weights and the SLSQP optimizer projects them into the allowed portfolio constraints. This is a heuristic event signal, not a learned directional-return target.

### 3.3 Financial & Regulatory Constraints (`app/module_a/constraints.py`)
The quadratic program is solved subject to strict investment constraints:
1. **Full Investment:**
   $$\sum_{i=1}^{15} w_i = 1.0000$$
2. **Long-Only Requirement:**
   $$w_i \ge 0, \quad \forall i \in \{1 \dots 15\}$$
3. **Single-Stock Deviation Limit:**
   $$|w_i - w_{i, \text{base}}| \le 0.02 \quad (\pm 2 \text{ percentage points})$$
4. **Sector Deviation Limit:**
   $$\left| \sum_{i \in \text{Sector}_k} w_i - \sum_{i \in \text{Sector}_k} w_{i, \text{base}} \right| \le 0.05 \quad (\pm 5 \text{ percentage points}), \quad \forall k$$
5. **Maximum One-Way Turnover Budget:**
   $$\frac{1}{2} \sum_{i=1}^{15} |w_i - w_{i, \text{current}}| \le 0.10 \quad (10\% \text{ maximum turnover})$$
   Enforced via auxiliary absolute-deviation variables $u_i \ge |w_i - w_{i, \text{current}}|$ such that $\frac{1}{2} \sum u_i \le 0.10$.
6. **Transaction Cost Model:**
   Transaction costs are 10 basis points per dollar traded; the estimated transaction cost fraction is $2\times0.0010\times$ one-way turnover.

---

## 4. Module B: Multi-Asset Stress Testing

### 4.1 Portfolio Structure (`data/module_b/synthetic_portfolio.json`)
A calibrated 30-position institutional reference portfolio comprising:
- **12 Commercial Loans:** $5,000,000 total Exposure at Default (EAD).
- **10 Corporate Bonds:** $3,500,000 total Market Value (MV).
- **4 Interest Rate Swaps:** $1,000,000 total Notional.
- **4 FX / Equity Options:** $500,000 reference Notional ($\text{contracts} \times \text{multiplier} \times \text{spot}$).
- **Total Positions:** 30 positions.
- **Total Derivative Reference Notional:** $1,500,000.

> [!IMPORTANT]
> **Non-Additive Balance Sheet Notice:** Loan EAD ($5.0M), Bond Market Value ($3.5M), and Derivative Reference Notional ($1.5M) represent fundamentally distinct financial risk measures across credit exposure, market liquidation value, and off-balance sheet notional. They are **not** summed into Net Asset Value (NAV).

### 4.2 Separation of Financial Horizons
RiskPulse AI enforces strict separation between valuation horizons:
- **Instantaneous Mark-to-Market (MTM):** Evaluates immediate trading book liquidation value shifts upon shock arrival ($t=0$).
- **Incremental 12-Month Expected Credit Loss (ECL):** Evaluates banking book loan impairment under IFRS 9 / CECL Stage 1 and Stage 2 credit migration over a 12-month horizon ($t=1 \text{ year}$).

### 4.3 Asset Pricing Models (`app/module_b/pricing/`)

#### Corporate Bonds (`pricing/bonds.py` — Duration + Rate-Convexity Sensitivity)
Bond price revaluation under risk-free rate shock $\Delta r$ (bps) and credit spread shock $\Delta s$ (bps):
$$\Delta r_{\text{dec}} = \frac{\Delta r_{\text{bps}}}{10000}, \quad \Delta s_{\text{dec}} = \frac{\Delta s_{\text{bps}}}{10000}$$
$$\Delta \text{MTM}_{\text{rate}} = \text{MV} \cdot \left( -MD \cdot \Delta r_{\text{dec}} + \frac{1}{2} C \cdot (\Delta r_{\text{dec}})^2 \right)$$
$$\Delta \text{MTM}_{\text{spread}} = -\text{MV} \cdot SD \cdot \Delta s_{\text{dec}}$$
$$\Delta \text{MTM}_{\text{Bond}} = \Delta \text{MTM}_{\text{rate}} + \Delta \text{MTM}_{\text{spread}}$$
where $MD$ is Modified Duration, $C$ is Convexity, and $SD$ is Spread Duration. Convexity is applied to risk-free yield moves; spread moves are linearized.

#### Commercial Loans (`pricing/loans.py` — 12-Month Incremental ECL)
Evaluates credit loss under baseline and stressed 12-month default probabilities:
$$\text{PD}_{\text{stressed}} = \min(1.0, \; \text{PD}_0 \cdot m_{\text{PD}})$$
$$\text{ECL}_{\text{base}} = \text{EAD} \cdot \text{PD}_0 \cdot \text{LGD}$$
$$\text{ECL}_{\text{stressed}} = \text{EAD} \cdot \text{PD}_{\text{stressed}} \cdot \text{LGD}$$
$$\Delta \text{ECL}_{\text{Loan}} = \text{ECL}_{\text{stressed}} - \text{ECL}_{\text{base}} = \text{EAD} \cdot (\text{PD}_{\text{stressed}} - \text{PD}_0) \cdot \text{LGD}$$

#### Interest Rate Swaps (`pricing/swaps.py` — Signed DV01 Sensitivity)
Evaluates fixed-for-floating swap sensitivity under interest rate shifts:
$$\text{sign} = \begin{cases} +1 & \text{if side} = \text{'pay\_fixed'} \\ -1 & \text{if side} = \text{'receive\_fixed'} \end{cases}$$
$$\Delta \text{MTM}_{\text{Swap}} = \text{sign} \cdot \text{DV01} \cdot \Delta r_{\text{bps}}$$

#### Options (`pricing/options.py` — Full European Black-Scholes Repricing)
Full non-linear repricing under equity price shock $\Delta S_{\%}$, volatility shift $\Delta \sigma$, and rate shock $\Delta r_{\text{bps}}$:
$$S_{\text{new}} = S_0 \cdot \left( 1 + \frac{\Delta S_{\%}}{100} \right), \quad \sigma_{\text{new}} = \sigma_0 + \Delta \sigma, \quad r_{\text{new}} = r_0 + \frac{\Delta r_{\text{bps}}}{10000}$$
$$\text{Price}_{\text{base}} = \text{BS}(S_0, K, T, \sigma_0, r_0, q)$$
$$\text{Price}_{\text{stressed}} = \text{BS}(S_{\text{new}}, K, T, \sigma_{\text{new}}, r_{\text{new}}, q)$$
$$\text{scale} = \text{contracts} \cdot \text{multiplier} \cdot (\text{+1 if long else -1})$$
$$\Delta \text{MTM}_{\text{Option}} = \left( \text{Price}_{\text{stressed}} - \text{Price}_{\text{base}} \right) \cdot \text{scale}$$
using analytical standard normal CDF $\Phi(d)$ with continuous dividend yield $q$.

### 4.4 Advanced Stress Capabilities

#### What-If Interactive Analysis (`app/module_b/what_if.py`)
Allows risk managers to scale individual position exposures with multiplier $\alpha_j \in [0.0, 2.0]$:
$$\Delta \text{Loss}_{\text{net}} = \sum_{j=1}^{30} \alpha_j \cdot \text{Loss}_j(\mathbf{x}_{\text{scenario}}) - \sum_{j=1}^{30} \text{Loss}_j(\mathbf{x}_{\text{scenario}})$$

#### Reverse Stress Testing (`app/module_b/reverse_stress.py`)
For a chosen budget and a scenario multiplier, the implementation samples a finite grid starting at 0 through a configured maximum multiplier (defaults: maximum 4, 81 points). It reports the **first sampled loss-budget breach**, not an exact threshold. Derivative hedges can make losses non-monotonic; therefore, the implementation deliberately uses a grid **rather than bisection**. Invalid shocked instrument values can truncate the grid.
