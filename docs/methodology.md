# RiskPulse AI — Quantitative Methodology & Modeling Specifications

**Project:** RiskPulse AI — Explainable Financial Risk Intelligence  
**Competition:** S&P Global & CRISIL Campus Hackathon 2026  
**Document:** Mathematical Specifications, Model Architecture & Risk Methodology  

---

## 1. System Overview

RiskPulse AI bridges the gap between unstructured financial data (news publications and social discourse) and quantitative financial portfolio risk management. The architecture is organized into four interconnected tiers:

1. **Multi-Source Ingestion & Provenance:** Real-time collection, normalization, entity mapping, and deduplication.
2. **Three-Stage Machine Learning Risk Engine:**
   - FinBERT Continuous Sentiment Regressor ($s \in [-1, +1]$)
   - FinBERT Event Taxonomy Classifier ($k \in \{1 \dots 10\}$)
   - XGBoost Return Impact Severity Engine (Decile scale $1 \dots 10$)
3. **Module A — Index Rebalancing Laboratory:** Signal-informed portfolio optimization with quadratic turnover control and regulatory concentration constraints.
4. **Module B — Multi-Asset Stress Testing Laboratory:** Horizon-separated scenario evaluation across trading book instruments (instantaneous Mark-to-Market) and banking book assets (12-month Expected Credit Loss).

---

## 2. Three-Stage Machine Learning Risk Engine

### 2.1 Model 1: Sentiment Regressor
- **Base Architecture:** `ProsusAI/finbert` (BERT-base-uncased fine-tuned on financial text).
- **Modification:** Replaced the discrete 3-class classification head with a dense linear projection layer:
  $$\hat{s} = \mathbf{w}_s^T \mathbf{h}_{\text{[CLS]}} + b_s, \quad \hat{s} \in [-1.0, +1.0]$$
  where $\mathbf{h}_{\text{[CLS]}} \in \mathbb{R}^{768}$ is the pooled contextual representation of the tokenized text.
- **Output:** Continuous sentiment intensity where $-1.0$ represents severe pessimism/downgrade and $+1.0$ represents strong optimism/expansion.

### 2.2 Model 2: Financial Event Classifier
- **Base Architecture:** `ProsusAI/finbert` fine-tuned on financial event sequences.
- **Classification Head:** 10-way softmax distribution over formal financial event taxonomies:
  $$P(\text{Event} = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{h}_{\text{[CLS]}} + b_k)}{\sum_{j=1}^{10} \exp(\mathbf{w}_j^T \mathbf{h}_{\text{[CLS]}} + b_j)}$$
- **Supported Taxonomy:**
  1. `Credit Event` (Default, rating downgrade, credit spread widening)
  2. `Dividend / Buyback` (Capital return announcements, dividend changes)
  3. `Earnings` (Quarterly reports, guidance revisions, revenue beats/misses)
  4. `Employment` (Layoffs, hiring freezes, workforce restructuring)
  5. `Executive Move` (CEO/CFO resignations, management turnover)
  6. `Legal / Regulatory` (Antitrust probes, compliance penalties, lawsuits)
  7. `Macroeconomic` (Interest rate decisions, inflation releases, GDP prints)
  8. `Merger / Acquisition` (Takeover bids, divestitures, buyout rumors)
  9. `Partnership` (Joint ventures, enterprise vendor agreements)
  10. `Product Launch` (Key product announcements, tech roadmap releases)

### 2.3 Model 3: XGBoost Impact Severity Engine
- **Objective:** Predict the return volatility and market movement decile ($1 \dots 10$) conditional on the detected event and current market environment.
- **Feature Vector Formulation:**
  $$\mathbf{z} = \left[ \hat{s}, \; P_1, P_2, \dots, P_{10}, \; \sigma_{15}, \; \mathbf{r}_{\text{regime}}, \; \mathbf{e}_{\text{sector}} \right]$$
  where:
  - $\hat{s}$: Continuous sentiment score from Model 1.
  - $P_1 \dots P_{10}$: Softmax probabilities from Model 2.
  - $\sigma_{15}$: 15-day rolling return volatility of the target equity.
  - $\mathbf{r}_{\text{regime}}$: Vector of binary market regime indicators (e.g., elevated market volatility, low volatility).
  - $\mathbf{e}_{\text{sector}}$: One-hot encoded sector membership.
- **Decile Mapping:** The continuous regression prediction is mapped into discrete impact tiers $1 \dots 10$ using calibrated historical quantile thresholds (`configs/impact_decile_thresholds.json`).

---

## 3. Module A: Constrained Index Rebalancing

### 3.1 Universe & Benchmark
- **Index Universe:** 15 large-cap US equities spanning Tech, Financials, Healthcare, and Consumer Goods.
- **Baseline Allocation:** Equal-weighted benchmark with baseline weights $w_{i, \text{base}} = \frac{1}{15} \approx 6.67\%$.

### 3.2 Optimization Objective Function
The optimal weight vector $\mathbf{w}^* = [w_1^*, \dots, w_{15}^*]^T$ minimizes the penalized distance from the risk-adjusted target allocation:
$$\min_{\mathbf{w}} \quad \frac{1}{2} \sum_{i=1}^{15} \left( w_i - w_{i, \text{target}}(\mathbf{s}) \right)^2 + \lambda \sum_{i=1}^{15} |w_i - w_{i, \text{base}}|$$
where:
- $w_{i, \text{target}}(\mathbf{s}) = w_{i, \text{base}} \cdot \left( 1 + \alpha \cdot \hat{s}_i \cdot \frac{\text{Impact}_i}{10} \right)$ is the raw unconstrained signal-tilted weight.
- $\lambda$ is the turnover aversion penalty parameter.

### 3.3 Financial & Regulatory Constraints
The solution is solved subject to strict real-world portfolio constraints:
1. **Full Investment:**
   $$\sum_{i=1}^{15} w_i = 1.0000$$
2. **Long-Only Constraint:**
   $$w_i \ge 0, \quad \forall i \in \{1 \dots 15\}$$
3. **Single-Name Concentration Cap:**
   $$w_i \le 0.15 \quad (15\%), \quad \forall i \in \{1 \dots 15\}$$
4. **Sector Concentration Cap:**
   $$\sum_{i \in \text{Sector}_k} w_i \le 0.35 \quad (35\%), \quad \forall k$$
5. **Turnover Budget Limit:**
   $$\frac{1}{2} \sum_{i=1}^{15} |w_i - w_{i, \text{base}}| \le 0.05 \quad (5\%)$$

---

## 4. Module B: Multi-Asset Stress Testing

### 4.1 Portfolio Structure
A calibrated 30-position synthetic financial institution portfolio ($10.0M total balance sheet exposure) containing:
- **10 Commercial & Corporate Loans** ($5.0M total EAD): Subject to banking book credit risk.
- **10 Corporate Bonds** ($3.5M total Market Value): Fixed-rate investment grade and high-yield instruments.
- **5 Interest Rate Swaps** ($1.0M total Notional): Receiver and payer fixed/floating contracts.
- **5 FX Options** ($0.5M total Notional): European calls and puts.

### 4.2 Separation of Financial Horizons
A core architectural feature of RiskPulse AI is the explicit separation of valuation horizons:
- **Instantaneous Mark-to-Market (MTM):** Reflects immediate trading book liquidation value shifts upon shock arrival ($t=0$).
- **12-Month Expected Credit Loss (ECL):** Reflects annual banking book loan loss provisioning under IFRS 9 / CECL Stage 1 and Stage 2 credit migration ($t=1\text{ year}$).

The system tracks these metrics separately and computes their net mixed-horizon effect while maintaining financial clarity.

### 4.3 Asset Pricing Models

#### Corporate Bonds (Duration + Convexity MTM)
The percentage price change of a fixed-coupon bond under a yield shock $\Delta y = \Delta y_{\text{rf}} + \Delta s_{\text{credit}}$ is given by the second-order Taylor expansion:
$$\frac{\Delta P}{P} \approx -MD \cdot \Delta y + \frac{1}{2} C \cdot (\Delta y)^2$$
$$\Delta \text{MTM}_{\text{Bond}} = \text{MV} \cdot \left( -MD \cdot \Delta y + \frac{1}{2} C \cdot (\Delta y)^2 \right)$$
where $MD$ is Modified Duration and $C$ is Convexity.

#### Commercial Loans (Basel II/III ECL)
Expected credit loss is evaluated under baseline and stressed conditions:
$$\text{ECL}_0 = \text{EAD} \times \text{PD}_0 \times \text{LGD}_0$$
$$\text{ECL}_{\text{stressed}} = \text{EAD} \times \min\left(1.0, \; \text{PD}_0 \cdot m_{\text{PD}}\right) \times \min\left(1.0, \; \text{LGD}_0 \cdot m_{\text{LGD}}\right)$$
$$\Delta \text{ECL}_{\text{Loan}} = \text{ECL}_{\text{stressed}} - \text{ECL}_0$$

#### Interest Rate Swaps (DV01 Re-pricing)
For an interest rate swap with dollar value of a basis point $\text{DV01}$:
$$\Delta \text{MTM}_{\text{Swap}} = \text{Direction} \times \text{DV01} \times \left( \frac{\Delta y_{\text{rf}}}{10^{-4}} \right)$$
where $\text{Direction} = +1$ for payer swaps and $-1$ for receiver swaps.

#### Options (Black-Scholes Delta-Gamma Re-pricing)
$$\Delta \text{MTM}_{\text{Option}} = \text{Notional} \cdot \left( \Delta \cdot \Delta S + \frac{1}{2} \Gamma \cdot (\Delta S)^2 + \mathcal{V} \cdot \Delta \sigma \right)$$

### 4.4 Advanced Stress Capabilities

#### What-If Analysis
Allows risk analysts to interactively adjust position exposure multipliers $\alpha_j \in [0.0, 2.0]$ to evaluate hedging or position sizing decisions:
$$\Delta \text{Loss}_{\text{net}} = \sum_{j} \alpha_j \cdot \text{Loss}_j(\mathbf{x}_{\text{scenario}}) - \sum_{j} \text{Loss}_j(\mathbf{x}_{\text{scenario}})$$

#### Reverse Stress Testing
Identifies the minimum shock scalar $\lambda^* \ge 1.0$ that causes total portfolio loss to breach a predefined institution capital buffer $B_{\text{limit}}$:
$$\lambda^* = \arg\min_{\lambda} \left\{ \text{TotalLoss}(\lambda \cdot \mathbf{x}_{\text{scenario}}) \ge B_{\text{limit}} \right\}$$
Solved via bisection root-finding on the interval $[1.0, 10.0]$ up to tolerance $\epsilon = 10^{-4}$.
