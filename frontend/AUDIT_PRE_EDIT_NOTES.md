# RiskPulse AI — Pre-Edit UI/UX Audit & Refinement Plan

**Date**: 2026-10-10  
**Target Audience**: Hackathon Reviewers (5-Minute Live Pitch), Financial Analysts & Institutional Risk Officers  
**Design Identity**: Obsidian Iris (`--bg: #10121c`, `--iris: #9b8cff`, `--amber: #f0b56b`)

---

## 1. Summary of Current UI/UX Shortcomings Across the 4 Pages

### 1.1 Command Center (Main Dashboard)
- **Hero & Identity Clutter**: The hero uses verbose marketing copy ("Financial narratives, measured", "See the story. Measure the risk. One signal. Two decisions.") instead of the direct institutional definition: *"Financial news and social signals transformed into explainable portfolio risk intelligence."*
- **KPI Metrics**: Includes "Ingested Signals: all locally stored records" alongside "Analyzed Signals", which confuses reviewers on whether raw unparsed records are actionable. Missing a dedicated "Source Freshness & Ingestion Health" KPI card.
- **Overcrowded Overview Grid**: `SOURCE WATCH` sits permanently beside the `RISK JOURNEY EXPLORER`, displaying raw DB collection records and diagnostic caveats in prime screen real estate.
- **Incomplete Event Feed Information**: The latest analyzed signals list truncates event information; it omits the AI-classified Sentiment pill and Event Category, showing only ticker, raw text, and impact score.
- **Reviewer Speed**: A judge cannot understand the system in 10-15 seconds because low-level ingestion diagnostics compete with the core signal-to-portfolio value proposition.

### 1.2 Event Intelligence (Model Explainability)
- **Information Overload in Diagnostics**: The third panel exposes all 5 raw event class probabilities simultaneously as dense horizontal bars directly below the primary prediction, crowding out the impact severity and scenario actions.
- **Model Limitation Visibility**: Disclaimers are currently small footnotes ("fineprint") rather than explicit analyst guardrails explaining that sentiment != price direction, confidence != accuracy, and impact 1-10 != probability of default or return.
- **Hidden Workflow Action**: The "Propose stress scenario" action is tucked at the very bottom of the diagnostics panel. Reviewers miss the critical bridge from AI signal to Module B portfolio stress testing.
- **Manual Inference Stature**: The manual inference text box occupies a full-width bottom section on equal footing with genuine ingested news/social streams, creating confusion regarding genuine vs. user-typed evidence.

### 1.3 Module A (Tactical Index Rebalancing)
- **Visual Allocation Hierarchy**: Before-and-after allocations are rendered solely as small inline dual-bars within a 15-row table without a high-level visual summary or sector breakdown.
- **Backtest Confusion**: The historical holdout backtest (357 historical trading sessions across 5 strategies) is rendered directly below the live recommendation table without a clear partition, causing reviewers to conflate historical backtests with real-time portfolio signals.
- **Missing Signal Attribution Visibility**: The "Why the weights changed" section only appears if changes exceed a threshold, but needs clearer visual association with the specific FinBERT/XGBoost signals that prompted the tilt.

### 1.4 Module B (Portfolio Stress Laboratory)
- **Three-Panel Squeeze**: `SCENARIO DESIGN`, `FOUR LOSS MECHANISMS`, and `STRESS RESULTS` are displayed in three tall side-by-side columns, causing inputs and outputs to feel cramped and requiring vertical scrolling to review assumptions and outcomes.
- **Measurement Horizon Disclosures**: The critical distinction between instantaneous mark-to-market revaluation (Bonds, Swaps, Options) and 12-month projected credit loss (Loans ECL) needs a more prominent visual callout to ensure analysts and judges never conflate MTM with 12-month ECL or view derivative notional as NAV.
- **Secondary Features**: What-If exposure scaling and Reverse Stress Testing should be seamlessly accessible via secondary tabs without cluttering the primary scenario execution workflow.

---

## 2. Target Design & Refinement Plan

1. **Global Cleanliness & Institutional Typography**:
   - Maintain the Obsidian Iris color palette (`#10121c` base, `#191d2b` panels, `#9b8cff` iris, `#f0b56b` amber, `#74b9f4` blue).
   - Ensure large, projector-friendly typography (14–16px body, 20–28px panel headers, 36–42px page titles).
   - Crisp card borders (`#363a54`), rounded corners (`12px`), and subtle hover states without distracting glow effects.

2. **Command Center Refinements**:
   - Institutional header: *"Financial news and social signals transformed into explainable portfolio risk intelligence."* with Live Freshness badge.
   - 4 Essential KPI Cards: Analyzed Events, Aggregate Sentiment, Average Impact Severity, Source Freshness & Health.
   - Streamlined, intuitive Risk Journey Explorer: `01 Source Evidence → 02 AI Risk Engine → 03 Portfolio Response`.
   - Rich Recent Event Feed: Ticker, source badge, headline, sentiment pill, event class tag, impact score, and direct one-click navigation to Event Intelligence.
   - Collapsible Ingestion & Source Status drawer for technical reviewers.

3. **Event Intelligence Refinements**:
   - Clear 3-column / 2-column responsive layout: Ingested Feed | Evidence & Provenance | Three-Model Diagnostics.
   - Primary model output highlight: Top event class + confidence prominently displayed; full 5-class distribution expandable under "Model Details".
   - Explicit Analyst Model Limitations banner (Sentiment != Direction; Impact 1-10 != Probability of Default; Product Launch distribution note).
   - Prominent "Propose Stress Scenario → Review in Stress Lab" workflow card.
   - Collapsible "Analyze Custom Text" section clearly designated as manual testing.

4. **Module A Refinements**:
   - Prominent "Generate Rebalance Recommendation" primary CTA.
   - Visual Before-vs-After Allocation comparison with sector tags and clear deltas.
   - Concise Constraint Guardrails (±1.5% max stock tilt, ±3.0% max sector tilt, 5% max turnover, 10 bps transaction cost).
   - Clean "Why the weights changed" explainability breakdown.
   - Dedicated collapsible "Historical Backtest Analysis" section with clear disclaimer that historical holdout is not a forecast.

5. **Module B Refinements**:
   - Clear synthetic portfolio summary metrics (Loan EAD, Bond MV, Derivative Reference Notional != NAV).
   - Prominent scenario selector (Credit Event, Macro, Geopolitical, M&A, Product Launch) with intensity picker and shock preview.
   - Explicit Analyst Review & Acknowledgment checkbox before execution.
   - Prominent "Run Reviewed Stress Test" CTA.
   - Distinct Results Breakdown: Instantaneous MTM P&L vs. Incremental 12m Loan ECL, with explicit warning on illustrative mixed-horizon metric.
   - Clean secondary tabs for What-If Analysis and Reverse Stress Testing.

---

## 3. Strict Contract & Safety Checklist
- [x] Retain exact string `'Risk Journey Explorer'`
- [x] Retain exact string `'EVENT STREAM'`
- [x] Retain exact string `'ALLOCATION SHIFT ANALYZER'`
- [x] Retain exact string `'SCENARIO DESIGN'`
- [x] Retain exact string `'FINBERT'`
- [x] Retain exact string `'XGBOOST'`
- [x] Retain theme tokens `--bg:#10121c`, `--iris:#9b8cff`, `--amber:#f0b56b`
- [x] Never invent fake data or mock responses
- [x] Preserve all 14 API endpoints in `src/api.ts`
- [x] Verify tests: `npm test`, `npm run check:contracts`, `npm run build`, `pytest -q`
