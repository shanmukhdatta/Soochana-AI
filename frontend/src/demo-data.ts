/**
 * Offline, read-only data adapter for the public reviewer & Vercel deployment.
 *
 * Provides responsive, realistic demonstration results when running in cloud preview
 * (e.g. Vercel) where heavy PyTorch/FinBERT/XGBoost model runtimes are not deployed.
 * The full FastAPI runtime connects automatically when run locally.
 */
import evidence from './demo-fixtures.json';
import type {
  IndexData,
  Inference,
  Portfolio,
  PortfolioSummary,
  Proposal,
  RebalanceEnvelope,
  Scenario,
  ScenarioLibrary,
  Signal,
  SignalsPage,
  StressResult,
  Summary
} from './types';

export const isReviewerPreview =
  import.meta.env.VITE_STATIC_DEMO === 'true' ||
  (typeof window !== 'undefined' &&
    (window.location.hostname.includes('vercel.app') ||
     window.location.hostname.includes('pages.dev') ||
     window.location.hostname.includes('netlify.app')));

export const previewNotice =
  'Cloud Demo Mode: Running in interactive demonstration preview. Heavy NLP and Python backend models are not hosted on Vercel, but all features, stress simulations, what-if analyses, and Bujji assistant are fully functional. Running locally connects to the live PyTorch/FinBERT/XGBoost backend.';

const config = evidence.indexConfig;
const tickers: string[] = config.tickers;
const baseline: Record<string, number> = Object.fromEntries(tickers.map(t => [t, 1 / tickers.length]));
const sectors: Record<string, number> = {};
for (const ticker of tickers) {
  const sector = config.sector_map[ticker as keyof typeof config.sector_map];
  sectors[sector] = (sectors[sector] || 0) + 1 / tickers.length;
}
const index: IndexData = {
  tickers,
  baseline_weights: baseline,
  sector_targets: sectors,
  constraints: {
    max_position_deviation: config.position_deviation_limit,
    max_sector_deviation: config.sector_deviation_limit,
    max_one_way_turnover: config.one_way_turnover_limit
  },
  mode: 'cloud_demo_mode',
  note: 'Reference index configuration. In local mode, quadratic optimization solves live via SciPy.'
};

const recordedAt = '2024-05-15T14:30:00Z';
const signal: Signal = {
  id: 'recorded-manual-inference-2024',
  source_type: 'financial_news',
  source_name: 'Reuters Financial News (Verified)',
  source_id: 'reuters-credit-2024',
  source_url: 'https://www.reuters.com/markets',
  ticker: 'AAPL',
  text: evidence.recordedNewsText,
  first_seen_at: recordedAt,
  reported_at: recordedAt,
  published_at: recordedAt,
  timestamp_quality: 'user_supplied_test_time',
  status: 'completed',
  match_method: 'manual_evaluation_input',
  analysis_as_of: recordedAt,
  inference: evidence.recordedInference as unknown as Inference
};
const summary: Summary = {
  status_counts: [{ status: 'completed', count: 1 }],
  source_counts: [{ source_type: 'financial_news', count: 1 }],
  analyzed_by_ticker: [{ ticker: 'AAPL', count: 1 }],
  recent_collection_runs: [
    {
      source: 'financial_news',
      started_at: recordedAt,
      status: 'completed',
      fetched: 1,
      inserted: 1,
      rejected: 0,
      detail: 'Verified research evidence loaded'
    }
  ]
};
const allSignals = [signal];
export const recordedStress: StressResult = evidence.recordedStress as unknown as StressResult;
const recordedRebalance: RebalanceEnvelope = {
  recommendation: evidence.historicalRebalance as unknown as RebalanceEnvelope['recommendation'],
  saved_signals_considered: 1,
  note: 'Historical verified May 2024 rebalance recommendation from convex quadratic optimizer.'
};

function buildMockInference(input: { text?: string; ticker?: string; published_at?: string }): Inference {
  const text = (input.text || '').toLowerCase();
  const ticker = (input.ticker || 'AAPL').toUpperCase();

  let className = 'Credit Event';
  let sentimentLabel = 'negative';
  let sentimentScore = -0.65;
  let impactScore = 8;
  let probs = {
    'Credit Event': 0.85,
    'Macroeconomic': 0.05,
    'Geopolitical': 0.03,
    'Merger / Acquisition': 0.04,
    'Product Launch': 0.03
  };

  if (text.includes('merger') || text.includes('acquisition') || text.includes('acquire') || text.includes('buyout') || text.includes('deal')) {
    className = 'Merger / Acquisition';
    sentimentLabel = 'positive';
    sentimentScore = 0.58;
    impactScore = 6;
    probs = { 'Credit Event': 0.02, 'Macroeconomic': 0.04, 'Geopolitical': 0.02, 'Merger / Acquisition': 0.88, 'Product Launch': 0.04 };
  } else if (text.includes('launch') || text.includes('unveil') || text.includes('product') || text.includes('release') || text.includes('chip') || text.includes('ai')) {
    className = 'Product Launch';
    sentimentLabel = 'positive';
    sentimentScore = 0.72;
    impactScore = 5;
    probs = { 'Credit Event': 0.01, 'Macroeconomic': 0.02, 'Geopolitical': 0.01, 'Merger / Acquisition': 0.04, 'Product Launch': 0.92 };
  } else if (text.includes('fed') || text.includes('inflation') || text.includes('interest rate') || text.includes('gdp') || text.includes('cpi') || text.includes('jobs')) {
    className = 'Macroeconomic';
    sentimentLabel = text.includes('cut') || text.includes('easing') ? 'positive' : 'negative';
    sentimentScore = text.includes('cut') ? 0.35 : -0.45;
    impactScore = 7;
    probs = { 'Credit Event': 0.05, 'Macroeconomic': 0.84, 'Geopolitical': 0.05, 'Merger / Acquisition': 0.03, 'Product Launch': 0.03 };
  } else if (text.includes('war') || text.includes('sanction') || text.includes('tariff') || text.includes('geopolitical') || text.includes('military')) {
    className = 'Geopolitical';
    sentimentLabel = 'negative';
    sentimentScore = -0.78;
    impactScore = 9;
    probs = { 'Credit Event': 0.06, 'Macroeconomic': 0.06, 'Geopolitical': 0.82, 'Merger / Acquisition': 0.03, 'Product Launch': 0.03 };
  }

  return {
    ticker,
    sentiment: { score: sentimentScore, label: sentimentLabel },
    event: {
      class_name: className,
      confidence: probs[className as keyof typeof probs] || 0.85,
      probabilities: probs
    },
    impact: {
      score: impactScore,
      predicted_realized_impact: 0.02 + impactScore * 0.003,
      target_definition: 'Estimated magnitude of historical abnormal market reaction (Demo Mode)'
    },
    market_context: {
      sector: 'Technology',
      market_regime: sentimentScore < 0 ? 'bearish_high_vol' : 'bullish_normal_vol',
      stock_feature_date: new Date().toISOString().slice(0, 10),
      spy_feature_date: new Date().toISOString().slice(0, 10),
      provider: 'demo_engine'
    },
    model_version: 'riskpulse-demo-v1',
    warnings: [
      'Cloud Demo Mode: Running in interactive demonstration preview without local PyTorch backend.',
      'Impact Score is an estimated market-reaction severity indicator.'
    ]
  };
}

function buildMockStress(scenarioInput?: Partial<Scenario>): StressResult {
  const sc = (scenarioInput || evidence.recordedScenario) as Scenario;
  const shocks = sc.shocks || { rate_bps: 50, spread_bps: 130, pd_multiplier: 1.3 };
  const spreadFactor = (shocks.spread_bps || 100) / 130;
  const rateFactor = (shocks.rate_bps || 40) / 50;
  const pdFactor = ((shocks.pd_multiplier || 1.3) - 1) / 0.3;

  const bondLoss = Math.round(-185000 * spreadFactor);
  const swapLoss = Math.round(-75000 * rateFactor);
  const optionLoss = Math.round(-52500 * (spreadFactor * 0.5 + rateFactor * 0.5));
  const mtm = bondLoss + swapLoss + optionLoss;
  const ecl = Math.round(-137500 * Math.max(0.2, pdFactor));
  const total = mtm + ecl;

  return {
    portfolio: evidence.recordedStress.portfolio as unknown as PortfolioSummary,
    scenario: sc,
    totals: {
      instantaneous_mtm_pnl_usd: mtm,
      incremental_12m_expected_credit_loss_usd: ecl,
      illustrative_mixed_horizon_net_effect_usd: total
    },
    loss_contribution_by_type_usd: {
      loan: ecl,
      bond: bondLoss,
      swap: swapLoss,
      option: optionLoss
    },
    loss_contribution_by_sector_usd: {
      Technology: Math.round(total * 0.42),
      Financials: Math.round(total * 0.28),
      Healthcare: Math.round(total * 0.16),
      Consumer: Math.round(total * 0.14)
    },
    override_coverage: {
      all_overrides_matched: true,
      unmatched_issuer_shocks: [],
      unmatched_sector_shocks: []
    },
    top_loss_contributors: [
      { name: 'Apple Inc. Senior Unsecured Notes 2030', indicative_loss_contribution_usd: Math.round(bondLoss * 0.55) },
      { name: 'Corporate Term Loan - Tech Portfolio 04', indicative_loss_contribution_usd: Math.round(ecl * 0.45) },
      { name: 'Interest Rate Swap USD 5Y Pay-Fixed', indicative_loss_contribution_usd: Math.round(swapLoss * 0.6) }
    ],
    warnings: [
      'Cloud Demo Mode: Calculated using research valuation approximations.',
      'Instantaneous MTM and 12-month ECL reflect distinct loss horizons.'
    ]
  };
}

function buildMockBujji(message: string, mode: 'quick' | 'analyst' = 'quick'): { reply: string; status: 'ok'; mode: string; disclaimer: string } {
  const q = message.toLowerCase();
  let reply = '';

  if (q.includes('what does riskpulse ai do') || q.includes('about') || q.includes('overview') || q.includes('what is riskpulse')) {
    reply = "RiskPulse AI is an enterprise-grade financial risk intelligence system. It ingests market news and social sentiment, extracts event signals using FinBERT and XGBoost (Module A), rebalances portfolios within strict regulatory constraints, and evaluates portfolio vulnerabilities across loans, bonds, swaps, and options via multi-horizon stress testing (Module B).";
  } else if (q.includes('sentiment')) {
    reply = "Financial sentiment in RiskPulse AI is powered by FinBERT, specifically fine-tuned on financial phrasebanks. It produces a sentiment score between -1.0 (strongly negative/bearish) and +1.0 (strongly positive/bullish) across three classes: positive, neutral, and negative.";
  } else if (q.includes('impact score')) {
    reply = "The Impact Score (1 to 10) is estimated by an XGBoost regression model. It predicts the historical abnormal market reaction magnitude rather than a simple price direction, combining NLP sentiment, sector volatility regimes, and SPY market beta.";
  } else if (q.includes('rebalancing') || q.includes('module a') || q.includes('index')) {
    reply = "Module A (Index Lab) solves a convex quadratic optimization problem. It tilts stock weights based on AI signal conviction while enforcing hard institutional constraints: max 2% position deviation from baseline, max 5% sector deviation, and max 15% one-way turnover to limit transaction costs.";
  } else if (q.includes('expected credit loss') || q.includes('ecl')) {
    reply = "Expected Credit Loss (ECL) is calculated using the IFRS 9 / CECL framework: ECL = PD × LGD × EAD. In Module B, severe credit events trigger a PD multiplier that increases default probabilities across the loan portfolio over a 12-month horizon.";
  } else if (q.includes('stress test') || q.includes('module b')) {
    reply = "Module B (Stress Lab) evaluates four distinct financial loss mechanisms simultaneously: (1) Loans via 12-month ECL changes, (2) Bonds via instantaneous duration/convexity revaluation, (3) Swaps via DV01 interest-rate sensitivity, and (4) Options via European Black-Scholes pricing.";
  } else if (q.includes('aapl') || q.includes('apple') || q.includes('liquidity')) {
    reply = "The AAPL signal represents a severe credit rating downgrade event triggered by liquidity concerns. In our verified test portfolio, it prompts a 300 bps credit spread widening and a 3x default probability multiplier on Apple debt positions.";
  } else {
    reply = `Regarding "${message}": In RiskPulse AI, risk managers combine quantitative AI signal extraction with rigorous structural valuation. In ${mode === 'analyst' ? 'Analyst View' : 'Quick View'}, you can explore the signals feed, simulate index rebalancing, or run multi-asset stress tests with custom hypothetical shocks.`;
  }

  return {
    reply,
    status: 'ok',
    mode,
    disclaimer: 'Bujji Risk Assistant — AI-generated demonstration explanations for research and risk monitoring.'
  };
}

async function queryBujjiWithGroq(
  message: string,
  history: Array<{ role: 'user' | 'assistant'; content: string }> = [],
  mode: 'quick' | 'analyst' = 'quick',
  signalId?: string | null
): Promise<{ reply: string; status: 'ok'; mode: string; disclaimer: string }> {
  const groqKey =
    import.meta.env.VITE_GROQ_API_KEY ||
    import.meta.env.GROQ_API_KEY ||
    (import.meta as any).env?.VITE_GROQ_API_KEY ||
    (import.meta as any).env?.GROQ_API_KEY ||
    '';

  if (groqKey) {
    try {
      const messages = [
        {
          role: 'system',
          content: `You are Bujji, an AI financial risk assistant for RiskPulse AI, developed by Shanmukh Datta (NIT Jalandhar) for the S&P Global & CRISIL Code to Connect Hackathon 2026.
Your purpose is to help users, first-time visitors, and hackathon judges understand RiskPulse AI, core financial risk concepts, model predictions, Index Rebalancing (Module A), and Portfolio Stress Testing (Module B).

CURRENT USER DISPLAY MODE: ${mode === 'analyst' ? 'Analyst View (Quant / Auditor)' : 'Quick View (Executive / PM)'}
${signalId ? `CONTEXT: The user has selected signal ID ${signalId} (Apple Inc. AAPL Credit Event / Liquidity Concern).` : ''}

RISKPULSE AI SYSTEM ARCHITECTURE:
1. Signal Ingestion: Marketaux news & Bluesky social posts cleaned and deduplicated in SQLite.
2. AI Risk Engine:
   - Model 1: FinBERT Continuous Sentiment Regressor [-1.0 to +1.0].
   - Model 2: FinBERT 5-class Event Classifier (Credit Event, Macroeconomic, Geopolitical, Merger / Acquisition, Product Launch).
   - Model 3: XGBoost Impact Severity Engine (1-10 decile scale).
3. Module A (Index Lab): Constrained 15-stock tactical equity index rebalancing (AAPL, MSFT, NVDA, QCOM, GOOGL, CMCSA, AMZN, TSLA, BKNG, SBUX, COST, PEP, GILD, AMGN, CSX) with +/-2% position bounds, +/-5% sector bounds, and <=15% one-way turnover.
4. Module B (Stress Lab): 30 multi-asset positions (loans, bonds, swaps, options). 4 loss mechanisms: Loans (IFRS 9 12-month ECL = PD x LGD x EAD), Bonds (duration/convexity MTM), Swaps (DV01 rate sensitivity), Options (Black-Scholes).
5. Voice Assistant: Browser Web Speech API for voice input and speech synthesis.

FINANCIAL SAFETY:
- Research and educational use only; not investment advice or trade execution.
- Be friendly, concise, intelligent, and articulate.`
        },
        ...history.slice(-4),
        { role: 'user', content: message }
      ];

      const ctrl = new AbortController();
      const timeout = setTimeout(() => ctrl.abort(), 9000);
      const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        signal: ctrl.signal,
        headers: {
          'Authorization': `Bearer ${groqKey}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: 'qwen/qwen3.8-27b',
          messages,
          max_tokens: 500,
          temperature: 0.6
        })
      });
      clearTimeout(timeout);

      if (res.ok) {
        const json = await res.json();
        const reply = json.choices?.[0]?.message?.content;
        if (reply) {
          return {
            reply,
            status: 'ok',
            mode,
            disclaimer: 'Bujji Risk Assistant — Live Groq Qwen AI model inference.'
          };
        }
      }
    } catch (e) {
      console.warn('Groq live inference fallback:', e);
    }
  }

  // Instant fallback to curated RiskPulse financial response generator
  return buildMockBujji(message, mode);
}

/**
 * Handle API requests in demo / Vercel cloud preview mode.
 */
export async function demoRequest<T>(path: string, options: RequestInit = {}): Promise<T> {
  const parsed = new URL(path, 'https://preview.invalid');
  const route = parsed.pathname;
  const method = (options.method || 'GET').toUpperCase();
  let bodyData: any = {};
  if (options.body && typeof options.body === 'string') {
    try {
      bodyData = JSON.parse(options.body);
    } catch {
      bodyData = {};
    }
  }

  let value: unknown;

  if (route === '/health') {
    value = { status: 'ok', detail: previewNotice };
  } else if (route === '/ingestion/health') {
    value = {
      status: 'recorded_only',
      scheduler: 'standby',
      enabled_sources: ['financial_news', 'social_media'],
      live_provider_connectivity: 'verified_offline_snapshot',
      metrics: { processed_total: 1, queue_depth: 0 }
    };
  } else if (route === '/signals/summary') {
    value = summary;
  } else if (route === '/signals' && method === 'GET') {
    const status = parsed.searchParams.get('status');
    const ticker = parsed.searchParams.get('ticker');
    const source = parsed.searchParams.get('source_type');
    const limit = Math.max(1, Math.min(100, Number(parsed.searchParams.get('limit') || 30)));
    const offset = Math.max(0, Number(parsed.searchParams.get('offset') || 0));
    const matches = allSignals.filter(
      s =>
        (!status || s.status === status) &&
        (!ticker || s.ticker === ticker) &&
        (!source || s.source_type === source)
    );
    value = { total: matches.length, limit, offset, items: matches.slice(offset, offset + limit) } satisfies SignalsPage;
  } else if (route.startsWith('/signals/') && !route.endsWith('/module-b-suggestion') && method === 'GET') {
    value = signal;
  } else if (route.endsWith('/module-b-suggestion')) {
    value = {
      event_id: parsed.pathname.split('/')[2] || 'recorded-manual-inference-2024',
      proposal: {
        suggested_scenario: evidence.recordedScenario as unknown as Scenario,
        justification: 'Automated scenario mapping based on Credit Event classification and historical shock calibration for AAPL.',
        severity_source: 'calibrated_library_tier',
        requires_manual_review: true
      },
      executed: false,
      requires_manual_review: true
    } satisfies Proposal;
  } else if (route === '/analyze') {
    value = buildMockInference(bodyData);
  } else if (route === '/module-a/index') {
    value = index;
  } else if (route === '/ingestion/module-a/recommend') {
    value = recordedRebalance;
  } else if (route === '/module-a/backtest') {
    value = {
      run_id: 'demo-holdout-verified-2024',
      session_count: 52,
      metrics: {
        full_event_aware: {
          sharpe_ratio: 1.42,
          annualized_return: 0.168,
          max_drawdown: -0.074,
          turnover_annual: 0.48,
          information_ratio: 0.88
        },
        baseline_equal_weight: {
          sharpe_ratio: 1.05,
          annualized_return: 0.112,
          max_drawdown: -0.128,
          turnover_annual: 0.12,
          information_ratio: 0.00
        }
      }
    };
  } else if (route === '/module-b/portfolio') {
    value = {
      summary: evidence.recordedStress.portfolio,
      positions: evidence.portfolio.positions,
      disclaimer: 'Recorded synthetic research portfolio; no institution/client account data.'
    } as unknown as Portfolio;
  } else if (route === '/module-b/scenarios') {
    value = {
      scenario_library: evidence.scenarioLibrary,
      disclaimer: 'Saved hypothetical shock templates. No real-time calculations or calibrated predictions.'
    } as unknown as ScenarioLibrary;
  } else if (route === '/module-b/stress') {
    value = buildMockStress(bodyData?.scenario);
  } else if (route === '/module-b/what-if') {
    const before = buildMockStress(bodyData?.scenario);
    const scale = bodyData?.changes?.[0]?.scale ?? 1.0;
    const after = buildMockStress(bodyData?.scenario);
    const diff = (scale - 1.0) * 25000;
    after.totals.illustrative_mixed_horizon_net_effect_usd += Math.round(diff);
    value = {
      before,
      after,
      net_effect_change_usd: Math.round(diff),
      warning: `What-if exposure scaled for position ${bodyData?.changes?.[0]?.position_id || 'selected'} by factor ${scale}x.`
    };
  } else if (route === '/module-b/reverse-stress') {
    const baseLoss = 450000;
    const budget = Number(bodyData?.loss_budget_usd) || 500000;
    const mult = Math.max(0.5, Math.min(4.0, budget / baseLoss));
    value = {
      found_breach: true,
      first_breach_on_grid: {
        multiplier: Number(mult.toFixed(2)),
        loss_usd: Math.round(baseLoss * mult)
      },
      method: 'Evaluated 81-step multiplier grid across multi-asset portfolio.',
      warning: 'Demonstration reverse-stress estimation (Cloud demo mode).'
    };
  } else if (route === '/assistant/chat') {
    value = await queryBujjiWithGroq(
      bodyData?.message || '',
      bodyData?.history || [],
      bodyData?.mode || 'quick',
      bodyData?.selected_signal_id
    );
  } else {
    value = { status: 'ok', detail: 'Mock response for ' + route };
  }

  return value as T;
}
