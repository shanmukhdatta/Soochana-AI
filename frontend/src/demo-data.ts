/**
 * Offline, read-only data adapter for the public reviewer build.
 *
 * All numeric model/portfolio outputs below are copied from the repository's
 * recorded JSON files via demo-fixtures.json, NOT computed at request time.
 * There is no simulated network response, ML inference, price feed, or pricing.
 * The normal Vite/FastAPI build does not call this adapter.
 */
import evidence from './demo-fixtures.json';
import type { IndexData, Inference, Portfolio, RebalanceEnvelope, ScenarioLibrary, Signal, SignalsPage, StressResult, Summary } from './types';

export const isReviewerPreview = import.meta.env.VITE_STATIC_DEMO === 'true';
export const previewNotice = 'Reviewer preview: saved, historical research evidence only. FinBERT/XGBoost inference and the Python calculation backend are not configured. No live news, market values, new prediction, or new scenario result is generated.';

const config = evidence.indexConfig;
const tickers: string[] = config.tickers;
const baseline: Record<string, number> = Object.fromEntries(tickers.map(t => [t, 1 / tickers.length]));
const sectors: Record<string, number> = {};
for (const ticker of tickers) {
  const sector = config.sector_map[ticker as keyof typeof config.sector_map];
  sectors[sector] = (sectors[sector] || 0) + 1 / tickers.length;
}
const index: IndexData = {
  tickers, baseline_weights: baseline, sector_targets: sectors,
  constraints: {
    max_position_deviation: config.position_deviation_limit,
    max_sector_deviation: config.sector_deviation_limit,
    max_one_way_turnover: config.one_way_turnover_limit
  },
  mode: 'recorded_preview_only',
  note: 'Reference index configuration only. Backend rebalancing disabled in Vercel preview.'
};

const recordedAt = '2024-05-15T14:30:00Z'; // input timestamp from scripts/test_live_module_b_e2e.py
const signal: Signal = {
  id: 'recorded-manual-inference-2024',
  source_type: 'financial_news',
  source_name: 'Recorded evaluation input (NOT source-verified)',
  source_id: 'research-evaluation-only',
  source_url: '',
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
  status_counts: [{status: 'completed', count: 1}],
  source_counts: [{source_type: 'financial_news', count: 1}],
  analyzed_by_ticker: [{ticker: 'AAPL', count: 1}],
  recent_collection_runs: []
};
const allSignals = [signal];
export const recordedStress: StressResult = evidence.recordedStress as unknown as StressResult;
const recordedRebalance: RebalanceEnvelope = {
  recommendation: evidence.historicalRebalance as unknown as RebalanceEnvelope['recommendation'],
  saved_signals_considered: 1,
  note: 'EXISTING recorded May 2024 example; not newly calculated, not based on current signals.'
};

function notAvailable(operation: string): never {
  throw new Error(operation + ' is unavailable in this reviewer preview: the trained NLP models / Python backend are not configured. Only saved, clearly identified research evidence is displayed.');
}

/** Map the existing full App's read-only API calls to archived data.
 * Unavailable calculations explicitly reject; never return made-up outputs.
 */
export async function demoRequest<T>(path: string, options: RequestInit = {}): Promise<T> {
  const parsed = new URL(path, 'https://preview.invalid');
  const route = parsed.pathname;
  const method = (options.method || 'GET').toUpperCase();
  let value: unknown;
  if (route === '/health') value = {status: 'models_not_configured', detail: previewNotice};
  else if (route === '/ingestion/health') value = {
    status: 'recorded_only', scheduler: 'unavailable', enabled_sources: [],
    live_provider_connectivity: 'not_tested_by_this_endpoint', metrics: []
  };
  else if (route === '/signals/summary') value = summary;
  else if (route === '/signals' && method === 'GET') {
    const status = parsed.searchParams.get('status');
    const ticker = parsed.searchParams.get('ticker');
    const source = parsed.searchParams.get('source_type');
    const limit = Math.max(1, Math.min(100, Number(parsed.searchParams.get('limit') || 30)));
    const offset = Math.max(0, Number(parsed.searchParams.get('offset') || 0));
    const matches = allSignals.filter(s => (!status || s.status === status) &&
      (!ticker || s.ticker === ticker) && (!source || s.source_type === source));
    value = {total: matches.length, limit, offset, items: matches.slice(offset, offset + limit)} satisfies SignalsPage;
  }
  else if (route === '/module-a/index') value = index;
  else if (route === '/module-b/portfolio') value = {
    summary: evidence.recordedStress.portfolio,
    positions: evidence.portfolio.positions,
    disclaimer: 'Recorded synthetic research portfolio; no institution/client account data.'
  } as unknown as Portfolio;
  else if (route === '/module-b/scenarios') value = {
    scenario_library: evidence.scenarioLibrary,
    disclaimer: 'Saved hypothetical shock templates. No real-time calculations or calibrated predictions.'
  } as unknown as ScenarioLibrary;
  else if (route === '/ingestion/module-a/recommend') value = recordedRebalance;
  else if (route === '/assistant/chat') value = {
    reply: 'Bujji is not configured in this reviewer preview. This is the original frontend with archived research evidence; chat requires the connected Python backend and its authorized assistant service.',
    status: 'unavailable',
    mode: 'quick',
    disclaimer: 'Research only.'
  };
  else if (route === '/analyze') notAvailable('New model inference');
  else if (route.endsWith('/module-b-suggestion')) notAvailable('New model-informed scenario suggestion');
  else if (route === '/module-a/backtest') notAvailable('Re-running the historical backtest');
  else if (route === '/module-b/stress') notAvailable('New stress valuation');
  else if (route === '/module-b/what-if') notAvailable('What-if calculation');
  else if (route === '/module-b/reverse-stress') notAvailable('Reverse-stress calculation');
  else notAvailable('API endpoint ' + route);
  return value as T;
}
