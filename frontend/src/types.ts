export type Page = 'overview' | 'events' | 'moduleA' | 'moduleB';
export interface Sentiment { score: number; label: string }
export interface ModelEvent { class_name: string; confidence: number; probabilities: Record<string, number> }
export interface Impact { score: number; predicted_realized_impact: number; target_definition?: string }
export interface Inference { ticker: string; sentiment: Sentiment; event: ModelEvent; impact: Impact; market_context?: Record<string, string>; model_version?: string; warnings?: string[] }
export interface Signal {
 id: string; source_type: 'financial_news' | 'social_media'; source_name: string; source_url: string; ticker: string;
 text: string; source_id: string; first_seen_at: string; reported_at: string | null; published_at?: string | null;
 timestamp_quality: string; status: string; match_method: string; analysis_as_of: string; inference: Inference | null;
 attempts?: number; last_error?: string | null;
}
export interface SignalsPage { total: number; limit: number; offset: number; items: Signal[] }
export interface CollectionRun { source: string; started_at: string; status: string; fetched: number; inserted: number; rejected: number; detail?: string }
export interface Summary {
 status_counts: Array<{status: string; count: number}>;
 source_counts: Array<{source_type: string; count: number}>;
 analyzed_by_ticker: Array<{ticker: string; count: number}>;
 recent_collection_runs: CollectionRun[];
}
export interface IndexData { tickers: string[]; baseline_weights: Record<string, number>; constraints: Record<string, number>; sector_targets?: Record<string, number>; mode: string; note?: string }
export interface WeightChange { ticker: string; before: number; after: number; change_percentage_points: number; signal: number; explanation: string }
export interface Rebalance { current_weights: Record<string,number>; target_weights: Record<string,number>; changes: WeightChange[]; one_way_turnover: number; estimated_transaction_cost_fraction: number; signal_counts: Record<string,number>; warnings?: string[] }
export interface RebalanceEnvelope { recommendation: Rebalance; saved_signals_considered: number; note: string }
export interface PortfolioSummary { name: string; currency: string; position_count: number; counts: Record<string,number>; loan_ead_usd: number; bond_market_value_usd: number; derivative_reference_notional_usd: number; exposure_warning: string }
export interface Position { id: string; name: string; type: 'loan'|'bond'|'swap'|'option'; ticker?: string | null; sector: string }
export interface Portfolio { summary: PortfolioSummary; positions: Position[]; disclaimer: string }
export interface Shocks { rate_bps: number; spread_bps: number; equity_pct: number; volatility_points: number; pd_multiplier: number }
export interface Scenario { name: string; event_class: string; severity: string; shocks: Shocks; issuer_shocks: Record<string,Shocks>; sector_shocks: Record<string,Shocks>; assumption_source: string }
export interface ScenarioLibrary { scenario_library: Record<string,Record<string,Scenario>>; disclaimer: string }
export interface StressResult { scenario: Scenario; portfolio: PortfolioSummary; totals: Record<string,number>; loss_contribution_by_type_usd: Record<string,number>; loss_contribution_by_sector_usd: Record<string,number>; override_coverage?: {all_overrides_matched: boolean;unmatched_issuer_shocks: string[];unmatched_sector_shocks: string[]}; top_loss_contributors?: Array<{name:string;indicative_loss_contribution_usd:number}>; warnings: string[] }
export interface Proposal { event_id: string; proposal: {suggested_scenario: Scenario; justification:string;severity_source:string; requires_manual_review: boolean}; executed: false; requires_manual_review: true }
