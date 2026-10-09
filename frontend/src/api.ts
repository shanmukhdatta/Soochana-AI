import type { IndexData, Inference, Portfolio, Proposal, RebalanceEnvelope, ScenarioLibrary, SignalsPage, Signal, StressResult, Summary } from './types';
const BASE = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '');
export class ApiError extends Error { constructor(public status: number, public path: string, message: string) { super(message); this.name='ApiError'; } }
async function request<T>(path: string, opts: RequestInit = {}): Promise<T> {
  const ctrl = new AbortController(); const timeout = setTimeout(() => ctrl.abort(), 18000);
  try {
    const res = await fetch(`${BASE}${path}`, { ...opts, signal:ctrl.signal, headers: { 'Accept':'application/json', ...(opts.body ? {'Content-Type':'application/json'} : {}), ...opts.headers } });
    if(!res.ok) { let message = `HTTP ${res.status}`; try { const err = await res.json(); message=typeof err.detail==='string'?err.detail:JSON.stringify(err.detail??err); }catch{/* preserve status */} throw new ApiError(res.status,path,message); }
    return await res.json() as T;
  } catch(e) {
    if(e instanceof Error && e.name==='AbortError') throw new Error(`Request timed out: ${path}`);
    throw e;
  } finally { clearTimeout(timeout); }
}
const post = <T>(path:string, data:unknown) => request<T>(path,{method:'POST',body:JSON.stringify(data)});
export const api = {
  health: () => request<{status:string;detail?:string}>('/health'),
  ingestionHealth: () => request<{status:string;scheduler:string;enabled_sources?:string[];live_provider_connectivity:string;metrics?:unknown}>('/ingestion/health'),
  summary: () => request<Summary>('/signals/summary'),
  signals: (opts:{limit?:number;offset?:number;ticker?:string;source_type?:string;status?:string}={}) => {
    const params = new URLSearchParams(); for(const [k,v] of Object.entries(opts))if(v!==undefined && v!=='')params.set(k,String(v));
    return request<SignalsPage>(`/signals?${params}`);
  },
  signal: (id:string) => request<Signal>(`/signals/${encodeURIComponent(id)}`),
  propose: (id:string) => request<Proposal>(`/signals/${encodeURIComponent(id)}/module-b-suggestion`),
  analyze: (input:{text:string;ticker:string;published_at:string;source:'manual'}) => post<Inference>('/analyze',input),
  index: () => request<IndexData>('/module-a/index'),
  rebalanceSaved: (limit=100) => post<RebalanceEnvelope>(`/ingestion/module-a/recommend?limit=${limit}`,{}),
  backtest: (dataset_id='holdout') => post<{run_id:string;metrics:Record<string,Record<string,unknown>>;session_count:number}>('/module-a/backtest',{dataset_id,strategy:'full_event_aware'}),
  portfolio: () => request<Portfolio>('/module-b/portfolio'),
  scenarios: () => request<ScenarioLibrary>('/module-b/scenarios'),
  stress: (scenario:unknown) => post<StressResult>('/module-b/stress',{scenario}),
  whatIf: (scenario:unknown, changes:Array<{position_id:string;scale:number}>) => post<{before:StressResult;after:StressResult;net_effect_change_usd:number;warning:string}>('/module-b/what-if',{scenario,changes}),
  reverseStress: (scenario:unknown, loss_budget_usd:number) => post<{found_breach:boolean;first_breach_on_grid?:{multiplier:number;loss_usd:number}|null;method:string;warning:string;truncated_at_multiplier?:number|null}>('/module-b/reverse-stress',{scenario,loss_budget_usd,max_multiplier:4,steps:81}),
};
