import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import {
  Activity, ArrowRight, ArrowUpRight, BrainCircuit, ChartNoAxesCombined,
  ChevronDown, CircleAlert, Clock3, ExternalLink, Gauge,
  LoaderCircle, LockKeyhole, Newspaper, Play, RadioTower, RefreshCw, Search, ShieldAlert, ShieldCheck,
  SlidersHorizontal, Sparkles, WandSparkles, X
} from 'lucide-react';
import { api } from './api';
import { isReviewerPreview, previewNotice, recordedStress } from './demo-data';
import { countStatus, eventCategoryCounts, fmtMoney, formatPercentFraction, recentSignalMeans, safeSourceUrl, scenarioForExecution, sentimentLabel } from './lib/helpers.mjs';
import type { IndexData, Inference, Page, Portfolio, Proposal, RebalanceEnvelope, Scenario, ScenarioLibrary, Signal, SignalsPage, StressResult, Summary } from './types';

type Load<T> = { value: T | null; error: string | null; pending: boolean };
const initial = <T,>(): Load<T> => ({ value:null,error:null,pending:true });
function errorMessage(err: unknown) { return err instanceof Error ? err.message : String(err); }
function asDate(v?:string|null) { if(!v) return 'Unknown'; const d=new Date(v);return Number.isNaN(d.getTime()) ? 'Unknown' : d.toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}); }
function scoreTone(score:number|null|undefined) { return typeof score==='number' ? score > .1?'pos':score<-.1?'neg':'neutral' : 'neutral'; }
function decimal(value:number|null|undefined, decimals=2) {return typeof value==='number'&&Number.isFinite(value)? value.toFixed(decimals):'—';}
const pill = (text:string,variant='neutral') => <span className={`pill ${variant}`}>{text}</span>;
const formatImpact=(n?:number|null)=>typeof n==='number'?`${n}/10`:'—';
const sourceName=(s?:string)=>s==='financial_news'?'NEWS':s==='social_media'?'SOCIAL':'MANUAL';
function Metric({label,value,sub,kind='iris'}:{label:string;value:string;sub:string;kind?:string}){return <div className="metric card"><span className="metric-label">{label}</span><strong className="metric-value">{value}</strong><span className="metric-sub">{sub}</span><i className={`metric-notch ${kind}`}/></div>}
function Empty({title,body}:{title:string;body:string}){return <div className="empty"><CircleAlert size={25}/><b>{title}</b><span>{body}</span></div>}
function PanelHeader({n,title,sub,accent='iris'}:{n:string;title:string;sub?:string;accent?:string}){return <div className="panelhead"><span className={`overline color-${accent}`}>{n} / {title}</span>{sub&&<span className="panel-sub">{sub}</span>}</div>}
function Field({label,children}:{label:string;children:React.ReactNode}){return <div className="field"><span className="field-label">{label}</span><div className="field-value">{children}</div></div>}
function SourceLink({url,label='Open original source'}:{url:string;label?:string}){const safe=safeSourceUrl(url);return safe?<a className="source-link" href={safe} target="_blank" rel="noopener noreferrer">{label} <ExternalLink size={12}/></a>:<span className="muted">Source URL unavailable</span>}
function Dot({tone='iris'}:{tone?:string}){return <span className={`dot ${tone}`}/>}
function ErrorBanner({message}:{message:string|null}){return message?<div className="error-banner" role="alert"><CircleAlert size={17}/>{message}</div>:null}
type ViewMode = 'quick' | 'analyst';
import { HomeLanding } from './components/HomeLanding';

// Architecture narrative: From market information to risk intelligence. The problem. What the challenge asks: Core technical requirements vs Additional implementation. Explore live intelligence.
function Home({onStart,onTab,onMode}:{onStart:()=>void;onTab:(p:Page)=>void;onMode?:(m:ViewMode)=>void}){
  return <HomeLanding onStart={onStart} onTab={onTab} onMode={onMode}/>;
}

function Shell({page,onPage,mode,onMode,children,summary,loading,refresh,selectedTicker}:{page:Page;onPage:(p:Page)=>void;mode:ViewMode;onMode:(m:ViewMode)=>void;children:React.ReactNode;summary:Summary|null;health:string;ingestion:string;loading:boolean;refresh:()=>void;selectedTicker:string}){
 const last=[...(summary?.recent_collection_runs??[])].sort((a,b)=>b.started_at.localeCompare(a.started_at))[0];
 const tabs:Array<[Page,string]>=[['events','AI Signals'],['moduleA','Index Lab'],['moduleB','Stress Lab']];
 return <div className={`mono-app mode-${mode}`} data-mode={mode}>
  <header className="mono-header"><button className="mono-brand mono-brand-button" onClick={()=>onPage('home')} aria-label="Go to introduction"><Activity size={25}/><span>RiskPulse <b>AI</b></span></button><nav className="mono-topnav" aria-label="Main navigation"><button onClick={()=>onPage('home')}>Home</button><button className="active" onClick={()=>onPage('events')}>Dashboard</button></nav><div className="mono-header-right"><span className="mono-source-status" title="This is last recorded collection time, not proof that polling is currently active">{last?`Last collection ${asDate(last.started_at)}`:'Collection time unknown'}</span><button className="mono-refresh" onClick={refresh} disabled={loading} aria-label="Refresh signals"><RefreshCw size={17} className={loading?'spin':''}/></button><div className="mode-toggle" role="group" aria-label="Display mode"><button aria-pressed={mode==='quick'} className={mode==='quick'?'selected':''} onClick={()=>onMode('quick')}>Quick View</button><button aria-pressed={mode==='analyst'} className={mode==='analyst'?'selected':''} onClick={()=>onMode('analyst')}>Analyst View</button></div></div></header>
  <main className="mono-main"><div className="mono-tabs" role="navigation" aria-label="Dashboard sections">{tabs.map(([id,label])=><button key={id} className={page===id?'active':''} onClick={()=>onPage(id)}>{label}</button>)}</div>{page==='events'&&selectedTicker?<div className="mono-context">Selected event: <strong>{selectedTicker}</strong><span> · Click any event in the 01 / EVENT STREAM below to inspect other stocks (archived evidence is explicitly labeled in this reviewer preview) · same evidence across both views</span></div>:page==='moduleA'?<div className="mono-context">Tactical Index Lab<span> · Constrains allocations across all 15 index constituents; the public preview displays an existing recorded example</span></div>:page==='moduleB'?<div className="mono-context">Multi-Asset Stress Lab<span> · Simulates shocks across 30 portfolio positions (Loans, Bonds, Derivatives) under 5 scenario families</span></div>:null}
  <div className="mono-mode-description">{mode==='quick'?'Quick View · essential inputs, actions and results':'Analyst View · complete model evidence, constraints and advanced analyses'}</div>
  {children}</main>
  <footer className="mono-footer"><span>RiskPulse AI · Research prototype · Simulated portfolios · No live trading</span><span>Models may misclassify events. Scenario outputs are hypothetical.</span></footer>
 </div>
}
function SignalSummary({summary,signals}:{summary:Summary|null;signals:Signal[]}){
 const means=recentSignalMeans(signals);return <div className="mono-stats"><div><span>Analyzed events</span><strong>{summary?countStatus(summary,'completed'):'—'}</strong></div><div><span>Average sentiment</span><strong>{decimal(means.sentiment,3)}</strong></div><div><span>Average impact</span><strong>{means.impact===null?'—':decimal(means.impact,1)+' / 10'}</strong></div></div>
}
function Overview({summary,signals,goSignal,goPage,health,ingestion}:{summary:Summary|null;signals:Signal[];goSignal:(s:Signal)=>void;goPage:(p:Page)=>void;health:string;ingestion:string}){
 const completed=signals.filter(x=>x.status==='completed'&&x.inference);const means=recentSignalMeans(signals);
 const sourceRows=summary?.recent_collection_runs??[];
 const news=sourceRows.filter(x=>['marketaux','gdelt'].includes(x.source.toLowerCase())).sort((a,b)=>b.started_at.localeCompare(a.started_at))[0];
 const social=sourceRows.filter(x=>x.source.toLowerCase()==='bluesky').sort((a,b)=>b.started_at.localeCompare(a.started_at))[0];
 const recent=completed.slice(0,5);
 const cats=eventCategoryCounts(completed);const max=Math.max(1,...Object.values(cats));
 return <div className="page">
   <div className="hero"><div><div className="eyebrow"><Sparkles size={14}/> Financial narratives, measured</div><h1>See the story. Measure the risk.<br/><em>One signal. Two decisions.</em></h1><p>Source-traceable market intelligence transformed into explainable research signals and two simulated portfolio workflows.</p></div><button className="primary-button hero-cta" onClick={()=>goPage('events')}>Explore events <ArrowUpRight size={17}/></button></div>
   <div className="metrics-grid"><Metric label={isReviewerPreview?'RECORDED INPUTS':'INGESTED SIGNALS'} value={summary?String((summary.status_counts??[]).reduce((a,x)=>a+Number(x.count||0),0)):'—'} sub="All locally stored records"/><Metric label={isReviewerPreview?'ARCHIVED MODEL OUTPUTS':'ANALYZED SIGNALS'} value={summary?String(countStatus(summary,'completed')):'—'} sub={isReviewerPreview?'Previously recorded model output':'Completed real inference'} kind="blue"/><Metric label="MEAN SENTIMENT" value={decimal(means.sentiment,3)} sub={`Recent ${means.count} loaded completed records`} kind="iris"/><Metric label="MEAN IMPACT" value={means.impact===null?'—':`${decimal(means.impact,1)} / 10`} sub="Recent loaded sample · magnitude" kind="amber"/></div>
   <div className="overview-center"><section className="panel journey"><PanelHeader n="01" title="RISK JOURNEY EXPLORER" sub="One evidence trail, three auditable stages"/><div className="journey-row"><div className="journey-box"><div className="journey-kicker blue">01 / SOURCE EVIDENCE</div><h3>Capture & verify</h3><p><Newspaper size={16}/> Financial news</p><p><RadioTower size={16}/> Social posts</p><small>Original link · ticker · observed time · de-duplication</small></div><ArrowRight className="journey-arrow"/><div className="journey-box bright"><div className="journey-kicker iris">02 / AI SIGNAL ENGINE</div><h3>Score & classify</h3><p><Activity size={16}/> Sentiment [−1,+1]</p><p><BrainCircuit size={16}/> Five event classes</p><p><Gauge size={16}/> Impact magnitude 1–10</p></div><ArrowRight className="journey-arrow"/><div className="journey-box"><div className="journey-kicker amber">03 / PORTFOLIO RESPONSE</div><h3>Simulate & review</h3><p><ChartNoAxesCombined size={16}/> Module A · Rebalance</p><p><ShieldAlert size={16}/> Module B · Stress</p><small>No trades. Stress requires review.</small></div></div><div className="journey-bottom"><span><Dot tone="iris"/> Select an actual signal in Event Intelligence to follow its journey.</span><button className="small-button" onClick={()=>goPage('events')}>Inspect signal <ArrowRight size={14}/></button></div></section>
   <section className="panel source-watch"><PanelHeader n="02" title="SOURCE WATCH" sub="Actual recorded collection history"/>{[[news,'NEWS','blue'],[social,'SOCIAL','iris']].map(([r,kind,tone],i)=>{const item=r as Summary['recent_collection_runs'][number]|undefined;return <div key={i} className="source-item"><Dot tone={String(tone)}/><div><strong>{kind==='NEWS'?'Financial news':'Public social posts'}</strong><small>{item?`${item.source} · ${item.status} · ${item.inserted} inserted`: 'No recorded run available'}</small><small>{item?asDate(item.started_at):'Status unknown'}</small></div></div>})}<div className="source-item"><Dot tone={health==='ready'?'green':'amber'}/><div><strong>Three-model risk engine</strong><small>{health==='ready'?'Ready for inference':'Unable to confirm current readiness'}</small><small>Ingestion: {ingestion}; live-provider status must come from collection runs</small></div></div><div className="source-note">No HTTP-success claim is inferred from the ingestion health endpoint alone.</div></section></div>
   <div className="overview-bottom"><section className="panel feed-panel"><PanelHeader n="03" title="LATEST ANALYZED SIGNALS" sub="Click any event to inspect its original source and model outputs"/>{recent.length?recent.map(s=><button className="signal-row" key={s.id} onClick={()=>goSignal(s)}><span className={`source-type ${s.source_type==='financial_news'?'news':'social'}`}>{sourceName(s.source_type)}</span><span className="signal-content"><b>{s.ticker} · {s.text}</b><small>{s.source_name} · {asDate(s.first_seen_at)}</small></span><span className="signal-impact">{formatImpact(s.inference?.impact.score)}<ChevronDown size={14}/></span></button>):<Empty title="No completed events loaded" body="Start the backend and ingestion scheduler; no demo values are substituted."/>}</section>
   <section className="panel distribution"><PanelHeader n="04" title="EVENT DISTRIBUTION" sub="Category counts in loaded completed records"/>{Object.keys(cats).length?Object.entries(cats).sort((a,b)=>b[1]-a[1]).map(([key,value])=><div key={key} className="bar-item"><span>{key}</span><div className="bar-track"><i style={{width:`${value/max*100}%`}}/></div><strong>{value}</strong></div>):<Empty title="No category data yet" body="Event probabilities appear after analyzed signals are returned."/>}</section></div>
   <div className="decision-grid"><button className="decision-card" onClick={()=>goPage('moduleA')}><span className="overline color-iris">TACTICAL INTELLIGENCE / A</span><h3>Index rebalancing</h3><p>From a real event to constrained weight changes across 15 stocks.</p><span>Inspect strategy <ArrowUpRight size={16}/></span></button><button className="decision-card" onClick={()=>goPage('moduleB')}><span className="overline color-amber">STRATEGIC INTELLIGENCE / B</span><h3>Portfolio stress lab</h3><p>Explore hypothetical credit, bond, swap and option sensitivity.</p><span>Explore scenario <ArrowUpRight size={16}/></span></button></div>
   <p className="page-disclaimer">Research demonstration. Models can misclassify financial discussion; synthetic exposures and historical backtests are not live holdings or forecasts.</p>
 </div>;
}
function Events({signals,selected,onSelect,onModuleB,tickers,analyzeManual}:{signals:Signal[];selected:Signal|null;onSelect:(s:Signal)=>void;onModuleB:(p:Proposal)=>void;tickers:string[];analyzeManual:(x:{text:string;ticker:string;published_at:string;source:'manual'})=>Promise<Inference>}){
 const [filter,setFilter]=useState(''); const [source,setSource]=useState('all');const [ticker,setTicker]=useState('');const [status,setStatus]=useState('completed');
 const [manualText,setManualText]=useState('');const [manualTicker,setManualTicker]=useState('AAPL');const [manual,setManual]=useState<Inference|null>(null);const [manualError,setManualError]=useState('');const [manualBusy,setManualBusy]=useState(false);
 const [proposal,setProposal]=useState<Proposal|null>(null);const [proposalError,setProposalError]=useState('');const [proposalBusy,setProposalBusy]=useState(false);
 const shown=useMemo(()=>signals.filter(s=>(source==='all'||s.source_type===source)&&(status==='all'||s.status===status)&&(!ticker||s.ticker===ticker)&&(!filter||`${s.text} ${s.ticker} ${s.source_name}`.toLowerCase().includes(filter.toLowerCase()))),[signals,source,status,ticker,filter]);
 const choose=(s:Signal)=>{onSelect(s);setProposal(null);setProposalError('');setManual(null)};
 const askProposal=async()=>{if(!selected)return;setProposalBusy(true);setProposalError('');try{setProposal(await api.propose(selected.id));}catch(e){setProposalError(errorMessage(e));}finally{setProposalBusy(false)}};
 const submit=async(e:FormEvent)=>{e.preventDefault();if(manualText.trim().length<5)return setManualError('At least five characters of financial text required.');setManualBusy(true);setManualError('');setManual(null);try{setManual(await analyzeManual({text:manualText.trim(),ticker:manualTicker,published_at:new Date().toISOString(),source:'manual'}));}catch(e){setManualError(errorMessage(e));}finally{setManualBusy(false)}};
 const inference=manual??selected?.inference;const record=manual?null:selected;const probabilities=inference?.event?.probabilities??{};
 return <div className="page"><div className="page-title"><div className="eyebrow">EVIDENCE-FIRST MODEL INSPECTION</div><h1>Follow the evidence.<br/><em>Inspect the inference.</em></h1><p>Every conclusion links back to the original source, stored processing time and model evidence.</p></div>
 <div className="filter-row"><label className="search-field"><Search size={17}/><input value={filter} onChange={e=>setFilter(e.target.value)} placeholder="Search event text or ticker"/></label><label className="compact-select">SOURCE<select value={source} onChange={e=>setSource(e.target.value)}><option value="all">News + social</option><option value="financial_news">News</option><option value="social_media">Social</option></select></label><label className="compact-select">STATUS<select value={status} onChange={e=>setStatus(e.target.value)}><option value="completed">Analyzed</option><option value="all">All</option><option value="pending">Pending</option><option value="retry">Retry</option><option value="dead">Failed</option></select></label><label className="compact-select">TICKER<select value={ticker} onChange={e=>setTicker(e.target.value)}><option value="">All tickers</option>{tickers.map(x=><option key={x}>{x}</option>)}</select></label></div>
 <div className="event-grid"><section className="panel event-feed"><PanelHeader n="01" title="EVENT STREAM" sub={`${shown.length} displayed of ${signals.length} locally fetched rows`}/><div className="event-list">{shown.length?shown.map(s=><button key={s.id} className={`event-item ${selected?.id===s.id&&!manual?'selected':''}`} onClick={()=>choose(s)}><div><span className={`source-type ${s.source_type==='financial_news'?'news':'social'}`}>{sourceName(s.source_type)}</span>{pill(s.status,s.status==='completed'?'good':s.status==='dead'?'bad':'neutral')}</div><b>{s.text}</b><div className="item-foot"><span>{s.ticker}</span><span>{asDate(s.first_seen_at)}</span></div></button>):<Empty title="No matching records" body="Try another filter or verify that the ingestion worker has processed records."/>}</div></section>
 <section className="panel evidence"><PanelHeader n="02" title="SELECTED EVIDENCE" sub="Original source and timestamp provenance"/>{record?<><div className="evidence-top">{pill(record.source_type==='financial_news'?'Financial news':'Public social post','iris')} {pill(record.status)}</div><h2>{record.text}</h2><div className="detail-fields"><Field label="TICKER">{record.ticker} <span className="muted">· {record.match_method}</span></Field><Field label="PROVIDER">{record.source_name}</Field><Field label="PUBLISHED">{record.published_at?asDate(record.published_at):'Unverified / unavailable'}</Field><Field label="FIRST OBSERVED">{asDate(record.first_seen_at)}</Field><Field label="TIMESTAMP QUALITY">{record.timestamp_quality.replaceAll('_',' ')}</Field><Field label="INFERENCE STATE">{record.status}</Field></div><div className="evidence-actions"><SourceLink url={record.source_url}/></div>{record.last_error&&<ErrorBanner message={`Processing: ${record.last_error}`}/>}</>:manual?<><h2>Manual inference result</h2><p>Source: manual user-provided text. It is not an ingested news or social record.</p></>:<Empty title="Choose an event" body="Select a source-linked record to inspect the evidence and model output."/>}</section>
 <section className="panel diagnostics"><PanelHeader n="03" title="THREE-MODEL DIAGNOSTICS" sub="Model outputs are estimates, not facts"/>{inference?<><div className="model-card"><span className="overline color-blue">FINBERT · SENTIMENT</span><strong className={scoreTone(inference.sentiment.score)}>{inference.sentiment.score>=0?'+':''}{decimal(inference.sentiment.score,3)}</strong><span>{sentimentLabel(inference.sentiment.score)} · −1 to +1 scale</span></div><div className="model-card"><span className="overline color-iris">FINBERT · EVENT</span><strong>{inference.event.class_name}</strong><span>Confidence {formatPercentFraction(inference.event.confidence,1)}</span><div className="probabilities">{Object.entries(probabilities).sort((a,b)=>b[1]-a[1]).map(([label,val])=><div className="mini-bar" key={label}><span>{label}</span><div><i style={{width:`${Math.min(100,Math.max(0,val*100))}%`}}/></div><b>{decimal(val*100,1)}%</b></div>)}</div></div><div className="model-card"><span className="overline color-amber">XGBOOST · IMPACT</span><strong>{formatImpact(inference.impact.score)}</strong><span>Estimated reaction magnitude {formatPercentFraction(inference.impact.predicted_realized_impact)}</span></div>{inference.warnings?.length?<div className="notice"><CircleAlert size={16}/>{inference.warnings.join(' · ')}</div>:null}</>:<Empty title="No inference available" body="Only completed records contain real FinBERT and XGBoost outputs."/>}<p className="fineprint">An impact score does not indicate return direction, default probability, or a guaranteed price change.</p>{record?.status==='completed'&&<><button className="small-button full" disabled={proposalBusy} onClick={askProposal}>{proposalBusy?<LoaderCircle className="spin" size={15}/>:<SlidersHorizontal size={15}/>} Propose stress scenario</button><ErrorBanner message={proposalError}/>{proposal&&<div className="proposal-box"><span>Proposed: {proposal.proposal.suggested_scenario.name}</span><small>{proposal.proposal.severity_source}</small><button className="primary-button" onClick={()=>onModuleB(proposal)}>Review in Stress Lab <ArrowRight size={14}/></button></div>}</>}</section></div>
 <section className="panel manual-panel"><div><PanelHeader n="04" title="MANUAL AI INFERENCE" sub="Run the existing models on your own text; requires available market data"/><p>This is a separate manual test, never presented as ingested news.</p></div><form className="manual-form" onSubmit={submit}><label className="form-label">SUPPORTED TICKER<select value={manualTicker} onChange={e=>setManualTicker(e.target.value)}>{(tickers.length?tickers:['AAPL']).map(t=><option key={t}>{t}</option>)}</select></label><label className="form-label manual-text">FINANCIAL TEXT<textarea minLength={5} maxLength={10000} rows={3} value={manualText} onChange={e=>setManualText(e.target.value)} placeholder="Paste a financial statement or headline to analyze…"/></label><button className="primary-button" type="submit" disabled={manualBusy||manualText.trim().length<5}>{manualBusy?<LoaderCircle size={16} className="spin"/>:<WandSparkles size={16}/>} Analyze text</button></form><ErrorBanner message={manualError}/></section>
 </div>;
}
function ModuleA({index,indexError,rebalance,setRebalance,mode}:{index:IndexData|null;indexError:string|null;rebalance:RebalanceEnvelope|null;setRebalance:(r:RebalanceEnvelope)=>void;mode:ViewMode}){
 const [showAll,setShowAll]=useState(false);const [busy,setBusy]=useState(false);const [error,setError]=useState('');const [run,setRun]=useState<{metrics:Record<string,Record<string,unknown>>;session_count:number}|null>(null);const [backBusy,setBackBusy]=useState(false);const [backError,setBackError]=useState('');
 const reb=rebalance?.recommendation;const tickers=index?.tickers??[];const limitPct=index?.constraints??{};const proposed=reb?.target_weights??{};const current=reb?.current_weights??index?.baseline_weights??{};
 const maximum=Math.max(.1,...Object.values(current).map(Number),...Object.values(proposed).map(Number));
 const ranked=[...tickers].sort((a,b)=>Math.abs((proposed[b]??current[b]??0)-(current[b]??0))-Math.abs((proposed[a]??current[a]??0)-(current[a]??0)));
 const visibleTickers=mode==='analyst'||showAll?tickers:ranked.slice(0,5);
 const onRebalance=async()=>{setBusy(true);setError('');try{setRebalance(await api.rebalanceSaved(100));}catch(e){setError(errorMessage(e));}finally{setBusy(false)}};
 const onBacktest=async()=>{setBackBusy(true);setBackError('');try{setRun(await api.backtest('holdout'));}catch(e){setBackError(errorMessage(e));}finally{setBackBusy(false)}};
 return <div className="page"><div className="page-title"><div className="eyebrow">TACTICAL INDEX LAB / MODULE A</div><h1>Turn signals into constraints.<br/><em>Not automatic trades.</em></h1><p>A research-only, fifteen-stock index optimizer that proposes bounded allocations from saved AI predictions.</p></div>
 <div className="metrics-grid"><Metric label="SUPPORTED CONSTITUENTS" value={index?String(tickers.length):'—'} sub="From live Module A config"/><Metric label="MAX ONE-WAY TURNOVER" value={formatPercentFraction(limitPct.max_one_way_turnover)} sub="Rebalance constraint" kind="amber"/><Metric label="MAX STOCK TILT" value={typeof limitPct.max_position_deviation==='number'?`±${decimal(limitPct.max_position_deviation*100,1)} pp`:'—'} sub="Deviation from reference"/><Metric label="MAX SECTOR TILT" value={typeof limitPct.max_sector_deviation==='number'?`±${decimal(limitPct.max_sector_deviation*100,1)} pp`:'—'} sub="Sector exposure deviation" kind="blue"/></div>
 <ErrorBanner message={indexError}/><div className="module-a-grid"><section className="panel allocation"><PanelHeader n="01" title="ALLOCATION SHIFT ANALYZER" sub="Blue = current, iris = recommended. Percentages from API results only."/>{tickers.length?<div className="allocation-table"><div className="alloc-head"><span>TICKER</span><span>CURRENT / PROPOSED WEIGHT</span><span>CURRENT</span><span>PROPOSED</span></div>{visibleTickers.map(t=><div className="alloc-row" key={t}><strong>{t}</strong><div className="alloc-track"><i style={{width:`${(current[t]??0)/maximum*100}%`}}/><em style={{width:`${proposed[t]===undefined?0:proposed[t]/maximum*100}%`}}/></div><span>{formatPercentFraction(current[t])}</span><b className="iris-text">{proposed[t]===undefined?'—':formatPercentFraction(proposed[t])}</b></div>)}{mode==='quick'&&tickers.length>5&&<button className="mono-expand" onClick={()=>setShowAll(!showAll)}>{showAll?'Show top changes':'View all 15 stocks'} <ChevronDown size={15}/></button>}</div>:<Empty title="Index not loaded" body="Start FastAPI and confirm GET /module-a/index."/>}</section>
 <section className="panel strategy-panel"><PanelHeader n="02" title="CONSTRAINT GUARDRAILS" sub="Propose, verify, explain"/>{[['Stock tilt',formatPercentFraction(limitPct.max_position_deviation)],['Sector deviation',formatPercentFraction(limitPct.max_sector_deviation)],['One-way turnover',formatPercentFraction(limitPct.max_one_way_turnover)],['Trade execution','Disabled']].map(([a,b])=><div className="constraint-row" key={a}><strong>{b}</strong><span>{a}</span></div>)}<div className="notice"><LockKeyhole size={16}/>No brokerage integration or automatic order placement.</div><button className="primary-button full" disabled={busy||!index} onClick={onRebalance}>{busy?<LoaderCircle size={16} className="spin"/>:<WandSparkles size={16}/>} {isReviewerPreview?'View recorded rebalance':'Generate from saved signals'}</button><ErrorBanner message={error}/>{reb&&<div className="result-mini"><div><span>Signals considered</span><b>{rebalance?.saved_signals_considered}</b></div><div><span>One-way turnover</span><b>{formatPercentFraction(reb.one_way_turnover)}</b></div><div><span>Modeled transaction costs</span><b>{formatPercentFraction(reb.estimated_transaction_cost_fraction,4)}</b></div>{reb.warnings?.map((w,i)=><p key={i}>{w}</p>)}</div>}</section></div>
 <section className="panel backtest"><div className="between"><PanelHeader n="03" title="HISTORICAL STRATEGY COMPARISON" sub="Requires local holdout dataset; runs research backtest on demand"/><button className="secondary-button" onClick={onBacktest} disabled={backBusy||isReviewerPreview} title={isReviewerPreview?'Requires local Python backend':undefined}>{backBusy?<LoaderCircle size={15} className="spin"/>:<Play size={15}/>} Run holdout backtest</button></div><ErrorBanner message={backError}/>{run?<><p className="muted">Completed {run.session_count} sessions. The server determines available metrics and strategy coverage.</p><div className="backtest-grid">{Object.entries(run.metrics).map(([name,met])=><div className="backtest-metric" key={name}><b>{name.replaceAll('_',' ')}</b>{Object.entries(met).slice(0,6).map(([k,v])=><div key={k}><span>{k.replaceAll('_',' ')}</span><strong>{typeof v==='number'?decimal(v,4):String(v)}</strong></div>)}</div>)}</div></>:<Empty title="No backtest run in this session" body="Click Run holdout backtest to load real server-computed metrics. We do not display preset returns as live results."/>}<p className="fineprint">Research simulation with idealized same-session closing fills. Past holdout performance is not investment performance or a forecast.</p></section>
 {reb?.changes?.some(x=>Math.abs(x.change_percentage_points)>.0001)&&<section className="panel explanation"><PanelHeader n="04" title="WHY THE WEIGHTS CHANGED"/>{reb.changes.filter(x=>Math.abs(x.change_percentage_points)>.0001).map(x=><div className="explanation-item" key={x.ticker}><strong>{x.ticker} <span className={x.change_percentage_points>=0?'pos':'neg'}>{x.change_percentage_points>=0?'+':''}{decimal(x.change_percentage_points,3)} pp</span></strong><p>{x.explanation}</p></div>)}</section>}
 </div>;
}
function ModuleB({portfolio,library,proposal,clearProposal,mode}:{portfolio:Portfolio|null;library:ScenarioLibrary|null;proposal:Proposal|null;clearProposal:()=>void;mode:ViewMode}){
 const [family,setFamily]=useState('');const [tier,setTier]=useState('moderate');const [issuer,setIssuer]=useState('');
 const [shock,setShock]=useState<Scenario|null>(null);const [reviewed,setReviewed]=useState(false);
 const [result,setResult]=useState<StressResult|null>(()=>isReviewerPreview?recordedStress:null);const [busy,setBusy]=useState(false);const [error,setError]=useState('');
 const [activeTab,setTab]=useState<'stress'|'whatif'|'reverse'>('stress');const [positionId,setPositionId]=useState('');const [scale,setScale]=useState('0.75');const [lossBudget,setLossBudget]=useState('100000');const [whatIf,setWhatIf]=useState<{before:StressResult;after:StressResult;net_effect_change_usd:number;warning:string}|null>(null);const [reverse,setReverse]=useState<{found_breach:boolean;first_breach_on_grid?:{multiplier:number;loss_usd:number}|null;method:string;warning:string;truncated_at_multiplier?:number|null}|null>(null);
 const families=Object.keys(library?.scenario_library??{});const templates=library?.scenario_library[family]??{};
 const selected=shock ?? (proposal?.proposal.suggested_scenario ?? templates[tier] ?? null);
 useEffect(()=>{if(families.length&&!family)setFamily(families[0]);},[families, family]);
 const changeFamily=(v:string)=>{setFamily(v);setShock(null);setReviewed(false);setResult(null);setWhatIf(null);setReverse(null);if(proposal)clearProposal()};
 const changeTier=(v:string)=>{setTier(v);setShock(null);setReviewed(false);setResult(null);if(proposal)clearProposal()};
 const setField=(name:keyof Scenario['shocks'],n:string)=>{if(!selected)return;setShock({...selected,shocks:{...selected.shocks,[name]:Number(n)}});setReviewed(false);setResult(null)};
 // The proposal's issuer shocks are already resolved, while the catalog may contain __ISSUER__ placeholders.
 const resolveScenario=()=>scenarioForExecution(selected,issuer) as unknown as Scenario;
 const run=async()=>{setBusy(true);setError('');setResult(null);try{if(!reviewed)throw Error('Review the scenario assumptions and acknowledge them first.');setResult(await api.stress(resolveScenario()));setTab('stress')}catch(e){setError(errorMessage(e))}finally{setBusy(false)}};
 const runWhatIf=async()=>{setBusy(true);setError('');try{if(!reviewed)throw Error('Review assumptions first.');if(!positionId)throw Error('Choose a portfolio position.');const n=Number(scale);if(!Number.isFinite(n)||n<0||n>2)throw Error('Position scaling must be between 0 and 2.');setWhatIf(await api.whatIf(resolveScenario(),[{position_id:positionId,scale:n}]))}catch(e){setError(errorMessage(e))}finally{setBusy(false)}};
 const runReverse=async()=>{setBusy(true);setError('');try{if(!reviewed)throw Error('Review assumptions first.');const n=Number(lossBudget);if(!Number.isFinite(n)||n<=0)throw Error('Loss budget must be positive.');setReverse(await api.reverseStress(resolveScenario(),n))}catch(e){setError(errorMessage(e))}finally{setBusy(false)}};
 const stats=portfolio?.summary;const counts=stats?.counts;
 return <div className="page"><div className="page-title"><div className="eyebrow">MULTI-ASSET RESEARCH / MODULE B</div><h1>Pressure-test the portfolio.<br/><em>Before pressure arrives.</em></h1><p>Review hypothetical shocks across loans, bonds, swaps and options. Keep mark-to-market and projected credit-loss horizons separate.</p></div>
 <div className="metrics-grid"><Metric label="SYNTHETIC POSITIONS" value={stats?String(stats.position_count):'—'} sub="Research positions only"/><Metric label="LOAN EAD" value={fmtMoney(stats?.loan_ead_usd)} sub={`${counts?.loan??'—'} loans · exposure at default`} kind="blue"/><Metric label="BOND MARKET VALUE" value={fmtMoney(stats?.bond_market_value_usd)} sub={`${counts?.bond??'—'} bonds · reference value`}/><Metric label="DERIVATIVE NOTIONAL" value={fmtMoney(stats?.derivative_reference_notional_usd)} sub="Reference notional ≠ NAV" kind="amber"/></div>
 {proposal&&<div className="proposal-alert"><Sparkles size={18}/><div><b>Scenario proposed from a genuine saved signal</b><p>{proposal.proposal.justification} The proposed shocks are not model-predicted rates or PDs.</p></div><button onClick={()=>{clearProposal();setShock(null);setReviewed(false);setResult(null)}} className="icon-btn" aria-label="Dismiss proposal"><X size={16}/></button></div>}
 <div className="stress-grid"><section className="panel scenario-panel"><PanelHeader n="01" title="SCENARIO DESIGN" sub="User-reviewable illustrative shock" accent="amber"/><div className="form-stack"><label className="form-label">EVENT FAMILY<select value={proposal?.proposal.suggested_scenario.event_class??family} disabled={!!proposal} onChange={e=>changeFamily(e.target.value)}>{families.map(x=><option key={x} value={x}>{x}</option>)}</select></label><label className="form-label">SCENARIO INTENSITY<select value={proposal?.proposal.suggested_scenario.severity??tier} disabled={!!proposal} onChange={e=>changeTier(e.target.value)}>{Object.keys(templates).map(x=><option key={x} value={x}>{x}</option>)}</select></label><label className="form-label">ISSUER (only for issuer-specific templates)<select value={issuer} onChange={e=>{setIssuer(e.target.value);setReviewed(false)}}><option value="">Select ticker when required</option>{['AAPL','MSFT','NVDA','QCOM','GOOGL','CMCSA','AMZN','TSLA','BKNG','SBUX','COST','PEP','GILD','AMGN','CSX'].map(x=><option key={x}>{x}</option>)}</select></label>{selected?<div className="shock-fields">{([['rate_bps','Rates (basis points)'],['spread_bps','Credit spread (bps)'],['equity_pct','Equity shock (%)'],['volatility_points','Volatility points'],['pd_multiplier','PD multiplier']] as const).map(([k,label])=><label className="form-label" key={k}>{label}<input type="number" step="any" value={selected.shocks[k]} onChange={e=>setField(k,e.target.value)}/></label>)}</div>:<Empty title="No scenario library" body="GET /module-b/scenarios must return scenario templates."/>}<div className="notice"><CircleAlert size={16}/>Impact score is not a calibrated prediction of bond spreads, PD or interest rates.</div><label className="checkline"><input type="checkbox" checked={reviewed} onChange={e=>setReviewed(e.target.checked)}/><span>I reviewed these hypothetical assumptions and understand no trades or portfolio orders will be executed.</span></label><button className="primary-button full" disabled={!selected||busy||!reviewed} onClick={run}>{busy?<LoaderCircle size={16} className="spin"/>:<ShieldCheck size={16}/>} Run reviewed stress test</button><ErrorBanner message={error}/></div></section>
 {mode==='analyst'&&<section className="panel mechanisms"><PanelHeader n="02" title="FOUR LOSS MECHANISMS" sub="One shock, distinct measurement bases"/>{[['LOANS',`${counts?.loan??'—'} positions`,'12-month ECL change','PD × LGD × EAD','blue'],['BONDS',`${counts?.bond??'—'} positions`,'Instantaneous price revaluation','Duration + convexity','iris'],['SWAPS',`${counts?.swap??'—'} positions`,'Signed interest-rate sensitivity','DV01 × rate shock','amber'],['OPTIONS',`${counts?.option??'—'} positions`,'Option mark-to-market','European Black–Scholes','rose']].map(x=><div key={x[0]} className="mechanism"><span className={`mech-strip ${x[4]}`}/><div><b className={`color-${x[4]}`}>{x[0]}</b><h3>{x[2]}</h3><p>{x[3]}</p></div><span className="mechanism-count">{x[1]}</span></div>)}<div className="notice small">Projected 12-month credit losses are not contemporaneous marked-to-market changes. The optional mixed-horizon indicator requires explicit labeling.</div></section>}
 <section className="panel stress-result"><PanelHeader n="03" title={isReviewerPreview?'RECORDED STRESS EXAMPLE':'STRESS RESULTS'} sub={result?result.scenario.name:'Awaiting reviewed scenario'} accent="amber"/>{result?<><div className="outcome-top"><span>Instantaneous MTM P&amp;L</span><strong>{fmtMoney(result.totals.instantaneous_mtm_pnl_usd)}</strong></div><div className="outcome-top"><span>Incremental 12-month ECL</span><strong>{fmtMoney(result.totals.incremental_12m_expected_credit_loss_usd)}</strong></div><div className="warning-band">Mixed-horizon illustrative effect: {fmtMoney(result.totals.illustrative_mixed_horizon_net_effect_usd)} (not NAV or VaR)</div><div className="split-title">INDICATIVE CONTRIBUTION BY INSTRUMENT</div>{Object.entries(result.loss_contribution_by_type_usd).map(([name,value])=><div className="result-line" key={name}><span>{name.toUpperCase()}</span><b>{fmtMoney(value)}</b></div>)}{result.override_coverage&&!result.override_coverage.all_overrides_matched&&<div className="notice">Unmatched issuer/sector shocks are reported by the backend. Review before interpretation.</div>}</>:<Empty title="No outcome generated" body="Choose, review and execute a synthetic scenario. No sample losses are displayed."/>}</section></div>
 {mode==='analyst'&&<section className="panel additional-stress"><div className="tabs"><button className={activeTab==='stress'?'active':''} onClick={()=>setTab('stress')}>Stress context</button><button className={activeTab==='whatif'?'active':''} onClick={()=>setTab('whatif')}>What-if exposure</button><button className={activeTab==='reverse'?'active':''} onClick={()=>setTab('reverse')}>Reverse stress</button></div>{activeTab==='stress'?<div className="tab-content"><b>Read the results carefully</b><p>Instantaneous marked-to-market profit/loss and projected 12-month incremental expected credit loss use different time horizons. No combined value should be interpreted as regulatory capital, NAV or guaranteed loss.</p><span className="fineprint">Scenarios are hypothetical, and all portfolio positions are synthetic.</span></div>:activeTab==='whatif'?<div className="tab-content"><p>Scale the exposure of a single synthetic position, then compare the stress results before and after. Does not model hedging or financing costs.</p><div className="tab-controls"><label className="form-label">POSITION<select value={positionId} onChange={e=>setPositionId(e.target.value)}><option value="">Select position</option>{portfolio?.positions.map(p=><option key={p.id} value={p.id}>{p.id} · {p.name}</option>)}</select></label><label className="form-label">EXPOSURE MULTIPLIER<input type="number" min="0" max="2" step="0.05" value={scale} onChange={e=>setScale(e.target.value)}/></label><button className="secondary-button" disabled={busy||!reviewed||!positionId} onClick={runWhatIf}>Run what-if</button></div>{whatIf&&<div className="compact-results"><span>Before mixed-horizon net: {fmtMoney(whatIf.before.totals.illustrative_mixed_horizon_net_effect_usd)}</span><span>After: {fmtMoney(whatIf.after.totals.illustrative_mixed_horizon_net_effect_usd)}</span><b>Change: {fmtMoney(whatIf.net_effect_change_usd)}</b><small>{whatIf.warning}</small></div>}</div>:<div className="tab-content"><p>Find the first sampled severity multiplier that breaches a user-chosen illustrative mixed-horizon loss budget. This is not VaR or a globally optimal threshold.</p><div className="tab-controls"><label className="form-label">LOSS BUDGET (USD)<input type="number" min="1" step="1000" value={lossBudget} onChange={e=>setLossBudget(e.target.value)}/></label><button className="secondary-button" disabled={busy||!lossBudget} onClick={runReverse}>Run reverse stress</button></div>{reverse&&<div className="compact-results"><b>{reverse.found_breach?'First sampled threshold breach found':'No breach in evaluated grid'}</b><span>{reverse.first_breach_on_grid?`Multiplier ${decimal(reverse.first_breach_on_grid.multiplier,2)}, indicative loss ${fmtMoney(reverse.first_breach_on_grid.loss_usd)}`:'No threshold within feasible evaluated factors'}</span><small>{reverse.method}</small></div>}</div>}</section>}
 </div>;
}

function BujjiAssistant({ mode, selectedSignalId }: { mode: ViewMode; selectedSignalId?: string | null }) {
  const [open, setOpen] = useState(() => typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('bujji') === 'open');
  const [messages, setMessages] = useState<Array<{ role: 'user' | 'assistant'; content: string; status?: string }>>([
    {
      role: 'assistant',
      content: "Hi! I'm Bujji, your RiskPulse AI guide. Ask me about financial news, AI risk signals, index rebalancing, or portfolio stress testing."
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const starters = [
    "What does RiskPulse AI do?",
    "What is financial sentiment?",
    "What does an impact score of 7/10 mean?",
    "How does Index Rebalancing work?",
    "What is Expected Credit Loss?",
    "How does Stress Testing work?"
  ];

  useEffect(() => {
    if (open && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, open, loading]);

  const send = async (textToSend?: string) => {
    const q = (textToSend ?? input).trim();
    if (!q || loading) return;
    setInput('');
    const newMessages = [...messages, { role: 'user' as const, content: q }];
    setMessages(newMessages);
    setLoading(true);
    try {
      const history = newMessages
        .slice(-6)
        .map(m => ({ role: m.role, content: m.content }));
      const res = await api.assistantChat({
        message: q,
        history,
        mode,
        selected_signal_id: selectedSignalId ?? null,
      });
      setMessages(prev => [...prev, { role: 'assistant', content: res.reply, status: res.status }]);
    } catch {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: "Sorry, I encountered an issue reaching the assistant service. Please check the backend connection.",
          status: 'error',
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  };

  return (
    <>
      <button
        type="button"
        className="bujji-launcher"
        onClick={() => setOpen(o => !o)}
        aria-label={open ? "Close Bujji AI Assistant" : "Open Bujji AI Assistant"}
        title={open ? "Close Bujji" : "Ask Bujji (RiskPulse AI Guide)"}
      >
        {open ? (
          <span style={{ fontSize: 18, fontWeight: 700 }}>✕</span>
        ) : (
          <svg viewBox="0 0 24 24" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
        )}
      </button>

      {open && (
        <div className="bujji-panel" role="dialog" aria-label="Bujji Financial Assistant">
          <div className="bujji-header">
            <div className="bujji-header-left">
              <span className="bujji-title">Bujji</span>
              <span className="bujji-mode-pill">{mode === 'analyst' ? 'Analyst View' : 'Quick View'}</span>
            </div>
            <button
              type="button"
              className="bujji-close-btn"
              onClick={() => setOpen(false)}
              aria-label="Close assistant"
              title="Close assistant"
            >
              ✕
            </button>
          </div>

          <div className="bujji-body" ref={scrollRef}>
            {messages.map((m, i) => (
              <div key={i} className={`bujji-msg ${m.role}`}>
                <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
              </div>
            ))}

            {messages.length === 1 && !loading && (
              <div className="bujji-starters">
                <div style={{ fontSize: 11, color: '#6c757d', fontWeight: 600, marginBottom: 2 }}>Suggested questions:</div>
                {starters.map((s, idx) => (
                  <button key={idx} type="button" className="bujji-starter-btn" onClick={() => void send(s)}>
                    {s}
                  </button>
                ))}
              </div>
            )}

            {loading && (
              <div className="bujji-loading">
                <span>●</span> Bujji is thinking...
              </div>
            )}
          </div>

          <div className="bujji-footer">
            <div className="bujji-input-row">
              <input
                className="bujji-input"
                placeholder="Ask Bujji anything..."
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={loading}
              />
              <button
                type="button"
                className="bujji-send-btn"
                onClick={() => void send()}
                disabled={loading || !input.trim()}
                aria-label="Send message"
              >
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              </button>
            </div>
            <div className="bujji-disclaimer">
              AI-generated explanations may contain errors. Research use only; not investment advice.
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default function App(){
 const [page,setPage]=useState<Page>('home');const [mode,setMode]=useState<ViewMode>('quick'); const [summary,setSummary]=useState<Load<Summary>>(initial()); const [signalPage,setSignalPage]=useState<Load<SignalsPage>>(initial());const [health,setHealth]=useState('unknown');const [ingestion,setIngestion]=useState('unknown');
 const [idx,setIdx]=useState<Load<IndexData>>(initial());const [portfolio,setPortfolio]=useState<Load<Portfolio>>(initial());const [library,setLibrary]=useState<Load<ScenarioLibrary>>(initial());
 const [selected,setSelected]=useState<Signal|null>(null);const [proposal,setProposal]=useState<Proposal|null>(null);const [rebalance,setRebalance]=useState<RebalanceEnvelope|null>(null);const [refreshing,setRefreshing]=useState(false);const [lastUpdated,setLastUpdated]=useState<string|null>(null);
 useEffect(()=>{
  const m=new URLSearchParams(window.location.search).get('mode') as ViewMode;
  if(m==='quick'||m==='analyst')setMode(m);
  const h=window.location.hash.replace('#','') as Page;
  if(['home','events','moduleA','moduleB'].includes(h))setPage(h);
 },[]);
 const refresh=useCallback(async()=>{setRefreshing(true);const [a,b,c,d,e,f,g]=await Promise.allSettled([api.summary(),api.signals({limit:100}),api.health(),api.ingestionHealth(),api.index(),api.portfolio(),api.scenarios()]);
 setSummary(a.status==='fulfilled'?{value:a.value,error:null,pending:false}:{value:null,error:errorMessage(a.reason),pending:false});
 setSignalPage(b.status==='fulfilled'?{value:b.value,error:null,pending:false}:{value:null,error:errorMessage(b.reason),pending:false});
 if(b.status==='fulfilled'&&b.value.items.length>0){setSelected(prev=>prev??(b.value.items.find(x=>x.status==='completed'&&x.inference)??b.value.items[0]));}
 setHealth(c.status==='fulfilled'?c.value.status:'unavailable');setIngestion(d.status==='fulfilled'?d.value.status:'unavailable');
 setIdx(e.status==='fulfilled'?{value:e.value,error:null,pending:false}:{value:null,error:errorMessage(e.reason),pending:false});
 setPortfolio(f.status==='fulfilled'?{value:f.value,error:null,pending:false}:{value:null,error:errorMessage(f.reason),pending:false});
 setLibrary(g.status==='fulfilled'?{value:g.value,error:null,pending:false}:{value:null,error:errorMessage(g.reason),pending:false});
 setLastUpdated(new Date().toISOString());setRefreshing(false);
 },[]);
 useEffect(()=>{void refresh();const timer=setInterval(()=>void refresh(),60_000);return()=>clearInterval(timer)},[refresh]);
 const signals=signalPage.value?.items??[];
 const choose=(s:Signal)=>{setSelected(s);setPage('events')};
 const goProposal=(p:Proposal)=>{setProposal(p);setPage('moduleB')};
 const hasProblem=summary.error||signalPage.error;
 return <>
  {isReviewerPreview&&<div role="status" style={{background:'#f3f4f6',color:'#1f2937',borderBottom:'1px solid #e5e7eb',padding:'10px 24px',fontSize:13,lineHeight:1.5,display:'flex',justifyContent:'space-between',alignItems:'center',flexWrap:'wrap',gap:8}}><div><strong>☁️ Cloud Demo Preview (Models Offline)</strong> — {previewNotice}</div><a href="https://github.com/shanmukhdatta/nitj-shanmukhdatta-hackthon" target="_blank" rel="noopener noreferrer" style={{color:'#111827',fontWeight:600,textDecoration:'underline'}}>GitHub Repository &amp; Code</a></div>}
  {page==='home'?<Home onStart={()=>setPage('events')} onTab={setPage} onMode={setMode}/>:<Shell page={page} onPage={setPage} mode={mode} onMode={setMode} summary={summary.value} health={health} ingestion={ingestion} loading={refreshing} refresh={()=>void refresh()} selectedTicker={selected?.ticker??''}>
  <div className="dataset-strip"><span><Dot tone={hasProblem?'amber':'green'}/>{isReviewerPreview?'Cloud Demo Mode (Interactive)':hasProblem?'Backend connection incomplete':'Backend responses loaded'}</span><span><Clock3 size={13}/> {lastUpdated?`UI refreshed ${asDate(lastUpdated)}`:'Waiting for first response'}</span><span><LockKeyhole size={13}/> Simulation / research only</span></div>
  <ErrorBanner message={hasProblem?`Backend read failed: ${hasProblem}. Start FastAPI on 127.0.0.1:8000; dev Vite proxies /api.`:null}/>
  {page==='overview'?<Overview summary={summary.value} signals={signals} goSignal={choose} goPage={setPage} health={health} ingestion={ingestion}/>:page==='events'?<><SignalSummary summary={summary.value} signals={signals}/><Events signals={signals} selected={selected} onSelect={setSelected} onModuleB={goProposal} tickers={idx.value?.tickers??[]} analyzeManual={api.analyze}/></>:page==='moduleA'?<ModuleA index={idx.value} indexError={idx.error} rebalance={rebalance} setRebalance={setRebalance} mode={mode}/>:<ModuleB portfolio={portfolio.value} library={library.value} proposal={proposal} clearProposal={()=>setProposal(null)} mode={mode}/>}
  </Shell>}
  <BujjiAssistant mode={mode} selectedSignalId={selected?.id}/>
 </>;
}

