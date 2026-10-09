import { useCallback, useEffect, useMemo, useState, type FormEvent } from 'react';
import {
  Activity, ArrowRight, ArrowUpRight, BrainCircuit, ChartNoAxesCombined,
  ChevronDown, ChevronRight, CircleAlert, Clock3, ExternalLink, Gauge, LayoutDashboard,
  LoaderCircle, LockKeyhole, Newspaper, Play, RadioTower, RefreshCw, Search, ShieldAlert, ShieldCheck,
  SlidersHorizontal, Sparkles, WandSparkles, X
} from 'lucide-react';
import { api } from './api';
import { countStatus, eventCategoryCounts, fmtMoney, formatPercentFraction, recentSignalMeans, safeSourceUrl, scenarioForExecution, sentimentLabel } from './lib/helpers.mjs';
import type { IndexData, Inference, Page, Portfolio, Proposal, RebalanceEnvelope, Scenario, ScenarioLibrary, Signal, SignalsPage, StressResult, Summary } from './types';

type Load<T> = { value: T | null; error: string | null; pending: boolean };
const initial = <T,>(): Load<T> => ({ value: null, error: null, pending: true });
function errorMessage(err: unknown) { return err instanceof Error ? err.message : String(err); }
function asDate(v?: string | null) {
  if (!v) return 'Unknown';
  const d = new Date(v);
  return Number.isNaN(d.getTime()) ? 'Unknown' : d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}
function scoreTone(score: number | null | undefined) {
  return typeof score === 'number' ? (score > 0.1 ? 'pos' : score < -0.1 ? 'neg' : 'neutral') : 'neutral';
}
function decimal(value: number | null | undefined, decimals = 2) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(decimals) : '—';
}
const pill = (text: string, variant = 'neutral') => <span className={`pill ${variant}`}>{text}</span>;
const formatImpact = (n?: number | null) => typeof n === 'number' ? `${n}/10` : '—';
const sourceName = (s?: string) => s === 'financial_news' ? 'NEWS' : s === 'social_media' ? 'SOCIAL' : 'MANUAL';

function Metric({ label, value, sub, kind = 'iris' }: { label: string; value: string; sub: string; kind?: string }) {
  return (
    <div className="metric card">
      <span className="metric-label">{label}</span>
      <strong className="metric-value">{value}</strong>
      <span className="metric-sub">{sub}</span>
      <i className={`metric-notch ${kind}`} />
    </div>
  );
}

function Empty({ title, body }: { title: string; body: string }) {
  return (
    <div className="empty">
      <CircleAlert size={22} />
      <b>{title}</b>
      <span>{body}</span>
    </div>
  );
}

function PanelHeader({ n, title, sub, accent = 'iris' }: { n: string; title: string; sub?: string; accent?: string }) {
  return (
    <div className="panelhead">
      <span className={`overline color-${accent}`}>{n} / {title}</span>
      {sub && <span className="panel-sub">{sub}</span>}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="field">
      <span className="field-label">{label}</span>
      <div className="field-value">{children}</div>
    </div>
  );
}

function SourceLink({ url, label = 'Open original source' }: { url: string; label?: string }) {
  const safe = safeSourceUrl(url);
  return safe ? (
    <a className="source-link" href={safe} target="_blank" rel="noopener noreferrer">
      {label} <ExternalLink size={12} />
    </a>
  ) : (
    <span className="muted">Source URL unavailable</span>
  );
}

function Dot({ tone = 'iris' }: { tone?: string }) {
  return <span className={`dot ${tone}`} />;
}

function ErrorBanner({ message }: { message: string | null }) {
  return message ? (
    <div className="error-banner" role="alert">
      <CircleAlert size={16} />
      <span>{message}</span>
    </div>
  ) : null;
}

function Shell({
  page, onPage, children, summary, health, ingestion, loading, refresh, selectedTicker
}: {
  page: Page;
  onPage: (p: Page) => void;
  children: React.ReactNode;
  summary: Summary | null;
  health: string;
  ingestion: string;
  loading: boolean;
  refresh: () => void;
  selectedTicker: string;
}) {
  const links: [Page, string, typeof LayoutDashboard][] = [
    ['overview', 'Command center', LayoutDashboard],
    ['events', 'Event intelligence', RadioTower],
    ['moduleA', 'Index rebalancing', ChartNoAxesCombined],
    ['moduleB', 'Stress laboratory', ShieldAlert]
  ];
  const recents = summary?.recent_collection_runs ?? [];
  const last = recents.length ? [...recents].sort((a, b) => b.started_at.localeCompare(a.started_at))[0] : null;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Activity size={22} /></div>
          <div className="brand-word">
            <strong>RISK</strong>
            <span>PULSE / AI</span>
          </div>
        </div>
        <span className="sidebar-kicker">WORKSPACE / 01</span>
        <nav aria-label="Main navigation">
          {links.map(([key, label, Icon], i) => (
            <button
              key={key}
              className={`nav-link ${page === key ? 'active' : ''}`}
              onClick={() => onPage(key)}
            >
              <span className="nav-num">0{i + 1}</span>
              <Icon size={16} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-monitor">
          <span className="sidebar-kicker">INGESTION & RISK ENGINE</span>
          <span className="monitor-row"><Dot tone="blue" />Marketaux / GDELT · News</span>
          <span className="monitor-row"><Dot tone="iris" />Bluesky · Social Media</span>
          <span className="monitor-row">
            <Dot tone={health === 'ready' ? 'green' : 'amber'} />
            Risk Engine · {health === 'ready' ? 'Ready' : 'Checking status'}
          </span>
        </div>
        <div className="sidebar-bottom">
          <b>RESEARCH ENVIRONMENT</b>
          <p>Explainable decision-support signals. Synthetic portfolios and simulations — no live brokerage orders.</p>
          <span>S&amp;P GLOBAL &amp; CRISIL · 2026</span>
        </div>
      </aside>
      <main className="main">
        <header className="topbar">
          <div className="crumb">
            <span>RISKPULSE AI</span>
            <b>/ {page === 'overview' ? 'COMMAND CENTER' : page === 'events' ? 'EVENT INTELLIGENCE' : page === 'moduleA' ? 'TACTICAL INDEX LAB' : 'STRESS LABORATORY'}</b>
          </div>
          <div className="top-actions">
            <span className="sync-text">
              {loading ? 'Refreshing data…' : last ? `Source observed: ${asDate(last.started_at)}` : 'Collection time unrecorded'}
            </span>
            <button className="icon-btn" onClick={refresh} disabled={loading} aria-label="Refresh data" title="Refresh data">
              <RefreshCw size={16} className={loading ? 'spin' : ''} />
            </button>
            <span className="env-tag">
              <Dot tone={ingestion === 'configured' ? 'green' : 'amber'} />
              {ingestion === 'configured' ? 'LIVE INGESTION ACTIVE' : 'VERIFY SOURCES'}
            </span>
            <span className="avatar" title="RiskPulse AI Operator">RP</span>
          </div>
        </header>
        {selectedTicker && (
          <div className="context-ribbon">
            <Sparkles size={14} />
            <span>Active event context: <b>{selectedTicker}</b></span>
            <span>Evidence provenance tracked across all modules</span>
          </div>
        )}
        {children}
      </main>
    </div>
  );
}

function Overview({
  summary, signals, goSignal, goPage, health, ingestion
}: {
  summary: Summary | null;
  signals: Signal[];
  goSignal: (s: Signal) => void;
  goPage: (p: Page) => void;
  health: string;
  ingestion: string;
}) {
  const completed = signals.filter(x => x.status === 'completed' && x.inference);
  const means = recentSignalMeans(signals);
  const sourceRows = summary?.recent_collection_runs ?? [];
  const news = sourceRows.filter(x => ['marketaux', 'gdelt'].includes(x.source.toLowerCase())).sort((a, b) => b.started_at.localeCompare(a.started_at))[0];
  const social = sourceRows.filter(x => x.source.toLowerCase() === 'bluesky').sort((a, b) => b.started_at.localeCompare(a.started_at))[0];
  const recent = completed.slice(0, 6);
  const cats = eventCategoryCounts(completed);
  const max = Math.max(1, ...Object.values(cats));
  const [showDiagnostics, setShowDiagnostics] = useState(false);

  return (
    <div className="page">
      <div className="hero">
        <div>
          <div className="eyebrow"><Sparkles size={13} /> Institutional Risk Engine</div>
          <h1>Financial news and social signals transformed into explainable portfolio risk intelligence.</h1>
          <p>Two genuine data streams processed through FinBERT and XGBoost to power constrained tactical index rebalancing and multi-asset stress testing.</p>
        </div>
        <button className="primary-button hero-cta" onClick={() => goPage('events')}>
          Explore Analyzed Events <ArrowUpRight size={16} />
        </button>
      </div>

      <div className="metrics-grid">
        <Metric
          label="ANALYZED EVENTS"
          value={summary ? `${countStatus(summary, 'completed')}` : '—'}
          sub={`Real FinBERT & XGBoost inference (${summary ? String((summary.status_counts ?? []).reduce((a, x) => a + Number(x.count || 0), 0)) : '—'} total records)`}
          kind="iris"
        />
        <Metric
          label="AGGREGATE SENTIMENT"
          value={decimal(means.sentiment, 3)}
          sub={`[-1, +1] tone scale across ${means.count} loaded completed records`}
          kind="blue"
        />
        <Metric
          label="AVERAGE IMPACT SEVERITY"
          value={means.impact === null ? '—' : `${decimal(means.impact, 1)} / 10`}
          sub="Modeled market reaction magnitude"
          kind="amber"
        />
        <Metric
          label="SOURCE FRESHNESS & HEALTH"
          value={news ? asDate(news.started_at) : 'Active'}
          sub={`Ingestion: ${ingestion} · AI Engine: ${health}`}
          kind="green"
        />
      </div>

      <div className="overview-center">
        <section className="panel journey">
          <PanelHeader
            n="01"
            title="RISK JOURNEY EXPLORER"
            sub="End-to-end evidence trail: From raw news and social posts to auditable portfolio actions"
          />
          <div className="journey-row">
            <div className="journey-box">
              <div className="journey-kicker blue">01 / SOURCE EVIDENCE</div>
              <h3>Capture &amp; Verify</h3>
              <p><Newspaper size={15} /> Financial news (Marketaux)</p>
              <p><RadioTower size={15} /> Public social posts (Bluesky)</p>
              <small>Original link · Ticker entity matching · Stored timestamp quality · De-duplication</small>
            </div>
            <ArrowRight className="journey-arrow" />
            <div className="journey-box bright">
              <div className="journey-kicker iris">02 / AI RISK ENGINE</div>
              <h3>Score &amp; Classify</h3>
              <p><Activity size={15} /> Continuous Sentiment [−1, +1]</p>
              <p><BrainCircuit size={15} /> 5-Class Event Classification</p>
              <p><Gauge size={15} /> Impact Severity Score 1–10</p>
              <small>FinBERT tone · FinBERT event category · XGBoost frozen deciles severity</small>
            </div>
            <ArrowRight className="journey-arrow" />
            <div className="journey-box">
              <div className="journey-kicker amber">03 / PORTFOLIO RESPONSE</div>
              <h3>Simulate &amp; Review</h3>
              <p><ChartNoAxesCombined size={15} /> Module A · 15-Stock Rebalance</p>
              <p><ShieldAlert size={15} /> Module B · Multi-Asset Stress Lab</p>
              <small>Constrained portfolio optimization and multi-asset shock sensitivity. No live brokerage trades.</small>
            </div>
          </div>
          <div className="journey-bottom">
            <span><Dot tone="iris" /> Review real signals below to inspect original text, model confidence, and downstream simulations.</span>
            <button className="small-button" onClick={() => goPage('events')}>
              Open Event Explorer <ArrowRight size={13} />
            </button>
          </div>
        </section>
      </div>

      <div className="overview-bottom">
        <section className="panel feed-panel">
          <PanelHeader
            n="02"
            title="LATEST ANALYZED FINANCIAL SIGNALS"
            sub="Click any event to inspect full provenance, model probabilities, and propose stress scenarios"
          />
          {recent.length ? (
            recent.map(s => {
              const sentScore = s.inference?.sentiment?.score;
              const eventClass = s.inference?.event?.class_name;
              return (
                <button className="signal-row" key={s.id} onClick={() => goSignal(s)}>
                  <span className={`source-type ${s.source_type === 'financial_news' ? 'news' : 'social'}`}>
                    {sourceName(s.source_type)}
                  </span>
                  <div className="signal-content">
                    <span className="signal-headline">
                      <b>[{s.ticker}]</b> {s.text}
                    </span>
                    <div className="signal-meta-row">
                      <span>{s.source_name}</span>
                      <span>·</span>
                      <span>{asDate(s.first_seen_at)}</span>
                      {typeof sentScore === 'number' && (
                        <>
                          <span>·</span>
                          {pill(`${sentScore >= 0 ? '+' : ''}${decimal(sentScore, 2)} ${sentimentLabel(sentScore)}`, scoreTone(sentScore))}
                        </>
                      )}
                      {eventClass && (
                        <>
                          <span>·</span>
                          {pill(eventClass, 'iris')}
                        </>
                      )}
                    </div>
                  </div>
                  <span className="signal-impact" title="XGBoost modeled impact score">
                    {formatImpact(s.inference?.impact.score)}
                    <ChevronRight size={14} />
                  </span>
                </button>
              );
            })
          ) : (
            <Empty title="No completed events loaded" body="Start the backend and ingestion scheduler; no demo values are substituted." />
          )}
        </section>

        <section className="panel distribution">
          <PanelHeader
            n="03"
            title="EVENT CATEGORY DISTRIBUTION"
            sub="FinBERT event classifications across completed records"
          />
          {Object.keys(cats).length ? (
            Object.entries(cats).sort((a, b) => b[1] - a[1]).map(([key, value]) => (
              <div key={key} className="bar-item">
                <span>{key}</span>
                <div className="bar-track">
                  <i style={{ width: `${(value / max) * 100}%` }} />
                </div>
                <strong>{value}</strong>
              </div>
            ))
          ) : (
            <Empty title="No category data yet" body="Event probabilities appear after analyzed signals are returned." />
          )}
          <p className="fineprint" style={{ marginTop: '16px' }}>
            Note: FinBERT exhibits a known empirical distribution skew toward Product Launch on general corporate announcements.
          </p>
        </section>
      </div>

      <div className="decision-grid">
        <button className="decision-card" onClick={() => goPage('moduleA')}>
          <div>
            <span className="overline color-iris">TACTICAL INTELLIGENCE / MODULE A</span>
            <h3>15-Stock Tactical Index Rebalancing</h3>
            <p>Translate analyzed event signals into constrained portfolio weight recommendations with sector tilt guardrails and turnover limits.</p>
          </div>
          <span>Inspect Rebalancing Recommendation <ArrowUpRight size={15} /></span>
        </button>
        <button className="decision-card" onClick={() => goPage('moduleB')}>
          <div>
            <span className="overline color-amber">STRATEGIC INTELLIGENCE / MODULE B</span>
            <h3>Multi-Asset Portfolio Stress Testing</h3>
            <p>Simulate hypothetical shock scenarios across synthetic credit loans, marked-to-market bonds, interest-rate swaps, and equity options.</p>
          </div>
          <span>Review Stress Laboratory <ArrowUpRight size={15} /></span>
        </button>
      </div>

      <div className="expandable-panel">
        <div
          className="expandable-summary"
          onClick={() => setShowDiagnostics(!showDiagnostics)}
          role="button"
          tabIndex={0}
        >
          <span>TECHNICAL INGESTION DIAGNOSTICS &amp; COLLECTION AUDIT</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '11px', color: 'var(--muted)' }}>
              {showDiagnostics ? 'Hide details' : 'Show collection runs & provider health'}
            </span>
            <ChevronDown size={15} style={{ transform: showDiagnostics ? 'rotate(180deg)' : 'none', transition: 'transform 0.15s ease' }} />
          </div>
        </div>
        {showDiagnostics && (
          <div className="expandable-content">
            <div className="source-grid">
              <div className="source-item">
                <Dot tone="blue" />
                <div>
                  <strong>Financial News (Marketaux / GDELT)</strong>
                  <small>{news ? `${news.source} · ${news.status} · ${news.inserted} inserted` : 'No recorded run available'}</small>
                  <small>{news ? asDate(news.started_at) : 'Status unknown'}</small>
                </div>
              </div>
              <div className="source-item">
                <Dot tone="iris" />
                <div>
                  <strong>Social Media (Bluesky Public Firehose)</strong>
                  <small>{social ? `${social.source} · ${social.status} · ${social.inserted} inserted` : 'No recorded run available'}</small>
                  <small>{social ? asDate(social.started_at) : 'Status unknown'}</small>
                </div>
              </div>
              <div className="source-item">
                <Dot tone={health === 'ready' ? 'green' : 'amber'} />
                <div>
                  <strong>Three-Model AI Engine Readiness</strong>
                  <small>{health === 'ready' ? 'FinBERT (Sentiment & Event) + XGBoost (Impact) loaded' : 'Unable to confirm current readiness'}</small>
                  <small>Ingestion state: {ingestion}; zero mock successes tolerated</small>
                </div>
              </div>
            </div>
            <div className="source-note" style={{ marginTop: '12px', fontSize: '11px', color: 'var(--muted)' }}>
              Provenance Note: All records are ingested through genuine REST adapters with SQLite WAL transaction guarantees. No synthetic demo news is injected.
            </div>
          </div>
        )}
      </div>

      <p className="page-disclaimer">
        Research demonstration for S&amp;P Global &amp; CRISIL Hackathon 2026. AI models provide decision-support estimates. Simulated index weights and synthetic multi-asset portfolios do not constitute real investments or automated trading orders.
      </p>
    </div>
  );
}

function Events({
  signals, selected, onSelect, onModuleB, tickers, analyzeManual
}: {
  signals: Signal[];
  selected: Signal | null;
  onSelect: (s: Signal) => void;
  onModuleB: (p: Proposal) => void;
  tickers: string[];
  analyzeManual: (x: { text: string; ticker: string; published_at: string; source: 'manual' }) => Promise<Inference>;
}) {
  const [filter, setFilter] = useState('');
  const [source, setSource] = useState('all');
  const [ticker, setTicker] = useState('');
  const [status, setStatus] = useState('completed');

  const [manualText, setManualText] = useState('');
  const [manualTicker, setManualTicker] = useState('AAPL');
  const [manual, setManual] = useState<Inference | null>(null);
  const [manualError, setManualError] = useState('');
  const [manualBusy, setManualBusy] = useState(false);
  const [showManualSection, setShowManualSection] = useState(false);

  const [proposal, setProposal] = useState<Proposal | null>(null);
  const [proposalError, setProposalError] = useState('');
  const [proposalBusy, setProposalBusy] = useState(false);
  const [showAllProbabilities, setShowAllProbabilities] = useState(false);

  const shown = useMemo(
    () => signals.filter(s =>
      (source === 'all' || s.source_type === source) &&
      (status === 'all' || s.status === status) &&
      (!ticker || s.ticker === ticker) &&
      (!filter || `${s.text} ${s.ticker} ${s.source_name}`.toLowerCase().includes(filter.toLowerCase()))
    ),
    [signals, source, status, ticker, filter]
  );

  const choose = (s: Signal) => {
    onSelect(s);
    setProposal(null);
    setProposalError('');
    setManual(null);
  };

  const askProposal = async () => {
    if (!selected) return;
    setProposalBusy(true);
    setProposalError('');
    try {
      setProposal(await api.propose(selected.id));
    } catch (e) {
      setProposalError(errorMessage(e));
    } finally {
      setProposalBusy(false);
    }
  };

  const submitManual = async (e: FormEvent) => {
    e.preventDefault();
    if (manualText.trim().length < 5) return setManualError('At least five characters of financial text required.');
    setManualBusy(true);
    setManualError('');
    setManual(null);
    try {
      setManual(await analyzeManual({
        text: manualText.trim(),
        ticker: manualTicker,
        published_at: new Date().toISOString(),
        source: 'manual'
      }));
    } catch (e) {
      setManualError(errorMessage(e));
    } finally {
      setManualBusy(false);
    }
  };

  const inference = manual ?? selected?.inference;
  const record = manual ? null : selected;
  const probabilities = inference?.event?.probabilities ?? {};

  return (
    <div className="page">
      <div className="page-title">
        <div className="eyebrow"><RadioTower size={13} /> Transparent AI Diagnostics</div>
        <h1>Event Intelligence &amp; Model Explainability</h1>
        <p>Inspect genuine financial news and social media signals. Audit FinBERT continuous sentiment, FinBERT event classification, and XGBoost impact severity with full provenance.</p>
      </div>

      <div className="filter-row">
        <label className="search-field">
          <Search size={16} />
          <input
            value={filter}
            onChange={e => setFilter(e.target.value)}
            placeholder="Search headline text, ticker, or source..."
          />
        </label>
        <label className="compact-select">
          SOURCE TYPE
          <select value={source} onChange={e => setSource(e.target.value)}>
            <option value="all">All Sources (News + Social)</option>
            <option value="financial_news">Financial News</option>
            <option value="social_media">Social Media</option>
          </select>
        </label>
        <label className="compact-select">
          INFERENCE STATUS
          <select value={status} onChange={e => setStatus(e.target.value)}>
            <option value="completed">Analyzed (Completed)</option>
            <option value="all">All Records</option>
            <option value="pending">Pending</option>
            <option value="retry">Retry</option>
            <option value="dead">Failed</option>
          </select>
        </label>
        <label className="compact-select">
          TICKER
          <select value={ticker} onChange={e => setTicker(e.target.value)}>
            <option value="">All Tickers</option>
            {tickers.map(x => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
      </div>

      <div className="event-grid">
        <section className="panel event-feed">
          <PanelHeader
            n="01"
            title="EVENT STREAM"
            sub={`${shown.length} displayed of ${signals.length} locally ingested signals`}
          />
          <div className="event-list">
            {shown.length ? (
              shown.map(s => (
                <button
                  key={s.id}
                  className={`event-item ${selected?.id === s.id && !manual ? 'selected' : ''}`}
                  onClick={() => choose(s)}
                >
                  <div>
                    <span className={`source-type ${s.source_type === 'financial_news' ? 'news' : 'social'}`}>
                      {sourceName(s.source_type)}
                    </span>
                    {pill(s.status, s.status === 'completed' ? 'good' : s.status === 'dead' ? 'bad' : 'neutral')}
                  </div>
                  <b>{s.text}</b>
                  <div className="item-foot">
                    <span><strong>{s.ticker}</strong> · {s.source_name}</span>
                    <span>{asDate(s.first_seen_at)}</span>
                  </div>
                </button>
              ))
            ) : (
              <Empty
                title="No matching records"
                body="Adjust filters or verify ingestion worker has fetched records from Marketaux and Bluesky."
              />
            )}
          </div>
        </section>

        <section className="panel evidence">
          <PanelHeader
            n="02"
            title="SELECTED EVIDENCE"
            sub="Original source metadata, entity match, and timestamp provenance"
          />
          {record ? (
            <>
              <div className="evidence-top">
                {pill(record.source_type === 'financial_news' ? 'Financial News' : 'Public Social Post', 'iris')}
                {pill(record.status, record.status === 'completed' ? 'good' : 'neutral')}
              </div>
              <h2>{record.text}</h2>
              <div className="detail-fields">
                <Field label="TICKER ENTITY">{record.ticker} <span className="muted">· Match: {record.match_method}</span></Field>
                <Field label="SOURCE PROVIDER">{record.source_name}</Field>
                <Field label="PUBLISHED TIME">{record.published_at ? asDate(record.published_at) : 'Unverified / unavailable'}</Field>
                <Field label="FIRST OBSERVED">{asDate(record.first_seen_at)}</Field>
                <Field label="TIMESTAMP QUALITY">{record.timestamp_quality.replaceAll('_', ' ')}</Field>
                <Field label="INFERENCE STATE">{record.status}</Field>
              </div>
              <div className="evidence-actions">
                <SourceLink url={record.source_url} label="Open original article / post link" />
              </div>
              {record.last_error && <ErrorBanner message={`Processing error: ${record.last_error}`} />}
            </>
          ) : manual ? (
            <>
              <div className="evidence-top">
                {pill('Manual User Inference', 'amber')}
              </div>
              <h2>&ldquo;{manualText}&rdquo;</h2>
              <div className="detail-fields">
                <Field label="TICKER ENTITY">{manualTicker} <span className="muted">· Manual entry</span></Field>
                <Field label="SOURCE TYPE">Ad-hoc analyst query</Field>
                <Field label="INSPECTED AT">{asDate(new Date().toISOString())}</Field>
              </div>
              <p className="fineprint" style={{ marginTop: '14px' }}>
                This is a direct manual inference request processed by the AI engine. It is not part of the ingested live stream.
              </p>
            </>
          ) : (
            <Empty
              title="Select an event from the stream"
              body="Click any financial news article or social post on the left to inspect its evidence and model diagnostics."
            />
          )}
        </section>

        <section className="panel diagnostics">
          <PanelHeader
            n="03"
            title="THREE-MODEL DIAGNOSTICS"
            sub="FinBERT &amp; XGBoost output breakdown"
          />
          {inference ? (
            <>
              <div className="model-card">
                <span className="overline color-blue">FINBERT · SENTIMENT ANALYSIS</span>
                <strong className={scoreTone(inference.sentiment.score)}>
                  {inference.sentiment.score >= 0 ? '+' : ''}{decimal(inference.sentiment.score, 3)}
                </strong>
                <span>
                  {sentimentLabel(inference.sentiment.score)} tone on [−1, +1] continuous scale
                </span>
              </div>

              <div className="model-card">
                <span className="overline color-iris">FINBERT · EVENT CLASSIFICATION</span>
                <strong style={{ color: 'var(--lav)' }}>{inference.event.class_name}</strong>
                <span>Classification confidence: <b>{formatPercentFraction(inference.event.confidence, 1)}</b></span>
                
                <button
                  type="button"
                  className="small-button"
                  style={{ alignSelf: 'flex-start', marginTop: '6px', fontSize: '11px', padding: '4px 8px' }}
                  onClick={() => setShowAllProbabilities(!showAllProbabilities)}
                >
                  {showAllProbabilities ? 'Hide probability distribution' : 'Show full 5-class distribution'}
                </button>

                {showAllProbabilities && (
                  <div className="probabilities">
                    {Object.entries(probabilities).sort((a, b) => b[1] - a[1]).map(([label, val]) => (
                      <div className="mini-bar" key={label}>
                        <span>{label}</span>
                        <div><i style={{ width: `${Math.min(100, Math.max(0, val * 100))}%` }} /></div>
                        <b>{decimal(val * 100, 1)}%</b>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="model-card">
                <span className="overline color-amber">XGBOOST · IMPACT SEVERITY ENGINE</span>
                <strong style={{ color: 'var(--amber)' }}>{formatImpact(inference.impact.score)}</strong>
                <span>Estimated reaction magnitude: <b>{formatPercentFraction(inference.impact.predicted_realized_impact)}</b></span>
              </div>

              <div className="notice small">
                <CircleAlert size={15} />
                <div>
                  <strong>Model Limitations &amp; Guardrails:</strong>
                  <div>• Sentiment reflects linguistic tone, not guaranteed stock return direction.</div>
                  <div>• Event confidence measures model classification certainty, not real-world ground truth.</div>
                  <div>• Impact Score (1–10) measures reaction magnitude, not default probability or return %.</div>
                </div>
              </div>

              {inference.warnings?.length ? (
                <div className="notice small">
                  <CircleAlert size={15} />
                  <span>{inference.warnings.join(' · ')}</span>
                </div>
              ) : null}

              {record?.status === 'completed' && (
                <div style={{ marginTop: '14px' }}>
                  <button
                    className="primary-button full"
                    disabled={proposalBusy}
                    onClick={askProposal}
                  >
                    {proposalBusy ? <LoaderCircle className="spin" size={15} /> : <SlidersHorizontal size={15} />}
                    Propose Stress Scenario from Signal
                  </button>
                  <ErrorBanner message={proposalError} />
                  {proposal && (
                    <div className="proposal-box">
                      <span>Proposed: {proposal.proposal.suggested_scenario.name}</span>
                      <small>Severity mapping: {proposal.proposal.severity_source}</small>
                      <button
                        className="secondary-button"
                        style={{ marginTop: '6px' }}
                        onClick={() => onModuleB(proposal)}
                      >
                        Review in Stress Lab <ArrowRight size={14} />
                      </button>
                    </div>
                  )}
                </div>
              )}
            </>
          ) : (
            <Empty
              title="No model output available"
              body="Select a completed record to view FinBERT sentiment, event classification, and XGBoost impact scores."
            />
          )}
        </section>
      </div>

      <div className="expandable-panel">
        <div
          className="expandable-summary"
          onClick={() => setShowManualSection(!showManualSection)}
          role="button"
          tabIndex={0}
        >
          <span>ANALYZE CUSTOM FINANCIAL TEXT (MANUAL AI INFERENCE)</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '11px', color: 'var(--muted)' }}>
              {showManualSection ? 'Collapse tester' : 'Test custom headline / text with FinBERT & XGBoost'}
            </span>
            <ChevronDown size={15} style={{ transform: showManualSection ? 'rotate(180deg)' : 'none', transition: 'transform 0.15s ease' }} />
          </div>
        </div>
        {showManualSection && (
          <div className="expandable-content">
            <p style={{ fontSize: '13px', color: 'var(--sub)', margin: '0 0 14px' }}>
              Test the active AI Risk Engine on arbitrary financial headlines or press releases. This runs live inference without polluting the genuine ingested news database.
            </p>
            <form className="manual-form" onSubmit={submitManual}>
              <label className="form-label">
                TICKER
                <select value={manualTicker} onChange={e => setManualTicker(e.target.value)}>
                  {(tickers.length ? tickers : ['AAPL', 'MSFT', 'NVDA', 'AMZN']).map(t => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </label>
              <label className="form-label manual-text">
                FINANCIAL HEADLINE / TEXT
                <textarea
                  minLength={5}
                  maxLength={10000}
                  rows={3}
                  value={manualText}
                  onChange={e => setManualText(e.target.value)}
                  placeholder="Paste a financial headline, earnings snippet, or regulatory statement to analyze..."
                />
              </label>
              <button
                className="primary-button"
                type="submit"
                disabled={manualBusy || manualText.trim().length < 5}
              >
                {manualBusy ? <LoaderCircle size={15} className="spin" /> : <WandSparkles size={15} />}
                Analyze Text
              </button>
            </form>
            <ErrorBanner message={manualError} />
          </div>
        )}
      </div>
    </div>
  );
}

function ModuleA({
  index, indexError, rebalance, setRebalance
}: {
  index: IndexData | null;
  indexError: string | null;
  rebalance: RebalanceEnvelope | null;
  setRebalance: (r: RebalanceEnvelope) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [run, setRun] = useState<{ metrics: Record<string, Record<string, unknown>>; session_count: number } | null>(null);
  const [backBusy, setBackBusy] = useState(false);
  const [backError, setBackError] = useState('');
  const [showBacktestSection, setShowBacktestSection] = useState(false);

  const reb = rebalance?.recommendation;
  const tickers = index?.tickers ?? [];
  const limitPct = index?.constraints ?? {};
  const proposed = reb?.target_weights ?? {};
  const current = reb?.current_weights ?? index?.baseline_weights ?? {};
  const maximum = Math.max(0.1, ...Object.values(current).map(Number), ...Object.values(proposed).map(Number));

  const onRebalance = async () => {
    setBusy(true);
    setError('');
    try {
      setRebalance(await api.rebalanceSaved(100));
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const onBacktest = async () => {
    setBackBusy(true);
    setBackError('');
    try {
      setRun(await api.backtest('holdout'));
    } catch (e) {
      setBackError(errorMessage(e));
    } finally {
      setBackBusy(false);
    }
  };

  return (
    <div className="page">
      <div className="page-title">
        <div className="eyebrow"><ChartNoAxesCombined size={13} /> Tactical Optimization</div>
        <h1>Tactical Index Rebalancing — Module A</h1>
        <p>Translate analyzed event signals into constrained portfolio allocations across the 15-stock basket. Enforces strict deviation guardrails and turnover limits.</p>
      </div>

      <div className="metrics-grid">
        <Metric
          label="INDEX CONSTITUENTS"
          value={index ? String(tickers.length) : '—'}
          sub="15 large-cap reference stocks"
          kind="iris"
        />
        <Metric
          label="MAX ONE-WAY TURNOVER"
          value={formatPercentFraction(limitPct.max_one_way_turnover)}
          sub="Rebalancing constraint per session"
          kind="amber"
        />
        <Metric
          label="MAX STOCK TILT"
          value={typeof limitPct.max_position_deviation === 'number' ? `±${decimal(limitPct.max_position_deviation * 100, 1)} pp` : '—'}
          sub="Max deviation from benchmark weight"
          kind="blue"
        />
        <Metric
          label="MAX SECTOR TILT"
          value={typeof limitPct.max_sector_deviation === 'number' ? `±${decimal(limitPct.max_sector_deviation * 100, 1)} pp` : '—'}
          sub="Sector exposure guardrail"
          kind="green"
        />
      </div>

      <ErrorBanner message={indexError} />

      <div className="module-a-grid">
        <section className="panel allocation">
          <PanelHeader
            n="01"
            title="ALLOCATION SHIFT ANALYZER"
            sub="Visual weight comparison: Blue = Reference benchmark, Iris = AI Recommended target. Real backend allocations only."
          />
          {tickers.length ? (
            <div className="allocation-table">
              <div className="alloc-head">
                <span>TICKER</span>
                <span>CURRENT / PROPOSED WEIGHT</span>
                <span>CURRENT</span>
                <span>PROPOSED</span>
              </div>
              {tickers.map(t => {
                const curVal = current[t] ?? 0;
                const propVal = proposed[t];
                return (
                  <div className="alloc-row" key={t}>
                    <strong>{t}</strong>
                    <div className="alloc-track">
                      <i style={{ width: `${(curVal / maximum) * 100}%` }} title={`Current: ${formatPercentFraction(curVal)}`} />
                      <em
                        style={{ width: `${propVal === undefined ? 0 : (propVal / maximum) * 100}%` }}
                        title={`Proposed: ${propVal === undefined ? 'Pending' : formatPercentFraction(propVal)}`}
                      />
                    </div>
                    <span>{formatPercentFraction(curVal)}</span>
                    <b className="iris-text">
                      {propVal === undefined ? 'Pending' : formatPercentFraction(propVal)}
                    </b>
                  </div>
                );
              })}
            </div>
          ) : (
            <Empty title="Index not loaded" body="Start FastAPI backend and verify GET /module-a/index endpoint." />
          )}
        </section>

        <section className="panel strategy-panel">
          <PanelHeader
            n="02"
            title="CONSTRAINT GUARDRAILS"
            sub="Bounded optimization guardrails enforced by quadratic programming"
          />
          {[
            ['Max Stock Tilt', formatPercentFraction(limitPct.max_position_deviation)],
            ['Max Sector Deviation', formatPercentFraction(limitPct.max_sector_deviation)],
            ['Max One-Way Turnover', formatPercentFraction(limitPct.max_one_way_turnover)],
            ['Modeled Transaction Cost', '10 bps'],
            ['Trade Execution Status', 'Disabled (Simulated)']
          ].map(([a, b]) => (
            <div className="constraint-row" key={a}>
              <strong>{b}</strong>
              <span>{a}</span>
            </div>
          ))}

          <div className="notice small">
            <LockKeyhole size={15} />
            <span>Simulation only. No automated broker integration or real order placement.</span>
          </div>

          <button
            className="primary-button full"
            disabled={busy || !index}
            onClick={onRebalance}
          >
            {busy ? <LoaderCircle size={15} className="spin" /> : <WandSparkles size={15} />}
            Generate Rebalance Recommendation
          </button>
          <ErrorBanner message={error} />

          {reb && (
            <div className="result-mini">
              <div>
                <span>Signals evaluated</span>
                <b>{rebalance?.saved_signals_considered} saved events</b>
              </div>
              <div>
                <span>One-way turnover</span>
                <b>{formatPercentFraction(reb.one_way_turnover)}</b>
              </div>
              <div>
                <span>Estimated transaction cost</span>
                <b>{formatPercentFraction(reb.estimated_transaction_cost_fraction, 4)}</b>
              </div>
              {reb.warnings?.map((w, i) => (
                <p key={i}>{w}</p>
              ))}
            </div>
          )}
        </section>
      </div>

      {reb?.changes?.some(x => Math.abs(x.change_percentage_points) > 0.0001) && (
        <section className="panel explanation">
          <PanelHeader
            n="03"
            title="WHY THE WEIGHTS CHANGED"
            sub="Attribution of recommended allocation changes to ingested news and social signals"
          />
          {reb.changes.filter(x => Math.abs(x.change_percentage_points) > 0.0001).map(x => (
            <div className="explanation-item" key={x.ticker}>
              <strong>
                {x.ticker}
                <span className={x.change_percentage_points >= 0 ? 'pos' : 'neg'}>
                  {x.change_percentage_points >= 0 ? '+' : ''}{decimal(x.change_percentage_points, 3)} pp
                </span>
              </strong>
              <p>{x.explanation}</p>
            </div>
          ))}
        </section>
      )}

      <div className="expandable-panel">
        <div
          className="expandable-summary"
          onClick={() => setShowBacktestSection(!showBacktestSection)}
          role="button"
          tabIndex={0}
        >
          <span>HISTORICAL STRATEGY COMPARISON (357-SESSION HOLDOUT BACKTEST)</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '11px', color: 'var(--muted)' }}>
              {showBacktestSection ? 'Hide backtest results' : 'Compare 5 strategies across historical holdout'}
            </span>
            <ChevronDown size={15} style={{ transform: showBacktestSection ? 'rotate(180deg)' : 'none', transition: 'transform 0.15s ease' }} />
          </div>
        </div>
        {showBacktestSection && (
          <div className="expandable-content">
            <div className="between" style={{ marginBottom: '14px' }}>
              <div>
                <strong>Evaluate historical performance across 5 strategy benchmarks:</strong>
                <p style={{ margin: '4px 0 0', fontSize: '12px', color: 'var(--sub)' }}>
                  Runs offline backtest on 357 historical trading sessions. Real server calculations only.
                </p>
              </div>
              <button className="secondary-button" onClick={onBacktest} disabled={backBusy}>
                {backBusy ? <LoaderCircle size={14} className="spin" /> : <Play size={14} />}
                Run Holdout Backtest
              </button>
            </div>
            <ErrorBanner message={backError} />
            {run ? (
              <div className="backtest-grid">
                {Object.entries(run.metrics).map(([name, met]) => (
                  <div className="backtest-metric" key={name}>
                    <b>{name.replaceAll('_', ' ')}</b>
                    {Object.entries(met).slice(0, 6).map(([k, v]) => (
                      <div key={k}>
                        <span>{k.replaceAll('_', ' ')}</span>
                        <strong>{typeof v === 'number' ? decimal(v, 4) : String(v)}</strong>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            ) : (
              <Empty
                title="Backtest not executed in this session"
                body="Click 'Run Holdout Backtest' to load authentic server-computed metrics across holdout data. No synthetic returns are displayed."
              />
            )}
            <p className="fineprint" style={{ marginTop: '16px' }}>
              Academic &amp; Research Disclaimer: Holdout backtest uses historical closing prices with idealized same-session fill assumptions. Past simulation performance is not indicative of live investment returns and does not constitute a trading recommendation.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function ModuleB({
  portfolio, library, proposal, clearProposal
}: {
  portfolio: Portfolio | null;
  library: ScenarioLibrary | null;
  proposal: Proposal | null;
  clearProposal: () => void;
}) {
  const [family, setFamily] = useState('');
  const [tier, setTier] = useState('moderate');
  const [issuer, setIssuer] = useState('');

  const [shock, setShock] = useState<Scenario | null>(null);
  const [reviewed, setReviewed] = useState(false);

  const [result, setResult] = useState<StressResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const [activeTab, setTab] = useState<'stress' | 'whatif' | 'reverse'>('stress');
  const [positionId, setPositionId] = useState('');
  const [scale, setScale] = useState('0.75');
  const [lossBudget, setLossBudget] = useState('100000');
  const [whatIf, setWhatIf] = useState<{ before: StressResult; after: StressResult; net_effect_change_usd: number; warning: string } | null>(null);
  const [reverse, setReverse] = useState<{ found_breach: boolean; first_breach_on_grid?: { multiplier: number; loss_usd: number } | null; method: string; warning: string; truncated_at_multiplier?: number | null } | null>(null);

  const families = Object.keys(library?.scenario_library ?? {});
  const templates = library?.scenario_library[family] ?? {};
  const selected = shock ?? (proposal?.proposal.suggested_scenario ?? templates[tier] ?? null);

  useEffect(() => {
    if (families.length && !family) setFamily(families[0]);
  }, [families, family]);

  const changeFamily = (v: string) => {
    setFamily(v);
    setShock(null);
    setReviewed(false);
    setResult(null);
    setWhatIf(null);
    setReverse(null);
    if (proposal) clearProposal();
  };

  const changeTier = (v: string) => {
    setTier(v);
    setShock(null);
    setReviewed(false);
    setResult(null);
    if (proposal) clearProposal();
  };

  const setField = (name: keyof Scenario['shocks'], n: string) => {
    if (!selected) return;
    setShock({ ...selected, shocks: { ...selected.shocks, [name]: Number(n) } });
    setReviewed(false);
    setResult(null);
  };

  const resolveScenario = () => scenarioForExecution(selected, issuer) as unknown as Scenario;

  const run = async () => {
    setBusy(true);
    setError('');
    setResult(null);
    try {
      if (!reviewed) throw Error('Please review the scenario assumptions and acknowledge them before execution.');
      setResult(await api.stress(resolveScenario()));
      setTab('stress');
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const runWhatIf = async () => {
    setBusy(true);
    setError('');
    try {
      if (!reviewed) throw Error('Please acknowledge assumptions first.');
      if (!positionId) throw Error('Please select a synthetic portfolio position.');
      const n = Number(scale);
      if (!Number.isFinite(n) || n < 0 || n > 2) throw Error('Position scaling must be between 0.0 and 2.0.');
      setWhatIf(await api.whatIf(resolveScenario(), [{ position_id: positionId, scale: n }]));
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const runReverse = async () => {
    setBusy(true);
    setError('');
    try {
      if (!reviewed) throw Error('Please acknowledge assumptions first.');
      const n = Number(lossBudget);
      if (!Number.isFinite(n) || n <= 0) throw Error('Loss budget must be a positive USD amount.');
      setReverse(await api.reverseStress(resolveScenario(), n));
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const stats = portfolio?.summary;
  const counts = stats?.counts;

  return (
    <div className="page">
      <div className="page-title">
        <div className="eyebrow"><ShieldAlert size={13} /> Institutional Risk Laboratory</div>
        <h1>Portfolio Stress Laboratory — Module B</h1>
        <p>Perform multi-asset sensitivity analysis across credit loans, bonds, interest-rate swaps, and options. Preserves clear separation between instantaneous mark-to-market and 12-month projected loan credit loss horizons.</p>
      </div>

      <div className="metrics-grid">
        <Metric
          label="SYNTHETIC POSITIONS"
          value={stats ? String(stats.position_count) : '—'}
          sub="30 multi-asset synthetic holdings"
          kind="iris"
        />
        <Metric
          label="LOAN EAD (CREDIT)"
          value={fmtMoney(stats?.loan_ead_usd)}
          sub={`${counts?.loan ?? '—'} corporate loans · Exposure at Default`}
          kind="blue"
        />
        <Metric
          label="BOND MARKET VALUE"
          value={fmtMoney(stats?.bond_market_value_usd)}
          sub={`${counts?.bond ?? '—'} fixed-income bonds · Revaluation base`}
          kind="green"
        />
        <Metric
          label="DERIVATIVE NOTIONAL"
          value={fmtMoney(stats?.derivative_reference_notional_usd)}
          sub="Reference notional volume ≠ NAV"
          kind="amber"
        />
      </div>

      {proposal && (
        <div className="proposal-alert">
          <Sparkles size={18} />
          <div>
            <b>Scenario Proposed from Saved Event Signal</b>
            <p>{proposal.proposal.justification} Shocks are illustrative hypothetical parameters, not calibrated default probabilities or interest-rate forecasts.</p>
          </div>
          <button
            onClick={() => { clearProposal(); setShock(null); setReviewed(false); setResult(null); }}
            className="icon-btn"
            aria-label="Dismiss proposal"
            title="Dismiss proposal"
          >
            <X size={15} />
          </button>
        </div>
      )}

      <div className="stress-grid">
        <section className="panel scenario-panel">
          <PanelHeader
            n="01"
            title="SCENARIO DESIGN"
            sub="Reviewable illustrative financial shocks"
            accent="amber"
          />
          <div className="form-stack">
            <label className="form-label">
              EVENT FAMILY
              <select
                value={proposal?.proposal.suggested_scenario.event_class ?? family}
                disabled={!!proposal}
                onChange={e => changeFamily(e.target.value)}
              >
                {families.map(x => <option key={x} value={x}>{x}</option>)}
              </select>
            </label>

            <label className="form-label">
              SCENARIO INTENSITY
              <select
                value={proposal?.proposal.suggested_scenario.severity ?? tier}
                disabled={!!proposal}
                onChange={e => changeTier(e.target.value)}
              >
                {Object.keys(templates).map(x => <option key={x} value={x}>{x}</option>)}
              </select>
            </label>

            <label className="form-label">
              ISSUER OVERRIDE (for issuer-specific templates)
              <select value={issuer} onChange={e => { setIssuer(e.target.value); setReviewed(false); }}>
                <option value="">Select issuer ticker when required</option>
                {['AAPL', 'MSFT', 'NVDA', 'QCOM', 'GOOGL', 'CMCSA', 'AMZN', 'TSLA', 'BKNG', 'SBUX', 'COST', 'PEP', 'GILD', 'AMGN', 'CSX'].map(x => (
                  <option key={x} value={x}>{x}</option>
                ))}
              </select>
            </label>

            {selected ? (
              <div className="shock-fields">
                {([
                  ['rate_bps', 'Rates (bps)'],
                  ['spread_bps', 'Spread (bps)'],
                  ['equity_pct', 'Equity (%)'],
                  ['volatility_points', 'Vol Points'],
                  ['pd_multiplier', 'PD Multiplier']
                ] as const).map(([k, label]) => (
                  <label className="form-label" key={k}>
                    {label}
                    <input
                      type="number"
                      step="any"
                      value={selected.shocks[k]}
                      onChange={e => setField(k, e.target.value)}
                    />
                  </label>
                ))}
              </div>
            ) : (
              <Empty title="No scenario library" body="GET /module-b/scenarios must return scenario templates." />
            )}

            <div className="notice small">
              <CircleAlert size={15} />
              <span>Impact scores and severity tiers are illustrative parameters, not calibrated forecasts of bond spreads or PD.</span>
            </div>

            <label className="checkline">
              <input
                type="checkbox"
                checked={reviewed}
                onChange={e => setReviewed(e.target.checked)}
              />
              <span>I reviewed these hypothetical assumptions and acknowledge that no live trades will be executed.</span>
            </label>

            <button
              className="primary-button full"
              disabled={!selected || busy || !reviewed}
              onClick={run}
            >
              {busy ? <LoaderCircle size={15} className="spin" /> : <ShieldCheck size={15} />}
              Run Reviewed Stress Test
            </button>
            <ErrorBanner message={error} />
          </div>
        </section>

        <section className="panel mechanisms">
          <PanelHeader
            n="02"
            title="FOUR LOSS MECHANISMS"
            sub="Dedicated valuation engines per asset class"
          />
          {[
            ['LOANS', `${counts?.loan ?? '—'} positions`, '12-Month Expected Credit Loss (ECL)', 'PD × LGD × EAD credit deterioration', 'blue'],
            ['BONDS', `${counts?.bond ?? '—'} positions`, 'Instantaneous Price Revaluation', 'Duration & convexity yield shift', 'iris'],
            ['SWAPS', `${counts?.swap ?? '—'} positions`, 'Signed Interest-Rate Sensitivity', 'DV01 × benchmark rate shock', 'amber'],
            ['OPTIONS', `${counts?.option ?? '—'} positions`, 'Option Mark-to-Market Revaluation', 'European Black–Scholes pricing formula', 'rose']
          ].map(x => (
            <div key={x[0]} className="mechanism">
              <span className={`mech-strip ${x[4]}`} />
              <div>
                <b className={`color-${x[4]}`}>{x[0]}</b>
                <h3>{x[2]}</h3>
                <p>{x[3]}</p>
              </div>
              <span className="mechanism-count">{x[1]}</span>
            </div>
          ))}
          <div className="notice small">
            <CircleAlert size={15} />
            <span>Measurement Horizon Distinction: Projected 12-month loan credit losses operate on a forward horizon, while Bonds, Swaps, and Options are revalued instantaneously.</span>
          </div>
        </section>

        <section className="panel stress-result">
          <PanelHeader
            n="03"
            title="STRESS RESULTS"
            sub={result ? result.scenario.name : 'Awaiting reviewed scenario execution'}
            accent="amber"
          />
          {result ? (
            <>
              <div className="outcome-top">
                <span>Instantaneous MTM P&amp;L (Bonds, Swaps, Options)</span>
                <strong className={result.totals.instantaneous_mtm_pnl_usd >= 0 ? 'pos' : 'neg'}>
                  {fmtMoney(result.totals.instantaneous_mtm_pnl_usd)}
                </strong>
              </div>
              <div className="outcome-top">
                <span>Incremental 12-Month Loan ECL (Credit Deterioration)</span>
                <strong style={{ color: 'var(--amber)' }}>
                  {fmtMoney(result.totals.incremental_12m_expected_credit_loss_usd)}
                </strong>
              </div>
              <div className="warning-band">
                Mixed-Horizon Illustrative Net: {fmtMoney(result.totals.illustrative_mixed_horizon_net_effect_usd)}
                <div style={{ fontSize: '11px', marginTop: '4px', opacity: 0.9 }}>
                  (Illustrative only — not NAV or VaR. Combines instantaneous MTM with 12-month expected credit loss.)
                </div>
              </div>

              <div className="split-title">INDICATIVE CONTRIBUTION BY INSTRUMENT</div>
              {Object.entries(result.loss_contribution_by_type_usd).map(([name, value]) => (
                <div className="result-line" key={name}>
                  <span>{name.toUpperCase()}</span>
                  <b>{fmtMoney(value)}</b>
                </div>
              ))}
              {result.override_coverage && !result.override_coverage.all_overrides_matched && (
                <div className="notice small">
                  <span>Unmatched issuer/sector shocks reported by backend. Review scenario setup before formal interpretation.</span>
                </div>
              )}
            </>
          ) : (
            <Empty
              title="No stress result yet"
              body="Select a scenario, review the assumptions checkbox, and click 'Run Reviewed Stress Test'."
            />
          )}
        </section>
      </div>

      <section className="panel additional-stress">
        <div className="tabs">
          <button className={activeTab === 'stress' ? 'active' : ''} onClick={() => setTab('stress')}>
            Horizon &amp; Methodology Context
          </button>
          <button className={activeTab === 'whatif' ? 'active' : ''} onClick={() => setTab('whatif')}>
            What-If Exposure Scaling
          </button>
          <button className={activeTab === 'reverse' ? 'active' : ''} onClick={() => setTab('reverse')}>
            Reverse Stress Testing
          </button>
        </div>

        {activeTab === 'stress' ? (
          <div className="tab-content">
            <b>Financial Interpretation &amp; Horizon Guidance:</b>
            <p>
              Instantaneous marked-to-market profit/loss and projected 12-month incremental expected credit losses utilize fundamentally distinct time horizons. No combined value should be interpreted as regulatory capital, VaR, or guaranteed economic loss.
            </p>
            <span className="fineprint">
              Scenarios are hypothetical illustrative simulations. All portfolio holdings are synthetic.
            </span>
          </div>
        ) : activeTab === 'whatif' ? (
          <div className="tab-content">
            <p>
              Scale the exposure of a single synthetic holding (0.0x to 2.0x) to evaluate hedging sensitivity under the active shock scenario.
            </p>
            <div className="tab-controls">
              <label className="form-label">
                SYNTHETIC POSITION
                <select value={positionId} onChange={e => setPositionId(e.target.value)}>
                  <option value="">Select holding</option>
                  {portfolio?.positions.map(p => (
                    <option key={p.id} value={p.id}>{p.id} · {p.name}</option>
                  ))}
                </select>
              </label>
              <label className="form-label">
                EXPOSURE MULTIPLIER
                <input
                  type="number"
                  min="0"
                  max="2"
                  step="0.05"
                  value={scale}
                  onChange={e => setScale(e.target.value)}
                />
              </label>
              <button className="secondary-button" disabled={busy || !reviewed} onClick={runWhatIf}>
                Run What-If Analysis
              </button>
            </div>
            {whatIf && (
              <div className="compact-results">
                <span>Before mixed-horizon net: <b>{fmtMoney(whatIf.before.totals.illustrative_mixed_horizon_net_effect_usd)}</b></span>
                <span>After mixed-horizon net: <b>{fmtMoney(whatIf.after.totals.illustrative_mixed_horizon_net_effect_usd)}</b></span>
                <b style={{ color: 'var(--lav)' }}>Net Change: {fmtMoney(whatIf.net_effect_change_usd)}</b>
                <small>{whatIf.warning}</small>
              </div>
            )}
          </div>
        ) : (
          <div className="tab-content">
            <p>
              Determine the lowest severity multiplier that breaches a user-defined loss budget on the evaluated stress grid.
            </p>
            <div className="tab-controls">
              <label className="form-label">
                LOSS BUDGET (USD)
                <input
                  type="number"
                  min="1"
                  step="1000"
                  value={lossBudget}
                  onChange={e => setLossBudget(e.target.value)}
                />
              </label>
              <button className="secondary-button" disabled={busy || !reviewed} onClick={runReverse}>
                Run Reverse Stress Test
              </button>
            </div>
            {reverse && (
              <div className="compact-results">
                <b>{reverse.found_breach ? 'Sampled Threshold Breach Identified' : 'No Breach within Feasible Grid'}</b>
                <span>
                  {reverse.first_breach_on_grid
                    ? `Breach Multiplier: ${decimal(reverse.first_breach_on_grid.multiplier, 2)}x · Indicative Loss: ${fmtMoney(reverse.first_breach_on_grid.loss_usd)}`
                    : 'Evaluated multipliers up to maximum grid factor did not breach chosen loss budget.'}
                </span>
                <small>{reverse.method}</small>
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>('overview');
  const [summary, setSummary] = useState<Load<Summary>>(initial());
  const [signalPage, setSignalPage] = useState<Load<SignalsPage>>(initial());
  const [health, setHealth] = useState('unknown');
  const [ingestion, setIngestion] = useState('unknown');

  const [idx, setIdx] = useState<Load<IndexData>>(initial());
  const [portfolio, setPortfolio] = useState<Load<Portfolio>>(initial());
  const [library, setLibrary] = useState<Load<ScenarioLibrary>>(initial());

  const [selected, setSelected] = useState<Signal | null>(null);
  const [proposal, setProposal] = useState<Proposal | null>(null);
  const [rebalance, setRebalance] = useState<RebalanceEnvelope | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    const [a, b, c, d, e, f, g] = await Promise.allSettled([
      api.summary(),
      api.signals({ limit: 100 }),
      api.health(),
      api.ingestionHealth(),
      api.index(),
      api.portfolio(),
      api.scenarios()
    ]);

    setSummary(a.status === 'fulfilled' ? { value: a.value, error: null, pending: false } : { value: null, error: errorMessage(a.reason), pending: false });
    setSignalPage(b.status === 'fulfilled' ? { value: b.value, error: null, pending: false } : { value: null, error: errorMessage(b.reason), pending: false });
    setHealth(c.status === 'fulfilled' ? c.value.status : 'unavailable');
    setIngestion(d.status === 'fulfilled' ? d.value.status : 'unavailable');
    setIdx(e.status === 'fulfilled' ? { value: e.value, error: null, pending: false } : { value: null, error: errorMessage(e.reason), pending: false });
    setPortfolio(f.status === 'fulfilled' ? { value: f.value, error: null, pending: false } : { value: null, error: errorMessage(f.reason), pending: false });
    setLibrary(g.status === 'fulfilled' ? { value: g.value, error: null, pending: false } : { value: null, error: errorMessage(g.reason), pending: false });
    setLastUpdated(new Date().toISOString());
    setRefreshing(false);
  }, []);

  useEffect(() => {
    void refresh();
    const timer = setInterval(() => void refresh(), 60_000);
    return () => clearInterval(timer);
  }, [refresh]);

  const signals = signalPage.value?.items ?? [];
  const choose = (s: Signal) => {
    setSelected(s);
    setPage('events');
  };
  const goProposal = (p: Proposal) => {
    setProposal(p);
    setPage('moduleB');
  };
  const hasProblem = summary.error || signalPage.error;

  return (
    <Shell
      page={page}
      onPage={setPage}
      summary={summary.value}
      health={health}
      ingestion={ingestion}
      loading={refreshing}
      refresh={() => void refresh()}
      selectedTicker={selected?.ticker ?? ''}
    >
      <div className="dataset-strip">
        <span>
          <Dot tone={hasProblem ? 'amber' : 'green'} />
          {hasProblem ? 'Backend connection incomplete' : 'Live backend responses loaded'}
        </span>
        <span>
          <Clock3 size={13} />
          {lastUpdated ? `Refreshed ${asDate(lastUpdated)}` : 'Awaiting first response'}
        </span>
        <span>
          <LockKeyhole size={13} />
          Simulation &amp; research only
        </span>
      </div>
      <ErrorBanner
        message={hasProblem ? `Backend read failed: ${hasProblem}. Start FastAPI on 127.0.0.1:8000; dev Vite proxies /api.` : null}
      />
      {page === 'overview' ? (
        <Overview
          summary={summary.value}
          signals={signals}
          goSignal={choose}
          goPage={setPage}
          health={health}
          ingestion={ingestion}
        />
      ) : page === 'events' ? (
        <Events
          signals={signals}
          selected={selected}
          onSelect={setSelected}
          onModuleB={goProposal}
          tickers={idx.value?.tickers ?? []}
          analyzeManual={api.analyze}
        />
      ) : page === 'moduleA' ? (
        <ModuleA
          index={idx.value}
          indexError={idx.error}
          rebalance={rebalance}
          setRebalance={setRebalance}
        />
      ) : (
        <ModuleB
          portfolio={portfolio.value}
          library={library.value}
          proposal={proposal}
          clearProposal={() => setProposal(null)}
        />
      )}
    </Shell>
  );
}
