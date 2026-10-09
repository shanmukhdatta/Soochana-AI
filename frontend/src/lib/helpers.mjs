/** Pure, auditable formatting and financial-display helpers shared with Node tests. */
export function countStatus(summary, status) {
  const rows = Array.isArray(summary?.status_counts) ? summary.status_counts : [];
  return Number(rows.find(row => row?.status === status)?.count ?? 0);
}
export function countSource(summary, source) {
  const rows = Array.isArray(summary?.source_counts) ? summary.source_counts : [];
  return Number(rows.find(row => row?.source_type === source)?.count ?? 0);
}
export function safeNumber(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}
export function sentimentLabel(score) {
  if (safeNumber(score) === null) return 'Not analyzed';
  if (score > 0.1) return 'Positive';
  if (score < -0.1) return 'Negative';
  return 'Neutral';
}
export function formatPercentFraction(value, digits = 2) {
  const n = safeNumber(value);
  return n === null ? '—' : `${(n * 100).toFixed(digits)}%`;
}
export function fmtMoney(value) {
  const n = safeNumber(value);
  if (n === null) return '—';
  const abs = Math.abs(n);
  if (abs >= 1_000_000) return `${n < 0 ? '−' : ''}$${(abs / 1_000_000).toFixed(2)}m`;
  if (abs >= 1000) return `${n < 0 ? '−' : ''}$${(abs / 1000).toFixed(1)}k`;
  return `${n < 0 ? '−' : ''}$${abs.toFixed(2)}`;
}
export function safeSourceUrl(raw) {
  try { const u = new URL(raw); return u.protocol === 'https:' || u.protocol === 'http:' ? u.href : null; }
  catch { return null; }
}
export function scenarioForExecution(scenario, issuerTicker) {
  if (!scenario || typeof scenario !== 'object') throw Error('Select a scenario first');
  const copy = structuredClone(scenario);
  if (Object.prototype.hasOwnProperty.call(copy.issuer_shocks ?? {}, '__ISSUER__')) {
    const ticker = (issuerTicker ?? '').toUpperCase().trim();
    if (!/^[A-Z][A-Z0-9.-]{0,9}$/.test(ticker)) throw Error('Issuer-specific scenario requires a valid issuer ticker.');
    const extra = copy.issuer_shocks.__ISSUER__;
    delete copy.issuer_shocks.__ISSUER__;
    if (Object.prototype.hasOwnProperty.call(copy.issuer_shocks, ticker)) throw Error('Issuer already has a scenario shock override.');
    copy.issuer_shocks[ticker] = extra;
  }
  return copy;
}
export function eventCategoryCounts(signals) {
  const map = {};
  for (const item of signals ?? []) {
    const cat = item?.inference?.event?.class_name;
    if (typeof cat === 'string' && cat) map[cat] = (map[cat] ?? 0) + 1;
  }
  return map;
}
export function recentSignalMeans(signals) {
  const ready = (signals ?? []).filter(x => x?.status === 'completed' && x?.inference);
  const sentiments = ready.map(x => safeNumber(x.inference?.sentiment?.score)).filter(x => x !== null);
  const severities = ready.map(x => safeNumber(x.inference?.impact?.score)).filter(x => x !== null);
  const average = xs => xs.length ? xs.reduce((a,b)=>a+b,0) / xs.length : null;
  return { sentiment: average(sentiments), impact: average(severities), count: ready.length };
}
