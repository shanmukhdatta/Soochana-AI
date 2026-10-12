import { useState, useEffect } from 'react';
import {
  ShieldCheck, Sparkles, Plus, Trash2, RefreshCw,
  TrendingDown, AlertCircle, Compass
} from 'lucide-react';
import { api } from '../api';

interface PositionInput {
  ticker: string;
  weight: number;
}

interface Preset {
  id: string;
  name: string;
  description: string;
  total_capital: number;
  currency: string;
  positions: PositionInput[];
}

export function CustomPortfolioStudio({ onAskBujji }: { onAskBujji?: (query: string) => void }) {
  // Input states
  const [positions, setPositions] = useState<PositionInput[]>([
    { ticker: 'NVDA', weight: 0.35 },
    { ticker: 'AAPL', weight: 0.25 },
    { ticker: 'MSFT', weight: 0.20 },
    { ticker: 'AMZN', weight: 0.10 },
    { ticker: 'JPM', weight: 0.10 },
  ]);
  const [totalCapital, setTotalCapital] = useState<number>(10000);
  const [currency, setCurrency] = useState<'USD' | 'INR'>('USD');
  const [presets, setPresets] = useState<Preset[]>([]);

  // Execution states
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<any | null>(null);

  // Load presets on mount
  useEffect(() => {
    let isMounted = true;
    api.customPortfolioPresets()
      .then((data) => {
        if (isMounted && data?.presets) {
          setPresets(data.presets);
        }
      })
      .catch(() => {
        // Fallback default presets if offline
        setPresets([
          {
            id: 'mag_seven',
            name: 'Big Tech & AI Leaders',
            description: 'High growth tech giants powering generative AI.',
            total_capital: 10000,
            currency: 'USD',
            positions: [
              { ticker: 'NVDA', weight: 0.30 },
              { ticker: 'MSFT', weight: 0.25 },
              { ticker: 'AAPL', weight: 0.20 },
              { ticker: 'AMZN', weight: 0.15 },
              { ticker: 'GOOGL', weight: 0.10 },
            ]
          },
          {
            id: 'balanced_core',
            name: 'All-Weather Balanced',
            description: 'Tech, financials, consumer essentials, and healthcare.',
            total_capital: 25000,
            currency: 'USD',
            positions: [
              { ticker: 'MSFT', weight: 0.25 },
              { ticker: 'JPM', weight: 0.25 },
              { ticker: 'PG', weight: 0.20 },
              { ticker: 'JNJ', weight: 0.15 },
              { ticker: 'COST', weight: 0.15 },
            ]
          }
        ]);
      });
    return () => { isMounted = false; };
  }, []);

  const totalWeight = positions.reduce((acc, p) => acc + (Number(p.weight) || 0), 0);
  const totalWeightPct = Math.round(totalWeight * 100);

  const handleWeightChange = (index: number, val: number) => {
    const updated = [...positions];
    updated[index].weight = Math.max(0, Math.min(1, val));
    setPositions(updated);
  };

  const handleTickerChange = (index: number, val: string) => {
    const updated = [...positions];
    updated[index].ticker = val.toUpperCase().trim();
    setPositions(updated);
  };

  const handleAddPosition = () => {
    if (positions.length >= 15) return;
    setPositions([...positions, { ticker: '', weight: 0 }]);
  };

  const handleRemovePosition = (index: number) => {
    if (positions.length <= 2) return;
    const updated = positions.filter((_, i) => i !== index);
    setPositions(updated);
  };

  const handleNormalize = () => {
    if (totalWeight <= 0) {
      const eq = 1 / positions.length;
      setPositions(positions.map((p) => ({ ...p, weight: Number(eq.toFixed(3)) })));
      return;
    }
    setPositions(positions.map((p) => ({
      ...p,
      weight: Number((p.weight / totalWeight).toFixed(3))
    })));
  };

  const handleLoadPreset = (preset: Preset) => {
    setPositions(preset.positions);
    setTotalCapital(preset.total_capital);
    setCurrency((preset.currency as 'USD' | 'INR') || 'USD');
    setResults(null);
    setError(null);
  };

  const handleRunAnalysis = async () => {
    // Validate
    const validPositions = positions.filter((p) => p.ticker.length > 0 && p.weight > 0);
    if (validPositions.length < 2) {
      setError('Please add at least 2 stock tickers with positive weights.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const res = await api.analyzeCustomPortfolio({
        positions: validPositions,
        total_capital: totalCapital,
        currency,
      });
      setResults(res);
    } catch (err: any) {
      setError(err?.message || 'Failed to analyze portfolio. Please verify ticker symbols.');
    } finally {
      setLoading(false);
    }
  };

  const curSymbol = currency === 'INR' ? '₹' : '$';

  return (
    <div className="custom-portfolio-studio" style={{ padding: '1rem 0' }}>
      {/* Studio Header */}
      <div className="card hero-custom-studio" style={{
        background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: '16px',
        padding: '1.75rem',
        marginBottom: '1.5rem',
        boxShadow: '0 20px 40px -15px rgba(0, 0, 0, 0.5)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#38bdf8', fontSize: '0.875rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              <ShieldCheck size={18} /> Personal AI Risk Shield
            </div>
            <h1 style={{ fontSize: '1.85rem', fontWeight: 700, color: '#f8fafc', margin: '0.5rem 0' }}>
              Custom Portfolio Studio
            </h1>
            <p style={{ color: '#94a3b8', fontSize: '0.95rem', maxWidth: '720px', lineHeight: 1.5, margin: 0 }}>
              Input your personal stock holdings and capital. Our 3-stage FinBERT + XGBoost engine evaluates live market news,
              filters out sensational panic chatter, optimizes your risk-adjusted weights, and protects your savings.
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'rgba(15, 23, 42, 0.6)', padding: '6px', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.08)' }}>
            <span style={{ fontSize: '0.8rem', color: '#94a3b8', padding: '0 8px' }}>Currency:</span>
            <button
              onClick={() => setCurrency('USD')}
              className={`pill ${currency === 'USD' ? 'blue' : 'neutral'}`}
              style={{ cursor: 'pointer', padding: '4px 12px' }}
            >
              $ USD
            </button>
            <button
              onClick={() => setCurrency('INR')}
              className={`pill ${currency === 'INR' ? 'blue' : 'neutral'}`}
              style={{ cursor: 'pointer', padding: '4px 12px' }}
            >
              ₹ INR
            </button>
          </div>
        </div>

        {/* Quick Presets */}
        {presets.length > 0 && (
          <div style={{ marginTop: '1.25rem', paddingTop: '1.25rem', borderTop: '1px solid rgba(255, 255, 255, 0.08)', display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.8rem', color: '#64748b', fontWeight: 600 }}>QUICK PRESETS:</span>
            {presets.map((pr) => (
              <button
                key={pr.id}
                onClick={() => handleLoadPreset(pr)}
                className="button-preset"
                style={{
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  color: '#e2e8f0',
                  borderRadius: '20px',
                  padding: '5px 14px',
                  fontSize: '0.8rem',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  transition: 'all 0.2s'
                }}
              >
                <Sparkles size={12} color="#38bdf8" /> {pr.name}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Main Grid: Input Builder vs Analysis Action */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>
        {/* Left: Interactive Portfolio Builder */}
        <div className="card" style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px', padding: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#f1f5f9', margin: 0 }}>
                1. Your Stock Holdings
              </h3>
              <span style={{ fontSize: '0.8rem', color: '#64748b' }}>Enter tickers and weight percentage</span>
            </div>
            <button
              onClick={handleAddPosition}
              disabled={positions.length >= 15}
              style={{
                background: '#2563eb',
                color: '#fff',
                border: 'none',
                borderRadius: '8px',
                padding: '6px 12px',
                fontSize: '0.8rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px'
              }}
            >
              <Plus size={14} /> Add Stock
            </button>
          </div>

          {/* Investment Amount Input */}
          <div style={{ background: 'rgba(255, 255, 255, 0.02)', border: '1px solid #1e293b', borderRadius: '10px', padding: '10px 14px', marginBottom: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Total Portfolio Value:</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ color: '#38bdf8', fontWeight: 600 }}>{curSymbol}</span>
              <input
                type="number"
                min="100"
                step="500"
                value={totalCapital}
                onChange={(e) => setTotalCapital(Number(e.target.value) || 1000)}
                style={{
                  background: '#1e293b',
                  border: '1px solid #334155',
                  color: '#fff',
                  borderRadius: '6px',
                  padding: '4px 8px',
                  width: '120px',
                  fontWeight: 600,
                  fontSize: '0.9rem'
                }}
              />
            </div>
          </div>

          {/* Ticker & Weight List */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', maxHeight: '380px', overflowY: 'auto', paddingRight: '4px' }}>
            {positions.map((pos, idx) => {
              const cashVal = Math.round(pos.weight * totalCapital);
              return (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    background: '#1e293b',
                    padding: '8px 12px',
                    borderRadius: '10px',
                    border: '1px solid rgba(255, 255, 255, 0.05)'
                  }}
                >
                  <input
                    type="text"
                    placeholder="TICKER"
                    value={pos.ticker}
                    onChange={(e) => handleTickerChange(idx, e.target.value)}
                    style={{
                      width: '75px',
                      background: '#0f172a',
                      border: '1px solid #334155',
                      color: '#38bdf8',
                      fontWeight: 700,
                      borderRadius: '6px',
                      padding: '6px 8px',
                      fontSize: '0.85rem',
                      textTransform: 'uppercase'
                    }}
                  />
                  <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={pos.weight}
                      onChange={(e) => handleWeightChange(idx, parseFloat(e.target.value))}
                      style={{ flex: 1, accentColor: '#38bdf8', cursor: 'pointer' }}
                    />
                    <span style={{ minWidth: '45px', textAlign: 'right', fontSize: '0.85rem', fontWeight: 600, color: '#f8fafc' }}>
                      {Math.round(pos.weight * 100)}%
                    </span>
                  </div>
                  <span style={{ fontSize: '0.75rem', color: '#64748b', minWidth: '70px', textAlign: 'right' }}>
                    {curSymbol}{cashVal.toLocaleString()}
                  </span>
                  <button
                    onClick={() => handleRemovePosition(idx)}
                    disabled={positions.length <= 2}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: positions.length <= 2 ? '#334155' : '#ef4444',
                      cursor: positions.length <= 2 ? 'not-allowed' : 'pointer',
                      padding: '4px'
                    }}
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              );
            })}
          </div>

          {/* Allocation Bar & Auto-balance */}
          <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid #1e293b' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                Total Allocation: <strong style={{ color: totalWeightPct === 100 ? '#10b981' : '#f59e0b' }}>{totalWeightPct}%</strong>
              </span>
              {totalWeightPct !== 100 && (
                <button
                  onClick={handleNormalize}
                  style={{
                    background: 'rgba(56, 189, 248, 0.1)',
                    border: '1px solid rgba(56, 189, 248, 0.3)',
                    color: '#38bdf8',
                    borderRadius: '6px',
                    padding: '3px 8px',
                    fontSize: '0.75rem',
                    cursor: 'pointer'
                  }}
                >
                  Auto-Balance to 100%
                </button>
              )}
            </div>
            <div style={{ height: '6px', width: '100%', background: '#334155', borderRadius: '3px', overflow: 'hidden' }}>
              <div
                style={{
                  height: '100%',
                  width: `${Math.min(100, totalWeightPct)}%`,
                  background: totalWeightPct === 100 ? '#10b981' : totalWeightPct > 100 ? '#ef4444' : '#38bdf8',
                  transition: 'width 0.3s'
                }}
              />
            </div>
          </div>
        </div>

        {/* Right: Trigger Box & Executive Readiness */}
        <div className="card" style={{
          background: 'linear-gradient(180deg, #0f172a 0%, #1e293b 100%)',
          border: '1px solid #1e293b',
          borderRadius: '16px',
          padding: '1.5rem',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between'
        }}>
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#f1f5f9', margin: '0 0 0.5rem 0' }}>
              2. Real-Time AI Analysis Pipeline
            </h3>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', lineHeight: 1.6 }}>
              When you click the button below, the system executes:
            </p>
            <ul style={{ color: '#cbd5e1', fontSize: '0.85rem', paddingLeft: '1.25rem', lineHeight: 1.8 }}>
              <li><strong>Live News Ingestion:</strong> Queries current headlines for your selected stocks.</li>
              <li><strong>FinBERT NLP Scoring:</strong> Classifies events and predicts continuous sentiment [-1, +1].</li>
              <li><strong>XGBoost Volatility Engine:</strong> Forecasts market impact magnitude deciles.</li>
              <li><strong>Fake News / Panic Filter:</strong> Labels emotional chatter vs. genuine balance sheet threats.</li>
              <li><strong>Convex Quadratic Optimizer:</strong> Minimizes tracking risk while tilting into AI signals.</li>
            </ul>
          </div>

          <div style={{ marginTop: '1.5rem' }}>
            {error && (
              <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid #ef4444', borderRadius: '8px', padding: '10px 12px', color: '#fca5a5', fontSize: '0.85rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <AlertCircle size={16} /> {error}
              </div>
            )}

            <button
              onClick={handleRunAnalysis}
              disabled={loading}
              style={{
                width: '100%',
                background: 'linear-gradient(135deg, #0284c7 0%, #2563eb 100%)',
                color: '#fff',
                border: 'none',
                borderRadius: '12px',
                padding: '14px',
                fontSize: '1rem',
                fontWeight: 700,
                cursor: loading ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                boxShadow: '0 10px 25px -5px rgba(37, 99, 235, 0.4)',
                opacity: loading ? 0.7 : 1,
                transition: 'all 0.2s'
              }}
            >
              {loading ? (
                <>
                  <RefreshCw size={18} className="spin" /> Ingesting News & Running Models...
                </>
              ) : (
                <>
                  <ShieldCheck size={18} /> Analyze & Shield My Portfolio
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Results Section (Rendered on success) */}
      {results && (
        <div className="results-container" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Executive Overview Banner */}
          <div className="card" style={{
            background: '#0f172a',
            border: '1px solid rgba(56, 189, 248, 0.3)',
            borderRadius: '16px',
            padding: '1.5rem',
            boxShadow: '0 15px 35px -10px rgba(0, 0, 0, 0.5)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', marginBottom: '1rem' }}>
              <div>
                <span className="pill blue" style={{ marginBottom: '6px', display: 'inline-block' }}>
                  AI RISK ASSESSMENT COMPLETE
                </span>
                <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#f8fafc', margin: 0 }}>
                  Portfolio Health & Rebalancing Prescription
                </h2>
              </div>

              {onAskBujji && (
                <button
                  onClick={() => onAskBujji(`Explain my custom portfolio analysis: my health score is ${results.health_score.before.score}/100 and target trades recommend adjusting ${results.execution_orders.map((o: any) => o.ticker).join(', ')}.`)}
                  style={{
                    background: 'rgba(255, 255, 255, 0.08)',
                    border: '1px solid rgba(255, 255, 255, 0.15)',
                    color: '#e2e8f0',
                    borderRadius: '8px',
                    padding: '8px 14px',
                    fontSize: '0.85rem',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px'
                  }}
                >
                  <Sparkles size={14} color="#38bdf8" /> Ask Bujji to Explain This
                </button>
              )}
            </div>

            <p style={{ color: '#cbd5e1', fontSize: '0.95rem', lineHeight: 1.6, background: 'rgba(15, 23, 42, 0.8)', padding: '12px 16px', borderRadius: '10px', borderLeft: '4px solid #38bdf8', margin: '0 0 1.25rem 0' }}>
              {results.summary}
            </p>

            {/* Metric Score Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
              {/* Health Score Gauge */}
              <div style={{ background: '#1e293b', padding: '1.25rem', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.05)', textAlign: 'center' }}>
                <span style={{ fontSize: '0.8rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Portfolio Health Score
                </span>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', margin: '0.5rem 0' }}>
                  <span style={{ fontSize: '2.4rem', fontWeight: 800, color: results.health_score.before.score >= 80 ? '#10b981' : results.health_score.before.score >= 65 ? '#f59e0b' : '#ef4444' }}>
                    {results.health_score.before.score}
                  </span>
                  <span style={{ fontSize: '1.1rem', color: '#64748b' }}>/ 100</span>
                </div>
                <span className={`pill ${results.health_score.before.score >= 80 ? 'green' : results.health_score.before.score >= 65 ? 'amber' : 'red'}`}>
                  {results.health_score.before.rating}
                </span>
              </div>

              {/* Volatility Reduction */}
              <div style={{ background: '#1e293b', padding: '1.25rem', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
                <span style={{ fontSize: '0.8rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Annual Volatility
                </span>
                <div style={{ margin: '0.5rem 0', display: 'flex', alignItems: 'baseline', gap: '8px' }}>
                  <strong style={{ fontSize: '1.75rem', color: '#f8fafc' }}>
                    {results.volatility_comparison.after_pct}%
                  </strong>
                  <span style={{ fontSize: '0.85rem', color: '#94a3b8', textDecoration: 'line-through' }}>
                    {results.volatility_comparison.before_pct}%
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#10b981', fontSize: '0.8rem', fontWeight: 600 }}>
                  <TrendingDown size={14} /> Volatility trimmed by {results.volatility_comparison.volatility_reduction_pct}%
                </div>
              </div>

              {/* 1-Day Value at Risk (VaR 95%) */}
              <div style={{ background: '#1e293b', padding: '1.25rem', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
                <span style={{ fontSize: '0.8rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Daily Value at Risk (95% VaR)
                </span>
                <div style={{ margin: '0.5rem 0' }}>
                  <strong style={{ fontSize: '1.75rem', color: '#38bdf8' }}>
                    {results.volatility_comparison.daily_var_95_cash_after}
                  </strong>
                </div>
                <span style={{ fontSize: '0.8rem', color: '#64748b' }}>
                  Before: {results.volatility_comparison.daily_var_95_cash_before}
                </span>
              </div>
            </div>
          </div>

          {/* Actionable Rebalancing Orders Table */}
          <div className="card" style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px', padding: '1.5rem' }}>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: '#f8fafc', margin: '0 0 0.5rem 0' }}>
              Actionable Execution Sheet (What to Buy & Sell)
            </h3>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: '1rem' }}>
              Specific dollar and weight adjustments recommended to align your portfolio with institutional risk constraints.
            </p>

            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #334155', color: '#64748b' }}>
                    <th style={{ padding: '10px 12px' }}>STOCK</th>
                    <th style={{ padding: '10px 12px' }}>SECTOR</th>
                    <th style={{ padding: '10px 12px' }}>CURRENT</th>
                    <th style={{ padding: '10px 12px' }}>RECOMMENDED</th>
                    <th style={{ padding: '10px 12px' }}>TRADE ACTION</th>
                    <th style={{ padding: '10px 12px' }}>CASH DELTA</th>
                    <th style={{ padding: '10px 12px' }}>AI SIGNAL REASON</th>
                  </tr>
                </thead>
                <tbody>
                  {results.execution_orders.map((order: any, idx: number) => {
                    const isBuy = order.action.includes('EXPAND');
                    const isSell = order.action.includes('TRIM');
                    return (
                      <tr key={idx} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                        <td style={{ padding: '12px', fontWeight: 700, color: '#38bdf8' }}>
                          {order.ticker}
                        </td>
                        <td style={{ padding: '12px', color: '#94a3b8' }}>
                          {order.sector}
                        </td>
                        <td style={{ padding: '12px', color: '#e2e8f0' }}>
                          {order.current_weight_pct}% <span style={{ color: '#64748b' }}>({order.current_cash})</span>
                        </td>
                        <td style={{ padding: '12px', fontWeight: 600, color: '#f8fafc' }}>
                          {order.target_weight_pct}% <span style={{ color: '#64748b' }}>({order.target_cash})</span>
                        </td>
                        <td style={{ padding: '12px' }}>
                          <span className={`pill ${isBuy ? 'green' : isSell ? 'red' : 'neutral'}`}>
                            {order.action}
                          </span>
                        </td>
                        <td style={{ padding: '12px', fontWeight: 700, color: isBuy ? '#10b981' : isSell ? '#ef4444' : '#94a3b8' }}>
                          {order.trade_cash_delta}
                        </td>
                        <td style={{ padding: '12px', color: '#94a3b8', fontSize: '0.8rem', maxWidth: '320px' }}>
                          {order.reason}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Panic & Fake News Filter ("FUD vs. Real Threat") */}
          <div className="card" style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px', padding: '1.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '0.5rem' }}>
              <Compass size={20} color="#38bdf8" />
              <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: '#f8fafc', margin: 0 }}>
                Panic & Fake News Filter (FUD vs. Real Risk)
              </h3>
            </div>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: '1.25rem' }}>
              Every headline fetched for your stocks is filtered through FinBERT + XGBoost to differentiate between emotional clickbait noise and genuine balance sheet threats.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1rem' }}>
              {results.scored_news_feed.map((news: any, idx: number) => {
                const isPanicNoise = news.fud_type === 'PANIC_NOISE';
                const isRealRisk = news.fud_type === 'GENUINE_RISK';
                const isGrowth = news.fud_type === 'GROWTH_CATALYST';

                return (
                  <div
                    key={idx}
                    style={{
                      background: '#1e293b',
                      borderRadius: '12px',
                      padding: '1.25rem',
                      border: isPanicNoise ? '1px solid #38bdf8' : isRealRisk ? '1px solid #ef4444' : isGrowth ? '1px solid #10b981' : '1px solid rgba(255, 255, 255, 0.08)',
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'space-between',
                      gap: '10px'
                    }}
                  >
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                        <span style={{ fontWeight: 800, color: '#38bdf8', fontSize: '0.9rem' }}>
                          {news.ticker}
                        </span>
                        <span className={`pill ${isPanicNoise ? 'blue' : isRealRisk ? 'red' : isGrowth ? 'green' : 'neutral'}`}>
                          {isPanicNoise ? '🛡️ PANIC NOISE' : isRealRisk ? '⚠️ GENUINE RISK' : isGrowth ? '🚀 GROWTH CATALYST' : 'ROUTINE NEWS'}
                        </span>
                      </div>
                      <p style={{ fontSize: '0.88rem', color: '#f8fafc', fontWeight: 500, margin: '0 0 8px 0', lineHeight: 1.4 }}>
                        "{news.headline}"
                      </p>
                    </div>

                    <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '8px 10px', borderRadius: '8px', fontSize: '0.78rem', color: '#cbd5e1' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                        <span>Classification: <strong>{news.event_class}</strong></span>
                        <span>Impact: <strong>{news.impact_decile}/10</strong></span>
                      </div>
                      <span style={{ color: isPanicNoise ? '#38bdf8' : isRealRisk ? '#fca5a5' : isGrowth ? '#86efac' : '#94a3b8' }}>
                        {news.fud_message}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Everyday Life Stress Scenarios */}
          <div className="card" style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px', padding: '1.5rem' }}>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: '#f8fafc', margin: '0 0 0.5rem 0' }}>
              Real-World Crisis Stress Simulator
            </h3>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: '1.25rem' }}>
              See how your portfolio performs across practical macroeconomic crises, and how much money our AI rebalancing protects.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
              {results.stress_scenarios.map((sc: any, idx: number) => (
                <div
                  key={idx}
                  style={{
                    background: '#1e293b',
                    borderRadius: '12px',
                    padding: '1.25rem',
                    border: '1px solid rgba(255, 255, 255, 0.05)',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                    gap: '12px'
                  }}
                >
                  <div>
                    <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f8fafc', margin: '0 0 4px 0' }}>
                      {sc.name}
                    </h4>
                    <p style={{ fontSize: '0.8rem', color: '#94a3b8', margin: 0, lineHeight: 1.4 }}>
                      {sc.description}
                    </p>
                  </div>

                  <div style={{ background: 'rgba(15, 23, 42, 0.8)', padding: '10px 12px', borderRadius: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#94a3b8', marginBottom: '6px' }}>
                      <span>Loss Without AI:</span>
                      <strong style={{ color: '#ef4444' }}>{sc.current_dollar_impact} ({sc.current_portfolio_loss_pct}%)</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#94a3b8', marginBottom: '6px' }}>
                      <span>Loss After Rebalance:</span>
                      <strong style={{ color: '#38bdf8' }}>{sc.optimized_dollar_impact} ({sc.optimized_portfolio_loss_pct}%)</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', fontWeight: 700, paddingTop: '6px', borderTop: '1px solid rgba(255, 255, 255, 0.08)' }}>
                      <span style={{ color: '#10b981' }}>Capital Shielded:</span>
                      <span style={{ color: '#10b981' }}>{sc.capital_shielded}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
