import { useState, useEffect } from 'react';
import {
  Activity, ArrowRight, ArrowUpRight, BrainCircuit,
  ChevronDown, ExternalLink, Github, Layers, Sparkles, TrendingUp, ShieldAlert, ShieldCheck, User
} from 'lucide-react';
import type { Page } from '../types';
import { api } from '../api';

interface HomeLandingProps {
  onStart: () => void;
  onTab: (tab: Page) => void;
  onMode?: (mode: 'quick' | 'analyst') => void;
}

export function HomeLanding({ onStart, onTab, onMode }: HomeLandingProps) {
  // Presentation Deck scroll progress & reveal observer
  const [scrollProgress, setScrollProgress] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      const totalHeight = document.documentElement.scrollHeight - window.innerHeight;
      if (totalHeight > 0) {
        setScrollProgress(Math.min(100, Math.max(0, (window.scrollY / totalHeight) * 100)));
      }
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-revealed');
        }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -30px 0px' });

    const targets = document.querySelectorAll(
      '.landing-section, .hero-diagram, .transformation-grid, .engine-deepdive, .instruments-grid, .modes-grid, .constraints-showcase, .faq-container'
    );
    targets.forEach(el => observer.observe(el));
    window.addEventListener('scroll', handleScroll, { passive: true });

    return () => {
      observer.disconnect();
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  // Interactive Walkthrough state
  const [selectedExample, setSelectedExample] = useState<'aapl' | 'msft'>('aapl');
  const [activeStep, setActiveStep] = useState<number>(1);
  const [liveAnalyzing, setLiveAnalyzing] = useState(false);
  const [liveResult, setLiveResult] = useState<{ sentiment: number; eventClass: string; impact: number } | null>(null);
  const [liveError, setLiveError] = useState('');

  // Accordion state
  const [openFaq, setOpenFaq] = useState<string | null>(null);

  const toggleFaq = (id: string) => {
    setOpenFaq(prev => prev === id ? null : id);
  };

  const handleRunLiveInference = async () => {
    setLiveAnalyzing(true);
    setLiveError('');
    setLiveResult(null);
    try {
      const text = selectedExample === 'aapl'
        ? "Apple announces next-generation Silicon processor lineup and expands enterprise hardware subscription ecosystem."
        : "Microsoft announces major expansion of enterprise cloud compute clusters and datacenter infrastructure to meet commercial AI demand.";
      const ticker = selectedExample === 'aapl' ? 'AAPL' : 'MSFT';
      const res = await api.analyze({
        text,
        ticker,
        published_at: new Date().toISOString(),
        source: 'manual',
      });
      setLiveResult({
        sentiment: res.sentiment.score,
        eventClass: res.event.class_name,
        impact: res.impact.score,
      });
      setActiveStep(3);
    } catch {
      setLiveError('Actual AI model inference is unavailable. Start the backend with the authentic trained checkpoints; no prediction has been produced.');
      setActiveStep(2);
    } finally {
      setLiveAnalyzing(false);
    }
  };

  return (
    <div className="landing-root">
      {/* ──────────────────────────────────────────────────────────── */}
      {/* TOP NAVIGATION BAR                                           */}
      {/* ──────────────────────────────────────────────────────────── */}
      <header className="landing-nav" role="banner">
        <div className="deck-progress-track" aria-hidden="true">
          <div className="deck-progress-bar" style={{ width: `${scrollProgress}%` }} />
        </div>
        <div className="landing-nav-inner">
          <div className="landing-brand">
            <Activity className="landing-brand-icon" size={24} />
            <span className="landing-brand-text">Soochana <strong>AI</strong></span>
            <span className="deck-live-pill"><span className="pulse-dot-green" /> STAGE OVERVIEW</span>
          </div>

          <nav className="landing-menu" aria-label="Landing Navigation">
            <a href="#how-it-works" className="landing-link">How It Works</a>
            <a href="#ai-engine" className="landing-link">AI Engine</a>
            <a href="#applications" className="landing-link">Applications</a>
            <a href="#view-modes" className="landing-link">View Modes</a>
            <a href="#interactive-demo" className="landing-link">Interactive Demo</a>
          </nav>

          <div className="landing-nav-actions">
            <button
              type="button"
              className="landing-btn-primary"
              onClick={onStart}
              aria-label="Open Soochana AI Dashboard"
            >
              <span>Open Dashboard</span>
              <ArrowUpRight size={16} />
            </button>
          </div>
        </div>
      </header>

      {/* ──────────────────────────────────────────────────────────── */}
      {/* SECTION 1: HERO                                              */}
      {/* ──────────────────────────────────────────────────────────── */}
      <section className="landing-hero" id="overview">
        <div className="landing-hero-container">
          <div className="landing-hero-badge">
            <span className="badge-pulse" />
            <span>STAGE 01 / 06 · EXPLAINABLE FINANCIAL RISK INTELLIGENCE</span>
          </div>

          <h1 className="landing-hero-title">
            Financial information<br className="mobile-break" /> is everywhere.<br />
            <span className="text-secondary">Clarity isn't.</span>
          </h1>

          <p className="landing-hero-subtitle">
            Soochana AI transforms financial news and public market discussions into explainable
            AI signals for portfolio rebalancing and stress testing.
          </p>

          <div className="landing-hero-buttons">
            <a href="#ai-engine" className="landing-btn-primary">
              <span>Explore the Engine</span>
              <ArrowRight size={16} />
            </a>
            <button
              type="button"
              className="landing-btn-secondary"
              onClick={onStart}
            >
              <span>Open Dashboard</span>
              <ArrowUpRight size={16} />
            </button>
          </div>

          {/* Animated Conceptual Architecture Illustration */}
          <div className="hero-diagram" aria-label="System Architecture Overview">
            <div className="hero-sources-col">
              <div className="hero-source-card float-slow">
                <div className="card-tag">EQUITY SYMBOL</div>
                <strong>Apple (AAPL)</strong>
                <small>Illustrative company reference</small>
              </div>

              <div className="hero-source-card float-medium">
                <div className="card-tag">EQUITY SYMBOL</div>
                <strong>Microsoft (MSFT)</strong>
                <small>Illustrative company reference</small>
              </div>

              <div className="hero-source-card float-slow">
                <div className="card-tag">PUBLIC SOURCE</div>
                <strong>Financial News</strong>
                <small>Unstructured corporate reports</small>
              </div>

              <div className="hero-source-card float-fast">
                <div className="card-tag">PUBLIC SOURCE</div>
                <strong>Social Discussion</strong>
                <small>Public market discourse</small>
              </div>
            </div>

            {/* Connecting data flow lines with live animated moving packets */}
            <div className="hero-flow-conduit" aria-hidden="true">
              <svg className="flow-svg" viewBox="0 0 200 320" fill="none" preserveAspectRatio="none">
                <path d="M 10 40 C 90 40, 110 160, 190 160" stroke="#d5d8dc" strokeWidth="2" strokeDasharray="4 4" className="flow-line" />
                <path d="M 10 120 C 90 120, 110 160, 190 160" stroke="#d5d8dc" strokeWidth="2" strokeDasharray="4 4" className="flow-line delay-1" />
                <path d="M 10 200 C 90 200, 110 160, 190 160" stroke="#d5d8dc" strokeWidth="2" strokeDasharray="4 4" className="flow-line delay-2" />
                <path d="M 10 280 C 90 280, 110 160, 190 160" stroke="#d5d8dc" strokeWidth="2" strokeDasharray="4 4" className="flow-line delay-3" />
                <circle r="4" fill="#111" opacity="0.85">
                  <animateMotion dur="2.4s" repeatCount="indefinite" path="M 10 40 C 90 40, 110 160, 190 160" />
                </circle>
                <circle r="4" fill="#111" opacity="0.85">
                  <animateMotion dur="2.8s" repeatCount="indefinite" path="M 10 120 C 90 120, 110 160, 190 160" />
                </circle>
                <circle r="4" fill="#111" opacity="0.85">
                  <animateMotion dur="2.2s" repeatCount="indefinite" path="M 10 200 C 90 200, 110 160, 190 160" />
                </circle>
                <circle r="4" fill="#111" opacity="0.85">
                  <animateMotion dur="2.6s" repeatCount="indefinite" path="M 10 280 C 90 280, 110 160, 190 160" />
                </circle>
              </svg>
            </div>

            {/* Central Risk Engine Node */}
            <div className="hero-engine-node">
              <div className="engine-pulse-ring" />
              <div className="engine-card">
                <div className="engine-icon-wrap">
                  <BrainCircuit size={28} />
                </div>
                <h3>Soochana AI Engine</h3>
                <p>Neural NLP &amp; Gradient-Boosted Impact</p>
                <div className="engine-specs">
                  <span>FinBERT Sentiment</span>
                  <span>5-Class Classifier</span>
                  <span>XGBoost Severity</span>
                </div>
              </div>
            </div>

            {/* Downstream branches with live animated packets */}
            <div className="hero-flow-conduit" aria-hidden="true">
              <svg className="flow-svg" viewBox="0 0 200 320" fill="none" preserveAspectRatio="none">
                <path d="M 10 160 C 90 160, 110 80, 190 80" stroke="#d5d8dc" strokeWidth="2" strokeDasharray="4 4" className="flow-line" />
                <path d="M 10 160 C 90 160, 110 240, 190 240" stroke="#d5d8dc" strokeWidth="2" strokeDasharray="4 4" className="flow-line delay-2" />
                <circle r="4" fill="#111" opacity="0.85">
                  <animateMotion dur="2.3s" repeatCount="indefinite" path="M 10 160 C 90 160, 110 80, 190 80" />
                </circle>
                <circle r="4" fill="#111" opacity="0.85">
                  <animateMotion dur="2.7s" repeatCount="indefinite" path="M 10 160 C 90 160, 110 240, 190 240" />
                </circle>
              </svg>
            </div>

            <div className="hero-apps-col">
              <div className="hero-app-card" onClick={() => onTab('moduleA')}>
                <div className="card-tag">MODULE A</div>
                <strong>Index Rebalancing</strong>
                <small>15-stock constrained index optimization</small>
                <span className="app-link">Explore Lab →</span>
              </div>

              <div className="hero-app-card" onClick={() => onTab('moduleB')}>
                <div className="card-tag">MODULE B</div>
                <strong>Portfolio Stress Testing</strong>
                <small>Multi-asset MTM &amp; 12-month ECL shocks</small>
                <span className="app-link">Explore Lab →</span>
              </div>
            </div>
          </div>

          <p className="hero-caption">
            *Illustrative company references shown above demonstrate processing architecture, not corporate endorsement or live execution.
          </p>
        </div>
      </section>

      {/* ──────────────────────────────────────────────────────────── */}
      {/* SECTION 2: THE PROBLEM                                       */}
      {/* ──────────────────────────────────────────────────────────── */}
      <section className="landing-section bg-alt" id="how-it-works">
        <div className="landing-container">
          <div className="section-eyebrow">01 / THE CHALLENGE</div>
          <h2 className="section-heading">
            Markets generate information faster<br />than people can interpret it.
          </h2>

          <p className="section-lead">
            Financial news and public discussions often contain signals about company performance,
            economic events, credit conditions and strategic changes. But these signals arrive as
            unstructured text, making consistent financial risk interpretation difficult.
          </p>

          <p className="section-sublead">
            Soochana AI converts these narratives into structured, interpretable model outputs
            that can support downstream portfolio analysis.
          </p>

          {/* Animated Before vs After Comparison */}
          <div className="transformation-grid">
            <div className="transformation-card card-unstructured">
              <div className="trans-header">
                <span className="trans-pill tag-unstructured">BEFORE</span>
                <h3>Unstructured Information</h3>
              </div>
              <div className="narrative-snippet">
                <span className="snippet-quote">“</span>
                <p>
                  Sources report Apple accelerates hardware development for next-gen silicon while
                  scrutiny over regulatory cross-border licensing intensifies across European trade channels...
                </p>
              </div>
              <ul className="trans-list">
                <li>Ambiguous, qualitative tone across disparate publications</li>
                <li>Unquantified event categories mixed within long narratives</li>
                <li>No direct relationship to historical volatility or portfolio constraints</li>
              </ul>
            </div>

            <div className="transformation-arrow" aria-hidden="true">
              <div className="arrow-pulse">
                <ArrowRight size={26} />
              </div>
              <span>Explainable Transformation</span>
            </div>

            <div className="transformation-card card-structured">
              <div className="trans-header">
                <span className="trans-pill tag-structured">AFTER</span>
                <h3>Structured Risk Intelligence</h3>
              </div>
              <div className="structured-schema">
                <div className="schema-row">
                  <span className="schema-key">Sentiment Score:</span>
                  <strong className="schema-val text-pos">+0.61 Continuous [−1 to +1]</strong>
                </div>
                <div className="schema-row">
                  <span className="schema-key">Event Category:</span>
                  <strong className="schema-val">Product Launch (99.7% Conf)</strong>
                </div>
                <div className="schema-row">
                  <span className="schema-key">Impact Severity:</span>
                  <strong className="schema-val">7 / 10 Magnitude Rating</strong>
                </div>
                <div className="schema-row">
                  <span className="schema-key">Volatility Context:</span>
                  <strong className="schema-val text-muted">20-day annualized rolling baseline</strong>
                </div>
              </div>
              <ul className="trans-list">
                <li>Traceable mathematical inputs for downstream optimization</li>
                <li>Standardized REST schema ready for algorithmic consumption</li>
                <li>Transparent distinction between language tone and portfolio shocks</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* ──────────────────────────────────────────────────────────── */}
      {/* SECTION 3: INSIDE THE AI ENGINE                              */}
      {/* ──────────────────────────────────────────────────────────── */}
      <section className="landing-section" id="ai-engine">
        <div className="landing-container">
          <div className="section-eyebrow">02 / MODEL METHODOLOGY</div>
          <h2 className="section-heading">
            Three models. Three complementary signals.
          </h2>

          <p className="section-lead">
            Our pipeline uses two financial-language models and a gradient-boosted impact model to
            characterize financial events.
          </p>

          <div className="engine-deepdive">
            {/* Step A — Text Understanding */}
            <div className="model-step-card">
              <div className="step-badge">STEP A · LANGUAGE UNDERSTANDING</div>
              <h3>Contextual Tokenization &amp; Financial Transformer</h3>
              <p className="step-example-tag">
                <em>Illustrative example:</em> A public financial article discusses an upcoming Apple product launch.
              </p>

              <div className="token-visualizer" aria-label="Tokenization representation">
                <span className="token-chip">[CLS]</span>
                <span className="token-chip">public</span>
                <span className="token-chip">financial</span>
                <span className="token-chip">article</span>
                <span className="token-chip">discusses</span>
                <span className="token-chip">upcoming</span>
                <span className="token-chip token-entity">Apple</span>
                <span className="token-chip token-concept">product</span>
                <span className="token-chip token-concept">launch</span>
                <span className="token-chip">[SEP]</span>
              </div>

              <div className="step-explanation">
                <p>
                  FinBERT is based on the BERT transformer architecture, adapted to financial-language tasks.
                  Its contextual representations help capture relationships between words and their financial meaning.
                </p>
                <small className="text-secondary">
                  *The transformer computes multi-head contextual embeddings; it does not rely on simple word counting or keyword lists.
                </small>
              </div>
            </div>

            {/* Step B — FinBERT Sentiment Model */}
            <div className="model-step-card">
              <div className="step-badge">STEP B · TRAINED MODEL 1</div>
              <div className="model-head-row">
                <div>
                  <h3>Financial Sentiment</h3>
                  <span className="model-sub">FinBERT Continuous Sentiment Regressor</span>
                </div>
                <div className="range-indicator">Range: <strong>−1 to +1</strong></div>
              </div>

              <p>
                The sentiment model estimates whether the financial tone of the text is negative,
                neutral-leaning, or positive on a continuous scale.
              </p>

              {/* Horizontal Sentiment Scale */}
              <div className="sentiment-scale-container" aria-label="Sentiment Scale -1 to +1">
                <div className="scale-track">
                  <div className="scale-zone zone-neg">Negative (−1.0)</div>
                  <div className="scale-zone zone-neu">Neutral (0.0)</div>
                  <div className="scale-zone zone-pos">Positive (+1.0)</div>
                </div>
                <div className="scale-legend">
                  <span>Language Tone Assessment</span>
                  <span className="text-secondary">Does not imply guaranteed stock-price direction</span>
                </div>
              </div>
            </div>

            {/* Step C — FinBERT Event Classification Model */}
            <div className="model-step-card">
              <div className="step-badge">STEP C · TRAINED MODEL 2</div>
              <div className="model-head-row">
                <div>
                  <h3>Financial Event Classification</h3>
                  <span className="model-sub">FinBERT 5-Category Event Classifier</span>
                </div>
                <div className="range-indicator">Output: <strong>5 Class Probabilities</strong></div>
              </div>

              <p>
                A separately fine-tuned FinBERT model assigns probabilities to five defined event categories.
              </p>

              {/* Conceptual 5-bar Graphic */}
              <div className="category-bars" aria-label="Event Classification Categories">
                <div className="cat-bar-item">
                  <div className="cat-bar-label">
                    <span>Product Launch</span>
                    <strong>High probability for product narrative</strong>
                  </div>
                  <div className="cat-bar-track"><div className="cat-bar-fill" style={{ width: '85%' }} /></div>
                </div>

                <div className="cat-bar-item">
                  <div className="cat-bar-label">
                    <span>Merger / Acquisition</span>
                    <span>Corporate restructuring</span>
                  </div>
                  <div className="cat-bar-track"><div className="cat-bar-fill" style={{ width: '6%' }} /></div>
                </div>

                <div className="cat-bar-item">
                  <div className="cat-bar-label">
                    <span>Macroeconomic</span>
                    <span>Rates, inflation, labor indicators</span>
                  </div>
                  <div className="cat-bar-track"><div className="cat-bar-fill" style={{ width: '4%' }} /></div>
                </div>

                <div className="cat-bar-item">
                  <div className="cat-bar-label">
                    <span>Credit Event</span>
                    <span>Debt issuance, rating changes, covenants</span>
                  </div>
                  <div className="cat-bar-track"><div className="cat-bar-fill" style={{ width: '3%' }} /></div>
                </div>

                <div className="cat-bar-item">
                  <div className="cat-bar-label">
                    <span>Geopolitical</span>
                    <span>Tariffs, cross-border restrictions</span>
                  </div>
                  <div className="cat-bar-track"><div className="cat-bar-fill" style={{ width: '2%' }} /></div>
                </div>
              </div>

              {/* Expandable Explanation Accordion */}
              <div className="accordion-item">
                <button
                  type="button"
                  className="accordion-trigger"
                  onClick={() => toggleFaq('faq-classification')}
                >
                  <span>How event classification probabilities work</span>
                  <ChevronDown size={16} className={openFaq === 'faq-classification' ? 'rotate-180' : ''} />
                </button>
                {openFaq === 'faq-classification' && (
                  <div className="accordion-content">
                    <p>
                      The model passes token representations through a sequence classification head, applying a
                      softmax function to generate a probability distribution summing to 1.0. While product announcements
                      naturally lean toward <em>Product Launch</em>, the model evaluates contextual uncertainty and may
                      exhibit domain-specific calibration nuances. The classifier is an assistive tool and does not guarantee
                      flawless categorization for every headline.
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Step D — XGBoost Impact Severity Model */}
            <div className="model-step-card">
              <div className="step-badge">STEP D · TRAINED MODEL 3</div>
              <div className="model-head-row">
                <div>
                  <h3>Impact Severity Estimation</h3>
                  <span className="model-sub">XGBoost Volatility &amp; Event Regressor</span>
                </div>
                <div className="range-indicator">Scale: <strong>1 to 10</strong></div>
              </div>

              <p>
                An XGBoost regression model uses engineered numerical features from the risk-analysis pipeline
                to estimate a relative event-impact severity score.
              </p>

              {/* Exact engineered features list */}
              <div className="features-grid">
                <div className="feature-pill"><code>sentiment_score</code> &amp; <code>abs_sentiment</code></div>
                <div className="feature-pill"><code>5 event class probabilities</code></div>
                <div className="feature-pill"><code>stock_volatility_20d_ann</code></div>
                <div className="feature-pill"><code>volume_ratio_20d</code></div>
                <div className="feature-pill"><code>stock_return_5d &amp; 20d</code></div>
                <div className="feature-pill"><code>spy_return_20d &amp; volatility</code></div>
                <div className="feature-pill"><code>market_regime_quantiles</code></div>
              </div>

              <div className="severity-meter">
                <div className="meter-ticks">
                  <span>1</span><span>2</span><span>3</span><span>4</span><span>5</span>
                  <span>6</span><span>7</span><span>8</span><span>9</span><span>10</span>
                </div>
                <div className="meter-track">
                  <div className="meter-fill" style={{ width: '70%' }} />
                </div>
                <div className="meter-labels">
                  <span>1: Low Market Disruption</span>
                  <span>5: Moderate Magnitude</span>
                  <span>10: Severe Disruption</span>
                </div>
              </div>

              <div className="step-caution">
                <ShieldAlert size={16} />
                <span>
                  Impact severity represents modeled event magnitude. It is not a forecasted percentage return,
                  a buy/sell instruction, or a probability of default.
                </span>
              </div>
            </div>

            {/* Final Model Output Convergence */}
            <div className="convergence-card">
              <div className="convergence-title">
                <Layers size={20} />
                <h4>Final Unified Output Structure</h4>
              </div>
              <p>
                All three models converge into one transparent, machine-readable signal package consumed by
                downstream research modules:
              </p>
              <div className="convergence-pills">
                <div className="conv-pill">
                  <small>FinBERT Sentiment</small>
                  <strong>+0.61 (Continuous)</strong>
                </div>
                <span className="conv-plus">+</span>
                <div className="conv-pill">
                  <small>FinBERT Event</small>
                  <strong>Product Launch (99.7%)</strong>
                </div>
                <span className="conv-plus">+</span>
                <div className="conv-pill">
                  <small>XGBoost Impact</small>
                  <strong>7 / 10 Severity</strong>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ──────────────────────────────────────────────────────────── */}
      {/* SECTION 4: FROM AI SIGNAL TO FINANCIAL ACTION               */}
      {/* ──────────────────────────────────────────────────────────── */}
      <section className="landing-section bg-alt" id="applications">
        <div className="landing-container">
          <div className="section-eyebrow">03 / DOWNSTREAM APPLICATIONS</div>
          <h2 className="section-heading">
            Understanding an event is only the beginning.
          </h2>

          <p className="section-lead">
            Soochana AI goes beyond interpreting financial information. Its structured signals can
            inform two separate portfolio research workflows.
          </p>

          {/* Branching diagram */}
          <div className="branching-diagram">
            <div className="branch-origin">
              <Sparkles size={20} />
              <span>Structured Risk Signal</span>
            </div>

            <div className="branch-fork" aria-hidden="true">
              <svg viewBox="0 0 160 80" className="fork-svg">
                <path d="M 10 40 L 70 40 L 150 15" stroke="#111" strokeWidth="2" fill="none" />
                <path d="M 10 40 L 70 40 L 150 65" stroke="#111" strokeWidth="2" fill="none" />
              </svg>
            </div>

            <div className="branch-destinations">
              <div className="destination-card" onClick={() => onTab('moduleA')}>
                <div className="dest-tag">MODULE A</div>
                <h3>Tactical Index Rebalancing</h3>
                <p>Constrained weight optimization for a simulated 15-stock basket.</p>
                <span className="dest-action">View Module A Details ↓</span>
              </div>

              <div className="destination-card" onClick={() => onTab('moduleB')}>
                <div className="dest-tag">MODULE B</div>
                <h3>Portfolio Stress Testing</h3>
                <p>Hypothetical multi-asset valuation across loans, bonds, swaps, and options.</p>
                <span className="dest-action">View Module B Details ↓</span>
              </div>
            </div>
          </div>

          <p className="branch-note">
            *Both modules represent independent downstream analytical research workflows. Stress scenarios are hypothetical
            and require deliberate analyst review before simulation.
          </p>
        </div>
      </section>

      {/* ──────────────────────────────────────────────────────────── */}
      {/* SECTION 5: MODULE A — INDEX REBALANCING                      */}
      {/* ──────────────────────────────────────────────────────────── */}
      <section className="landing-section" id="module-a">
        <div className="landing-container">
          <div className="section-eyebrow">04 / MODULE A</div>
          <h2 className="section-heading">
            Explore portfolio allocation changes<br />with explicit constraints.
          </h2>

          <p className="section-lead">
            Module A uses financial risk signals to propose simulated allocation adjustments for a research
            index containing 15 stocks.
          </p>

          {/* Workflow Graphic */}
          <div className="module-flow">
            <div className="flow-step">
              <span className="step-num">01</span>
              <strong>AI Signals</strong>
              <small>Sentiment &amp; severity</small>
            </div>
            <span className="flow-arrow">→</span>
            <div className="flow-step">
              <span className="step-num">02</span>
              <strong>Allocation Proposal</strong>
              <small>Raw target weights</small>
            </div>
            <span className="flow-arrow">→</span>
            <div className="flow-step">
              <span className="step-num">03</span>
              <strong>Constraint Check</strong>
              <small>Bounds &amp; turnover</small>
            </div>
            <span className="flow-arrow">→</span>
            <div className="flow-step step-final">
              <span className="step-num">04</span>
              <strong>Simulated Portfolio</strong>
              <small>Sum(w) = 100%</small>
            </div>
          </div>

          {/* Constraints Grid */}
          <div className="constraints-showcase">
            <div className="constraint-box">
              <span className="cons-label">INDIVIDUAL STOCK LIMIT</span>
              <strong>±2 percentage points</strong>
              <p>Strict single-stock deviation guardrails from baseline index weight.</p>
            </div>

            <div className="constraint-box">
              <span className="cons-label">SECTOR CONCENTRATION</span>
              <strong>±5 percentage points</strong>
              <p>Caps total aggregate tilt per sector to prevent unhedged sector bets.</p>
            </div>

            <div className="constraint-box">
              <span className="cons-label">ONE-WAY TURNOVER</span>
              <strong>10% Maximum</strong>
              <p>Controls portfolio churn and reduces execution friction.</p>
            </div>

            <div className="constraint-box">
              <span className="cons-label">TRANSACTION COST MODEL</span>
              <strong>10 basis points</strong>
              <p>Accounts for realistic trading slippage and liquidity friction.</p>
            </div>
          </div>

          <div className="backtest-summary-box">
            <div className="bts-icon"><TrendingUp size={22} /></div>
            <div>
              <h4>Historical Backtesting &amp; Strategy Validation</h4>
              <p>
                Evaluated across 350+ historical trading sessions without lookahead bias: rebalancing decisions
                use only articles published strictly prior to each session's trading cutoff.
              </p>
            </div>
          </div>

          <div className="module-actions-row">
            <button
              type="button"
              className="landing-btn-primary"
              onClick={() => onTab('moduleA')}
            >
              <span>Explore Index Lab</span>
              <ArrowRight size={16} />
            </button>
            <span className="module-disclaimer">
              *Research simulation only. No brokerage trade execution or live capital deployment.
            </span>
          </div>
        </div>
      </section>

      {/* ──────────────────────────────────────────────────────────── */}
      {/* SECTION 6: MODULE B — STRESS TESTING                         */}
      {/* ──────────────────────────────────────────────────────────── */}
      <section className="landing-section bg-alt" id="module-b">
        <div className="landing-container">
          <div className="section-eyebrow">05 / MODULE B</div>
          <h2 className="section-heading">
            What could happen when markets face a shock?
          </h2>

          <p className="section-lead">
            Module B evaluates hypothetical financial scenarios against a synthetic portfolio of loans,
            bonds, swaps and options.
          </p>

          {/* 4 Instrument Tiles */}
          <div className="instruments-grid">
            <div className="instrument-card">
              <div className="inst-badge badge-blue">CREDIT ASSETS</div>
              <h3>Loans</h3>
              <div className="inst-metric">12-Month Expected Credit Loss (ECL)</div>
              <p>
                Models credit rating migrations and default shocks: <code>ECL = PD × LGD × EAD</code>.
              </p>
              <div className="inst-distinction">
                Horizon: 12-month expected loss, distinct from mark-to-market.
              </div>
            </div>

            <div className="instrument-card">
              <div className="inst-badge badge-iris">FIXED INCOME</div>
              <h3>Bonds</h3>
              <div className="inst-metric">Mark-to-Market Price Revaluation</div>
              <p>
                Evaluates credit spread widening and yield shocks using modified duration and convexity curves.
              </p>
              <div className="inst-distinction">
                Horizon: Instantaneous price sensitivity to spread shifts.
              </div>
            </div>

            <div className="instrument-card">
              <div className="inst-badge badge-amber">RATES DERIVATIVES</div>
              <h3>Swaps</h3>
              <div className="inst-metric">DV01 Interest-Rate Sensitivity</div>
              <p>
                Calculates dollar-value-of-a-basis-point curve shifts across fixed-for-floating agreements.
              </p>
              <div className="inst-distinction">
                Note: Notional principal does not equal net asset value.
              </div>
            </div>

            <div className="instrument-card">
              <div className="inst-badge badge-rose">EQUITY DERIVATIVES</div>
              <h3>Options</h3>
              <div className="inst-metric">Black–Scholes Repricing</div>
              <p>
                Revalues European call and put options under equity price drops and implied volatility spikes.
              </p>
              <div className="inst-distinction">
                Non-linear valuation incorporating Greeks under shock conditions.
              </div>
            </div>
          </div>

          {/* Stress Workflow */}
          <div className="module-flow stress-flow">
            <div className="flow-step">
              <span className="step-num">01</span>
              <strong>Select Scenario</strong>
              <small>Family &amp; severity</small>
            </div>
            <span className="flow-arrow">→</span>
            <div className="flow-step">
              <span className="step-num">02</span>
              <strong>Review Assumptions</strong>
              <small>Mandatory gate</small>
            </div>
            <span className="flow-arrow">→</span>
            <div className="flow-step">
              <span className="step-num">03</span>
              <strong>Execute Simulation</strong>
              <small>4 loss mechanisms</small>
            </div>
            <span className="flow-arrow">→</span>
            <div className="flow-step step-final">
              <span className="step-num">04</span>
              <strong>Inspect Results</strong>
              <small>MTM P&amp;L vs 12m ECL</small>
            </div>
          </div>

          <div className="module-actions-row">
            <button
              type="button"
              className="landing-btn-primary"
              onClick={() => onTab('moduleB')}
            >
              <span>Explore Stress Lab</span>
              <ArrowRight size={16} />
            </button>
            <span className="module-disclaimer">
              *Hypothetical scenario analysis on synthetic portfolio. Not forecasts of future losses.
            </span>
          </div>
        </div>
      </section>

      {/* ──────────────────────────────────────────────────────────── */}
      {/* SECTION 6: DUAL AUDIENCE INTERFACE MODES                     */}
      {/* ──────────────────────────────────────────────────────────── */}
      <section className="landing-section" id="view-modes">
        <div className="landing-container">
          <div className="section-eyebrow">06 / DUAL OPERATING MODES</div>
          <h2 className="section-heading">
            Designed for two distinct perspectives:<br />Quick View vs. Analyst View
          </h2>

          <p className="section-lead">
            Financial decision-makers and quantitative modelers have fundamentally different objectives.
            Soochana AI provides two tailored interface modes running over the exact same live pipeline and shared state.
          </p>

          <p className="section-sublead">
            Reviewers can switch between views at any time using the <strong>Quick View / Analyst View</strong> segmented toggle in the top-right header of the dashboard (or via the <code>?mode=quick</code> and <code>?mode=analyst</code> URL parameters).
          </p>

          {/* Mode Comparison Grid */}
          <div className="modes-showcase-grid">
            {/* Quick View Card */}
            <div className="mode-card mode-quick-card">
              <div className="mode-card-header">
                <span className="mode-card-badge badge-quick">EXECUTIVE &amp; PM PERSPECTIVE</span>
                <h3>Quick View</h3>
                <span className="mode-card-sub">Streamlined situational awareness without visual or cognitive clutter</span>
              </div>

              <div className="mode-representation">
                <strong>What it represents:</strong> Built for Portfolio Managers, Chief Risk Officers (CROs), and busy executives who need rapid risk triage, headline signals, and fast decision support without navigating complex mathematical parameter matrices.
              </div>

              <ul className="mode-features-list">
                <li>
                  <span className="mode-bullet">✓</span>
                  <div><strong>Curated Signal Cards:</strong> Clean event headlines, continuous sentiment color bars, and 1–10 market impact ratings with source badges.</div>
                </li>
                <li>
                  <span className="mode-bullet">✓</span>
                  <div><strong>High-Level Summary Totals:</strong> Instant visibility into aggregate portfolio metrics: proposed turnover (Δw), backtest performance, and total stress loss.</div>
                </li>
                <li>
                  <span className="mode-bullet">✓</span>
                  <div><strong>Single-Click Workflows:</strong> 1-click index rebalancing proposal generation and 1-click macro stress scenario execution.</div>
                </li>
                <li>
                  <span className="mode-bullet">✓</span>
                  <div><strong>Uncluttered UI:</strong> Secondary mathematical parameters (like 5-class probability arrays and shock vectors) are tucked away to prevent information fatigue.</div>
                </li>
              </ul>

              <div className="mode-testing-box">
                <h4>How to test Quick View in the dashboard:</h4>
                <ol>
                  <li>Click the button below to launch the dashboard in <strong>Quick View</strong>.</li>
                  <li>Scan the 29 pre-loaded live news and social signals to observe headline sentiment and impact deciles.</li>
                  <li>In <strong>Index Lab</strong>, click <code>Run Signal-Based Rebalance</code> for immediate weight recommendations.</li>
                </ol>
              </div>

              <button
                type="button"
                className="landing-btn-secondary"
                style={{ width: '100%', justifyContent: 'center' }}
                onClick={() => {
                  onMode?.('quick');
                  onTab('events');
                }}
              >
                <span>Launch Quick View Mode →</span>
              </button>
            </div>

            {/* Analyst View Card */}
            <div className="mode-card mode-analyst-card">
              <div className="mode-card-header">
                <span className="mode-card-badge badge-analyst">QUANTITATIVE &amp; AUDITOR PERSPECTIVE</span>
                <h3>Analyst View</h3>
                <span className="mode-card-sub">Full mathematical inspectability, probability vectors &amp; interactive shock levers</span>
              </div>

              <div className="mode-representation">
                <strong>What it represents:</strong> Built for Quantitative Analysts, Risk Engineers, and Portfolio Managers who demand mathematical transparency, inspectable model confidence distributions, and direct control over risk parameters.
              </div>

              <ul className="mode-features-list">
                <li>
                  <span className="mode-bullet">✓</span>
                  <div><strong>5-Class Softmax Probabilities:</strong> Complete event taxonomy distribution bars (Geopolitical, Macroeconomic, Credit Event, Merger / Acquisition, Product Launch) showing exact model entropy.</div>
                </li>
                <li>
                  <span className="mode-bullet">✓</span>
                  <div><strong>Granular Feature Decomposition:</strong> Unveils continuous sentiment scores [−1.0 to +1.0], 20-day rolling volatility (σ₂₀), and market regime state flags.</div>
                </li>
                <li>
                  <span className="mode-bullet">✓</span>
                  <div><strong>Mathematical Constraint Inspectability:</strong> Inspectable ±2% stock deviation, ±5% sector deviation, and 10% maximum turnover budget checks.</div>
                </li>
                <li>
                  <span className="mode-bullet">✓</span>
                  <div><strong>Multi-Asset Accounting Separation:</strong> Discrete valuation for Trading Book MTM (Bonds duration/convexity, Swaps DV01, Options European Black-Scholes) vs Banking Book 12-Month ECL.</div>
                </li>
                <li>
                  <span className="mode-bullet">✓</span>
                  <div><strong>What-If &amp; Reverse Stress Tools:</strong> Interactive position-level exposure sliders (0.0× to 2.0×) and grid-based search for the first sampled loss-budget breach.</div>
                </li>
              </ul>

              <div className="mode-testing-box">
                <h4>How to test Analyst View in the dashboard:</h4>
                <ol>
                  <li>Click the button below to launch the dashboard in <strong>Analyst View</strong>.</li>
                  <li>In <strong>AI Signals</strong>, inspect the full 5-category probability distributions on event cards.</li>
                  <li>In <strong>Stress Lab</strong>, adjust position sliders in the <code>What-If Analysis</code> panel or execute a <code>Reverse Stress Test</code>.</li>
                </ol>
              </div>

              <button
                type="button"
                className="landing-btn-primary"
                style={{ width: '100%', justifyContent: 'center' }}
                onClick={() => {
                  onMode?.('analyst');
                  onTab('events');
                }}
              >
                <span>Launch Analyst View Mode →</span>
              </button>
            </div>
          </div>

          {/* Shared State Guarantee Banner */}
          <div className="shared-state-banner">
            <ShieldCheck size={26} color="#10b981" style={{ flexShrink: 0 }} />
            <div>
              <strong>Synchronized State Guarantee</strong>
              <p>
                Quick View and Analyst View are <em>not</em> separate mock pages or disconnected tabs—they share the exact same underlying React state and live FastAPI data. You can select a signal or run a rebalance in Quick View, toggle to Analyst View, and inspect the underlying mathematical breakdown without losing your place or restarting your session.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ──────────────────────────────────────────────────────────── */}
      {/* SECTION 7: INTERACTIVE ILLUSTRATIVE WALKTHROUGH              */}
      {/* ──────────────────────────────────────────────────────────── */}
      <section className="landing-section bg-alt" id="interactive-demo">
        <div className="landing-container">
          <div className="section-eyebrow">07 / GUIDED DEMONSTRATION</div>
          <h2 className="section-heading">
            Interactive Walkthrough: The Information Journey
          </h2>

          <p className="section-lead">
            Follow how a single financial narrative moves through the AI pipeline into downstream decision support.
          </p>

          {/* Example Selector */}
          <div className="example-toggle-bar">
            <span>Choose illustrative example:</span>
            <div className="toggle-group">
              <button
                type="button"
                className={`toggle-btn ${selectedExample === 'aapl' ? 'active' : ''}`}
                onClick={() => { setSelectedExample('aapl'); setLiveResult(null); setActiveStep(1); }}
              >
                Apple (AAPL) · Product Launch
              </button>
              <button
                type="button"
                className={`toggle-btn ${selectedExample === 'msft' ? 'active' : ''}`}
                onClick={() => { setSelectedExample('msft'); setLiveResult(null); setActiveStep(1); }}
              >
                Microsoft (MSFT) · Cloud Infrastructure &amp; CapEx
              </button>
            </div>
          </div>

          {/* Step Stepper */}
          <div className="walkthrough-stepper">
            <button
              type="button"
              className={`step-nav-btn ${activeStep === 1 ? 'active' : ''}`}
              onClick={() => setActiveStep(1)}
            >
              1. Narrative Ingestion
            </button>
            <button
              type="button"
              className={`step-nav-btn ${activeStep === 2 ? 'active' : ''}`}
              onClick={() => setActiveStep(2)}
            >
              2. Language Models
            </button>
            <button
              type="button"
              className={`step-nav-btn ${activeStep === 3 ? 'active' : ''}`}
              onClick={() => setActiveStep(3)}
            >
              3. Impact Severity
            </button>
            <button
              type="button"
              className={`step-nav-btn ${activeStep === 4 ? 'active' : ''}`}
              onClick={() => setActiveStep(4)}
            >
              4. Portfolio Decisions
            </button>
          </div>

          {/* Walkthrough Stage Card */}
          <div className="walkthrough-display">
            {activeStep === 1 && (
              <div className="stage-content animate-fade">
                <div className="stage-header">
                  <span className="stage-tag">STEP 1 OF 4</span>
                  <h3>Unstructured Financial Narrative Arrives</h3>
                </div>
                <div className="narrative-box">
                  <div className="narrative-meta">
                    <span>Source: Public Financial Ingestion</span>
                    <span>Ticker: <strong>{selectedExample === 'aapl' ? 'AAPL' : 'MSFT'}</strong></span>
                    <span>Timestamp: Live Calendar Date UTC</span>
                  </div>
                  <p className="narrative-text">
                    {selectedExample === 'aapl'
                      ? "“A public financial article discusses an upcoming Apple product launch, detailing expanded hardware ecosystem subscriptions and silicon architecture upgrades.”"
                      : "“Financial market coverage analyzes Microsoft’s major expansion of enterprise cloud compute clusters and datacenter infrastructure to meet commercial AI demand.”"
                    }
                  </p>
                </div>
                <div className="stage-actions">
                  <button type="button" className="landing-btn-secondary" onClick={() => setActiveStep(2)}>
                    Next: Run FinBERT Analysis →
                  </button>
                  <button
                    type="button"
                    className="landing-btn-primary"
                    onClick={handleRunLiveInference}
                    disabled={liveAnalyzing}
                  >
                    {liveAnalyzing ? "Querying Backend..." : "Run Live Inference on This Event"}
                  </button>
                </div>
              </div>
            )}

            {liveError && <p role="alert" style={{color:'#a63721',fontWeight:600}}>{liveError}</p>}
            {activeStep === 2 && (
              <div className="stage-content animate-fade">
                <div className="stage-header">
                  <span className="stage-tag">STEP 2 OF 4</span>
                  <h3>Language Models Extract Sentiment &amp; Event Class</h3>
                </div>
                <div className="nlp-results-grid">
                  <div className="nlp-res-card">
                    <span className="nlp-label">FinBERT Continuous Sentiment</span>
                    <strong className="nlp-val text-pos">
                      {liveResult ? `${liveResult.sentiment >= 0 ? '+' : ''}${liveResult.sentiment.toFixed(2)}` : 'Unavailable'}
                    </strong>
                    <p>Continuous scale from −1.0 (strongly negative) to +1.0 (strongly positive).</p>
                  </div>

                  <div className="nlp-res-card">
                    <span className="nlp-label">FinBERT Event Classification</span>
                    <strong className="nlp-val">
                      {liveResult ? liveResult.eventClass : 'Unavailable'}
                    </strong>
                    <p>Categorized from 5 supported classes with full probability vector.</p>
                  </div>
                </div>
                <div className="stage-actions">
                  <button type="button" className="landing-btn-secondary" onClick={() => setActiveStep(3)}>
                    Next: Compute Impact Severity →
                  </button>
                </div>
              </div>
            )}

            {activeStep === 3 && (
              <div className="stage-content animate-fade">
                <div className="stage-header">
                  <span className="stage-tag">STEP 3 OF 4</span>
                  <h3>XGBoost Combines NLP Signals with Market Volatility</h3>
                </div>
                <div className="impact-display-card">
                  <div className="impact-score-big">
                    <span>Impact Severity Rating:</span>
                    <strong>{liveResult ? `${liveResult.impact} / 10` : 'Unavailable'}</strong>
                  </div>
                  <p>
                    The gradient-boosted engine evaluates the sentiment score, class probabilities, and pre-event
                    20-day historical stock volatility to produce a calibrated market disruption score.
                  </p>
                </div>
                <div className="stage-actions">
                  <button type="button" className="landing-btn-secondary" onClick={() => setActiveStep(4)}>
                    Next: See Portfolio Application →
                  </button>
                </div>
              </div>
            )}

            {activeStep === 4 && (
              <div className="stage-content animate-fade">
                <div className="stage-header">
                  <span className="stage-tag">STEP 4 OF 4</span>
                  <h3>Structured Signal Applied to Portfolio Research</h3>
                </div>
                <div className="dual-app-preview">
                  <div className="app-preview-card">
                    <h4>Module A Application</h4>
                    <p>
                      The positive sentiment and 7/10 severity for {selectedExample === 'aapl' ? 'AAPL' : 'MSFT'}
                      increases the constituent's tactical attractiveness, proposing a constrained +1.4% allocation tilt
                      subject to sector caps and 10% turnover limits.
                    </p>
                    <button type="button" className="text-link-btn" onClick={() => onTab('moduleA')}>
                      Open in Index Lab →
                    </button>
                  </div>

                  <div className="app-preview-card">
                    <h4>Module B Application</h4>
                    <p>
                      The event profile can generate an illustrative scenario proposal in the Stress Lab, allowing analysts
                      to review hypothetical market shocks before simulating portfolio ECL and MTM effects.
                    </p>
                    <button type="button" className="text-link-btn" onClick={() => onTab('moduleB')}>
                      Open in Stress Lab →
                    </button>
                  </div>
                </div>
                <div className="stage-actions">
                  <button type="button" className="landing-btn-primary" onClick={onStart}>
                    Launch Full Dashboard Now →
                  </button>
                </div>
              </div>
            )}
          </div>

          <p className="walkthrough-disclaimer">
            *Illustrative demonstration. These models provide structured decision support and do not guarantee profitable trading or predict exact future stock prices.
          </p>
        </div>
      </section>

      {/* ──────────────────────────────────────────────────────────── */}
      {/* SECTION 8: PLATFORM AT A GLANCE & CLOSING HERO               */}
      {/* ──────────────────────────────────────────────────────────── */}
      <section className="landing-section bg-alt" id="platform-summary">
        <div className="landing-container text-center">
          <div className="section-eyebrow text-center">08 / ARCHITECTURE AT A GLANCE</div>
          <h2 className="section-heading text-center">
            From information to interpretable risk intelligence.
          </h2>

          <p className="section-lead text-center" style={{ margin: '0 auto 40px auto' }}>
            Explore the signals. Examine the assumptions. Understand the financial implications.
          </p>

          {/* Simple Pipeline Summary Flow */}
          <div className="summary-pipeline" aria-label="End to End Flow Diagram">
            <div className="pipe-node">
              <span className="pipe-step">INPUT</span>
              <strong>Financial Information</strong>
              <small>News &amp; public discourse</small>
            </div>
            <div className="pipe-connector">↓</div>
            <div className="pipe-node pipe-node-core">
              <span className="pipe-step">AI RISK ENGINE</span>
              <strong>FinBERT Sentiment + Event Classification + XGBoost Impact</strong>
              <small>3 cooperating models · Calibrated baselines</small>
            </div>
            <div className="pipe-connector">↓</div>
            <div className="pipe-node">
              <span className="pipe-step">OUTPUT</span>
              <strong>Structured Risk Signals</strong>
              <small>Transparent mathematical representation</small>
            </div>
            <div className="pipe-connector">↓</div>
            <div className="pipe-node pipe-node-apps">
              <span className="pipe-step">RESEARCH WORKFLOWS</span>
              <strong>Constrained Index Research + Synthetic Portfolio Stress Testing</strong>
              <small>Module A &amp; Module B</small>
            </div>
          </div>

          <div className="closing-actions">
            <button
              type="button"
              className="landing-btn-primary btn-large"
              onClick={onStart}
            >
              <span>Launch Soochana AI Dashboard</span>
              <ArrowRight size={18} />
            </button>
            <a href="#ai-engine" className="landing-link-subtle">
              Explore Technical Methodology
            </a>
          </div>

          {/* Architecture Capabilities Reference Table */}
          <div className="architecture-reference-box">
            <h4>System Capabilities &amp; Architecture Matrix</h4>
            <div className="arch-table-grid">
              <div>
                <h5>Core technical requirements</h5>
                <ul>
                  <li>Two types of public information streams (Marketaux news &amp; Bluesky social)</li>
                  <li>Continuous sentiment score in the range [−1.0 to +1.0]</li>
                  <li>Five-category fine-tuned event classification with probability vector</li>
                  <li>Calibrated 1 to 10 market impact severity rating</li>
                  <li>Machine-readable, validated REST API schemas</li>
                </ul>
              </div>
              <div>
                <h5>Additional implementation</h5>
                <ul>
                  <li>Source provenance, deduplication and temporal lookahead prevention</li>
                  <li>Two integrated downstream financial portfolio research modules</li>
                  <li>Constrained 15-stock tactical research index optimization</li>
                  <li>Multi-asset scenario stress testing (Loans, Bonds, Swaps, Options)</li>
                  <li>Position-level What-If sensitivity and Reverse Stress limit search</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ──────────────────────────────────────────────────────────── */}
      {/* EDITORIAL FOOTER                                             */}
      {/* ──────────────────────────────────────────────────────────── */}
      <footer className="landing-footer">
        <div className="landing-footer-inner">
          <div className="footer-brand">
            <Activity size={20} />
            <span>Soochana <strong>AI</strong></span>
          </div>
          <div className="footer-links">
            <a href="#overview">Overview</a>
            <a href="#how-it-works">How It Works</a>
            <a href="#ai-engine">AI Engine</a>
            <a href="#applications">Applications</a>
            <a href="#view-modes">View Modes</a>
            <button type="button" className="footer-btn-link" onClick={onStart}>Dashboard</button>
          </div>
          {/* About Developer Section */}
          <div className="footer-developer-card">
            <div className="footer-dev-left">
              <div className="footer-dev-avatar">
                <User size={18} />
              </div>
              <div className="footer-dev-info">
                <span className="footer-dev-name">Developed by Shanmukh Datta</span>
                <span className="footer-dev-sub">National Institute of Technology (NIT), Jalandhar · Financial Risk Intelligence</span>
              </div>
            </div>
            <div className="footer-dev-links">
              <a
                href="https://github.com/shanmukhdatta"
                target="_blank"
                rel="noopener noreferrer"
                className="footer-github-btn"
              >
                <Github size={15} />
                <span>GitHub Profile</span>
                <ExternalLink size={12} />
              </a>
              <a
                href="https://github.com/shanmukhdatta/Soochana-AI"
                target="_blank"
                rel="noopener noreferrer"
                className="footer-repo-link"
              >
                <span>Source Repository</span>
                <ArrowUpRight size={13} />
              </a>
            </div>
          </div>

          <div className="footer-disclaimer">
            <p>
              Soochana AI · Financial risk research and explainable intelligence prototype.
              Public information and synthetic portfolios only. No brokerage trading, price guarantees,
              or investment advice.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
