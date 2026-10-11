import { useEffect, useMemo, useState } from 'react';
import { apiGet, type Overview, type Stock, type StockDetail, type Performance, type Replay } from './api';

const number = (v: number | null | undefined, digits = 2) => v == null || !Number.isFinite(v) ? '—' : v.toFixed(digits);
const percent = (v: number | null | undefined, digits = 2) => v == null || !Number.isFinite(v) ? '—' : `${v > 0 ? '+' : ''}${v.toFixed(digits)}%`;
const signed = (v: number | null | undefined, digits = 2) => v == null || !Number.isFinite(v) ? '—' : `${v > 0 ? '+' : ''}${v.toFixed(digits)}`;
const tone = (v: number | null | undefined) => v == null ? 'dark' : v < 0 ? 'negative' : 'positive';
const dateLabel = (value: string | null | undefined) => value ? value.slice(0, 10) : 'Unavailable';
const scrollToId = (id: string) => window.setTimeout(() => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' }), 0);

function Badge({ children, blush = false }: { children: React.ReactNode; blush?: boolean }) {
  return <span className={blush ? 'badge blush' : 'badge'}>{children}</span>;
}
function Metric({ label, value, note, toneClass = 'dark' }: { label: string; value: string; note: string; toneClass?: string }) {
  return <div className="metric"><span className="eyebrow">{label}</span><strong className={`mono ${toneClass}`}>{value}</strong><small>{note}</small></div>;
}
function Header({ onHome }: { onHome: () => void }) {
  const go = (id: string) => { onHome(); scrollToId(id); };
  return <header className="site-header">
    <button className="logo" onClick={onHome} aria-label="ReplicaVire home"><span className="brand-mark" aria-hidden="true"><i /><i /><i /></span><span className="wordmark">ReplicaVire</span><span className="descriptor">EQUITY RESEARCH</span></button>
    <nav className="desktop-nav" aria-label="Main navigation">
      <button onClick={onHome}>Dashboard</button><button onClick={() => go('screener')}>Stock Screener</button>
      <button onClick={() => go('performance')}>Model Performance</button><button onClick={() => go('methodology')}>Methodology</button>
      <button onClick={() => go('about')}>About</button>
    </nav>
    <div className="header-actions"><button className="button dark launch" onClick={() => go('screener')}>Launch Dashboard</button></div>
  </header>;
}
function TrendChart({ values, labels, title, height = 180 }: { values: (number | null)[]; labels?: string[]; title: string; height?: number }) {
  const valid = values.map((v, i) => v === null || !Number.isFinite(v) ? null : { i, v }).filter((p): p is { i: number; v: number } => p !== null);
  if (valid.length < 2) return <div className="no-chart">No time-series data available</div>;
  const min = Math.min(...valid.map(p => p.v));
  const max = Math.max(...valid.map(p => p.v));
  const scale = max - min || 1;
  const denom = Math.max(1, values.length - 1);
  const points = valid.map(p => `${20 + 560 * p.i / denom},${135 - 108 * (p.v - min) / scale}`).join(' ');
  return <div className="live-chart" role="img" aria-label={`${title}, ${valid.length} values`}>
    <svg viewBox="0 0 600 170" preserveAspectRatio="none" style={{ height }}>
      {[35, 75, 115, 145].map(y => <line key={y} x1="20" x2="580" y1={y} y2={y} stroke="#d9ddd5" strokeWidth="1" />)}
      <polyline points={points} fill="none" stroke="#23684f" strokeWidth="2.8" strokeLinejoin="round" strokeLinecap="round" />
      <text x="20" y="18" fontSize="11" fill="#69716c">{max.toFixed(2)}</text>
      <text x="20" y="158" fontSize="11" fill="#69716c">{min.toFixed(2)}</text>
    </svg>
    {labels && labels.length > 1 && <div className="chart-labels"><span>{labels[0]?.slice(0, 10)}</span><span>{labels[labels.length - 1]?.slice(0, 10)}</span></div>}
  </div>;
}
function MiniTrend({ stock }: { stock: Stock }) {
  const values = stock.price_history.map(p => p.close);
  return <div className="mini-trend"><TrendChart values={values} title={`${stock.ticker} last ${values.length} trading closes`} height={44} /></div>;
}
function Hero({ overview, connected }: { overview: Overview | null; connected: boolean }) {
  const available = overview?.stocks.filter(s => s.prediction_date).length ?? 0;
  const withNews = overview?.stocks.filter(s => s.news_count_5d !== null).length ?? 0;
  return <section className="hero">
    <div className="hero-copy"><Badge>AI-POWERED FINANCIAL NEWS RESEARCH</Badge><h1>Understand the sentiment. Predict the movement.</h1>
      <p>Explore real market-price history, financial news sentiment, and Ridge versus neural-network estimates of five-day returns relative to SPY.</p>
      <div className="actions"><button className="button primary" onClick={() => scrollToId('screener')}>Explore Stocks</button><button className="button secondary" onClick={() => scrollToId('methodology')}>How It Works</button></div>
      <div className="hero-stats">
        <Metric label="WATCHED STOCKS" value={overview ? String(overview.stocks.length) : '—'} note="From backend config.yaml" />
        <Metric label="MODEL PAIRS" value={String(available)} note="Latest date-matched predictions" />
        <Metric label="NEWS FEATURES" value={String(withNews)} note="Tracked stocks with sentiment features" />
      </div>
    </div>
    <div className="panel preview">
      <div className="section-row"><div><span className="eyebrow">RESEARCH CONNECTION</span><h3>Backend data status</h3></div><Badge blush={!connected}>{connected ? 'TIGERDATA FEED' : 'NOT CONNECTED'}</Badge></div>
      <div className="preview-chart"><span className="eyebrow">CURRENT DATA PIPELINE</span>
        <p>Alpaca prices + Finnhub/Alpaca financial news → TigerData → Ridge + MLP → ReplicaVire</p>
        <p className="fine-print">The backend currently predicts five-day <strong>excess return versus SPY</strong>, not a stock's absolute price return. The original Google Play review snapshots aren't integrated here yet.</p>
      </div>
      <div className="three-grid"><Metric label="PRICE SOURCE" value="Alpaca" note="Stored daily bars" /><Metric label="SENTIMENT" value="News" note="VADER / optional Cortex" /><Metric label="BENCHMARK" value="SPY" note="Excess-return target" /></div>
    </div>
  </section>;
}

type SortField = 'ticker' | 'price' | 'sentiment' | 'mean_excess_pct' | 'spread_pp';
function Screener({ stocks, onSelect, loading }: { stocks: Stock[]; onSelect: (s: Stock) => void; loading: boolean }) {
  const [query, setQuery] = useState('');
  const [sector, setSector] = useState('All');
  const [sentiment, setSentiment] = useState('All');
  const [order, setOrder] = useState<SortField>('mean_excess_pct');
  const [descending, setDescending] = useState(true);
  const sectors = ['All', ...new Set(stocks.map(s => s.sector))];
  const visible = useMemo(() => stocks.filter(s => `${s.ticker} ${s.company}`.toLowerCase().includes(query.toLowerCase()))
    .filter(s => sector === 'All' || s.sector === sector)
    .filter(s => sentiment === 'All' || (s.sentiment !== null && (sentiment === 'Positive' ? s.sentiment > 0 : s.sentiment < 0)))
    .sort((a, b) => {
      const av = a[order], bv = b[order];
      if (av == null && bv == null) return a.ticker.localeCompare(b.ticker);
      if (av == null) return 1;
      if (bv == null) return -1;
      const comparison = typeof av === 'string' ? av.localeCompare(String(bv)) : Number(av) - Number(bv);
      return descending ? -comparison : comparison;
    }), [stocks, query, sector, sentiment, order, descending]);
  return <section className="page-section" id="screener">
    <div className="section-heading"><div><span className="eyebrow">01 / LIVE RESEARCH SCREENER</span><h2>Stock Screener</h2></div><small>{visible.length} stocks · News sentiment · Five-day excess-return predictions</small></div>
    <div className="filters">
      <label className="search"><img src="/assets/c647e.svg" alt="" width="16" height="16" /><input value={query} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setQuery(e.target.value)} placeholder="Search ticker or company" /></label>
      <select aria-label="Sector filter" value={sector} onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setSector(e.target.value)}>{sectors.map(s => <option key={s}>{s}</option>)}</select>
      <select aria-label="Sentiment filter" value={sentiment} onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setSentiment(e.target.value)}><option>All</option><option>Positive</option><option>Negative</option></select>
      <select aria-label="Sort by" value={order} onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setOrder(e.target.value as SortField)}><option value="mean_excess_pct">Model mean</option><option value="sentiment">News sentiment</option><option value="price">Price</option><option value="spread_pp">Model spread</option><option value="ticker">Ticker</option></select>
      <button className="sort-button" onClick={() => setDescending(!descending)}>{descending ? '↓ DESC' : '↑ ASC'}</button>
    </div>
    <div className="table-wrap">{loading ? <div className="empty-state">Loading real data…</div> : visible.length === 0 ? <div className="empty-state"><h3>No stocks available</h3><p>Check the API connection or adjust your search and filters. No demo prices or predictions are substituted.</p></div> : <table>
      <thead><tr><th>TICKER / COMPANY</th><th>LAST CLOSE</th><th>1D</th><th>NEWS SENTIMENT (5D)</th><th>NEWS COUNT (5D)</th><th>MLP EXCESS 5D</th><th>RIDGE EXCESS 5D</th><th>MEAN EXCESS 5D</th><th>SPREAD</th><th>PRICE TREND</th></tr></thead>
      <tbody>{visible.map(s => <tr key={s.ticker} tabIndex={0} role="button" onKeyDown={(e: React.KeyboardEvent<HTMLTableRowElement>) => (e.key === 'Enter' || e.key === ' ') && onSelect(s)} onClick={() => onSelect(s)}>
        <td><b>{s.ticker}</b><small>{s.company}</small></td><td className="mono">{s.price == null ? '—' : `$${number(s.price)}`}<small>{dateLabel(s.price_date)}</small></td>
        <td className={`mono ${tone(s.day_change_pct)}`}>{percent(s.day_change_pct)}</td>
        <td className={`mono ${tone(s.sentiment)}`}>{signed(s.sentiment, 3)}</td><td className="mono">{s.news_count_5d ?? '—'}</td>
        <td className={`mono ${tone(s.mlp_excess_pct)}`}>{percent(s.mlp_excess_pct)}</td>
        <td className={`mono ${tone(s.ridge_excess_pct)}`}>{percent(s.ridge_excess_pct)}</td>
        <td className={`mono strong ${tone(s.mean_excess_pct)}`}>{percent(s.mean_excess_pct)}</td><td className="mono">{s.spread_pp == null ? '—' : `${number(s.spread_pp)}pp`}</td><td><MiniTrend stock={s} /></td>
      </tr>)}</tbody>
    </table>}</div>
    <p className="fine-print">Values come from TigerData when connected. Missing values remain unavailable. Model numbers are excess-return estimates relative to SPY, not expected stock returns. The mean is an unweighted average of the two estimates, not a calibrated ensemble.</p>
  </section>;
}
function ModelPerformance() {
  const [data, setData] = useState<Performance | null>(null);
  const [error, setError] = useState('');
  useEffect(() => { const ctrl = new AbortController(); apiGet<Performance>('/api/performance', ctrl.signal).then(setData).catch(e => { if (!ctrl.signal.aborted) setError(e.message); }); return () => ctrl.abort(); }, []);
  return <section className="page-section" id="performance"><div className="section-heading"><div><span className="eyebrow">02 / VERIFIED TRAINING OUTPUTS</span><h2>Model Performance</h2></div><Badge blush>{data?.status === 'available' ? 'TEST RESULTS' : 'RESULTS UNAVAILABLE'}</Badge></div>
    {data?.status === 'available' ? <><div className="model-grid">{data.models.map(m => <article className="panel model-card" key={m.name}><div className="section-row"><h3>{m.name}</h3><Badge>HELD-OUT TEST</Badge></div>
      <div className="metric-grid"><Metric label="DIRECTIONAL HIT RATE" value={m.hit_rate == null ? '—' : percent(100 * m.hit_rate, 1)} note="Sign of five-day excess return" toneClass="positive" />
        <Metric label="MEAN SPEARMAN IC" value={number(m.ic_mean, 3)} note="Rank correlation per day" />
        <Metric label="EVALUATED ROWS" value={m.n_rows?.toLocaleString() ?? '—'} note="Held-out predictions" />
        <Metric label="IC DAYS" value={m.n_days?.toString() ?? '—'} note="Dates in correlation summary" /></div>
    </article>)}</div>
      {data.curve && data.curve.length > 1 && <div className="model-grid"><div className="panel"><span className="eyebrow">COST-AWARE AGREEMENT STRATEGY · CUMULATIVE</span><TrendChart values={data.curve.map(p => p.agreement * 100)} labels={data.curve.map(p => p.date)} title="Agreement strategy cumulative return in percent" height={200} /></div>
      <div className="panel"><span className="eyebrow">SPY BENCHMARK · CUMULATIVE</span><TrendChart values={data.curve.map(p => p.spy * 100)} labels={data.curve.map(p => p.date)} title="SPY cumulative return in percent" height={200} /></div></div>}
      <p className="fine-print">Training: {data.meta?.train_dates?.join(' → ') ?? '—'} · Test: {data.meta?.test_dates?.join(' → ') ?? '—'}. {data.meta?.caveat ?? ''}</p></>
    : <div className="panel empty-state"><h3>Real backtest results not available yet</h3><p>{error || data?.reason || 'Loading evaluation data…'}</p><p>Run <code>python -m src.train</code> to generate <code>results/metrics.json</code>. We will not show fabricated accuracy percentages.</p></div>}
  </section>;
}
function HistoricalReplay({ ticker }: { ticker: string }) {
  const [day, setDay] = useState(new Date().toISOString().slice(0, 10));
  const [data, setData] = useState<Replay | null>(null);
  const [error, setError] = useState('');
  useEffect(() => { const ctrl = new AbortController(); setData(null); setError('');
    apiGet<Replay>(`/api/replay?symbol=${encodeURIComponent(ticker)}&selected_date=${encodeURIComponent(day)}`, ctrl.signal)
      .then(setData).catch(e => { if (!ctrl.signal.aborted) setError(e.message); });
    return () => ctrl.abort();
  }, [ticker, day]);
  return <section className="panel replay"><div className="section-row"><div><span className="eyebrow">HISTORICAL REPLAY · {ticker}</span><h3>What was knowable on that date?</h3></div><Badge blush>{data?.status === 'retrospective_only' ? 'RETROSPECTIVE ONLY' : 'NOT VERIFIED'}</Badge></div>
    <div className="replay-grid"><div className="calendar-card"><label className="eyebrow" htmlFor="replay-date">Decision date</label><input id="replay-date" type="date" value={day} max={new Date().toISOString().slice(0, 10)} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setDay(e.target.value)} /></div>
      <div className="replay-result"><div className="three-grid"><Metric label="SELECTED DATE" value={day} note="End-of-day decision" /><Metric label="LAST STORED FORECAST" value={dateLabel(data?.latest_research_date)} note="Not point-in-time verified" /><Metric label="REALIZED RETURN" value="—" note="Not shown without valid replay" /></div>
        <p>{error || data?.reason || 'Checking historical availability…'}</p></div></div>
    <p className="fine-print">No historical date is marked verified because the existing pipeline does not preserve fully immutable feature, ingestion, and model-training snapshots for an auditable point-in-time forecast.</p>
  </section>;
}
function Methodology() {
  const steps = [
    ['01', 'Collect', 'Alpaca adjusted bars and published financial news', '/assets/fbd86.svg'],
    ['02', 'Score', 'News sentiment using VADER or optional Snowflake Cortex', '/assets/3e265.svg'],
    ['03', 'Train', 'TigerData features, Ridge regression, and MLP models', '/assets/969c9.svg'],
    ['04', 'Compare', 'Five-day excess return versus SPY and honest held-out metrics', '/assets/4dcef.svg'],
  ];
  return <section className="methodology" id="methodology"><span className="eyebrow">03 / METHODOLOGY</span>
    <div className="method-intro"><h2>A clear path<br />from insight to outlook.</h2><p>Readable model research with transparent inputs, real source dates, and cautious interpretation.</p></div>
    <div className="process-grid">{steps.map(([num, title, desc, icon]) => <article key={num}><span className="step">{num}</span><div className="method-icon"><img src={icon} alt="" /></div><h3>{title}</h3><p>{desc}</p></article>)}</div>
    <p className="method-note">The original Duolingo Google Play review collector is separate from this financial-news pipeline. A review-specific integration is future work and is not represented as live.</p>
  </section>;
}
function About() {
  return <section className="page-section about" id="about"><span className="eyebrow">04 / ABOUT REPLICAVIRE</span><h2>Built for the curious investor.</h2>
    <p>ReplicaVire organizes financial news signals and model outputs to help compare research ideas. Forecasts are estimates and should never be treated as guaranteed outcomes.</p>
    <div className="pricing"><article className="panel"><span className="eyebrow">FREE RESEARCH</span><h3>Research dashboard</h3><strong>$0 / month</strong><Badge blush>PROPOSED PLAN</Badge><p>Tracked stocks, sentiment scores, and research methodology.</p></article>
      <article className="panel pro"><span className="eyebrow">PRO</span><h3>Advanced research</h3><strong>$29 / month</strong><Badge blush>HYPOTHETICAL</Badge><p>Proposed historical diagnostics and deeper model comparisons. Payments are not implemented.</p></article></div>
  </section>;
}
function StockAnalysis({ ticker, onBack }: { ticker: string; onBack: () => void }) {
  const [detail, setDetail] = useState<StockDetail | null>(null);
  const [error, setError] = useState('');
  const [range, setRange] = useState(30);
  useEffect(() => { const ctrl = new AbortController(); setDetail(null); setError('');
    apiGet<StockDetail>(`/api/stocks/${ticker}`, ctrl.signal).then(setDetail).catch(e => { if (!ctrl.signal.aborted) setError(e.message); }); return () => ctrl.abort();
  }, [ticker]);
  if (error || !detail) return <main className="analysis-page"><div className="analysis-bar"><button className="back" onClick={onBack}>← Back to screener</button></div><div className="panel placeholder-page">{error || 'Loading stock details from TigerData…'}</div></main>;
  const s = detail.stock;
  const history = detail.history.slice(-range).filter(x => x.close != null);
  const sent = detail.sentiment_history.slice(-range);
  return <main className="analysis-page">
    <div className="analysis-bar"><button className="back" onClick={onBack}>← Back to screener</button><span>{ticker} · Source: financial news + TigerData</span></div>
    <div className="disclosure"><Badge>REAL DATABASE VALUES</Badge><b>Predictions estimate excess return versus SPY, not absolute stock return. Blank fields mean unavailable.</b></div>
    <div className="analysis-layout"><div className="analysis-main">
      <section className="panel stock-summary"><div className="section-row"><div><span className="eyebrow">{ticker}</span><h2>{s.company}</h2></div><Badge blush={!s.prediction_date}>{s.prediction_date ? `MODEL AS OF ${dateLabel(s.prediction_date)}` : 'NO MATCHED FORECAST'}</Badge></div>
        <div className="four-grid"><Metric label="LAST STORED CLOSE" value={s.price == null ? '—' : `$${number(s.price)}`} note={dateLabel(s.price_date)} />
          <Metric label="NEWS SENTIMENT · 5D" value={signed(s.sentiment, 3)} note="Weighted by available scored articles" toneClass={tone(s.sentiment)} />
          <Metric label="NEWS SENTIMENT · 1D" value={signed(s.sentiment_1d, 3)} note="Backend feature" toneClass={tone(s.sentiment_1d)} />
          <Metric label="NEWS ITEMS · 5D" value={s.news_count_5d?.toString() ?? '—'} note="Latest feature window" /></div>
        <p className="fine-print">Sentiment features as of {dateLabel(s.feature_date)}. Source: Finnhub/Alpaca financial news, not Google Play reviews.</p>
      </section>
      <section className="panel"><div className="section-row"><span className="eyebrow">STORED STOCK PRICE HISTORY</span><div className="tabs">{[10, 30, 90, 150].map(n => <button className={n === range ? 'selected' : ''} onClick={() => setRange(n)} key={n}>{n}D</button>)}</div></div>
        <TrendChart values={history.map(p => p.close)} labels={history.map(p => p.ts)} title={`${ticker} historical daily closing prices`} height={220} /></section>
      <section className="panel"><span className="eyebrow">NEWS SENTIMENT TREND · FIVE-DAY AVERAGE</span>
        <TrendChart values={sent.map(p => p.sent_5d)} labels={sent.map(p => p.d)} title={`${ticker} five-day average news sentiment`} height={200} />
        <p className="fine-print">Sentiment uses the existing backend's five-day publication-time window.</p></section>
      <section className="panel reviews"><div className="section-row"><span className="eyebrow">RECENT FINANCIAL NEWS</span><small>{detail.articles.length} stored articles</small></div>
        {detail.articles.length ? detail.articles.slice(0, 12).map((a, i) => <article className="review" key={`${a.url}-${i}`}>
          <small>{dateLabel(a.event_time)} · {a.publisher || a.source_api}</small><Badge blush={a.sentiment != null && a.sentiment < 0}>{a.sentiment == null ? 'UNSCORED' : `Sentiment ${signed(a.sentiment)}`}</Badge>
          <p><a href={a.url} target="_blank" rel="noopener noreferrer">{a.headline || 'Open source article ↗'}</a></p>
          {a.summary && <small>{a.summary.slice(0, 270)}</small>}
        </article>) : <div className="empty-state">No stored articles for this stock yet.</div>}
      </section>
    </div>
    <aside className="analysis-aside">
      <section className="panel prediction-cards"><span className="eyebrow">5D EXCESS RETURN VS SPY</span>
        <Metric label="NEURAL NETWORK · MLP" value={percent(s.mlp_excess_pct)} note="Raw MLP estimate" toneClass={tone(s.mlp_excess_pct)} />
        <Metric label="RIDGE REGRESSION" value={percent(s.ridge_excess_pct)} note="Raw Ridge estimate" toneClass={tone(s.ridge_excess_pct)} />
        <Metric label="SIMPLE MEAN" value={percent(s.mean_excess_pct)} note="Unweighted mean, not a trained ensemble" toneClass={tone(s.mean_excess_pct)} /></section>
      <section className="panel"><span className="eyebrow">FORECAST DISAGREEMENT</span><h3>{s.spread_pp == null ? '—' : `${number(s.spread_pp)} percentage points`}</h3><p className="fine-print">Difference between the two excess-return estimates. This is not a confidence score.</p></section>
      <section className="panel"><span className="eyebrow">DATA PROVENANCE</span><p>Last price: {dateLabel(s.price_date)}</p><p>News feature: {dateLabel(s.feature_date)}</p><p>Prediction: {dateLabel(s.prediction_date)}</p><p className="fine-print">{detail.source_note}</p></section>
    </aside></div>
    <div className="analysis-replay"><HistoricalReplay ticker={ticker} /></div>
  </main>;
}
function Footer() {
  return <footer><div><span className="wordmark">ReplicaVire</span><span>Know the data. Understand the limits.</span></div><p>Research prototype, not investment advice. Forecasts and live-data status depend on the connected backend. Trading and paid subscriptions are not provided by this website.</p><small>© 2026 ReplicaVire · PALMETTO HACKS</small></footer>;
}
export default function App() {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [apiError, setApiError] = useState('');
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0);
  useEffect(() => { const ctrl = new AbortController(); setLoading(true); setApiError('');
    apiGet<Overview>('/api/overview', ctrl.signal).then(setOverview).catch(e => { if (!ctrl.signal.aborted) { setOverview(null); setApiError(e.message); } }).finally(() => { if (!ctrl.signal.aborted) setLoading(false); });
    return () => ctrl.abort();
  }, [refresh]);
  const goHome = () => { setSelected(null); window.scrollTo({ top: 0, behavior: 'smooth' }); };
  return <div className="app"><Header onHome={goHome} />
    {selected ? <StockAnalysis ticker={selected} onBack={() => { setSelected(null); scrollToId('screener'); }} /> : <main>
      {apiError && <div className="connection-error" role="alert"><b>Backend unavailable:</b> {apiError}. Start the Python API and verify TigerData is configured. <button className="button secondary" onClick={() => setRefresh(v => v + 1)}>Retry</button></div>}
      <Hero overview={overview} connected={!!overview} />
      <Screener stocks={overview?.stocks ?? []} loading={loading} onSelect={s => { setSelected(s.ticker); window.scrollTo(0, 0); }} />
      <ModelPerformance /><Methodology /><About />
    </main>}
    <Footer />
  </div>;
}
