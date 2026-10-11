import { useEffect, useMemo, useRef, useState } from 'react';
import { apiGet, apiSend, ApiError, type Overview, type Stock, type StockDetail, type Performance, type Replay, type Alert, type Settings, type Session, type DirectionMode, type Horizons, type Driver, type RegressorOutput, type SentimentEvent, type HorizonMetrics } from './api';

const HORIZON_ORDER = ['6h', '12h', '18h', '1d', '3d', '5d'];
const GICS_SECTORS =['Information Technology', 'Communication Services', 'Consumer Discretionary', 'Consumer Staples', 'Health Care', 'Financials', 'Energy', 'Industrials', 'Materials', 'Utilities', 'Real Estate'];
const money = (v: number | null | undefined) => v == null || !Number.isFinite(v) ? '—' : `$${Math.round(v).toLocaleString()}`;
const timeLabel = (value: string | null | undefined) => {
  if (!value) return '—';
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? value : d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
};
const arrowFor = (d: string | null | undefined) => d === 'long' ? '▲' : d === 'short' ? '▼' : '–';
const directionTone = (d: string | null | undefined) => d === 'long' ? 'positive' : d === 'short' ? 'negative' : 'dark';
const sellSide = (a: Alert) => a.action === 'SELL' || a.action === 'SHORT';

const number = (v: number | null | undefined, digits = 2) => v == null || !Number.isFinite(v) ? '—' : v.toFixed(digits);
const percent = (v: number | null | undefined, digits = 2) => v == null || !Number.isFinite(v) ? '—' : `${v > 0 ? '+' : ''}${v.toFixed(digits)}%`;
const signed = (v: number | null | undefined, digits = 2) => v == null || !Number.isFinite(v) ? '—' : `${v > 0 ? '+' : ''}${v.toFixed(digits)}`;
const tone = (v: number | null | undefined) => v == null ? 'dark' : v < 0 ? 'negative' : 'positive';
const dateLabel = (value: string | null | undefined) => value ? value.slice(0, 10) : 'Unavailable';
const dateTimeLabel = (value: string | null | undefined) => {
  if (!value) return 'Unavailable';
  const d = new Date(value);
  return value.length <= 10 || Number.isNaN(d.getTime()) ? value.slice(0, 10) : d.toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
};
const rate = (v: number | null | undefined, digits = 1) => v == null || !Number.isFinite(v) ? '—' : `${(100 * v).toFixed(digits)}%`;
const levelTone = (v: number | null | undefined) => v == null ? 'dark' : v > 50 ? 'positive' : v < 50 ? 'negative' : 'dark';
const trendLabel = (v: number | null | undefined) => v == null || !Number.isFinite(v) ? '' : `${v > 0 ? '▲' : v < 0 ? '▼' : '–'} ${Math.abs(v).toFixed(1)}`;
// Database-supplied links are only rendered when they are plain web URLs.
const safeUrl = (u: string | null | undefined) => u && /^https?:\/\//i.test(u) ? u : null;
const featureValue = (v: number | null | undefined) => v == null || !Number.isFinite(v) ? '—' : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(2);
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
      <button onClick={() => go('recommendations')}>Recommendations</button><button onClick={() => go('autotrading')}>Auto-Trading</button>
      <button onClick={() => go('performance')}>Model Performance</button><button onClick={() => go('methodology')}>Methodology</button>
      <button onClick={() => go('about')}>About</button>
    </nav>
    <div className="header-actions"><button className="button dark launch" onClick={() => go('screener')}>Launch Dashboard</button></div>
  </header>;
}
function TrendChart({ values, labels, title, height = 180, compare }: { values: (number | null)[]; labels?: string[]; title: string; height?: number; compare?: (number | null)[] }) {
  const toPoints = (series: (number | null)[]) => series.map((v, i) => v === null || !Number.isFinite(v) ? null : { i, v }).filter((p): p is { i: number; v: number } => p !== null);
  const valid = toPoints(values);
  const other = compare ? toPoints(compare) : [];
  if (valid.length < 2) return <div className="no-chart">No time-series data available</div>;
  const all = [...valid, ...other];
  const min = Math.min(...all.map(p => p.v));
  const max = Math.max(...all.map(p => p.v));
  const scale = max - min || 1;
  const denom = Math.max(1, values.length - 1);
  const line = (pts: { i: number; v: number }[]) => pts.map(p => `${20 + 560 * p.i / denom},${135 - 108 * (p.v - min) / scale}`).join(' ');
  const points = line(valid);
  return <div className="live-chart" role="img" aria-label={`${title}, ${valid.length} values`}>
    <svg viewBox="0 0 600 170" preserveAspectRatio="none" style={{ height }}>
      {[35, 75, 115, 145].map(y => <line key={y} x1="20" x2="580" y1={y} y2={y} stroke="#d9ddd5" strokeWidth="1" />)}
      {other.length > 1 && <polyline points={line(other)} fill="none" stroke="#69716c" strokeWidth="2" strokeDasharray="6 5" strokeLinejoin="round" strokeLinecap="round" />}
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
  const available = overview?.stocks.filter(s => s.horizons).length ?? 0;
  const withNews = overview?.stocks.filter(s => s.sentiment_level !== null).length ?? 0;
  return <section className="hero">
    <div className="hero-copy"><Badge>AI-POWERED FINANCIAL NEWS RESEARCH</Badge><h1>Understand the sentiment. Predict the movement.</h1>
      <p>Explore real market-price history, LLM-read news and SEC filings, and a fusion neural network's return forecasts for six horizons from six hours to five days.</p>
      <div className="actions"><button className="button primary" onClick={() => scrollToId('screener')}>Explore Stocks</button><button className="button secondary" onClick={() => scrollToId('methodology')}>How It Works</button></div>
      <div className="hero-stats">
        <Metric label="WATCHED STOCKS" value={overview ? String(overview.stocks.length) : '—'} note="From backend config.yaml" />
        <Metric label="LIVE FORECASTS" value={String(available)} note="Stocks with stored horizon forecasts" />
        <Metric label="SENTIMENT TRACKED" value={String(withNews)} note="Stocks with sentiment-agent events" />
      </div>
    </div>
    <div className="panel preview">
      <div className="section-row"><div><span className="eyebrow">RESEARCH CONNECTION</span><h3>Backend data status</h3></div><Badge blush={!connected}>{connected ? 'TIGERDATA FEED' : 'NOT CONNECTED'}</Badge></div>
      <div className="preview-chart"><span className="eyebrow">CURRENT DATA PIPELINE</span>
        <p>Alpaca 15-min bars + Finnhub/Alpaca news + SEC EDGAR filings + FINRA short volume → Tiger Data → 15 regressors + Snowflake Cortex sentiment agent → fusion neural network → 6 horizons</p>
        <p className="fine-print">Forecasts are a stock's <strong>raw return in percent</strong> (mean and uncertainty) for 6h, 12h, 18h, 1d, 3d and 5d; the decision layer picks one horizon per stock. The original Google Play review snapshots aren't integrated here yet.</p>
      </div>
      <div className="three-grid"><Metric label="PRICE SOURCE" value="Alpaca" note="15-minute and daily bars" /><Metric label="SENTIMENT" value="LLM" note="Snowflake Cortex agent" /><Metric label="HORIZONS" value="6" note="6h to 5d raw return" /></div>
    </div>
  </section>;
}

function SectorChips({ sectors, selected, onChange }: { sectors: string[]; selected: string[]; onChange: (next: string[]) => void }) {
  const toggle = (s: string) => onChange(selected.includes(s) ? selected.filter(x => x !== s) : [...selected, s]);
  return <div className="chips" role="group" aria-label="Sector filter">
    <button type="button" className={selected.length === 0 ? 'chip selected' : 'chip'} aria-pressed={selected.length === 0} onClick={() => onChange([])}>All</button>
    {sectors.map(s => <button type="button" key={s} className={selected.includes(s) ? 'chip selected' : 'chip'} aria-pressed={selected.includes(s)} onClick={() => toggle(s)}>{s}</button>)}
  </div>;
}
function Conviction({ height }: { height: number | null | undefined }) {
  if (height == null || !Number.isFinite(height)) return <span className="mono">—</span>;
  const width = Math.max(0, Math.min(height, 3)) / 3 * 100;
  return <div className="conviction" title={`${height.toFixed(2)}σ conviction`}><div className="conviction-bar"><i style={{ width: `${width}%` }} /></div><small className="mono">{height.toFixed(2)}σ</small></div>;
}

function SentimentLevel({ level, trend }: { level: number | null; trend: number | null }) {
  return <span className={`mono ${levelTone(level)}`}>{level == null ? '—' : level.toFixed(0)}{trend != null && <small className={`mono ${trend > 0 ? 'positive' : trend < 0 ? 'negative' : ''}`} title="Change in level over the last 24 hours">{trendLabel(trend)}</small>}</span>;
}
// Center-zero bar for signed quantities (driver attributions, regressor outputs).
function SignedBar({ value, scale }: { value: number | null; scale: number }) {
  if (value == null || !Number.isFinite(value) || !(scale > 0)) return <div className="signed-bar" />;
  const w = Math.min(Math.abs(value) / scale, 1) * 50;
  return <div className="signed-bar"><i className={value < 0 ? 'neg' : ''} style={{ left: value < 0 ? `${50 - w}%` : '50%', width: `${w}%` }} /></div>;
}

type SortField = 'ticker' | 'price' | 'day_change_pct' | 'sentiment_level' | 'sentiment_trend' | 'news_count_5d' | 'height' | 'expected_pct' | 'suggested_usd';
function Screener({ stocks, sectors, onSelect, loading }: { stocks: Stock[]; sectors: string[]; onSelect: (s: Stock) => void; loading: boolean }) {
  const [query, setQuery] = useState('');
  const [sector, setSector] = useState<string[]>([]);
  const [sentiment, setSentiment] = useState('All');
  const [order, setOrder] = useState<SortField>('height');
  const [descending, setDescending] = useState(true);
  const visible = useMemo(() => stocks.filter(s => `${s.ticker} ${s.company}`.toLowerCase().includes(query.toLowerCase()))
    .filter(s => sector.length === 0 || sector.includes(s.sector))
    .filter(s => sentiment === 'All'
      || (sentiment === 'Positive' && s.sentiment_level !== null && s.sentiment_level > 50)
      || (sentiment === 'Negative' && s.sentiment_level !== null && s.sentiment_level < 50)
      || (sentiment === 'Improving' && s.sentiment_trend !== null && s.sentiment_trend > 0)
      || (sentiment === 'Deteriorating' && s.sentiment_trend !== null && s.sentiment_trend < 0))
    .sort((a, b) => {
      const av = a[order], bv = b[order];
      if (av == null && bv == null) return a.ticker.localeCompare(b.ticker);
      if (av == null) return 1;
      if (bv == null) return -1;
      const comparison = typeof av === 'string' ? av.localeCompare(String(bv)) : Number(av) - Number(bv);
      return descending ? -comparison : comparison;
    }), [stocks, query, sector, sentiment, order, descending]);
  return <section className="page-section" id="screener">
    <div className="section-heading"><div><span className="eyebrow">01 / LIVE RESEARCH SCREENER</span><h2>Stock Screener</h2></div><small>{visible.length} stocks · LLM news sentiment · Fusion-network forecasts (6h–5d)</small></div>
    <div className="filters">
      <label className="search"><img src="/assets/c647e.svg" alt="" width="16" height="16" /><input value={query} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setQuery(e.target.value)} placeholder="Search ticker or company" /></label>
      <select aria-label="Sentiment filter" value={sentiment} onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setSentiment(e.target.value)}><option>All</option><option value="Positive">Positive (&gt; 50)</option><option value="Negative">Negative (&lt; 50)</option><option>Improving</option><option>Deteriorating</option></select>
      <select aria-label="Sort by" value={order} onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setOrder(e.target.value as SortField)}><option value="height">Conviction</option><option value="expected_pct">Expected move</option><option value="suggested_usd">Suggested $</option><option value="sentiment_level">Sentiment level</option><option value="sentiment_trend">Sentiment 24h change</option><option value="news_count_5d">News count (5D)</option><option value="day_change_pct">1D change</option><option value="price">Price</option><option value="ticker">Ticker</option></select>
      <button className="sort-button" onClick={() => setDescending(!descending)}>{descending ? '↓ DESC' : '↑ ASC'}</button>
    </div>
    <SectorChips sectors={sectors} selected={sector} onChange={setSector} />
    <div className="table-wrap">{loading ? <div className="empty-state">Loading real data…</div> : visible.length === 0 ? <div className="empty-state"><h3>No stocks available</h3><p>Check the API connection or adjust your search and filters. No demo prices or predictions are substituted.</p></div> : <table>
      <thead><tr><th>TICKER / COMPANY</th><th>LAST CLOSE</th><th>1D</th><th>SIGNAL</th><th>CONVICTION</th><th>SUGGESTED $</th><th>SENTIMENT (0–100)</th><th>NEWS COUNT (5D)</th><th>PRICE TREND</th></tr></thead>
      <tbody>{visible.map(s => <tr key={s.ticker} tabIndex={0} role="button" onKeyDown={(e: React.KeyboardEvent<HTMLTableRowElement>) => (e.key === 'Enter' || e.key === ' ') && onSelect(s)} onClick={() => onSelect(s)}>
        <td><b>{s.ticker}</b><small>{s.company}</small></td><td className="mono">{s.price == null ? '—' : `$${number(s.price)}`}<small>{dateLabel(s.price_date)}</small></td>
        <td className={`mono ${tone(s.day_change_pct)}`}>{percent(s.day_change_pct)}</td>
        <td className={`mono strong ${directionTone(s.direction)}`}>{arrowFor(s.direction)} {percent(s.expected_pct, 1)}<small>{s.timeframe ? `over ${s.timeframe}` : (s.direction ?? 'flat').toUpperCase()}</small></td>
        <td><Conviction height={s.height} /></td>
        <td className="mono">{money(s.suggested_usd)}</td>
        <td><SentimentLevel level={s.sentiment_level} trend={s.sentiment_trend} /></td><td className="mono">{s.news_count_5d ?? '—'}</td>
        <td><MiniTrend stock={s} /></td>
      </tr>)}</tbody>
    </table>}</div>
    <p className="fine-print">Values come from TigerData when connected. Missing values remain unavailable. The signal is the fusion neural network's raw expected return for the horizon the decision layer chose; conviction is that forecast in sigma units. Sentiment is the LLM agent's latest 0–100 level (50 = neutral) with its 24-hour change. Signal, conviction and suggested size are backend-generated for paper trading only.</p>
  </section>;
}
function ModelPerformance() {
  const [data, setData] = useState<Performance | null>(null);
  const [error, setError] = useState('');
  useEffect(() => { const ctrl = new AbortController(); apiGet<Performance>('/api/performance', ctrl.signal).then(setData).catch(e => { if (!ctrl.signal.aborted) setError(e.message); }); return () => ctrl.abort(); }, []);
  const curve = data?.curve ?? [];
  const regressors = [...(data?.regressors ?? [])].sort((a, b) => (b.ic_mean ?? -Infinity) - (a.ic_mean ?? -Infinity));
  const pairs = data?.redundant_pairs ?? [];
  const backtest = Object.entries(data?.backtest ?? {});
  const coverage = (m: HorizonMetrics) => m.coverage_1sigma == null ? '—' : rate(m.coverage_1sigma);
  return <section className="page-section" id="performance"><div className="section-heading"><div><span className="eyebrow">04 / VERIFIED TRAINING OUTPUTS</span><h2>Model Performance</h2></div><Badge blush>{data?.status === 'available' ? 'TEST RESULTS' : 'RESULTS UNAVAILABLE'}</Badge></div>
    {data?.status === 'available' ? <>
      <div className="panel perf-panel"><div className="section-row"><div><span className="eyebrow">FUSION NEURAL NETWORK VS BASELINE · BY HORIZON</span><h3>Held-out test accuracy</h3></div><Badge>HELD-OUT TEST</Badge></div>
        {data.horizons.length ? <div className="data-table-wrap"><table className="data-table"><thead><tr><th>HORIZON</th><th>NN IC</th><th>NN HIT RATE</th><th>1σ COVERAGE</th><th>BASELINE IC</th><th>BASELINE HIT RATE</th><th>ROWS</th></tr></thead>
          <tbody>{data.horizons.map(h => <tr key={h.horizon}><td className="mono strong">{h.horizon}</td><td className={`mono ${tone(h.nn.ic_mean)}`}>{number(h.nn.ic_mean, 3)}</td><td className="mono">{rate(h.nn.hit_rate)}</td><td className="mono">{coverage(h.nn)}</td>
            <td className={`mono ${tone(h.baseline.ic_mean)}`}>{number(h.baseline.ic_mean, 3)}</td><td className="mono">{rate(h.baseline.hit_rate)}</td><td className="mono">{h.nn.n_rows?.toLocaleString() ?? '—'}</td></tr>)}</tbody></table></div>
          : <div className="empty-state">No per-horizon metrics in metrics.json.</div>}
        <p className="fine-print">IC = mean daily Spearman rank correlation between forecast and realized raw return. Hit rate = share of rows with the correct sign. 1σ coverage = share of realized returns inside mu ± sigma (about 68% if sigma is calibrated).</p></div>
      <div className="perf-grid">
        <div className="panel perf-panel"><span className="eyebrow">15 REGRESSOR FAMILIES · NEXT-DAY IC</span>
          {regressors.length ? <div className="data-table-wrap"><table className="data-table"><thead><tr><th>FAMILY</th><th>IC</th><th>HIT RATE</th></tr></thead>
            <tbody>{regressors.map(r => <tr key={r.key}><td>{r.label}</td><td className={`mono ${tone(r.ic_mean)}`}>{number(r.ic_mean, 3)}</td><td className="mono">{rate(r.hit_rate)}</td></tr>)}</tbody></table></div>
            : <div className="empty-state">No regressor metrics in metrics.json.</div>}
          <p className="fine-print">{pairs.length ? <>Highly correlated (possibly redundant) families: {pairs.map(p => `${p.a_label} ↔ ${p.b_label} (ρ ${number(p.corr)})`).join('; ')}.</> : 'No redundant regressor pairs were flagged.'}</p></div>
        <div className="panel perf-panel"><span className="eyebrow">COST-AWARE BACKTEST · CUMULATIVE RETURN %</span>
          <TrendChart values={curve.map(p => p.nn == null ? null : p.nn * 100)} compare={curve.map(p => p.spy == null ? null : p.spy * 100)} labels={curve.map(p => p.date)} title="Fusion network strategy versus SPY cumulative return in percent" height={200} />
          {curve.length > 1 && <div className="legend"><span className="green-line">Fusion network</span><span className="dash-line">SPY</span></div>}
          {backtest.length > 0 && <div className="data-table-wrap"><table className="data-table"><thead><tr><th>STRATEGY</th><th>TOTAL RETURN</th><th>SHARPE</th><th>MAX DRAWDOWN</th><th>PERIODS</th></tr></thead>
            <tbody>{backtest.map(([name, b]) => <tr key={name}><td className="mono strong">{name === 'nn' ? 'Fusion NN' : name === 'spy' ? 'SPY' : name.charAt(0).toUpperCase() + name.slice(1)}</td><td className={`mono ${tone(b.total_return)}`}>{b.total_return == null ? '—' : percent(100 * b.total_return, 1)}</td><td className="mono">{number(b.sharpe)}</td><td className="mono">{b.max_drawdown == null ? '—' : percent(100 * b.max_drawdown, 1)}</td><td className="mono">{b.n_periods ?? '—'}</td></tr>)}</tbody></table></div>}
        </div>
      </div>
      <p className="fine-print">Training: {data.meta?.train_dates?.join(' → ') ?? '—'} · Test: {data.meta?.test_dates?.join(' → ') ?? '—'}{data.meta?.folds ? ` · ${data.meta.folds} walk-forward folds` : ''}{data.meta?.n_test_rows ? ` · ${data.meta.n_test_rows.toLocaleString()} test rows` : ''}{data.meta?.trained_at ? ` · Trained ${dateLabel(data.meta.trained_at)}` : ''}. {data.meta?.caveat ?? ''}</p></>
    : <div className="panel empty-state"><h3>Real backtest results not available yet</h3><p>{error || data?.reason || 'Loading evaluation data…'}</p><p>Results appear here after the models are trained (<code>python -m src.train</code>) and published to Tiger Data (<code>python -m src.publish_models</code>). Until then nothing is shown: we never display made-up accuracy numbers.</p></div>}
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
      <div className="replay-result"><div className="three-grid"><Metric label="SELECTED DATE" value={day} note="End-of-day decision" /><Metric label="LAST STORED SIGNAL" value={dateLabel(data?.latest_research_date)} note="Not point-in-time verified" /><Metric label="REALIZED RETURN" value="—" note="Not shown without valid replay" /></div>
        <p>{error || data?.reason || 'Checking historical availability…'}</p></div></div>
    <p className="fine-print">No historical date is marked verified because the existing pipeline does not preserve fully immutable feature, ingestion, and model-training snapshots for an auditable point-in-time forecast.</p>
  </section>;
}
function Methodology() {
  const steps = [
    ['01', 'Collect', 'Alpaca 15-minute bars, Finnhub/Alpaca news, SEC EDGAR filings and FINRA short volume, stored in Tiger Data with the time each row became knowable', '/assets/fbd86.svg'],
    ['02', 'Read', 'A Snowflake Cortex sentiment agent reads each document and records direction, a 0–100 level, novelty, materiality and drivers', '/assets/3e265.svg'],
    ['03', 'Model', '15 regressor families (reversal, momentum, volume, volatility, earnings, analysts, short selling, insiders and more) each predict the next-day return', '/assets/969c9.svg'],
    ['04', 'Fuse', 'A fusion neural network combines the regressors and sentiment into a forecast with uncertainty for 6h, 12h, 18h, 1d, 3d and 5d, scored on held-out data', '/assets/4dcef.svg'],
  ];
  return <section className="methodology" id="methodology"><span className="eyebrow">05 / METHODOLOGY</span>
    <div className="method-intro"><h2>A clear path<br />from insight to outlook.</h2><p>Readable model research with transparent inputs, real source dates, and cautious interpretation.</p></div>
    <div className="process-grid">{steps.map(([num, title, desc, icon]) => <article key={num}><span className="step">{num}</span><div className="method-icon"><img src={icon} alt="" /></div><h3>{title}</h3><p>{desc}</p></article>)}</div>
    <p className="method-note">The original Duolingo Google Play review collector is separate from this financial-news pipeline. A review-specific integration is future work and is not represented as live.</p>
  </section>;
}
function About() {
  return <section className="page-section about" id="about"><span className="eyebrow">06 / ABOUT REPLICAVIRE</span><h2>Built for the curious investor.</h2>
    <p>ReplicaVire organizes financial news signals and model outputs to help compare research ideas. Forecasts are estimates and should never be treated as guaranteed outcomes.</p>
    <div className="pricing"><article className="panel"><span className="eyebrow">FREE RESEARCH</span><h3>Research dashboard</h3><strong>$0 / month</strong><Badge blush>PROPOSED PLAN</Badge><p>Tracked stocks, sentiment scores, and research methodology.</p></article>
      <article className="panel pro"><span className="eyebrow">PRO</span><h3>Advanced research</h3><strong>$29 / month</strong><Badge blush>HYPOTHETICAL</Badge><p>Proposed historical diagnostics and deeper model comparisons. Payments are not implemented.</p></article></div>
  </section>;
}
function HorizonForecast({ horizons, chosen }: { horizons: Horizons | null; chosen: string | null }) {
  const keys = horizons ? [...HORIZON_ORDER.filter(h => h in horizons), ...Object.keys(horizons).filter(h => !HORIZON_ORDER.includes(h))] : [];
  const extent = Math.max(0.5, ...keys.map(h => Math.abs(horizons![h]?.mu ?? 0) + Math.abs(horizons![h]?.sigma ?? 0)));
  const pos = (v: number) => 50 + 50 * Math.max(-1, Math.min(1, v / extent));
  return <section className="panel horizon-panel"><div className="section-row"><div><span className="eyebrow">FORECAST BY HORIZON · FUSION NEURAL NETWORK</span><h3>Expected raw return ± 1σ</h3></div>{chosen && <Badge>CHOSEN: {chosen}</Badge>}</div>
    {keys.length === 0 ? <div className="empty-state">No horizon forecasts are stored for this stock yet.</div> : <div className="hz-rows" role="list">
      {keys.map(h => {
        const mu = horizons![h]?.mu ?? null;
        const sigma = horizons![h]?.sigma ?? null;
        return <div className={h === chosen ? 'hz-row chosen' : 'hz-row'} role="listitem" key={h} aria-label={`${h}: ${percent(mu)}${sigma != null ? ` plus or minus ${number(sigma)}%` : ''}`}>
          <span className="mono strong">{h}</span>
          <div className="hz-track"><i className="hz-zero" />
            {mu != null && <i className={mu < 0 ? 'hz-bar neg' : 'hz-bar'} style={{ left: `${Math.min(pos(0), pos(mu))}%`, width: `${Math.abs(pos(mu) - pos(0))}%` }} />}
            {mu != null && sigma != null && <i className="hz-whisker" style={{ left: `${pos(mu - Math.abs(sigma))}%`, width: `${pos(mu + Math.abs(sigma)) - pos(mu - Math.abs(sigma))}%` }} />}
            {mu != null && <i className="hz-dot" style={{ left: `${pos(mu)}%` }} />}
          </div>
          <span className={`mono ${tone(mu)}`}>{percent(mu)}<small>{sigma == null ? 'σ unavailable' : `± ${number(sigma)}%`}</small></span>
        </div>;
      })}
      <div className="hz-axis mono"><span /><div><span>−{extent.toFixed(1)}%</span><span>0</span><span>+{extent.toFixed(1)}%</span></div><span /></div>
    </div>}
    <p className="fine-print">Bars show the mean forecast (mu); whiskers span ± one predicted standard deviation (sigma). The highlighted row is the horizon the decision layer chose for the signal.</p>
  </section>;
}
function WhyPanel({ reason, drivers }: { reason: string | null; drivers: Driver[] | null }) {
  const list = drivers ?? [];
  const scale = Math.max(0, ...list.map(d => Math.abs(d.attribution ?? 0)));
  return <section className="panel why-panel"><span className="eyebrow">WHY</span>
    <p>{reason || 'No explanation was provided by the backend for this signal.'}</p>
    {list.length > 0 && <div className="driver-list">{list.map((d, i) => <div className="driver-row" key={`${d.name}-${i}`}>
      <span>{d.label || d.name}<small className="mono">value {featureValue(d.value)}</small></span>
      <SignedBar value={d.attribution} scale={scale} />
      <span className={`mono ${tone(d.attribution)}`}>{signed(d.attribution, 3)}</span>
    </div>)}</div>}
    <p className="fine-print">{list.length ? 'Top model inputs by integrated-gradients attribution; green pushes the forecast up, red pushes it down.' : 'Driver attributions are unavailable for this signal.'}</p>
  </section>;
}
function RegressorTable({ regressors }: { regressors: RegressorOutput[] | null }) {
  const list = [...(regressors ?? [])].sort((a, b) => Math.abs(b.yhat_pct ?? 0) - Math.abs(a.yhat_pct ?? 0));
  const scale = Math.max(0, ...list.map(r => Math.abs(r.yhat_pct ?? 0)));
  return <section className="panel"><div className="section-row"><span className="eyebrow">15 REGRESSORS · NEXT-DAY RETURN ESTIMATES</span><small>Sorted by size</small></div>
    {list.length ? <div className="data-table-wrap"><table className="data-table"><thead><tr><th>FAMILY</th><th>ESTIMATE</th><th /></tr></thead>
      <tbody>{list.map(r => <tr key={r.key}><td>{r.label}</td><td className={`mono strong ${tone(r.yhat_pct)}`}>{percent(r.yhat_pct, 3)}</td><td className="bar-cell"><SignedBar value={r.yhat_pct} scale={scale} /></td></tr>)}</tbody></table></div>
      : <div className="empty-state">No regressor outputs are stored for this stock yet.</div>}
    <p className="fine-print">Each family's model predicts the next-day raw return on its own; the fusion network weighs them together with sentiment.</p>
  </section>;
}
const sentimentBadge = (d: string | null) => d === 'improving' ? 'badge' : d === 'deteriorating' ? 'badge blush' : 'badge neutral';
function SentimentPanel({ ticker, levels, labels, events }: { ticker: string; levels: (number | null)[]; labels: string[]; events: SentimentEvent[] }) {
  return <section className="panel reviews"><div className="section-row"><span className="eyebrow">LLM SENTIMENT AGENT · LEVEL (0–100), LAST 90 DAYS</span><small>{events.length} recent events</small></div>
    <TrendChart values={levels} labels={labels} title={`${ticker} sentiment level over the last 90 days`} height={200} />
    {events.length ? events.slice(0, 12).map((e, i) => <article className="review" key={`${e.published_at}-${i}`}>
      <small>{dateTimeLabel(e.published_at)}</small>
      <span className={sentimentBadge(e.direction)}>{(e.direction ?? 'unrated').toUpperCase()}</span>
      <small className={`mono ${tone(e.delta)}`}>Δ {signed(e.delta, 2)}</small>
      <small className="mono">Level {e.level == null ? '—' : e.level.toFixed(0)} · Materiality {number(e.materiality)} · Novelty {number(e.novelty)}</small>
      {e.summary && <p>{e.summary}</p>}
      {e.drivers.length > 0 && <div className="tag-list">{e.drivers.map((d, j) => <span className="tag" key={j}>{d}</span>)}</div>}
      {(e.doc_title || safeUrl(e.doc_url)) && <small className="doc-source">{e.doc_source && <b>{e.doc_source.toUpperCase()}{e.doc_type && e.doc_type !== 'news' ? ` ${e.doc_type}` : ''} · </b>}
        {safeUrl(e.doc_url) ? <a href={safeUrl(e.doc_url)!} target="_blank" rel="noopener noreferrer">{e.doc_title || 'Source document'} ↗</a> : e.doc_title}</small>}
    </article>) : <div className="empty-state">No sentiment events are stored for this stock yet.</div>}
    <p className="fine-print">Each event is one news article or filing read by the agent: direction and delta describe the change it implies, level is the running 0–100 sentiment, materiality and novelty are 0–1.</p>
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
  const sent = detail.sentiment_history;
  return <main className="analysis-page">
    <div className="analysis-bar"><button className="back" onClick={onBack}>← Back to screener</button><span>{ticker} · Source: news, SEC filings + TigerData</span></div>
    <div className="disclosure"><Badge>REAL DATABASE VALUES</Badge><b>Forecasts are raw stock returns in percent from the fusion neural network, not excess returns versus SPY. Blank fields mean unavailable.</b></div>
    <div className="analysis-layout"><div className="analysis-main">
      <section className="panel stock-summary"><div className="section-row"><div><span className="eyebrow">{ticker}</span><h2>{s.company}</h2></div><Badge blush={!s.signal_date}>{s.signal_date ? `MODEL AS OF ${dateTimeLabel(s.signal_ts ?? s.signal_date).toUpperCase()}` : 'NO STORED FORECAST'}</Badge></div>
        <div className="four-grid"><Metric label="LAST STORED CLOSE" value={s.price == null ? '—' : `$${number(s.price)}`} note={dateLabel(s.price_date)} />
          <Metric label="SENTIMENT LEVEL" value={s.sentiment_level == null ? '—' : s.sentiment_level.toFixed(0)} note="0–100 · 50 is neutral" toneClass={levelTone(s.sentiment_level)} />
          <Metric label="SENTIMENT · 24H" value={signed(s.sentiment_trend, 1)} note="Change in level" toneClass={tone(s.sentiment_trend)} />
          <Metric label="DOCUMENTS · 5D" value={s.news_count_5d?.toString() ?? '—'} note="Read by the sentiment agent" /></div>
        <p className="fine-print">Latest sentiment event {dateTimeLabel(s.sentiment_date)}. Sources: Finnhub/Alpaca news and SEC EDGAR filings, not Google Play reviews.</p>
      </section>
      <HorizonForecast horizons={detail.horizons ?? s.horizons} chosen={s.timeframe} />
      <section className="panel"><div className="section-row"><span className="eyebrow">STORED STOCK PRICE HISTORY</span><div className="tabs">{[10, 30, 90, 150].map(n => <button className={n === range ? 'selected' : ''} onClick={() => setRange(n)} key={n}>{n}D</button>)}</div></div>
        <TrendChart values={history.map(p => p.close)} labels={history.map(p => p.ts)} title={`${ticker} historical daily closing prices`} height={220} /></section>
      <RegressorTable regressors={detail.regressors} />
      <SentimentPanel ticker={ticker} levels={sent.map(p => p.level)} labels={sent.map(p => p.published_at)} events={detail.sentiment_events} />
      <section className="panel reviews"><div className="section-row"><span className="eyebrow">RECENT NEWS & FILINGS</span><small>{detail.articles.length} stored documents</small></div>
        {detail.articles.length ? detail.articles.slice(0, 12).map((a, i) => <article className="review" key={`${a.url}-${i}`}>
          <small>{dateTimeLabel(a.published_at)}{a.publisher ? ` · ${a.publisher}` : ''}</small>
          {a.source && <Badge>{a.source.toUpperCase()}</Badge>}{a.doc_type && <span className={a.doc_type === 'news' ? 'badge neutral' : 'badge blush'}>{a.doc_type.toUpperCase()}</span>}
          <p>{safeUrl(a.url) ? <a href={safeUrl(a.url)!} target="_blank" rel="noopener noreferrer">{a.title || 'Open source document ↗'}</a> : (a.title || 'Untitled document')}</p>
          {a.body && <small>{a.body.slice(0, 300)}</small>}
        </article>) : <div className="empty-state">No stored news or filings for this stock yet.</div>}
      </section>
    </div>
    <aside className="analysis-aside">
      <section className="panel signal-panel"><div className="section-row"><span className="eyebrow">SIGNAL</span><span className={`badge ${s.direction === 'short' ? 'blush' : ''}`}>{arrowFor(s.direction)} {(s.direction ?? 'flat').toUpperCase()}</span></div>
        <div className="metric-grid">
          <Metric label="EXPECTED MOVE" value={percent(s.expected_pct, 1)} note={s.timeframe ? `Over ${s.timeframe}` : 'Timeframe unavailable'} toneClass={tone(s.expected_pct)} />
          <Metric label="CONVICTION" value={s.height == null ? '—' : `${number(s.height)}σ`} note="Signal height, sigma units" />
          <Metric label="TIMEFRAME" value={s.timeframe ?? '—'} note="Chosen horizon" />
          <Metric label="SUGGESTED SIZE" value={money(s.suggested_usd)} note="Paper trading only" />
        </div>
        <Conviction height={s.height} />
        <p className="fine-print">Model-generated signal. Not investment advice.</p></section>
      <WhyPanel reason={s.reason} drivers={detail.drivers} />
      <section className="panel"><span className="eyebrow">DATA PROVENANCE</span><p>Last price: {dateLabel(s.price_date)}</p><p>Latest sentiment event: {dateTimeLabel(s.sentiment_date)}</p><p>Signal: {dateTimeLabel(s.signal_ts ?? s.signal_date)}</p><p className="fine-print">{detail.source_note}</p></section>
    </aside></div>
    <div className="analysis-replay"><HistoricalReplay ticker={ticker} /></div>
  </main>;
}
const alertHeadline = (a: Alert) => a.mode === 'auto'
  ? `Executed: ${a.action} ${money(a.size_usd)} ${a.symbol}`
  : `${a.action} ${a.symbol}${a.act_by ? ` by ${timeLabel(a.act_by)}` : ''}`;
const moveLabel = (a: Alert) => `${percent(a.expected_pct, 1)}${a.timeframe ? ` over ${a.timeframe}` : ''}`;

function Recommendations({ alerts, loaded, notice, authenticated, onExecute, onDismiss, onDetails }: { alerts: Alert[]; loaded: boolean; notice: string; authenticated: boolean; onExecute: (id: number) => void; onDismiss: (id: number) => void; onDetails: (symbol: string) => void }) {
  return <section className="page-section" id="recommendations">
    <div className="section-heading"><div><span className="eyebrow">02 / SIGNAL ALERTS</span><h2>Recommendations</h2></div><Badge blush>PAPER TRADING ONLY</Badge></div>
    {notice && <div className="connection-error" role="alert">{notice}</div>}
    {alerts.length === 0 ? <div className="panel empty-state"><h3>No recommendations yet</h3><p>{loaded ? 'Alerts appear here when the backend emits a signal. None are fabricated.' : 'Checking the backend for alerts…'}</p></div>
      : <div className="alert-list">{alerts.map(a => <article className="panel alert-card" key={a.id}>
        <div className="alert-head">
          <Badge blush={sellSide(a)}>{a.action}</Badge>
          <button className="symbol-link" onClick={() => onDetails(a.symbol)}>{a.symbol}</button>
          <span className={`mono strong ${tone(a.expected_pct)}`}>{moveLabel(a)}</span>
          <span className="alert-tags"><span className="badge">{a.mode.toUpperCase()}</span><span className={a.status === 'new' ? 'badge blush' : 'badge'}>{a.status.toUpperCase()}</span></span>
        </div>
        <p>{a.reason}</p>
        <small className="mono">Act by {timeLabel(a.act_by)} · Size {money(a.size_usd)} · Conviction {a.height == null ? '—' : `${number(a.height)}σ`} · Created {timeLabel(a.created_at)}</small>
        {a.status === 'new' && a.mode === 'recommend' && <div className="actions"><button className="button primary" onClick={() => onExecute(a.id)}>Execute (paper)</button><button className="button secondary" onClick={() => onDismiss(a.id)}>Dismiss</button></div>}
      </article>)}</div>}
    <p className="fine-print">Alerts are model outputs, not investment advice. Execution places paper orders through Alpaca only.{authenticated ? '' : ' Log in under Auto-Trading to execute.'}</p>
  </section>;
}

function AutoTrading({ authenticated, onAuth, sectors }: { authenticated: boolean | null; onAuth: (v: boolean) => void; sectors: string[] }) {
  const [passphrase, setPassphrase] = useState('');
  const [settings, setSettings] = useState<Settings | null>(null);
  const [error, setError] = useState('');
  const [saved, setSaved] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => { setError(''); setSaved('');
    if (!authenticated) { setSettings(null); return; }
    const ctrl = new AbortController();
    apiGet<Settings>('/api/settings', ctrl.signal).then(setSettings).catch(e => { if (!ctrl.signal.aborted) { if (e instanceof ApiError && e.status === 401) onAuth(false); else setError(e.message); } });
    return () => ctrl.abort();
  }, [authenticated]);
  const set = <K extends keyof Settings>(key: K, value: Settings[K]) => { setSettings(prev => prev ? { ...prev, [key]: value } : prev); setSaved(''); };
  const login = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault(); setBusy(true); setError('');
    try { await apiSend<Session>('/api/login', 'POST', { passphrase }); setPassphrase(''); onAuth(true); }
    catch (err) { setError(err instanceof ApiError && err.status === 401 ? 'Incorrect passphrase.' : (err as Error).message); }
    finally { setBusy(false); }
  };
  const logout = async () => { try { await apiSend('/api/logout', 'POST'); } catch { /* cookie may already be gone */ } onAuth(false); };
  const invalid = settings != null && settings.min_usd > settings.max_usd;
  const save = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault(); if (!settings || invalid) return;
    setBusy(true); setError('');
    try { setSettings(await apiSend<Settings>('/api/settings', 'PUT', settings)); setSaved(`Saved at ${new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}`); }
    catch (err) { if (err instanceof ApiError && err.status === 401) onAuth(false); else setError((err as Error).message); }
    finally { setBusy(false); }
  };
  const modes: [DirectionMode, string][] = [['both', 'Long & short'], ['long_only', 'Long only'], ['short_only', 'Short only']];
  return <section className="page-section" id="autotrading">
    <div className="section-heading"><div><span className="eyebrow">03 / AUTOMATION RULES</span><h2>Auto-Trading</h2></div><Badge blush>PAPER TRADING ONLY</Badge></div>
    {authenticated === null ? <div className="panel empty-state">Checking session…</div>
      : !authenticated ? <form className="panel login-form" onSubmit={login}>
        <h3>Log in to manage automation</h3>
        <p>Settings and paper execution are locked behind the operator passphrase.</p>
        <label className="field"><span className="eyebrow">PASSPHRASE</span><input type="password" autoComplete="current-password" value={passphrase} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setPassphrase(e.target.value)} required /></label>
        {error && <p className="negative">{error}</p>}
        <button className="button primary" type="submit" disabled={busy || !passphrase}>{busy ? 'Logging in…' : 'Log in'}</button>
      </form>
      : !settings ? <div className="panel empty-state">{error || 'Loading settings…'}</div>
      : <form className="panel settings-form" onSubmit={save}>
        <label className="toggle-row"><input type="checkbox" checked={settings.auto_trading} onChange={(e: React.ChangeEvent<HTMLInputElement>) => set('auto_trading', e.target.checked)} /><span className="toggle" aria-hidden="true" /><span><b>Auto-trading {settings.auto_trading ? 'ON' : 'OFF'}</b><small>When on, qualifying signals place paper orders automatically. When off, they arrive as recommendations.</small></span></label>
        <div className="field full"><span className="eyebrow">SECTORS (NONE SELECTED = ALL)</span><SectorChips sectors={sectors} selected={settings.sectors} onChange={v => set('sectors', v)} /></div>
        <label className="field"><span className="eyebrow">MIN $ PER TRADE</span><input type="number" min={0} step={10} value={settings.min_usd} onChange={(e: React.ChangeEvent<HTMLInputElement>) => set('min_usd', Number(e.target.value))} /></label>
        <label className="field"><span className="eyebrow">MAX $ PER TRADE</span><input type="number" min={0} step={10} value={settings.max_usd} onChange={(e: React.ChangeEvent<HTMLInputElement>) => set('max_usd', Number(e.target.value))} /></label>
        {invalid && <p className="negative full">Min $ per trade must be less than or equal to Max $ per trade.</p>}
        <label className="field"><span className="eyebrow">MINIMUM CONVICTION (σ)</span><input type="number" min={0} max={5} step={0.1} value={settings.min_height} onChange={(e: React.ChangeEvent<HTMLInputElement>) => set('min_height', Number(e.target.value))} /><small>Signals below this height are ignored.</small></label>
        <label className="field"><span className="eyebrow">MAX OPEN POSITIONS</span><input type="number" min={0} step={1} value={settings.max_positions} onChange={(e: React.ChangeEvent<HTMLInputElement>) => set('max_positions', Number(e.target.value))} /></label>
        <fieldset className="field radio-group"><legend className="eyebrow">DIRECTION MODE</legend>
          {modes.map(([value, label]) => <label key={value}><input type="radio" name="direction_mode" value={value} checked={settings.direction_mode === value} onChange={() => set('direction_mode', value)} />{label}</label>)}
        </fieldset>
        <label className="field"><span className="eyebrow">DAY-TRADE SPIKE THRESHOLD (σ)</span><input type="number" min={0} max={5} step={0.1} value={settings.day_trade_min_height} onChange={(e: React.ChangeEvent<HTMLInputElement>) => set('day_trade_min_height', Number(e.target.value))} /><small>Same-day exits are blocked unless conviction exceeds this.</small></label>
        {error && <p className="negative full">{error}</p>}
        <div className="actions full"><button className="button primary" type="submit" disabled={busy || invalid}>{busy ? 'Saving…' : 'Save settings'}</button><button className="button secondary" type="button" onClick={logout}>Log out</button>{saved && <small className="positive">{saved}</small>}</div>
      </form>}
    <p className="fine-print">Orders route to an Alpaca paper account only. No real money is traded. Not investment advice.</p>
  </section>;
}

type Toast = { alert: Alert; error?: string };
function ToastItem({ toast, onClose, onExecute, onDismiss, onDetails }: { toast: Toast; onClose: () => void; onExecute: () => void; onDismiss: () => void; onDetails: () => void }) {
  const [hover, setHover] = useState(false);
  useEffect(() => { if (hover) return; const t = window.setTimeout(onClose, 12000); return () => window.clearTimeout(t); }, [hover, toast.alert.id]);
  const a = toast.alert;
  const actionable = a.mode === 'recommend' && a.status === 'new';
  return <div className={`toast ${sellSide(a) ? 'sell' : 'buy'}`} role="status" onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
    <div className="section-row"><b>{alertHeadline(a)}</b><button className="toast-close" aria-label="Close notification" onClick={onClose}>×</button></div>
    <span className={`mono ${tone(a.expected_pct)}`}>{moveLabel(a)}{a.height != null ? ` · ${number(a.height)}σ` : ''}</span>
    {a.reason && <small>{a.reason}</small>}
    {toast.error && <small className="negative">{toast.error}</small>}
    <div className="toast-actions">
      {actionable && <button className="button primary" onClick={onExecute}>Execute</button>}
      <button className="button secondary" onClick={onDetails}>Details</button>
      {actionable && <button className="button secondary" onClick={onDismiss}>Dismiss</button>}
    </div>
  </div>;
}

function Footer() {
  return <footer><div><span className="wordmark">ReplicaVire</span><span>Know the data. Understand the limits.</span></div><p>Research prototype. Forecasts and live-data status depend on the connected backend. Paper trading only via Alpaca. Not investment advice.</p><small>© 2026 ReplicaVire · PALMETTO HACKS</small></footer>;
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
  const [authenticated, setAuthenticated] = useState<boolean | null>(null);
  useEffect(() => { apiGet<Session>('/api/session').then(r => setAuthenticated(!!r.authenticated)).catch(() => setAuthenticated(false)); }, []);

  // Alerts: initial load + poll every 15s for anything newer than the highest id seen.
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [alertsLoaded, setAlertsLoaded] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [notice, setNotice] = useState('');
  const maxSeen = useRef<number | null>(null);
  useEffect(() => {
    let cancelled = false;
    const pushToasts = (items: Alert[]) => { if (items.length) setToasts(prev => [...items.map(alert => ({ alert })), ...prev.filter(t => !items.some(i => i.id === t.alert.id))].slice(0, 3)); };
    const poll = async () => {
      try {
        const first = maxSeen.current === null;
        const data = await apiGet<{ alerts: Alert[] }>(first ? '/api/alerts' : `/api/alerts?since=${maxSeen.current}`);
        if (cancelled) return;
        const incoming = data.alerts ?? [];
        maxSeen.current = Math.max(maxSeen.current ?? 0, ...incoming.map(a => a.id));
        if (incoming.length) {
          const ids = new Set(incoming.map(a => a.id));
          setAlerts(prev => [...incoming, ...prev.filter(a => !ids.has(a.id))].sort((x, y) => y.id - x.id));
          const cutoff = Date.now() - 10 * 60 * 1000;
          pushToasts(first ? incoming.filter(a => a.status === 'new' && Date.parse(a.created_at) >= cutoff).slice(0, 3) : incoming.slice(0, 3));
        }
        setAlertsLoaded(true);
      } catch { /* fail silently; next poll retries */ }
    };
    poll();
    const timer = window.setInterval(poll, 15000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, []);
  const setStatus = (id: number, status: Alert['status']) => setAlerts(prev => prev.map(a => a.id === id ? { ...a, status } : a));
  const closeToast = (id: number) => setToasts(prev => prev.filter(t => t.alert.id !== id));
  const executeAlert = async (id: number) => {
    try {
      const r = await apiSend<{ status?: Alert['status'] }>(`/api/alerts/${id}/execute`, 'POST');
      setStatus(id, r.status ?? 'requested'); closeToast(id); setNotice('');
    } catch (e) {
      const msg = e instanceof ApiError && e.status === 401 ? 'Log in under Auto-Trading to execute paper trades.' : (e as Error).message;
      if (e instanceof ApiError && e.status === 401) setAuthenticated(false);
      setNotice(msg); setToasts(prev => prev.map(t => t.alert.id === id ? { ...t, error: msg } : t));
    }
  };
  const dismissAlert = async (id: number) => {
    closeToast(id);
    try { await apiSend(`/api/alerts/${id}/dismiss`, 'POST'); setStatus(id, 'dismissed'); setNotice(''); } catch (e) { setNotice((e as Error).message); }
  };
  const openStock = (ticker: string) => { setSelected(ticker); window.scrollTo(0, 0); };
  const sectors = overview?.sectors?.length ? overview.sectors : GICS_SECTORS;

  const goHome = () => { setSelected(null); window.scrollTo({ top: 0, behavior: 'smooth' }); };
  return <div className="app"><Header onHome={goHome} />
    <div className="toast-stack" aria-live="polite">{toasts.map(t => <ToastItem key={t.alert.id} toast={t} onClose={() => closeToast(t.alert.id)} onExecute={() => executeAlert(t.alert.id)} onDismiss={() => dismissAlert(t.alert.id)} onDetails={() => { closeToast(t.alert.id); openStock(t.alert.symbol); }} />)}</div>
    {selected ? <StockAnalysis ticker={selected} onBack={() => { setSelected(null); scrollToId('screener'); }} /> : <main>
      {apiError && <div className="connection-error" role="alert"><b>Backend unavailable:</b> {apiError}. Start the Python API and verify TigerData is configured. <button className="button secondary" onClick={() => setRefresh(v => v + 1)}>Retry</button></div>}
      <Hero overview={overview} connected={!!overview} />
      <Screener stocks={overview?.stocks ?? []} sectors={sectors} loading={loading} onSelect={s => openStock(s.ticker)} />
      <Recommendations alerts={alerts} loaded={alertsLoaded} notice={notice} authenticated={!!authenticated} onExecute={executeAlert} onDismiss={dismissAlert} onDetails={openStock} />
      <AutoTrading authenticated={authenticated} onAuth={setAuthenticated} sectors={sectors} />
      <ModelPerformance /><Methodology /><About />
    </main>}
    <Footer />
  </div>;
}
