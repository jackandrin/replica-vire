import { useMemo, useState } from "react";

type Stock = {
  ticker: string;
  company: string;
  sector: string;
  price: number;
  day: number;
  sentiment: number;
  momentum: number;
  nn: number;
  regression: number;
  combined: number;
  spread: number;
};

const stocks: Stock[] = [
  { ticker: "DUOL", company: "Duolingo Inc.", sector: "Consumer", price: 142.5, day: 2.4, sentiment: 0.82, momentum: 0.12, nn: 6.8, regression: 6.2, combined: 6.5, spread: 0.6 },
  { ticker: "UBER", company: "Uber Technologies Inc.", sector: "Technology", price: 48.2, day: 1.8, sentiment: 0.74, momentum: 0.08, nn: 5.4, regression: 5.1, combined: 5.3, spread: 0.3 },
  { ticker: "SPOT", company: "Spotify Technology SA", sector: "Communication", price: 184.1, day: -0.9, sentiment: -0.31, momentum: -0.06, nn: 4.1, regression: 3.8, combined: 4.0, spread: 0.3 },
  { ticker: "NFLX", company: "Netflix Inc.", sector: "Communication", price: 612.4, day: 0.6, sentiment: 0.58, momentum: 0.04, nn: 3.7, regression: 3.4, combined: 3.6, spread: 0.3 },
  { ticker: "RBLX", company: "Roblox Corp.", sector: "Technology", price: 56.8, day: -1.2, sentiment: -0.42, momentum: -0.09, nn: 2.9, regression: 2.6, combined: 2.8, spread: 0.3 },
];

const smoothScroll = (id: string) => {
  window.setTimeout(() => document.getElementById(id)?.scrollIntoView({ behavior: "smooth" }), 0);
};

function Logo({ onClick }: { onClick?: () => void }) {
  return (
    <button className="logo" onClick={onClick} aria-label="ReplicaVire home">
      <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
      <span className="wordmark">ReplicaVire</span>
      <span className="descriptor">EQUITY RESEARCH</span>
    </button>
  );
}

function Header({ onHome }: { onHome: () => void }) {
  const go = (id: string) => { onHome(); smoothScroll(id); };
  return (
    <header className="site-header">
      <Logo onClick={onHome} />
      <nav className="desktop-nav" aria-label="Primary navigation">
        <button className="active" onClick={onHome}>Dashboard</button>
        <button onClick={() => go("screener")}>Stock Screener</button>
        <button onClick={() => go("performance")}>Model Performance</button>
        <button onClick={() => go("methodology")}>Methodology</button>
        <button onClick={() => go("about")}>About</button>
      </nav>
      <div className="header-actions">
        <button className="watchlist" aria-label="Illustrative watchlist"><img src="/assets/8df8b.svg" alt="" width="15" height="15" /> Watchlist <span className="mono">03</span></button>
        <button className="button dark launch" onClick={() => go("screener")}>Launch Dashboard</button>
      </div>
    </header>
  );
}

function Badge({ children, blush = false }: { children: React.ReactNode; blush?: boolean }) {
  return <span className={blush ? "badge blush" : "badge"}>{children}</span>;
}

function Metric({ label, value, note, tone = "green" }: { label: string; value: string; note: string; tone?: "green" | "dark" | "red" }) {
  return (
    <div className="metric">
      <span className="eyebrow">{label}</span>
      <strong className={`mono ${tone}`}>{value}</strong>
      <small>{note}</small>
    </div>
  );
}

function Chart({ kind = "price" }: { kind?: "price" | "compare" | "model" | "spark" }) {
  const sources = kind === "price"
    ? ["/assets/8993c.svg"]
    : kind === "compare"
      ? ["/assets/a6c3f.svg", "/assets/64a19.svg"]
      : kind === "model"
        ? ["/assets/478d5.svg", "/assets/5e914.png"]
        : ["/assets/61974.svg"];
  return (
    <div className={`chart ${kind}`} aria-label={`${kind} illustrative chart`}>
      <span className="grid-lines" />
      {sources.map((source) => <img key={source} src={source} alt="" />)}
    </div>
  );
}

function Hero() {
  return (
    <section className="hero">
      <div className="hero-copy">
        <Badge>AI-POWERED SENTIMENT RESEARCH</Badge>
        <h1>Understand the sentiment. Predict the movement.</h1>
        <p>Discover investment opportunities through AI-powered customer sentiment analysis, neural network predictions, and historical market data.</p>
        <div className="actions">
          <button className="button primary" onClick={() => smoothScroll("screener")}>Explore Stocks</button>
          <button className="button secondary" onClick={() => smoothScroll("methodology")}>How It Works</button>
        </div>
        <div className="hero-stats">
          <Metric label="SOURCES" value="12k+" note="Illustrative reviews processed daily" tone="dark" />
          <Metric label="MODELS" value="2" note="Neural network + regression" tone="dark" />
          <Metric label="COVERAGE" value="150+" note="Illustrative US-listed coverage" tone="dark" />
        </div>
      </div>
      <div className="panel preview">
        <div className="section-row">
          <div><span className="eyebrow">DASHBOARD PREVIEW</span><h3>Today’s sentiment-led opportunities</h3></div>
          <Badge>ILLUSTRATIVE</Badge>
        </div>
        <div className="three-grid">
          <Metric label="NEURAL NETWORK" value="+4.8%" note="Median five-day predicted return" />
          <Metric label="SENTIMENT MOMENTUM" value="+0.18" note="Seven-day score change" />
          <Metric label="FORECAST SPREAD" value="0.6pp" note="Difference between estimates" />
        </div>
        <div className="preview-chart">
          <div className="section-row"><span className="eyebrow">SENTIMENT VS. PRICE</span><small>30-day trend</small></div>
          <Chart kind="compare" />
          <div className="legend"><span className="green-line">Sentiment</span><span>Price</span></div>
        </div>
      </div>
    </section>
  );
}

function InfoTip({ text }: { text: string }) {
  return <span className="info-tip" tabIndex={0} data-tip={text} aria-label={text}>i</span>;
}

function Screener({ onSelect }: { onSelect: (stock: Stock) => void }) {
  const [query, setQuery] = useState("");
  const [sector, setSector] = useState("All");
  const [sentiment, setSentiment] = useState("All");
  const [returns, setReturns] = useState("All");
  const [sort, setSort] = useState<keyof Stock>("combined");
  const [direction, setDirection] = useState<"asc" | "desc">("desc");
  const [loading, setLoading] = useState(false);

  const update = (fn: () => void) => {
    setLoading(true);
    fn();
    window.setTimeout(() => setLoading(false), 240);
  };

  const filtered = useMemo(() => stocks
    .filter((stock) => `${stock.ticker} ${stock.company}`.toLowerCase().includes(query.toLowerCase()))
    .filter((stock) => sector === "All" || stock.sector === sector)
    .filter((stock) => sentiment === "All" || (sentiment === "Positive" ? stock.sentiment > 0 : stock.sentiment < 0))
    .filter((stock) => returns === "All" || (returns === "4+" ? stock.combined >= 4 : stock.combined < 4))
    .sort((a, b) => {
      const av = a[sort], bv = b[sort];
      const comparison = typeof av === "string" ? av.localeCompare(String(bv)) : Number(av) - Number(bv);
      return direction === "asc" ? comparison : -comparison;
    }), [query, sector, sentiment, returns, sort, direction]);

  const clear = () => { setQuery(""); setSector("All"); setSentiment("All"); setReturns("All"); };
  return (
    <section className="page-section" id="screener">
      <div className="section-heading">
        <div><span className="eyebrow">01 / STOCK SCREENER</span><h2>Stock Screener</h2></div>
        <small>{filtered.length} stocks · Illustrative sentiment and prediction data</small>
      </div>
      <div className="filters">
        <label className="search"><img src="/assets/c647e.svg" alt="" width="16" height="16" /><input value={query} onChange={(e) => update(() => setQuery(e.target.value))} placeholder="Search ticker or company" /></label>
        <select aria-label="Sector filter" value={sector} onChange={(e) => update(() => setSector(e.target.value))}>
          <option>All</option><option>Consumer</option><option>Technology</option><option>Communication</option>
        </select>
        <select aria-label="Sentiment filter" value={sentiment} onChange={(e) => update(() => setSentiment(e.target.value))}>
          <option>All</option><option>Positive</option><option>Negative</option>
        </select>
        <select aria-label="Predicted return filter" value={returns} onChange={(e) => update(() => setReturns(e.target.value))}>
          <option value="All">Return: All</option><option value="4+">Return: 4%+</option><option value="<4">Return: Under 4%</option>
        </select>
        <select aria-label="Sort column" value={sort} onChange={(e) => setSort(e.target.value as keyof Stock)}>
          <option value="combined">Sort: Predicted return</option><option value="sentiment">Sort: Sentiment</option><option value="spread">Sort: Spread</option><option value="price">Sort: Price</option><option value="ticker">Sort: Ticker</option>
        </select>
        <button className="sort-button" onClick={() => setDirection(direction === "asc" ? "desc" : "asc")} aria-label={`Sort ${direction === "asc" ? "descending" : "ascending"}`}>{direction === "asc" ? "↑ ASC" : "↓ DESC"}</button>
      </div>
      <div className="table-wrap">
        {loading ? (
          <div className="loading-state"><i /><i /><i /><span>Filtering illustrative signals…</span></div>
        ) : filtered.length === 0 ? (
          <div className="empty-state">
            <span className="empty-icon">⌕</span><h3>No results found</h3>
            <p>Try adjusting your filters or search for a different ticker or company.</p>
            <button className="button secondary" onClick={clear}>Clear filters</button>
          </div>
        ) : (
          <table>
            <thead><tr>
              <th>TICKER / COMPANY</th><th>PRICE</th><th>1D</th>
              <th>SENTIMENT <InfoTip text="Average illustrative sentiment score from −1 to +1." /></th>
              <th>7D MOMENTUM <InfoTip text="Change in average sentiment over seven days; score units, not stock return." /></th>
              <th>NN 5D</th><th>REG 5D</th>
              <th>COMBINED <InfoTip text="Illustrative weighted average predicted five-day return." /></th>
              <th>SPREAD <InfoTip text="Absolute difference between model estimates in percentage points; not confidence." /></th><th>SPARKLINE</th>
            </tr></thead>
            <tbody>{filtered.map((stock, index) => (
              <tr key={stock.ticker} onClick={() => onSelect(stock)} tabIndex={0} onKeyDown={(e) => e.key === "Enter" && onSelect(stock)} className={index === 0 ? "selected-row" : ""}>
                <td><b>{stock.ticker}</b><small>{stock.company}</small></td>
                <td className="mono">{stock.price.toFixed(2)}<small>Sample close</small></td>
                <td className={`mono ${stock.day >= 0 ? "positive" : "negative"}`}>{stock.day > 0 ? "+" : ""}{stock.day}%</td>
                <td className={`mono ${stock.sentiment >= 0 ? "positive" : "negative"}`}>{stock.sentiment > 0 ? "+" : ""}{stock.sentiment}</td>
                <td className={`mono ${stock.momentum >= 0 ? "positive" : "negative"}`}>{stock.momentum > 0 ? "+" : ""}{stock.momentum}</td>
                <td className="mono positive">+{stock.nn}%</td><td className="mono positive">+{stock.regression}%</td>
                <td className="mono positive strong">+{stock.combined}%</td><td className="mono positive">{stock.spread}pp</td>
                <td><Chart kind="spark" /></td>
              </tr>
            ))}</tbody>
          </table>
        )}
      </div>
      <p className="fine-print">Local mock data only. Search, filters, sorting and row links are interactive; there is no live market-data, payment, or trading integration.</p>
    </section>
  );
}

const modelCards = [
  { name: "Neural Network", accuracy: "78%", mae: "3.4 pp", rmse: "4.6 pp" },
  { name: "Regression Model", accuracy: "72%", mae: "4.1 pp", rmse: "5.2 pp" },
  { name: "Combined Forecast", accuracy: "76%", mae: "3.2 pp", rmse: "4.3 pp" },
];

function ModelPerformance() {
  const [period, setPeriod] = useState("1Y");
  return (
    <section className="page-section" id="performance">
      <div className="section-heading">
        <div><span className="eyebrow">02 / MODEL PERFORMANCE ANALYTICS</span><h2>Model Performance</h2></div>
        <span><Badge blush>DEMO DATA</Badge> <small>Illustrative metrics · no verified backtest results</small></span>
      </div>
      <div className="tabs">{["1Y", "2Y", "3Y", "5Y"].map((value) => <button key={value} className={period === value ? "selected" : ""} onClick={() => setPeriod(value)}>{value}</button>)}</div>
      <div className="model-grid">{modelCards.map((model) => (
        <article className="panel model-card" key={model.name}>
          <div className="section-row"><div><span className="eyebrow">ILLUSTRATIVE MODEL</span><h3>{model.name}</h3></div><Badge>DEMO DATA</Badge></div>
          <div className="metric-grid">
            <Metric label="DIRECTIONAL ACCURACY" value={model.accuracy} note="Illustrative up/down direction" />
            <Metric label="MAE" value={model.mae} note="Sample absolute error" tone="dark" />
            <Metric label="RMSE" value={model.rmse} note="Sample weighted error" />
            <Metric label="EVALUATED PREDICTIONS" value="240" note="Illustrative count" />
          </div>
          <span className="eyebrow">PREDICTED / REALIZED · SAMPLE 5D %</span>
          <Chart kind="model" />
          <div className="legend"><span className="green-line">Predicted</span><span>Sample return</span><span className="red-line">Benchmark</span></div>
          <p className="warning">ILLUSTRATIVE ONLY · VERIFIED RESULTS UNAVAILABLE</p>
        </article>
      ))}</div>
      <HistoricalReplay />
      <div className="state-reference">
        <div className="section-row"><span className="eyebrow">COMPONENT STATES · INTERACTIVE REFERENCE</span><small>Local mock interface only</small></div>
        <div className="state-grid">
          <article><span className="eyebrow">LOADING STATE</span><div className="skeleton"><i /><i /><i /></div><b>Filtering signals…</b></article>
          <article className="error-card"><span className="eyebrow">ERROR STATE</span><Badge blush>DATA UNAVAILABLE</Badge><b>Unable to load local model data</b><p>This demonstration state does not represent a failed live integration.</p><button className="button dark">Retry</button></article>
          <article><span className="eyebrow">EMPTY STATE</span><b>No results found</b><p>Adjust the screener filters or search for another ticker.</p><button className="button secondary" onClick={() => smoothScroll("screener")}>Open screener</button></article>
          <article className="selected-card"><span className="eyebrow">SELECTED + HOVER</span><b>Active controls are outlined</b><p>Buttons, rows, tabs, dates, and filter fields include visible hover, selected, and keyboard-focus states.</p></article>
        </div>
      </div>
    </section>
  );
}

function HistoricalReplay() {
  const [date, setDate] = useState(22);
  const [windowDays, setWindowDays] = useState(30);
  const status = date < 20 ? "Insufficient historical data" : "Retrospective analysis only";
  const blanks = [null, null, ...Array.from({ length: 31 }, (_, i) => i + 1), null, null];
  return (
    <div className="panel replay">
      <span className="eyebrow">HISTORICAL REPLAY</span>
      <div className="replay-grid">
        <div className="calendar-card">
          <p>January 2025 · calendar preview</p>
          <div className="calendar weekdays">{["M", "T", "W", "T", "F", "S", "S"].map((day, i) => <span key={`${day}-${i}`}>{day}</span>)}</div>
          <div className="calendar">{blanks.map((day, i) => <button key={i} disabled={!day} className={date === day ? "selected" : ""} onClick={() => day && setDate(day)}>{day}</button>)}</div>
          <div className="window-tabs">{[7, 14, 30].map((days) => <button key={days} className={windowDays === days ? "selected" : ""} onClick={() => setWindowDays(days)}>{days} days</button>)}</div>
          <small>Mock controls · no live historical integration</small>
        </div>
        <div className="replay-result">
          <div className="section-row"><b>Selected state</b><Badge blush>{status}</Badge></div>
          <div className="replay-values">
            <Metric label="DATE SELECTOR" value={`Jan ${date}, 2025`} note="Illustrative date" tone="dark" />
            <Metric label="SENTIMENT WINDOW" value={`${windowDays} days`} note="Selected window" tone="dark" />
            <Metric label="ELIGIBLE REVIEWS" value="—" note="Withheld" tone="dark" />
            <Metric label="HISTORICAL FORECASTS" value="—" note="Withheld" tone="dark" />
            <Metric label="REALIZED RETURN" value="—" note="Withheld" tone="red" />
          </div>
          <p>{status === "Retrospective analysis only" ? "Data was collected or analyzed after the selected cutoff. Historical outputs are withheld; retrospective research is not a verified backtest." : "The selected range lacks enough verified reviews, sentiment history, or forecast evidence. No historical result can be shown."}</p>
        </div>
      </div>
      <div className="integrity-states">
        <article><b>Verified historical data</b><Badge>Example appearance only</Badge><p>Shown only when point-in-time evidence exists. No supplied dates meet this standard.</p></article>
        <article className="current"><b>Retrospective analysis only</b><Badge blush>Current default</Badge><p>Historical outputs stay withheld when evidence came after the cutoff.</p></article>
        <article><b>Insufficient historical data</b><Badge blush>Empty state</Badge><p>Used when verified reviews, sentiment history, or forecast evidence are incomplete.</p></article>
      </div>
    </div>
  );
}

function Methodology() {
  const steps = [
    ["01", "Collect", "Use local illustrative review and price samples.", "/assets/fbd86.svg"],
    ["02", "Analyze", "Measure sample sentiment, satisfaction, and topic trends.", "/assets/3e265.svg"],
    ["03", "Train", "Compare illustrative neural-network and regression estimates.", "/assets/969c9.svg"],
    ["04", "Illuminate", "Present forecasts, model alignment, assumptions, and supporting context.", "/assets/4dcef.svg"],
  ];
  return (
    <section className="methodology" id="methodology">
      <span className="eyebrow">03 / METHODOLOGY</span>
      <div className="method-intro"><h2>A clear path<br />from insight to outlook.</h2><p>Explore a transparent range of possible outcomes with visible inputs that make every research thesis easier to understand and evaluate.</p></div>
      <div className="process-grid">{steps.map(([number, title, copy, icon]) => (
        <article key={number}><span className="step">{number}</span><div className="method-icon"><img src={icon} alt="" /></div><h3>{title}</h3><p>{copy}</p><small>DEMONSTRATION WORKFLOW</small></article>
      ))}</div>
      <p className="method-note">This local demonstration highlights a thoughtful, transparent research workflow. Snowflake AI, TigerData, market-data, payment, and trading services are represented conceptually and are not connected.</p>
    </section>
  );
}

function About() {
  return (
    <section className="page-section about" id="about">
      <span className="eyebrow">04 / ABOUT LEDGER</span>
      <h2>Built for the patient investor.</h2>
      <p>Useful research starts with better questions. What makes a business durable? What expectations are already in its price? What would have to change for the thesis to fail?</p>
      <div className="pricing">
        <article className="panel"><span className="eyebrow">FREE RESEARCH</span><h3>Core sentiment signals</h3><strong>$0 / month · Free</strong><Badge blush>PROPOSED · HYPOTHETICAL</Badge><p>Dashboard access, stock screener, and illustrative model comparisons.</p><button className="button secondary">Start free</button></article>
        <article className="panel pro"><span className="eyebrow">PRO</span><h3>Expanded stock coverage</h3><strong>$29 / month · Pro</strong><Badge blush>PROPOSED · HYPOTHETICAL</Badge><p>Advanced sentiment factors, historical replay, and deeper research signals.</p><button className="button primary">View hypothetical plan</button></article>
      </div>
    </section>
  );
}

function Home({ onSelect }: { onSelect: (stock: Stock) => void }) {
  return <><Hero /><Screener onSelect={onSelect} /><ModelPerformance /><Methodology /><About /></>;
}

function Review({ stars, score, children }: { stars: string; score: string; children: React.ReactNode }) {
  return <article className="review"><span className="positive">{stars} · sample</span><Badge blush>SAMPLE / DEMO</Badge><p>“{children}”</p><small><b className={score.startsWith("-") ? "negative" : "positive"}>{score}</b> illustrative AI score</small></article>;
}

function Analysis({ stock, onBack }: { stock: Stock; onBack: () => void }) {
  const [range, setRange] = useState("1M");
  const [mode, setMode] = useState<"Compare" | "Overlay">("Compare");
  return (
    <main className="analysis-page">
      <div className="analysis-bar"><button className="back" onClick={onBack}>← Back to screener</button><span>Illustrative analysis · {stock.ticker}</span></div>
      <div className="disclosure"><Badge blush>DEMO DATA</Badge><b>Prices, reviews, predictions, and sentiment values are illustrative and not verified results.</b></div>
      <div className="analysis-layout">
        <div className="analysis-main">
          <section className="panel stock-summary">
            <div className="section-row"><div><span className="eyebrow">{stock.company.toUpperCase()}</span><h2>{stock.company.replace(/ Inc\.| SA| Corp\./, "")}</h2></div><Badge>{stock.sentiment > 0 ? "BULLISH SENTIMENT" : "BEARISH SENTIMENT"}</Badge></div>
            <div className="four-grid">
              <Metric label="CURRENT PRICE" value={stock.price.toFixed(2)} note={`${stock.day > 0 ? "+" : ""}${stock.day}% today · sample`} tone="dark" />
              <Metric label="SENTIMENT SCORE" value={`${stock.sentiment > 0 ? "+" : ""}${stock.sentiment}`} note={stock.sentiment > 0 ? "Bullish" : "Bearish"} />
              <Metric label="7D MOMENTUM" value={`${stock.momentum > 0 ? "+" : ""}${stock.momentum}`} note={stock.momentum > 0 ? "Rising" : "Falling"} />
              <Metric label="30D SENTIMENT CHANGE" value="+0.18" note="Illustrative" />
            </div>
            <div><span className="eyebrow">RESEARCH THESIS</span><p>Strong user retention and premium conversion trends support the illustrative bull case. Sentiment momentum and both local mock models point to a positive five-day outlook.</p></div>
            <div><span className="eyebrow">KEY RISK</span><p className="fine-print">Competition from AI-powered learning tools and discretionary spending pressure remain the primary sample watchpoints.</p></div>
          </section>
          <section className="panel">
            <div className="section-row"><span className="eyebrow">PRICE CHART</span><div className="tabs">{["1D", "1W", "1M", "1Y"].map((r) => <button className={range === r ? "selected" : ""} key={r} onClick={() => setRange(r)}>{r}</button>)}</div></div>
            <Chart kind="price" /><div className="legend"><span className="green-line">Illustrative closing price · {range}</span></div>
          </section>
          <section className="panel">
            <span className="eyebrow">SENTIMENT VS. PERFORMANCE</span>
            <div className="tabs mode-tabs">{(["Compare", "Overlay"] as const).map((value) => <button className={mode === value ? "selected" : ""} key={value} onClick={() => setMode(value)}>{value}{mode === value ? " ✓" : ""}</button>)}</div>
            <Chart kind="compare" /><p className="fine-print">{mode === "Compare" ? "Separate scales compare direction and shape." : "Price is normalized to the sentiment scale for visual overlay only."} No causal relationship is implied.</p>
          </section>
          <section className="panel reviews">
            <span className="eyebrow">RECENT CUSTOMER REVIEWS</span>
            <Review stars="★★★★★ · 5/5 · Jan 21, 2025" score="+0.94">The gamified learning experience is still the strongest part of the app. Premium adoption is improving.</Review>
            <Review stars="★★★★☆ · 4/5 · Jan 20, 2025" score="+0.88">Retention is holding up better than expected, and the community remains engaged.</Review>
            <Review stars="★★☆☆☆ · 2/5 · Jan 19, 2025" score="-0.31">Concerns around AI competition are still present, but the core product remains sticky.</Review>
          </section>
        </div>
        <aside className="analysis-aside">
          <section className="panel prediction-cards"><span className="eyebrow">PREDICTION CARDS</span><Metric label="NEURAL NETWORK" value={`+${stock.nn}%`} note="Illustrative five-day prediction" /><Metric label="REGRESSION" value={`+${stock.regression}%`} note="Illustrative five-day prediction" /><Metric label="COMBINED" value={`+${stock.combined}%`} note="Illustrative weighted estimate" /></section>
          <section className="panel"><div className="section-row"><span className="eyebrow">FORECAST SPREAD</span><b className="mono positive">{stock.spread} pp</b></div><div className="spread-bar"><i style={{ width: `${Math.max(stock.spread * 15, 8)}%` }} /></div><p className="fine-print">Forecast spread, not confidence. This is the absolute percentage-point difference between model estimates.</p></section>
          <section className="panel"><span className="eyebrow">WHAT IS DRIVING THIS PREDICTION?</span><ul><li>Strong user retention</li><li>Premium conversion growth</li><li>AI learning adoption</li></ul><p className="fine-print">Contributing factors are illustrative, not causal guarantees.</p></section>
          <section className="panel review-analytics"><span className="eyebrow">REVIEW ANALYTICS</span><div className="three-grid"><Metric label="REVIEW COUNT" value="1,248" note="Illustrative sample" tone="dark" /><Metric label="AVG STAR RATING" value="4.7 / 5" note="Illustrative sample" tone="dark" /><Metric label="NEGATIVE REVIEWS" value="12%" note="Illustrative sample" tone="dark" /></div><Chart kind="spark" /><p className="fine-print">Themes, distribution, volume, and review details are illustrative and unverified.</p></section>
        </aside>
      </div>
      <div className="analysis-replay"><HistoricalReplay /></div>
    </main>
  );
}

function Footer() {
  return <footer><div><span className="wordmark">ReplicaVire</span><span>Know the business. Understand the opportunity.</span></div><p>Illustrative design and educational research only. All prices, reviews, predictions, model performance, and historical states are sample or hypothetical—not verified market information or investment advice. No live integrations are connected.</p><small>© 2026 ReplicaVire · RESEARCH DEMONSTRATION / USD</small></footer>;
}

export default function App() {
  const [selected, setSelected] = useState<Stock | null>(null);
  const home = () => { setSelected(null); window.scrollTo({ top: 0, behavior: "smooth" }); };
  return (
    <div className="app">
      <Header onHome={home} />
      {selected ? <Analysis stock={selected} onBack={() => { setSelected(null); smoothScroll("screener"); }} /> : <main><Home onSelect={(stock) => { setSelected(stock); window.scrollTo(0, 0); }} /></main>}
      <Footer />
    </div>
  );
}
