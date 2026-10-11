import { useEffect, useMemo, useState } from "react"
import { apiGet, type Performance as PerformanceData, type Stock } from "../api"
import { buildPortfolio, sigmaOf, type Holding } from "./portfolio"
import {
  Badge,
  Button,
  Icon,
  Onboarding,
  Shell,
  type Profile,
  type Screen,
} from "./parts"
import "./trading.css"

const usd = (v: number) => `$${v.toFixed(2)}`
const pct = (v: number | null | undefined, digits = 2) =>
  v == null || !Number.isFinite(v) ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(digits)}%`
const tone = (v: number | null | undefined) =>
  v == null ? "" : v < 0 ? "negative" : "positive"
const DEFAULT_PROFILE: Profile = {
  budget: 250,
  risk: "Balanced",
  horizon: "5+ years",
  sectors: ["No preference"],
}

function Donut({ holdings, budget }: { holdings: Holding[]; budget: number }) {
  let offset = 25
  return (
    <svg
      className="donut"
      viewBox="0 0 42 42"
      role="img"
      aria-label="Portfolio allocation chart"
    >
      <circle cx="21" cy="21" r="15.9155" fill="none" stroke="#ebe9e2" strokeWidth="5" />
      {holdings.map((item) => {
        const current = offset
        offset -= item.weight
        return (
          <circle
            key={item.stock.ticker}
            cx="21"
            cy="21"
            r="15.9155"
            fill="none"
            stroke={item.color}
            strokeWidth="5"
            strokeDasharray={`${item.weight} ${100 - item.weight}`}
            strokeDashoffset={current}
          />
        )
      })}
      <text x="21" y="19" textAnchor="middle" className="donut-label">
        BUDGET
      </text>
      <text x="21" y="25" textAnchor="middle" className="donut-value">
        ${Math.round(budget)}
      </text>
    </svg>
  )
}

function PickCard({ holding }: { holding: Holding }) {
  const [open, setOpen] = useState(false)
  const s = holding.stock
  return (
    <article className={`pick-card ${open ? "open" : ""}`}>
      <button className="pick-main" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="ticker-tile" style={{ backgroundColor: holding.color }}>
          {s.ticker.slice(0, 2)}
        </span>
        <span className="pick-identity">
          <span>
            <strong>{s.ticker}</strong>
            <Badge>{s.sector}</Badge>
          </span>
          <small>{s.company}</small>
        </span>
        <span className="pick-allocation">
          <small>Allocation</small>
          <strong>
            {holding.weight.toFixed(0)}% <em>{usd(holding.amount)}</em>
          </strong>
        </span>
        {holding.spark ? (
          <svg className="sparkline" viewBox="0 0 96 42" preserveAspectRatio="none">
            <path d={holding.spark} fill="none" stroke={holding.color} strokeWidth="2.2" strokeLinecap="round" />
            <path d={`${holding.spark} L94 42 L2 42 Z`} fill={holding.color} opacity=".08" />
          </svg>
        ) : (
          <span className="sparkline" />
        )}
        <Badge tone="green">
          <Icon name="sparkle" size={13} /> {holding.score}
        </Badge>
        <span className="expand-icon">
          <Icon name="chevron" size={18} />
        </span>
      </button>
      <div className="pick-reason">
        <span>Why it fits</span>
        <p>{holding.reason}</p>
      </div>
      {open && (
        <div className="pick-details">
          <div>
            <small>News sentiment (0-100)</small>
            <strong className={tone(s.sentiment_level == null ? null : s.sentiment_level - 50)}>
              {s.sentiment_level == null ? "—" : s.sentiment_level.toFixed(0)}
            </strong>
          </div>
          <div>
            <small>Expected move ({s.timeframe ?? "—"})</small>
            <strong className={tone(s.expected_pct)}>{pct(s.expected_pct)}</strong>
          </div>
          <div>
            <small>Uncertainty (±1σ)</small>
            <strong>{sigmaOf(s) == null ? "—" : `±${(sigmaOf(s) as number).toFixed(2)}%`}</strong>
          </div>
          <p>
            Last close {s.price == null ? "unavailable" : usd(s.price)} ({s.price_date?.slice(0, 10) ?? "no date"}).
            Forecasts are raw-return estimates over the chosen timeframe, not price targets.
          </p>
        </div>
      )}
    </article>
  )
}

function Results({
  profile,
  stocks,
  loading,
  error,
  onStart,
  onEdit,
}: {
  profile: Profile
  stocks: Stock[]
  loading: boolean
  error: string
  onStart: () => void
  onEdit: () => void
}) {
  const portfolio = useMemo(() => buildPortfolio(stocks, profile), [stocks, profile])
  const { holdings } = portfolio
  return (
    <div className="page">
      <div className="page-heading result-heading">
        <div>
          <span className="eyebrow">Your model-built portfolio</span>
          <h1>{profile.risk} picks, built for you.</h1>
          <p>
            Ranked by the fusion network's conviction, adjusted for forecast uncertainty. Stocks only; the
            tracked universe contains no ETFs or bonds.
          </p>
        </div>
        <div className="profile-summary">
          <div>
            <small>Starting budget</small>
            <strong>{usd(profile.budget)}</strong>
          </div>
          <div>
            <small>Risk style</small>
            <strong>{profile.risk}</strong>
          </div>
          <div>
            <small>Time horizon</small>
            <strong>{profile.horizon}</strong>
          </div>
          <Button kind="ghost" onClick={onEdit}>
            <Icon name="edit" size={16} /> Edit
          </Button>
        </div>
      </div>
      {loading ? (
        <div className="allocation-card">Loading live research data…</div>
      ) : error ? (
        <div className="allocation-card" role="alert">
          <strong>Backend unavailable:</strong> {error}. No portfolio is shown without real data.
        </div>
      ) : holdings.length === 0 ? (
        <div className="allocation-card">
          No stocks currently have a date-matched model forecast, so a portfolio can't be built.
        </div>
      ) : (
        <>
          {portfolio.sectorFallback && (
            <div className="allocation-card" role="status">
              Fewer than three forecasted stocks match your sector choices, so all sectors were considered.
            </div>
          )}
          <section className="allocation-card">
            <div className="section-title">
              <div>
                <span className="section-index">01</span>
                <h2>Your allocation</h2>
              </div>
              <Badge tone="blue">{holdings.length} holdings</Badge>
            </div>
            <div className="allocation-content">
              <Donut holdings={holdings} budget={profile.budget} />
              <div className="allocation-legend">
                {holdings.map((item) => (
                  <div key={item.stock.ticker}>
                    <span className="legend-color" style={{ backgroundColor: item.color }} />
                    <strong>{item.stock.ticker}</strong>
                    <span>{item.stock.company}</span>
                    <em>{item.weight.toFixed(0)}%</em>
                    <small>{usd(item.amount)}</small>
                  </div>
                ))}
              </div>
            </div>
          </section>
          <section className="picks-section">
            <div className="section-title">
              <div>
                <span className="section-index">02</span>
                <h2>What’s inside</h2>
              </div>
              <p>Tap any holding to see the signals behind it.</p>
            </div>
            <div className="pick-list">
              {holdings.map((holding) => (
                <PickCard key={holding.stock.ticker} holding={holding} />
              ))}
            </div>
          </section>
          <div className="start-card">
            <div className="start-icon">
              <Icon name="briefcase" size={25} />
            </div>
            <div>
              <span className="eyebrow">Ready when you are</span>
              <h2>See how the strategy has performed.</h2>
              <p>Review held-out backtest results for the underlying models against SPY.</p>
            </div>
            <Button onClick={onStart}>
              View performance <Icon name="arrow" />
            </Button>
          </div>
        </>
      )}
    </div>
  )
}

function PerformanceChart({ curve }: { curve: NonNullable<PerformanceData["curve"]> }) {
  const w = 840
  const h = 220
  const a = curve.map((p) => (p.nn ?? 0) * 100)
  const m = curve.map((p) => (p.spy ?? 0) * 100)
  const all = [...a, ...m, 0]
  const min = Math.min(...all)
  const max = Math.max(...all)
  const span = max - min || 1
  const x = (i: number) => (w * i) / Math.max(1, curve.length - 1)
  const y = (v: number) => 200 - (180 * (v - min)) / span
  const path = (vs: number[]) => vs.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ")
  const ticks = [0, 1, 2, 3, 4].map((i) => max - (span * i) / 4)
  const mid = curve[Math.floor(curve.length / 2)]
  return (
    <div className="chart-wrap">
      <div className="chart-y-axis">
        {ticks.map((t) => (
          <span key={t}>{t.toFixed(1)}%</span>
        ))}
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="performance-chart" role="img" aria-label="Cumulative backtest return, strategy versus SPY">
        {[20, 65, 110, 155, 200].map((gy) => (
          <line key={gy} x1="0" y1={gy} x2={w} y2={gy} className="grid-line" />
        ))}
        <path d={`${path(a)} L${w} ${h} L0 ${h}Z`} className="chart-area" />
        <path d={path(m)} className="market-line" />
        <path d={path(a)} className="portfolio-line" />
      </svg>
      <div className="chart-x-axis">
        <span>{curve[0].date.slice(0, 10)}</span>
        <span>{mid.date.slice(0, 10)}</span>
        <span>{curve[curve.length - 1].date.slice(0, 10)}</span>
      </div>
    </div>
  )
}

function Performance({ profile, stocks }: { profile: Profile; stocks: Stock[] }) {
  const [data, setData] = useState<PerformanceData | null>(null)
  const [error, setError] = useState("")
  const [selected, setSelected] = useState<Holding | null>(null)
  const holdings = useMemo(() => buildPortfolio(stocks, profile).holdings, [stocks, profile])
  useEffect(() => {
    const ctrl = new AbortController()
    apiGet<PerformanceData>("/api/performance", ctrl.signal)
      .then(setData)
      .catch((e) => {
        if (!ctrl.signal.aborted) setError(e.message)
      })
    return () => ctrl.abort()
  }, [])
  const strat = data?.backtest?.nn
  const spy = data?.backtest?.spy
  const pctOf = (v: number | null | undefined) => (v == null ? null : v * 100)
  const available = data?.status === "available" && data.curve && data.curve.length > 1
  const ahead = strat?.total_return != null && spy?.total_return != null ? 100 * (strat.total_return - spy.total_return) : null
  return (
    <div className="page performance-page">
      <div className="page-heading performance-heading">
        <div>
          <span className="eyebrow">Backtest · held-out test period</span>
          <h1>Strategy performance</h1>
          <p>
            Results are from the research pipeline's cost-aware backtest of the fusion network's daily long-short signals, not a
            live paper account. Past results don't guarantee future returns.
          </p>
        </div>
        <div className="last-updated">
          <span className="status-dot" /> {data?.meta?.trained_at ? `Trained ${data.meta.trained_at.slice(0, 10)}` : "Training date unavailable"}
        </div>
      </div>
      {!available ? (
        <section className="chart-card">
          <h2>Backtest results not available yet</h2>
          <p>{error || data?.reason || "Loading evaluation data…"}</p>
          <p>
            Run <code>python -m src.train</code> in <code>hackathon-trader</code> to generate them. No placeholder
            numbers are shown.
          </p>
        </section>
      ) : (
        <>
          <div className="kpi-grid">
            <article>
              <div>
                <span>Strategy return</span>
                <Icon name="growth" size={18} />
              </div>
              <strong className={tone(strat?.total_return)}>{pct(pctOf(strat?.total_return))}</strong>
              <small>Agreement strategy, test period</small>
            </article>
            <article>
              <div>
                <span>vs. SPY</span>
                {ahead != null && <Badge tone="blue">{ahead >= 0 ? "Ahead" : "Behind"}</Badge>}
              </div>
              <strong className={tone(ahead)}>{pct(ahead)}</strong>
              <small>SPY {pct(pctOf(spy?.total_return))}</small>
            </article>
            <article>
              <div>
                <span>Sharpe ratio</span>
                <Icon name="chart" size={18} />
              </div>
              <strong>{strat?.sharpe == null ? "—" : strat.sharpe.toFixed(2)}</strong>
              <small>Non-overlapping periods</small>
            </article>
            <article>
              <div>
                <span>Max drop</span>
                <Icon name="chart" size={18} />
              </div>
              <strong>{pct(pctOf(strat?.max_drawdown))}</strong>
              <small>Worst peak-to-trough</small>
            </article>
          </div>
          <section className="chart-card">
            <div className="chart-header">
              <div>
                <h2>Cumulative return</h2>
                <div className="chart-legend">
                  <span>
                    <i className="dot-blue" />
                    Agreement strategy
                  </span>
                  <span>
                    <i className="dot-dark" />
                    SPY
                  </span>
                </div>
              </div>
            </div>
            <PerformanceChart curve={data.curve!} />
            <p className="paper-note">
              Training {data.meta?.train_dates?.join(" → ") ?? "—"} · Test {data.meta?.test_dates?.join(" → ") ?? "—"}.{" "}
              {data.meta?.caveat ?? ""}
            </p>
          </section>
        </>
      )}
      {holdings.length > 0 && (
        <div className="dashboard-grid">
          <section className="holdings-card">
            <div className="section-title">
              <div>
                <h2>Proposed holdings</h2>
                <Badge>{holdings.length}</Badge>
              </div>
              <p>Click a holding for its model signals.</p>
            </div>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Holding</th>
                    <th>Amount</th>
                    <th>Est. shares</th>
                    <th>Last close</th>
                    <th>1D</th>
                    <th>Weight</th>
                  </tr>
                </thead>
                <tbody>
                  {holdings.map((h) => (
                    <tr key={h.stock.ticker} onClick={() => setSelected(h)}>
                      <td>
                        <span className="table-ticker">{h.stock.ticker}</span>
                        <small>{h.stock.company}</small>
                      </td>
                      <td>{usd(h.amount)}</td>
                      <td>{h.shares.toFixed(3)}</td>
                      <td>{h.stock.price == null ? "—" : usd(h.stock.price)}</td>
                      <td className={tone(h.stock.day_change_pct)}>{pct(h.stock.day_change_pct)}</td>
                      <td>
                        <div className="weight-bar">
                          <span style={{ width: `${h.weight}%` }} />
                        </div>
                        {h.weight.toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      )}
      {selected && (
        <>
          <button className="panel-scrim" aria-label="Close details" onClick={() => setSelected(null)} />
          <aside className="why-panel">
            <button className="panel-close" onClick={() => setSelected(null)} aria-label="Close panel">
              <Icon name="close" />
            </button>
            <span className="eyebrow">Model signals</span>
            <div className="panel-title">
              <span className="ticker-tile" style={{ backgroundColor: selected.color }}>
                {selected.stock.ticker.slice(0, 2)}
              </span>
              <div>
                <h2>{selected.stock.ticker}</h2>
                <p>{selected.stock.company}</p>
              </div>
            </div>
            <div className="score-block">
              <span>
                <Icon name="sparkle" /> Rank score
              </span>
              <strong>
                {selected.score}
                <small>/100</small>
              </strong>
              <div>
                <span style={{ width: `${selected.score}%` }} />
              </div>
            </div>
            <div className="panel-section">
              <h3>Why this pick?</h3>
              <p>{selected.reason}</p>
            </div>
            <div className="signal-list">
              <div>
                <span>Expected move ({selected.stock.timeframe ?? "—"})</span>
                <strong className={tone(selected.stock.expected_pct)}>{pct(selected.stock.expected_pct)}</strong>
              </div>
              <div>
                <span>Conviction</span>
                <strong>{selected.stock.height == null ? "—" : selected.stock.height.toFixed(1)}</strong>
              </div>
              <div>
                <span>News items (5D)</span>
                <strong>{selected.stock.news_count_5d ?? "—"}</strong>
              </div>
            </div>
            <div className="panel-note">
              <Icon name="shield" />
              <p>This analysis is for research and simulation, not investment advice.</p>
            </div>
          </aside>
        </>
      )}
    </div>
  )
}

export default function TradingDashboard({
  stocks,
  loading,
  error,
  onExit,
}: {
  stocks: Stock[]
  loading: boolean
  error: string
  onExit: () => void
}) {
  const [screen, setScreen] = useState<Screen>("onboarding")
  const [profile, setProfile] = useState<Profile>(DEFAULT_PROFILE)
  return (
    <div className="tv">
      <Shell screen={screen} setScreen={setScreen} onExit={onExit}>
        {screen === "onboarding" && (
          <Onboarding
            initial={profile}
            onComplete={(p) => {
              setProfile(p)
              setScreen("results")
              window.scrollTo(0, 0)
            }}
          />
        )}
        {screen === "results" && (
          <Results
            profile={profile}
            stocks={stocks}
            loading={loading}
            error={error}
            onStart={() => setScreen("performance")}
            onEdit={() => setScreen("onboarding")}
          />
        )}
        {screen === "performance" && <Performance profile={profile} stocks={stocks} />}
      </Shell>
    </div>
  )
}
