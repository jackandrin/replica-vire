# ReplicaVire v2 — Plan (draft for review)

Status: **DRAFT — nothing implemented yet.** Builds on the existing `hackathon-trader/` pipeline and `frontend/` app.
Open decisions are marked **[DECIDE]** and collected in §12.

---

## 0. What changes from v1

| Area | v1 (today) | v2 (this plan) |
|---|---|---|
| Bars | Daily, Alpaca IEX | **15-min bars** (+ hourly/daily continuous aggregates in Tiger Data) |
| Target | 5-day excess return vs SPY | Returns at **6h, 12h, 18h, 1d, 3d, 5d**, each with μ and σ |
| Level-1 models | 1 Ridge on 15 features | **15 per-family regressors**, each predicting next-day return (kept for explanations) |
| Sentiment | VADER/Cortex score per headline (−1..1) | **RAG agent on Cortex** that judges *how sentiment changed* vs. what it already knew → level + trend |
| Fusion | MLP 32→16 → (μ, σ) | **NN 32 → 64 → 32 → 16 → 12** (μ, σ for 6 horizons) |
| Decision | Agreement filter, weekly rebalance | **Direction vector** (expected %, timeframe, height) → sized between user min/max |
| UI | Read-only research site | **Signals, Stock detail with "why", Recommendations + top toasts, Auto-trading, Performance** |
| Runs | Once a day at 16:20 ET | Worker every 15 min during extended hours |

Kept as-is: Tiger Data as the system of record, Alpaca **paper** trading (`paper=True` stays hard-coded), Finnhub + EDGAR + Alpaca news, leakage tests, honest evaluation, ridge fallback.

---

## 1. Architecture

```
                     ┌──────────────────────── WORKER (Ubuntu box, systemd timers) ─────────────────────────┐
 Alpaca bars ──────► │ ingest_bars (15m) ─► features ─► 15 regressors ─┐                                    │
 FINRA short vol ──► │ ingest_events (earnings, analyst, insider, short vol)                                │
 Finnhub/Alpaca news►│ ingest_docs ─► chunk+embed (Cortex) ─► RAG sentiment (Cortex COMPLETE) ─► sentiment  │
 SEC EDGAR ────────► │                                                  │                         state    │
                     │                                  32-dim input ◄──┴──── context features ◄────┘       │
                     │                                       │                                              │
                     │                          Fusion NN (μ,σ × 6 horizons) + attribution                  │
                     │                                       │                                              │
                     │       decision engine: direction vector → sizing → alerts / orders → exit manager    │
                     └───────────────────────────────────────┼──────────────────────────────────────────────┘
                                                             ▼
                                  TIGER DATA (TimescaleDB + pgvector) — all data, models' outputs, settings
                                                             ▲
                     FastAPI on Vercel (reads + authenticated writes to settings / order_requests)
                                                             ▲
                     React dashboard (polls signals & alerts; toasts; auto-trading controls)
```

Key design choice: **Vercel never talks to Alpaca.** The website only writes rows (settings, "execute this recommendation" requests) to Tiger Data; the worker, which holds the Alpaca keys, executes them. That keeps trading keys off the public site, and Vercel's serverless functions can't run schedulers or hold websockets anyway.

---

## 2. Data sources

| Data | Source | Notes |
|---|---|---|
| 15-min bars, stocks + SPY + 11 sector ETFs | Alpaca | Try `feed=sip` for history (free plan allows SIP if the request ends ≥15 min ago); `iex` for the live edge. No tick data needed for these horizons. |
| Extended hours (4:00–20:00 ET) | Alpaca | Needed because 12h/18h targets land outside the regular session. IEX after-hours prints are thin. |
| News (headline, summary, body) | Alpaca news (`include_content=true`), Finnhub company-news | Existing `ingest_news.py`, extended to keep bodies. |
| Filings | SEC EDGAR (`data.sec.gov` submissions JSON + filing docs) | 8-K, 10-Q, 10-K, Form 4. Use **acceptance timestamp**. 10 req/s, User-Agent header required. |
| Earnings dates + EPS surprises | Finnhub `/calendar/earnings`, `/stock/earnings` | |
| Analyst recommendation trends | Finnhub `/stock/recommendation` | Monthly granularity on free tier. |
| Insider transactions | EDGAR Form 4 (or Finnhub `/stock/insider-transactions`) | |
| Daily short-sale volume | FINRA Reg SHO daily files (free) | Published after the close → usable from the next day only. |
| Sector labels | Static `symbols.csv` (GICS sector per ticker) | |

`check_apis.py` gets one probe per new endpoint so we confirm free-tier access before building on it.

**Universe** **[DECIDE]**: expand from 20 to **55 stocks = 5 large caps per GICS sector**, plus SPY and the 11 SPDR sector ETFs (XLK, XLF, XLV, XLY, XLP, XLE, XLI, XLB, XLU, XLRE, XLC). The current 20 have no Materials, Utilities, Real Estate or Industrials names, so most sector filters would be empty.

---

## 3. Horizons and labels

Decision times: every hour on the half hour, 9:30–15:30 ET (7 per day), later extendable to premarket/after-hours.

Horizons: **6h, 12h, 18h, 1d (24h), 3d, 5d** in wall-clock time (3d/5d = trading days).

Label for horizon h at decision time t = `price(first tradeable bar at or after t+h) / price(t) − 1`.

**[DECIDE] the overnight problem.** The market is closed 20:00–4:00 ET and weekends. From a 10:00 decision, 18h lands at 4:00 next morning; from 14:00, 12h lands at 2:00 AM. With the rule above those targets roll forward to the next available price, so 12h and 18h often both mean "first premarket price tomorrow." Options:
- **(a) Recommended:** keep the rule, allow exits in extended hours with limit orders. 12h and 18h stay distinct for some decision times, and the model learns which horizon is reachable.
- (b) Express 6h/12h/18h in *market hours* (regular session = 6.5h), so 12h ≈ 2 sessions. Cleaner labels, but "6h" then means nearly a full trading day.

Train on raw returns (what you actually earn). Market and sector context are inputs, so the NN can separate market moves from stock-specific ones.

---

## 4. The 15 regressors (next-day predictors)

Each regressor is a small **ridge regression on one feature family** (1–4 raw features) predicting **next-day (1d) return**. Keeping each family separate is what makes the per-reason explanation possible.

I could find 15 with published evidence and free data. In my judgment, **#1–6 and #14 have the strongest next-day evidence**. #7–10 overlap heavily with #1, because RSI, %B and MA-distance are transforms of recent returns. #11, #13 and #15 are slow or sparse event signals.

| # | Family | Features | Evidence | Data |
|---|---|---|---|---|
| 1 | Short-term reversal | 1d, 5d return (raw and vs sector) | Jegadeesh 1990; Lehmann 1990 | bars |
| 2 | Overnight vs intraday split | last overnight gap, last intraday return | Lou, Polk & Skouras 2019 | bars |
| 3 | Intraday seasonality | return in the same half-hour slot on past days | Heston, Korajczyk & Sadka 2010 | 15-min bars |
| 4 | Market intraday momentum | SPY first-30-min return, SPY prior-day return × beta | Gao, Han, Li & Zhou 2018 | SPY bars |
| 5 | Sector lead–lag | sector ETF 1d return, stock residual vs sector | Moskowitz & Grinblatt 1999; Hou 2007 | sector ETFs |
| 6 | Abnormal volume | volume / 20d avg (same time of day) | Gervais, Kaniel & Mingelgrin 2001 | bars (ratio only, because IEX volume is partial) |
| 7 | Volatility state | 5d/21d realized vol, ATR, idiosyncratic vol | Ang, Hodrick, Xing & Zhang 2006 | bars |
| 8 | Oscillators | RSI(2), RSI(14), Bollinger %B | practitioner; overlaps #1 | bars |
| 9 | VWAP / MA distance | price vs session VWAP, vs 10/20/50-day MA | practitioner | bars |
| 10 | Trend | MACD histogram, 21d momentum | Jegadeesh & Titman 1993 (longer horizon) | bars |
| 11 | 52-week-high proximity | close / 52w high | George & Hwang 2004 | daily agg |
| 12 | Earnings event | days to/since earnings, last EPS surprise (SUE) | Bernard & Thomas 1989 (PEAD) | Finnhub |
| 13 | Analyst changes | change in recommendation score | Womack 1996 | Finnhub |
| 14 | Short-sale activity | FINRA short-volume ratio vs 20d avg | Diether, Lee & Werner 2009 | FINRA |
| 15 | Insider trading | net insider buys, 30d, opportunistic trades only | Cohen, Malloy & Pomorski 2012 | EDGAR Form 4 |

Deliberately excluded: P/E, revenue growth, margins and other fundamentals. They don't move next-day prices except through earnings events, which #12 already captures.

Expectations: each regressor will have a tiny R² (≈0.001–0.01). That is normal for daily returns, not a bug.

Built-in check: after training, compute the correlation matrix of the 15 out-of-fold predictions. Any pair above 0.8 gets merged (most likely #1/#8/#9). The freed slot can go to **liquidity** (Amihud 2002) or a **VIX-proxy regime** regressor.

Each regressor stores per signal: `yhat_k`, and per-feature contributions `coef × standardized value` (used in the explanation).

---

## 5. Sentiment RAG agent (Snowflake Cortex + Tiger Data)

Goal (as you specified): judge **how sentiment is moving**, not an absolute percentage. A stock going from 100 → 80 is still "positive" but trending down, and that matters.

**Storage (all in Tiger Data):** `documents` (raw text, source, `published_at`, URL, hash) and `doc_chunks` (text + `vector(768)` embedding, pgvector). Embeddings come from Cortex `EMBED_TEXT_768`; vectors are written back to Tiger.

**Per new document for ticker T at time t:**
1. Dedupe syndicated copies by normalized-headline hash.
2. Retrieve context **with `published_at < t` only** (point-in-time):
   - the ticker's current *sentiment memory* (running summary + level),
   - the top-k most similar chunks from the last 90 days,
   - the last few sentiment events.
3. Call Cortex `COMPLETE` with a JSON-schema response:
   ```json
   { "direction": "improving|deteriorating|stable",
     "delta": -1.0..1.0,          // change vs. prior view, not an absolute score
     "level": 0..100,             // updated overall view
     "novelty": 0..1,             // new info vs. rehash of known news
     "materiality": 0..1,         // could this move the price?
     "horizon": "hours|days|weeks",
     "drivers": ["guidance cut", "..."],
     "updated_summary": "..." }
   ```
4. Write a `sentiment_events` row and update `sentiment_state`.

Asking the model for change relative to what it already knew is also more reliable than asking for an absolute score each time.

**Features sent to the NN (4):** `level` (materiality×novelty-weighted EWMA), `trend_24h`, `trend_5d` (time-decayed slopes of level), `intensity` (weighted doc count vs normal).

**Cost and risk controls:**
- A cheap model triages materiality; only material docs go to the larger model.
- Store `model`, `prompt_version` and raw output with every event.
- Backfill: start with 1 year of headlines and summaries; estimate Cortex credits on 500 docs first.
- Use the **same model** for backfill and live, so training and serving see the same distribution.
- **Known bias:** on historical articles the LLM may already "know" what happened afterward, so backtested sentiment value is optimistic. We disclose that on the Performance page.

"Reviews" as a source: **[DECIDE]** (see §12).

---

## 6. Fusion network

**Input (32):**
| Block | Count | Contents |
|---|---|---|
| Regressor outputs | 15 | `yhat_1..15` (out-of-fold during training) |
| Sentiment | 4 | level, trend_24h, trend_5d, intensity |
| Context | 13 | SPY 1d return, SPY 21d vol, VIX-proxy level and change, sector ETF 5d trend, beta, log dollar volume, spread proxy, minutes to close, session flag, day-of-week (sin, cos), days to earnings |

**Network:** 32 → 64 → 32 → 16 (GELU, LayerNorm, dropout 0.2) → **12 outputs** = (μ, log σ²) for each of the 6 horizons. Loss: Gaussian NLL summed over horizons. This extends the existing `MLPModel` in `models.py`.

**Training protocol (stacking without leakage):**
- Walk-forward with an expanding window and monthly refits.
- Regressors are fitted on past folds only, and their out-of-fold predictions become NN training inputs.
- Purge gap ≥ 5 trading days (the longest horizon) between train and test; extend `splits.py`.
- **Baseline:** one ridge on all raw features, per horizon. The NN ships only if it beats the baseline on **validation** IC. Otherwise ridge drives signals (existing fallback culture).

**Attribution:** Integrated Gradients (Captum) over the 32 inputs for every live signal. This tells us which regressors and sentiment features the NN actually relied on, so the explanation cites those rather than raw regressor outputs that the NN may have overridden.

---

## 7. Decision engine: the direction vector and sizing

For each (stock, decision time):
1. For each horizon h, compute a cost-adjusted score `z_h = (μ_h − cost_h) / σ_h`.
2. Pick `h* = argmax |z_h|` among horizons the user allows.
3. **Direction vector** = `(expected % change = μ_h*, timeframe = h*)`.
   **Height** = `|z_h*|`, i.e. conviction in σ units.

   I made height confidence-adjusted on purpose. A big predicted move with huge uncertainty should not get max size.
4. **Size:**
   ```
   height < h_min                 → no trade
   size = min_usd + (max_usd − min_usd) × clip((height − h_min) / (h_full − h_min), 0, 1)
   ```
   `min_usd` and `max_usd` come from user settings. `h_min` and `h_full` are config values, calibrated on validation data.
5. **Portfolio guards** (applied after sizing): buying power, max open positions, max exposure per sector, max total exposure, **daily loss limit → auto-disable**, global kill switch (`TRADING_ENABLED=0`, already exists).
6. **Direction rules** (default **long-only** **[DECIDE]**):
   - μ > 0 → buy.
   - μ < 0 on a held position → sell.
   - μ < 0 on a non-held stock → "avoid", no order.
7. **Exits** (exit manager runs every 5 min):
   - at `entry_time + h*`,
   - or earlier if the signal flips with height ≥ `h_min`,
   - or on a stop at −k·σ.
   Extended-hours exits use limit orders.
8. **Pattern-day-trader rule:** a 6h round trip is a day trade. Alpaca paper simulates PDT limits for accounts under $25k. The engine counts day trades and skips same-day exits when the limit would be hit. FINRA has been revising this rule, so we'll check the current status when we build it.

---

## 8. Explanations ("why")

Every signal stores an `explanation` JSON:
- top 3 attributed inputs, with sign and size,
- for each regressor among them, its top feature contributions,
- sentiment `drivers` plus links to source documents.

The rendered sentence comes from a **template**, e.g.:

> **SELL NVDA by 2:30 PM ET** — expected −1.4% over 6h (1.8σ). Why: strong 5-day reversal signal (+3.9% run-up), sentiment trending down after this morning's 8-K (−12 pts/24h), volume 2.3× normal.

An optional Cortex rewrite can smooth the wording, but only from those facts, with no new numbers. The template is the default because it can't hallucinate.

---

## 9. Dashboard (extends `frontend/`)

Global elements: GICS **sector filter chips** (Energy, Materials, Industrials, Consumer Discretionary, Consumer Staples, Health Care, Financials, Information Technology, Communication Services, Utilities, Real Estate). "Manufacturing" maps to Industrials (+ Materials). Also a paper-trading badge, kill-switch status and the toast layer.

| Page | Contents |
|---|---|
| **Signals** (replaces Screener) | Table: ticker, sector, direction arrow, expected %, timeframe, height bar, suggested $, sentiment level and trend sparkline, status (held / recommended / avoid). Sector chips filter it. |
| **Stock detail** | 15-min/hourly price chart with a forecast fan across the 6 horizons. "Why" panel with attribution bars and the template sentence. All 15 regressor outputs. Sentiment timeline (level + trend) with clickable LLM drivers and source docs. |
| **Recommendations** | Feed of alerts (buy/sell/exit) with reason, expected %, by-time. Actions: **Execute (paper)** / Dismiss / Details. Execute creates an `order_request`, which the worker places. |
| **Auto-Trading** | Enable toggle; sectors; **min / max $ per trade**; max open positions; max total and per-sector exposure; daily loss limit; minimum height; allowed horizons; long-only vs shorts; **paper (locked)**. Below: open positions with exit countdowns, an activity log, and a big Stop button. |
| **Performance** | Per-horizon IC, hit rate, calibration (does ±1σ cover ~68%?), cost-aware backtest, NN vs ridge baseline, and the LLM look-ahead caveat. |

**Toast popups:** slide down from the top center and stack up to 3. Each shows ticker, action, by-time and a one-line reason, with Execute / Details / Dismiss buttons. They auto-hide after 12s unless hovered. The frontend polls `GET /api/alerts?since=<last_id>` every 15s; I chose polling because Vercel can't hold websockets. An optional browser Notification fires when the tab is in the background. In auto mode the toasts report executed trades ("Bought $420 AAPL — …").

**Auth:** the API gains write endpoints that control trading, so they can't be public. For the hackathon: a single-user passphrase login → signed HTTP-only cookie, checked on all non-GET routes **[DECIDE]**.

**API additions** (`web_api.py`):
- `GET /api/signals?sectors=`
- `GET /api/stocks/{sym}` (extended)
- `GET /api/alerts?since=`, `POST /api/alerts/{id}/dismiss`
- `GET|PUT /api/settings`
- `POST /api/order-requests`
- `GET /api/positions`
- `POST /api/kill-switch`
- `POST /api/login`
- `GET /api/performance` (per horizon)

---

## 10. Tiger Data schema additions

| Table | Type | Purpose |
|---|---|---|
| `symbols` | table | ticker, name, GICS sector, CIK, is_etf |
| `bars_15m` | hypertable | replaces daily `bars` as the source; `bars_1h`, `bars_1d` become **continuous aggregates** |
| `events` | hypertable | earnings, analyst changes, insider trades, short volume (typed rows with `available_at`) |
| `documents`, `doc_chunks` | table + pgvector | RAG corpus |
| `sentiment_events`, `sentiment_state` | hypertables | per-doc judgments; rolling level/trend |
| `features_intraday` | hypertable | raw family features per (symbol, decision ts) |
| `regressor_outputs` | hypertable | `yhat` + contributions per regressor |
| `forecasts` | hypertable | μ, σ per horizon, model version |
| `signals` | hypertable | direction vector, height, size, explanation JSON |
| `alerts` | table | toast/recommendation feed |
| `user_settings` | table | auto-trading config |
| `order_requests` | table | website → worker queue |
| `orders` (extend), `positions_meta` | table | adds horizon and planned exit time |

Every time-stamped row carries both **event time** and **available time**. Leakage tests check that features at t only use rows with `available_at ≤ t`.

---

## 11. Build phases (4-person team, roles from PLAN.md)

| Phase | Owner | Work | Done when |
|---|---|---|---|
| **P0 Foundation** | A | schema v2, `symbols.csv`, 15-min bar backfill, continuous aggregates, new API probes | `bars_15m` filled for the universe; `check_apis` green |
| **P1 Features + regressors** | B → C | 15 families, 6-horizon labels, walk-forward OOF ridges, collinearity report, leakage tests | OOF predictions stored; tests green |
| **P2 Sentiment RAG** (parallel with P1) | B/A | docs + chunks + embeddings, prompt + JSON schema, state updater, 1-year backfill | `sentiment_state` populated; 20 events hand-checked |
| **P3 Fusion NN** | C | 32→64→32→16→12, NLL, baseline comparison, Integrated Gradients | Validation IC per horizon reported vs ridge |
| **P4 Decision + worker** | D | direction vector, sizing, guards, alerts, order queue, exit manager, systemd timers (15-min) | Paper orders placed and exited automatically for one day |
| **P5 API + dashboard** (starts at P0 using mocked JSON) | D + frontend | 5 pages, toasts, auth, settings | Full click-through on live data |
| **P6 Evaluate + harden** | all | per-horizon metrics, backtest after costs, demo script, backup recording | Code freeze |

**MVP cut line**, if time runs short, in this order:
1. Ship the 7 strongest regressors (#1–6, #14) instead of 15.
2. Run sentiment on headlines and summaries only.
3. Keep ridge driving signals if the NN doesn't beat it.
4. Ship Recommendations + toasts before Auto-Trading.

A working simple system beats a broken complex one, which is the same rule as in PLAN.md §4.

---

## 12. Decisions needed from you

1. **Time left / demo time?** This sets how much of P1–P5 is realistic. The existing PLAN.md assumes a 24-hour hackathon.
2. **12h/18h semantics:** (a) wall-clock with roll-forward and extended-hours exits (recommended), or (b) market hours.
3. **Universe:** 55 stocks (5 per GICS sector, recommended), or keep the current 20?
4. **Long-only or allow shorts?** Default: long-only.
5. **"Reviews":** which reviews? Analyst reviews, product/app reviews (like the old Google Play collector), or skip?
6. **Auth:** single-user passphrase OK for the hackathon?
7. **Paper only:** I'm keeping `paper=True` hard-coded. Real-money auto-trading is out of scope for this plan.
