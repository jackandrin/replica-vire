# ReplicaVire trader (v2)

ML paper-trading pipeline: **Alpaca 15-min bars + Alpaca/Finnhub news + SEC EDGAR filings + FINRA short volume →
Tiger Data → 15 per-family regressors + Snowflake Cortex sentiment agent → fusion neural network (6 horizons) →
direction vector → recommendations / Alpaca paper orders → React dashboard**. Paper trading only. Not financial advice.

- Training on USC Theia: **[docs/TRAINING.md](docs/TRAINING.md)**
- Data contract (tables, bundle files, time semantics): **[docs/V2_CONTRACT.md](docs/V2_CONTRACT.md)**
- Design rationale: [../PLAN_V2.md](../PLAN_V2.md). `PLAN.md` is the original 24-hour hackathon plan (v1, historical).

## Architecture

```
 ingest_intraday ─┐                                   ┌─ 15 family ridges (next-day) ─┐
 ingest_events ───┼─► Tiger Data ─► featurize ────────┤                               ├─► fusion NN ─► (mu, sigma) x 6
 ingest_docs ─────┘        │        (point-in-time)    └─ sentiment + context ─────────┘     horizons
        │                  ▼                                                                  │
        └──► sentiment_agent (Cortex / local Llama): RAG over prior coverage → level + trend  ▼
                                                                    signals.py: horizon choice, conviction, sizing
                                                                                              │
                         React dashboard ◄── web_api.py ◄── signals / alerts ◄── autotrade.py ─┴─► Alpaca PAPER
```

| Layer | Module |
|---|---|
| Universe (5 per GICS sector + SPY, VIXY, 11 sector ETFs) | `sectors.py`, `config.yaml: tickers` |
| Ingestion | `ingest_intraday.py`, `ingest_events.py`, `ingest_docs.py`, `edgar.py`, `export_dataset.py` |
| Sentiment agent | `sentiment_agent.py`, `llm_backends.py` |
| Features + labels (6h, 12h, 18h, 1d, 3d, 5d) | `featurize.py` |
| Level 1: 15 regressors, walk-forward out-of-fold | `regressors.py` |
| Level 2: fusion network 32→64→32→16→12, integrated gradients | `fusion.py` |
| Training / evaluation | `train.py`, `evaluate.py`, `splits.py` |
| Live inference | `predict.py` → `signals` table |
| Decisions, alerts, paper orders | `signals.py`, `autotrade.py` |
| Live cycle | `run_cycle.py` + `deploy/hackathon-cycle.timer` |
| Dashboard API | `web_api.py` (frontend in `../frontend`) |

## Setup

```bash
cd hackathon-trader && ./setup.sh          # Ubuntu: venv, torch CPU, deps, .env, offline tests (--systemd for the timer)
```
Windows: `powershell -ExecutionPolicy Bypass -File .\setup.ps1`. Then fill in `.env`:

| Service | `.env` variable(s) | Notes |
|---|---|---|
| Alpaca (paper keys) | `ALPACA_API_KEY`, `ALPACA_API_SECRET` | 15-min + daily bars, news, paper orders |
| Finnhub (free) | `FINNHUB_API_KEY` | news (~1 year), analyst recommendation trends |
| SEC EDGAR | `SEC_USER_AGENT` | "Your Name your@email" (required by SEC fair-access policy) |
| Tiger Data | `TIGER_DATABASE_URL` | keep `?sslmode=require` |
| Snowflake | `SNOWFLAKE_*` | Cortex `COMPLETE` + `EMBED_TEXT_768` for the live sentiment agent |
| Dashboard | `DASHBOARD_PASSPHRASE`, `SESSION_SECRET` | login for Auto-Trading settings / executing recommendations |

FINRA short-volume files need no key.

## Commands

| Step | Command |
|---|---|
| Tables | `python -m src.init_db --tiger` |
| API check | `python -m src.check_apis` |
| Download all training data + export bundle | `bash scripts/backfill.sh [--bg]` |
| Sentiment agent (live, Cortex) | `python -m src.sentiment_agent` |
| Sentiment backfill on a GPU | `sbatch deploy/sentiment.sbatch` (see docs/TRAINING.md) |
| Train | `python -m src.train` (or `sbatch deploy/theia.sbatch`) |
| Publish a trained model to Tiger Data (website + bot use it) | `python -m src.publish_models` |
| Predict latest bar (newest published model) | `python -m src.predict` |
| Recommendations / paper orders | `python -m src.autotrade [--no-orders]` |
| Live cycle | `python -m src.run_cycle [--dry-run]` |
| Everything on one machine | `bash scripts/run_all.sh [--place]` |
| API / dashboard | `python -m uvicorn src.web_api:app --port 8000` and `cd ../frontend && npm run dev` |
| Tests (offline) | `pytest -q` |

## How it works

**Decisions** happen at 10:00, 11:00, …, 15:00 and 15:45 ET. A decision at `t` sees only bars that ended by `t`,
daily bars of earlier days (plus today's open), and events / sentiment whose `available_at` / `published_at` ≤ `t`
(`tests/test_v2_pipeline.py::test_no_lookahead` rebuilds features with the future deleted and checks nothing changes).

**Labels.** 6h/12h/18h are wall-clock; when `t+h` falls outside the session the price rolls forward to the next
open ("first price tomorrow"). 1d/3d/5d are the same time 1/3/5 trading days later. Raw returns.

**Level 1.** 15 ridge regressions, one per evidence family (reversal, overnight/intraday, intraday seasonality, market
momentum, sector lead-lag, abnormal volume, volatility, RSI/Bollinger, VWAP/MAs, trend, 52-week high, earnings,
analyst ratings, short selling, insider trading), each predicting next-day return. Their walk-forward out-of-fold
predictions feed level 2, so the network never sees in-sample regressor outputs.

**Level 2.** 32 inputs (15 regressor outputs, 4 sentiment, 13 context) → 64 → 32 → 16 → `(mu, log σ²)` for 6
horizons, Gaussian NLL, early stopping. A per-horizon ridge on all raw features is the baseline it must beat.

**Sentiment agent.** For each new document the LLM sees the stock's prior view (level + summary), recent events and
the most similar earlier coverage (retrieval, point-in-time) and reports how sentiment *changed*: direction, delta,
updated level, novelty, materiality, drivers. Features: time-decayed level, 24h change, 5d change, intensity.

**Decisions.** Per horizon: `score = sign(mu)·max(|mu| − cost, 0)/σ`, `height = |score| / validation scale`. The
strongest allowed horizon becomes the direction vector `(expected %, timeframe)`; order size scales from the user's
min $ to max $ with height. Horizons that exit the same session (day trades) need `day_trade_min_height`.
Long/short/both filters, sector filters, max positions, planned exits at the horizon, closes on flipped signals.

## Safety

- Paper only: `TradingClient(..., paper=True)` is hard-coded. `TRADING_ENABLED=0` in `.env` blocks all orders.
- The website never talks to Alpaca: it only queues alerts; the worker executes them.
- `run_cycle` stops before trading if bars or predictions fail; `autotrade` refuses new positions on signals
  older than 3 hours.

## Limitations

Daily and intraday returns are mostly noise: expect small ICs (0.00–0.05). A single held-out period; survivorship
in a hand-picked large-cap universe; free-tier data gaps (Finnhub analyst history is short, IEX volume if SIP is
refused); LLM sentiment on historical articles may carry look-ahead from the model's training data; paper fills
ignore slippage.
