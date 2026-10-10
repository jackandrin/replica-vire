# Hackathon Trader

A 24-hour ML paper-trading pipeline: **Alpaca + Finnhub → TigerData → features/sentiment → Ridge & tiny MLP → agreement filter → Alpaca paper orders → Streamlit dashboard**, with results mirrored to **Snowflake**.

It predicts the **5-day excess return vs SPY** for 20 liquid US stocks, trades only when two different models agree, and reports results honestly (IC, hit rate, cost-aware backtest vs SPY, known limitations). Paper trading only. Not financial advice.

> Hour-by-hour schedule, roles, fallbacks, demo script: **[PLAN.md](PLAN.md)**

## Architecture

```
 Alpaca bars ─┐                                   ┌─> Ridge ──┐
              ├─> TigerData (TimescaleDB) ─> features/labels ─┤           ├─> agreement filter ─> sizing ─> Alpaca PAPER orders
 Finnhub +   ─┘    bars, news(+sentiment)        (pandas)    └─> MLP[μ,σ]┘                                   │
 Alpaca news                    │                                  │                                          │
                                └────────── Snowflake (FEATURES, LABELS, PREDICTIONS, ORDERS) <───────────────┘
                                                      Streamlit dashboard <── TigerData + Alpaca + results/
```

| Layer | Tech | Module |
|---|---|---|
| Raw time-series | TigerData (managed TimescaleDB, `bars` hypertable) | `ingest_bars.py`, `ingest_news.py`, `sql/tiger_schema.sql` |
| Sentiment | VADER (default) or Snowflake Cortex `SENTIMENT` | `sentiment.py` |
| Features / labels | pandas (leakage-tested) | `features.py` |
| Models | scikit-learn Ridge, PyTorch MLP (32→16, dropout, early stopping, Gaussian NLL) | `models.py`, `train.py` |
| Evaluation | IC, hit rate, non-overlapping backtest vs SPY | `evaluate.py` |
| Decisions | agreement filter + conviction sizing | `strategy.py`, `decide.py` |
| Warehouse | Snowflake | `sync_snowflake.py` |
| UI | Streamlit | `dashboard.py` |
| Ops | systemd service + timer, tmux | `deploy/`, `setup.sh` |

## Quick start on Ubuntu (22.04 / 24.04)

### Option A: single-file installer (recommended)

The bundled `install.sh` contains the whole project. Copy that one file to the server and run it:

```bash
scp install.sh user@SERVER:~
ssh user@SERVER
bash install.sh                 # installs to ~/hackathon-trader
# or: bash install.sh /opt/trader --systemd      (custom dir + systemd units)
```

It installs apt packages (python3, venv, build tools, git, tmux), creates `.venv`, installs the CPU build of PyTorch, installs `requirements.txt`, creates `.env` (mode 600), and runs the offline tests.

### Option B: from a folder / git clone

```bash
cd hackathon-trader
chmod +x setup.sh scripts/*.sh
./setup.sh                      # flags: --systemd  --skip-apt  --no-torch
```

### Option C: Windows (develop locally first, then move to the server)

```powershell
cd hackathon-trader
powershell -ExecutionPolicy Bypass -File .\setup.ps1      # venv, torch CPU, deps, .env, tests
notepad .env                                              # fill in keys
powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1   # whole pipeline (dry-run orders)
.\.venv\Scripts\streamlit run src\dashboard.py            # dashboard
```

Use any `python -m src.<module>` command from the table below via `.\.venv\Scripts\python -m src.<module>`.
To move to Ubuntu later: run `bash build_installer.sh` (Git Bash/WSL) and copy `dist/install.sh`; your TigerData data stays put because it lives in the cloud. Copy `.env` over separately.

### Then (Ubuntu)

```bash
cd ~/hackathon-trader
nano .env                                   # fill in keys (see "Accounts and keys")
source .venv/bin/activate
python -m src.init_db --tiger               # create tables in TigerData
python -m src.check_apis                    # every service should say PASS (Snowflake can wait)
bash scripts/run_all.sh                     # whole pipeline, dry-run orders
# when happy:
python -m src.decide                        # submit PAPER orders
streamlit run src/dashboard.py --server.address 0.0.0.0 --server.port 8501
```

Server requirements: ~2 vCPU, 4 GB RAM (2 GB + swap works), ~3 GB disk (PyTorch dominates), outbound HTTPS to Alpaca, Finnhub, TigerData and Snowflake. No GPU needed.

## Accounts and keys

| Service | What you need | `.env` variable(s) |
|---|---|---|
| Alpaca | Paper trading account → API key/secret (use the **paper** keys) | `ALPACA_API_KEY`, `ALPACA_API_SECRET` |
| Finnhub | Free API key | `FINNHUB_API_KEY` |
| TigerData | Create a service (Timescale Cloud) → copy the connection string; TimescaleDB is preinstalled | `TIGER_DATABASE_URL` (keep `?sslmode=require`) |
| Snowflake | Trial account → account identifier (`orgname-accountname`), user, password | `SNOWFLAKE_*` |

Snowflake notes: the user needs rights to create a warehouse/database (the trial `ACCOUNTADMIN` works). If your account enforces MFA/network policies for password login, use a programmatic access token as `SNOWFLAKE_PASSWORD`, or put the server's IP on the allow list. Cortex `SENTIMENT` availability depends on account region/edition; `--cortex` automatically falls back to VADER.

## Pipeline commands

| Step | Command | Notes |
|---|---|---|
| Create tables | `python -m src.init_db --tiger` | idempotent; add `--snowflake` to create warehouse/db/schema |
| API check | `python -m src.check_apis` | Hour-0 gate |
| Daily bars | `python -m src.ingest_bars [--incremental]` | tickers + SPY, adjusted, feed from `config.yaml` |
| News | `python -m src.ingest_news [--source both\|finnhub\|alpaca] [--days N] [--only AAPL]` | `bash scripts/backfill_news_bg.sh` for background run |
| Sentiment | `python -m src.sentiment [--cortex]` | fills `news.sentiment` |
| Features + labels | `python -m src.features` | |
| Vertical slice | `python -m src.vertical_slice [--place]` | Hour-6 chain test |
| Train + evaluate | `python -m src.train` | writes `models/` and `results/` |
| Predict latest | `python -m src.predict` | writes `predictions` |
| Trade (paper) | `python -m src.decide [--dry-run]` | target-weight rebalance |
| Snowflake sync | `python -m src.sync_snowflake` | optional |
| Daily cycle | `python -m src.run_cycle [--dry-run] [--retrain] [--skip-snowflake]` | stops before trading if a critical step fails |
| Tests | `pytest -q` | offline leakage/split tests |

## How it works

**Label.** `y_excess_5d = open[T+6]/open[T+1] − 1 − same for SPY`. You decide after the close on day T, enter at the next open, hold five trading days.

**Features (15).** Returns (1/5/10/21d), volatility (10/21d), volume ratio vs 20d, RSI(14), MACD histogram / price, SPY 5d return and 21d vol, relative 5d return, news sentiment (1d, 5d) and 5d article count. Articles published after 16:00 ET count toward the next trading day.

**Split.** Last 20% of dates held out as the test set. Last 15% of the training dates is validation (ridge alpha, MLP early stopping). A 7-trading-day gap separates every split (> label horizon). Scaler and winsorization are fit on the training portion only.

**Models.** Ridge baseline. MLP 32→16 with dropout, AdamW, Gaussian negative log-likelihood outputting `[μ, σ]`, early stopping on validation. The MLP is deliberately tiny (~10–15k rows). It may not beat ridge, and that is an acceptable result.

**Agreement filter** (`strategy.py`). Trade a name only if both models have μ > 0, both rank it in their top 40%, and the MLP's μ/σ conviction clears `min_conviction`. Take at most `top_k` names, weight by conviction (clipped), cap each at `max_position`, and total at `max_gross`. Everything is long-only; the rest stays in cash.

**Backtest.** Rebalances every 5 trading days (no overlapping holds), applies `cost_bps` round-trip, and compares each strategy to SPY at 100% **and** SPY at the strategy's gross exposure, plus an equal-weight universe.

**Orders.** `decide.py` computes target dollar weights, diffs them against Alpaca positions, and submits notional market `DAY` orders (paper only). Every order is stored in `orders` with the predictions that caused it. Orders placed while the market is closed queue until the next open.

## Running it unattended on the server

```bash
./setup.sh --systemd --skip-apt
systemctl list-timers hackathon-cycle.timer       # next run (16:20 America/New_York, Mon-Fri)
journalctl -u hackathon-dashboard -f               # dashboard logs
tail -f logs/cycle.log                             # cycle logs
sudo systemctl start hackathon-cycle.service       # run a cycle now
```

Dashboard access: either open port 8501 (`sudo ufw allow 8501/tcp` and your cloud firewall) or, safer, tunnel from your laptop:

```bash
ssh -L 8501:localhost:8501 user@SERVER        # then browse http://localhost:8501
```

Long manual jobs: run inside `tmux new -s trader` so an SSH drop does not kill them.

## Kill switch and safety

- Paper trading only; `decide.py` constructs `TradingClient(..., paper=True)`.
- `TRADING_ENABLED=0` in `.env` makes `decide.py` plan but never submit.
- `run_cycle` aborts (no orders) if bars, features or predictions fail, so it never trades on stale data.
- `.env` is `chmod 600` and git-ignored; never commit keys.

## Repository layout

```
hackathon-trader/
├── PLAN.md                  hour-by-hour plan, roles, fallbacks, demo script
├── README.md
├── setup.sh                 Ubuntu setup (apt, venv, torch CPU, deps, .env, tests, optional systemd)
├── config.yaml              tickers, split, model and strategy parameters
├── .env.example
├── requirements.txt
├── sql/                     tiger_schema.sql, snowflake_schema.sql
├── src/
│   ├── common.py            config, env, DB access, upsert
│   ├── init_db.py  check_apis.py  vertical_slice.py
│   ├── ingest_bars.py  ingest_news.py  sentiment.py
│   ├── features.py  splits.py  models.py  evaluate.py  train.py
│   ├── strategy.py  predict.py  decide.py
│   ├── sync_snowflake.py  run_cycle.py  dashboard.py
├── tests/test_leakage.py
├── scripts/                 run_all.sh, backfill_news_bg.sh
└── deploy/                  systemd service + timer templates
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Missing env var ...` | Fill the value in `.env`; commands load it automatically. |
| TigerData `SSL`/`timeout` | Keep `?sslmode=require`; confirm the service is running and not IP-restricted. |
| Alpaca 403 on bars | You may be using live keys or SIP; keep `alpaca_feed: iex` and use paper keys. |
| Finnhub 429 | The scraper retries with backoff; rerun later or use `--source alpaca`. |
| Very few news rows | Free Finnhub only goes back ~1 year; Alpaca news covers the rest. Sentiment features are 0 where there is no news. |
| `train` says "only N labelled rows" | Run ingestion, sentiment and features first. |
| Snowflake login fails | Check account identifier format, MFA/network policy, role; or skip (non-blocking). |
| No trades in the dry-run | The agreement filter can legitimately return nothing. See "Fallback ladder" in PLAN.md. |
| Orders stuck "accepted" | Market closed; they fill at the next open. |
| `pip` fails building a wheel | `sudo apt install -y build-essential python3-dev`, then rerun `setup.sh`. |

## Limitations (say these out loud in the demo)

Small universe, short history, a single regime in the test window, one split instead of walk-forward, overlapping labels (naive IC t-stats are inflated), free-tier IEX volume, headline-level VADER sentiment, survivorship in a hand-picked ticker list, paper fills without realistic slippage, and some researcher degrees of freedom from tuning. A small positive IC or a null result is the expected honest outcome.
