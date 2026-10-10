# 24-Hour Hackathon Plan

Goal: a working **end-to-end loop early**, an **honest evaluation**, and a **clear demo**.
Everything here maps to a file in this repo, so every hour has a command and a "done" test.

---

## 1. Decisions to lock in Hour 0 (do not revisit)

| Decision | Value | Where it lives |
|---|---|---|
| Universe | 20 liquid US stocks + SPY (benchmark, never traded) | `config.yaml: tickers` |
| History | ~3 years of daily bars | `history_days: 1100` |
| Label | 5-day **excess** return vs SPY, entering at the **next open** | `src/features.py` |
| Decision time | After the close (16:00 ET) on day T | `src/features.py` docstring |
| Models | Ridge (baseline) + tiny MLP (32→16, outputs μ and σ) | `src/models.py` |
| Split | Last 20% of dates = test; 7-trading-day gap between every split | `config.yaml: split` |
| Trading | Alpaca **paper** only, long-only, ≤50% gross, ≤10% per name | `config.yaml: strategy` |
| Alpaca feed | `iex` (free tier). Volume is only IEX's share of market volume, so use `vol_ratio` (relative), not absolute volume | `config.yaml` |
| Sentiment | VADER by default; Snowflake Cortex `SENTIMENT` if it works (`--cortex`) | `src/sentiment.py` |

Roles (4 people). One owner per row; others review.

| Role | Owns | Primary files |
|---|---|---|
| **A. Data engineer** | TigerData schema, ingestion, Snowflake sync, ops on the Ubuntu box | `sql/`, `ingest_*.py`, `sync_snowflake.py`, `deploy/` |
| **B. Features / NLP** | Features, labels, sentiment, leakage tests | `features.py`, `sentiment.py`, `tests/` |
| **C. Modeler** | Splits, ridge, MLP, metrics, backtest | `models.py`, `splits.py`, `train.py`, `evaluate.py` |
| **D. Trading + demo** | Decision layer, orders, dashboard, slides | `strategy.py`, `decide.py`, `dashboard.py` |

Everyone shares one Ubuntu server and one TigerData database. Each person works in a git branch or a separate clone; only A touches schema changes.

---

## 2. Interface contracts (so people can work in parallel)

These tables are the hand-offs. If a column changes, tell the downstream owner.

| Table (TigerData) | Written by | Read by | Key |
|---|---|---|---|
| `bars` (hypertable) | A | B, D | `(symbol, ts)` |
| `news` | A (`sentiment` filled by B) | B | `(symbol, url)` |
| `features` | B | C, D | `(symbol, d)` |
| `labels` | B | C | `(symbol, d)` |
| `predictions` | C (`predict.py`) | D | `(symbol, asof, model)` |
| `orders` | D | dashboard, Snowflake | `client_order_id` |
| `run_log` | all | dashboard | `id` |

Time semantics (the whole leakage story):
- `bars.ts` / `news.event_time` = **event time** (when it happened). `ingested_at` = when we stored it.
- `features.d = T` uses only bars dated ≤ T and news published ≤ 16:00 ET on T. News after 16:00 ET counts for the next trading day.
- `labels.d = T` looks forward: `open[T+6]/open[T+1] − 1` minus the same for SPY.
- Rows with features but no label (the most recent ~6 days) are the **prediction set**.

---

## 3. Hour-by-hour

### Hours 0–1 — Lock decisions, keys, one call per API
**Everyone**: create accounts and keys while A runs setup.
- A: provision Ubuntu box → `bash install.sh` (see README). Create TigerData service; paste `TIGER_DATABASE_URL` into `.env`.
- B/C: Alpaca paper keys; Finnhub key. Share keys via a password manager, never chat or git.
- D: Snowflake trial account (can wait until hour 6; see fallback).

**Gate:** `python -m src.init_db --tiger && python -m src.check_apis` shows PASS for Alpaca (account, bars, news), Finnhub, TigerData. Snowflake may still FAIL.

### Hours 1–5 — Parallel ingestion
- A: `python -m src.ingest_bars` (≈1 min). Then **immediately** `bash scripts/backfill_news_bg.sh` (≈5–10 min; rate-limited, runs in background, survives SSH drops).
- A: confirm in SQL: `SELECT symbol, count(*), min(ts), max(ts) FROM bars GROUP BY 1;` → 21 symbols, ~750 rows each.
- B: while news loads, run `pytest -q` and read `features.py`; add any extra feature ideas to a list (do **not** implement yet).
- C: write the evaluation harness against `tests/` synthetic data; confirm `src.train` logic on synthetic data.
- D: run the vertical slice scaffolding (next section) and design dashboard layout.

**Gate:** raw tables filled; `news` row count > ~10k; ingestion is re-runnable (`--incremental`).

### Hours 5–6 — CHECKPOINT: thin vertical slice (the most important hour)
```
python -m src.vertical_slice --symbol AAPL            # bars -> feature -> dummy prediction
python -m src.vertical_slice --symbol AAPL --place    # + one $10 Alpaca paper order
```
**Gate:** the order appears in the Alpaca paper dashboard (or is queued if the market is closed). **If this does not work by hour 6, stop adding anything and fix the chain.**

### Hours 6–11 — Features and labels
- B: `python -m src.sentiment && python -m src.features`. Verify:
  ```sql
  SELECT count(*), min(d), max(d) FROM features;
  SELECT count(*) FROM labels;
  ```
  Expect ≈ 20 × ~700 = ~14k feature rows; ~30 fewer label rows per symbol than feature rows.
- B: run `pytest -q` (leakage tests). **Do not proceed to modelling if they fail.**
- A: `python -m src.init_db --snowflake && python -m src.sync_snowflake`. **Timebox to hour 10.** If Snowflake is not working by then, move on (see fallbacks).

**Gate:** `features` and `labels` populated, tests green, Snowflake tables exist (or fallback invoked).

### Hours 8–14 — Models and honest evaluation
- C: `python -m src.train`. Outputs `results/metrics.json`, `backtest_curve.csv`, models.
- Read the log: IC and hit rate for ridge, MLP and their average; backtest vs SPY at equal gross exposure.
- Allowed tuning: ridge alphas, MLP width/dropout/weight decay, **evaluated on validation only**. The test set is touched once per final configuration. Write down every config you tried (this is your multiple-testing disclosure).
- Expected honest outcome: IC between −0.02 and +0.05; hit rate 50–54%; MLP roughly ties ridge. That is a fine demo.

**Gate:** `results/metrics.json` exists and the numbers have been sanity-checked against the leakage checklist (section 6).

### Hours 14–18 — Decision layer, dashboard, unattended cycle
- D: `python -m src.predict && python -m src.decide --dry-run` → inspect the plan. Then `python -m src.decide` for real paper orders.
- D: dashboard: `streamlit run src/dashboard.py` (4 tabs: predictions & agreement, positions & orders, honest results, pipeline health).
- A: `python -m src.run_cycle` end to end; then `./setup.sh --systemd` to install the daily timer + dashboard service.

**Gate:** `python -m src.run_cycle` completes unattended with exit 0 and `run_log` shows all steps OK.

### Hours 18–20 — Stabilize, then CODE FREEZE at hour 20
- Run the entire pipeline from scratch **twice** (`bash scripts/run_all.sh`), once on a fresh clone to prove `install.sh` works.
- Fix bugs only. No new features. Tag the commit `demo`.

### Hours 20–24 — Slides, rehearsal, backup, sleep
- 20–21: slides (outline in section 7). 21–22: first full rehearsal with a timer.
- 22–23: **record a backup screen recording** of the full demo (dashboard + TigerData + Snowflake + Alpaca).
- Sleep rotation: two people sleep first, two after the first rehearsal. Do not leave the demo to the one person who stayed up all night.
- Last hour: pre-place the paper orders (section 5), take screenshots of fills, rehearse once more.

---

## 4. Fallback ladder (use the first one that applies)

| Problem | Fallback |
|---|---|
| Snowflake not working at hour 10 | Train in pandas (already the default); `sync_snowflake` after the models are done. You can still say it is in the stack. |
| Cortex `SENTIMENT` unavailable | VADER (default). Mention it honestly. |
| Finnhub history short or rate-limited | Alpaca news (default source includes both; go back years). `--source alpaca`. |
| Alpaca SIP not permitted | Stay on `iex`. |
| TigerData connection flaky | Retry; verify `sslmode=require`; check the server's outbound IP is not blocked; fall back to local Postgres+TimescaleDB only as a last resort. |
| MLP unstable / worse than ridge | Report it. Keep both; agreement filter still works (it just rarely agrees). |
| Agreement filter yields no trades | Raise `agree_top_frac` to 0.5 in `config.yaml`; report the change. |
| Live demo outside market hours | Show logged orders, queued orders and prior fills (section 5). |
| Everything is on fire at hour 18 | Ship ridge only + dashboard + paper order log. A working simple system beats a broken complex one. |

---

## 5. Hackathon-specific traps

1. **Market hours.** Paper `DAY` orders submitted when the market is closed queue until the next open. Place orders **before** the demo and show fills + the `orders` table. Never depend on a live fill during judging.
2. **Leakage.** One rule: features for day T use only data timestamped before T's decision time; test strictly after train. Enforced by `tests/test_leakage.py` and an `assert` in `train.py`.
3. **Rate limits.** Finnhub free ≈ 60 calls/min → backfill early in the background. Everything is cached in TigerData.
4. **Snowflake time sink.** Timebox to hour 10.
5. **Overselling.** Judges respect "IC of 0.03, small edge, here is why we are skeptical" more than a perfect backtest.
6. **Naive t-stats lie.** 5-day labels overlap day to day, so the naive IC t-stat is inflated. The smoke test on pure random-walk prices still printed IC ≈ 0.07, naive t ≈ 2.8. Do **not** present the naive t-stat as significance; present IC, hit rate, the test-period regime, and "not significant after accounting for overlap".
7. **Shared secrets.** `.env` is `chmod 600` and git-ignored. Never paste keys in Slack/Discord.
8. **Paper only.** `decide.py` hard-codes `paper=True`. `TRADING_ENABLED=0` in `.env` is the kill switch.

---

## 6. Leakage checklist (run before trusting any number)

- [ ] `pytest -q` passes (features at T unchanged when future data is removed; after-close news shifts a day; label uses next open and horizon 5; split gaps).
- [ ] `train.py` prints fit / val / test date ranges, each separated by ≥7 trading days.
- [ ] Scaler and winsorization are fit on **fit/train only**, never on test.
- [ ] No hyperparameter was chosen by looking at test metrics (validation only).
- [ ] If IC > 0.10 or hit rate > 58%: assume a bug before celebrating. Look for a feature that peeks (e.g. a close-to-close return that overlaps the label).
- [ ] Backtest includes costs (`cost_bps`) and rebalances every 5 days (non-overlapping).
- [ ] Benchmarks shown: SPY at 100% and SPY at the strategy's gross exposure, plus equal-weight universe.

---

## 7. Demo (3–5 minutes)

| Time | Show | Source |
|---|---|---|
| 0:00 | Architecture diagram in one slide | README diagram |
| 0:30 | **Raw data in TigerData** (`SELECT ... FROM bars`, `news`) | psql / Tiger console |
| 1:00 | **Features in Snowflake** (`SELECT * FROM FEATURES`) | Snowflake worksheet |
| 1:30 | Today's predictions from both models; agreement filter picks | dashboard tab 1 |
| 2:15 | Alpaca paper positions and the order log | dashboard tab 2 + Alpaca UI |
| 3:00 | **Honest results**: IC, hit rate, backtest vs SPY, limitations | dashboard tab 3 |
| 4:00 | Limitations + what we would do next | slide |

Limitations slide (be specific): one regime in the test window; 20 names; free-tier IEX volume; VADER on headlines; overlapping labels; paper fills ≠ live slippage; one split, no walk-forward; selection effects from the config tweaks we tried.

Likely judge questions and honest answers:
- *Is this profitable?* We do not claim that. IC and hit rate show a small or no edge; here is the uncertainty.
- *How do you prevent leakage?* Decision time convention, tests, split gaps, scaler fit on train only.
- *Why two models?* The agreement filter trades only when an interpretable linear model and a nonlinear model both agree; the MLP's σ gives a conviction measure for sizing.
- *What breaks in production?* Regime shift, data outages, survivorship in the ticker list, costs/slippage.

---

## 8. Time budget cheat-sheet

| Task | Expected wall-clock |
|---|---|
| `install.sh` on a fresh Ubuntu VM | 3–6 min (PyTorch CPU download is the bulk) |
| `ingest_bars` | ~1 min |
| `ingest_news` (both sources, 20 tickers) | ~5–15 min |
| `sentiment` (VADER, ~20–50k articles) | ~1–2 min |
| `features` | seconds |
| `train` (CPU) | ~1–3 min |
| `run_cycle` (daily) | ~1–2 min |
