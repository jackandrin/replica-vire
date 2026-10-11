# v2 data contract

Shared by ingestion, the sentiment agent, training (`src/train.py`) and live inference (`src/predict.py`).
All timestamps are **timezone-aware UTC** unless a column is a plain `date` (US/Eastern trading day).
Every row that a model can see carries the moment it became **knowable** (`available_at` / `published_at`);
features at decision time `t` may only use rows with that value `<= t`.

## Pipeline

```
 (internet: laptop / server / HPC login node)                    (HPC compute node, no internet needed)
 ingest_intraday ─┐
 ingest_docs ─────┼─► Tiger Data ─► export_dataset ─► data/bundle/*.parquet ─► sentiment_agent --backend hf (GPU)
 ingest_events ───┘        ▲                                         │            writes sentiment_events.parquet
                           │                                         └──────────► train.py  ─► models/v2/
                           └── sentiment_agent --import-bundle  ◄── copy back ──┘
 live (server, every 15 min): ingest_* --incremental → sentiment_agent --backend cortex → predict → autotrade
```

## Universe

`src/sectors.py`: `SYMBOLS` (tradable stocks → company, GICS sector), `SECTOR_ETF`, `MARKET="SPY"`,
`VOL_PROXY="VIXY"`, `CONTEXT_SYMBOLS` (market + vol proxy + 11 sector ETFs; bars only, never traded).
Tradable list = `config.yaml: tickers`.

## Bundle files (`data/bundle/`) and matching Tiger Data tables

| File / table | Columns | Notes |
|---|---|---|
| `symbols` | symbol, company, sector, sector_etf, kind (`stock`/`etf`), cik (nullable) | |
| `bars_15m` (hypertable on ts, PK symbol+ts, compressed after 7 days) | symbol, ts (bar **start**), open, high, low, close, volume, vwap, trade_count | Alpaca, `adjustment=all`; all hours Alpaca returns. Features use the regular session (09:30–16:00 ET). |
| `bars_1d` (PK symbol+d) | symbol, d, open, high, low, close, volume, vwap | Alpaca daily, `adjustment=all` |
| `events` (PK symbol+kind+event_time+key) | symbol, kind, event_time, available_at, value (float), key (text, dedupe id), payload (JSON text) | kinds below |
| `documents` (PK doc_id, UNIQUE symbol+url) | doc_id (int64), symbol, source (`alpaca`/`finnhub`/`edgar`), doc_type (`news`/`8-K`/`10-Q`/`10-K`), published_at, title, body (plain text ≤ 8000 chars), url, content_hash | EDGAR `published_at` = acceptance datetime |
| `sentiment_events` (PK doc_id+symbol) | doc_id, symbol, published_at, direction (`improving`/`deteriorating`/`stable`), delta [-1,1], level [0,100], novelty [0,1], materiality [0,1], horizon (`hours`/`days`/`weeks`), drivers (JSON text list of str), summary (text), backend, model, prompt_version | written by `src/sentiment_agent.py` |
| `manifest.json` | created_at, start, end, symbols, row counts per file | |

### `events.kind`

| kind | event_time | available_at | value | payload |
|---|---|---|---|---|
| `earnings` | 8-K (Item 2.02) acceptance | same | NaN | {accession} |
| `insider` | transaction date 16:00 ET | Form 4 acceptance | signed USD (+buy code P, −sale code S) | {code, shares, price, owner, officer, director} |
| `analyst` | Finnhub recommendation `period` | period + 31 days (conservative) | (2·strongBuy + buy − sell − 2·strongSell) / total | counts |
| `short_volume` | trade date 16:00 ET | next calendar day 08:00 ET | ShortVolume / TotalVolume (FINRA CNMS file) | {short_volume, total_volume} |

## Model outputs (written by `src/predict.py`)

`signals` table (one row per symbol per trading day, overwritten by later intraday runs), existing columns plus:
`ts` (decision time), `horizons` (JSON: `{"6h": {"mu": pct, "sigma": pct}, ...}` for 6h,12h,18h,1d,3d,5d),
`regressors` (JSON: `{family: yhat_pct}` for the 15 families), `drivers` (JSON list of
`{"name", "label", "value", "attribution"}`, top inputs by integrated-gradients attribution).
`expected_pct` / `timeframe` / `height` = the chosen horizon (see `src/signals.py`).

`results/metrics.json` (written by `src/train.py`):
```json
{"horizons": {"6h": {"nn": {"ic_mean":0, "hit_rate":0, "coverage_1sigma":0, "n_rows":0, "n_days":0},
                     "baseline": {...}}, "...": {}},
 "regressors": {"reversal": {"ic_mean":0, "hit_rate":0}, "...": {}},
 "redundant_pairs": [["oscillators","reversal",0.83]],
 "backtest": {"nn": {"total_return":0,"sharpe":0,"max_drawdown":0,"n_periods":0}, "baseline": {}, "spy": {}},
 "curve": [{"date":"2025-01-02","nn":0.0,"baseline":0.0,"spy":0.0}],
 "meta": {"train_dates":["..",".."], "test_dates":["..",".."], "n_train_rows":0, "n_test_rows":0,
          "trained_at":"..", "caveat":"..", "folds":0}}
```
