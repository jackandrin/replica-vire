-- TigerData (TimescaleDB). Idempotent. Run with: python -m src.init_db --tiger
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS bars (
    symbol       text             NOT NULL,
    ts           timestamptz      NOT NULL,   -- event_time: bar open timestamp from Alpaca
    open         double precision,
    high         double precision,
    low          double precision,
    close        double precision,
    volume       double precision,
    vwap         double precision,
    trade_count  double precision,
    ingested_at  timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, ts)
);
SELECT create_hypertable('bars', 'ts', if_not_exists => TRUE, chunk_time_interval => INTERVAL '180 days');

CREATE TABLE IF NOT EXISTS news (
    id           bigserial PRIMARY KEY,
    symbol       text        NOT NULL,
    source_api   text        NOT NULL,       -- finnhub | alpaca
    event_time   timestamptz NOT NULL,       -- when the article was published
    ingested_at  timestamptz NOT NULL DEFAULT now(),
    headline     text,
    summary      text,
    publisher    text,
    url          text        NOT NULL,
    sentiment    real,                       -- -1..1, filled by src/sentiment.py
    UNIQUE (symbol, url)
);
CREATE INDEX IF NOT EXISTS news_symbol_time ON news (symbol, event_time);

CREATE TABLE IF NOT EXISTS features (
    symbol text NOT NULL, d date NOT NULL,
    ret_1 double precision, ret_5 double precision, ret_10 double precision, ret_21 double precision,
    vol_10 double precision, vol_21 double precision, vol_ratio double precision,
    rsi_14 double precision, macd_hist double precision,
    mkt_ret_5 double precision, mkt_vol_21 double precision, rel_ret_5 double precision,
    sent_1d double precision, sent_5d double precision, news_cnt_5d double precision,
    computed_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, d)
);

CREATE TABLE IF NOT EXISTS labels (
    symbol text NOT NULL, d date NOT NULL,
    y_raw_5d double precision, y_spy_5d double precision, y_excess_5d double precision,
    computed_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, d)
);

CREATE TABLE IF NOT EXISTS predictions (
    symbol text NOT NULL, asof date NOT NULL, model text NOT NULL,
    mu double precision, sigma double precision,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, asof, model)
);

CREATE TABLE IF NOT EXISTS orders (
    client_order_id text PRIMARY KEY,
    alpaca_order_id text, symbol text NOT NULL, side text NOT NULL,
    notional double precision, weight double precision,
    mu_ridge double precision, mu_mlp double precision, sigma_mlp double precision,
    status text, asof date, created_at timestamptz NOT NULL DEFAULT now()
);

-- Direction vector per symbol/decision date, written by src/autotrade.py, read by the web API
CREATE TABLE IF NOT EXISTS signals (
    symbol text NOT NULL, asof date NOT NULL,
    direction text, expected_pct double precision, timeframe text, height double precision,
    suggested_usd double precision, reason text,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, asof)
);

-- Single-user dashboard settings (id = 1)
CREATE TABLE IF NOT EXISTS user_settings (
    id int PRIMARY KEY, data jsonb NOT NULL, updated_at timestamptz NOT NULL DEFAULT now()
);

-- Recommendations / auto-trade notices shown as dashboard toasts.
-- status: new -> requested (user clicked Execute) -> executed | error ; or dismissed
CREATE TABLE IF NOT EXISTS alerts (
    id bigserial PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(),
    asof date NOT NULL, symbol text NOT NULL, action text NOT NULL,
    expected_pct double precision, timeframe text, act_by timestamptz,
    size_usd double precision, height double precision, reason text,
    mode text NOT NULL DEFAULT 'recommend', status text NOT NULL DEFAULT 'new', detail text,
    UNIQUE (symbol, asof, action)
);

CREATE TABLE IF NOT EXISTS run_log (
    id bigserial PRIMARY KEY, run_at timestamptz NOT NULL DEFAULT now(),
    step text NOT NULL, ok boolean NOT NULL, detail text
);

-- ============================================================================
-- v2 tables (docs/V2_CONTRACT.md). All timestamps UTC. init_db splits on semicolons,
-- so no DO blocks here: guarded statements only.
-- ============================================================================
CREATE TABLE IF NOT EXISTS symbols (
    symbol      text PRIMARY KEY,
    company     text,
    sector      text,
    sector_etf  text,
    kind        text NOT NULL DEFAULT 'stock',   -- stock | etf
    cik         text,                            -- 10-digit zero-padded SEC CIK, nullable
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS bars_15m (
    symbol       text             NOT NULL,
    ts           timestamptz      NOT NULL,      -- bar START (Alpaca timestamp)
    open         double precision,
    high         double precision,
    low          double precision,
    close        double precision,
    volume       double precision,
    vwap         double precision,
    trade_count  double precision,
    PRIMARY KEY (symbol, ts)
);
SELECT create_hypertable('bars_15m', 'ts', if_not_exists => TRUE, chunk_time_interval => INTERVAL '30 days');
-- Enable compression only if not yet enabled (re-running ALTER on compressed chunks errors on older Timescale).
CREATE OR REPLACE FUNCTION _v2_enable_bars_15m_compression() RETURNS void LANGUAGE sql AS $$
    ALTER TABLE bars_15m SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol',
                              timescaledb.compress_orderby = 'ts DESC')
$$;
SELECT _v2_enable_bars_15m_compression() FROM timescaledb_information.hypertables
 WHERE hypertable_name = 'bars_15m' AND NOT compression_enabled;
DROP FUNCTION IF EXISTS _v2_enable_bars_15m_compression();
SELECT add_compression_policy('bars_15m', INTERVAL '7 days', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS bars_1d (
    symbol  text             NOT NULL,
    d       date             NOT NULL,           -- US/Eastern trading day
    open    double precision,
    high    double precision,
    low     double precision,
    close   double precision,
    volume  double precision,
    vwap    double precision,
    PRIMARY KEY (symbol, d)
);

-- kind: earnings | insider | analyst | short_volume (see contract for event_time / available_at rules)
CREATE TABLE IF NOT EXISTS events (
    symbol        text        NOT NULL,
    kind          text        NOT NULL,
    event_time    timestamptz NOT NULL,
    available_at  timestamptz NOT NULL,
    value         double precision,
    key           text        NOT NULL,          -- dedupe id: accession, accession-row, period or date
    payload       text,                          -- JSON
    ingested_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, kind, event_time, key)
);
CREATE INDEX IF NOT EXISTS events_symbol_available ON events (symbol, available_at);

CREATE TABLE IF NOT EXISTS documents (
    doc_id        bigserial PRIMARY KEY,
    symbol        text        NOT NULL,
    source        text        NOT NULL,          -- alpaca | finnhub | edgar
    doc_type      text        NOT NULL,          -- news | 8-K | 10-Q | 10-K
    published_at  timestamptz NOT NULL,          -- EDGAR: acceptance datetime
    title         text,
    body          text,                          -- plain text, <= 8000 chars
    url           text        NOT NULL,
    content_hash  text,                          -- sha1 of normalized title + body
    ingested_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (symbol, url)
);
CREATE INDEX IF NOT EXISTS documents_symbol_published ON documents (symbol, published_at);

-- Written by src/sentiment_agent.py
CREATE TABLE IF NOT EXISTS sentiment_events (
    doc_id          bigint      NOT NULL,
    symbol          text        NOT NULL,
    published_at    timestamptz NOT NULL,
    direction       text,                        -- improving | deteriorating | stable
    delta           double precision,            -- [-1, 1]
    level           double precision,            -- [0, 100]
    novelty         double precision,            -- [0, 1]
    materiality     double precision,            -- [0, 1]
    horizon         text,                        -- hours | days | weeks
    drivers         text,                        -- JSON list of str
    summary         text,
    backend         text,
    model           text,
    prompt_version  text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id, symbol)
);
CREATE INDEX IF NOT EXISTS sentiment_events_symbol_published ON sentiment_events (symbol, published_at);

-- v2 model outputs on the existing tables (written by src/predict.py and src/autotrade.py)
ALTER TABLE signals ADD COLUMN IF NOT EXISTS ts timestamptz,
    ADD COLUMN IF NOT EXISTS horizons jsonb, ADD COLUMN IF NOT EXISTS regressors jsonb,
    ADD COLUMN IF NOT EXISTS drivers jsonb, ADD COLUMN IF NOT EXISTS exit_at timestamptz,
    ADD COLUMN IF NOT EXISTS model_version text;
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS exit_at timestamptz;

-- Trained models published from the HPC (src/publish_models.py). The dashboard reads metrics from model_runs;
-- src/predict.py loads the newest model_artifacts version. Only trusted machines should hold write access:
-- model files are Python pickles and are executed when loaded.
CREATE TABLE IF NOT EXISTS model_runs (
    version text PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(),
    trained_through date, metrics jsonb NOT NULL, meta jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS model_artifacts (
    version text NOT NULL REFERENCES model_runs (version) ON DELETE CASCADE,
    name text NOT NULL, content bytea NOT NULL, sha256 text NOT NULL,
    PRIMARY KEY (version, name)
);

-- Live paper portfolios built from the onboarding form. All portfolios share ONE Alpaca paper account:
-- portfolio_orders is the virtual ledger that attributes fills to each portfolio (positions = sum of fills).
-- The browser keeps the secret token and only its sha256 is stored here.
CREATE TABLE IF NOT EXISTS portfolios (
    id text PRIMARY KEY,
    token_hash text NOT NULL,
    budget double precision NOT NULL,
    risk text NOT NULL, horizon text, sectors jsonb NOT NULL DEFAULT '[]'::jsonb,
    targets jsonb NOT NULL DEFAULT '[]'::jsonb,
    status text NOT NULL DEFAULT 'active',
    spy_start double precision,
    last_rebalance date,
    client_hash text,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS portfolios_created ON portfolios (created_at);

CREATE TABLE IF NOT EXISTS portfolio_orders (
    client_order_id text PRIMARY KEY,
    portfolio_id text NOT NULL REFERENCES portfolios (id) ON DELETE CASCADE,
    alpaca_order_id text, symbol text NOT NULL, side text NOT NULL,
    notional double precision, qty double precision,
    filled_qty double precision NOT NULL DEFAULT 0, filled_avg_price double precision,
    status text, reason text, asof date,
    created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS portfolio_orders_pf ON portfolio_orders (portfolio_id, created_at);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    portfolio_id text NOT NULL REFERENCES portfolios (id) ON DELETE CASCADE,
    asof date NOT NULL,
    value double precision NOT NULL, cash double precision NOT NULL, spy_price double precision,
    PRIMARY KEY (portfolio_id, asof)
);
