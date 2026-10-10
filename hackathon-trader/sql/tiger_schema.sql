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

CREATE TABLE IF NOT EXISTS run_log (
    id bigserial PRIMARY KEY, run_at timestamptz NOT NULL DEFAULT now(),
    step text NOT NULL, ok boolean NOT NULL, detail text
);
