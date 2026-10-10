"""Hour-0 check: one call per external service.  python -m src.check_apis"""
from datetime import datetime, timedelta, timezone

import requests

from .common import env, tiger

results = {}


def check(name):
    def deco(fn):
        try:
            results[name] = ("PASS", fn())
        except Exception as e:  # noqa: BLE001
            results[name] = ("FAIL", str(e)[:200])
        return fn
    return deco


@check("alpaca-paper-account")
def _alpaca_account():
    from alpaca.trading.client import TradingClient
    a = TradingClient(env("ALPACA_API_KEY"), env("ALPACA_API_SECRET"), paper=True).get_account()
    return f"status={a.status} equity={a.equity} buying_power={a.buying_power}"


@check("alpaca-bars")
def _alpaca_bars():
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    c = StockHistoricalDataClient(env("ALPACA_API_KEY"), env("ALPACA_API_SECRET"))
    df = c.get_stock_bars(StockBarsRequest(symbol_or_symbols="SPY", timeframe=TimeFrame.Day, feed="iex",
                                     start=datetime.now(timezone.utc) - timedelta(days=10))).df
    return f"{len(df)} SPY bars"


@check("alpaca-news")
def _alpaca_news():
    r = requests.get("https://data.alpaca.markets/v1beta1/news", params={"symbols": "AAPL", "limit": 3},
                     headers={"APCA-API-KEY-ID": env("ALPACA_API_KEY"),
                              "APCA-API-SECRET-KEY": env("ALPACA_API_SECRET")}, timeout=20)
    r.raise_for_status()
    return f"{len(r.json().get('news', []))} articles"


@check("finnhub")
def _finnhub():
    r = requests.get("https://finnhub.io/api/v1/quote",
                     params={"symbol": "AAPL", "token": env("FINNHUB_API_KEY")}, timeout=20)
    r.raise_for_status()
    return f"AAPL last={r.json().get('c')}"


@check("tigerdata")
def _tiger():
    with tiger() as conn, conn.cursor() as cur:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname='timescaledb'")
        row = cur.fetchone()
    return f"timescaledb {row[0] if row else 'NOT INSTALLED (run init_db first)'}"


@check("snowflake (optional)")
def _snowflake():
    import snowflake.connector
    c = snowflake.connector.connect(account=env("SNOWFLAKE_ACCOUNT"), user=env("SNOWFLAKE_USER"),
                                    password=env("SNOWFLAKE_PASSWORD"))
    v = c.cursor().execute("SELECT CURRENT_VERSION()").fetchone()[0]
    c.close()
    return f"version {v}"


if __name__ == "__main__":
    for k, (s, d) in results.items():
        print(f"{s:5} {k:24} {d}")
