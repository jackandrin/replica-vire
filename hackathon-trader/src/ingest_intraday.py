"""v2 bars: Alpaca 15-min + daily bars (tickers + sectors.CONTEXT_SYMBOLS, adjustment=all) -> TigerData
bars_15m / bars_1d, plus the symbols table. Feed = config data.feed (sip), auto-falls back to iex.
python -m src.ingest_intraday [--incremental] [--only AAPL,SPY] [--timeframe both|15m|1d] [--symbols-only]"""
import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd

from .common import cfg, env, get_logger, query_df, tiger, universe, upsert
from .sectors import CONTEXT_SYMBOLS, MARKET, SECTOR_ETF, SYMBOLS, VOL_PROXY, sector_etf_of, sector_of

log = get_logger("ingest_intraday")
COLS_15M = ["symbol", "ts", "open", "high", "low", "close", "volume", "vwap", "trade_count"]
COLS_1D = ["symbol", "d", "open", "high", "low", "close", "volume", "vwap"]
ETF_NAMES = {MARKET: ("SPDR S&P 500 ETF", "Market"), VOL_PROXY: ("ProShares VIX Short-Term Futures ETF", "Volatility"),
             **{etf: (f"{sec} Select Sector SPDR", sec) for sec, etf in SECTOR_ETF.items()}}
_feed = {"name": None}  # sticky for the run: once SIP is refused we stay on IEX


def all_symbols() -> list:
    return list(dict.fromkeys(universe() + CONTEXT_SYMBOLS))


def symbols_frame(syms: list) -> pd.DataFrame:
    rows = []
    for s in syms:
        if s in ETF_NAMES:
            rows.append({"symbol": s, "company": ETF_NAMES[s][0], "sector": ETF_NAMES[s][1],
                         "sector_etf": s, "kind": "etf"})
        else:
            rows.append({"symbol": s, "company": SYMBOLS.get(s, (s,))[0], "sector": sector_of(s),
                         "sector_etf": sector_etf_of(s), "kind": "stock"})
    df = pd.DataFrame(rows)
    try:
        from . import edgar
        m = edgar.ticker_cik_map()
        ciks = [m.get(s.replace(".", "-")) if k == "stock" else None for s, k in zip(df.symbol, df.kind)]
        df["cik"] = [f"{c:010d}" if c else None for c in ciks]
    except Exception as e:  # noqa: BLE001  CIK is optional; never block bars on SEC
        log.warning("CIK lookup skipped (%s)", str(e)[:160])
    if "cik" in df and df["cik"].isna().all():
        df = df.drop(columns="cik")  # keep any CIKs already stored
    return df


def upsert_symbols(syms: list) -> int:
    df = symbols_frame(syms)
    with tiger() as conn:
        n = upsert(conn, "symbols", df, ["symbol"])
    log.info("symbols: upserted %d rows (cik %s)", n, "included" if "cik" in df else "skipped")
    return n


def _client():
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(env("ALPACA_API_KEY"), env("ALPACA_API_SECRET"))


def _sip_refused(e: Exception) -> bool:
    s = str(e).lower()
    return any(k in s for k in ("subscription", "sip", "forbidden", "403", "not permitted", "not entitled"))


def fetch_bars(client, symbol: str, tf: str, start: datetime, end) -> pd.DataFrame:
    """One Alpaca request (SDK paginates). tf in {'15m', '1d'}. Falls back to IEX if SIP is refused."""
    from alpaca.data.enums import Adjustment
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    timeframe = TimeFrame(15, TimeFrameUnit.Minute) if tf == "15m" else TimeFrame.Day
    feed = _feed["name"] or cfg()["data"].get("feed", "iex")
    while True:
        e_ = end
        if feed == "sip":  # free plans may only query SIP data older than 15 minutes
            e_ = min(end or datetime.now(timezone.utc), datetime.now(timezone.utc) - timedelta(minutes=16))
        if e_ is not None and e_ <= start:
            return pd.DataFrame()
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=start, end=e_,
                               adjustment=Adjustment.ALL, feed=feed)
        try:
            df = client.get_stock_bars(req).df
            _feed["name"] = feed
            break
        except Exception as e:  # noqa: BLE001
            if feed != "iex" and _sip_refused(e):
                log.warning("feed %s refused (%s); falling back to iex for this run", feed, str(e)[:160])
                feed = _feed["name"] = "iex"
                continue
            raise
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index().rename(columns={"timestamp": "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for c in ("vwap", "trade_count"):
        if c not in df:
            df[c] = float("nan")
    if tf == "15m":
        return df[COLS_15M]
    df["d"] = df["ts"].dt.tz_convert("America/New_York").dt.date
    return df[COLS_1D]


def _last(table: str, col: str) -> dict:
    try:
        df = query_df(f"SELECT symbol, max({col}) AS m FROM {table} GROUP BY symbol")
        return {r.symbol: pd.Timestamp(r.m) for r in df.itertuples() if r.m is not None and not pd.isna(r.m)}
    except Exception as e:  # noqa: BLE001
        log.warning("could not read last %s from %s (%s); doing full pull", col, table, e)
        return {}


def _windows(start: datetime, end: datetime, days: int):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


def ingest(syms: list, tf: str, start: datetime, incremental: bool) -> int:
    table, col, keys = ("bars_15m", "ts", ["symbol", "ts"]) if tf == "15m" else ("bars_1d", "d", ["symbol", "d"])
    last = _last(table, col) if incremental else {}
    client, total, now = _client(), 0, datetime.now(timezone.utc)
    for s in syms:
        s0 = start
        if s in last:
            t = last[s]
            t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
            s0 = max(start, (t - pd.Timedelta(days=3)).to_pydatetime())
        n = 0
        try:  # per symbol per year (15m) keeps each frame ~20k rows
            for a, b in _windows(s0, now, 365 if tf == "15m" else 3660):
                df = fetch_bars(client, s, tf, a, b)
                if not df.empty:
                    with tiger() as conn:
                        n += upsert(conn, table, df, keys)
        except Exception as e:  # noqa: BLE001  one bad symbol must not stop the run
            log.error("%s %s failed: %s", s, tf, str(e)[:300])
        total += n
        log.info("%s %s: %d bars since %s (feed %s)", s, tf, n, s0.date(), _feed["name"])
    return total


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--incremental", action="store_true", help="per symbol: from last stored bar minus 3 days")
    ap.add_argument("--only", help="comma-separated symbols (testing)")
    ap.add_argument("--timeframe", default="both", choices=["both", "15m", "1d"])
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (default config data.start)")
    ap.add_argument("--symbols-only", action="store_true", help="only refresh the symbols table")
    a = ap.parse_args(argv)
    syms = [s.strip().upper() for s in a.only.split(",")] if a.only else all_symbols()
    try:
        upsert_symbols(all_symbols())
    except Exception as e:  # noqa: BLE001
        log.error("symbols upsert failed: %s", e)
    if a.symbols_only:
        return
    start = pd.Timestamp(a.start or cfg()["data"]["start"], tz="UTC").to_pydatetime()
    for tf in (["15m", "1d"] if a.timeframe == "both" else [a.timeframe]):
        n = ingest(syms, tf, start, a.incremental)
        log.info("%s done: %d bars upserted for %d symbols", tf, n, len(syms))


if __name__ == "__main__":
    main()
