"""Daily bars (tickers + benchmark) from Alpaca -> TigerData.bars.
python -m src.ingest_bars [--incremental]"""
import argparse
from datetime import datetime, timedelta, timezone

from alpaca.data.enums import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .common import cfg, env, get_logger, query_df, tiger, universe, upsert

log = get_logger("ingest_bars")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--incremental", action="store_true", help="only re-pull the last ~10 days")
    a = ap.parse_args(argv)
    c = cfg()
    symbols = universe() + [c["benchmark"]]
    start = datetime.now(timezone.utc) - timedelta(days=c["history_days"])
    if a.incremental:
        last = query_df("SELECT max(ts) AS m FROM bars")["m"].iloc[0]
        if last is not None and str(last) != "NaT":
            start = last.to_pydatetime() - timedelta(days=10)
    client = StockHistoricalDataClient(env("ALPACA_API_KEY"), env("ALPACA_API_SECRET"))
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start,
                           adjustment=Adjustment.ALL, feed=c["alpaca_feed"])
    df = client.get_stock_bars(req).df.reset_index().rename(columns={"timestamp": "ts"})
    df = df[["symbol", "ts", "open", "high", "low", "close", "volume", "vwap", "trade_count"]]
    with tiger() as conn:
        n = upsert(conn, "bars", df, ["symbol", "ts"])
    log.info("upserted %d bars for %d symbols (since %s)", n, df.symbol.nunique(), start.date())


if __name__ == "__main__":
    main()
