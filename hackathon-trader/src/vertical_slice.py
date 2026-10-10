"""Hour-6 checkpoint: bars -> one feature -> dummy prediction -> (optionally) one tiny paper order.
Proves the whole chain connects before any real modelling.
python -m src.vertical_slice [--symbol AAPL] [--place]"""
import argparse

from .common import env, get_logger, query_df

log = get_logger("slice")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--place", action="store_true", help="actually submit a $10 paper order")
    a = ap.parse_args(argv)

    bars = query_df("SELECT ts, close FROM bars WHERE symbol=%s ORDER BY ts DESC LIMIT 10", (a.symbol,))
    if len(bars) < 6:
        raise RuntimeError("not enough bars in TigerData; run python -m src.ingest_bars")
    close = bars.sort_values("ts")["close"].reset_index(drop=True)
    ret_5 = close.iloc[-1] / close.iloc[-6] - 1          # one feature
    pred = 1 if ret_5 > 0 else -1                        # dummy "model": momentum sign
    log.info("%s ret_5=%.4f -> dummy prediction %+d", a.symbol, ret_5, pred)

    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
    tc = TradingClient(env("ALPACA_API_KEY"), env("ALPACA_API_SECRET"), paper=True)
    log.info("account equity=%s market_open=%s", tc.get_account().equity, tc.get_clock().is_open)
    if pred > 0 and a.place:
        o = tc.submit_order(MarketOrderRequest(symbol=a.symbol, notional=10, side=OrderSide.BUY,
                                               time_in_force=TimeInForce.DAY))
        log.info("submitted paper order id=%s status=%s", o.id, o.status)
    else:
        log.info("no order placed (prediction %+d, --place=%s)", pred, a.place)


if __name__ == "__main__":
    main()
