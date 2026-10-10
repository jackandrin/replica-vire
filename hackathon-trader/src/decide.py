"""Turn the latest predictions into Alpaca PAPER orders (target-weight rebalance) and log them.
python -m src.decide [--dry-run]
Paper only: TradingClient(paper=True) is hard-coded. Kill switch: TRADING_ENABLED=0 in .env."""
import argparse
import uuid

import pandas as pd

from .common import cfg, env, get_logger, tiger, universe, upsert
from .predict import latest_pivot
from .strategy import select_agreement

log = get_logger("decide")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="compute and log the plan, submit nothing")
    a = ap.parse_args(argv)
    s = cfg()["strategy"]

    p = latest_pivot()
    if p.empty:
        raise RuntimeError("no predictions; run src.predict first")
    asof = p["asof"].iloc[0]
    picks = select_agreement(p, s)
    log.info("asof %s | %d names pass the agreement filter: %s", asof, len(picks), list(picks.symbol))

    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
    tc = TradingClient(env("ALPACA_API_KEY"), env("ALPACA_API_SECRET"), paper=True)
    equity = float(tc.get_account().equity)
    held = {x.symbol: float(x.market_value) for x in tc.get_all_positions() if x.symbol in universe()}
    target = {r.symbol: r.weight * equity for r in picks.itertuples()}
    meta = picks.set_index("symbol")

    live = env("TRADING_ENABLED", False) != "0" and not a.dry_run
    if not tc.get_clock().is_open:
        log.info("market closed: DAY orders will queue until the next open")
    rows = []
    for sym in sorted(set(held) | set(target)):
        diff = target.get(sym, 0.0) - held.get(sym, 0.0)
        if abs(diff) < s["min_order_usd"]:
            continue
        side = "buy" if diff > 0 else "sell"
        coid = f"hk-{asof}-{sym}-{uuid.uuid4().hex[:6]}"
        status, oid = "planned", None
        if live:
            try:
                o = tc.submit_order(MarketOrderRequest(
                    symbol=sym, notional=round(abs(diff), 2), side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY, client_order_id=coid))
                status, oid = str(o.status.value if hasattr(o.status, "value") else o.status), str(o.id)
            except Exception as e:  # noqa: BLE001
                status = f"error: {str(e)[:120]}"
        m = meta.loc[sym] if sym in meta.index else None
        rows.append({"client_order_id": coid, "alpaca_order_id": oid, "symbol": sym, "side": side,
                     "notional": round(abs(diff), 2), "weight": float(m.weight) if m is not None else 0.0,
                     "mu_ridge": float(m.mu_ridge) if m is not None else None,
                     "mu_mlp": float(m.mu_mlp) if m is not None else None,
                     "sigma_mlp": float(m.sigma_mlp) if m is not None else None,
                     "status": status, "asof": asof})
        log.info("%s %s $%.2f -> %s", side.upper(), sym, abs(diff), status)
    if rows:
        with tiger() as conn:
            upsert(conn, "orders", pd.DataFrame(rows), ["client_order_id"])
    log.info("done: %d orders (%s)", len(rows), "submitted" if live else "NOT submitted")


if __name__ == "__main__":
    main()
