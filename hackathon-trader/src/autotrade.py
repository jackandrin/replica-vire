"""Signals -> recommendations (dashboard toasts) -> optional auto-trading on Alpaca PAPER.
python -m src.autotrade [--no-orders]

1. Reads today's direction vectors from `signals` (written by src/predict.py).
2. Compares them with open positions: planned exits whose horizon has passed, closes on a flipped signal
   (same-day closes only on a large spike: day-trade rule), and new opens sized by conviction.
3. Writes `alerts` (BUY / SELL / SHORT / COVER) with the reason and the planned exit time.
4. If settings.auto_trading is on, executes them; otherwise it executes only alerts the user clicked
   "Execute" on in the dashboard (status = 'requested').
Paper only: TradingClient(paper=True) is hard-coded. Kill switch: TRADING_ENABLED=0 in .env."""
import argparse
import math
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd

from .common import cfg, env, get_logger, query_df, tiger, upsert
from .sectors import SYMBOLS
from .signals import plan_actions

log = get_logger("autotrade")


def load_settings() -> dict:
    st = dict(cfg()["autotrade"])
    try:
        d = query_df("SELECT data FROM user_settings WHERE id = 1")
        if not d.empty:
            st.update(d["data"].iloc[0])
    except Exception as e:  # noqa: BLE001  table missing -> config defaults
        log.warning("user_settings unavailable (%s); using config defaults", e)
    return st


def opened_today(syms) -> set:
    try:
        d = query_df("SELECT DISTINCT symbol FROM orders WHERE side IN ('buy','short') AND status NOT LIKE 'error%' "
                     "AND (created_at AT TIME ZONE 'America/New_York')::date = (now() AT TIME ZONE 'America/New_York')::date")
        return set(d["symbol"]) & set(syms)
    except Exception:  # noqa: BLE001
        return set()


def due_exits(held: dict) -> set:
    """Held symbols whose most recent executed open has a planned exit time in the past."""
    if not held:
        return set()
    d = query_df("SELECT DISTINCT ON (symbol) symbol, exit_at FROM alerts WHERE symbol = ANY(%s) "
                 "AND action IN ('BUY','SHORT') AND status = 'executed' ORDER BY symbol, id DESC", (list(held),))
    now = pd.Timestamp.now(tz="UTC")
    return {r.symbol for r in d.itertuples() if pd.notna(r.exit_at) and pd.Timestamp(r.exit_at) <= now}


def execute(tc, a: dict, price: float, asof) -> str:
    """Place one paper order and log it in `orders`. Returns the Alpaca status."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
    coid = f"rv-{asof}-{a['symbol']}-{uuid.uuid4().hex[:6]}"
    act = a["action"]
    if act in ("SELL", "COVER"):
        o = tc.close_position(a["symbol"])
    elif act == "BUY":
        o = tc.submit_order(MarketOrderRequest(symbol=a["symbol"], notional=round(a["size_usd"], 2),
                                               side=OrderSide.BUY, time_in_force=TimeInForce.DAY, client_order_id=coid))
    else:  # SHORT: Alpaca does not allow fractional shorts -> whole shares
        qty = math.floor(a["size_usd"] / price) if price else 0
        if qty < 1:
            return "skipped: size below one share"
        o = tc.submit_order(MarketOrderRequest(symbol=a["symbol"], qty=qty, side=OrderSide.SELL,
                                               time_in_force=TimeInForce.DAY, client_order_id=coid))
    status = str(getattr(o.status, "value", o.status))
    with tiger() as conn:
        upsert(conn, "orders", pd.DataFrame([{
            "client_order_id": coid, "alpaca_order_id": str(o.id), "symbol": a["symbol"], "side": act.lower(),
            "notional": a.get("size_usd"), "status": status, "asof": asof}]), ["client_order_id"])
    return status


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-orders", action="store_true", help="write alerts only, never trade")
    a = ap.parse_args(argv)
    st = load_settings()
    sig = query_df("SELECT symbol, asof, ts, direction, expected_pct, timeframe, height, suggested_usd, exit_at, reason "
                   "FROM signals WHERE asof = (SELECT max(asof) FROM signals)")
    if sig.empty:
        raise RuntimeError("no signals; run src.predict first")
    asof = sig["asof"].iloc[0]
    reasons = dict(zip(sig.symbol, sig.reason))
    sig_ts = pd.Timestamp(sig["ts"].max()) if sig["ts"].notna().any() else pd.Timestamp.now(tz="UTC")

    from alpaca.trading.client import TradingClient
    tc = TradingClient(env("ALPACA_API_KEY"), env("ALPACA_API_SECRET"), paper=True)
    held = {x.symbol: float(x.qty) for x in tc.get_all_positions()}
    sectors = {s: v[1] for s, v in SYMBOLS.items()}
    actions = plan_actions(sig, held, st, opened_today(held), sectors, due_exits(held))
    now = datetime.now(timezone.utc)
    if now - sig_ts.to_pydatetime() > timedelta(hours=3):  # holiday / data outage: never open on stale signals
        log.warning("signals are from %s (stale): only planned exits are allowed", sig_ts)
        actions = [x for x in actions if x["why"] == "horizon"]
    mode = "auto" if st.get("auto_trading") else "recommend"
    rows = []
    for x in actions:
        if x["why"] == "horizon":
            reason = f"Planned exit: the position's horizon has been reached. {reasons.get(x['symbol'], '')}".strip()
        elif x["why"] == "flip":
            reason = "Signal flipped against the position. " + reasons.get(x["symbol"], "")
        else:
            reason = reasons.get(x["symbol"], "")
        act_by = now if x["why"] != "signal" else (sig_ts + timedelta(minutes=30)).to_pydatetime()
        rows.append({**x, "asof": asof, "act_by": act_by, "reason": reason, "mode": mode,
                     "exit_at": None if pd.isna(x.get("exit_at")) else x.get("exit_at")})
    with tiger() as conn, conn.cursor() as cur:  # new alerts only; re-runs do not duplicate or reset status
        for r in rows:
            cur.execute("""INSERT INTO alerts (asof, symbol, action, expected_pct, timeframe, act_by, size_usd,
                           height, reason, mode, exit_at) VALUES (%(asof)s, %(symbol)s, %(action)s, %(expected_pct)s,
                           %(timeframe)s, %(act_by)s, %(size_usd)s, %(height)s, %(reason)s, %(mode)s, %(exit_at)s)
                           ON CONFLICT (symbol, asof, action) DO NOTHING""", r)
    log.info("%d alerts (%s mode)", len(rows), mode)

    if env("TRADING_ENABLED", False) == "0" or a.no_orders:
        log.info("orders disabled (TRADING_ENABLED=0 or --no-orders)")
        return
    todo = query_df("SELECT id, symbol, action, size_usd FROM alerts WHERE status = 'requested' "
                    "OR (status = 'new' AND mode = 'auto' AND asof = %s)", (asof,))
    prices = query_df("SELECT DISTINCT ON (symbol) symbol, close FROM bars_15m "
                      "WHERE ts > now() - interval '5 days' ORDER BY symbol, ts DESC")
    price = dict(zip(prices.symbol, prices.close))
    for r in todo.itertuples():
        try:
            status = execute(tc, r._asdict(), price.get(r.symbol), asof)
            new, detail = ("error", status) if status.startswith("skipped") else ("executed", status)
        except Exception as e:  # noqa: BLE001
            new, detail = "error", str(e)[:200]
        with tiger() as conn, conn.cursor() as cur:
            cur.execute("UPDATE alerts SET status = %s, detail = %s WHERE id = %s", (new, detail, int(r.id)))
        log.info("%s %s -> %s (%s)", r.action, r.symbol, new, detail)


if __name__ == "__main__":
    main()
