"""Live paper portfolios: onboarding profile -> model-built portfolio -> real Alpaca PAPER orders.

All portfolios share ONE Alpaca paper account. `portfolio_orders` is a virtual ledger: a portfolio's positions are the
sum of the fills of its own orders (client_order_id = pf-<portfolio id>-...), so nothing is attributed by guesswork.
Nothing here backtests: values come from real fills and the latest stored prices.

Used by the web API (create / read) and by the daily cycle (`python -m src.portfolio`: reconcile fills, rebalance on
the newest signals, snapshot value vs SPY). Pure Python + psycopg2 so it runs on the slim Vercel image
(alpaca-py is imported lazily). Paper only: TradingClient(paper=True) is hard-coded.
Kill switch: TRADING_ENABLED=0 stops every order submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
from collections import defaultdict
from datetime import date, datetime
from zoneinfo import ZoneInfo

from .common import env, get_logger, tiger
from .sectors import GICS_SECTORS, SYMBOLS

log = get_logger("portfolio")
ET = ZoneInfo("America/New_York")

RISKS = ("Conservative", "Balanced", "Growth")
HORIZONS = ("1 year", "3 years", "5+ years", "10+ years")
PICK_COUNT = {"Conservative": 6, "Balanced": 5, "Growth": 4}
# How strongly forecast uncertainty (the network's 1-sigma, in percent) is penalised.
DISAGREEMENT_PENALTY = {"Conservative": 1.0, "Balanced": 0.5, "Growth": 0.0}
MIN_BUDGET = 10.0
MIN_ORDER_USD = 1.0           # Alpaca's minimum notional
REBALANCE_BAND = 0.02         # ignore drifts smaller than 2% of the portfolio value (and than $1)
MAX_SIGNAL_AGE_DAYS = 5
TERMINAL = {"filled", "canceled", "expired", "rejected", "done_for_day", "replaced", "stopped", "suspended"}


class PortfolioError(Exception):
    """An expected failure with an HTTP-style status the API can return as-is."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def trading_enabled() -> bool:
    return os.getenv("TRADING_ENABLED", "1").strip() != "0"


def is_open(status: str | None) -> bool:
    s = (status or "").lower()
    return not (s in TERMINAL or s.startswith("error"))


def num(v):
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _json(v):
    if v is None or isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return None


def sigma_of(stock: dict) -> float | None:
    """1-sigma (percent) of the fusion network's forecast at the stock's chosen timeframe."""
    horizons, tf = _json(stock.get("horizons")), stock.get("timeframe")
    cell = horizons.get(tf) if isinstance(horizons, dict) and tf else None
    return num(cell.get("sigma")) if isinstance(cell, dict) else None


# ----------------------------------------------------------------------------------------------- profile + selection
def normalize_profile(budget, risk, horizon, sectors) -> dict:
    budget = num(budget)
    top = _float_env("PORTFOLIO_MAX_BUDGET", 10000)
    if budget is None or budget < MIN_BUDGET or budget > top:
        raise PortfolioError(422, f"Budget must be between ${MIN_BUDGET:.0f} and ${top:,.0f}")
    if risk not in RISKS:
        raise PortfolioError(422, f"Risk must be one of {', '.join(RISKS)}")
    if horizon not in HORIZONS:
        raise PortfolioError(422, f"Horizon must be one of {', '.join(HORIZONS)}")
    wanted = [s for s in (sectors or []) if s != "No preference"]
    bad = [s for s in wanted if s not in GICS_SECTORS]
    if bad:
        raise PortfolioError(422, f"Unknown sectors: {bad}")
    return {"budget": round(budget, 2), "risk": risk, "horizon": horizon, "sectors": sorted(set(wanted))}


def select_portfolio(stocks: list[dict], risk: str, sectors: list[str]) -> dict:
    """Long-only target weights from the latest signals.

    Eligible = a LONG signal with a price. Score = conviction - penalty * sigma (riskier profiles tolerate more
    uncertainty). Conservative: equal weight, Balanced: rank-linear, Growth: rank-squared (concentrated).
    Fewer than three eligible stocks in the chosen sectors -> all sectors are considered."""
    usable = [s for s in stocks if s.get("direction") == "long" and (num(s.get("price")) or 0) > 0
              and (num(s.get("expected_pct")) or 0) > 0]
    wanted = [s for s in sectors if s != "No preference"]
    in_sector = [s for s in usable if SYMBOLS.get(s["symbol"], ("", ""))[1] in wanted] if wanted else usable
    fallback = bool(wanted) and len(in_sector) < 3
    pool = usable if fallback else in_sector
    penalty = DISAGREEMENT_PENALTY.get(risk, 0.5)

    def score(s):
        return (num(s.get("height")) or 0) - penalty * (sigma_of(s) or 0)

    ranked = sorted(pool, key=score, reverse=True)
    picks = ranked[:PICK_COUNT.get(risk, 5)]
    raw = [1 if risk == "Conservative" else (len(picks) - i) ** (2 if risk == "Growth" else 1) for i in range(len(picks))]
    total = sum(raw) or 1
    targets = []
    for i, s in enumerate(picks):
        sym, tf, sg = s["symbol"], s.get("timeframe"), sigma_of(s)
        company, sector = SYMBOLS.get(sym, (sym, "Other"))
        targets.append({
            "symbol": sym, "weight": raw[i] / total,
            "score": round(100 * (1 - i / (len(ranked) - 1)) if len(ranked) > 1 else 100),
            "reason": s.get("reason") or (f"The fusion network expects {num(s.get('expected_pct')):+.2f}% over {tf or '—'}"
                                          f"{f' (±{sg:.2f}%)' if sg is not None else ''}; {company} is in {sector}."),
        })
    return {"targets": targets, "sector_fallback": fallback, "eligible": len(usable)}


# ------------------------------------------------------------------------------------------------------- data access
def load_candidates(cur) -> tuple[date | None, list[dict]]:
    """(asof, latest signal rows + last price) or (None, [])."""
    cur.execute("SELECT max(asof) FROM signals")
    asof = cur.fetchone()[0]
    if asof is None:
        return None, []
    cur.execute("SELECT * FROM signals WHERE asof = %s", (asof,))
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    prices = latest_prices(cur, [r["symbol"] for r in rows])
    for r in rows:
        r["price"] = prices.get(r["symbol"])
    return asof, rows


def latest_prices(cur, symbols) -> dict[str, float]:
    """Last stored close: 15-minute bars first, then daily bars."""
    symbols = sorted(set(symbols))
    out: dict[str, float] = {}
    if not symbols:
        return out
    cur.execute("""SELECT DISTINCT ON (symbol) symbol, close FROM bars_15m
                   WHERE symbol = ANY(%s) AND close IS NOT NULL AND ts > now() - interval '10 days'
                   ORDER BY symbol, ts DESC""", (symbols,))
    out.update({s: float(c) for s, c in cur.fetchall()})
    missing = [s for s in symbols if s not in out]
    if missing:
        cur.execute("""SELECT DISTINCT ON (symbol) symbol, close FROM bars_1d
                       WHERE symbol = ANY(%s) AND close IS NOT NULL ORDER BY symbol, d DESC""", (missing,))
        out.update({s: float(c) for s, c in cur.fetchall()})
    return out


def _orders(cur, pid: str) -> list[dict]:
    cur.execute("""SELECT client_order_id, symbol, side, notional, qty, filled_qty, filled_avg_price, status, reason,
                          asof, created_at FROM portfolio_orders WHERE portfolio_id = %s ORDER BY created_at, client_order_id""",
                (pid,))
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _portfolio(cur, pid: str) -> dict | None:
    cur.execute("""SELECT id, budget, risk, horizon, sectors, targets, status, spy_start, last_rebalance, created_at
                   FROM portfolios WHERE id = %s""", (pid,))
    r = cur.fetchone()
    if not r:
        return None
    pf = dict(zip(["id", "budget", "risk", "horizon", "sectors", "targets", "status", "spy_start", "last_rebalance",
                   "created_at"], r))
    pf["sectors"], pf["targets"] = _json(pf["sectors"]) or [], _json(pf["targets"]) or []
    return pf


def ledger(orders: list[dict], budget: float) -> dict:
    """Virtual account from fills: signed quantity and cost basis per symbol, settled cash, cash reserved by open buys."""
    qty, cost = defaultdict(float), defaultdict(float)
    cash, reserved, open_symbols = float(budget), 0.0, set()
    for o in orders:
        sym, fq = o["symbol"], num(o.get("filled_qty")) or 0.0
        amount = fq * (num(o.get("filled_avg_price")) or 0.0)
        if o["side"] == "buy":
            qty[sym] += fq
            cost[sym] += amount
            cash -= amount
        else:
            if qty[sym] > 0:
                cost[sym] *= max(qty[sym] - fq, 0.0) / qty[sym]
            qty[sym] -= fq
            cash += amount
        if is_open(o.get("status")):
            open_symbols.add(sym)
            if o["side"] == "buy":
                reserved += max((num(o.get("notional")) or 0.0) - amount, 0.0)
    qty = {s: q for s, q in qty.items() if abs(q) > 1e-9}
    return {"qty": qty, "cost": {s: cost[s] for s in qty}, "cash": cash, "reserved": reserved,
            "open_symbols": open_symbols}


def total_value(led: dict, prices: dict) -> float:
    return led["cash"] + sum(q * (prices.get(s) or led["cost"][s] / q) for s, q in led["qty"].items())


def reserved_positions(cur) -> dict[str, float]:
    """Shares per symbol that belong to portfolios (so the global autotrader leaves them alone)."""
    cur.execute("""SELECT symbol, sum(CASE WHEN side = 'buy' THEN filled_qty ELSE -filled_qty END)
                   FROM portfolio_orders GROUP BY symbol""")
    return {s: float(q) for s, q in cur.fetchall() if q and q > 1e-9}


# -------------------------------------------------------------------------------------------------------- Alpaca I/O
def trading_client():
    """Alpaca PAPER client. ALPACA_PORTFOLIO_API_KEY/SECRET can point portfolios at their own paper account."""
    from alpaca.trading.client import TradingClient
    key = os.getenv("ALPACA_PORTFOLIO_API_KEY", "").strip() or env("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_PORTFOLIO_API_SECRET", "").strip() or env("ALPACA_API_SECRET")
    return TradingClient(key, secret, paper=True)


def _status(o) -> str:
    return str(getattr(o.status, "value", o.status))


def _apply(cur, coid: str, o) -> None:
    cur.execute("""UPDATE portfolio_orders SET alpaca_order_id = %s, status = %s, filled_qty = %s,
                   filled_avg_price = %s, updated_at = now() WHERE client_order_id = %s""",
                (str(o.id), _status(o), num(o.filled_qty) or 0.0, num(o.filled_avg_price), coid))


def place_order(tc, cur, pid: str, symbol: str, side: str, *, coid: str, asof, reason: str,
                notional: float | None = None, qty: float | None = None) -> dict | None:
    """Record the order, then submit it. The row is committed first, so a retry never double-submits (None = skipped)."""
    cur.execute("""INSERT INTO portfolio_orders (client_order_id, portfolio_id, symbol, side, notional, qty, status, reason, asof)
                   VALUES (%s,%s,%s,%s,%s,%s,'pending_submit',%s,%s) ON CONFLICT (client_order_id) DO NOTHING
                   RETURNING client_order_id""", (coid, pid, symbol, side, notional, qty, reason, asof))
    if not cur.fetchone():
        return None
    cur.connection.commit()
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
    try:
        req = MarketOrderRequest(symbol=symbol, side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                                 time_in_force=TimeInForce.DAY, client_order_id=coid,
                                 **({"notional": round(notional, 2)} if notional else {"qty": round(qty, 6)}))
        _apply(cur, coid, tc.submit_order(req))
    except Exception as e:  # noqa: BLE001  one rejected order must not stop the rest
        log.warning("order %s failed: %s", coid, e)
        cur.execute("UPDATE portfolio_orders SET status = %s, updated_at = now() WHERE client_order_id = %s",
                    (f"error: {str(e)[:150]}", coid))
    cur.connection.commit()
    return {"symbol": symbol, "side": side}


def reconcile(cur, tc, pid: str | None = None) -> int:
    """Pull the latest status/fills from Alpaca for every non-terminal order (one portfolio, or all)."""
    cur.execute("""SELECT client_order_id, status, created_at < now() - interval '10 minutes' FROM portfolio_orders
                   WHERE (%s::text IS NULL OR portfolio_id = %s)""", (pid, pid))
    n = 0
    for coid, status, stale in cur.fetchall():
        if not is_open(status):
            continue
        try:
            _apply(cur, coid, tc.get_order_by_client_id(coid))
            n += 1
        except Exception as e:  # noqa: BLE001
            if status == "pending_submit" and stale:  # crashed between INSERT and submit: Alpaca never saw it
                cur.execute("UPDATE portfolio_orders SET status = 'error: not submitted', updated_at = now() "
                            "WHERE client_order_id = %s", (coid,))
            else:
                log.warning("reconcile %s: %s", coid, e)
    return n


# ------------------------------------------------------------------------------------------------------ create / read
def new_credentials() -> tuple[str, str, str]:
    """(public id, secret token, sha256(token)). Only the hash is stored."""
    token = secrets.token_urlsafe(32)
    return secrets.token_urlsafe(9), token, hashlib.sha256(token.encode()).hexdigest()


def token_ok(cur, pid: str, token: str) -> bool:
    cur.execute("SELECT token_hash FROM portfolios WHERE id = %s", (pid,))
    r = cur.fetchone()
    return bool(r) and secrets.compare_digest(r[0], hashlib.sha256(token.encode()).hexdigest())


def _guards(cur, client_hash: str, budget: float) -> None:
    """The create endpoint is public and moves (paper) money: cap per-client rate, daily volume and total capital."""
    cur.execute("SELECT count(*) FROM portfolios WHERE client_hash = %s AND created_at > now() - interval '1 hour'",
                (client_hash,))
    if cur.fetchone()[0] >= _float_env("PORTFOLIO_MAX_PER_CLIENT_HOUR", 5):
        raise PortfolioError(429, "Too many portfolios from this connection. Try again later.")
    cur.execute("SELECT count(*) FROM portfolios WHERE created_at > now() - interval '24 hours'")
    if cur.fetchone()[0] >= _float_env("PORTFOLIO_MAX_PER_DAY", 200):
        raise PortfolioError(429, "The daily portfolio limit has been reached. Try again tomorrow.")
    cur.execute("SELECT coalesce(sum(budget), 0) FROM portfolios WHERE status = 'active'")
    if float(cur.fetchone()[0]) + budget > _float_env("PORTFOLIO_MAX_TOTAL_USD", 100000):
        raise PortfolioError(503, "The paper account has no capacity for another portfolio right now.")


def create_portfolio(cur, profile: dict, client_hash: str) -> tuple[str, str]:
    """Generate the portfolio from the latest signals, store it, and place the initial paper buys. Returns (id, token)."""
    p = normalize_profile(profile.get("budget"), profile.get("risk"), profile.get("horizon"), profile.get("sectors"))
    _guards(cur, client_hash, p["budget"])
    asof, cands = load_candidates(cur)
    if asof is None or (datetime.now(ET).date() - asof).days > MAX_SIGNAL_AGE_DAYS:
        raise PortfolioError(503, "No fresh model signals yet, so a portfolio can't be built. Try again after the next cycle.")
    sel = select_portfolio(cands, p["risk"], p["sectors"])
    trade = trading_enabled() and bool(sel["targets"])
    tc = None
    if trade:
        try:
            tc = trading_client()
            if float(tc.get_account().buying_power) < p["budget"]:
                raise PortfolioError(503, "The paper account has insufficient buying power right now.")
        except PortfolioError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("alpaca unavailable: %s", e)
            raise PortfolioError(503, "The paper-trading broker is unavailable. Please try again shortly.") from None
    pid, token, token_hash = new_credentials()
    spy = latest_prices(cur, ["SPY"]).get("SPY")
    cur.execute("""INSERT INTO portfolios (id, token_hash, budget, risk, horizon, sectors, targets, spy_start, last_rebalance,
                   client_hash) VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s)""",
                (pid, token_hash, p["budget"], p["risk"], p["horizon"], json.dumps(p["sectors"]),
                 json.dumps(sel["targets"]), spy, asof, client_hash))
    cur.execute("INSERT INTO portfolio_snapshots (portfolio_id, asof, value, cash, spy_price) VALUES (%s,%s,%s,%s,%s)",
                (pid, datetime.now(ET).date(), p["budget"], p["budget"], spy))
    cur.connection.commit()
    if trade:
        for t in sel["targets"]:
            amount = math.floor(p["budget"] * t["weight"] * 100) / 100
            if amount >= MIN_ORDER_USD:
                place_order(tc, cur, pid, t["symbol"], "buy", coid=f"pf-{pid}-init-{t['symbol']}", asof=asof,
                            reason=t["reason"], notional=amount)
    return pid, token


def refresh(cur, pid: str) -> None:
    """Best effort: pull fresh fill status for this portfolio's open orders before showing it."""
    if not any(is_open(o["status"]) for o in _orders(cur, pid)):
        return
    try:
        reconcile(cur, trading_client(), pid)
    except Exception as e:  # noqa: BLE001  stale status is better than an error page
        log.warning("refresh %s skipped: %s", pid, e)


def view(cur, pid: str) -> dict | None:
    """Everything the dashboard shows for one portfolio: targets, real positions, order states, value vs SPY."""
    pf = _portfolio(cur, pid)
    if not pf:
        return None
    orders = _orders(cur, pid)
    led = ledger(orders, pf["budget"])
    symbols = set(led["qty"]) | {t["symbol"] for t in pf["targets"]} | {"SPY"}
    prices = latest_prices(cur, symbols)
    value = total_value(led, prices)
    holdings = sum(q * (prices.get(s) or 0) for s, q in led["qty"].items())
    meta = lambda s: SYMBOLS.get(s, (s, "Other"))  # noqa: E731
    positions = []
    for s, q in sorted(led["qty"].items(), key=lambda kv: -kv[1] * (prices.get(kv[0]) or 0)):
        px, basis = prices.get(s), led["cost"][s]
        mv = q * px if px else None
        positions.append({"symbol": s, "company": meta(s)[0], "sector": meta(s)[1], "qty": q,
                          "avg_cost": basis / q if q else None, "price": px, "market_value": mv,
                          "pnl": mv - basis if mv is not None else None,
                          "pnl_pct": 100 * (mv / basis - 1) if mv is not None and basis else None,
                          "weight": 100 * mv / value if mv is not None and value else None})
    targets = [{**t, "company": meta(t["symbol"])[0], "sector": meta(t["symbol"])[1], "price": prices.get(t["symbol"]),
                "target_usd": value * t["weight"], "weight": 100 * t["weight"]} for t in pf["targets"]]
    cur.execute("SELECT asof, value, spy_price FROM portfolio_snapshots WHERE portfolio_id = %s ORDER BY asof", (pid,))
    spy0, spy = pf["spy_start"], prices.get("SPY")
    history = [{"date": d, "value": v, "return_pct": 100 * (v / pf["budget"] - 1),
                "spy_return_pct": 100 * (sp / spy0 - 1) if sp and spy0 else None} for d, v, sp in cur.fetchall()]
    queued = [o for o in orders if is_open(o["status"])]
    notes = []
    if not pf["targets"]:
        notes.append("The model has no long signals for your profile right now, so the portfolio is held in cash. "
                     "It is re-checked after each daily model run.")
    if queued:
        notes.append(f"{len(queued)} paper order(s) are queued with Alpaca. Orders placed outside market hours fill "
                     "at the next open.")
    if not trading_enabled():
        notes.append("Paper trading is switched off on the server, so no orders are being placed.")
    return {
        "id": pf["id"], "status": pf["status"], "created_at": pf["created_at"], "last_rebalance": pf["last_rebalance"],
        "profile": {"budget": pf["budget"], "risk": pf["risk"], "horizon": pf["horizon"], "sectors": pf["sectors"]},
        "summary": {"value": value, "cash": led["cash"], "invested": holdings, "pnl": value - pf["budget"],
                    "pnl_pct": 100 * (value / pf["budget"] - 1),
                    "spy_return_pct": 100 * (spy / spy0 - 1) if spy and spy0 else None,
                    "pending_orders": len(queued)},
        "targets": targets, "positions": positions, "history": history, "notes": notes,
        "orders": [{k: o[k] for k in ("symbol", "side", "notional", "filled_qty", "filled_avg_price", "status",
                                      "reason", "created_at")} for o in sorted(orders, key=lambda o: o["created_at"], reverse=True)[:50]],
    }


# ------------------------------------------------------------------------------------------------- daily cycle steps
def rebalance(cur, tc, pf: dict, asof, cands: list[dict]) -> int:
    """Move one portfolio toward the target weights of the newest signals. Idempotent per (portfolio, asof)."""
    sel = select_portfolio(cands, pf["risk"], pf["sectors"])
    cur.execute("UPDATE portfolios SET targets = %s::jsonb, last_rebalance = %s WHERE id = %s",
                (json.dumps(sel["targets"]), asof, pf["id"]))
    led = ledger(_orders(cur, pf["id"]), pf["budget"])
    prices = latest_prices(cur, set(led["qty"]) | {t["symbol"] for t in sel["targets"]})
    value = total_value(led, prices)
    want = {t["symbol"]: value * t["weight"] for t in sel["targets"]}
    band = max(MIN_ORDER_USD, REBALANCE_BAND * value)
    sells, buys = [], []
    for sym in set(want) | set(led["qty"]):
        px = prices.get(sym)
        if sym in led["open_symbols"] or not px:
            continue
        held = led["qty"].get(sym, 0.0)
        delta = want.get(sym, 0.0) - held * px
        if held > 0 and sym not in want:
            sells.append((sym, held, held * px))
        elif held > 0 and delta <= -band:
            sells.append((sym, -delta / px, -delta))
        elif delta >= band:
            buys.append((sym, delta))
    placed = 0
    free = led["cash"] - led["reserved"] + sum(n for _, _, n in sells)  # sells and buys all queue for the next open
    for sym, q, _ in sells:
        placed += bool(place_order(tc, cur, pf["id"], sym, "sell", coid=f"pf-{pf['id']}-{asof}-{sym}-sell", asof=asof,
                                   reason="Rebalance to the latest model signals", qty=q))
    for sym, delta in sorted(buys, key=lambda b: -b[1]):
        amount = math.floor(min(delta, free) * 100) / 100
        if amount >= MIN_ORDER_USD:
            reason = next((t["reason"] for t in sel["targets"] if t["symbol"] == sym), "Rebalance")
            placed += bool(place_order(tc, cur, pf["id"], sym, "buy", coid=f"pf-{pf['id']}-{asof}-{sym}-buy", asof=asof,
                                       reason=reason, notional=amount))
            free -= amount
    return placed


def snapshot(cur, pf: dict) -> None:
    led = ledger(_orders(cur, pf["id"]), pf["budget"])
    prices = latest_prices(cur, set(led["qty"]) | {"SPY"})
    cur.execute("""INSERT INTO portfolio_snapshots (portfolio_id, asof, value, cash, spy_price) VALUES (%s,%s,%s,%s,%s)
                   ON CONFLICT (portfolio_id, asof) DO UPDATE SET value = EXCLUDED.value, cash = EXCLUDED.cash,
                   spy_price = EXCLUDED.spy_price""",
                (pf["id"], datetime.now(ET).date(), total_value(led, prices), led["cash"], prices.get("SPY")))


def liquidate(cur, tc, pf: dict) -> int:
    """Sell every settled position of a closed portfolio (one order per symbol per day)."""
    led = ledger(_orders(cur, pf["id"]), pf["budget"])
    day, placed = datetime.now(ET).date(), 0
    for sym, q in led["qty"].items():
        if q > 0 and sym not in led["open_symbols"]:
            placed += bool(place_order(tc, cur, pf["id"], sym, "sell", coid=f"pf-{pf['id']}-close-{day}-{sym}", asof=day,
                                       reason="Portfolio closed by its owner", qty=q))
    return placed


def close(cur, pid: str) -> None:
    """Stop managing a portfolio: cancel unfilled buys, sell what it holds, never rebalance it again."""
    cur.execute("UPDATE portfolios SET status = 'closed' WHERE id = %s", (pid,))
    cur.connection.commit()
    if not trading_enabled():
        return
    try:
        tc = trading_client()
        for o in _orders(cur, pid):
            if o["side"] == "buy" and is_open(o["status"]):
                cur.execute("SELECT alpaca_order_id FROM portfolio_orders WHERE client_order_id = %s", (o["client_order_id"],))
                aid = cur.fetchone()[0]
                if aid:
                    try:
                        tc.cancel_order_by_id(aid)
                    except Exception as e:  # noqa: BLE001  already filled/cancelled
                        log.info("cancel %s: %s", aid, e)
        reconcile(cur, tc, pid)
        liquidate(cur, tc, _portfolio(cur, pid))
    except Exception as e:  # noqa: BLE001  the daily cycle finishes the job
        log.warning("close %s: will finish in the next cycle (%s)", pid, e)


def main(argv=None) -> int:
    """python -m src.portfolio [--dry-run]: reconcile fills, rebalance on the newest signals, snapshot."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="reconcile and snapshot only, never submit orders")
    a = ap.parse_args(argv)
    with tiger() as conn, conn.cursor() as cur:
        cur.execute("""SELECT id FROM portfolios WHERE status = 'active' OR EXISTS (
                           SELECT 1 FROM portfolio_orders o WHERE o.portfolio_id = portfolios.id
                           AND o.status NOT IN ('filled','canceled','expired','rejected','done_for_day','replaced')
                           AND o.status NOT LIKE 'error%') OR (status = 'closed' AND EXISTS (
                           SELECT 1 FROM portfolio_orders o WHERE o.portfolio_id = portfolios.id AND o.filled_qty > 0))
                       ORDER BY created_at""")
        ids = [r[0] for r in cur.fetchall()]
        if not ids:
            log.info("no portfolios to process")
            return 0
        asof, cands = load_candidates(cur)
        fresh = asof is not None and (datetime.now(ET).date() - asof).days <= MAX_SIGNAL_AGE_DAYS
        try:
            tc = trading_client()
        except RuntimeError as e:  # no Alpaca keys on this machine: still snapshot
            log.warning("%s; skipping reconcile and rebalance", e)
            tc = None
        for pid in ids:
            try:
                if tc:
                    reconcile(cur, tc, pid)
                    conn.commit()
                pf = _portfolio(cur, pid)
                if tc and trading_enabled() and not a.dry_run:
                    if pf["status"] == "closed":
                        liquidate(cur, tc, pf)
                    elif fresh and pf["last_rebalance"] != asof:
                        log.info("%s: %d rebalance order(s)", pid, rebalance(cur, tc, pf, asof, cands))
                if pf["status"] == "active":
                    snapshot(cur, pf)
                conn.commit()
            except Exception:  # noqa: BLE001  one bad portfolio must not block the others
                conn.rollback()
                log.exception("portfolio %s failed", pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
