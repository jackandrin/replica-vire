"""Direction vector + sizing + trade actions. Pure functions (no DB, no Alpaca) so they are unit-testable.

For each stock the fusion network gives (mu_h, sigma_h) for 6 horizons. Per horizon:
  score_h  = sign(mu_h) * max(|mu_h| - round-trip cost, 0) / sigma_h        (cost-adjusted Sharpe-like score)
  height_h = |score_h| / score_scale_h      (score_scale_h = std of score_h on validation, saved by train.py,
                                             so a height of 1 means "a typical-strength signal for this horizon")
Chosen horizon = the allowed horizon with the largest height. Direction vector = (expected % = mu_h*,
timeframe = h*), its height = height_h*. Horizons whose exit lands in the SAME session (a day trade) are only
allowed when their height clears day_trade_min_height ("a huge sudden change").
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
SIGNAL_COLS = ["symbol", "direction", "expected_pct", "timeframe", "height", "suggested_usd", "exit_at", "h_idx"]


def exit_time(t: datetime, h: str) -> datetime:
    """When a position opened at t for horizon h should close (weekday calendar; holidays roll at runtime)."""
    t = t.astimezone(ET)

    def next_open(x):
        x = (x + timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)
        while x.weekday() >= 5:
            x += timedelta(days=1)
        return x
    if h.endswith("h"):
        T = t + timedelta(hours=int(h[:-1]))
        o, c = T.replace(hour=9, minute=30, second=0, microsecond=0), T.replace(hour=16, minute=0, second=0, microsecond=0)
        if T.weekday() < 5 and o < T <= c:
            return T
        if T.weekday() < 5 and T <= o:
            return o
        return next_open(T)
    T, n = t, int(h[:-1])
    while n:
        T += timedelta(days=1)
        n -= T.weekday() < 5
    return T


def size_usd(height: float, st: dict) -> float:
    """Linear between min_usd (at min_height) and max_usd (at full_height); 0 below min_height."""
    if height < st["min_height"]:
        return 0.0
    span = st.get("full_height", 2.0) - st["min_height"]
    frac = 1.0 if span <= 0 else min(max((height - st["min_height"]) / span, 0.0), 1.0)
    return round(st["min_usd"] + (st["max_usd"] - st["min_usd"]) * frac, 2)


def build_signals(p: pd.DataFrame, st: dict, horizons: list, score_scale: dict, cost_bps: float = 5,
                  allowed=None) -> pd.DataFrame:
    """p: symbol, ts (aware datetime), mu_<h>, sigma_<h> (fractions). Returns SIGNAL_COLS."""
    if p.empty:
        return pd.DataFrame(columns=SIGNAL_COLS)
    allowed = set(allowed or horizons)
    cost = 2 * cost_bps / 1e4
    mu = np.column_stack([p[f"mu_{h}"].values for h in horizons])
    sg = np.column_stack([p[f"sigma_{h}"].values for h in horizons])
    score = np.sign(mu) * np.clip(np.abs(mu) - cost, 0, None) / np.where(sg > 0, sg, np.nan)
    height = np.abs(score) / np.array([score_scale.get(h) or 1.0 for h in horizons])
    height = np.nan_to_num(height)
    rows = []
    for i, r in enumerate(p.itertuples()):
        t = pd.Timestamp(r.ts).to_pydatetime()
        cand = []
        for j, h in enumerate(horizons):
            if h not in allowed:
                continue
            ex = exit_time(t, h)
            same_day = ex.astimezone(ET).date() == t.astimezone(ET).date()
            if same_day and height[i, j] < st["day_trade_min_height"]:
                continue  # day trades only on a large sudden signal
            cand.append((height[i, j], j, ex))
        hgt, j, ex = max(cand) if cand else (0.0, horizons.index("1d") if "1d" in horizons else 0, None)
        d = "flat"
        if hgt >= st["min_height"] and score[i, j] != 0:
            d = "long" if score[i, j] > 0 else "short"
        mode = st.get("direction_mode", "both")
        if (mode == "long_only" and d == "short") or (mode == "short_only" and d == "long"):
            d = "flat"
        rows.append({"symbol": r.symbol, "direction": d, "expected_pct": 100 * float(mu[i, j]),
                     "timeframe": horizons[j], "height": float(hgt),
                     "suggested_usd": size_usd(hgt, st) if d != "flat" else 0.0, "exit_at": ex, "h_idx": j})
    return pd.DataFrame(rows, columns=SIGNAL_COLS)


def plan_actions(sig: pd.DataFrame, held: dict, st: dict, opened_today: set, sectors: dict,
                 due_exits: set = frozenset()) -> list:
    """Turn signals into actions given current positions.
    held: symbol -> signed qty (+long, -short). opened_today: symbols opened this session (closing them is a day
    trade: only allowed when height >= day_trade_min_height). due_exits: held symbols whose planned exit time
    has passed. Returns dicts: symbol, action (BUY|SELL|SHORT|COVER), size_usd, height, expected_pct, timeframe,
    exit_at, why ("signal" | "flip" | "horizon")."""
    allowed = set(st.get("sectors") or [])
    out, opens = [], []
    by_sym = {r.symbol: r for r in sig.itertuples()}
    for sym, q in held.items():  # planned exits first (horizon reached)
        if q and sym in due_exits:
            r = by_sym.get(sym)
            out.append({"symbol": sym, "action": "SELL" if q > 0 else "COVER", "size_usd": None,
                        "height": float(r.height) if r else 0.0, "expected_pct": float(r.expected_pct) if r else 0.0,
                        "timeframe": r.timeframe if r else None, "exit_at": None, "why": "horizon"})
    done = {a["symbol"] for a in out}
    for r in sig.itertuples():
        if r.symbol in done or (allowed and sectors.get(r.symbol) not in allowed):
            continue
        q = held.get(r.symbol, 0.0)
        base = {"symbol": r.symbol, "height": float(r.height), "expected_pct": float(r.expected_pct),
                "timeframe": r.timeframe, "exit_at": r.exit_at}
        close = "SELL" if q > 0 and r.direction == "short" else "COVER" if q < 0 and r.direction == "long" else None
        if close:
            if r.symbol in opened_today and r.height < st["day_trade_min_height"]:
                continue  # day-trade restriction
            out.append({**base, "action": close, "size_usd": None, "exit_at": None, "why": "flip"})
        elif q == 0 and r.direction in ("long", "short"):
            opens.append({**base, "action": "BUY" if r.direction == "long" else "SHORT",
                          "size_usd": float(r.suggested_usd), "why": "signal"})
    slots = max(0, st["max_positions"] - sum(1 for q in held.values() if q != 0))
    opens = sorted(opens, key=lambda a: -a["height"])[:slots]
    return out + [a for a in opens if a["size_usd"] > 0]
