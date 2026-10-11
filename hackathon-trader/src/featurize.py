"""Decision-time features and multi-horizon labels (v2). Pure pandas/numpy: no DB, no network.

Used by src/train.py (on the HPC, from the parquet bundle) and src/predict.py (live, from Tiger Data).

Time conventions (docs/V2_CONTRACT.md):
  * Regular session 09:30-16:00 ET is cut into 26 fifteen-minute slots; slot k covers [09:30+15k, 09:45+15k).
  * A decision at time t (e.g. 10:00 ET) sees the close of the bar ENDING at t, daily bars of PRIOR days
    (plus today's open), and events / sentiment with available_at / published_at <= t. Nothing else.
  * Labels: 6h/12h/18h are wall-clock. If t+h falls inside a session the price is the close of the last bar
    ending at or before t+h; otherwise it rolls forward to the next regular-session open ("first price
    tomorrow"). 1d/3d/5d are the same slot 1/3/5 trading days later. Label = price / price(t) - 1 (raw return).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ET = "America/New_York"
OPEN_MIN, SLOT_MIN, N_SLOTS = 570, 15, 26          # 09:30 = 570 minutes after midnight
WARMUP_DAYS = 260                                   # 52-week features need ~1 year of daily history
HORIZONS = ["6h", "12h", "18h", "1d", "3d", "5d"]

FAMILIES = {
    "reversal": ["r_today", "r_prev1", "r_5d", "r_5d_rel_sector"],
    "overnight_intraday": ["gap_today", "intraday_today", "prev_intraday", "prev_overnight"],
    "intraday_seasonality": ["seas_1h", "seas_close", "seas_overnight"],
    "market_momentum": ["spy_first30", "spy_today", "spy_prev1", "beta_x_spy_today"],
    "sector_leadlag": ["etf_today", "etf_prev1", "etf_r5", "resid_today"],
    "abnormal_volume": ["cumvol_ratio", "prev_vol_ratio"],
    "volatility": ["rv5", "rv21", "atr14", "range_today_atr"],
    "oscillators": ["rsi2", "rsi14", "bb20"],
    "vwap_ma_distance": ["vwap_dist", "ma10_dist", "ma20_dist", "ma50_dist"],
    "trend": ["macd_hist", "mom21", "mom63"],
    "high_52w": ["hi52_dist", "lo52_dist"],
    "earnings": ["earn_days_since", "earn_ear", "earn_days_to_est"],
    "analyst": ["analyst_score", "analyst_d1", "analyst_d3"],
    "short_selling": ["short_ratio", "short_ratio_z"],
    "insider": ["insider_net30", "insider_buys90", "insider_sells90"],
}
FAMILY_LABELS = {
    "reversal": "Short-term reversal", "overnight_intraday": "Overnight vs intraday",
    "intraday_seasonality": "Intraday seasonality", "market_momentum": "Market momentum",
    "sector_leadlag": "Sector lead-lag", "abnormal_volume": "Abnormal volume", "volatility": "Volatility state",
    "oscillators": "RSI / Bollinger", "vwap_ma_distance": "VWAP & moving averages", "trend": "Trend (MACD, momentum)",
    "high_52w": "52-week high", "earnings": "Earnings event", "analyst": "Analyst ratings",
    "short_selling": "Short selling", "insider": "Insider trading",
}
SENTIMENT = ["sent_level", "sent_trend_24h", "sent_trend_5d", "sent_intensity"]
CONTEXT = ["spy_rv21", "spy_mom21", "vixy_r1", "vixy_r5", "etf_mom21", "beta60", "log_dvol20", "hl_spread5",
           "min_to_close", "dow_sin", "dow_cos", "month_end", "earn_window"]
RAW_FEATURES = [c for cols in FAMILIES.values() for c in cols]
LABELS = [f"y_{h}" for h in HORIZONS]
assert len(FAMILIES) + len(SENTIMENT) + len(CONTEXT) == 32  # fusion-network input width

LABEL_TEXT = {  # feature -> readable name (explanations in the dashboard)
    "r_today": "return since yesterday's close", "r_prev1": "yesterday's return", "r_5d": "5-day return",
    "r_5d_rel_sector": "5-day return vs sector", "gap_today": "overnight gap", "intraday_today": "return since open",
    "prev_intraday": "yesterday open-to-close", "prev_overnight": "yesterday's gap",
    "seas_1h": "typical next-hour move at this time", "seas_close": "typical move to the close at this time",
    "seas_overnight": "typical overnight move", "spy_first30": "S&P 500 first 30 minutes",
    "spy_today": "S&P 500 today", "spy_prev1": "S&P 500 yesterday", "beta_x_spy_today": "beta x S&P 500 today",
    "etf_today": "sector ETF today", "etf_prev1": "sector ETF yesterday", "etf_r5": "sector ETF 5-day",
    "resid_today": "move vs sector today", "cumvol_ratio": "volume vs normal for this time",
    "prev_vol_ratio": "yesterday's volume vs normal", "rv5": "5-day volatility", "rv21": "1-month volatility",
    "atr14": "average true range", "range_today_atr": "today's range vs ATR", "rsi2": "RSI(2)", "rsi14": "RSI(14)",
    "bb20": "Bollinger position", "vwap_dist": "price vs VWAP", "ma10_dist": "price vs 10-day average",
    "ma20_dist": "price vs 20-day average", "ma50_dist": "price vs 50-day average", "macd_hist": "MACD histogram",
    "mom21": "1-month momentum", "mom63": "3-month momentum", "hi52_dist": "distance from 52-week high",
    "lo52_dist": "distance from 52-week low", "earn_days_since": "time since earnings",
    "earn_ear": "last earnings reaction", "earn_days_to_est": "time to next expected earnings",
    "analyst_score": "analyst consensus", "analyst_d1": "analyst change (1 month)",
    "analyst_d3": "analyst change (3 months)", "short_ratio": "short-volume ratio", "short_ratio_z": "short volume vs normal",
    "insider_net30": "net insider buying (30d)", "insider_buys90": "insider buys (90d)",
    "insider_sells90": "insider sales (90d)", "sent_level": "news sentiment level",
    "sent_trend_24h": "sentiment change (24h)", "sent_trend_5d": "sentiment change (5d)",
    "sent_intensity": "news intensity", "spy_rv21": "market volatility", "spy_mom21": "market 1-month trend",
    "vixy_r1": "VIX futures yesterday", "vixy_r5": "VIX futures 5-day", "etf_mom21": "sector 1-month trend",
    "beta60": "beta", "log_dvol20": "liquidity", "hl_spread5": "trading-range spread", "min_to_close": "time to close",
    "dow_sin": "day of week", "dow_cos": "day of week", "month_end": "days to month end",
    "earn_window": "earnings window",
}


# ------------------------------------------------------------------ grids
def slot_of_time(hhmm: str) -> int:
    """Decision time 'HH:MM' -> index of the bar that ENDS at that time."""
    h, m = map(int, hhmm.split(":"))
    k = (h * 60 + m - OPEN_MIN) // SLOT_MIN - 1
    if not 0 <= k < N_SLOTS:
        raise ValueError(f"decision time {hhmm} is outside 09:45-16:00 ET")
    return k


def trading_days(bars_15m: pd.DataFrame, bars_1d: pd.DataFrame | None = None, market: str = "SPY") -> np.ndarray:
    """Union of the market's intraday and daily dates (live runs load a year of daily but ~40 days of 15-min)."""
    m = bars_15m[bars_15m.symbol == market]
    d = set(pd.to_datetime(m["ts"], utc=True).dt.tz_convert(ET).dt.date)
    if bars_1d is not None:
        d |= set(pd.to_datetime(bars_1d.loc[bars_1d.symbol == market, "d"]).dt.date)
    return np.array(sorted(d), dtype="datetime64[D]")


def intraday_cube(g: pd.DataFrame, days: np.ndarray) -> dict:
    """One symbol's regular-session 15-min bars -> (n_days, 26) arrays (NaN where missing)."""
    et = pd.to_datetime(g["ts"], utc=True).dt.tz_convert(ET)
    slot = ((et.dt.hour * 60 + et.dt.minute - OPEN_MIN) // SLOT_MIN).values
    di = np.searchsorted(days, et.dt.date.values.astype("datetime64[D]"))
    ok = (slot >= 0) & (slot < N_SLOTS) & (di < len(days))
    ok &= days[np.minimum(di, len(days) - 1)] == et.dt.date.values.astype("datetime64[D]")
    out = {}
    for c in ("open", "high", "low", "close", "volume", "vwap"):
        a = np.full((len(days), N_SLOTS), np.nan)
        a[di[ok], slot[ok]] = g[c].values[ok].astype(float)
        out[c] = a
    return out


def daily_arrays(g: pd.DataFrame, days: np.ndarray) -> dict:
    s = g.assign(d=pd.to_datetime(g["d"]).dt.normalize()).set_index("d").sort_index()
    s = s[~s.index.duplicated(keep="last")].reindex(pd.DatetimeIndex(days))
    return {c: s[c].values.astype(float) for c in ("open", "high", "low", "close", "volume")}


def lag(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    if n < len(a):
        out[n:] = a[: len(a) - n]
    return out


def roll(a: np.ndarray, n: int, fn: str, minp: int | None = None) -> np.ndarray:
    """Rolling stat over PRIOR days only (window ends at t-1)."""
    r = getattr(pd.Series(a).rolling(n, min_periods=minp or max(2, n // 2)), fn)()
    return lag(r.values, 1)


def today_open(b: dict) -> np.ndarray:
    """Today's open from the first 15-min bar (the daily bar does not exist yet intraday), else the daily open."""
    o = b["i"]["open"][:, 0]
    return np.where(np.isnan(o), b["d"]["open"], o)


def ffill_slots(a: np.ndarray, first: np.ndarray) -> np.ndarray:
    """Forward-fill missing bars within each day; a day's leading gap gets `first` (that day's open)."""
    df = pd.DataFrame(a.T).ffill().T.values
    return np.where(np.isnan(df), first[:, None], df)


def _rsi_prev(close: np.ndarray, n: int):
    d = np.diff(close, prepend=np.nan)
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().values
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().values
    return lag(up, 1), lag(dn, 1)


def _rsi_now(up_prev, dn_prev, ch, n):
    up = (up_prev * (n - 1) + np.clip(ch, 0, None)) / n
    dn = (dn_prev * (n - 1) + np.clip(-ch, 0, None)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 - 100 / (1 + up / dn)


# ------------------------------------------------------------------ price-based families
def price_block(sym: dict, spy: dict, etf: dict, vixy: dict | None, days: np.ndarray, slots: list) -> dict:
    """All bar-derived features + labels for one symbol. Returns {name: (n_days, n_slots) array}."""
    K = np.array(slots)
    D = sym["d"]
    c_d, h_d, l_d, v_d = D["close"], D["high"], D["low"], D["volume"]
    o_d = today_open(sym)
    I = sym["i"]
    C = ffill_slots(I["close"], o_d)
    V = np.nan_to_num(I["volume"])
    H = np.fmax.accumulate(np.where(np.isnan(I["high"]), -np.inf, I["high"]), axis=1)
    L = np.fmin.accumulate(np.where(np.isnan(I["low"]), np.inf, I["low"]), axis=1)
    P = C[:, K]                                           # price at each decision
    prev_c = lag(c_d, 1)
    o_prev = lag(D["open"], 1)
    col = lambda a: a[:, None]                            # noqa: E731  daily -> broadcast over slots
    f = {}

    def ret(b):
        return ffill_slots(b["i"]["close"], today_open(b))[:, K], lag(b["d"]["close"], 1)

    # 1 reversal
    f["r_today"] = P / col(prev_c) - 1
    f["r_prev1"] = col(prev_c / lag(c_d, 2) - 1)
    f["r_5d"] = P / col(lag(c_d, 5)) - 1
    eP, e_prev = ret(etf)
    etf_r5 = eP / col(lag(etf["d"]["close"], 5)) - 1
    f["r_5d_rel_sector"] = f["r_5d"] - etf_r5
    # 2 overnight vs intraday
    f["gap_today"] = col(o_d / prev_c - 1)
    f["intraday_today"] = P / col(o_d) - 1
    f["prev_intraday"] = col(prev_c / o_prev - 1)
    f["prev_overnight"] = col(o_prev / lag(c_d, 2) - 1)
    # 3 intraday seasonality: same slot on the previous 20 days
    nxt = np.minimum(K + 4, N_SLOTS - 1)
    r1h = C[:, nxt] / C[:, K] - 1
    rcl = C[:, [N_SLOTS - 1]] / C[:, K] - 1
    f["seas_1h"] = np.column_stack([roll(r1h[:, j], 20, "mean") for j in range(len(K))])
    f["seas_close"] = np.column_stack([roll(rcl[:, j], 20, "mean") for j in range(len(K))])
    on = o_d / prev_c - 1
    f["seas_overnight"] = col(pd.Series(on).rolling(20, min_periods=10).mean().values)  # includes today's gap (known)
    # 4 market momentum
    sP, s_prev = ret(spy)
    s_c, s_o = spy["d"]["close"], today_open(spy)
    spy_c1 = ffill_slots(spy["i"]["close"], s_o)[:, 1]
    f["spy_first30"] = col(spy_c1 / s_o - 1) * (K >= 1)
    f["spy_today"] = sP / col(s_prev) - 1
    f["spy_prev1"] = col(s_prev / lag(s_c, 2) - 1)
    r_d, s_r = c_d / prev_c - 1, s_c / lag(s_c, 1) - 1
    cov = lag(pd.Series(r_d).rolling(60, min_periods=30).cov(pd.Series(s_r)).values, 1)
    var = lag(pd.Series(s_r).rolling(60, min_periods=30).var().values, 1)
    beta = np.clip(cov / var, -1, 4)
    f["beta_x_spy_today"] = col(beta) * f["spy_today"]
    # 5 sector lead-lag
    f["etf_today"] = eP / col(e_prev) - 1
    f["etf_prev1"] = col(e_prev / lag(etf["d"]["close"], 2) - 1)
    f["etf_r5"] = etf_r5
    f["resid_today"] = f["r_today"] - f["etf_today"]
    # 6 abnormal volume
    cv = np.cumsum(V, axis=1)[:, K]
    f["cumvol_ratio"] = cv / np.column_stack([roll(cv[:, j], 20, "mean") for j in range(len(K))])
    f["prev_vol_ratio"] = col(lag(v_d, 1) / roll(lag(v_d, 1), 20, "mean"))
    # 7 volatility
    f["rv5"] = col(roll(r_d, 5, "std"))
    f["rv21"] = col(roll(r_d, 21, "std"))
    tr = np.fmax(h_d - l_d, np.fmax(np.abs(h_d - prev_c), np.abs(l_d - prev_c)))
    atr = roll(tr, 14, "mean")
    f["atr14"] = col(atr / prev_c)
    f["range_today_atr"] = (H[:, K] - L[:, K]) / col(atr)
    # 8 oscillators (yesterday's state + the current price)
    ch = P - col(prev_c)
    for n in (2, 14):
        up, dn = _rsi_prev(c_d, n)
        f[f"rsi{n}"] = _rsi_now(col(up), col(dn), ch, n)
    ma20, sd20 = roll(c_d, 20, "mean"), roll(c_d, 20, "std")
    f["bb20"] = (P - col(ma20)) / col(2 * sd20)
    # 9 VWAP / moving-average distance
    vw = np.where(np.isnan(I["vwap"]), C, I["vwap"])
    with np.errstate(divide="ignore", invalid="ignore"):
        svwap = np.cumsum(vw * V, axis=1) / np.cumsum(V, axis=1)
    f["vwap_dist"] = P / svwap[:, K] - 1
    for n in (10, 20, 50):
        f[f"ma{n}_dist"] = P / col(roll(c_d, n, "mean")) - 1
    # 10 trend
    ema = lambda s, n: pd.Series(s).ewm(span=n, adjust=False).mean().values  # noqa: E731
    macd = ema(c_d, 12) - ema(c_d, 26)
    f["macd_hist"] = col(lag((macd - ema(macd, 9)) / c_d, 1))
    f["mom21"] = col(prev_c / lag(c_d, 22) - 1)
    f["mom63"] = col(prev_c / lag(c_d, 64) - 1)
    # 11 52-week high/low
    f["hi52_dist"] = P / col(roll(h_d, 252, "max", 200)) - 1
    f["lo52_dist"] = P / col(roll(l_d, 252, "min", 200)) - 1
    # context (price-based part)
    f["spy_rv21"] = col(roll(s_r, 21, "std"))
    f["spy_mom21"] = col(s_prev / lag(s_c, 22) - 1)
    if vixy is not None:
        vc = vixy["d"]["close"]
        f["vixy_r1"] = col(lag(vc, 1) / lag(vc, 2) - 1)
        f["vixy_r5"] = col(lag(vc, 1) / lag(vc, 6) - 1)
    else:
        f["vixy_r1"] = f["vixy_r5"] = np.zeros_like(P)
    f["etf_mom21"] = col(e_prev / lag(etf["d"]["close"], 22) - 1)
    f["beta60"] = col(beta)
    f["log_dvol20"] = col(np.log1p(roll(c_d * v_d, 20, "mean")))
    f["hl_spread5"] = col(roll((h_d - l_d) / c_d, 5, "mean"))
    f["min_to_close"] = np.broadcast_to((N_SLOTS - 1 - K) * SLOT_MIN / 390.0, P.shape).copy()
    dow = pd.DatetimeIndex(days).dayofweek.values
    f["dow_sin"], f["dow_cos"] = col(np.sin(2 * np.pi * dow / 5)), col(np.cos(2 * np.pi * dow / 5))
    month_last = (pd.DatetimeIndex(days) + pd.offsets.MonthEnd(0)).values.astype("datetime64[D]")
    left = np.busday_count(days, month_last)  # weekdays left in the month: calendar only, no data needed
    f["month_end"] = col(left / 22.0)
    # labels
    f.update(labels(C, o_d, days, K))
    f["_price"] = P
    f["_has_intraday"] = col(~np.isnan(I["close"]).all(axis=1))
    f["_dvol20"] = col(roll(c_d * v_d, 20, "mean"))
    f["_atr_pct"] = col(atr / prev_c)
    return f


def price_at_offsets(C: np.ndarray, o_d: np.ndarray, days: np.ndarray, K: np.ndarray, hours: float) -> np.ndarray:
    """Price at (decision time + `hours` wall clock), rolling closed-market times to the next open."""
    n = len(days)
    cal = days.astype("datetime64[D]")
    out = np.full((n, len(K)), np.nan)
    for j, k in enumerate(K):
        t_min = OPEN_MIN + (k + 1) * SLOT_MIN + hours * 60
        day_off, minute = int(t_min // 1440), t_min % 1440
        target = cal + np.timedelta64(day_off, "D")
        idx = np.searchsorted(cal, target)               # first trading day >= target date
        idx_c = np.minimum(idx, n - 1)
        is_td = (idx < n) & (cal[idx_c] == target)
        in_sess = is_td & (minute > OPEN_MIN) & (minute <= OPEN_MIN + N_SLOTS * SLOT_MIN)
        slot = int(np.clip((minute - OPEN_MIN) // SLOT_MIN - 1, 0, N_SLOTS - 1))
        before_open = is_td & (minute <= OPEN_MIN)
        nxt = np.where(is_td & ~before_open, idx + 1, idx)  # after the close or a non-trading day -> next open
        nxt_ok = nxt < n
        roll_px = np.where(nxt_ok, o_d[np.minimum(nxt, n - 1)], np.nan)
        out[:, j] = np.where(in_sess, C[idx_c, slot], roll_px)
    return out


def labels(C, o_d, days, K) -> dict:
    P = C[:, K]
    out = {}
    for h in HORIZONS:
        if h.endswith("h"):
            px = price_at_offsets(C, o_d, days, K, float(h[:-1]))
        else:
            n = int(h[:-1])
            px = np.full_like(P, np.nan)
            px[:-n] = C[n:, :][:, K]
        out[f"y_{h}"] = px / P - 1
    return out


# ------------------------------------------------------------------ event families
def _ns(x) -> np.ndarray:
    """Timestamps -> int64 nanoseconds since epoch, whatever resolution pandas inferred."""
    return pd.to_datetime(x, utc=True).dt.as_unit("ns").astype("int64").values


def _asof_idx(times: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.searchsorted(times, t, side="right") - 1


def event_block(ev: pd.DataFrame, t_ns: np.ndarray, day_idx: np.ndarray, days: np.ndarray, c_d: np.ndarray,
                dvol20: np.ndarray) -> dict:
    """ev: one symbol's events. t_ns: decision times (int64 ns, flattened). day_idx: day index per decision."""
    f = {}
    n = len(t_ns)
    # 12 earnings
    e = ev[ev.kind == "earnings"].sort_values("available_at")
    if len(e):
        av = pd.to_datetime(e["available_at"], utc=True)
        a_ns = _ns(av)
        et = av.dt.tz_convert(ET)
        e_day = np.searchsorted(days, et.dt.date.values.astype("datetime64[D]"))
        mins = (et.dt.hour * 60 + et.dt.minute).values
        is_td = (e_day < len(days)) & (days[np.minimum(e_day, len(days) - 1)] == et.dt.date.values.astype("datetime64[D]"))
        react = np.where(is_td & (mins < OPEN_MIN + N_SLOTS * SLOT_MIN), e_day, e_day + (is_td))  # day whose close reflects it
        react = np.minimum(react, len(days) - 1)
        ear = c_d[react] / c_d[np.maximum(react - 2, 0)] - 1
        i = _asof_idx(a_ns, t_ns)
        since = np.where(i >= 0, day_idx - e_day[np.maximum(i, 0)], 63).astype(float)
        f["earn_days_since"] = np.clip(since, 0, 63) / 63
        f["earn_days_to_est"] = np.clip(63 - since, -10, 63) / 63
        f["earn_window"] = (since <= 1).astype(float)
        # reaction only once its close is known: react day < decision day
        order = np.argsort(react, kind="stable")
        r_sorted, ear_sorted = react[order], ear[order]
        j = np.searchsorted(r_sorted, day_idx, side="left") - 1
        f["earn_ear"] = np.where(j >= 0, ear_sorted[np.maximum(j, 0)], 0.0)
    else:
        f["earn_days_since"], f["earn_days_to_est"] = np.ones(n), np.zeros(n)
        f["earn_window"], f["earn_ear"] = np.zeros(n), np.zeros(n)
    # 13 analyst consensus
    a = ev[ev.kind == "analyst"].sort_values("available_at")
    if len(a):
        a_ns = _ns(a["available_at"])
        v = a["value"].astype(float).values
        i = _asof_idx(a_ns, t_ns)
        g = lambda k: np.where(i - k >= 0, v[np.maximum(i - k, 0)], np.nan)  # noqa: E731
        f["analyst_score"] = np.nan_to_num(g(0))
        f["analyst_d1"] = np.nan_to_num(g(0) - g(1))
        f["analyst_d3"] = np.nan_to_num(g(0) - g(3))
    else:
        f["analyst_score"] = f["analyst_d1"] = f["analyst_d3"] = np.zeros(n)
    # 14 short selling
    s = ev[ev.kind == "short_volume"].sort_values("available_at")
    if len(s):
        s_ns = _ns(s["available_at"])
        v = s["value"].astype(float).values
        mu = pd.Series(v).rolling(20, min_periods=5).mean().shift(1).values
        sd = pd.Series(v).rolling(20, min_periods=5).std().shift(1).values
        z = (v - mu) / sd
        i = _asof_idx(s_ns, t_ns)
        ok = i >= 0
        f["short_ratio"] = np.where(ok, v[np.maximum(i, 0)], np.nan)
        f["short_ratio_z"] = np.where(ok, z[np.maximum(i, 0)], np.nan)
    else:
        f["short_ratio"], f["short_ratio_z"] = np.full(n, np.nan), np.zeros(n)
    # 15 insider trading
    ins = ev[ev.kind == "insider"].sort_values("available_at")
    if len(ins):
        i_ns = _ns(ins["available_at"])
        v = ins["value"].astype(float).values
        day = np.int64(86_400 * 10**9)

        def window(vals, days_back):
            cs = np.concatenate([[0.0], np.cumsum(vals)])
            return cs[_asof_idx(i_ns, t_ns) + 1] - cs[_asof_idx(i_ns, t_ns - days_back * day) + 1]
        f["insider_net30"] = window(v, 30) / np.where(dvol20 > 0, dvol20, np.nan)
        f["insider_buys90"] = np.log1p(window((v > 0).astype(float), 90))
        f["insider_sells90"] = np.log1p(window((v < 0).astype(float), 90))
    else:
        f["insider_net30"] = f["insider_buys90"] = f["insider_sells90"] = np.zeros(n)
    return f


def sentiment_block(se: pd.DataFrame, t_ns: np.ndarray, tau_days: float = 3.0) -> dict:
    """Agent events -> level (time-decayed, materiality x novelty weighted), 24h trend (weighted LLM deltas),
    5d trend (level change), intensity (recent weighted count). Point-in-time: only published_at <= t."""
    n = len(t_ns)
    out = {"sent_level": np.full(n, 50.0), "sent_trend_24h": np.zeros(n), "sent_trend_5d": np.zeros(n),
           "sent_intensity": np.zeros(n)}
    if se is None or se.empty:
        return out
    se = se.sort_values("published_at")
    e_ns = _ns(se["published_at"])
    w = (se["materiality"].astype(float) * (0.25 + 0.75 * se["novelty"].astype(float))).clip(0.02, 1).values
    lvl, dlt = se["level"].astype(float).values, se["delta"].astype(float).values
    day = 86_400e9

    def level_at(tq):
        order = np.argsort(tq)
        res = np.full(len(tq), 50.0)
        num = den = 0.0
        last_t, j = None, 0
        for q in order:
            while j < len(e_ns) and e_ns[j] <= tq[q]:
                if last_t is not None:
                    decay = np.exp(-(e_ns[j] - last_t) / (tau_days * day))
                    num, den = num * decay, den * decay
                num, den, last_t = num + w[j] * lvl[j], den + w[j], e_ns[j]
                j += 1
            if last_t is not None:
                decay = np.exp(-(tq[q] - last_t) / (tau_days * day))
                d = den * decay
                res[q] = (num * decay + 50.0 * 0.05) / (d + 0.05)  # shrink toward neutral when coverage is thin
        return res

    out["sent_level"] = level_at(t_ns)
    out["sent_trend_5d"] = out["sent_level"] - level_at(t_ns - np.int64(5 * day))
    cs_d = np.concatenate([[0.0], np.cumsum(w * dlt)])
    cs_w = np.concatenate([[0.0], np.cumsum(w)])
    i_now = _asof_idx(e_ns, t_ns) + 1
    i_24 = _asof_idx(e_ns, t_ns - np.int64(day)) + 1
    out["sent_trend_24h"] = cs_d[i_now] - cs_d[i_24]
    out["sent_intensity"] = np.log1p(cs_w[i_now] - cs_w[i_24])
    return out


# ------------------------------------------------------------------ assemble
def build(bars_15m: pd.DataFrame, bars_1d: pd.DataFrame, events: pd.DataFrame | None,
          sentiment: pd.DataFrame | None, symbols: list, sector_etf: dict, decision_times: list,
          market: str = "SPY", vol_proxy: str = "VIXY", min_day: int = WARMUP_DAYS, log=None) -> pd.DataFrame:
    """Return one row per (symbol, decision ts) with RAW_FEATURES + SENTIMENT + CONTEXT + LABELS.
    Labels are NaN where the future is not yet known (the live prediction rows)."""
    days = trading_days(bars_15m, bars_1d, market)
    if min_day < 0:  # live: only the last `-min_day` days
        min_day = max(0, len(days) + min_day)
    slots = [slot_of_time(t) for t in decision_times]
    K = np.array(slots)
    b15 = dict(tuple(bars_15m.groupby("symbol")))
    b1d = dict(tuple(bars_1d.groupby("symbol")))
    ev = dict(tuple(events.groupby("symbol"))) if events is not None and len(events) else {}
    se = dict(tuple(sentiment.groupby("symbol"))) if sentiment is not None and len(sentiment) else {}

    def bundle(sym):
        if sym not in b15 or sym not in b1d:
            return None
        return {"i": intraday_cube(b15[sym], days), "d": daily_arrays(b1d[sym], days)}

    spy, vixy = bundle(market), bundle(vol_proxy)
    if spy is None:
        raise RuntimeError(f"no bars for the market symbol {market}")
    etfs = {}
    # decision timestamps (UTC ns) for the whole grid
    base = pd.DatetimeIndex(days).as_unit("ns").tz_localize(ET)
    t_grid = np.stack([(base + pd.Timedelta(minutes=OPEN_MIN + (k + 1) * SLOT_MIN)).tz_convert("UTC").asi8
                       for k in K], axis=1)
    frames = []
    for sym in symbols:
        b = bundle(sym)
        if b is None or np.isnan(b["d"]["close"]).mean() > 0.5:
            if log:
                log.warning("%s: missing or sparse bars, skipped", sym)
            continue
        etf_sym = sector_etf.get(sym, market)
        if etf_sym not in etfs:
            etfs[etf_sym] = bundle(etf_sym) or spy
        with np.errstate(divide="ignore", invalid="ignore"):  # inf/NaN become NaN below; Prep imputes them
            f = price_block(b, spy, etfs[etf_sym], vixy, days, slots)
        rows = np.arange(min_day, len(days))
        t_ns = t_grid[rows].ravel()
        day_idx = np.repeat(rows, len(K))
        shape = f["_price"].shape
        flat = {k: np.broadcast_to(v, shape)[rows].ravel() for k, v in f.items()}
        flat.update(event_block(ev.get(sym, pd.DataFrame(columns=["kind", "available_at", "value"])),
                                t_ns, day_idx, days, b["d"]["close"], flat["_dvol20"]))
        flat.update(sentiment_block(se.get(sym), t_ns))
        df = pd.DataFrame(flat)
        df.insert(0, "ts", pd.to_datetime(t_ns, utc=True))
        df.insert(0, "symbol", sym)
        df["d"] = days[day_idx]
        df["slot"] = np.tile(K, len(rows))
        frames.append(df[np.isfinite(df["_price"]) & df["_has_intraday"].astype(bool)])
    out = pd.concat(frames, ignore_index=True)
    feat = RAW_FEATURES + SENTIMENT + CONTEXT
    out[feat] = out[feat].replace([np.inf, -np.inf], np.nan)
    return out.sort_values(["ts", "symbol"]).reset_index(drop=True)
