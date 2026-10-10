"""Features + labels. Decision time = market close of day T (16:00 ET).

LEAKAGE RULES (enforced by tests/test_leakage.py):
  * feature row (symbol, T) uses only bars with date <= T and news published <= 16:00 ET on T
  * label = open[T+1+H] / open[T+1] - 1  (enter at next open, hold H trading days), minus the same for SPY

python -m src.features
"""
import numpy as np
import pandas as pd

from .common import cfg, get_logger, query_df, tiger, upsert

log = get_logger("features")

FEATURES = ["ret_1", "ret_5", "ret_10", "ret_21", "vol_10", "vol_21", "vol_ratio", "rsi_14", "macd_hist",
            "mkt_ret_5", "mkt_vol_21", "rel_ret_5", "sent_1d", "sent_5d", "news_cnt_5d"]
LABELS = ["y_raw_5d", "y_spy_5d", "y_excess_5d"]


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def price_features(g: pd.DataFrame) -> pd.DataFrame:
    c = g["close"]
    r = c.pct_change()
    out = pd.DataFrame(index=g.index)
    out["ret_1"] = r
    for n in (5, 10, 21):
        out[f"ret_{n}"] = c.pct_change(n)
    out["vol_10"] = r.rolling(10).std()
    out["vol_21"] = r.rolling(21).std()
    out["vol_ratio"] = g["volume"] / g["volume"].rolling(20).mean()
    out["rsi_14"] = _rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    return out


def assign_news_days(news: pd.DataFrame, days: list) -> pd.DataFrame:
    """Map each article to the first trading day whose 16:00 ET close is >= publish time."""
    if news.empty:
        return news.assign(d=[])
    t = pd.to_datetime(news["event_time"], utc=True).dt.tz_convert("America/New_York")
    nd = t.dt.normalize().dt.tz_localize(None) + pd.to_timedelta((t.dt.hour >= 16).astype(int), unit="D")
    days_ts = pd.to_datetime(pd.Series(days)).values
    pos = np.searchsorted(days_ts, nd.values)  # first trading day >= nd
    ok = pos < len(days_ts)  # later than the last bar -> belongs to the future, dropped
    out = news.loc[ok].copy()
    out["d"] = [days[p] for p in pos[ok]]
    return out


def build(bars: pd.DataFrame, news: pd.DataFrame, benchmark: str, horizon: int = 5):
    """bars: symbol,d,open,high,low,close,volume.  news: symbol,event_time,sentiment."""
    bars = bars.sort_values(["symbol", "d"])
    spy = bars[bars.symbol == benchmark].set_index("d")
    days = list(spy.index)
    if len(days) < 60:
        raise RuntimeError("benchmark has too few bars; run ingest_bars first")

    spy_ret5 = spy["close"].pct_change(5)
    spy_vol21 = spy["close"].pct_change().rolling(21).std()
    spy_entry = spy["open"].shift(-1)
    spy_y = spy["open"].shift(-1 - horizon) / spy_entry - 1

    nws = assign_news_days(news.dropna(subset=["sentiment"]), days)
    daily = nws.groupby(["symbol", "d"])["sentiment"].agg(["sum", "count"]) if len(nws) else None

    feats, labs = [], []
    for sym, g in bars[bars.symbol != benchmark].groupby("symbol"):
        g = g.set_index("d").reindex(days)  # align to the benchmark trading calendar
        if g["close"].notna().mean() < 0.9:
            log.warning("%s: <90%% bar coverage, skipping", sym)
            continue
        f = price_features(g)
        f["mkt_ret_5"] = spy_ret5
        f["mkt_vol_21"] = spy_vol21
        f["rel_ret_5"] = f["ret_5"] - f["mkt_ret_5"]
        if daily is not None and sym in daily.index.get_level_values(0):
            s = daily.loc[sym].reindex(days).fillna(0.0)
        else:
            s = pd.DataFrame({"sum": 0.0, "count": 0.0}, index=days)
        f["sent_1d"] = s["sum"] / s["count"].clip(lower=1)
        f["sent_5d"] = s["sum"].rolling(5, min_periods=1).sum() / s["count"].rolling(5, min_periods=1).sum().clip(lower=1)
        f["news_cnt_5d"] = s["count"].rolling(5, min_periods=1).sum()
        f.insert(0, "symbol", sym)
        f.index.name = "d"
        feats.append(f.reset_index())

        y_raw = g["open"].shift(-1 - horizon) / g["open"].shift(-1) - 1
        lab = pd.DataFrame({"symbol": sym, "y_raw_5d": y_raw, "y_spy_5d": spy_y})
        lab["y_excess_5d"] = lab["y_raw_5d"] - lab["y_spy_5d"]
        lab.index.name = "d"
        labs.append(lab.reset_index())

    F = pd.concat(feats, ignore_index=True).dropna(subset=FEATURES)
    L = pd.concat(labs, ignore_index=True).dropna(subset=LABELS)
    return F[["symbol", "d"] + FEATURES], L[["symbol", "d"] + LABELS]


def load_inputs():
    bars = query_df("SELECT symbol, (ts AT TIME ZONE 'America/New_York')::date AS d, open, high, low, close, volume "
                    "FROM bars ORDER BY symbol, ts")
    news = query_df("SELECT symbol, event_time, sentiment FROM news WHERE sentiment IS NOT NULL")
    return bars, news


def main(argv=None):
    c = cfg()
    bars, news = load_inputs()
    F, L = build(bars, news, c["benchmark"], c["label"]["horizon_days"])
    with tiger() as conn:
        nf = upsert(conn, "features", F, ["symbol", "d"])
        nl = upsert(conn, "labels", L, ["symbol", "d"])
    log.info("features=%d labels=%d (latest feature date %s)", nf, nl, F["d"].max())


if __name__ == "__main__":
    main()
