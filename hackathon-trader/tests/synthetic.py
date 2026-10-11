"""Synthetic market data in the bundle format (docs/V2_CONTRACT.md) for offline tests."""
import numpy as np
import pandas as pd

ET = "America/New_York"


def make_bundle(symbols=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ", "KKK", "LLL"),
                context=("SPY", "VIXY", "XLK"), n_days=420, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=n_days)
    b15, b1d = [], []
    for s in list(symbols) + list(context):
        px = 100.0
        for d in days:
            px *= np.exp(rng.normal(0, 0.006))           # overnight
            o = px
            closes = px * np.exp(np.cumsum(rng.normal(0, 0.002, 26)))
            opens = np.concatenate([[o], closes[:-1]])
            vol = rng.integers(1_000, 5_000, 26).astype(float)
            start = pd.Timestamp(d.date(), tz=ET) + pd.Timedelta(minutes=570)
            ts = [start + pd.Timedelta(minutes=15 * k) for k in range(26)]
            ts.append(pd.Timestamp(d.date(), tz=ET) + pd.Timedelta(hours=8))   # a pre-market bar (must be ignored)
            hi = np.maximum(opens, closes) * 1.001
            lo = np.minimum(opens, closes) * 0.999
            for k in range(27):
                j = min(k, 25)
                b15.append((s, ts[k].tz_convert("UTC"), opens[j], hi[j], lo[j], closes[j], vol[j],
                            (opens[j] + closes[j]) / 2, 10.0))
            b1d.append((s, d.date(), o, hi.max(), lo.min(), closes[-1], vol.sum(), closes.mean()))
            px = closes[-1]
    bars_15m = pd.DataFrame(b15, columns=["symbol", "ts", "open", "high", "low", "close", "volume", "vwap",
                                          "trade_count"])
    bars_1d = pd.DataFrame(b1d, columns=["symbol", "d", "open", "high", "low", "close", "volume", "vwap"])
    ev, se = [], []
    for s in symbols:
        for i in range(30, n_days, 63):   # quarterly earnings after the close
            t = pd.Timestamp(days[i].date(), tz=ET) + pd.Timedelta(hours=16, minutes=5)
            ev.append((s, "earnings", t, t, np.nan, f"e{i}", "{}"))
        for i in range(n_days - 1):       # daily short volume, known next morning
            t = pd.Timestamp(days[i].date(), tz=ET) + pd.Timedelta(hours=16)
            ev.append((s, "short_volume", t, t + pd.Timedelta(hours=16), rng.uniform(0.3, 0.6), f"s{i}", "{}"))
        for i in range(10, n_days, 21):
            t = pd.Timestamp(days[i].date(), tz=ET)
            ev.append((s, "analyst", t, t + pd.Timedelta(days=31), rng.uniform(-1, 2), f"a{i}", "{}"))
            ev.append((s, "insider", t + pd.Timedelta(hours=16), t + pd.Timedelta(days=2),
                       rng.normal(0, 1e5), f"i{i}", "{}"))
        for i in range(n_days):
            if rng.random() < 0.5:
                t = pd.Timestamp(days[i].date(), tz=ET) + pd.Timedelta(hours=int(rng.integers(7, 20)))
                se.append((len(se), s, t, "stable", rng.uniform(-0.5, 0.5), rng.uniform(20, 80),
                           rng.uniform(0, 1), rng.uniform(0, 1), "days", "[]", "", "fake", "fake", "v1"))
    events = pd.DataFrame(ev, columns=["symbol", "kind", "event_time", "available_at", "value", "key", "payload"])
    for c in ("event_time", "available_at"):
        events[c] = pd.to_datetime(events[c], utc=True)
    sent = pd.DataFrame(se, columns=["doc_id", "symbol", "published_at", "direction", "delta", "level", "novelty",
                                     "materiality", "horizon", "drivers", "summary", "backend", "model",
                                     "prompt_version"])
    sent["published_at"] = pd.to_datetime(sent["published_at"], utc=True)
    return {"bars_15m": bars_15m, "bars_1d": bars_1d, "events": events, "sentiment_events": sent}


def write_bundle(tmp_path, **kw):
    b = make_bundle(**kw)
    for k, v in b.items():
        v.to_parquet(tmp_path / f"{k}.parquet", index=False)
    return b
