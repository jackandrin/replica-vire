"""Live inference: latest decision time -> 15 regressors -> fusion network -> direction vectors -> `signals`.
python -m src.predict [--models models/v2]

Loads a short window from Tiger Data (~45 days of 15-min bars, ~14 months of daily bars, events, recent
sentiment), builds features for the most recent completed 15-min bar only, and writes one `signals` row per
symbol with the chosen horizon, all six (mu, sigma), the 15 regressor outputs and the top attributed drivers."""
import argparse
import json
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

from . import featurize as fz
from .common import ROOT, cfg, get_logger, query_df, tiger, upsert
from .regressors import predict_all
from .sectors import CONTEXT_SYMBOLS, MARKET, VOL_PROXY, sector_etf_of
from .signals import build_signals

log = get_logger("predict")


def load_models(path):
    path = ROOT / path
    meta = json.loads((path / "meta.json").read_text())
    return meta, joblib.load(path / "regressors.joblib"), joblib.load(path / "fusion.joblib")


def decision_hhmm(bars_15m: pd.DataFrame) -> str:
    """End time of the market's latest regular-session bar, clamped to 10:00-16:00 ET."""
    m = bars_15m[bars_15m.symbol == MARKET]
    et = pd.to_datetime(m["ts"], utc=True).dt.tz_convert(fz.ET) + pd.Timedelta(minutes=15)
    last = et[et.dt.date == et.dt.date.max()]
    mins = (last.dt.hour * 60 + last.dt.minute).values
    mins = mins[(mins > fz.OPEN_MIN) & (mins <= fz.OPEN_MIN + fz.N_SLOTS * fz.SLOT_MIN)]
    end = int(np.clip(mins.max() if len(mins) else 960, 600, 960))
    return f"{end // 60:02d}:{end % 60:02d}"


def _fmt(name, v):
    if name.startswith("reg_"):
        return f"{100 * v:+.2f}%"
    if name == "sent_level":
        return f"{v:.0f}/100"
    return f"{v:+.3g}"


def _label(name):
    return f"{fz.FAMILY_LABELS[name[4:]]} regressor" if name.startswith("reg_") else fz.LABEL_TEXT.get(name, name)


def score(frames: dict, meta: dict, regs: dict, net, st: dict, symbols: list, hhmm: str | None = None) -> pd.DataFrame:
    """Pure scoring (no DB). frames: bars_15m, bars_1d, events, sentiment_events. Returns rows for `signals`."""
    hhmm = hhmm or decision_hhmm(frames["bars_15m"])
    df = fz.build(frames["bars_15m"], frames["bars_1d"], frames.get("events"), frames.get("sentiment_events"),
                  symbols, {s: sector_etf_of(s) for s in symbols}, [hhmm], MARKET, VOL_PROXY, min_day=-1)
    if df.empty:
        return df
    df = df[df.ts == df.ts.max()].reset_index(drop=True)
    df = pd.concat([df, predict_all(regs, df)], axis=1)
    X = df[meta["nn_inputs"]].values
    mu, sg = net.predict(X)
    H = meta["horizons"]
    p = df[["symbol", "ts"]].copy()
    for j, h in enumerate(H):
        p[f"mu_{h}"], p[f"sigma_{h}"] = mu[:, j], sg[:, j]
    sig = build_signals(p, st, H, meta["score_scale"], meta.get("cost_bps", 5))
    attr = net.attribute(X, sig["h_idx"].values)
    names = meta["nn_inputs"]
    out = []
    for i, r in sig.iterrows():
        drivers = [{"name": names[k], "label": _label(names[k]), "value": float(X[i, k]),
                    "attribution": float(100 * attr[i, k])} for k in np.argsort(-np.abs(attr[i]))[:5]]
        sign = 1 if r.expected_pct >= 0 else -1
        why = []
        for k in [k for k in np.argsort(-attr[i] * sign) if attr[i, k] * sign > 0][:3]:
            n = names[k]
            txt = f"{_label(n)} {_fmt(n, X[i, k])}"
            if n.startswith("reg_"):  # name the raw feature that moved this regressor most
                fam = n[4:]
                c = regs[fam].contributions(df.iloc[[i]])[0]
                col = regs[fam].cols[int(np.argmax(np.abs(c)))]
                txt += f" (mostly {fz.LABEL_TEXT.get(col, col)}: {df.loc[i, col]:+.3g})"
            why.append(txt)
        if r.direction == "flat":
            reason = "No trade: the expected move after costs is too small for your minimum conviction."
        else:
            ex = pd.Timestamp(r.exit_at).tz_convert(fz.ET).strftime("%a %I:%M %p ET") if r.exit_at else "?"
            reason = (f"{'Buy' if r.direction == 'long' else 'Short'}: expected {r.expected_pct:+.2f}% over "
                      f"{r.timeframe} (conviction {r.height:.1f}), exit around {ex}.")
        if why:
            reason += " Why: " + "; ".join(why) + "."
        out.append({"symbol": r.symbol, "asof": df.loc[i, "d"], "ts": df.loc[i, "ts"], "direction": r.direction,
                    "expected_pct": r.expected_pct, "timeframe": r.timeframe, "height": r.height,
                    "suggested_usd": r.suggested_usd, "exit_at": r.exit_at, "reason": reason,
                    "horizons": json.dumps({h: {"mu": float(100 * mu[i, j]), "sigma": float(100 * sg[i, j])}
                                            for j, h in enumerate(H)}),
                    "regressors": json.dumps({f: float(100 * df.loc[i, f"reg_{f}"]) for f in fz.FAMILIES}),
                    "drivers": json.dumps(drivers), "model_version": meta["version"]})
    return pd.DataFrame(out)


def load_live() -> dict:
    syms = list(cfg()["tickers"]) + CONTEXT_SYMBOLS
    now = pd.Timestamp.now(tz="UTC")

    def q(sql, days):
        return query_df(sql, (syms, (now - timedelta(days=days)).to_pydatetime()))
    return {
        "bars_15m": q("SELECT symbol, ts, open, high, low, close, volume, vwap FROM bars_15m "
                      "WHERE symbol = ANY(%s) AND ts >= %s", 45),
        "bars_1d": q("SELECT symbol, d, open, high, low, close, volume FROM bars_1d "
                     "WHERE symbol = ANY(%s) AND d >= %s", 430),
        "events": q("SELECT symbol, kind, event_time, available_at, value FROM events "
                    "WHERE symbol = ANY(%s) AND available_at >= %s", 400),
        "sentiment_events": q("SELECT doc_id, symbol, published_at, delta, level, novelty, materiality "
                              "FROM sentiment_events WHERE symbol = ANY(%s) AND published_at >= %s", 12),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models/v2")
    a = ap.parse_args(argv)
    from .autotrade import load_settings
    meta, regs, net = load_models(a.models)
    rows = score(load_live(), meta, regs, net, load_settings(), list(cfg()["tickers"]))
    if rows.empty:
        raise RuntimeError("no feature rows for the latest bar: run ingest_intraday first")
    with tiger() as conn:
        upsert(conn, "signals", rows, ["symbol", "asof"])
    log.info("signals at %s: %d long, %d short, %d flat", rows.ts.iloc[0], (rows.direction == "long").sum(),
             (rows.direction == "short").sum(), (rows.direction == "flat").sum())


if __name__ == "__main__":
    main()
