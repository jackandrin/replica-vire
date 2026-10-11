"""v2 features / labels / training on synthetic data. No network or database."""
import json

import numpy as np
import pandas as pd
import pytest

from src import featurize as fz
from tests.synthetic import ET, make_bundle, write_bundle

TIMES = ["10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "15:45"]


@pytest.fixture(scope="module")
def data():
    return make_bundle()


def _build(b, syms=("AAA", "BBB"), times=TIMES):
    return fz.build(b["bars_15m"], b["bars_1d"], b["events"], b["sentiment_events"], list(syms),
                    {"AAA": "XLK"}, times)


def _bar(b, sym, day, hhmm, col="close"):
    t = (pd.Timestamp(f"{day} {hhmm}", tz=ET)).tz_convert("UTC")
    g = b["bars_15m"]
    return float(g[(g.symbol == sym) & (g.ts == t)][col].iloc[0])


def test_32_inputs_and_families():
    assert len(fz.FAMILIES) == 15 and len(fz.SENTIMENT) + len(fz.CONTEXT) == 17


def test_labels_roll_forward_to_next_open(data):
    df = _build(data, ("AAA",), ["10:00", "15:00"])
    days = sorted(df.d.unique())
    d0, d1 = pd.Timestamp(days[50]).date(), pd.Timestamp(days[51]).date()
    r10 = df[(df.d == days[50]) & (df.slot == fz.slot_of_time("10:00"))].iloc[0]
    r15 = df[(df.d == days[50]) & (df.slot == fz.slot_of_time("15:00"))].iloc[0]
    p10 = _bar(data, "AAA", d0, "09:45")                  # bar ending 10:00
    assert r10["_price"] == pytest.approx(p10)
    assert r10["y_6h"] == pytest.approx(_bar(data, "AAA", d0, "15:45") / p10 - 1)      # 16:00 close, same day
    next_open = _bar(data, "AAA", d1, "09:30", "open")
    assert r10["y_12h"] == pytest.approx(next_open / p10 - 1)                         # 22:00 -> next open
    p15 = _bar(data, "AAA", d0, "14:45")
    assert r15["y_6h"] == pytest.approx(next_open / p15 - 1)                          # 21:00 -> next open
    assert r15["y_18h"] == pytest.approx(next_open / p15 - 1)                          # 09:00 next day -> open
    assert r10["y_1d"] == pytest.approx(_bar(data, "AAA", d1, "09:45") / p10 - 1)     # same slot next day


def test_friday_rolls_to_monday(data):
    df = _build(data, ("AAA",), ["15:00"])
    fri = next(d for d in sorted(df.d.unique()) if pd.Timestamp(d).dayofweek == 4)
    days = sorted(df.d.unique())
    mon = pd.Timestamp(days[days.index(fri) + 1]).date()
    r = df[df.d == fri].iloc[0]
    assert r["y_18h"] == pytest.approx(_bar(data, "AAA", mon, "09:30", "open") / r["_price"] - 1)


def test_no_lookahead(data):
    """Features at t must be identical whether or not anything after t exists."""
    full = _build(data)
    days = sorted(full.d.unique())
    cut = (pd.Timestamp(days[100]).tz_localize(None).tz_localize(ET) + pd.Timedelta(hours=12)).tz_convert("UTC")
    b = data
    trunc = {
        "bars_15m": b["bars_15m"][b["bars_15m"].ts + pd.Timedelta(minutes=15) <= cut],
        "bars_1d": b["bars_1d"][pd.to_datetime(b["bars_1d"].d) < pd.Timestamp(days[100])],  # today's daily bar not final
        "events": b["events"][b["events"].available_at <= cut],
        "sentiment_events": b["sentiment_events"][b["sentiment_events"].published_at <= cut],
    }
    part = _build(trunc)
    cols = fz.RAW_FEATURES + fz.SENTIMENT + fz.CONTEXT
    m = full.merge(part, on=["symbol", "ts"], suffixes=("", "_p"))
    m = m[m.ts <= cut]
    assert len(m) > 100 and (m.ts == cut).any()
    for c in cols:
        a, p = m[c].values.astype(float), m[c + "_p"].values.astype(float)
        assert np.allclose(a, p, equal_nan=True, rtol=1e-9, atol=1e-12), f"{c} changes when the future is removed"


def test_events_respect_available_at(data):
    df = _build(data, ("AAA",))
    ev = data["events"]
    first_analyst = ev[(ev.symbol == "AAA") & (ev.kind == "analyst")].available_at.min()
    before = df[df.ts < first_analyst]
    assert (before["analyst_score"] == 0).all()


def test_train_quick_end_to_end(tmp_path):
    torch = pytest.importorskip("torch")  # noqa: F841
    from src import train
    syms = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ", "KKK", "LLL"]
    write_bundle(tmp_path)
    train.main(["--bundle", str(tmp_path), "--out", str(tmp_path / "models"), "--results", str(tmp_path / "res"),
                "--quick", "--only", ",".join(syms), "--device", "cpu"])
    met = json.loads((tmp_path / "res/metrics.json").read_text())
    assert set(met["horizons"]) == set(fz.HORIZONS) and len(met["regressors"]) == 15
    assert met["meta"]["test_dates"][0] > met["meta"]["train_dates"][1]
    # random-walk data: no real edge should appear
    assert abs(met["horizons"]["1d"]["nn"]["ic_mean"]) < 0.15
    for f in ("regressors.joblib", "fusion.joblib", "baseline.joblib", "meta.json"):
        assert (tmp_path / "models" / f).exists()
    # live scoring with a short intraday window, like src/predict.py
    from src.predict import load_models, score
    b = make_bundle()
    cutoff = b["bars_15m"].ts.max() - pd.Timedelta(days=45)
    frames = {**b, "bars_15m": b["bars_15m"][b["bars_15m"].ts >= cutoff]}
    meta, regs, net = load_models(tmp_path / "models")
    st = {"min_usd": 50, "max_usd": 500, "min_height": 0.5, "full_height": 2.0, "direction_mode": "both",
          "max_positions": 10, "day_trade_min_height": 2.5, "sectors": []}
    rows = score(frames, meta, regs, net, st, syms, "11:00")
    assert len(rows) == len(syms) and rows.ts.nunique() == 1
    r = rows.iloc[0]
    assert set(json.loads(r.horizons)) == set(fz.HORIZONS) and len(json.loads(r.regressors)) == 15
    assert len(json.loads(r.drivers)) == 5 and r.reason
