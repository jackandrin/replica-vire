"""Leakage + split checks. Run: pytest -q   (no network or database needed)."""
import numpy as np
import pandas as pd

from src.features import FEATURES, build
from src.splits import time_split


def _synthetic(n=200, syms=("AAA", "BBB", "SPY"), seed=0):
    rng = np.random.default_rng(seed)
    days = [d.date() for d in pd.bdate_range("2024-01-02", periods=n)]
    rows = []
    for s in syms:
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        for i, d in enumerate(days):
            rows.append(dict(symbol=s, d=d, open=px[i] * 0.999, high=px[i] * 1.01, low=px[i] * 0.99,
                             close=px[i], volume=1e6 * (1 + rng.random())))
    return pd.DataFrame(rows), days


def _news(days):
    t = pd.Timestamp(days[120], tz="America/New_York")
    return pd.DataFrame({
        "symbol": ["AAA"] * 4,
        "event_time": [t + pd.Timedelta(hours=10), t + pd.Timedelta(hours=17),  # before / after the close
                       t + pd.Timedelta(days=1, hours=9), t + pd.Timedelta(days=3)],
        "sentiment": [0.5, -0.5, 0.9, 0.1]})


def test_features_at_T_ignore_the_future():
    bars, days = _synthetic()
    news = _news(days)
    T = days[120]
    full, _ = build(bars, news, "SPY")
    cut, _ = build(bars[bars.d <= T], news, "SPY")
    a = full[(full.symbol == "AAA") & (full.d == T)][FEATURES].values
    b = cut[(cut.symbol == "AAA") & (cut.d == T)][FEATURES].values
    assert a.shape == b.shape == (1, len(FEATURES))
    np.testing.assert_allclose(a, b, equal_nan=True)


def test_after_close_news_counts_for_next_day():
    bars, days = _synthetic()
    F, _ = build(bars, _news(days), "SPY")
    row_T = F[(F.symbol == "AAA") & (F.d == days[120])].iloc[0]
    assert abs(row_T["sent_1d"] - 0.5) < 1e-9          # only the 10:00 article
    row_T1 = F[(F.symbol == "AAA") & (F.d == days[121])].iloc[0]
    assert abs(row_T1["sent_1d"] - (-0.5 + 0.9) / 2) < 1e-9  # 17:00 (T) + 09:00 (T+1)


def test_label_uses_next_open_and_horizon():
    bars, days = _synthetic()
    F, L = build(bars, _news(days), "SPY", horizon=5)
    a = bars[bars.symbol == "AAA"].set_index("d")
    i = 50
    expect = a["open"].iloc[i + 6] / a["open"].iloc[i + 1] - 1
    got = L[(L.symbol == "AAA") & (L.d == days[i])]["y_raw_5d"].iloc[0]
    assert abs(got - expect) < 1e-12


def test_time_split_has_gaps_and_order():
    dates = np.repeat(np.arange(1000), 20)
    fit, val, train, test = time_split(dates, 0.2, 0.15, 7)
    assert train.max() < test.min() and test.min() - train.max() > 7
    assert fit.max() < val.min() and val.min() - fit.max() > 7
