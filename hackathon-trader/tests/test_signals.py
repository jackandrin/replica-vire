"""Horizon choice, direction vector, sizing, direction filters, day-trade rule and actions (no DB / Alpaca)."""
from datetime import datetime

import pandas as pd

from src.signals import ET, build_signals, exit_time, plan_actions, size_usd

H = ["6h", "12h", "18h", "1d", "3d", "5d"]
SCALE = {h: 1.0 for h in H}  # height = cost-adjusted mu / sigma
ST = {"min_usd": 50, "max_usd": 500, "min_height": 0.5, "full_height": 2.0, "direction_mode": "both",
      "max_positions": 10, "day_trade_min_height": 2.5, "sectors": []}
T10 = datetime(2026, 10, 6, 10, 0, tzinfo=ET)  # a Tuesday


def frame(rows, t=T10):
    """rows: symbol -> {h: (mu, sigma)}; unspecified horizons get a flat forecast."""
    out = []
    for s, hs in rows.items():
        r = {"symbol": s, "ts": t}
        for h in H:
            mu, sg = hs.get(h, (0.0, 0.02))
            r[f"mu_{h}"], r[f"sigma_{h}"] = mu, sg
        out.append(r)
    return pd.DataFrame(out)


def test_exit_times():
    assert exit_time(T10, "6h") == datetime(2026, 10, 6, 16, 0, tzinfo=ET)          # same day close
    assert exit_time(T10, "12h") == datetime(2026, 10, 7, 9, 30, tzinfo=ET)         # rolls to next open
    fri = datetime(2026, 10, 9, 15, 0, tzinfo=ET)
    assert exit_time(fri, "18h") == datetime(2026, 10, 12, 9, 30, tzinfo=ET)        # Friday -> Monday open
    assert exit_time(fri, "1d") == datetime(2026, 10, 12, 15, 0, tzinfo=ET)


def test_size_scales_between_min_and_max():
    assert size_usd(0.4, ST) == 0 and size_usd(0.5, ST) == 50
    assert size_usd(1.25, ST) == 275 and size_usd(9.0, ST) == 500


def test_picks_strongest_horizon_and_direction():
    s = build_signals(frame({"UP": {"3d": (0.03, 0.03)}, "DOWN": {"1d": (-0.02, 0.02)}}), ST, H, SCALE).set_index("symbol")
    assert s.loc["UP", "direction"] == "long" and s.loc["UP", "timeframe"] == "3d"
    assert s.loc["DOWN", "direction"] == "short" and s.loc["DOWN", "timeframe"] == "1d"
    assert s.loc["UP", "suggested_usd"] > 0 and s.loc["UP", "expected_pct"] == 3.0


def test_cost_kills_tiny_moves():
    s = build_signals(frame({"TINY": {"1d": (0.0005, 0.001)}}), ST, H, SCALE)
    assert s.iloc[0].direction == "flat"


def test_day_trade_only_on_spike():
    mild = build_signals(frame({"X": {"6h": (0.01, 0.02), "1d": (0.006, 0.02)}}), ST, H, SCALE).iloc[0]
    assert mild.timeframe != "6h"   # 6h from 10:00 exits today -> blocked below the spike threshold
    spike = build_signals(frame({"X": {"6h": (0.08, 0.02)}}), ST, H, SCALE).iloc[0]
    assert spike.timeframe == "6h" and spike.height >= ST["day_trade_min_height"]


def test_direction_filters():
    f = frame({"UP": {"1d": (0.03, 0.02)}, "DOWN": {"1d": (-0.03, 0.02)}})
    lo = build_signals(f, {**ST, "direction_mode": "long_only"}, H, SCALE).set_index("symbol")
    so = build_signals(f, {**ST, "direction_mode": "short_only"}, H, SCALE).set_index("symbol")
    assert lo.loc["DOWN", "direction"] == "flat" and lo.loc["UP", "direction"] == "long"
    assert so.loc["UP", "direction"] == "flat" and so.loc["DOWN", "direction"] == "short"


def test_actions_open_close_exit_and_sector_filter():
    s = build_signals(frame({"UP": {"1d": (0.03, 0.02)}, "DOWN": {"1d": (-0.03, 0.02)}}), ST, H, SCALE)
    acts = {a["symbol"]: a["action"] for a in plan_actions(s, {}, ST, set(), {})}
    assert acts == {"UP": "BUY", "DOWN": "SHORT"}
    acts = {a["symbol"]: a["action"] for a in plan_actions(s, {"UP": -3, "DOWN": 2}, ST, set(), {})}
    assert acts == {"UP": "COVER", "DOWN": "SELL"}
    only = plan_actions(s, {}, {**ST, "sectors": ["Energy"]}, set(), {"UP": "Energy", "DOWN": "Utilities"})
    assert [a["symbol"] for a in only] == ["UP"]
    due = plan_actions(s, {"OLD": 5}, ST, set(), {}, due_exits={"OLD"})
    assert {"symbol": "OLD", "action": "SELL"}.items() <= due[0].items() and due[0]["why"] == "horizon"


def test_flip_close_blocked_same_day_unless_spike():
    s = build_signals(frame({"DOWN": {"1d": (-0.03, 0.02)}}), ST, H, SCALE)
    assert plan_actions(s, {"DOWN": 2}, ST, {"DOWN"}, {}) == []
    spike = plan_actions(s, {"DOWN": 2}, {**ST, "day_trade_min_height": 0.1}, {"DOWN"}, {})
    assert spike[0]["action"] == "SELL"


def test_max_positions_caps_new_opens():
    s = build_signals(frame({"UP": {"1d": (0.03, 0.02)}}), ST, H, SCALE)
    assert plan_actions(s, {"X": 1}, {**ST, "max_positions": 1}, set(), {}) == []
