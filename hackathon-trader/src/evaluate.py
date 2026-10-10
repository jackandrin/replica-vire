"""Metrics (IC, hit rate) and a simple non-overlapping backtest vs SPY."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .strategy import select_agreement, select_single


def ic_by_day(df: pd.DataFrame, pred: str, target: str = "y_excess_5d") -> pd.Series:
    out = {}
    for d, g in df.groupby("d"):
        if len(g) >= 5 and g[pred].nunique() > 1:
            out[d] = spearmanr(g[pred], g[target])[0]
    return pd.Series(out, dtype=float)


def pred_metrics(df: pd.DataFrame, pred: str, target: str = "y_excess_5d") -> dict:
    ic = ic_by_day(df, pred, target)
    n = len(ic)
    return {
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std()),
        "ic_tstat_naive": float(ic.mean() / ic.std() * np.sqrt(n)) if n > 1 and ic.std() > 0 else None,
        "n_days": int(n),
        "pooled_spearman": float(spearmanr(df[pred], df[target])[0]),
        "hit_rate": float((np.sign(df[pred]) == np.sign(df[target])).mean()),
        "n_rows": int(len(df)),
    }


def _stats(r: pd.Series, periods_per_year: float) -> dict:
    r = r.fillna(0.0)
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    sd = r.std()
    return {"total_return": float(eq.iloc[-1] - 1),
            "sharpe": float(r.mean() / sd * np.sqrt(periods_per_year)) if sd and sd > 0 else None,
            "max_drawdown": float(dd), "n_periods": int(len(r))}


def backtest(test: pd.DataFrame, s: dict, horizon: int = 5):
    """test needs: symbol, d, y_raw_5d, y_spy_5d, mu_ridge, mu_mlp, sigma_mlp.
    Rebalance every `horizon` trading days so holding periods do not overlap."""
    dates = sorted(test["d"].unique())[::horizon]
    cost = 2 * s["cost_bps"] / 1e4  # round trip, assuming a full turnover each period
    rows = []
    for d in dates:
        p = test[test.d == d]
        ret = {"d": d, "spy": p["y_spy_5d"].iloc[0], "ew_universe": p["y_raw_5d"].mean()}
        ret["spy_at_gross"] = ret["spy"] * s["max_gross"]
        for name, sel in (("ridge", select_single(p, "mu_ridge", s)),
                          ("mlp", select_single(p, "mu_mlp", s)),
                          ("agreement", select_agreement(p, s))):
            if sel.empty:
                ret[name], ret[name + "_n"] = 0.0, 0
                continue
            r = sel.merge(p[["symbol", "y_raw_5d"]], on="symbol")
            ret[name] = float((r["weight"] * r["y_raw_5d"]).sum() - r["weight"].sum() * cost)
            ret[name + "_n"] = int(len(r))
        rows.append(ret)
    curve = pd.DataFrame(rows).set_index("d")
    ppy = 252 / horizon
    stats = {c: _stats(curve[c], ppy) for c in ("ridge", "mlp", "agreement", "spy", "spy_at_gross", "ew_universe")}
    for c in ("ridge", "mlp", "agreement"):
        stats[c]["avg_positions"] = float(curve[c + "_n"].mean())
        stats[c]["pct_periods_beating_spy_at_gross"] = float((curve[c] > curve["spy_at_gross"]).mean())
    return curve, stats
