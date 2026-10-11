"""Metrics (cross-sectional IC, hit rate, sigma calibration) and a simple long-short backtest (v2)."""
import numpy as np
import pandas as pd


def ic_by_ts(df: pd.DataFrame, pred: str, target: str) -> pd.Series:
    """Spearman rank correlation across symbols at each decision timestamp."""
    d = df[["ts", pred, target]].dropna()
    d = d[d.groupby("ts")[pred].transform("size") >= 5]
    if d.empty:
        return pd.Series(dtype=float)
    r = d.groupby("ts")[[pred, target]].rank()
    r["ts"] = d["ts"].values
    return r.groupby("ts").apply(lambda g: g[pred].corr(g[target]) if g[pred].nunique() > 1 else np.nan).dropna()


def pred_metrics(df: pd.DataFrame, pred: str, target: str, sigma: str | None = None) -> dict:
    ok = df[[pred, target]].notna().all(axis=1)
    d = df[ok]
    ic = ic_by_ts(d, pred, target)
    out = {"ic_mean": float(ic.mean()) if len(ic) else None, "ic_std": float(ic.std()) if len(ic) > 1 else None,
           "hit_rate": float((np.sign(d[pred]) == np.sign(d[target])).mean()) if len(d) else None,
           "coverage_1sigma": None, "n_rows": int(len(d)), "n_days": int(d["d"].nunique()) if len(d) else 0}
    if sigma and len(d):
        out["coverage_1sigma"] = float(((d[target] - d[pred]).abs() <= d[sigma]).mean())  # ~0.68 if calibrated
    return out


def long_short(df: pd.DataFrame, score: str, target: str, k: int = 5, cost_bps: float = 5) -> pd.Series:
    """Each decision timestamp: long the top-k, short the bottom-k by `score`, hold for the target's horizon.
    Cost = one round trip on both legs. Pass one decision slot per day with a 1d target -> non-overlapping."""
    rets = {}
    for ts, g in df.dropna(subset=[score, target]).groupby("ts"):
        if len(g) < 2 * k:
            continue
        g = g.sort_values(score)
        rets[ts] = g[target].iloc[-k:].mean() / 2 - g[target].iloc[:k].mean() / 2 - 2 * cost_bps / 1e4
    return pd.Series(rets, dtype=float).sort_index()


def stats(r: pd.Series, periods_per_year: float = 252) -> dict:
    if r.empty:
        return {"total_return": None, "sharpe": None, "max_drawdown": None, "n_periods": 0}
    eq = (1 + r.fillna(0)).cumprod()
    sd = r.std()
    return {"total_return": float(eq.iloc[-1] - 1),
            "sharpe": float(r.mean() / sd * np.sqrt(periods_per_year)) if sd and sd > 0 else None,
            "max_drawdown": float((eq / eq.cummax() - 1).min()), "n_periods": int(len(r))}
