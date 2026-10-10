"""Time-based splits (no torch dependency)."""
import numpy as np


def time_split(dates, test_frac: float, val_frac: float, gap: int):
    """Return (fit, val, train_all, test) arrays of dates. Gaps are in trading days (unique dates)."""
    u = np.sort(np.unique(dates))
    n = len(u)
    t0 = int(n * (1 - test_frac))
    train_all = u[: t0 - gap]
    test = u[t0:]
    v0 = int(len(train_all) * (1 - val_frac))
    fit = train_all[: v0 - gap]
    val = train_all[v0:]
    return fit, val, train_all, test


def winsor(y: np.ndarray, lo=0.01, hi=0.99):
    a, b = np.quantile(y, [lo, hi])
    return np.clip(y, a, b)
