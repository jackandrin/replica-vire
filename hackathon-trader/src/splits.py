"""Time-based splits by trading day, with an embargo gap (no torch dependency)."""
import numpy as np


def split_days(days, test_frac: float, embargo: int):
    """Return (dev_days, test_days): test = last test_frac of unique days; `embargo` days dropped in between."""
    u = np.sort(np.unique(days))
    t0 = int(len(u) * (1 - test_frac))
    return u[: max(0, t0 - embargo)], u[t0:]


def winsor(y: np.ndarray, lo=0.01, hi=0.99):
    a, b = np.nanquantile(y, [lo, hi])
    return np.clip(y, a, b)
