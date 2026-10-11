"""The 15 per-family ridge regressors (level 1) and walk-forward out-of-fold predictions.

Each regressor sees ONE feature family (src/featurize.py: FAMILIES) and predicts the next-day (1d) return.
Keeping families separate is what lets the dashboard say *which* kind of evidence drove a signal.
Out-of-fold (OOF) predictions are the fusion network's training inputs, so the network never learns to trust
regressor outputs that were fitted on the same rows (stacking leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class Prep:
    """Clip to train 1st/99th percentiles, fill NaN with the train median, standardize. Fit on train only."""

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        self.lo, self.hi = np.nanpercentile(X, 1, axis=0), np.nanpercentile(X, 99, axis=0)
        Xc = np.clip(X, self.lo, self.hi)
        self.med = np.nan_to_num(np.nanmedian(Xc, axis=0))
        Xc = np.where(np.isnan(Xc), self.med, Xc)
        self.mu, self.sd = Xc.mean(axis=0), Xc.std(axis=0)
        self.sd[self.sd < 1e-12] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.clip(np.asarray(X, dtype=float), self.lo, self.hi)
        X = np.where(np.isnan(X), self.med, X)
        return (X - self.mu) / self.sd


def winsor_bounds(y: np.ndarray, lo=0.5, hi=99.5):
    return tuple(np.nanpercentile(y, [lo, hi]))


class FamilyRegressor:
    def __init__(self, family: str, cols: list, alpha: float):
        self.family, self.cols, self.alpha = family, list(cols), alpha

    def fit(self, df: pd.DataFrame, y: np.ndarray):
        ok = np.isfinite(y)
        self.prep = Prep().fit(df.loc[ok, self.cols].values)
        self.ybounds = winsor_bounds(y[ok])
        self.model = Ridge(alpha=self.alpha).fit(self.prep.transform(df.loc[ok, self.cols].values),
                                                 np.clip(y[ok], *self.ybounds))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.prep.transform(df[self.cols].values))

    def contributions(self, df: pd.DataFrame) -> np.ndarray:
        """Per-feature contribution to the prediction (coef x standardized value), shape (n, n_cols)."""
        return self.prep.transform(df[self.cols].values) * self.model.coef_


def day_blocks(days: np.ndarray, n_blocks: int) -> list:
    u = np.sort(np.unique(days))
    return [b for b in np.array_split(u, n_blocks) if len(b)]


def walk_forward_oof(df: pd.DataFrame, families: dict, target: str, folds: int, embargo: int, alphas: list,
                     log=None):
    """Expanding-window walk-forward over `folds`+1 contiguous day blocks. Block 0 is training-only.
    For block i the regressors are fitted on blocks < i minus an `embargo`-day gap, then predict block i.
    Alpha per family = the one with the best OOF correlation to the target.
    Returns (oof DataFrame of reg_<family> columns aligned to df.index with NaN for block 0, {family: alpha})."""
    blocks = day_blocks(df["d"].values, folds + 1)
    u = np.sort(np.unique(df["d"].values))
    y = df[target].values
    best = {}
    oof = pd.DataFrame(index=df.index, dtype=float)
    for fam, cols in families.items():
        scores = {}
        for a in alphas:
            pred = np.full(len(df), np.nan)
            for b in blocks[1:]:
                cut = u[max(0, np.searchsorted(u, b[0]) - embargo)]
                tr = (df["d"].values < cut) & np.isfinite(y)
                te = np.isin(df["d"].values, b)
                if tr.sum() < 500:
                    continue
                pred[te] = FamilyRegressor(fam, cols, a).fit(df[tr], y[tr]).predict(df[te])
            ok = np.isfinite(pred) & np.isfinite(y)
            scores[a] = (np.corrcoef(pred[ok], y[ok])[0, 1] if ok.sum() > 100 else -np.inf, pred)
        a_best = max(scores, key=lambda a: np.nan_to_num(scores[a][0], nan=-np.inf))
        best[fam] = a_best
        oof[f"reg_{fam}"] = scores[a_best][1]
        if log:
            log.info("regressor %-22s alpha=%-7g OOF corr=%+.4f", fam, a_best, scores[a_best][0])
    return oof, best


def fit_all(df: pd.DataFrame, families: dict, target: str, alphas: dict) -> dict:
    y = df[target].values
    return {fam: FamilyRegressor(fam, cols, alphas[fam]).fit(df, y) for fam, cols in families.items()}


def predict_all(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({f"reg_{f}": m.predict(df) for f, m in models.items()}, index=df.index)
