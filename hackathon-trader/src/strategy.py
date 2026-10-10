"""Decision layer shared by the backtest and live trading: agreement filter + position sizing."""
import pandas as pd

OUT_COLS = ["symbol", "weight", "mu_ridge", "mu_mlp", "sigma_mlp"]


def select_agreement(p: pd.DataFrame, s: dict) -> pd.DataFrame:
    """p: one decision date, columns symbol, mu_ridge, mu_mlp, sigma_mlp.
    Keep names where BOTH models are bullish, BOTH rank it in their top agree_top_frac,
    and the MLP's mu/sigma conviction clears min_conviction. Size by conviction, capped."""
    if p.empty:
        return pd.DataFrame(columns=OUT_COLS)
    p = p.copy()
    p["r_ridge"] = p["mu_ridge"].rank(pct=True)
    p["r_mlp"] = p["mu_mlp"].rank(pct=True)
    p["conv"] = p["mu_mlp"] / p["sigma_mlp"]
    cut = 1 - s["agree_top_frac"]
    ok = ((p.mu_ridge > 0) & (p.mu_mlp > 0) & (p.r_ridge >= cut) & (p.r_mlp >= cut)
          & (p.conv >= s["min_conviction"]))
    sel = p[ok].copy()
    if sel.empty:
        return pd.DataFrame(columns=OUT_COLS)
    sel["score"] = (sel.r_ridge + sel.r_mlp) / 2
    sel = sel.nlargest(s["top_k"], "score")
    c = sel["conv"].clip(0.2, 2.0)
    sel["weight"] = (c / c.sum() * s["max_gross"]).clip(upper=s["max_position"])
    return sel[OUT_COLS]


def select_single(p: pd.DataFrame, col: str, s: dict) -> pd.DataFrame:
    """Baseline: top_k by one model's mu (only if mu > 0), equal-weighted to max_gross."""
    sel = p[p[col] > 0].nlargest(s["top_k"], col).copy()
    if sel.empty:
        return pd.DataFrame(columns=OUT_COLS)
    sel["weight"] = min(s["max_gross"] / len(sel), s["max_position"])
    for c in ("mu_ridge", "mu_mlp", "sigma_mlp"):
        if c not in sel:
            sel[c] = float("nan")
    return sel[OUT_COLS]
