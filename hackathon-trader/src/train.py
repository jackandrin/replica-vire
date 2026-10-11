"""Train the v2 stack from the parquet bundle: 15 family regressors -> fusion network (6 horizons) + baseline.
Runs offline (no database, no network), so it can run on an HPC compute node.

    python -m src.train [--bundle data/bundle] [--out models/v2] [--device cuda] [--quick] [--only AAPL,MSFT]

Steps
  1. featurize.build: decision-time features + 6-horizon labels (cached to <bundle>/features.parquet)
  2. split by trading day: dev | embargo | held-out test (last model.test_frac of days)
  3. walk-forward out-of-fold predictions of the 15 regressors on dev (stacking without leakage)
  4. fusion network on [15 OOF regressor outputs + 4 sentiment + 13 context]; early stopping on a later
     validation slice of dev
  5. baseline: one ridge per horizon on all raw features (the network must beat this to be worth it)
  6. held-out test: per-horizon IC / hit rate / 1-sigma coverage, regressor ICs, redundancy, long-short backtest
  7. production regressors refit on all labelled rows; artifacts -> models/v2, results/metrics.json
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import featurize as fz
from .common import ROOT, cfg, get_logger
from .evaluate import long_short, pred_metrics, stats
from .fusion import FusionModel
from .regressors import Prep, fit_all, predict_all, walk_forward_oof, winsor_bounds
from .sectors import MARKET, VOL_PROXY, sector_etf_of
from .splits import split_days

log = get_logger("train")
REG_COLS = [f"reg_{f}" for f in fz.FAMILIES]
NN_INPUTS = REG_COLS + fz.SENTIMENT + fz.CONTEXT


def load_bundle(bundle: Path) -> dict:
    need = ["bars_15m", "bars_1d"]
    out = {}
    for name in need + ["events", "sentiment_events"]:
        p = bundle / f"{name}.parquet"
        if p.exists():
            out[name] = pd.read_parquet(p)
        elif name in need:
            raise FileNotFoundError(f"{p} missing: run python -m src.export_dataset first")
        else:
            log.warning("%s missing: its features fall back to neutral values", p.name)
            out[name] = None
    for k, v in out.items():
        log.info("loaded %-17s %s rows", k, "-" if v is None else f"{len(v):,}")
    return out


def build_features(bundle: Path, symbols: list, rebuild: bool) -> pd.DataFrame:
    cache = bundle / "features.parquet"
    inputs = [p for p in bundle.glob("*.parquet") if p.name != cache.name]
    if cache.exists() and any(p.stat().st_mtime > cache.stat().st_mtime for p in inputs):
        log.info("bundle changed since %s was built: rebuilding features", cache.name)
        rebuild = True
    if cache.exists() and not rebuild:
        df = pd.read_parquet(cache)
        if set(symbols) <= set(df.symbol.unique()):
            log.info("using cached %s (%s rows); pass --rebuild-features after new data", cache.name, f"{len(df):,}")
            return df[df.symbol.isin(symbols)].reset_index(drop=True)
    b = load_bundle(bundle)
    t0 = time.time()
    df = fz.build(b["bars_15m"], b["bars_1d"], b["events"], b["sentiment_events"], symbols,
                  {s: sector_etf_of(s) for s in symbols}, cfg()["model"]["decision_times"], MARKET, VOL_PROXY, log=log)
    log.info("built %s feature rows for %d symbols in %.0fs", f"{len(df):,}", df.symbol.nunique(), time.time() - t0)
    df.to_parquet(cache, index=False)
    return df


def ridge_baseline(train: pd.DataFrame, val: pd.DataFrame, cols: list, target: str, alphas: list):
    """Pick alpha on validation, refit on train+val. Returns (prep, model, ybounds)."""
    best = None
    tr, va = train.dropna(subset=[target]), val.dropna(subset=[target])
    prep = Prep().fit(tr[cols].values)
    yb = winsor_bounds(tr[target].values)
    for a in alphas:
        m = Ridge(alpha=a).fit(prep.transform(tr[cols].values), np.clip(tr[target].values, *yb))
        c = np.corrcoef(m.predict(prep.transform(va[cols].values)), va[target].values)[0, 1]
        if best is None or np.nan_to_num(c, nan=-1) > best[0]:
            best = (np.nan_to_num(c, nan=-1), a)
    both = pd.concat([tr, va])
    prep = Prep().fit(both[cols].values)
    yb = winsor_bounds(both[target].values)
    return prep, Ridge(alpha=best[1]).fit(prep.transform(both[cols].values), np.clip(both[target].values, *yb)), yb


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=cfg()["data"]["bundle_dir"])
    ap.add_argument("--out", default="models/v2")
    ap.add_argument("--results", default="results")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--only", help="comma-separated tickers (debugging)")
    ap.add_argument("--quick", action="store_true", help="few epochs/folds: smoke test the pipeline")
    ap.add_argument("--rebuild-features", action="store_true")
    a = ap.parse_args(argv)
    c, m = cfg(), cfg()["model"]
    bundle, out, res = (ROOT / a.bundle), (ROOT / a.out), (ROOT / a.results)
    out.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    symbols = a.only.split(",") if a.only else list(c["tickers"])
    folds, epochs = (2, 5) if a.quick else (m["folds"], m["nn"]["epochs"])

    df = build_features(bundle, symbols, a.rebuild_features)
    if df.empty:
        raise RuntimeError("no feature rows: check the bundle covers > 1 year of bars")
    labelled = df.dropna(subset=["y_1d"]).reset_index(drop=True)
    dev_d, test_d = split_days(labelled["d"].values, m["test_frac"], m["embargo_days"])
    dev = labelled[labelled.d.isin(dev_d)].reset_index(drop=True)
    test = labelled[labelled.d.isin(test_d)].reset_index(drop=True)
    log.info("dev %s..%s (%s rows) | embargo %d days | test %s..%s (%s rows)", dev_d[0], dev_d[-1], f"{len(dev):,}",
             m["embargo_days"], test_d[0], test_d[-1], f"{len(test):,}")
    assert dev["d"].max() < test["d"].min(), "dev must end strictly before test"

    # ---- level 1: out-of-fold regressor predictions on dev, plain predictions on test
    oof, alphas = walk_forward_oof(dev, fz.FAMILIES, "y_1d", folds, m["embargo_days"], m["ridge_alphas"], log)
    dev = pd.concat([dev, oof], axis=1)
    regs_dev = fit_all(dev, fz.FAMILIES, "y_1d", alphas)
    test = pd.concat([test, predict_all(regs_dev, test)], axis=1)
    nn_rows = dev.dropna(subset=REG_COLS).reset_index(drop=True)
    corr = nn_rows[REG_COLS].corr()
    redundant = [[a_[4:], b_[4:], round(float(corr.loc[a_, b_]), 3)] for i, a_ in enumerate(REG_COLS)
                 for b_ in REG_COLS[i + 1:] if abs(corr.loc[a_, b_]) > 0.8]
    if redundant:
        log.warning("highly correlated regressors (consider merging): %s", redundant)

    # ---- level 2: fusion network
    u = np.sort(nn_rows["d"].unique())
    v0 = int(len(u) * (1 - m["val_frac"]))
    tr = nn_rows[nn_rows.d < u[max(0, v0 - m["embargo_days"])]]
    va = nn_rows[nn_rows.d >= u[v0]]
    log.info("fusion net: train %s rows, val %s rows, %d inputs -> %d horizons",
             f"{len(tr):,}", f"{len(va):,}", len(NN_INPUTS), len(fz.HORIZONS))
    nnc = m["nn"]
    net = FusionModel(NN_INPUTS, fz.HORIZONS, hidden=nnc["hidden"], dropout=nnc["dropout"], lr=nnc["lr"],
                      weight_decay=nnc["weight_decay"], batch_size=nnc["batch_size"], epochs=epochs,
                      patience=nnc["patience"], seed=nnc["seed"], device=a.device)
    net.fit(tr[NN_INPUTS].values, tr[fz.LABELS].values, va[NN_INPUTS].values, va[fz.LABELS].values, log=log)
    log.info("fusion net best epoch %d, val NLL %.4f", net.best_epoch, net.val_nll)
    # height scale per horizon (src/signals.py): std of the cost-adjusted mu/sigma score on validation
    vmu, vsg = net.predict(va[NN_INPUTS].values)
    cost = 2 * m["cost_bps"] / 1e4
    vscore = np.sign(vmu) * np.clip(np.abs(vmu) - cost, 0, None) / vsg
    score_scale = {h: float(np.nanstd(vscore[:, j])) or 1.0 for j, h in enumerate(fz.HORIZONS)}
    mu, sg = net.predict(test[NN_INPUTS].values)
    for j, h in enumerate(fz.HORIZONS):
        test[f"nn_mu_{h}"], test[f"nn_sigma_{h}"] = mu[:, j], sg[:, j]

    # ---- baseline: one ridge per horizon on all raw features
    base_cols = fz.RAW_FEATURES + fz.SENTIMENT + fz.CONTEXT
    btr, bva = dev[dev.d < u[max(0, v0 - m["embargo_days"])]], dev[dev.d >= u[v0]]
    baseline = {}
    for h in fz.HORIZONS:
        baseline[h] = ridge_baseline(btr, bva, base_cols, f"y_{h}", m["ridge_alphas"])
        prep, model, _ = baseline[h]
        test[f"base_{h}"] = model.predict(prep.transform(test[base_cols].values))

    # ---- held-out evaluation
    metrics = {"horizons": {}, "regressors": {}, "redundant_pairs": redundant}
    for h in fz.HORIZONS:
        metrics["horizons"][h] = {"nn": pred_metrics(test, f"nn_mu_{h}", f"y_{h}", f"nn_sigma_{h}"),
                                  "baseline": pred_metrics(test, f"base_{h}", f"y_{h}")}
        k = metrics["horizons"][h]
        log.info("%-4s NN IC=%s hit=%.3f cov1σ=%.2f | baseline IC=%s hit=%.3f", h, _f(k["nn"]["ic_mean"]),
                 k["nn"]["hit_rate"] or 0, k["nn"]["coverage_1sigma"] or 0, _f(k["baseline"]["ic_mean"]),
                 k["baseline"]["hit_rate"] or 0)
    for f in fz.FAMILIES:
        r = pred_metrics(test, f"reg_{f}", "y_1d")
        metrics["regressors"][f] = {"ic_mean": r["ic_mean"], "hit_rate": r["hit_rate"]}

    first = fz.slot_of_time(m["decision_times"][0])  # one decision per day -> non-overlapping 1d holds
    day = test[test.slot == first].copy()
    day["nn_score"] = day["nn_mu_1d"] / day["nn_sigma_1d"]
    r_nn = long_short(day, "nn_score", "y_1d", cost_bps=m["cost_bps"])
    r_base = long_short(day, "base_1d", "y_1d", cost_bps=m["cost_bps"])
    spy = _spy_returns(bundle, test_d, m["decision_times"][0])
    curve = pd.DataFrame({"nn": r_nn, "baseline": r_base}).join(spy.rename("spy"), how="left").fillna(0.0)
    metrics["backtest"] = {k: stats(curve[k]) for k in ("nn", "baseline", "spy")}
    cum = (1 + curve).cumprod() - 1
    metrics["curve"] = [{"date": str(ts.date()), **{k: float(v) for k, v in row.items()}} for ts, row in cum.iterrows()]
    for k, v in metrics["backtest"].items():
        log.info("backtest %-8s ret=%s sharpe=%s maxdd=%s", k, _f(v["total_return"]), _f(v["sharpe"]), _f(v["max_drawdown"]))

    # ---- production artifacts: regressors refit on every labelled row; the network keeps its OOF training
    regs_all = fit_all(labelled, fz.FAMILIES, "y_1d", alphas)
    trained_at = datetime.now(timezone.utc).isoformat()
    joblib.dump(regs_all, out / "regressors.joblib")
    joblib.dump(net, out / "fusion.joblib")
    joblib.dump({h: (p_, m_) for h, (p_, m_, _) in baseline.items()}, out / "baseline.joblib")
    meta = {"version": trained_at, "families": fz.FAMILIES, "nn_inputs": NN_INPUTS, "horizons": fz.HORIZONS,
            "baseline_cols": base_cols, "score_scale": score_scale, "cost_bps": m["cost_bps"], "decision_times": m["decision_times"], "alphas": alphas,
            "trained_through": str(labelled["d"].max()), "symbols": sorted(df.symbol.unique().tolist())}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    metrics["meta"] = {"train_dates": [str(dev_d[0]), str(dev_d[-1])], "test_dates": [str(test_d[0]), str(test_d[-1])],
                       "n_train_rows": int(len(dev)), "n_test_rows": int(len(test)), "trained_at": trained_at,
                       "folds": folds, "nn_best_epoch": int(net.best_epoch), "quick": a.quick,
                       "caveat": ("Single held-out period; intraday rows overlap across horizons; LLM sentiment on "
                                  "historical articles may contain look-ahead from the model's training data.")}
    (res / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (res / "nn_history.json").write_text(json.dumps(net.history))
    test.to_parquet(res / "test_predictions.parquet", index=False)
    log.info("saved %s and %s", out, res / "metrics.json")


def _f(v):
    return "n/a" if v is None else f"{v:+.4f}"


def _spy_returns(bundle: Path, test_days, decision_time: str) -> pd.Series:
    """SPY's 1-day return from the same decision time (benchmark line). Reads only SPY's rows."""
    rd = lambda n: pd.read_parquet(bundle / f"{n}.parquet", filters=[("symbol", "==", MARKET)])  # noqa: E731
    spy = fz.build(rd("bars_15m"), rd("bars_1d"), None, None, [MARKET], {}, [decision_time], MARKET, VOL_PROXY)
    return spy[spy.d.isin(test_days)].set_index("ts")["y_1d"]


if __name__ == "__main__":
    main()
