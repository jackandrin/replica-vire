"""Train ridge + MLP on a strict time split, evaluate on the held-out last 20%.
python -m src.train
Outputs: models/{ridge,mlp}.joblib, models/meta.json, results/{metrics.json,test_predictions.csv,backtest_curve.csv}"""
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from .common import ROOT, cfg, get_logger, query_df
from .evaluate import backtest, pred_metrics
from .features import FEATURES
from .models import MLPModel, RidgeModel
from .splits import time_split, winsor

log = get_logger("train")


def load_dataset() -> pd.DataFrame:
    cols = ", ".join(f"f.{c}" for c in FEATURES)
    return query_df(f"SELECT f.symbol, f.d, {cols}, l.y_raw_5d, l.y_spy_5d, l.y_excess_5d "
                    "FROM features f JOIN labels l USING (symbol, d) ORDER BY f.d, f.symbol")


def main(argv=None):
    c = cfg()
    df = load_dataset()
    if len(df) < 500:
        raise RuntimeError(f"only {len(df)} labelled rows; run ingestion + features first")
    sp = c["split"]
    fit_d, val_d, train_d, test_d = time_split(df["d"].values, sp["test_frac"], sp["val_frac"], sp["gap_days"])
    log.info("fit %s..%s | val %s..%s | gap | test %s..%s", fit_d[0], fit_d[-1], val_d[0], val_d[-1], test_d[0], test_d[-1])
    assert train_d.max() < test_d.min(), "train must end strictly before test"

    sel = lambda ds: df[df["d"].isin(ds)]  # noqa: E731
    fit, val, train, test = sel(fit_d), sel(val_d), sel(train_d), sel(test_d).copy()
    y = "y_excess_5d"
    yfit, yval, ytrain = winsor(fit[y].values), winsor(val[y].values), winsor(train[y].values)

    ridge = RidgeModel(c["ridge"]["alphas"]).fit(fit[FEATURES].values, yfit, val[FEATURES].values, yval)
    ridge.refit(train[FEATURES].values, ytrain)
    mlp = MLPModel(**c["mlp"], use_gpu=os.environ.get("USE_GPU") == "1").fit(fit[FEATURES].values, yfit, val[FEATURES].values, yval)
    log.info("ridge alpha=%s | mlp best_epoch=%d device=%s", ridge.alpha, mlp.best_epoch, mlp.device_used)

    test["mu_ridge"], _ = ridge.predict(test[FEATURES].values)
    test["mu_mlp"], test["sigma_mlp"] = mlp.predict(test[FEATURES].values)
    test["mu_avg"] = (test["mu_ridge"] / test["mu_ridge"].std() + test["mu_mlp"] / test["mu_mlp"].std()) / 2

    metrics = {m: pred_metrics(test, m) for m in ("mu_ridge", "mu_mlp", "mu_avg")}
    curve, bt = backtest(test, c["strategy"], c["label"]["horizon_days"])
    metrics["backtest"] = bt
    metrics["meta"] = {"train_dates": [str(fit_d[0]), str(train_d[-1])], "test_dates": [str(test_d[0]), str(test_d[-1])],
                       "n_train_rows": int(len(train)), "n_test_rows": int(len(test)), "ridge_alpha": ridge.alpha,
                       "mlp_best_epoch": mlp.best_epoch, "trained_at": datetime.now(timezone.utc).isoformat(),
                       "caveat": "Overlapping 5-day labels inflate naive t-stats; test set is a single regime."}

    joblib.dump(ridge, ROOT / "models/ridge.joblib")
    joblib.dump(mlp, ROOT / "models/mlp.joblib")
    (ROOT / "models/meta.json").write_text(json.dumps({"features": FEATURES, "trained_through": str(train_d[-1])}))
    (ROOT / "results/metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    test.to_csv(ROOT / "results/test_predictions.csv", index=False)
    curve.to_csv(ROOT / "results/backtest_curve.csv")

    for m in ("mu_ridge", "mu_mlp", "mu_avg"):
        k = metrics[m]
        log.info("%-9s IC=%+.4f (naive t=%s)  hit=%.3f  n_days=%d", m, k["ic_mean"],
                 f'{k["ic_tstat_naive"]:.2f}' if k["ic_tstat_naive"] is not None else "n/a", k["hit_rate"], k["n_days"])
    for k, v in bt.items():
        log.info("backtest %-13s ret=%+.3f sharpe=%s maxdd=%.3f", k, v["total_return"],
                 f'{v["sharpe"]:.2f}' if v["sharpe"] is not None else "n/a", v["max_drawdown"])


if __name__ == "__main__":
    main()
