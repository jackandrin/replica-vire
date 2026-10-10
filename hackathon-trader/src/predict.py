"""Score the latest feature date with the saved models and write TigerData.predictions.
python -m src.predict"""
import joblib
import pandas as pd

from .common import ROOT, get_logger, query_df, tiger, upsert
from .features import FEATURES

log = get_logger("predict")


def main(argv=None):
    cols = ", ".join(FEATURES)
    df = query_df(f"SELECT symbol, d, {cols} FROM features WHERE d = (SELECT max(d) FROM features)")
    if df.empty:
        raise RuntimeError("no features; run src.features first")
    asof = df["d"].iloc[0]
    ridge = joblib.load(ROOT / "models/ridge.joblib")
    mlp = joblib.load(ROOT / "models/mlp.joblib")
    X = df[FEATURES].values
    mu_r, _ = ridge.predict(X)
    mu_m, sg_m = mlp.predict(X)
    out = pd.concat([
        pd.DataFrame({"symbol": df.symbol, "asof": asof, "model": "ridge", "mu": mu_r, "sigma": None}),
        pd.DataFrame({"symbol": df.symbol, "asof": asof, "model": "mlp", "mu": mu_m, "sigma": sg_m}),
    ])
    with tiger() as conn:
        upsert(conn, "predictions", out, ["symbol", "asof", "model"])
    log.info("wrote %d predictions for %s", len(out), asof)


def latest_pivot() -> pd.DataFrame:
    p = query_df("SELECT symbol, asof, model, mu, sigma FROM predictions "
                 "WHERE asof = (SELECT max(asof) FROM predictions)")
    if p.empty:
        return p
    r = p[p.model == "ridge"].set_index("symbol")["mu"].rename("mu_ridge")
    m = p[p.model == "mlp"].set_index("symbol")[["mu", "sigma"]].rename(columns={"mu": "mu_mlp", "sigma": "sigma_mlp"})
    out = pd.concat([r, m], axis=1).dropna().reset_index().rename(columns={"index": "symbol"})
    out["asof"] = p["asof"].iloc[0]
    return out


if __name__ == "__main__":
    main()
