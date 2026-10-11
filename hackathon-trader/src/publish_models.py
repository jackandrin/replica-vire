"""Publish a finished training run to Tiger Data so the website and the live bot use it.
python -m src.publish_models [--models models/v2] [--results results] [--no-sentiment] [--list]

Run it on any machine that can reach Tiger Data after src.train finishes (e.g. a Theia LOGIN node; compute nodes
usually have no internet). It uploads, in one transaction:
  * results/metrics.json + models/v2/meta.json -> model_runs   (the dashboard's Performance page reads this)
  * models/v2/*.joblib + meta.json             -> model_artifacts (src.predict loads the newest version)
and then imports data/bundle/sentiment_events.parquet into sentiment_events (skip with --no-sentiment).
Re-publishing the same version replaces it."""
import argparse
import hashlib
import json

import psycopg2

from .common import ROOT, cfg, get_logger, query_df, tiger

log = get_logger("publish_models")
ARTIFACTS = ["meta.json", "regressors.joblib", "fusion.joblib", "baseline.joblib"]


def publish(models_dir, results_dir) -> str:
    meta = json.loads((models_dir / "meta.json").read_text())
    metrics = json.loads((results_dir / "metrics.json").read_text())
    version = meta["version"]
    if (metrics.get("meta") or {}).get("trained_at") not in (None, version):
        raise RuntimeError(f"{results_dir / 'metrics.json'} is from a different run than {models_dir} "
                           f"({metrics['meta']['trained_at']} vs {version}); copy both from the same training job")
    files = {n: (models_dir / n).read_bytes() for n in ARTIFACTS}
    with tiger() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO model_runs (version, trained_through, metrics, meta) VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (version) DO UPDATE SET metrics = EXCLUDED.metrics, meta = EXCLUDED.meta, "
                    "trained_through = EXCLUDED.trained_through, created_at = now()",
                    (version, meta.get("trained_through"), json.dumps(metrics), json.dumps(meta)))
        for name, blob in files.items():
            cur.execute("INSERT INTO model_artifacts (version, name, content, sha256) VALUES (%s, %s, %s, %s) "
                        "ON CONFLICT (version, name) DO UPDATE SET content = EXCLUDED.content, sha256 = EXCLUDED.sha256",
                        (version, name, psycopg2.Binary(blob), hashlib.sha256(blob).hexdigest()))
    log.info("published model %s (%s, %.1f MB)", version, ", ".join(files), sum(map(len, files.values())) / 1e6)
    return version


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models/v2")
    ap.add_argument("--results", default="results")
    ap.add_argument("--no-sentiment", action="store_true", help="don't import data/bundle/sentiment_events.parquet")
    ap.add_argument("--list", action="store_true", help="show published versions and exit")
    a = ap.parse_args(argv)
    if a.list:
        print(query_df("SELECT version, created_at, trained_through FROM model_runs ORDER BY created_at DESC")
              .to_string(index=False))
        return
    publish(ROOT / a.models, ROOT / a.results)
    bundle = ROOT / cfg()["data"]["bundle_dir"] / "sentiment_events.parquet"
    if not a.no_sentiment and bundle.exists():
        from . import sentiment_agent
        sentiment_agent.main(["--import-bundle"])
    elif not a.no_sentiment:
        log.info("no %s: sentiment import skipped", bundle)


if __name__ == "__main__":
    main()
