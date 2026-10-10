"""One unattended daily cycle (run after the close, e.g. 16:20 ET on weekdays).
python -m src.run_cycle [--dry-run] [--skip-snowflake] [--retrain]"""
import argparse

from . import decide, features, ingest_bars, ingest_news, predict, sentiment, sync_snowflake, train
from .common import get_logger, log_step

log = get_logger("run_cycle")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-snowflake", action="store_true")
    ap.add_argument("--retrain", action="store_true", help="also retrain models (off by default)")
    a = ap.parse_args(argv)

    steps = [("ingest_bars", lambda: ingest_bars.main(["--incremental"])),
             ("ingest_news", lambda: ingest_news.main(["--days", "10"])),
             ("sentiment", lambda: sentiment.main([])),
             ("features", lambda: features.main([]))]
    if a.retrain:
        steps.append(("train", lambda: train.main([])))
    steps += [("predict", lambda: predict.main([])),
              ("decide", lambda: decide.main(["--dry-run"] if a.dry_run else []))]
    if not a.skip_snowflake:
        steps.append(("sync_snowflake", lambda: sync_snowflake.main([])))

    critical = {"ingest_bars", "features", "predict", "decide"}
    for name, fn in steps:
        try:
            fn()
            log_step(name, True)
        except Exception as e:  # noqa: BLE001
            log.exception("step %s failed", name)
            log_step(name, False, repr(e))
            if name in critical:
                log.error("critical step failed; stopping cycle (no orders placed from stale data)")
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
