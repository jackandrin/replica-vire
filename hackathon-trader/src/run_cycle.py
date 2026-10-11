"""One unattended live cycle: ingest -> sentiment agent -> predict -> recommendations / paper orders.
python -m src.run_cycle [--dry-run] [--with-events] [--skip-sentiment] [--skip-snowflake]
Scheduled by deploy/hackathon-cycle.timer (hourly 10:01-15:01 and 15:46 ET). Events refresh in the 15:46 run."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

from . import (autotrade, ingest_docs, ingest_events, ingest_intraday, portfolio, predict, sentiment_agent,
               sync_snowflake)
from .common import get_logger, log_step

log = get_logger("run_cycle")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="alerts only, no orders")
    ap.add_argument("--with-events", action="store_true", help="also refresh earnings/insider/analyst/short volume")
    ap.add_argument("--skip-sentiment", action="store_true")
    ap.add_argument("--skip-snowflake", action="store_true")
    a = ap.parse_args(argv)

    steps = [("ingest_intraday", lambda: ingest_intraday.main(["--incremental"])),
             ("ingest_docs", lambda: ingest_docs.main(["--incremental"]))]
    now = datetime.now(ZoneInfo("America/New_York"))
    if a.with_events or (now.hour, now.minute) >= (15, 30):  # events change slowly: refresh in the last run of the day
        steps.append(("ingest_events", lambda: ingest_events.main(["--incremental"])))
    if not a.skip_sentiment:
        steps.append(("sentiment_agent", lambda: sentiment_agent.main(["--backend", "cortex"])))
    steps += [("predict", lambda: predict.main([])),
              ("autotrade", lambda: autotrade.main(["--no-orders"] if a.dry_run else [])),
              ("portfolios", lambda: portfolio.main(["--dry-run"] if a.dry_run else []))]  # fills, rebalance, snapshots
    if not a.skip_snowflake:
        steps.append(("sync_snowflake", lambda: sync_snowflake.main([])))

    critical = {"ingest_intraday", "predict", "autotrade"}
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
