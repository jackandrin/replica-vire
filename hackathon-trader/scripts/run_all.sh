#!/usr/bin/env bash
# Full first run on ONE machine (no HPC): backfill -> Cortex sentiment -> train -> predict -> alerts (dry run).
# Usage: bash scripts/run_all.sh [--place]   (--place lets autotrade submit PAPER orders; default is dry-run)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
. .venv/bin/activate
mkdir -p logs

bash scripts/backfill.sh
python -m src.sentiment_agent --backend cortex --source bundle --sink bundle
python -m src.sentiment_agent --import-bundle
python -m src.export_dataset            # re-export so the bundle includes sentiment_events
python -m src.train --rebuild-features
python -m src.publish_models --no-sentiment   # metrics -> website, models -> predict (sentiment already imported)
python -m src.predict
if [ "${1:-}" = "--place" ]; then python -m src.autotrade; else python -m src.autotrade --no-orders; fi
python -m src.sync_snowflake || echo "snowflake sync failed (non-blocking)"
echo "Done. API: python -m uvicorn src.web_api:app --port 8000 ; frontend: cd ../frontend && npm run dev"
