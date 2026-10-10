#!/usr/bin/env bash
# Full pipeline, first run: backfill -> features -> train -> predict -> (dry-run) decide -> snowflake.
# Usage: bash scripts/run_all.sh [--place]   (--place submits real PAPER orders; default is dry-run)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
. .venv/bin/activate
mkdir -p logs

python -m src.init_db --tiger
python -m src.check_apis
python -m src.ingest_bars
python -m src.ingest_news 2>&1 | tee logs/ingest_news.log
python -m src.sentiment
python -m src.features
python -m src.train
python -m src.predict
if [ "${1:-}" = "--place" ]; then python -m src.decide; else python -m src.decide --dry-run; fi
python -m src.sync_snowflake || echo "snowflake sync failed (non-blocking)"
echo "Done. Dashboard: streamlit run src/dashboard.py --server.address 0.0.0.0 --server.port 8501"
