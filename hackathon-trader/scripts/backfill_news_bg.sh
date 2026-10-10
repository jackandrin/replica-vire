#!/usr/bin/env bash
# Hour 1: start the slow news backfill in the background so it survives SSH disconnects.
#   bash scripts/backfill_news_bg.sh ; tail -f logs/ingest_news.log
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
. .venv/bin/activate
mkdir -p logs
nohup python -m src.ingest_news > logs/ingest_news.log 2>&1 &
echo "started pid $! ; tail -f logs/ingest_news.log"
