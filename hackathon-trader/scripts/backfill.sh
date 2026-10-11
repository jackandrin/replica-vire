#!/usr/bin/env bash
# Download everything the v2 models train on into Tiger Data, then export the parquet bundle for the HPC.
# Needs internet + .env keys (ALPACA_*, FINNHUB_API_KEY, TIGER_DATABASE_URL, SEC_USER_AGENT). Resumable: re-run it.
#
#   bash scripts/backfill.sh              # foreground
#   bash scripts/backfill.sh --bg         # background (survives SSH drops); tail -f logs/backfill.log
#   SINCE=2022-01-01 bash scripts/backfill.sh   # shorter history (default: config.yaml data.start)
#
# Rough wall-clock for 55 stocks + 13 context ETFs since 2019: bars 10-30 min, events 30-90 min (EDGAR Form 4 at
# 8 req/s + ~1,900 FINRA daily files), documents several hours (Alpaca news with full text, EDGAR filings).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
if [ "${1:-}" = "--bg" ]; then
  mkdir -p logs
  nohup bash "$0" > logs/backfill.log 2>&1 &
  echo "started pid $! ; tail -f logs/backfill.log"
  exit 0
fi
. .venv/bin/activate
mkdir -p logs
S=()
if [ -n "${SINCE:-}" ]; then S=(--since "$SINCE"); fi

step() { echo; echo "=== $(date -Is) $*"; }
step "schema";          python -m src.init_db --tiger
step "API checks";      python -m src.check_apis || echo "some checks failed: the matching sources will be skipped"
step "bars (15m + 1d)"; python -m src.ingest_intraday ${SINCE:+--start "$SINCE"}
step "events";          python -m src.ingest_events "${S[@]}"
step "documents";       python -m src.ingest_docs "${S[@]}"
step "export bundle";   python -m src.export_dataset ${SINCE:+--start "$SINCE"}
du -sh data/bundle || true
cat data/bundle/manifest.json || true
echo
echo "Done. Next: python -m src.sentiment_agent --source bundle --sink bundle --estimate   (cost/size check)"
echo "Then copy hackathon-trader/ (with data/bundle) to Theia and follow docs/TRAINING.md"
