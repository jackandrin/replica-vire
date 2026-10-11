#!/bin/bash
# SLURM job for Theia (USC RC): train the v2 stack from the parquet bundle (src/train.py). No internet needed.
# Full walkthrough: docs/TRAINING.md
#
#   sbatch deploy/theia.sbatch                                  # full training (GPU if one is allocated)
#   sbatch --gres=gpu:1 deploy/theia.sbatch                     # ask for a GPU (the network is small; CPU is fine)
#   sbatch --export=ALL,MODE=quick deploy/theia.sbatch          # 5-minute smoke test of the whole pipeline
#   sbatch --export=ALL,TRAIN_ARGS="--rebuild-features" deploy/theia.sbatch   # after a new bundle
#   SENT=$(sbatch --parsable deploy/sentiment.sbatch); sbatch --dependency=afterok:$SENT deploy/theia.sbatch
#
# Outputs: models/v2/{regressors,fusion,baseline}.joblib + meta.json, results/metrics.json (shown on the dashboard's
# Performance page), results/test_predictions.parquet, results/nn_history.json. Copy models/ and results/ back to
# the server that runs src.predict.
#
# Cluster-specific flags (partition/account/qos) go on the command line or in the PLACEHOLDER lines below.
#
#SBATCH --job-name=rv-train
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-train-%j.out
#SBATCH --error=logs/slurm-train-%j.err
## PLACEHOLDER: uncomment and fill in what RC tells you to use
##SBATCH --partition=PARTITION_NAME
##SBATCH --account=ACCOUNT_NAME
##SBATCH --mail-user=wtpeyton@email.sc.edu
##SBATCH --mail-type=END,FAIL

set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/hackathon-trader}"
MODE="${MODE:-full}"            # full | quick
TRAIN_ARGS="${TRAIN_ARGS:-}"    # extra flags for src.train (e.g. --rebuild-features, --only AAPL,MSFT)
PY_MODULE="${PY_MODULE:-}"      # e.g. export PY_MODULE=python3 if the env needs a module
CUDA_MODULE="${CUDA_MODULE:-}"  # usually not needed: the pip CUDA wheel bundles its own runtime

cd "$APP_DIR"
mkdir -p logs

if [ -n "$PY_MODULE" ]; then module load "$PY_MODULE"; fi
if [ -n "$CUDA_MODULE" ]; then module load "$CUDA_MODULE"; fi
# shellcheck disable=SC1091
. .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job=${SLURM_JOB_ID:-local} node=$(hostname) mode=$MODE started=$(date -Is)"
python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
test -f data/bundle/bars_15m.parquet || { echo "data/bundle missing: run scripts/backfill.sh + export first"; exit 2; }
cat data/bundle/manifest.json 2>/dev/null || true

case "$MODE" in
  # shellcheck disable=SC2086
  full)  python -m src.train $TRAIN_ARGS ;;
  # shellcheck disable=SC2086
  quick) python -m src.train --quick --out models/v2-quick --results results/quick $TRAIN_ARGS ;;
  *) echo "unknown MODE=$MODE"; exit 2 ;;
esac

echo "finished=$(date -Is)"
