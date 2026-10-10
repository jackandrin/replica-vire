#!/bin/bash
# SLURM job for Theia (USC RC). Submit from the project folder:
#
#   CPU (recommended for this project):
#     sbatch deploy/theia.sbatch
#   GPU:
#     sbatch --gres=gpu:1 --export=ALL,USE_GPU=1 deploy/theia.sbatch
#
# Cluster-specific flags (partition/account/qos) go on the command line or in the PLACEHOLDER lines below.
# Command-line flags override the #SBATCH lines. Options for MODE: full | cycle | train
#     sbatch --export=ALL,MODE=cycle deploy/theia.sbatch
# Add PLACE=1 (MODE=full only) to submit real PAPER orders; the default is a dry run.
#
#SBATCH --job-name=hk-trader
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
## PLACEHOLDER: uncomment and fill in what RC tells you to use
##SBATCH --partition=PARTITION_NAME
##SBATCH --account=ACCOUNT_NAME
##SBATCH --mail-user=wtpeyton@email.sc.edu
##SBATCH --mail-type=END,FAIL

set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/hackathon-trader}"
MODE="${MODE:-full}"            # full = scripts/run_all.sh | cycle = src.run_cycle | train = train + predict + dry-run decide
PY_MODULE="${PY_MODULE:-}"      # e.g. export PY_MODULE=python3 if the env needs a module
CUDA_MODULE="${CUDA_MODULE:-}"  # usually not needed: the pip CUDA wheel bundles its own runtime

cd "$APP_DIR"
mkdir -p logs

if [ -n "$PY_MODULE" ]; then module load "$PY_MODULE"; fi
if [ -n "$CUDA_MODULE" ] && [ "${USE_GPU:-0}" = "1" ]; then module load "$CUDA_MODULE"; fi
# shellcheck disable=SC1091
. .venv/bin/activate

echo "job=$SLURM_JOB_ID node=$(hostname) mode=$MODE use_gpu=${USE_GPU:-0} started=$(date -Is)"
if [ "${USE_GPU:-0}" = "1" ]; then
  nvidia-smi -L || echo "WARNING: no GPU visible; training will fall back to CPU"
  python -c "import torch; print('cuda available:', torch.cuda.is_available())"
fi

# Show whether this compute node can reach the services (some clusters block outbound traffic); read the PASS/FAIL lines in the log.
python -m src.check_apis

case "$MODE" in
  full)  if [ "${PLACE:-0}" = "1" ]; then bash scripts/run_all.sh --place; else bash scripts/run_all.sh; fi ;;  # dry-run unless PLACE=1
  cycle) python -m src.run_cycle ;;
  train) python -m src.train && python -m src.predict && python -m src.decide --dry-run ;;
  *) echo "unknown MODE=$MODE"; exit 2 ;;
esac

echo "finished=$(date -Is)"
