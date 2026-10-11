#!/bin/bash
# SLURM GPU job for Theia (USC RC): backfill sentiment_events with the local HF model (src/sentiment_agent.py).
# Reads data/bundle/documents.parquet, writes data/bundle/sentiment_events.parquet (resumable: just resubmit).
#
#   sbatch deploy/sentiment.sbatch
#   sbatch --export=ALL,SA_ARGS="--since 2024-01-01 --only AAPL,MSFT" deploy/sentiment.sbatch
#
# Compute nodes may have no internet, so download the models ONCE on the login node first (same HF_HOME as below):
#   . .venv/bin/activate && pip install -r requirements-sentiment-hf.txt   # torch: install the CUDA wheel separately
#   export HF_HOME=$HOME/.cache/huggingface
#   huggingface-cli login            # Llama 3.1 is gated: accept Meta's license on its Hugging Face page first
#   huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*"
#   huggingface-cli download Snowflake/snowflake-arctic-embed-m-v1.5
# Check token/doc counts before spending GPU hours:
#   python -m src.sentiment_agent --source bundle --sink bundle --estimate
# Afterwards copy sentiment_events.parquet back and load it:  python -m src.sentiment_agent --import-bundle
#
# Cluster-specific flags (partition/account/qos) go on the command line or in the PLACEHOLDER lines below.
#
#SBATCH --job-name=hk-sentiment
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm-sent-%j.out
#SBATCH --error=logs/slurm-sent-%j.err
## PLACEHOLDER: uncomment and fill in what RC tells you to use
##SBATCH --partition=PARTITION_NAME
##SBATCH --account=ACCOUNT_NAME
##SBATCH --mail-user=wtpeyton@email.sc.edu
##SBATCH --mail-type=END,FAIL

set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/hackathon-trader}"
PY_MODULE="${PY_MODULE:-}"      # e.g. export PY_MODULE=python3 if the env needs a module
CUDA_MODULE="${CUDA_MODULE:-}"  # usually not needed: the pip CUDA wheel bundles its own runtime
HF_BATCH="${HF_BATCH:-16}"      # GPU micro-batch; halves itself on CUDA OOM
SA_ARGS="${SA_ARGS:-}"          # extra flags for src.sentiment_agent

cd "$APP_DIR"
mkdir -p logs

if [ -n "$PY_MODULE" ]; then module load "$PY_MODULE"; fi
if [ -n "$CUDA_MODULE" ]; then module load "$CUDA_MODULE"; fi
# shellcheck disable=SC1091
. .venv/bin/activate

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "job=$SLURM_JOB_ID node=$(hostname) hf_home=$HF_HOME started=$(date -Is)"
nvidia-smi -L || echo "WARNING: no GPU visible; generation on CPU will be extremely slow"
python -c "import torch; print('cuda available:', torch.cuda.is_available())"

# shellcheck disable=SC2086
python -m src.sentiment_agent --backend hf --source bundle --sink bundle --hf-batch "$HF_BATCH" $SA_ARGS

echo "finished=$(date -Is)"
