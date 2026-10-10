#!/usr/bin/env bash
# Ubuntu 22.04 / 24.04 setup for hackathon-trader.
#   ./setup.sh                 install system packages, venv, python deps, .env, run offline tests
#   ./setup.sh --systemd       ...also install the dashboard service + daily-cycle timer (needs sudo)
#   ./setup.sh --skip-apt      skip apt (no sudo / packages already present)
#   ./setup.sh --no-torch      skip PyTorch (ridge-only; MLP will not work)
#   ./setup.sh --gpu           install the default (CUDA) PyTorch wheel instead of the CPU one
# Safe to re-run.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
APP_DIR="$(pwd)"
WITH_SYSTEMD=0; SKIP_APT=0; NO_TORCH=0; GPU=0
for a in "$@"; do
  case "$a" in
    --systemd) WITH_SYSTEMD=1 ;;
    --skip-apt) SKIP_APT=1 ;;
    --no-torch) NO_TORCH=1 ;;
    --gpu) GPU=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

if [ -r /etc/os-release ]; then . /etc/os-release; fi
if [ "${ID:-}" != "ubuntu" ] && [ "${ID:-}" != "debian" ]; then
  echo "warning: tested on Ubuntu 22.04/24.04; detected '${PRETTY_NAME:-unknown}'" >&2
fi

if [ "$SKIP_APT" -eq 0 ]; then
  say "Installing system packages (apt)"
  $SUDO apt-get update -y
  $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-venv python3-pip python3-dev build-essential git curl ca-certificates tmux tzdata
fi

PYV=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')
python3 - <<'PY' || { echo "Python >= 3.10 required (found $PYV)"; exit 1; }
import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY
say "Using Python $PYV"

say "Creating virtualenv .venv"
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install --upgrade pip wheel setuptools

if [ "$NO_TORCH" -eq 0 ]; then
  if [ "$GPU" -eq 1 ]; then
    say "Installing PyTorch (default CUDA wheel, large download)"
    pip install torch
  else
    say "Installing PyTorch (CPU wheel)"
    pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
fi

say "Installing Python requirements"
pip install -r requirements.txt

say "Preparing .env and folders"
[ -f .env ] || cp .env.example .env
chmod 600 .env
mkdir -p data results models logs

say "Running offline tests (no network, no database)"
python -m pytest -q -p no:cacheprovider

if [ "$WITH_SYSTEMD" -eq 1 ]; then
  say "Installing systemd units"
  RUN_USER="${SUDO_USER:-$(id -un)}"
  for f in deploy/hackathon-dashboard.service deploy/hackathon-cycle.service deploy/hackathon-cycle.timer; do
    sed -e "s|__USER__|$RUN_USER|g" -e "s|__DIR__|$APP_DIR|g" "$f" | $SUDO tee "/etc/systemd/system/$(basename "$f")" >/dev/null
  done
  $SUDO systemctl daemon-reload
  $SUDO systemctl enable --now hackathon-dashboard.service
  $SUDO systemctl enable --now hackathon-cycle.timer
  echo "dashboard: http://<server-ip>:8501   (or: ssh -L 8501:localhost:8501 user@server)"
  systemctl list-timers hackathon-cycle.timer --no-pager || true
fi

cat <<EOF

Setup complete in $APP_DIR

Next steps
  1. Edit .env with your keys:           nano $APP_DIR/.env
  2. Activate the environment:           source $APP_DIR/.venv/bin/activate
  3. Create the TigerData tables:        python -m src.init_db --tiger
  4. Verify every API (Hour 0 gate):     python -m src.check_apis
  5. Follow PLAN.md hour by hour.        Quick path:  bash scripts/run_all.sh
EOF
