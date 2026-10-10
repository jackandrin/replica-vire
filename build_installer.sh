#!/usr/bin/env bash
# Pack hackathon-trader/ into ONE self-extracting file: dist/install.sh
# Copy that single file to the Ubuntu server and run:  bash install.sh [target_dir] [setup.sh flags]
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p dist
OUT=dist/install.sh

cat > "$OUT" <<'HEADER'
#!/usr/bin/env bash
# Self-extracting installer for hackathon-trader.
#   bash install.sh                       -> installs to ~/hackathon-trader
#   bash install.sh /opt/trader --systemd -> custom dir; extra flags go to setup.sh
set -euo pipefail
TARGET="$HOME/hackathon-trader"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then TARGET="$1"; shift; fi
echo "Extracting to $TARGET"
mkdir -p "$TARGET"
sed '1,/^__PAYLOAD_BELOW__$/d' "$0" | base64 -d | tar xz -C "$TARGET"
chmod +x "$TARGET/setup.sh" "$TARGET"/scripts/*.sh
cd "$TARGET"
exec ./setup.sh "$@"
exit 0
__PAYLOAD_BELOW__
HEADER

tar czf - --exclude='.venv' --exclude='__pycache__' --exclude='.env' --exclude='*.pyc' \
    --exclude='data' --exclude='models' --exclude='results' --exclude='logs' --exclude='.pytest_cache' \
    -C hackathon-trader . | base64 >> "$OUT"
chmod +x "$OUT"
echo "built $OUT ($(wc -c < "$OUT") bytes)"
