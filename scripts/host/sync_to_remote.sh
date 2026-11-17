#!/usr/bin/env bash
# Push this local shadow copy -> remote compute host (training/inference).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-user@your-gpu-host}"
REMOTE_PATH="${REMOTE_PATH:-/path/to/realtime_3dvision}"

# Default: sync code + configs. Use --all to also sync data/pretrained.
MODE="${1:-code}"

RSYNC_EXCLUDES=(
  --exclude '.git/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.pytest_cache/'
  --exclude '.mypy_cache/'
  --exclude 'tmp/'
  --exclude 'runs/'
  --exclude 'wandb/'
)

case "$MODE" in
  code)
    RSYNC_EXCLUDES+=(
      --exclude 'data/'
      --exclude 'pretrained/'
      --exclude 'datasets/'
    )
    ;;
  all) ;;
  *)
    echo "Usage: $0 [code|all]" >&2
    exit 1
    ;;
esac

echo "[sync_to_remote] ${ROOT}/ -> ${REMOTE_HOST}:${REMOTE_PATH}/  (mode=${MODE})"
rsync -avh --delete --info=stats2 "${RSYNC_EXCLUDES[@]}" \
  "${ROOT}/" "${REMOTE_HOST}:${REMOTE_PATH}/"
echo "[sync_to_remote] done"
