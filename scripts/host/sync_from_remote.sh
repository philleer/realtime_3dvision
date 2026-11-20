#!/usr/bin/env bash
# Pull remote compute artifacts back into this local shadow copy.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=_load_remote.sh
source "$(dirname "$0")/_load_remote.sh"

# Pull code + tmp inference artifacts by default; use --all for data/pretrained too.
MODE="${1:-results}"

RSYNC_EXCLUDES=(
  --exclude '.git/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.pytest_cache/'
  --exclude '.mypy_cache/'
  --exclude '.remote_host'
)

case "$MODE" in
  results)
    # Only bring back inference/viz products under tmp/ + source edits.
    rsync -avh --info=stats2 "${RSYNC_EXCLUDES[@]}" \
      --include 'tmp/***'/ \
      --include 'runs/***'/ \
      --include 'configs/***'/ \
      --include 'scripts/***'/ \
      --include 'srcs/***'/ \
      --include 'tests/***'/ \
      --include '*/' \
      --exclude '*' \
      "${REMOTE_HOST}:${REMOTE_PATH}/" "${ROOT}/"
    ;;
  all)
    rsync -avh --info=stats2 "${RSYNC_EXCLUDES[@]}" \
      "${REMOTE_HOST}:${REMOTE_PATH}/" "${ROOT}/"
    ;;
  *)
    echo "Usage: $0 [results|all]" >&2
    exit 1
    ;;
esac

echo "[sync_from_remote] done (mode=${MODE})"
