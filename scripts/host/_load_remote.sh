# Shared remote config loader for sync_*.sh (source me; do not run directly).
# Prefer env REMOTE_HOST / REMOTE_PATH; else load repo-root .remote_host (gitignored).

_HOST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_REMOTE_FILE="${_HOST_ROOT}/.remote_host"

if [[ -z "${REMOTE_HOST:-}" || -z "${REMOTE_PATH:-}" ]]; then
  if [[ -f "${_REMOTE_FILE}" ]]; then
    # shellcheck disable=SC1090
    set -a
    # shellcheck source=/dev/null
    source "${_REMOTE_FILE}"
    set +a
  fi
fi

if [[ -z "${REMOTE_HOST:-}" || -z "${REMOTE_PATH:-}" ]]; then
  echo "Missing REMOTE_HOST / REMOTE_PATH." >&2
  echo "Copy .remote_host.example -> .remote_host and fill in local values," >&2
  echo "or export REMOTE_HOST and REMOTE_PATH." >&2
  exit 1
fi
