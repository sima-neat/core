#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SITE_DIR="${DOCS_LINK_SITE_DIR:-${REPO_ROOT}/website/build}"
START_PATHS="${DOCS_LINK_START_PATHS:-all}"
PORT="${DOCS_LINK_CHECK_PORT:-}"
CONCURRENCY="${DOCS_LINK_CONCURRENCY:-25}"
RETRY_ERRORS_COUNT="${DOCS_LINK_RETRY_ERRORS_COUNT:-3}"
HOST="localhost"
SERVER_PID=""
SERVER_LOG=""
SERVE_DIR=""

die() {
  echo "check_docs_links: $*" >&2
  exit 1
}

cleanup() {
  local rc=$?
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  [[ -n "${SERVER_LOG}" ]] && rm -f "${SERVER_LOG}"
  [[ -n "${SERVE_DIR}" && "${SERVE_DIR}" != "${SITE_DIR}" ]] && rm -rf "${SERVE_DIR}"
  exit "${rc}"
}
trap cleanup EXIT

if [[ ! -d "${SITE_DIR}" ]]; then
  die "site directory not found: ${SITE_DIR}"
fi

landing_page="${SITE_DIR}/index.html"
if [[ -f "${landing_page}" ]] &&
   grep -Eq '<a[^>]*href="([^"/#][^":]*|\.\/[^":]*)"[^>]*class="overview-link-card"|<a[^>]*class="overview-link-card"[^>]*href="([^"/#][^":]*|\.\/[^":]*)"' "${landing_page}"; then
  grep -En '<a[^>]*href="([^"/#][^":]*|\.\/[^":]*)"[^>]*class="overview-link-card"|<a[^>]*class="overview-link-card"[^>]*href="([^"/#][^":]*|\.\/[^":]*)"' "${landing_page}" >&2 || true
  die "overview landing cards must use route-absolute /software/ links so /software resolves under the software route"
fi

if [[ -z "${PORT}" ]]; then
  PORT="$(python3 - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
fi

BASE_URL="http://${HOST}:${PORT}"
SERVER_LOG="$(mktemp "${TMPDIR:-/tmp}/sima-neat-docs-link-serve.XXXXXX")"
BASE_PATH="${DOCS_LINK_BASE_PATH:-${DOCS_BASE_URL:-/}}"

case "${BASE_PATH}" in
  "")
    BASE_PATH="/"
    ;;
  /*)
    ;;
  *)
    BASE_PATH="/${BASE_PATH}"
    ;;
esac

if [[ "${BASE_PATH}" != "/" ]]; then
  BASE_PATH="/${BASE_PATH#/}"
  BASE_PATH="${BASE_PATH%/}/"
  SERVE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sima-neat-docs-link-root.XXXXXX")"
  mkdir -p "${SERVE_DIR}${BASE_PATH%/}"
  cp -a "${SITE_DIR}/." "${SERVE_DIR}${BASE_PATH%/}/"
else
  SERVE_DIR="${SITE_DIR}"
fi

READY_URL="${BASE_URL}"
if [[ "${BASE_PATH}" != "/" ]]; then
  READY_URL="${BASE_URL}${BASE_PATH}"
fi

echo "Serving docs from ${SITE_DIR} at ${READY_URL}"
npx --yes serve@14.2.6 "${SERVE_DIR}" -l "${PORT}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"

for _ in $(seq 1 100); do
  if curl -fsS "${READY_URL}" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    cat "${SERVER_LOG}" >&2 || true
    die "docs server exited before becoming ready"
  fi
  sleep 0.2
done

if ! curl -fsS "${READY_URL}" >/dev/null 2>&1; then
  cat "${SERVER_LOG}" >&2 || true
  die "docs server did not become ready at ${READY_URL}"
fi

url_for_start_path() {
  case "$1" in
    all)
      printf '%s' "${READY_URL}"
      ;;
    http://*|https://*)
      printf '%s' "$1"
      ;;
    /*)
      printf '%s%s' "${BASE_URL}" "$1"
      ;;
    *)
      printf '%s/%s' "${BASE_URL}" "$1"
      ;;
  esac
}

for start_path in ${START_PATHS}; do
  start_url="$(url_for_start_path "${start_path}")"
  echo "Checking internal docs links from ${start_url}"
  npx --yes linkinator@7.6.1 "${start_url}" \
    --recurse \
    --concurrency "${CONCURRENCY}" \
    --retry-errors \
    --retry-errors-count "${RETRY_ERRORS_COUNT}" \
    --skip "^(mailto:|tel:|https?://(?!${HOST}:${PORT}))"
done
