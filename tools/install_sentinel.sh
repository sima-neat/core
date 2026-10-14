#!/usr/bin/env bash
# Install Sentinel, the board telemetry daemon that Model::benchmark() reads
# power, temperature, CPU, and memory from. Run it on a Modalix board.
#
# A running daemon is kept: reinstalling restarts it and would interrupt a trace
# another tool is recording. Exits non-zero when Sentinel is not running at the
# end.
#
# Environment:
# - SIMA_CLI: sima-cli executable (default: PATH, then its standard install
#   locations)
set -euo pipefail

readonly SENTINEL_API_SOCKET="/run/simaai-sentinel/api.sock"
SENTINEL_INSTALL_DIR=""

log() {
  printf '[install_sentinel] %s\n' "$*"
}

sentinel_version() {
  curl -fsS --max-time 5 --unix-socket "${SENTINEL_API_SOCKET}" http://localhost/v1/health 2>/dev/null |
    sed -n 's/.*"version": *"\([^"]*\)".*/\1/p'
}

sentinel_is_running() {
  systemctl is-active --quiet simaai-sentinel && [[ -n "$(sentinel_version)" ]]
}

# sima-cli lives in a virtualenv that non-login shells, such as the one
# sima-cli runs this installer in, may not have on PATH.
find_sima_cli() {
  local candidate
  for candidate in "${SIMA_CLI:-}" "$(command -v sima-cli || true)" \
    /data/sima-cli/.venv/bin/sima-cli "${HOME}/.sima-cli/.venv/bin/sima-cli"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

wait_for_sentinel() {
  local _
  for _ in {1..10}; do
    if sentinel_is_running; then
      return 0
    fi
    sleep 1
  done
  return 1
}

main() {
  if sentinel_is_running; then
    log "Sentinel $(sentinel_version) is running; skipping its installation."
    return 0
  fi

  local sima_cli
  if ! sima_cli="$(find_sima_cli)"; then
    echo "sima-cli is required to install Sentinel, the board telemetry daemon used by Model::benchmark()." >&2
    echo "Install sima-cli, then rerun this installer." >&2
    return 1
  fi

  SENTINEL_INSTALL_DIR="$(mktemp -d)"
  trap 'rm -rf "${SENTINEL_INSTALL_DIR}"' EXIT
  log "Installing Sentinel, the board telemetry daemon used by Model::benchmark()."
  # The caller has verified a Modalix board. --force skips sima-cli's own
  # board check, which classifies a PCIe card's Modalix userspace as Debian.
  # SIMA_CLI lets Sentinel's installer register its agent skill.
  if ! SIMA_CLI="${sima_cli}" SIMA_CLI_CHECK_FOR_UPDATE=0 \
    "${sima_cli}" neat install sentinel --prod --force --install-dir "${SENTINEL_INSTALL_DIR}"; then
    log "WARNING: 'sima-cli neat install sentinel' reported a failure; checking the daemon."
  fi

  if ! wait_for_sentinel; then
    echo "Sentinel is not running after installation. Run 'sima-cli neat install sentinel', then rerun this installer." >&2
    return 1
  fi
  log "Sentinel $(sentinel_version) is running."
}

main "$@"
