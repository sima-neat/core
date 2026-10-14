#!/usr/bin/env bash
# Install or upgrade Sentinel, the board telemetry daemon that Model::benchmark()
# reads power, temperature, CPU, and memory from. Run it on a Modalix board.
#
# Sentinel is reinstalled unless the running daemon is the latest production
# build. When that build cannot be resolved, a running daemon is kept. Exits
# non-zero when the latest build is not running at the end.
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

# Version of the running daemon, such as "main:766066b74f49"; empty when none runs.
running_sentinel_version() {
  systemctl is-active --quiet simaai-sentinel || return 0
  curl -fsS --max-time 5 --unix-socket "${SENTINEL_API_SOCKET}" http://localhost/v1/health 2>/dev/null |
    sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' || true
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

# Commit of the latest production build of Sentinel's main branch.
latest_sentinel_build() {
  SIMA_CLI_CHECK_FOR_UPDATE=0 "$1" neat install sentinel --prod --json 2>/dev/null |
    sed -n 's/.*"resolved_spec": *"\([^"]*\)".*/\1/p' || true
}

wait_for_sentinel_version() {
  local _
  for _ in {1..10}; do
    if [[ "$(running_sentinel_version)" == "$1" ]]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

main() {
  local running sima_cli="" build=""
  running="$(running_sentinel_version)"
  if sima_cli="$(find_sima_cli)"; then
    build="$(latest_sentinel_build "${sima_cli}")"
  fi

  if [[ -z "${build}" ]]; then
    if [[ -n "${running}" ]]; then
      log "WARNING: Could not resolve the latest Sentinel build; keeping Sentinel ${running}."
      return 0
    fi
    if [[ -z "${sima_cli}" ]]; then
      echo "sima-cli is required to install Sentinel, the board telemetry daemon used by Model::benchmark()." >&2
      echo "Install sima-cli, then rerun this installer." >&2
    else
      echo "Could not resolve the latest Sentinel build. Run 'sima-cli neat install sentinel --prod --json' to see why, then rerun this installer." >&2
    fi
    return 1
  fi

  local target="main:${build}"
  if [[ "${running}" == "${target}" ]]; then
    log "Sentinel ${running} is up to date."
    return 0
  fi

  SENTINEL_INSTALL_DIR="$(mktemp -d)"
  trap 'rm -rf "${SENTINEL_INSTALL_DIR}"' EXIT
  log "Installing Sentinel ${target} (running: ${running:-none})."
  # The caller has verified a Modalix board. --force skips sima-cli's own
  # board check, which classifies a PCIe card's Modalix userspace as Debian.
  # SIMA_CLI lets Sentinel's installer register its agent skill.
  if ! SIMA_CLI="${sima_cli}" SIMA_CLI_CHECK_FOR_UPDATE=0 \
    "${sima_cli}" neat install "sentinel@${build}" --prod --force --install-dir "${SENTINEL_INSTALL_DIR}"; then
    log "WARNING: 'sima-cli neat install sentinel' reported a failure; checking the daemon."
  fi

  if ! wait_for_sentinel_version "${target}"; then
    echo "Sentinel ${target} is not running after installation. Run 'sima-cli neat install sentinel', then rerun this installer." >&2
    return 1
  fi
  log "Sentinel ${target} is running."
}

main "$@"
