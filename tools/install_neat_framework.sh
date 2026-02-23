#!/usr/bin/env bash
set -euo pipefail

# install_neat_framework.sh
#
# Purpose:
# - Install SiMa NEAT wheel into a Python virtual environment (board mode only).
# - Install NEAT runtime .deb packages on Modalix board or into eLxr SDK sysroot.
#
# Behavior:
# - Auto-detects environment:
#   - eLxr SDK when /etc/sdk-release exists, or SYSROOT points to a valid directory.
#   - Modalix board when /etc/buildinfo reports MACHINE=modalix.
# - In eLxr SDK mode, extracts .deb payloads into SYSROOT using dpkg-deb -x.
# - In eLxr SDK mode, skips wheel installation (wheel targets aarch64 runtime on DevKit).
# - In Modalix board mode, installs .deb packages with apt (sudo).
#
# Expected working directory:
# - Directory containing:
#   - one .whl file
#   - sima-neat-*-Linux-core.deb
#   - neat-*.deb runtime dependencies
#
# Environment overrides:
# - PYNEAT_VENV_DIR: Python virtualenv path (default: $HOME/pyneat/.venv)
# - SUDO_PASSWORD / DEVKIT_PASSWORD: sudo password (preferred non-interactive override)
# - DEFAULT_SUDO_PASSWORD: fallback password (default: edgeai)
# - SYSROOT: SDK sysroot path override (default: /opt/toolchain/aarch64/modalix)

VENV_DIR="${PYNEAT_VENV_DIR:-$HOME/pyneat/.venv}"
SUDO_PASSWORD="${SUDO_PASSWORD:-${DEVKIT_PASSWORD:-}}"
DEFAULT_SUDO_PASSWORD="${DEFAULT_SUDO_PASSWORD:-edgeai}"
SDK_SYSROOT_DEFAULT="/opt/toolchain/aarch64/modalix"

log() {
  printf '[install_neat_framework] %s\n' "$*"
}

run_sudo() {
  if sudo -n true >/dev/null 2>&1; then
    sudo "$@"
    return 0
  fi

  local pw="${SUDO_PASSWORD}"
  if [[ -z "${pw}" ]]; then
    pw="${DEFAULT_SUDO_PASSWORD}"
  fi

  if printf '%s\n' "${pw}" | sudo -S -v >/dev/null 2>&1; then
    printf '%s\n' "${pw}" | sudo -S "$@"
    return 0
  fi

  if [[ -t 0 ]]; then
    read -r -s -p "Enter sudo password: " pw
    echo
    if [[ -z "${pw}" ]]; then
      echo "sudo password is required." >&2
      exit 1
    fi
    printf '%s\n' "${pw}" | sudo -S -v >/dev/null
    printf '%s\n' "${pw}" | sudo -S "$@"
    return 0
  fi

  echo "Unable to authenticate sudo. Set SUDO_PASSWORD or DEVKIT_PASSWORD." >&2
  exit 1
}

detect_env_mode() {
  if [[ -f /etc/sdk-release ]]; then
    echo "elxr-sdk"
    return 0
  fi
  if [[ -n "${SYSROOT:-}" && -d "${SYSROOT}" ]]; then
    echo "elxr-sdk"
    return 0
  fi
  if [[ -f /etc/buildinfo ]] && grep -qE '^MACHINE[[:space:]]*=[[:space:]]*modalix' /etc/buildinfo; then
    echo "modalix-board"
    return 0
  fi
  echo "modalix-board"
}

install_debs_into_sysroot() {
  local sysroot="${SYSROOT:-${SDK_SYSROOT_DEFAULT}}"
  if [[ ! -d "${sysroot}" ]]; then
    echo "SDK sysroot not found: ${sysroot}" >&2
    echo "Set SYSROOT to your eLxr SDK sysroot path." >&2
    exit 1
  fi

  if ! command -v dpkg-deb >/dev/null 2>&1; then
    echo "dpkg-deb is required to install DEBs into SDK sysroot." >&2
    exit 1
  fi

  log "Detected eLxr SDK environment; installing DEBs into sysroot: ${sysroot}"
  local use_sudo=""
  if [[ ! -w "${sysroot}" ]]; then
    use_sudo="1"
  fi

  local deb
  for deb in "${DEBS[@]}"; do
    if [[ -n "${use_sudo}" ]]; then
      run_sudo dpkg-deb -x "${deb}" "${sysroot}"
    else
      dpkg-deb -x "${deb}" "${sysroot}"
    fi
  done

  fix_sdk_symlinks "${sysroot}" "${use_sudo}"
}

fix_sdk_symlinks() {
  local sysroot="$1"
  local use_sudo="${2:-}"
  local legacy_dir="${sysroot}/usr/lib/sima-neat/gst-plugins"
  local canonical_dir="${sysroot}/usr/lib/aarch64-linux-gnu/neat/gst-plugins"
  local rel_target_prefix='../../aarch64-linux-gnu/neat/gst-plugins'

  if [[ ! -d "${legacy_dir}" || ! -d "${canonical_dir}" ]]; then
    return 0
  fi

  log "Repairing broken SDK symlinks in ${legacy_dir}"
  local repaired=0
  while IFS= read -r link_path; do
    local name target_abs new_target
    name="$(basename "${link_path}")"
    target_abs="${canonical_dir}/${name}"
    if [[ -f "${target_abs}" ]]; then
      new_target="${rel_target_prefix}/${name}"
      if [[ -n "${use_sudo}" ]]; then
        run_sudo ln -sfn "${new_target}" "${link_path}"
      else
        ln -sfn "${new_target}" "${link_path}"
      fi
      repaired=$((repaired + 1))
    fi
  done < <(find "${legacy_dir}" -maxdepth 1 -type l ! -exec test -e {} \; -print)

  if [[ "${repaired}" -gt 0 ]]; then
    log "Repaired ${repaired} broken symlink(s) in SDK sysroot."
  fi
}

install_debs_on_board() {
  log "Detected Modalix board environment; installing DEBs with apt."
  run_sudo apt install -y --allow-downgrades "${DEBS[@]}"
}

mapfile -t DEBS < <(find . -maxdepth 1 -type f \( -name 'sima-neat-*-Linux-core.deb' -o -name 'neat-*.deb' \) | sort)
if [[ "${#DEBS[@]}" -lt 1 ]]; then
  echo "No required DEB files found in current directory." >&2
  exit 1
fi

ENV_MODE="$(detect_env_mode)"
if [[ "${ENV_MODE}" == "elxr-sdk" ]]; then
  log "Skipping wheel installation in eLxr SDK environment."
  install_debs_into_sysroot
else
  python3 -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip

  WHEEL_FILE="$(ls -1 ./*.whl 2>/dev/null | head -n1 || true)"
  if [[ -z "${WHEEL_FILE}" ]]; then
    echo "No wheel file found in current directory." >&2
    exit 1
  fi
  "${VENV_DIR}/bin/python" -m pip install --force-reinstall "${WHEEL_FILE}"
  install_debs_on_board
fi
