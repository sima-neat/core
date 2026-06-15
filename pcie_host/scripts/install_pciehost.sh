#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: install_pciehost.sh [options]

Install sima-pcie-host Debian packages on this host, then provision PCIe SSH.

Options:
  --deb <path>             Install this runtime DEB instead of auto-detecting it
  --dev-deb <path>         Install this development DEB instead of auto-detecting it
  --search-dir <dir>       Directory to search for DEBs (default: current dir, then pcie_host/packaging)
  --runtime-only           Do not install sima-pcie-host-dev
  --skip-setup             Do not run pcie-setup.sh after package install
  --setup-args <s>         Extra arguments passed to pcie-setup.sh
  --setup-best-effort      Do not fail installation if pcie-setup.sh fails
  -h, --help               Show this help

Examples:
  ./install_pciehost.sh
  ./install_pciehost.sh --deb sima-pcie-host_0.4.0_amd64.deb
  ./install_pciehost.sh --runtime-only
  ./install_pciehost.sh --setup-args "--hosts 10.0.0.2"
  ./install_pciehost.sh --skip-setup

Environment:
  SIMAPCIE_RUNTIME_ONLY=1       Same as --runtime-only
  SIMAPCIE_SKIP_SETUP=1         Same as --skip-setup
  SIMAPCIE_SETUP_ARGS="..."     Same as --setup-args
  SIMAPCIE_SETUP_BEST_EFFORT=1  Same as --setup-best-effort
EOF
}

env_truthy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

DEB_PATH=""
DEV_DEB_PATH=""
SEARCH_DIR=""
RUNTIME_ONLY="OFF"
RUN_SETUP="ON"
SETUP_ARGS=""
SETUP_BEST_EFFORT="OFF"

if env_truthy "${SIMAPCIE_RUNTIME_ONLY:-}"; then
  RUNTIME_ONLY="ON"
fi
if env_truthy "${SIMAPCIE_SKIP_SETUP:-}"; then
  RUN_SETUP="OFF"
fi
if [[ -n "${SIMAPCIE_SETUP_ARGS:-}" ]]; then
  SETUP_ARGS="${SIMAPCIE_SETUP_ARGS}"
fi
if env_truthy "${SIMAPCIE_SETUP_BEST_EFFORT:-}"; then
  SETUP_BEST_EFFORT="ON"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deb)
      DEB_PATH="${2:-}"
      if [[ -z "${DEB_PATH}" ]]; then
        echo "ERROR: --deb requires a value" >&2
        exit 1
      fi
      shift 2
      ;;
    --dev-deb)
      DEV_DEB_PATH="${2:-}"
      if [[ -z "${DEV_DEB_PATH}" ]]; then
        echo "ERROR: --dev-deb requires a value" >&2
        exit 1
      fi
      shift 2
      ;;
    --search-dir)
      SEARCH_DIR="${2:-}"
      if [[ -z "${SEARCH_DIR}" ]]; then
        echo "ERROR: --search-dir requires a value" >&2
        exit 1
      fi
      shift 2
      ;;
    --runtime-only)
      RUNTIME_ONLY="ON"
      shift
      ;;
    --skip-setup)
      RUN_SETUP="OFF"
      shift
      ;;
    --setup-args)
      SETUP_ARGS="${2:-}"
      shift 2
      ;;
    --setup-best-effort)
      SETUP_BEST_EFFORT="ON"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

detect_deb_arch() {
  if command -v dpkg >/dev/null 2>&1; then
    dpkg --print-architecture
    return 0
  fi

  case "$(uname -m)" in
    x86_64)
      echo "amd64"
      ;;
    aarch64|arm64)
      echo "arm64"
      ;;
    *)
      echo "ERROR: cannot determine Debian architecture for $(uname -m)" >&2
      exit 1
      ;;
  esac
}

latest_matching_deb() {
  local dir="$1"
  local pattern="$2"
  find "${dir}" -maxdepth 1 -type f -name "${pattern}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 {print $2}'
}

absolute_path() {
  local path="$1"
  local dir base
  dir="$(cd "$(dirname "${path}")" && pwd)"
  base="$(basename "${path}")"
  printf '%s/%s\n' "${dir}" "${base}"
}

deb_arch="$(detect_deb_arch)"
case "${deb_arch}" in
  amd64|arm64)
    ;;
  *)
    echo "ERROR: unsupported host architecture for sima-pcie-host: ${deb_arch}" >&2
    exit 1
    ;;
esac

if [[ -n "${SEARCH_DIR}" ]]; then
  search_dirs=("${SEARCH_DIR}")
else
  search_dirs=("${PWD}" "${ROOT_DIR}/packaging")
fi

if [[ -z "${DEB_PATH}" ]]; then
  for dir in "${search_dirs[@]}"; do
    [[ -d "${dir}" ]] || continue
    DEB_PATH="$(latest_matching_deb "${dir}" "sima-pcie-host_*_${deb_arch}.deb")"
    [[ -n "${DEB_PATH}" ]] && break
  done
fi

if [[ "${RUNTIME_ONLY}" != "ON" && -z "${DEV_DEB_PATH}" ]]; then
  for dir in "${search_dirs[@]}"; do
    [[ -d "${dir}" ]] || continue
    DEV_DEB_PATH="$(latest_matching_deb "${dir}" "sima-pcie-host-dev_*_${deb_arch}.deb")"
    [[ -n "${DEV_DEB_PATH}" ]] && break
  done
fi

if [[ -z "${DEB_PATH}" || ! -f "${DEB_PATH}" ]]; then
  echo "ERROR: no sima-pcie-host DEB found for ${deb_arch}." >&2
  echo "       Search dirs: ${search_dirs[*]}" >&2
  exit 1
fi
if [[ "${RUNTIME_ONLY}" != "ON" && -n "${DEV_DEB_PATH}" && ! -f "${DEV_DEB_PATH}" ]]; then
  echo "ERROR: sima-pcie-host-dev DEB does not exist: ${DEV_DEB_PATH}" >&2
  exit 1
fi

DEB_PATH="$(absolute_path "${DEB_PATH}")"
if [[ -n "${DEV_DEB_PATH}" ]]; then
  DEV_DEB_PATH="$(absolute_path "${DEV_DEB_PATH}")"
fi

if [[ "$(id -u)" -eq 0 ]]; then
  apt_cmd=(apt)
elif command -v sudo >/dev/null 2>&1; then
  apt_cmd=(sudo apt)
else
  echo "ERROR: root privileges or sudo are required to install sima-pcie-host packages." >&2
  exit 1
fi

echo "Installing ${DEB_PATH}"
if [[ "${RUNTIME_ONLY}" != "ON" && -n "${DEV_DEB_PATH}" ]]; then
  echo "Installing ${DEV_DEB_PATH}"
  "${apt_cmd[@]}" install -y "${DEB_PATH}" "${DEV_DEB_PATH}"
else
  if [[ "${RUNTIME_ONLY}" != "ON" ]]; then
    echo "WARN: no sima-pcie-host-dev DEB found; installing runtime package only" >&2
  fi
  "${apt_cmd[@]}" install -y "${DEB_PATH}"
fi

if command -v gst-inspect-1.0 >/dev/null 2>&1; then
  gst-inspect-1.0 neatpciehost >/dev/null 2>&1 || true
fi

if [[ "${RUN_SETUP}" == "ON" ]]; then
  if ! command -v pcie-setup.sh >/dev/null 2>&1; then
    echo "ERROR: pcie-setup.sh was not found after package install." >&2
    exit 1
  fi

  echo "Running pcie-setup.sh ${SETUP_ARGS}"
  if [[ "${SETUP_BEST_EFFORT}" == "ON" ]]; then
    # shellcheck disable=SC2086
    pcie-setup.sh ${SETUP_ARGS} || {
      echo "WARN: pcie-setup.sh failed; package installation completed." >&2
    }
  else
    # shellcheck disable=SC2086
    pcie-setup.sh ${SETUP_ARGS}
  fi
fi

echo "sima-pcie-host installed."
