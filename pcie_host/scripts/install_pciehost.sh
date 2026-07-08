#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: install_pciehost.sh [options]

Install sima-pcie-host Debian packages on this host, then provision PCIe SSH.

Options:
  --deb <path>             Install this runtime DEB instead of auto-detecting it
  --dev-deb <path>         Install this development DEB instead of auto-detecting it
  --search-dir <dir>       Directory to search for DEBs (default: this script's directory)
  --python                 Install pyneatpcie wheel into a Python venv
  --python-wheel <path>    Install this pyneatpcie wheel instead of auto-detecting it
  --python-venv <dir>      Python venv path (default: ~/pyneatpcie)
  --runtime-only           Do not install sima-pcie-host-dev
  --skip-setup             Do not run pcie-setup.sh after package install
  --setup-args <s>         Extra arguments passed to pcie-setup.sh
  --setup-best-effort      Do not fail installation if pcie-setup.sh fails
  -h, --help               Show this help

Examples:
  ./install_pciehost.sh
  ./install_pciehost.sh --deb sima-pcie-host_<version>_amd64.deb
  ./install_pciehost.sh --runtime-only
  ./install_pciehost.sh --python
  ./install_pciehost.sh --setup-args "--hosts 10.0.0.2"
  ./install_pciehost.sh --skip-setup

Environment:
  SIMAPCIE_INSTALL_PYTHON=1     Same as --python
  SIMAPCIE_PYTHON_WHEEL=<path>  Same as --python-wheel
  SIMAPCIE_PYTHON_VENV=<dir>    Same as --python-venv
  SIMAPCIE_RUNTIME_ONLY=1       Same as --runtime-only
  SIMAPCIE_SKIP_SETUP=1         Same as --skip-setup
  SIMAPCIE_SETUP_ARGS="..."     Same as --setup-args
  SIMAPCIE_SETUP_BEST_EFFORT=1  Same as --setup-best-effort
  SUDO_PASSWORD=<password>       sudo password for non-interactive installs
  DEVKIT_PASSWORD=<password>     fallback sudo password for CI/devkit hosts
                                  If neither is set, sudo may prompt interactively.
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
PYTHON_WHEEL_PATH=""
PYTHON_VENV=""
SEARCH_DIR=""
INSTALL_PYTHON="OFF"
RUNTIME_ONLY="OFF"
RUN_SETUP="ON"
SETUP_ARGS=""
SETUP_BEST_EFFORT="OFF"

if env_truthy "${SIMAPCIE_RUNTIME_ONLY:-}"; then
  RUNTIME_ONLY="ON"
fi
if env_truthy "${SIMAPCIE_INSTALL_PYTHON:-}"; then
  INSTALL_PYTHON="ON"
fi
if [[ -n "${SIMAPCIE_PYTHON_WHEEL:-}" ]]; then
  PYTHON_WHEEL_PATH="${SIMAPCIE_PYTHON_WHEEL}"
fi
if [[ -n "${SIMAPCIE_PYTHON_VENV:-}" ]]; then
  PYTHON_VENV="${SIMAPCIE_PYTHON_VENV}"
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
    --python)
      INSTALL_PYTHON="ON"
      shift
      ;;
    --python-wheel)
      PYTHON_WHEEL_PATH="${2:-}"
      if [[ -z "${PYTHON_WHEEL_PATH}" ]]; then
        echo "ERROR: --python-wheel requires a value" >&2
        exit 1
      fi
      INSTALL_PYTHON="ON"
      shift 2
      ;;
    --python-venv)
      PYTHON_VENV="${2:-}"
      if [[ -z "${PYTHON_VENV}" ]]; then
        echo "ERROR: --python-venv requires a value" >&2
        exit 1
      fi
      INSTALL_PYTHON="ON"
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

latest_matching_file() {
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

run_sudo() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
    return
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    echo "ERROR: root privileges or sudo are required to install sima-pcie-host packages." >&2
    return 1
  fi

  if sudo -n true >/dev/null 2>&1; then
    sudo -n "$@"
    return
  fi

  local pw="${SUDO_PASSWORD:-${DEVKIT_PASSWORD:-}}"
  if [[ -z "${pw}" ]]; then
    sudo "$@"
    return
  fi

  if ! printf '%s\n' "${pw}" | sudo -S -p '' -v >/dev/null 2>&1; then
    echo "ERROR: sudo authentication failed." >&2
    return 1
  fi

  printf '%s\n' "${pw}" | sudo -S -p '' "$@"
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
  search_dirs=("${SCRIPT_DIR}")
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

if [[ "${INSTALL_PYTHON}" == "ON" && -z "${PYTHON_WHEEL_PATH}" ]]; then
  for dir in "${search_dirs[@]}"; do
    [[ -d "${dir}" ]] || continue
    PYTHON_WHEEL_PATH="$(latest_matching_file "${dir}" "pyneatpcie-*.whl")"
    [[ -n "${PYTHON_WHEEL_PATH}" ]] && break
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
if [[ "${INSTALL_PYTHON}" == "ON" ]]; then
  if [[ -z "${PYTHON_WHEEL_PATH}" || ! -f "${PYTHON_WHEEL_PATH}" ]]; then
    echo "ERROR: no pyneatpcie wheel found." >&2
    echo "       Search dirs: ${search_dirs[*]}" >&2
    exit 1
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is required to install pyneatpcie." >&2
    exit 1
  fi
  if [[ -z "${PYTHON_VENV}" ]]; then
    PYTHON_VENV="${HOME}/pyneatpcie"
  fi
fi

DEB_PATH="$(absolute_path "${DEB_PATH}")"
if [[ -n "${DEV_DEB_PATH}" ]]; then
  DEV_DEB_PATH="$(absolute_path "${DEV_DEB_PATH}")"
fi
if [[ -n "${PYTHON_WHEEL_PATH}" ]]; then
  PYTHON_WHEEL_PATH="$(absolute_path "${PYTHON_WHEEL_PATH}")"
fi

apt_install_opts=(install -y --reinstall --allow-downgrades)

echo "Installing ${DEB_PATH}"
if [[ "${RUNTIME_ONLY}" != "ON" && -n "${DEV_DEB_PATH}" ]]; then
  echo "Installing ${DEV_DEB_PATH}"
  run_sudo apt "${apt_install_opts[@]}" "${DEB_PATH}" "${DEV_DEB_PATH}"
else
  if [[ "${RUNTIME_ONLY}" != "ON" ]]; then
    echo "WARN: no sima-pcie-host-dev DEB found; installing runtime package only" >&2
  fi
  run_sudo apt "${apt_install_opts[@]}" "${DEB_PATH}"
fi

if command -v gst-inspect-1.0 >/dev/null 2>&1; then
  gst-inspect-1.0 neatpciehost >/dev/null 2>&1 || true
fi

if [[ "${INSTALL_PYTHON}" == "ON" ]]; then
  echo "Installing ${PYTHON_WHEEL_PATH} into ${PYTHON_VENV}"
  if [[ ! -x "${PYTHON_VENV}/bin/python" ]]; then
    python3 -m venv --system-site-packages "${PYTHON_VENV}"
  fi
  "${PYTHON_VENV}/bin/python" -m pip install --upgrade pip
  "${PYTHON_VENV}/bin/python" -m pip install --no-deps --force-reinstall "${PYTHON_WHEEL_PATH}"
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

if [[ "${INSTALL_PYTHON}" == "ON" ]]; then
  echo "sima-pcie-host installed. Python venv: ${PYTHON_VENV}"
else
  echo "sima-pcie-host installed."
fi
