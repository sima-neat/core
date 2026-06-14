#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: scripts/install-local.sh [options]

Install locally built sima-pcie-host Debian packages on this host.

Options:
  --deb <path>        Install this runtime DEB instead of auto-detecting it
  --dev-deb <path>    Install this development DEB instead of auto-detecting it
  --runtime-only      Do not install sima-pcie-host-dev
  --run-setup         Run pcie-setup.sh after package install
  --setup-args <s>    Extra arguments passed to pcie-setup.sh with --run-setup
  -h, --help          Show this help

Examples:
  scripts/install-local.sh
  scripts/install-local.sh --deb packaging/sima-pcie-host_0.4.0_amd64.deb
  scripts/install-local.sh --runtime-only
  scripts/install-local.sh --run-setup --setup-args "--hosts 10.0.0.2"
EOF
}

DEB_PATH=""
DEV_DEB_PATH=""
RUNTIME_ONLY="OFF"
RUN_SETUP="OFF"
SETUP_ARGS=""

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
    --runtime-only)
      RUNTIME_ONLY="ON"
      shift
      ;;
    --run-setup)
      RUN_SETUP="ON"
      shift
      ;;
    --setup-args)
      SETUP_ARGS="${2:-}"
      shift 2
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

if [[ -z "${DEB_PATH}" ]]; then
  DEB_PATH="$(find "${ROOT_DIR}/packaging" -maxdepth 1 \
    -type f -name 'sima-pcie-host_*.deb' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | awk 'NR == 1 {print $2}')"
fi

if [[ "${RUNTIME_ONLY}" != "ON" && -z "${DEV_DEB_PATH}" ]]; then
  DEV_DEB_PATH="$(find "${ROOT_DIR}/packaging" -maxdepth 1 \
    -type f -name 'sima-pcie-host-dev_*.deb' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | awk 'NR == 1 {print $2}')"
fi

if [[ -z "${DEB_PATH}" || ! -f "${DEB_PATH}" ]]; then
  echo "ERROR: no sima-pcie-host DEB found." >&2
  echo "       Build one first with: ./build.sh" >&2
  exit 1
fi
if [[ "${RUNTIME_ONLY}" != "ON" && -n "${DEV_DEB_PATH}" && ! -f "${DEV_DEB_PATH}" ]]; then
  echo "ERROR: sima-pcie-host-dev DEB does not exist: ${DEV_DEB_PATH}" >&2
  exit 1
fi

echo "Installing ${DEB_PATH}"
if [[ "${RUNTIME_ONLY}" != "ON" && -n "${DEV_DEB_PATH}" ]]; then
  echo "Installing ${DEV_DEB_PATH}"
  sudo apt install -y "${DEB_PATH}" "${DEV_DEB_PATH}"
else
  if [[ "${RUNTIME_ONLY}" != "ON" ]]; then
    echo "WARN: no sima-pcie-host-dev DEB found; installing runtime package only" >&2
  fi
  sudo apt install -y "${DEB_PATH}"
fi

if command -v gst-inspect-1.0 >/dev/null 2>&1; then
  gst-inspect-1.0 neatpciehost >/dev/null 2>&1 || true
fi

if [[ "${RUN_SETUP}" == "ON" ]]; then
  echo "Running pcie-setup.sh ${SETUP_ARGS}"
  # shellcheck disable=SC2086
  pcie-setup.sh ${SETUP_ARGS}
fi

echo "sima-pcie-host installed."
