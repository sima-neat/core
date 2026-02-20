#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
UNIT_NAME="simaai-appcomplex-workspace.service"
OLD_UNIT_NAME="simaai-appcomplex.service"
ENV_FILE="/etc/default/simaai-appcomplex-workspace"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--deb <path>] [--build] [--activate] [--switch-system]

Options:
  --deb <path>        Install an existing .deb package
  --build             Build package before install (default if --deb omitted)
  --activate          Enable/start workspace unit after install
  --switch-system     Stop/disable $OLD_UNIT_NAME after install
EOF
}

require_sudo() {
  if [ "$(id -u)" -ne 0 ]; then
    if ! command -v sudo >/dev/null 2>&1; then
      echo "sudo is required for install operations" >&2
      exit 1
    fi
  fi
}

run_as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

DEB_PATH=""
DO_BUILD=0
DO_ACTIVATE=0
DO_SWITCH=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --deb)
      DEB_PATH="${2:-}"
      shift 2
      ;;
    --build)
      DO_BUILD=1
      shift
      ;;
    --activate)
      DO_ACTIVATE=1
      shift
      ;;
    --switch-system)
      DO_SWITCH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [ -z "$DEB_PATH" ]; then
  DO_BUILD=1
fi

if [ "$DO_BUILD" -eq 1 ]; then
  DEB_PATH="$("$ROOT_DIR/scripts/release/build_appcomplex_workspace_deb.sh" | tail -n 1)"
fi

if [ ! -f "$DEB_PATH" ]; then
  echo "Package not found: $DEB_PATH" >&2
  exit 1
fi

require_sudo

echo "[install] Installing $DEB_PATH"
run_as_root dpkg -i "$DEB_PATH"

if [ "$DO_SWITCH" -eq 1 ]; then
  run_as_root sed -i 's/^APP_COMPLEX_SWITCH_SYSTEM_SERVICE=.*/APP_COMPLEX_SWITCH_SYSTEM_SERVICE=1/' "$ENV_FILE"
fi

if [ "$DO_ACTIVATE" -eq 1 ]; then
  run_as_root sed -i 's/^APP_COMPLEX_ACTIVATE_ON_INSTALL=.*/APP_COMPLEX_ACTIVATE_ON_INSTALL=1/' "$ENV_FILE"
  run_as_root systemctl daemon-reload
  run_as_root systemctl enable "$UNIT_NAME"
  run_as_root systemctl restart "$UNIT_NAME"
fi

if [ "$DO_SWITCH" -eq 1 ]; then
  run_as_root systemctl stop "$OLD_UNIT_NAME" || true
  run_as_root systemctl disable "$OLD_UNIT_NAME" || true
fi

echo "[install] Completed"
echo "[install] Workspace unit status:"
run_as_root systemctl is-active "$UNIT_NAME" || true
echo "[install] System unit status:"
run_as_root systemctl is-active "$OLD_UNIT_NAME" || true
