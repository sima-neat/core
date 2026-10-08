#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ssh_card \
  "REMOTE_DEVKIT_PASSWORD=$(shell_quote "${DEVKIT_PASSWORD:-}") \
   REMOTE_SUDO_PASSWORD=$(shell_quote "${DEVKIT_PASSWORD:-}") \
   bash -s" <<REMOTE_BOOTSTRAP
set -euo pipefail
export DEVKIT_PASSWORD="\${REMOTE_DEVKIT_PASSWORD}"
export SUDO_PASSWORD="\${REMOTE_SUDO_PASSWORD}"
$(remote_sudo_wrapper_script)
setup_remote_sudo_wrapper
trap 'rm -rf "\${REMOTE_SUDO_WRAPPER_DIR:-}" "\${venv_probe:-}"' EXIT

venv_probe="\$(mktemp -d /tmp/sima-neat-venv-probe.XXXXXX)"
if python3 -m venv "\${venv_probe}" >/dev/null 2>&1; then
  echo "PCIe card Python venv support is available."
  exit 0
fi
rm -rf "\${venv_probe}"
venv_probe=""

python_venv_package="\$(python3 -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}-venv")')"
echo "Installing missing PCIe card Python venv support: \${python_venv_package}"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
if ! sudo apt-get install -y "\${python_venv_package}"; then
  echo "Version-specific package unavailable; trying python3-venv." >&2
  sudo apt-get install -y python3-venv
fi

venv_probe="\$(mktemp -d /tmp/sima-neat-venv-probe.XXXXXX)"
python3 -m venv "\${venv_probe}"
echo "PCIe card Python venv support is ready."
REMOTE_BOOTSTRAP
