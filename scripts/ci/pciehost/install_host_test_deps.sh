#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
sanitize_path

if ldconfig -p 2>/dev/null | grep -q 'libopencv_imgcodecs'; then
  echo "OpenCV runtime dependency already available."
  exit 0
fi

echo "Installing PCIe hardware test runtime dependencies"
run_host_sudo apt-get update
run_host_sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y libopencv-dev
