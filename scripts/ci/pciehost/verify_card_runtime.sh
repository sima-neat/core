#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
sanitize_path

ssh_card "set -e; \
  command -v pcie-pipeline-builder; \
  gst-inspect-1.0 neatpciesrc >/dev/null; \
  gst-inspect-1.0 neatpciesink >/dev/null"
echo "PCIe card runtime verified."
