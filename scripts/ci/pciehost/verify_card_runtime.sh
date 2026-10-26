#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
sanitize_path

ssh_card bash -s <<'REMOTE_VERIFY'
set -e
export GST_PLUGIN_PATH_1_0=/usr/lib/aarch64-linux-gnu/neat/gst-plugins
unset GST_PLUGIN_PATH
command -v pcie-pipeline-builder
gst-inspect-1.0 neatpciesrc >/dev/null
gst-inspect-1.0 neatpciesink >/dev/null
for directory in /run/sima-neat/pcie /var/log/sima-neat/pcie; do
  if [ ! -d "${directory}" ] || [ ! -w "${directory}" ] || [ ! -x "${directory}" ]; then
    echo "PCIe runtime directory is missing or not writable/searchable: ${directory}" >&2
    echo "Reconfigure sima-neat on the card and check its tmpfiles configuration and user permissions." >&2
    exit 1
  fi
done
REMOTE_VERIFY
echo "PCIe card runtime verified."
