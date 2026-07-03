#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
sanitize_path

package_spec="core/pciehost/amd64@${REF_NAME}:${SHORT_SHA}"
echo "Installing PCIe host package ${package_spec} from Vulcan env ${VULCAN_ENV}"
SIMA_CLI_CHECK_FOR_UPDATE=0 \
  sima-cli neat install \
    --env "${VULCAN_ENV}" \
    --install-dir "${PACKAGE_DIR}" \
    "${package_spec}"

/usr/bin/gst-inspect-1.0 neatpciehost >/dev/null
echo "PCIe host package installed and neatpciehost is inspectable."
