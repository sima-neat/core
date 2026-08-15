#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
sanitize_path

host_distro="$(simapcie_detect_host_distro)"
if [[ -n "${SIMAPCIE_EXPECTED_HOST_DISTRO:-}" &&
  "${host_distro}" != "${SIMAPCIE_EXPECTED_HOST_DISTRO}" ]]; then
  echo "ERROR: detected host distro ${host_distro} does not match expected ${SIMAPCIE_EXPECTED_HOST_DISTRO}." >&2
  exit 1
fi

package_spec="core/pciehost/${host_distro}/amd64@${REF_NAME}:${SHORT_SHA}"
echo "Installing PCIe host package ${package_spec} from Vulcan env ${VULCAN_ENV}"
SIMA_CLI_CHECK_FOR_UPDATE=0 \
  sima-cli neat install \
    --env "${VULCAN_ENV}" \
    --install-dir "${PACKAGE_DIR}" \
    "${package_spec}"

/usr/bin/gst-inspect-1.0 neatpciehost >/dev/null
echo "PCIe host package installed and neatpciehost is inspectable."
