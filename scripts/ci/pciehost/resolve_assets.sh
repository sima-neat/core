#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
sanitize_path

"${WORKSPACE}/pcie_host/scripts/resolve_hardware_test_assets.sh" \
  --workspace "${WORKSPACE}" \
  --extras-root "${EXTRAS_DIR}" \
  --cache-dir "${WORK_DIR}/assets" \
  --env-file "${ASSET_ENV_FILE}"

# shellcheck disable=SC1090
source "${ASSET_ENV_FILE}"
export SIMAPCIE_YOLOV8_MODEL SIMAPCIE_TEST_IMAGE SIMAPCIE_BOXDECODE_IMAGE
echo "Resolved PCIe hardware test assets."
