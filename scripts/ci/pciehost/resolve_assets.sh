#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
sanitize_path

# Prints the extracted MPK path of one Model Browser package.
model_browser_mpk() {
  python3 "${WORKSPACE}/scripts/ci/prepare_model_browser_fixture.py" \
    --id "$1" --variant "$2" --output "${WORK_DIR}/assets"
}

# An explicit model variable still wins over the Model Browser package.
if [[ -z "${SIMAPCIE_YOLOV8_MODEL:-${SIMA_MODEL_TAR:-${SIMA_YOLO_TAR:-}}}" ]]; then
  SIMAPCIE_YOLOV8_MODEL="$(model_browser_mpk yolov8s w8a8)"
  export SIMAPCIE_YOLOV8_MODEL
fi

"${WORKSPACE}/pcie_host/scripts/resolve_hardware_test_assets.sh" \
  --workspace "${WORKSPACE}" \
  --extras-root "${EXTRAS_DIR}" \
  --cache-dir "${WORK_DIR}/assets" \
  --env-file "${ASSET_ENV_FILE}"

# The mla_only tests run one INT8 and one BF16 build.
MLA_ONLY_MODEL="$(model_browser_mpk yolo26n w8a8)"
MLA_ONLY_BF16_MODEL="$(model_browser_mpk yolo26n w16a16)"
echo "SIMAPCIE_MLA_ONLY_MODEL=${MLA_ONLY_MODEL}" >>"${ASSET_ENV_FILE}"
echo "SIMAPCIE_MLA_ONLY_BF16_MODEL=${MLA_ONLY_BF16_MODEL}" >>"${ASSET_ENV_FILE}"

# shellcheck disable=SC1090
source "${ASSET_ENV_FILE}"
export SIMAPCIE_YOLOV8_MODEL SIMAPCIE_MLA_ONLY_MODEL SIMAPCIE_MLA_ONLY_BF16_MODEL SIMAPCIE_TEST_IMAGE SIMAPCIE_BOXDECODE_IMAGE
echo "Resolved PCIe hardware test assets."
