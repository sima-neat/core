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

# The Model Zoo yolo_v8s build tessellates on the CVU and cannot serve mla_only; yolo26n INT8 can.
(cd "${WORK_DIR}/assets" && SIMA_CLI_CHECK_FOR_UPDATE=0 sima-cli download \
  "https://docs.sima.ai/pkg_downloads/SDK2.1.3/models/modalix/yolo26-detection/yolo26n-det-int8-b1.tar.gz")
echo "SIMAPCIE_MLA_ONLY_MODEL=${WORK_DIR}/assets/yolo26n-det-int8-b1.tar.gz" >>"${ASSET_ENV_FILE}"
(cd "${WORK_DIR}/assets" && SIMA_CLI_CHECK_FOR_UPDATE=0 sima-cli download \
  "https://docs.sima.ai/pkg_downloads/SDK2.1.3/models/modalix/yolo26-detection/yolo26n-det-bf16-mla_tess-b1.tar.gz")
echo "SIMAPCIE_MLA_ONLY_BF16_MODEL=${WORK_DIR}/assets/yolo26n-det-bf16-mla_tess-b1.tar.gz" >>"${ASSET_ENV_FILE}"

# shellcheck disable=SC1090
source "${ASSET_ENV_FILE}"
export SIMAPCIE_YOLOV8_MODEL SIMAPCIE_MLA_ONLY_MODEL SIMAPCIE_MLA_ONLY_BF16_MODEL SIMAPCIE_TEST_IMAGE SIMAPCIE_BOXDECODE_IMAGE
echo "Resolved PCIe hardware test assets."
