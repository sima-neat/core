#!/usr/bin/env bash
# Prove the mla_only route receives through the plugin's driver mapping: every returned DATA
# frame must show up as `host-read-mapped` in the neatpciehost trace, and none when the
# SIMA_PCIE_HOST_RX_MODE=copy override is set. Skips (77) without an MLA-only capable model or
# with a plugin that has no rx-mode property.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${SIMAPCIE_MLA_INT8_MODEL:-${SIMAPCIE_YOLOV8_MODEL:-}}"
LOG_DIR="${SIMAPCIE_STRESS_LOG_DIR:-${TMPDIR:-/tmp}/sima-pcie-host-single-copy-$$}"

if [[ -z "${MODEL}" || ! -f "${MODEL}" ]]; then
  echo "SKIP: SIMAPCIE_MLA_INT8_MODEL / SIMAPCIE_YOLOV8_MODEL is not set to a readable model"
  exit 77
fi
if [[ ! -x "${SCRIPT_DIR}/test_tensor_mla_only_run" ]]; then
  echo "ERROR: missing executable: ${SCRIPT_DIR}/test_tensor_mla_only_run" >&2
  exit 1
fi
if ! gst-inspect-1.0 neatpciehost 2>/dev/null | grep -q '^  rx-mode'; then
  echo "SKIP: the installed neatpciehost has no rx-mode property"
  exit 77
fi

rm -rf "${LOG_DIR}"
mkdir -p "${LOG_DIR}"

run_mode() {
  local mode="$1"
  local log="${LOG_DIR}/${mode}.log"
  local status=0

  echo "rx-mode=${mode}" >&2
  SIMA_PCIE_HOST_RX_MODE="${mode}" GST_DEBUG=neatpciehost:4 GST_DEBUG_FILE="${log}" \
    "${SCRIPT_DIR}/test_tensor_mla_only_run" --model "${MODEL}" >"${LOG_DIR}/${mode}.out" 2>&1 ||
    status=$?
  if [[ "${status}" -ne 0 ]]; then
    cat "${LOG_DIR}/${mode}.out"
    exit "${status}"
  fi
  grep -c 'host-read-mapped' "${log}" || true
}

mapped="$(run_mode mapped)"
copied="$(run_mode copy)"
echo "  mapped reads: rx-mode=mapped ${mapped}, rx-mode=copy ${copied}"
if [[ "${mapped}" -eq 0 ]]; then
  echo "ERROR: rx-mode=mapped delivered no mapped DATA frame" >&2
  exit 1
fi
if [[ "${copied}" -ne 0 ]]; then
  echo "ERROR: SIMA_PCIE_HOST_RX_MODE=copy still delivered mapped frames" >&2
  exit 1
fi
echo "done"
