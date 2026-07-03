#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
sanitize_path

load_state
if [[ -z "${TEST_DIR:-}" || ! -d "${TEST_DIR}" ]]; then
  echo "ERROR: PCIe host CTest directory is not available. Run extract_extras.sh first." >&2
  exit 1
fi

export SIMAPCIE_CARD_HOST="${CARD_HOST}"
export SIMAPCIE_USER="${CARD_USER}"
export SIMAPCIE_CARD_ID="${CARD_ID}"
export SIMAPCIE_QUEUE="${QUEUE}"
export SIMAPCIE_READINESS_TIMEOUT_MS="${READINESS_TIMEOUT_MS}"
export SIMAPCIE_PULL_TIMEOUT_MS="${PULL_TIMEOUT_MS}"
export SIMAPCIE_CARD_GST_DEBUG="${SIMAPCIE_CARD_GST_DEBUG:-}"
export SIMAPCIE_CARD_GST_DEBUG_FILE="${SIMAPCIE_CARD_GST_DEBUG_FILE:-/tmp/q${QUEUE}-card.gst.log}"
export SIMAPCIE_SYNC_ITERATIONS="${SYNC_ITERATIONS}"
export SIMAPCIE_TEST_ITERATIONS="${TEST_ITERATIONS}"
export SIMAPCIE_STRESS_QUEUES="${STRESS_QUEUES}"
export SIMAPCIE_STRESS_ITERATIONS="${STRESS_ITERATIONS}"

ctest \
  --test-dir "${TEST_DIR}" \
  --output-on-failure \
  --no-tests=error
