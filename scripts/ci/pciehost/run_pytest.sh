#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
sanitize_path

load_state
python_venv="${SIMAPCIE_PYTHON_VENV:-${HOME}/pyneatpcie}"
python_bin="${SIMAPCIE_PYTHON_BIN:-${python_venv}/bin/python}"
pytest_dir="${WORKSPACE}/pcie_host/python/tests"

if [[ ! -x "${python_bin}" ]]; then
  echo "ERROR: expected pyneatpcie Python at ${python_bin}" >&2
  echo "       The PCIe host package should install pyneatpcie via install_pciehost.sh --python." >&2
  exit 1
fi
if [[ ! -d "${pytest_dir}" ]]; then
  echo "ERROR: pyneatpcie pytest directory not found: ${pytest_dir}" >&2
  exit 1
fi

"${python_bin}" - <<'PY'
import pyneatpcie
print(f"pyneatpcie={pyneatpcie.__version__}")
PY

"${python_bin}" -m pip install pytest pillow numpy

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

"${python_bin}" -m pytest -q "${pytest_dir}"
