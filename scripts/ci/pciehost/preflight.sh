#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
sanitize_path
require_cmd sima-cli
require_cmd /usr/bin/gst-inspect-1.0
require_cmd ctest
require_cmd ssh
require_cmd tar
require_cmd find

if [[ -z "${REF_NAME}" || -z "${SHORT_SHA}" ]]; then
  echo "ERROR: REF_NAME and SHORT_SHA/GITHUB_SHA are required." >&2
  exit 1
fi

rm -rf "${WORK_DIR}"
mkdir -p "${PACKAGE_DIR}" "${EXTRAS_DIR}" "${ARTIFACT_DIR}"

ping -c 1 -W 5 "${CARD_HOST}" >/dev/null
echo "PCIe host hardware workspace prepared at ${WORK_DIR}"
