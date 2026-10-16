#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cli="$(command -v "${SIMA_CLI_BIN:-sima-cli}")"
cli="$(readlink -f "${cli}")"
python="${SIMA_CLI_PYTHON:-$(dirname "${cli}")/python}"
if [[ ! -x "${python}" ]]; then
  echo "Cannot locate selected sima-cli Python; set SIMA_CLI_PYTHON explicitly." >&2
  exit 1
fi
exec "${python}" "${root}/tools/ci/verify_core_dist.py" "$@"
