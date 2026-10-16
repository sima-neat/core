#!/usr/bin/env bash
# Reuse the installed canonical test registration and owner; never build/launch
# a second, unguarded native suite here. Pass the exact canonical CTest argv.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${COVERAGE_DIR:?Set a fresh per-lane COVERAGE_DIR}"
exec python3 "${ROOT_DIR}/tools/ci/check_test_coverage.py" \
  --profile "${COVERAGE_PROFILE:-strict}" --require-guarded \
  --output-dir "${COVERAGE_DIR}" -- "$@"
