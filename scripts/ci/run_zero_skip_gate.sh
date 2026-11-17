#!/usr/bin/env bash
# Account for the installed canonical lane without changing its execution mode.
# Pass the exact canonical CTest argv; guarded qualification remains opt-in.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${COVERAGE_DIR:?Set a fresh per-lane COVERAGE_DIR}"
exec python3 "${ROOT_DIR}/tools/ci/check_test_coverage.py" \
  --profile "${COVERAGE_PROFILE:-strict}" \
  --output-dir "${COVERAGE_DIR}" -- "$@"
