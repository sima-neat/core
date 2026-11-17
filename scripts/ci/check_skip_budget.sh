#!/usr/bin/env bash
# Advisory source trend only. Actual zero-skip enforcement lives in the
# canonical lane's registration/result accounting (run_zero_skip_gate.sh).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
OUT_DIR="${OUT_DIR:-build-zero-skip}"
mkdir -p "${OUT_DIR}"
TOTAL="$( { grep -RE 'skip_test\(|skip_test_exception\(|skip_long_test\(|skip_long_test_exception\(|SkipTest' tests --include='*.cpp' --include='*.h' || true; } | wc -l)"
printf '{"advisory_source_callsites": %s, "runtime_skips": null, "qualification": false}\n' "${TOTAL}" > "${OUT_JSON:-${OUT_DIR}/skip_trend.json}"
echo "[skip-trend] ${TOTAL} source references; not skipped executions or a qualification gate."
