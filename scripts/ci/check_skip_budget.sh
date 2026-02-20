#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BASELINE_FILE="${BASELINE_FILE:-tests/metrics/skip_budget_baseline.txt}"
TARGET_MAX_STRICT="${TARGET_MAX_STRICT:-9}"
OUT_DIR="${OUT_DIR:-build-zero-skip}"
OUT_JSON="${OUT_JSON:-${OUT_DIR}/skip_trend.json}"
mkdir -p "${OUT_DIR}"

if [[ ! -f "${BASELINE_FILE}" ]]; then
  echo "[skip-budget] missing baseline file: ${BASELINE_FILE}" >&2
  exit 1
fi

BASELINE_COUNT="$(tr -d '[:space:]' < "${BASELINE_FILE}")"
if [[ -z "${BASELINE_COUNT}" ]]; then
  echo "[skip-budget] baseline file is empty: ${BASELINE_FILE}" >&2
  exit 1
fi

TOTAL_SKIP_CALLS="$(grep -R "skip_test(\\|skip_test_exception(\\|skip_long_test(\\|skip_long_test_exception(\\|SkipTest" \
  -n tests \
  --include='*.cpp' \
  --include='*.h' \
  | wc -l | tr -d '[:space:]')"

STRICT_SKIP_CALLS="$(
  {
    grep -R "skip_test(\\|skip_test_exception(\\|SkipTest" \
      -n tests \
      --include='*.cpp' \
      --include='*.h' \
      | grep -vE "tests/test_utils\\.h|tests/test_main\\.h|tests/stress/|tests/e2e_pipelines/|tests/e2e_testing/modelzoo_bulk_models_test\\.cpp" \
      || true
  } | wc -l | tr -d '[:space:]'
)"

DELTA="$((STRICT_SKIP_CALLS - BASELINE_COUNT))"

cat > "${OUT_JSON}" <<JSON
{
  "baseline_strict": ${BASELINE_COUNT},
  "current_strict": ${STRICT_SKIP_CALLS},
  "target_max_strict": ${TARGET_MAX_STRICT},
  "current_total": ${TOTAL_SKIP_CALLS},
  "delta": ${DELTA},
  "legacy_non_long": ${STRICT_SKIP_CALLS}
}
JSON

printf '[skip-budget] baseline_strict=%s current_strict=%s target_max_strict=%s total=%s delta=%s\n' \
  "${BASELINE_COUNT}" "${STRICT_SKIP_CALLS}" "${TARGET_MAX_STRICT}" "${TOTAL_SKIP_CALLS}" "${DELTA}"

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "## Skip Budget"
    echo
    echo "| Metric | Value |"
    echo "|---|---:|"
    echo "| Strict Baseline | ${BASELINE_COUNT} |"
    echo "| Strict Current | ${STRICT_SKIP_CALLS} |"
    echo "| Strict Target Max | ${TARGET_MAX_STRICT} |"
    echo "| Total Callsites (incl. long) | ${TOTAL_SKIP_CALLS} |"
    echo "| Delta (strict) | ${DELTA} |"
    echo
  } >> "${GITHUB_STEP_SUMMARY}"
fi

if (( STRICT_SKIP_CALLS > BASELINE_COUNT )); then
  echo "[skip-budget] FAIL: strict skip callsite count increased" >&2
  exit 1
fi

if (( STRICT_SKIP_CALLS > TARGET_MAX_STRICT )); then
  echo "[skip-budget] FAIL: strict skip callsites exceed target (<10 required)" >&2
  exit 1
fi

echo "[skip-budget] PASS"
