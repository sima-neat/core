#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BUILD_DIR="${BUILD_DIR:-build-soak-lane}"
SOAK_DURATION_HOURS="${SOAK_DURATION_HOURS:-2}"
SOAK_INTERVAL_SECONDS="${SOAK_INTERVAL_SECONDS:-900}"
SOAK_LEAK_STEP_KB="${SOAK_LEAK_STEP_KB:-32768}"
SOAK_MONOTONIC_LIMIT="${SOAK_MONOTONIC_LIMIT:-3}"
SOAK_REPORT="${SOAK_REPORT:-${BUILD_DIR}/soak_report.csv}"

SOAK_TESTS=(
  stress_graph_execution_test
  stress_run_try_push_holder_race_test
  stress_session_validate_rtsp_churn_test
  stress_graph_strict_sync_join_race_test
)

join_regex() {
  local -n arr_ref=$1
  local regex=""
  local t
  for t in "${arr_ref[@]}"; do
    if [[ -n "${regex}" ]]; then
      regex+="|"
    fi
    regex+="${t}"
  done
  printf '^(%s)$\n' "${regex}"
}

regex="$(join_regex SOAK_TESTS)"

echo "[soak-lane] configuring build in ${BUILD_DIR}..."
cmake -S . -B "${BUILD_DIR}" -DSIMANEAT_BUILD_SAMPLES=OFF

echo "[soak-lane] building soak stress targets..."
build_targets=(unit_modalix_contract_preflight_test "${SOAK_TESTS[@]}")
cmake --build "${BUILD_DIR}" --target "${build_targets[@]}" -j"${CMAKE_BUILD_PARALLEL_LEVEL:-8}"

echo "[soak-lane] running Modalix preflight..."
ctest --test-dir "${BUILD_DIR}" -R "^unit_modalix_contract_preflight_test$" --output-on-failure

mkdir -p "$(dirname "${SOAK_REPORT}")"
echo "timestamp,vmrss_kb,threads,fd_count,iteration" > "${SOAK_REPORT}"

export SIMA_STRESS_ITERS="${SIMA_STRESS_ITERS:-200}"

start_epoch="$(date +%s)"
end_epoch="$(python3 - <<'PY'
import time
import os
hours = float(os.environ.get("SOAK_DURATION_HOURS", "2"))
print(int(time.time() + hours * 3600))
PY
)"

iteration=0
prev_rss=0
monotonic_growth=0

echo "[soak-lane] running until $(date -d "@${end_epoch}" +"%Y-%m-%d %H:%M:%S") ..."

while [[ "$(date +%s)" -lt "${end_epoch}" ]]; do
  iteration=$((iteration + 1))
  timestamp="$(date +"%Y-%m-%dT%H:%M:%S")"

  vmrss_kb="$(awk '/VmRSS/ {print $2}' /proc/$$/status 2>/dev/null || echo 0)"
  threads="$(awk '/Threads/ {print $2}' /proc/$$/status 2>/dev/null || echo 0)"
  fd_count="$(ls /proc/$$/fd 2>/dev/null | wc -l | tr -d ' ')"

  echo "${timestamp},${vmrss_kb},${threads},${fd_count},${iteration}" >> "${SOAK_REPORT}"
  echo "[soak-lane] iteration=${iteration} vmrss_kb=${vmrss_kb} threads=${threads} fd_count=${fd_count}"

  ctest --test-dir "${BUILD_DIR}/tests" --output-on-failure -R "${regex}"

  if [[ "${prev_rss}" -gt 0 ]] && [[ "${vmrss_kb}" -gt $((prev_rss + SOAK_LEAK_STEP_KB)) ]]; then
    monotonic_growth=$((monotonic_growth + 1))
  else
    monotonic_growth=0
  fi
  prev_rss="${vmrss_kb}"

  if [[ "${monotonic_growth}" -ge "${SOAK_MONOTONIC_LIMIT}" ]]; then
    echo "ERROR: potential monotonic leak trend detected (vmrss growth over ${SOAK_MONOTONIC_LIMIT} intervals)." >&2
    exit 1
  fi

  now_epoch="$(date +%s)"
  remaining=$((end_epoch - now_epoch))
  if [[ "${remaining}" -le 0 ]]; then
    break
  fi
  sleep_for="${SOAK_INTERVAL_SECONDS}"
  if [[ "${sleep_for}" -gt "${remaining}" ]]; then
    sleep_for="${remaining}"
  fi
  sleep "${sleep_for}"
done

echo "[soak-lane] completed. report=${SOAK_REPORT}"
