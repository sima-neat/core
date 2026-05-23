#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BUILD_DIR="${BUILD_DIR:-build-stress-gate}"
REPEAT_COUNT="${STRESS_REPEAT_COUNT:-20}"
export SIMA_STRESS_ITERS="${SIMA_STRESS_ITERS:-100}"

stress_tests=(
  stress_graph_strict_sync_join_race_test
  stress_pipeline_build_repeated_test
  stress_model_lifecycle_test
  stress_graph_lifecycle_test
  stress_graph_validate_rtsp_churn_test
  stress_run_async_pressure_test
  stress_run_try_push_holder_race_test
  stress_graph_execution_test
  stress_udp_json_burst_test
)

repeat_tests=(
  stress_graph_strict_sync_join_race_test
  stress_graph_validate_rtsp_churn_test
  stress_run_try_push_holder_race_test
  stress_udp_json_burst_test
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

stress_regex="$(join_regex stress_tests)"
repeat_regex="$(join_regex repeat_tests)"

echo "[stress-gate] configuring build in ${BUILD_DIR}..."
cmake -S . -B "${BUILD_DIR}" -DSIMANEAT_BUILD_SAMPLES=OFF

echo "[stress-gate] building stress targets..."
build_targets=(unit_modalix_contract_preflight_test "${stress_tests[@]}")
cmake --build "${BUILD_DIR}" --target "${build_targets[@]}" -j"${CMAKE_BUILD_PARALLEL_LEVEL:-8}"

echo "[stress-gate] running Modalix preflight..."
ctest --test-dir "${BUILD_DIR}" -R "^unit_modalix_contract_preflight_test$" --output-on-failure

echo "[stress-gate] running stress suite with SIMA_STRESS_ITERS=${SIMA_STRESS_ITERS}..."
ctest --test-dir "${BUILD_DIR}/tests" --output-on-failure -R "${stress_regex}"

if [[ "${REPEAT_COUNT}" -gt 1 ]]; then
  echo "[stress-gate] repeat-until-fail:${REPEAT_COUNT} for race-sensitive stress subset..."
  ctest --test-dir "${BUILD_DIR}/tests" \
    --output-on-failure \
    --repeat "until-fail:${REPEAT_COUNT}" \
    -R "${repeat_regex}"
fi

echo "[stress-gate] passed."
