#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BUILD_DIR="${BUILD_DIR:-build-mpk-security-gate}"
SAN_BUILD_DIR="${SAN_BUILD_DIR:-build-mpk-security-asan-ubsan}"

mpk_tests=(
  unit_mpk_loader_test
  unit_mpk_pipeline_adapter_test
  unit_pipeline_sequence_parse_test
  unit_modelpack_parse_test
  unit_modelpack_extract_test
  security_modelpack_archive_test
  security_modelpack_json_test
  security_modelpack_unicode_path_test
  security_modelpack_tar_integrity_test
  security_modelpack_json_robustness_test
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

regex="$(join_regex mpk_tests)"

echo "[mpk-security-gate] checking deterministic MPK fixtures..."
python3 tests/tools/make_mpk_fixtures.py --check

echo "[mpk-security-gate] configuring baseline build in ${BUILD_DIR}..."
cmake -S . -B "${BUILD_DIR}" -DSIMANEAT_BUILD_SAMPLES=OFF

echo "[mpk-security-gate] building MPK/security tests..."
baseline_targets=(unit_modalix_contract_preflight_test "${mpk_tests[@]}")
cmake --build "${BUILD_DIR}" --target "${baseline_targets[@]}" -j"${CMAKE_BUILD_PARALLEL_LEVEL:-8}"

echo "[mpk-security-gate] running Modalix preflight..."
ctest --test-dir "${BUILD_DIR}" --output-on-failure -R "^unit_modalix_contract_preflight_test$"

echo "[mpk-security-gate] running baseline MPK/security tests..."
ctest --test-dir "${BUILD_DIR}/tests" --output-on-failure -R "${regex}"

if [[ "${MPK_SECURITY_SKIP_SANITIZERS:-0}" == "1" ]]; then
  echo "[mpk-security-gate] skipping sanitizer subset (MPK_SECURITY_SKIP_SANITIZERS=1)."
  echo "[mpk-security-gate] passed."
  exit 0
fi

echo "[mpk-security-gate] configuring ASan/UBSan subset in ${SAN_BUILD_DIR}..."
cmake -S . -B "${SAN_BUILD_DIR}" \
  -DSIMANEAT_BUILD_SAMPLES=OFF \
  -DSIMA_ENABLE_ASAN=ON \
  -DSIMA_ENABLE_UBSAN=ON \
  -DSIMA_ENABLE_TSAN=OFF

export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0:abort_on_error=1:halt_on_error=1}"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=1}"

echo "[mpk-security-gate] building sanitizer MPK/security tests..."
san_targets=(unit_modalix_contract_preflight_test "${mpk_tests[@]}")
cmake --build "${SAN_BUILD_DIR}" --target "${san_targets[@]}" -j"${CMAKE_BUILD_PARALLEL_LEVEL:-8}"

echo "[mpk-security-gate] running sanitizer Modalix preflight..."
ctest --test-dir "${SAN_BUILD_DIR}" --output-on-failure -R "^unit_modalix_contract_preflight_test$"

echo "[mpk-security-gate] running sanitizer MPK/security tests..."
ctest --test-dir "${SAN_BUILD_DIR}/tests" --output-on-failure -R "${regex}"

echo "[mpk-security-gate] passed."
