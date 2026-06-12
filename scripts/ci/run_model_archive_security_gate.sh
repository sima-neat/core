#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BUILD_DIR="${BUILD_DIR:-build-model-archive-security-gate}"
SAN_BUILD_DIR="${SAN_BUILD_DIR:-build-model-archive-security-asan-ubsan}"

archive_security_tests=(
  unit_model_archive_loader_test
  unit_modelpack_extract_test
  security_modelpack_matrix_test
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

regex="$(join_regex archive_security_tests)"

echo "[model-archive-security-gate] configuring baseline build in ${BUILD_DIR}..."
cmake -S . -B "${BUILD_DIR}" -DSIMANEAT_BUILD_SAMPLES=OFF

echo "[model-archive-security-gate] building model archive/security tests..."
baseline_targets=(unit_modalix_contract_preflight_test "${archive_security_tests[@]}")
cmake --build "${BUILD_DIR}" --target "${baseline_targets[@]}" -j"${CMAKE_BUILD_PARALLEL_LEVEL:-8}"

echo "[model-archive-security-gate] checking deterministic model archive fixtures..."
python3 tests/tools/make_model_archive_fixtures.py \
  --root "${BUILD_DIR}/test-assets/model-archive" \
  --manifest "${BUILD_DIR}/test-assets/model-archive/fixtures_manifest.json" \
  --check

echo "[model-archive-security-gate] running Modalix preflight..."
ctest --test-dir "${BUILD_DIR}" --output-on-failure -R "^unit_modalix_contract_preflight_test$"

echo "[model-archive-security-gate] running baseline model archive/security tests..."
ctest --test-dir "${BUILD_DIR}/tests" --output-on-failure -R "${regex}"

if [[ "${MODEL_ARCHIVE_SECURITY_SKIP_SANITIZERS:-0}" == "1" ]]; then
  echo "[model-archive-security-gate] skipping sanitizer subset (MODEL_ARCHIVE_SECURITY_SKIP_SANITIZERS=1)."
  echo "[model-archive-security-gate] passed."
  exit 0
fi

echo "[model-archive-security-gate] configuring ASan/UBSan subset in ${SAN_BUILD_DIR}..."
cmake -S . -B "${SAN_BUILD_DIR}" \
  -DSIMANEAT_BUILD_SAMPLES=OFF \
  -DSIMA_ENABLE_ASAN=ON \
  -DSIMA_ENABLE_UBSAN=ON \
  -DSIMA_ENABLE_TSAN=OFF

export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0:abort_on_error=1:halt_on_error=1}"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=1}"

echo "[model-archive-security-gate] building sanitizer model archive/security tests..."
san_targets=(unit_modalix_contract_preflight_test "${archive_security_tests[@]}")
cmake --build "${SAN_BUILD_DIR}" --target "${san_targets[@]}" -j"${CMAKE_BUILD_PARALLEL_LEVEL:-8}"

echo "[model-archive-security-gate] checking sanitizer model archive fixtures..."
python3 tests/tools/make_model_archive_fixtures.py \
  --root "${SAN_BUILD_DIR}/test-assets/model-archive" \
  --manifest "${SAN_BUILD_DIR}/test-assets/model-archive/fixtures_manifest.json" \
  --check

echo "[model-archive-security-gate] running sanitizer Modalix preflight..."
ctest --test-dir "${SAN_BUILD_DIR}" --output-on-failure -R "^unit_modalix_contract_preflight_test$"

echo "[model-archive-security-gate] running sanitizer model archive/security tests..."
ctest --test-dir "${SAN_BUILD_DIR}/tests" --output-on-failure -R "${regex}"

echo "[model-archive-security-gate] passed."
