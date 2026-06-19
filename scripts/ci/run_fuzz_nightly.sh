#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BUILD_DIR="${BUILD_DIR:-build-fuzz-nightly}"
CORPUS_ROOT="${CORPUS_ROOT:-tests/fuzz/corpus}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${BUILD_DIR}/fuzz-artifacts}"
MAX_TOTAL_TIME="${MAX_TOTAL_TIME:-45}"
FUZZ_PREBUILT_ONLY="${FUZZ_PREBUILT_ONLY:-0}"
FUZZ_BIN_DIR="${FUZZ_BIN_DIR:-${BUILD_DIR}/tests/fuzz}"

FUZZ_TARGETS=(
  fuzz_modelpack_json
  fuzz_gst_pipeline_string
)

declare -A CORPUS_BY_TARGET=(
  [fuzz_modelpack_json]="modelpack_json"
  [fuzz_gst_pipeline_string]="gst_pipeline_string"
)

mkdir -p "${ARTIFACT_DIR}"

if [[ "${FUZZ_PREBUILT_ONLY}" != "1" ]]; then
  echo "[fuzz-nightly] configuring fuzz build in ${BUILD_DIR}..."
  cmake -S . -B "${BUILD_DIR}" -DFUZZING=ON

  echo "[fuzz-nightly] building fuzz targets..."
  cmake --build "${BUILD_DIR}" --target "${FUZZ_TARGETS[@]}" -j"${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
else
  echo "[fuzz-nightly] prebuilt-only mode enabled. Using binaries from ${FUZZ_BIN_DIR}."
fi

export ASAN_OPTIONS="${ASAN_OPTIONS:-abort_on_error=1:halt_on_error=1:detect_leaks=0}"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=1}"

for target in "${FUZZ_TARGETS[@]}"; do
  corpus_subdir="${CORPUS_BY_TARGET[${target}]}"
  corpus_dir="${CORPUS_ROOT}/${corpus_subdir}"
  bin="${FUZZ_BIN_DIR}/${target}"

  if [[ ! -x "${bin}" ]]; then
    echo "ERROR: missing fuzz binary: ${bin}" >&2
    exit 1
  fi
  if [[ ! -d "${corpus_dir}" ]]; then
    echo "ERROR: missing corpus directory: ${corpus_dir}" >&2
    exit 1
  fi

  echo "[fuzz-nightly] running ${target} with corpus ${corpus_dir}..."
  "${bin}" "${corpus_dir}" \
    -max_total_time="${MAX_TOTAL_TIME}" \
    -print_final_stats=1 \
    -artifact_prefix="${ARTIFACT_DIR}/${target}_"
done

echo "[fuzz-nightly] completed. artifacts=${ARTIFACT_DIR}"
