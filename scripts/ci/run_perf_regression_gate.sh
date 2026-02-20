#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BUILD_DIR="${BUILD_DIR:-build-perf-gate}"
PROFILE_DIR="${PROFILE_DIR:-tests/perf/baselines/v2/modalix_default}"
RESULTS_DIR="${RESULTS_DIR:-${BUILD_DIR}/perf_results}"
SCENARIO_TIMEOUT_SEC="${SIMA_PERF_SCENARIO_TIMEOUT_SEC:-180}"

echo "[perf-regression-gate] validating baseline schema (${PROFILE_DIR})..."
python3 tests/perf/tools/validate_perf_baselines.py \
  --profile-dir "${PROFILE_DIR}"

echo "[perf-regression-gate] running perf matrix..."
python3 tests/perf/tools/run_perf_matrix.py \
  --repo-root "${ROOT_DIR}" \
  --build-dir "${BUILD_DIR}" \
  --profile-dir "${PROFILE_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --scenario-timeout-sec "${SCENARIO_TIMEOUT_SEC}"

echo "[perf-regression-gate] validating generated result schema (${RESULTS_DIR})..."
python3 tests/perf/tools/validate_perf_result.py \
  --results-dir "${RESULTS_DIR}" \
  --summary

echo "[perf-regression-gate] completed. results: ${RESULTS_DIR}"
