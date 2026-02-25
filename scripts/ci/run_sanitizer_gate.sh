#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODE="${SANITIZER_MODE:-asan-ubsan}"
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
SKIP_BUILD="${SANITIZER_SKIP_BUILD:-0}"
TEST_DIR_OVERRIDE="${SANITIZER_TEST_DIR:-}"
export CMAKE_BUILD_PARALLEL_LEVEL

usage() {
  cat <<USAGE
Usage: SANITIZER_MODE=<asan-ubsan|tsan> bash scripts/ci/run_sanitizer_gate.sh [options]

Options:
  --mode <asan-ubsan|tsan>    Override sanitizer mode
  --test-dir <path>           Run tests from prebuilt installed artifacts (no build)
  --build-dir <path>          Override local CMake build directory
  -h, --help                  Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --test-dir)
      TEST_DIR_OVERRIDE="${2:-}"
      SKIP_BUILD=1
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

asan_ubsan_tests=(
  unit_builder_test
  unit_caps_bridge_test
  unit_graph_compiler_topology_test
  unit_graph_compiler_spec_validation_test
  unit_graph_join_bundle_test
  unit_graph_join_encoded_with_meta_test
  unit_graph_stamp_frame_id_test
  unit_graph_strict_sync_store_test
  unit_group_optiview_output_group_failure_test
  unit_contracts_test
  unit_detection_types_bbox_test
  unit_gst_data_adapter_edge_test
  unit_node_optiview_json_output_test
  unit_node_udp_output_fragment_test
  unit_nodes_test
  unit_outputspec_test
  unit_pipeline_build_wiring_test
  unit_model_infer_output_name_test
  unit_model_input_spec_contract_test
  unit_model_output_spec_contract_test
  unit_model_metadata_test
  unit_model_stage_fragments_test
  unit_run_holder_api_test
  unit_run_pull_variants_api_test
  unit_run_try_push_api_test
  unit_run_api_variants_test
  unit_session_io_roundtrip_test
  unit_session_io_negative_test
  unit_session_naming_transform_test
  unit_simaai_guard_test
  unit_stageconfig_mla_info_test
  unit_tensor_conversion_policy_test
  unit_tensor_image_mapping_test
  caps_negotiation_matrix_regression_test
  gst_data_adapter_runtime_regression_test
  pull_timeout_regression_test
  session_naming_pipeline_integration_test
  session_rtsp_lifecycle_regression_test
  session_validate_report_regression_test
  stage_routing_regression_test
  graph_deterministic_routing_regression_test
  hybrid_graph_basic_test
)

tsan_tests=(
  unit_builder_test
  unit_contracts_test
  unit_graph_strict_sync_store_test
  unit_run_try_push_api_test
  unit_run_holder_api_test
  unit_run_api_variants_test
  unit_session_naming_transform_test
  session_validate_report_regression_test
  session_naming_pipeline_integration_test
  stage_routing_regression_test
  graph_deterministic_routing_regression_test
  hybrid_graph_fanout_test
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

BUILD_DIR="${BUILD_DIR:-build-${MODE}-gate}"
TEST_DIR=""
PREFLIGHT_DIR=""

if [[ "${MODE}" == "asan-ubsan" ]]; then
  TESTS=("${asan_ubsan_tests[@]}")
  # Disable default leak detection in this gate because third-party runtime
  # initialization can report non-actionable leaks and hide project findings.
  export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0:abort_on_error=1:halt_on_error=1}"
  export UBSAN_OPTIONS="${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=1}"
elif [[ "${MODE}" == "tsan" ]]; then
  TESTS=("${tsan_tests[@]}")
  export TSAN_OPTIONS="${TSAN_OPTIONS:-halt_on_error=1:history_size=7}"
else
  echo "ERROR: Unsupported SANITIZER_MODE='${MODE}'. Use asan-ubsan or tsan." >&2
  exit 1
fi

echo "[sanitizer-gate] mode=${MODE} skip_build=${SKIP_BUILD}"

if [[ "${SKIP_BUILD}" == "1" ]]; then
  TEST_DIR="${TEST_DIR_OVERRIDE}"
  if [[ -z "${TEST_DIR}" ]]; then
    echo "ERROR: SANITIZER_TEST_DIR is required when SANITIZER_SKIP_BUILD=1." >&2
    exit 1
  fi
  if [[ ! -d "${TEST_DIR}" ]]; then
    echo "ERROR: SANITIZER_TEST_DIR does not exist: ${TEST_DIR}" >&2
    exit 1
  fi
  PREFLIGHT_DIR="${TEST_DIR}"
else
  if [[ "${MODE}" == "asan-ubsan" ]]; then
    cmake -S . -B "${BUILD_DIR}" \
      -DSIMANEAT_BUILD_SAMPLES=OFF \
      -DSIMA_ENABLE_ASAN=ON \
      -DSIMA_ENABLE_UBSAN=ON \
      -DSIMA_ENABLE_TSAN=OFF
  else
    cmake -S . -B "${BUILD_DIR}" \
      -DSIMANEAT_BUILD_SAMPLES=OFF \
      -DSIMA_ENABLE_ASAN=OFF \
      -DSIMA_ENABLE_UBSAN=OFF \
      -DSIMA_ENABLE_TSAN=ON
  fi
fi

regex="$(join_regex TESTS)"

if [[ "${SKIP_BUILD}" != "1" ]]; then
  echo "[sanitizer-gate] building test targets..."
  build_targets=(unit_modalix_contract_preflight_test "${TESTS[@]}")
  cmake --build "${BUILD_DIR}" --target "${build_targets[@]}" -j"${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
  TEST_DIR="${BUILD_DIR}/tests"
  PREFLIGHT_DIR="${BUILD_DIR}"
fi

#echo "[sanitizer-gate] running Modalix preflight..."
#ctest --test-dir "${PREFLIGHT_DIR}" --output-on-failure -R "^unit_modalix_contract_preflight_test$" --no-tests=error

echo "[sanitizer-gate] running tests..."
ctest --test-dir "${TEST_DIR}" --output-on-failure --no-tests=error -R "${regex}"

echo "[sanitizer-gate] ${MODE} passed."
