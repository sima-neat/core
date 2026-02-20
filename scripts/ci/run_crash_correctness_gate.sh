#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BUILD_DIR="${BUILD_DIR:-build-crash-correctness}"
REPEAT_COUNT="${CRASH_REPEAT_COUNT:-20}"

critical_tests=(
  unit_builder_test
  unit_caps_bridge_test
  unit_graph_compiler_topology_test
  unit_graph_compiler_spec_validation_test
  unit_graph_join_bundle_test
  unit_graph_join_encoded_with_meta_test
  unit_graph_stamp_frame_id_test
  unit_graph_strict_sync_store_test
  unit_contracts_test
  unit_detection_types_bbox_test
  unit_gst_data_adapter_edge_test
  unit_group_optiview_output_group_failure_test
  unit_nodes_test
  unit_node_optiview_json_output_test
  unit_outputspec_test
  unit_pipeline_build_wiring_test
  unit_pipeline_internal_build_test
  unit_run_holder_api_test
  unit_model_infer_output_name_test
  unit_model_input_spec_contract_test
  unit_model_output_spec_contract_test
  unit_model_metadata_test
  unit_model_stage_fragments_test
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
  output_appsink_policy_test
  async_stream_test
  hybrid_graph_basic_test
  hybrid_graph_fanout_test
  graph_deterministic_routing_regression_test
  caps_negotiation_matrix_regression_test
  gst_data_adapter_runtime_regression_test
  pull_timeout_regression_test
  session_naming_pipeline_integration_test
  session_rtsp_lifecycle_regression_test
  session_validate_report_regression_test
  stage_routing_regression_test
)

repeat_tests=(
  async_stream_test
  hybrid_graph_basic_test
  graph_deterministic_routing_regression_test
  caps_negotiation_matrix_regression_test
  gst_data_adapter_runtime_regression_test
  pull_timeout_regression_test
  unit_run_try_push_api_test
  unit_run_holder_api_test
  unit_run_pull_variants_api_test
  unit_model_infer_output_name_test
  unit_model_input_spec_contract_test
  unit_model_output_spec_contract_test
  unit_run_api_variants_test
  session_naming_pipeline_integration_test
  session_rtsp_lifecycle_regression_test
  session_validate_report_regression_test
  stage_routing_regression_test
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

critical_regex="$(join_regex critical_tests)"
repeat_regex="$(join_regex repeat_tests)"

echo "[crash-correctness-gate] configuring build in ${BUILD_DIR}..."
cmake -S . -B "${BUILD_DIR}" -DSIMANEAT_BUILD_SAMPLES=OFF

echo "[crash-correctness-gate] building critical test targets..."
build_targets=(unit_modalix_contract_preflight_test "${critical_tests[@]}")
cmake --build "${BUILD_DIR}" --target "${build_targets[@]}" -j"${CMAKE_BUILD_PARALLEL_LEVEL:-8}"

echo "[crash-correctness-gate] running Modalix preflight..."
ctest --test-dir "${BUILD_DIR}" -R "^unit_modalix_contract_preflight_test$" --output-on-failure

echo "[crash-correctness-gate] running critical test suite..."
ctest --test-dir "${BUILD_DIR}/tests" --output-on-failure -R "${critical_regex}"

if [[ "${REPEAT_COUNT}" -gt 1 ]]; then
  echo "[crash-correctness-gate] running repeat-until-fail:${REPEAT_COUNT} on crash-prone regressions..."
  ctest --test-dir "${BUILD_DIR}/tests" \
    --output-on-failure \
    --repeat "until-fail:${REPEAT_COUNT}" \
    -R "${repeat_regex}"
fi

echo "[crash-correctness-gate] passed."
