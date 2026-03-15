cmake_minimum_required(VERSION 3.16)

# ------------------------------------------------------------------------------
# PipelineSession tests
# ------------------------------------------------------------------------------
message(STATUS "Configuring PipelineSession tests")

enable_testing()
include(CMakeParseArguments)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(PIPELINESESSION_TEST_TARGETS "")
# ------------------------------------------------------------------------------
# Test blacklist (by executable name)
# ------------------------------------------------------------------------------
set(PIPELINESESSION_TEST_BLACKLIST
)

# ------------------------------------------------------------------------------
# Helper: fixture preparation
# ------------------------------------------------------------------------------
set(SIMA_TEST_MPK_FIXTURE_ROOT "${CMAKE_SOURCE_DIR}/tests/assets/mpk")
set(SIMA_TEST_MPK_FIXTURE_MANIFEST "${CMAKE_SOURCE_DIR}/tests/assets/mpk/fixtures_manifest.json")
set(SIMA_TEST_DECODER_FIXTURE "${CMAKE_SOURCE_DIR}/tests/assets/decoder/dynamic_caps.h264")

add_custom_target(prepare_test_fixtures
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/make_mpk_fixtures.py
    --root ${SIMA_TEST_MPK_FIXTURE_ROOT}
    --manifest ${SIMA_TEST_MPK_FIXTURE_MANIFEST}
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/make_decoder_dynamic_fixture.py
    --output ${SIMA_TEST_DECODER_FIXTURE}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Preparing deterministic test fixtures"
)

# ------------------------------------------------------------------------------
# Helper: add strict test executable + ctest entry
# ------------------------------------------------------------------------------
function(add_pipeline_test_strict name)
  if (name IN_LIST PIPELINESESSION_TEST_BLACKLIST)
    message(STATUS "  [tests] Skipping blacklisted test: ${name}")
    return()
  endif()

  set(options)
  set(oneValueArgs)
  set(multiValueArgs SOURCES ARGS)
  cmake_parse_arguments(APT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (APT_SOURCES)
    set(test_sources ${APT_SOURCES})
  else()
    set(test_sources ${APT_UNPARSED_ARGUMENTS})
  endif()

  add_executable(${name} ${test_sources})
  add_dependencies(${name} prepare_test_fixtures)

  target_link_libraries(${name}
    PRIVATE sima_neat_internal
  )

  # 🔑 CRITICAL: test include paths
  target_include_directories(${name}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}                 # tests/test_utils.h
      ${CMAKE_CURRENT_SOURCE_DIR}/e2e_pipelines   # e2e_utils.h
      ${CMAKE_CURRENT_SOURCE_DIR}/e2e_pipelines/obj_detection
  )

  # Base test (optional arguments)
  if (APT_ARGS)
    add_test(NAME ${name} COMMAND ${name} ${APT_ARGS})
  else()
    add_test(NAME ${name} COMMAND ${name})
  endif()

  set_tests_properties(${name} PROPERTIES
    RESOURCE_LOCK "dispatcher"
  )

  set_property(TEST ${name} APPEND PROPERTY LABELS "strict")

  message(STATUS "  [tests] Added strict test executable: ${name}")

  list(APPEND PIPELINESESSION_TEST_TARGETS ${name})
endfunction()

# ------------------------------------------------------------------------------
# Helper: add long test executable + ctest entry
# ------------------------------------------------------------------------------
function(add_pipeline_test_long name)
  add_pipeline_test_strict(${name} ${ARGN})
  if (name IN_LIST PIPELINESESSION_TEST_BLACKLIST)
    return()
  endif()

  set_property(TEST ${name} APPEND PROPERTY LABELS "long")
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
  set_property(TEST ${name} APPEND PROPERTY ENVIRONMENT "SIMA_TEST_IS_LONG=1")
endfunction()

# Backward-compatible alias: default all tests to strict unless explicitly long.
function(add_pipeline_test name)
  add_pipeline_test_strict(${name} ${ARGN})
endfunction()

# ------------------------------------------------------------------------------
# Unit tests (1:1 cpp mapping)
# ------------------------------------------------------------------------------
set(UNIT_TESTS
  unit_buffer_name_policy_test
  unit_builder_test
  unit_caps_bridge_test
  unit_contracts_test
  unit_detection_types_bbox_test
  unit_detessdequant_output_info_test
  unit_dispatcher_recovery_test
  unit_dispatcher_watchdog_test
  unit_error_code_taxonomy_test
  unit_encoded_sample_test
  unit_framestream_test
  unit_graph_compiler_determinism_test
  unit_graph_compiler_spec_validation_test
  unit_graph_compiler_topology_test
  unit_graph_join_bundle_test
  unit_graph_join_encoded_with_meta_test
  unit_graph_join_nodes_test
  unit_graph_stamp_frame_id_test
  unit_graph_backpressure_timeout_test
  unit_graph_run_option_validation_test
  unit_graph_stop_sequencing_test
  unit_graph_strict_sync_store_test
  unit_graph_strict_sync_test
  unit_header_neat_leaf_compile_test
  unit_header_neat_graph_compile_test
  unit_header_neat_models_compile_test
  unit_header_neat_node_groups_compile_test
  unit_header_neat_nodes_compile_test
  unit_header_neat_session_compile_test
  unit_header_neat_split_compile_test
  unit_header_neat_umbrella_compile_test
  unit_gst_data_adapter_edge_test
  unit_groups_test
  unit_gst_helpers_test
  unit_gst_init_guard_test
  unit_gst_video_meta_test
  unit_inputstream_format_change_test
  unit_inputstream_dynamic_policy_matrix_test
  unit_input_policy_test
  unit_inputstream_nv12_renegotiate_test
  unit_inputstream_renegotiate_test
  unit_inputstream_stop_timeout_detach_test
  unit_inputappsrc_stop_race_test
  unit_neatdecoder_keepalive_test
  unit_neatencoder_keepalive_test
  unit_neatmodel_async_api_test
  unit_neattensor_core_test
  unit_neattensor_cvmat_test
  unit_modelpack_extract_test
  unit_modelpack_parse_test
  unit_model_infer_output_name_test
  unit_model_input_spec_contract_test
  unit_model_output_spec_contract_test
  unit_model_metadata_test
  unit_model_stage_fragments_test
  unit_modalix_contract_preflight_test
  unit_mpk_loader_test
  unit_mpk_pipeline_adapter_test
  unit_node_h264_caps_fixup_fragment_test
  unit_node_optiview_json_output_test
  unit_node_quanttess_config_test
  unit_node_udp_output_fragment_test
  unit_outputspec_test
  unit_pcie_nodes_test
  unit_group_optiview_output_group_failure_test
  unit_group_udp_h264_output_group_fragment_test
  unit_group_udp_output_group_g_fragment_test
  unit_optiview_json_output_udp_test
  unit_optiview_output_group_test
  pull_timeout_regression_test
  property_config_json_test
  property_pipeline_string_test
  unit_pipeline_build_wiring_test
  unit_pipeline_internal_build_test
  unit_policy_defaults_test
  unit_run_holder_api_test
  unit_run_lifecycle_teardown_test
  unit_run_pull_variants_api_test
  unit_run_try_push_api_test
  unit_run_overflow_policy_parity_test
  unit_run_sample_message_modes_test
  unit_run_api_variants_test
  unit_samplespec_test
  unit_session_api_surface_test
  unit_session_build_input_strategy_test
  unit_session_io_roundtrip_test
  unit_session_io_negative_test
  unit_session_naming_transform_test
  unit_sima_boxdecode_config_test
  unit_simaai_guard_test
  unit_stageconfig_mla_info_test
  unit_simaai_allocator_share_test
  unit_sima_mla_static_contract_extractor_test
  unit_boxdecode_resolution_precedence_test
  unit_boxdecode_required_runtime_fields_test
  unit_quant_resolution_precedence_test
  unit_wiring_tensor_index_test
  unit_kernel200_envelope_guard_test
  unit_stage_transform_rule_registry_test
  unit_sima_plugin_manifest_strict_fallback_test
  unit_sima_plugin_manifest_strict_model_pipeline_test
  unit_sima_plugin_static_manifest_parse_test
  unit_sima_plugin_static_manifest_contract_test
  unit_sima_plugin_context_injection_test
  unit_sima_plugin_static_manifest_resolver_test
  unit_sima_plugins_test
  unit_pipeline_sequence_parse_test
  unit_tensor_conversion_policy_test
  unit_tensor_image_mapping_test
  unit_transport_output_nodes_test
  unit_nodes_test
  unit_plugin_failure_messages_test
)

foreach(test ${UNIT_TESTS})
  add_pipeline_test(
    ${test}
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_testing/${test}.cpp
  )

  if (NOT test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
    set_tests_properties(${test} PROPERTIES LABELS "unit")
  endif()
endforeach()

if (TARGET unit_plugin_failure_messages_test)
  set(SIMA_TEST_NEAT_PLUGIN_DIR "/usr/lib/aarch64-linux-gnu/neat/gst-plugins")
  set_tests_properties(unit_plugin_failure_messages_test PROPERTIES
    ENVIRONMENT "SIMA_GST_PLUGIN_DIR=${SIMA_TEST_NEAT_PLUGIN_DIR};GST_REGISTRY_1_0=${CMAKE_BINARY_DIR}/sima_gst_registry_unit_plugin_failure_messages_test_v2.bin")
endif()

if (TARGET unit_simaai_allocator_share_test)
  target_link_libraries(unit_simaai_allocator_share_test PRIVATE ${CMAKE_DL_LIBS})
endif()

if (TARGET unit_samplespec_test)
  target_include_directories(unit_samplespec_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
endif()

add_pipeline_test(
  unit_appsrcimage_test
  ${CMAKE_CURRENT_SOURCE_DIR}/unit_testing/unit_appsrcimage_test.cpp
  ARGS ${CMAKE_SOURCE_DIR}/test.jpg
)

add_pipeline_test(
  unit_image_group_runtime_test
  ${CMAKE_CURRENT_SOURCE_DIR}/unit_testing/unit_image_group_runtime_test.cpp
  ARGS ${CMAKE_SOURCE_DIR}/test.jpg
)

add_pipeline_test(
  unit_pipeline_io_test
  ${CMAKE_CURRENT_SOURCE_DIR}/unit_testing/unit_pipeline_io_test.cpp
  ARGS ${CMAKE_SOURCE_DIR}/test.jpg
)

add_pipeline_test(
  neat_tensor_device_placement_test
  ${CMAKE_CURRENT_SOURCE_DIR}/neat_tensor_device_placement_test.cpp
)

foreach(test unit_appsrcimage_test unit_image_group_runtime_test unit_pipeline_io_test neat_tensor_device_placement_test)
  if (NOT test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
    set_tests_properties(${test} PROPERTIES LABELS "unit")
  endif()
endforeach()

# ------------------------------------------------------------------------------
# Simple E2E tests
# ------------------------------------------------------------------------------
set(E2E_TESTS
  appsrc_videoconvert_fakesink_test
  appsrc_videoscale_fakesink_test
  async_stream_test
  caps_negotiation_matrix_regression_test
  decoder_download_test
  decoder_drop_on_resolution_change_test
  decoder_dynamic_caps_test
  decoder_quality_compare_test
  decoder_visible_caps_test
  hybrid_graph_basic_test
  hybrid_graph_bundle_test
  graph_deterministic_routing_regression_test
  hybrid_graph_fanout_test
  hybrid_graph_multistream_scheduler_test
  hybrid_graph_stage_model_test
  inputstream_nv12_format_change_test
  inputstream_nv12_stability_test
  inputstream_videoconvert_test
  inputstream_videoscale_test
  output_appsink_policy_test
  gst_data_adapter_runtime_regression_test
  optiview_json_udp_integration_test
  preproc_standalone_test
  session_naming_pipeline_integration_test
  session_rtsp_lifecycle_regression_test
  session_validate_report_regression_test
  stage_routing_regression_test
  udp_output_loopback_integration_test
)

foreach(test ${E2E_TESTS})
  add_pipeline_test(
    ${test}
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_testing/${test}.cpp
  )

  if (NOT test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
    set_tests_properties(${test} PROPERTIES LABELS "e2e")
  endif()
endforeach()

add_pipeline_test_long(
  modelzoo_bulk_models_test
  e2e_testing/modelzoo_bulk_models_test.cpp
)

if (NOT modelzoo_bulk_models_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_tests_properties(modelzoo_bulk_models_test PROPERTIES LABELS "e2e;modelzoo;long")
endif()

if (NOT decoder_quality_compare_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST decoder_quality_compare_test APPEND PROPERTY RESOURCE_LOCK "rtsp")
  set_property(TEST decoder_quality_compare_test
    APPEND PROPERTY ENVIRONMENT "SIMA_RTSP_PORT=8757"
  )
endif()

add_pipeline_test(
  preproc_dynamic_renegotiate_test
  ${CMAKE_CURRENT_SOURCE_DIR}/e2e_testing/preproc_dynamic_renegotiate_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/e2e_pipelines/e2e_utils.cpp
)

if (NOT preproc_dynamic_renegotiate_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_tests_properties(preproc_dynamic_renegotiate_test PROPERTIES LABELS "e2e")
endif()

# ------------------------------------------------------------------------------
# Stress tests
# ------------------------------------------------------------------------------
set(STRESS_TESTS
  stress_graph_strict_sync_join_race_test
  stress_udp_json_burst_test
  stress_pipeline_build_repeated_test
  stress_model_lifecycle_test
  stress_session_lifecycle_test
  stress_session_validate_rtsp_churn_test
  stress_run_async_pressure_test
  stress_run_try_push_holder_race_test
  stress_graph_execution_test
)

foreach(test ${STRESS_TESTS})
  add_pipeline_test_long(
    ${test}
    ${CMAKE_CURRENT_SOURCE_DIR}/stress/${test}.cpp
  )

  if (NOT test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
    set_tests_properties(${test} PROPERTIES LABELS "stress")
  endif()
endforeach()

# ------------------------------------------------------------------------------
# Security tests (MPK / archive / JSON abuse)
# ------------------------------------------------------------------------------
set(SECURITY_TESTS
  security_modelpack_archive_test
  security_modelpack_json_test
  security_modelpack_unicode_path_test
  security_modelpack_tar_integrity_test
  security_modelpack_json_robustness_test
)

foreach(test ${SECURITY_TESTS})
  add_pipeline_test(
    ${test}
    ${CMAKE_CURRENT_SOURCE_DIR}/security/${test}.cpp
  )

  if (NOT test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
    set_tests_properties(${test} PROPERTIES LABELS "security")
  endif()
endforeach()

# ------------------------------------------------------------------------------
# Performance suites (binary metrics emitters; not default ctest)
# ------------------------------------------------------------------------------
set(PERF_TARGETS
  perf_mpk_parse_smoke_test
  perf_runtime_session_sync_rgb_test
  perf_runtime_session_async_rgb_test
  perf_runtime_graph_fanout_test
  perf_runtime_graph_join_bundle_test
)

foreach(target ${PERF_TARGETS})
  add_executable(
    ${target}
    ${CMAKE_CURRENT_SOURCE_DIR}/perf/${target}.cpp
  )
  target_link_libraries(${target} PRIVATE sima_neat_internal)
  target_include_directories(${target}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/e2e_pipelines
      ${CMAKE_CURRENT_SOURCE_DIR}/e2e_pipelines/obj_detection
  )
  list(APPEND PIPELINESESSION_TEST_TARGETS ${target})
endforeach()

set(PERF_PYTHON_TESTS
  test_perf_schema
  test_perf_classifier
  test_perf_comparator
  test_perf_matrix_failfast
)

foreach(pytest ${PERF_PYTHON_TESTS})
  add_test(
    NAME ${pytest}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/perf/tools/${pytest}.py
  )
  set_tests_properties(${pytest} PROPERTIES LABELS "perf;unit")
endforeach()

# ------------------------------------------------------------------------------
# Fuzz harnesses (opt-in via -DFUZZING=ON)
# ------------------------------------------------------------------------------
if (FUZZING)
  if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "FUZZING=ON requires Clang/libFuzzer (set CC=clang CXX=clang++).")
  endif()

  function(add_pipeline_fuzz_target name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE sima_neat_internal)
    target_include_directories(${name}
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_pipelines
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_pipelines/obj_detection
    )
    target_compile_options(${name} PRIVATE
      -fsanitize=fuzzer,address,undefined
      -fno-omit-frame-pointer
      -O1
      -g
    )
    target_link_options(${name} PRIVATE -fsanitize=fuzzer,address,undefined)
    set_target_properties(${name} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/fuzz
    )
    list(APPEND PIPELINESESSION_TEST_TARGETS ${name})
  endfunction()

  add_pipeline_fuzz_target(
    fuzz_modelpack_json
    ${CMAKE_CURRENT_SOURCE_DIR}/fuzz/fuzz_modelpack_json.cpp
  )
  add_pipeline_fuzz_target(
    fuzz_pipeline_sequence
    ${CMAKE_CURRENT_SOURCE_DIR}/fuzz/fuzz_pipeline_sequence.cpp
  )
  add_pipeline_fuzz_target(
    fuzz_gst_pipeline_string
    ${CMAKE_CURRENT_SOURCE_DIR}/fuzz/fuzz_gst_pipeline_string.cpp
  )
endif()

# ------------------------------------------------------------------------------
# Tests with arguments
# ------------------------------------------------------------------------------
add_pipeline_test(
  file_read_test
  e2e_testing/file_read_test.cpp
  ARGS --image ${CMAKE_SOURCE_DIR}/test.jpg --w 256 --h 256
)

if (NOT file_read_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_tests_properties(file_read_test PROPERTIES LABELS "e2e")
endif()

# ------------------------------------------------------------------------------
# Crash/correctness gate labels
# ------------------------------------------------------------------------------
foreach(test
    pull_timeout_regression_test
    caps_negotiation_matrix_regression_test
    stage_routing_regression_test
    graph_deterministic_routing_regression_test
    async_stream_test
    hybrid_graph_basic_test
    hybrid_graph_fanout_test
    unit_builder_test
    unit_caps_bridge_test
    unit_graph_compiler_spec_validation_test
    unit_graph_compiler_topology_test
    unit_graph_join_bundle_test
    unit_graph_join_encoded_with_meta_test
    unit_graph_stamp_frame_id_test
    unit_graph_strict_sync_store_test
    unit_group_optiview_output_group_failure_test
    unit_node_optiview_json_output_test
    unit_modalix_contract_preflight_test
    unit_run_holder_api_test
    unit_run_pull_variants_api_test
    unit_run_try_push_api_test
    unit_run_sample_message_modes_test
    unit_contracts_test
    unit_detection_types_bbox_test
    unit_gst_data_adapter_edge_test
    unit_nodes_test
    unit_outputspec_test
    unit_model_input_spec_contract_test
    unit_model_output_spec_contract_test
    unit_model_metadata_test
    unit_model_stage_fragments_test
    unit_stageconfig_mla_info_test
    session_naming_pipeline_integration_test
    gst_data_adapter_runtime_regression_test
    unit_run_api_variants_test
    unit_modelpack_parse_test
    unit_modelpack_extract_test
    unit_pipeline_sequence_parse_test
    unit_mpk_loader_test
    unit_mpk_pipeline_adapter_test
    session_rtsp_lifecycle_regression_test
    session_validate_report_regression_test
    security_modelpack_archive_test
    security_modelpack_json_test)
  if (NOT test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
    set_property(TEST ${test} APPEND PROPERTY LABELS "crash-gate;regression")
  endif()
endforeach()

# ------------------------------------------------------------------------------
# Runtime-core labels
# ------------------------------------------------------------------------------
foreach(test
    unit_run_try_push_api_test
    unit_run_holder_api_test
    unit_run_pull_variants_api_test
    unit_run_sample_message_modes_test
    unit_model_infer_output_name_test
    unit_model_input_spec_contract_test
    unit_model_output_spec_contract_test
    unit_model_metadata_test
    unit_model_stage_fragments_test
    unit_session_io_roundtrip_test
    unit_session_io_negative_test
    unit_session_naming_transform_test
    unit_pipeline_build_wiring_test
    unit_pipeline_internal_build_test
    unit_gst_data_adapter_edge_test
    unit_caps_bridge_test
    unit_simaai_guard_test
    unit_tensor_conversion_policy_test
    unit_tensor_image_mapping_test
    unit_stageconfig_mla_info_test
    unit_detection_types_bbox_test
    unit_modelpack_parse_test
    unit_modelpack_extract_test
    unit_pipeline_sequence_parse_test
    unit_mpk_loader_test
    unit_mpk_pipeline_adapter_test
    unit_modalix_contract_preflight_test
    session_validate_report_regression_test
    session_rtsp_lifecycle_regression_test
    session_naming_pipeline_integration_test
    gst_data_adapter_runtime_regression_test
    stress_run_try_push_holder_race_test
    stress_session_validate_rtsp_churn_test
    stress_pipeline_build_repeated_test)
  if (NOT test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
    set_property(TEST ${test} APPEND PROPERTY LABELS "runtime-core")
  endif()
endforeach()

# ------------------------------------------------------------------------------
# Negative-path labels
# ------------------------------------------------------------------------------
foreach(test
    unit_run_try_push_api_test
    unit_run_holder_api_test
    unit_run_pull_variants_api_test
    unit_session_io_roundtrip_test
    unit_session_io_negative_test
    unit_graph_compiler_topology_test
    unit_graph_compiler_spec_validation_test
    unit_group_optiview_output_group_failure_test
    unit_node_optiview_json_output_test
    unit_modelpack_parse_test
    unit_modelpack_extract_test
    unit_pipeline_sequence_parse_test
    unit_mpk_loader_test
    security_modelpack_archive_test
    security_modelpack_json_test
    pull_timeout_regression_test
    session_validate_report_regression_test
    session_rtsp_lifecycle_regression_test
    gst_data_adapter_runtime_regression_test)
  if (NOT test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
    set_property(TEST ${test} APPEND PROPERTY LABELS "negative")
  endif()
endforeach()

# ------------------------------------------------------------------------------
# E2E pipelines – object detection
# ------------------------------------------------------------------------------
add_pipeline_test(
  async_yolov8_tput_test
  e2e_pipelines/obj_detection/async_yolov8_tput_test.cpp
  e2e_pipelines/e2e_utils.cpp
  e2e_pipelines/obj_detection/obj_detection_utils.cpp
  ARGS ${CMAKE_SOURCE_DIR}
)

if (NOT async_yolov8_tput_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_tests_properties(async_yolov8_tput_test
    PROPERTIES LABELS "e2e;obj_detection"
  )
endif()

add_pipeline_test(
  sync_yolov8_test
  e2e_pipelines/obj_detection/sync_yolov8_test.cpp
  e2e_pipelines/e2e_utils.cpp
  e2e_pipelines/obj_detection/obj_detection_utils.cpp
  ARGS ${CMAKE_SOURCE_DIR}
)

if (NOT sync_yolov8_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_tests_properties(sync_yolov8_test
    PROPERTIES LABELS "e2e;obj_detection"
  )
endif()

add_pipeline_test(
  sync_yolov8_test_detessdequant
  e2e_pipelines/obj_detection/sync_yolov8_test_detessdequant.cpp
  e2e_pipelines/e2e_utils.cpp
  e2e_pipelines/obj_detection/obj_detection_utils.cpp
  ARGS ${CMAKE_SOURCE_DIR}
)

if (NOT sync_yolov8_test_detessdequant IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_tests_properties(sync_yolov8_test_detessdequant
    PROPERTIES LABELS "e2e;obj_detection"
  )
endif()

add_pipeline_test(
  async_yolov11_pose_bf16_tput_test
  e2e_pipelines/obj_detection/async_yolov11_pose_bf16_tput_test.cpp
  e2e_pipelines/e2e_utils.cpp
  e2e_pipelines/obj_detection/obj_detection_utils.cpp
  ARGS ${CMAKE_SOURCE_DIR}
)

if (NOT async_yolov11_pose_bf16_tput_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_tests_properties(async_yolov11_pose_bf16_tput_test
    PROPERTIES
      LABELS "e2e;obj_detection"
      DISABLED TRUE
  )
endif()

add_pipeline_test(
  yolov8_stage_route_test
  e2e_pipelines/obj_detection/yolov8_stage_route_test.cpp
  e2e_pipelines/e2e_utils.cpp
  e2e_pipelines/obj_detection/obj_detection_utils.cpp
)

if (NOT yolov8_stage_route_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_tests_properties(yolov8_stage_route_test
    PROPERTIES LABELS "e2e;obj_detection"
  )
endif()

add_pipeline_test(
  graphpipes_yolo_optiview_16stream_test
  e2e_pipelines/obj_detection/graphpipes_yolo_optiview_16stream_test.cpp
  e2e_pipelines/e2e_utils.cpp
  e2e_pipelines/obj_detection/obj_detection_utils.cpp
  ARGS ${CMAKE_SOURCE_DIR}
)

if (NOT graphpipes_yolo_optiview_16stream_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_tests_properties(graphpipes_yolo_optiview_16stream_test
    PROPERTIES LABELS "e2e;obj_detection"
  )
endif()

# ------------------------------------------------------------------------------
# E2E pipelines – segmentation/depth examples
# ------------------------------------------------------------------------------
add_pipeline_test(
  depth_anything_v2_example_test
  e2e_pipelines/depth_estimation/depth_anything_v2_example_test.cpp
  ARGS ${CMAKE_SOURCE_DIR}
)

if (NOT depth_anything_v2_example_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_tests_properties(depth_anything_v2_example_test
    PROPERTIES LABELS "e2e;examples;depth"
  )
endif()

add_pipeline_test(
  fcn_hrnet_seg_overlay_example_test
  e2e_pipelines/segmentation/fcn_hrnet_seg_overlay_example_test.cpp
  ARGS ${CMAKE_SOURCE_DIR}
)

if (NOT fcn_hrnet_seg_overlay_example_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_tests_properties(fcn_hrnet_seg_overlay_example_test
    PROPERTIES LABELS "e2e;examples;segmentation"
  )
endif()

add_pipeline_test(
  yolov5_seg_overlay_example_test
  e2e_pipelines/segmentation/yolov5_seg_overlay_example_test.cpp
  ARGS ${CMAKE_SOURCE_DIR}
)

if (NOT yolov5_seg_overlay_example_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_tests_properties(yolov5_seg_overlay_example_test
    PROPERTIES LABELS "e2e;examples;segmentation"
  )
endif()

add_pipeline_test(
  yolov8n_seg_detess_simple_example_test
  e2e_pipelines/segmentation/yolov8n_seg_detess_simple_example_test.cpp
  ARGS ${CMAKE_SOURCE_DIR}
)

if (NOT yolov8n_seg_detess_simple_example_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_tests_properties(yolov8n_seg_detess_simple_example_test
    PROPERTIES LABELS "e2e;examples;segmentation"
  )
endif()

# ------------------------------------------------------------------------------
# E2E pipelines – image classification
# ------------------------------------------------------------------------------
add_pipeline_test(
  model_resnet50_multi_test
  e2e_pipelines/img_classification/model_resnet50_multi_test.cpp
)

if (NOT model_resnet50_multi_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_tests_properties(model_resnet50_multi_test
    PROPERTIES LABELS "e2e;img_classification"
  )
endif()

add_pipeline_test(
  model_unique_names_test
  e2e_pipelines/img_classification/model_unique_names_test.cpp
)

if (NOT model_unique_names_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_tests_properties(model_unique_names_test
    PROPERTIES LABELS "e2e;img_classification"
  )
endif()

add_pipeline_test(
  resnet50_modelpack_test
  e2e_pipelines/img_classification/resnet50_modelpack_test.cpp
)

if (NOT resnet50_modelpack_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST resnet50_modelpack_test APPEND PROPERTY RESOURCE_LOCK "rtsp")
  set_property(TEST resnet50_modelpack_test
    APPEND PROPERTY ENVIRONMENT "SIMA_RTSP_PORT_BASE=8557"
  )
  add_test(
    NAME resnet50_modelpack_direct
    COMMAND resnet50_modelpack_test --goldfish --only-direct
  )
  add_test(
    NAME resnet50_modelpack_jpeg
    COMMAND resnet50_modelpack_test --goldfish --only-jpeg
  )
  add_test(
    NAME resnet50_modelpack_rtsp
    COMMAND resnet50_modelpack_test --goldfish --only-rtsp
  )

  set_tests_properties(
    resnet50_modelpack_direct
    resnet50_modelpack_jpeg
    resnet50_modelpack_rtsp
    PROPERTIES
      LABELS "e2e;img_classification"
      RESOURCE_LOCK "dispatcher"
  )
  set_property(TEST resnet50_modelpack_rtsp APPEND PROPERTY RESOURCE_LOCK "rtsp")
  set_property(TEST resnet50_modelpack_rtsp
    APPEND PROPERTY ENVIRONMENT "SIMA_RTSP_PORT_BASE=8567"
  )
endif()

add_pipeline_test(
  rtsp_pull_frame_test
  e2e_pipelines/img_classification/rtsp_pull_frame_test.cpp
  ARGS --local --image ${CMAKE_SOURCE_DIR}/test.jpg
)

if (NOT rtsp_pull_frame_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_tests_properties(rtsp_pull_frame_test
    PROPERTIES LABELS "e2e;img_classification"
  )
  set_property(TEST rtsp_pull_frame_test APPEND PROPERTY RESOURCE_LOCK "rtsp")
endif()

add_pipeline_test(
  hybrid_graph_rtsp_resnet50_tput_test
  e2e_pipelines/img_classification/hybrid_graph_rtsp_resnet50_tput_test.cpp
)

if (NOT hybrid_graph_rtsp_resnet50_tput_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_tests_properties(hybrid_graph_rtsp_resnet50_tput_test
    PROPERTIES LABELS "e2e;img_classification"
  )
  set_property(TEST hybrid_graph_rtsp_resnet50_tput_test APPEND PROPERTY RESOURCE_LOCK "rtsp")
endif()

set(SIMANEAT_TEST_INSTALL_TARGETS ${PIPELINESESSION_TEST_TARGETS})

if (SIMANEAT_SANITIZER_GATE_ONLY_EXTRAS)
  set(SIMANEAT_SANITIZER_GATE_TESTS
    unit_modalix_contract_preflight_test
  )

  if (SIMA_ENABLE_TSAN)
    list(APPEND SIMANEAT_SANITIZER_GATE_TESTS
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
  else()
    list(APPEND SIMANEAT_SANITIZER_GATE_TESTS
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
  endif()

  set(SIMANEAT_TEST_INSTALL_TARGETS "")
  foreach(_t IN LISTS SIMANEAT_SANITIZER_GATE_TESTS)
    if (TARGET "${_t}")
      list(APPEND SIMANEAT_TEST_INSTALL_TARGETS "${_t}")
    else()
      message(WARNING "Sanitizer gate install target not found: ${_t}")
    endif()
  endforeach()

  list(REMOVE_DUPLICATES SIMANEAT_TEST_INSTALL_TARGETS)
  message(STATUS "Sanitizer gate-only extras install enabled (SIMA_ENABLE_TSAN=${SIMA_ENABLE_TSAN})")
  message(STATUS "Installing ${SIMANEAT_TEST_INSTALL_TARGETS}")
endif()

install(
  TARGETS ${SIMANEAT_TEST_INSTALL_TARGETS}
  RUNTIME DESTINATION lib/sima-neat/tests
  COMPONENT extras
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION share/sima-neat/tests
  COMPONENT extras
  FILES_MATCHING
    PATTERN "*.cpp"
    PATTERN "*.h"
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/assets/mpk/
  DESTINATION share/sima-neat/tests/assets/mpk
  COMPONENT extras
  FILES_MATCHING
    PATTERN "*.json"
    PATTERN "*.tar"
    PATTERN "*.mpk"
)

# Keep standard extras behavior (full tests tree install), but trim payload
# aggressively for sanitizer gate artifact builds.
if (SIMANEAT_SANITIZER_GATE_ONLY_EXTRAS)
  install(
    DIRECTORY ${CMAKE_BINARY_DIR}/tests/
    DESTINATION lib/sima-neat/tests
    COMPONENT extras
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING
      PATTERN "CTestTestfile.cmake"
  )
else()
  install(
    DIRECTORY ${CMAKE_BINARY_DIR}/tests/
    DESTINATION lib/sima-neat/tests
    COMPONENT extras
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING
      PATTERN "*"
  )
endif()
