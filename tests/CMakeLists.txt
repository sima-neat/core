cmake_minimum_required(VERSION 3.16)

# ------------------------------------------------------------------------------
# PipelineSession tests
# ------------------------------------------------------------------------------
message(STATUS "Configuring PipelineSession tests")

enable_testing()
include(CMakeParseArguments)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

option(SIMANEAT_CTEST_FOR_DEVKIT
  "Emit CTestTestfile with absolute DevKit /usr/lib paths (skip configure-time EXISTS checks)"
  OFF)

if (SIMANEAT_CTEST_FOR_DEVKIT)
  set(SIMA_TEST_NEAT_PLUGIN_DIR "/usr/lib/aarch64-linux-gnu/neat/gst-plugins")
  set(SIMA_TEST_NEAT_LD_LIBRARY_PATH
    "/usr/lib/aarch64-linux-gnu/neat/gst-plugins:/usr/lib/aarch64-linux-gnu/neat/runtime:/lib/aarch64-linux-gnu/neat/runtime")
  message(STATUS "  [tests] SIMANEAT_CTEST_FOR_DEVKIT=ON; SIMA_TEST_NEAT_PLUGIN_DIR=${SIMA_TEST_NEAT_PLUGIN_DIR}")
  message(STATUS "  [tests] SIMA_TEST_NEAT_LD_LIBRARY_PATH=${SIMA_TEST_NEAT_LD_LIBRARY_PATH}")
else()
set(SIMA_TEST_NEAT_PLUGIN_DIR "${SIMANEAT_CTEST_NEAT_PLUGIN_DIR}" CACHE PATH
    "NEAT GStreamer plugin directory used by tests")
if (NOT SIMA_TEST_NEAT_PLUGIN_DIR)
  set(_SIMA_DEFAULT_NEAT_PLUGIN_DIRS
    "/usr/lib/aarch64-linux-gnu/neat/gst-plugins"
    "/lib/aarch64-linux-gnu/neat/gst-plugins"
    "/usr/lib/neat/gst-plugins"
    "/lib/neat/gst-plugins"
  )
  foreach(_sima_candidate IN LISTS _SIMA_DEFAULT_NEAT_PLUGIN_DIRS)
    if (EXISTS "${_sima_candidate}")
      set(SIMA_TEST_NEAT_PLUGIN_DIR "${_sima_candidate}")
      break()
    endif()
  endforeach()
endif()
message(STATUS "  [tests] SIMA_TEST_NEAT_PLUGIN_DIR=${SIMA_TEST_NEAT_PLUGIN_DIR}")

set(_SIMA_TEST_NEAT_RUNTIME_DIR_CANDIDATES
  "${SIMA_TEST_NEAT_PLUGIN_DIR}"
  ${SIMANEAT_EXPORTED_NEAT_RUNTIME_DIR_CANDIDATES}
  ${SIMANEAT_REPO_NEAT_RUNTIME_DIR_CANDIDATES}
  ${SIMANEAT_INSTALLED_NEAT_RUNTIME_DIR_CANDIDATES}
  "/usr/lib/aarch64-linux-gnu/neat/runtime"
  "/lib/aarch64-linux-gnu/neat/runtime"
  "/usr/lib/neat/runtime"
  "/lib/neat/runtime")
set(_SIMA_TEST_NEAT_RUNTIME_DIRS "")
foreach(_sima_runtime_dir IN LISTS _SIMA_TEST_NEAT_RUNTIME_DIR_CANDIDATES)
  if (EXISTS "${_sima_runtime_dir}")
    list(APPEND _SIMA_TEST_NEAT_RUNTIME_DIRS "${_sima_runtime_dir}")
  endif()
endforeach()
if (DEFINED ENV{LD_LIBRARY_PATH} AND NOT "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
  list(APPEND _SIMA_TEST_NEAT_RUNTIME_DIRS "$ENV{LD_LIBRARY_PATH}")
endif()
list(REMOVE_DUPLICATES _SIMA_TEST_NEAT_RUNTIME_DIRS)
string(JOIN ":" SIMA_TEST_NEAT_LD_LIBRARY_PATH ${_SIMA_TEST_NEAT_RUNTIME_DIRS})
message(STATUS "  [tests] SIMA_TEST_NEAT_LD_LIBRARY_PATH=${SIMA_TEST_NEAT_LD_LIBRARY_PATH}")
endif()

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
# Python test source compilation
# ------------------------------------------------------------------------------
#
# Core test coverage is not limited to CTest-registered C++ executables: keep the
# Python-side test/tutorial sources syntax-checked during a normal test build too.
file(GLOB_RECURSE SIMANEAT_PYTHON_TEST_SOURCES
  CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/python/*.py"
  "${CMAKE_SOURCE_DIR}/tests/*.py"
  "${CMAKE_SOURCE_DIR}/tutorials/*.py"
)

set(SIMANEAT_PYTHON_TEST_COMPILE_DIRS
  "${CMAKE_SOURCE_DIR}/python"
  "${CMAKE_SOURCE_DIR}/tests"
  "${CMAKE_SOURCE_DIR}/tutorials"
)

set(SIMANEAT_PYTHON_TEST_PYCACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/pycache")
set(SIMANEAT_PYTHON_TEST_COMPILE_STAMP
  "${CMAKE_CURRENT_BINARY_DIR}/python_tests_compiled.stamp")

add_custom_command(
  OUTPUT "${SIMANEAT_PYTHON_TEST_COMPILE_STAMP}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${SIMANEAT_PYTHON_TEST_PYCACHE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E env
    "PYTHONPYCACHEPREFIX=${SIMANEAT_PYTHON_TEST_PYCACHE_DIR}"
    ${Python3_EXECUTABLE} -m compileall -q ${SIMANEAT_PYTHON_TEST_COMPILE_DIRS}
  COMMAND ${CMAKE_COMMAND} -E touch "${SIMANEAT_PYTHON_TEST_COMPILE_STAMP}"
  DEPENDS ${SIMANEAT_PYTHON_TEST_SOURCES}
  COMMENT "Compiling core Python test and tutorial sources"
  VERBATIM
)

add_custom_target(compile_core_python_tests ALL
  DEPENDS "${SIMANEAT_PYTHON_TEST_COMPILE_STAMP}"
)

if (TARGET _pyneat_core)
  add_dependencies(compile_core_python_tests _pyneat_core)
endif()

# ------------------------------------------------------------------------------
# Helper: add strict test executable + ctest entry
# ------------------------------------------------------------------------------
function(add_pipeline_test_strict name)
  if (name IN_LIST PIPELINESESSION_TEST_BLACKLIST)
    message(STATUS "  [tests] Skipping blacklisted test: ${name}")
    return()
  endif()

  set(options LOCK_DISPATCHER)
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

  simaneat_append_ctest_env(${name})

  if (APT_LOCK_DISPATCHER)
    set_tests_properties(${name} PROPERTIES
      RESOURCE_LOCK "dispatcher"
    )
  endif()

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

function(append_pipeline_test_env name)
  simaneat_append_ctest_env(${name})
endfunction()

# ------------------------------------------------------------------------------
# Unit tests (1:1 cpp mapping)
# ------------------------------------------------------------------------------
add_pipeline_test(
  hybrid_graph_rtsp_resnet50_tput_test
  e2e_pipelines/img_classification/hybrid_graph_rtsp_resnet50_tput_test.cpp
)

if (NOT hybrid_graph_rtsp_resnet50_tput_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST hybrid_graph_rtsp_resnet50_tput_test APPEND PROPERTY LABELS "e2e;img_classification")
  set_property(TEST hybrid_graph_rtsp_resnet50_tput_test APPEND PROPERTY RESOURCE_LOCK "rtsp")
endif()

set(UNIT_TESTS
  unit_caps_bridge_test
  unit_contract_compiler_boxdecode_test
  unit_contract_compiler_mla_test
  unit_contract_compiler_processcvu_test
  unit_contract_architecture_guard_test
  unit_contract_input_only_rules_test
  unit_contract_render_manifest_equivalence_test
  unit_contracts_test
  unit_boxdecode_render_manifest_from_model_test
  unit_model_route_retarget_helper_test
  unit_dequant_processcvu_contract_equivalence_test
  unit_dequant_node_fragment_test
  unit_detection_types_bbox_test
  unit_detessdequant_output_info_test
  unit_route_capability_mpk_static_contract_test
  unit_route_plan_detess_adapter_test
  unit_dispatcher_recovery_test
  unit_dispatcher_watchdog_test
  unit_error_code_taxonomy_test
  unit_encoded_sample_test
  unit_env_util_profile_test
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
  unit_graph_stage_emitter_test
  unit_graph_stop_sequencing_test
  unit_graph_strict_sync_store_test
  unit_graph_strict_sync_test
  unit_genai_prompt_builder_test
  unit_genai_image_types_test
  unit_genai_header_compile_surface_test
  unit_genai_text_tensor_test
  unit_header_neat_compile_surface_test
  unit_verbose_options_surface_test
  unit_gst_data_adapter_edge_test
  unit_groups_test
  unit_gst_helpers_test
  unit_gst_init_guard_test
  unit_gst_video_meta_test
  unit_inputstream_dynamic_policy_matrix_test
  unit_input_policy_test
  unit_input_planner_semantics_test
  unit_preprocess_meta_validation_test
  unit_preprocess_meta_template_dynamic_test
  unit_preproc_contract_rules_test
  unit_inputstream_stop_timeout_detach_test
  unit_inputappsrc_stop_race_test
  unit_neatdecoder_keepalive_test
  unit_neatencoder_keepalive_test
  unit_neatmodel_async_api_test
  unit_neattensor_core_test
  unit_neattensor_cvmat_test
  unit_tensorbuffer_packed_runtime_parent_test
  unit_tensorbuffer_padded_hwc_gap_inference_test
  unit_tensorbuffer_runtime_contract_identity_overlay_test
  unit_tensorbuffer_strided_parent_view_materialize_test
  unit_modelpack_extract_test
  unit_asset_utils_readable_model_tar_test
  unit_model_infer_output_name_test
  unit_model_input_spec_contract_test
  unit_model_output_spec_contract_test
  unit_post_region_materialization_test
  unit_model_metadata_test
  unit_model_route_planner_test
  unit_resnet50_mla_handoff_segment_test
  unit_cast_fanout_contract_test
  unit_mla_boundary_output_contract_test
  unit_route_graph_slice_kernel_test
  unit_modelpack_mla_handoff_segment_test
  unit_model_stage_fragments_test
  unit_modalix_contract_preflight_test
  unit_mpk_graph_dump_test
  unit_mpk_fuse_sibling_pre_mla_test
  unit_mpk_contract_legacy_ingress_test
  unit_mla_elf_io_topology_test
  unit_mpk_loader_test
  unit_mpk_pipeline_adapter_test
  unit_outputspec_test
  unit_pcie_nodes_test
  pull_timeout_regression_test
  property_config_json_test
  property_pipeline_string_test
  unit_pipeline_internal_build_test
  unit_power_telemetry_test
  unit_policy_defaults_test
  unit_run_api_matrix_test
  unit_run_lifecycle_teardown_test
  unit_run_overflow_policy_parity_test
  unit_runtime_metrics_test
  unit_samplespec_test
  unit_sample_bundle_holder_test
  unit_session_api_surface_test
  unit_session_build_input_strategy_test
  unit_session_io_contract_test
  unit_session_naming_transform_test
  session_stage_contract_equivalence_test
  unit_simaai_guard_test
  unit_simaai_allocator_share_test
  unit_static_spec_builders_dtype_token_test
  unit_sima_boxdecode_static_contract_extractor_test
  unit_boxdecode_uses_upstream_cast_contract_test
  unit_sima_boxdecode_node_fragment_test
  unit_kernel200_envelope_guard_test
  unit_processcvu_binding_table_uses_runtime_outputs_test
  unit_processcvu_preproc_runtime_vs_exposed_outputs_test
  unit_stage_transform_rule_registry_test
  unit_sima_plugin_static_manifest_parse_test
  unit_sima_plugin_static_manifest_contract_test
  unit_sima_plugin_context_injection_test
  unit_sima_plugin_context_late_child_injection_test
  unit_sima_plugin_context_copy_test
  unit_sima_plugin_context_attach_lifetime_test
  unit_sima_plugin_context_replacement_test
  unit_sima_plugin_context_destroyed_pipeline_test
  unit_sima_plugins_test
  unit_pipeline_sequence_parse_test
  unit_processmla_model_load_failure_test
  unit_processmla_requires_explicit_output_contract_test
  unit_tensor_set_meta_route_contract_test
  unit_tensorbuffer_bind_preserve_source_view_test
  unit_tensorbuffer_publish_view_segment_name_precedence_test
  unit_sample_tensor_set_materialized_parent_offsets_test
  unit_yolov8_mla_handoff_segment_test
  unit_yolov8_contract_subset_test
  unit_agg_model_managed_no_standalone_json_test
  unit_tensor_list_api_test
  unit_tensor_conversion_policy_test
  unit_tensor_image_mapping_test
  unit_transport_shared_parent_input_test
  unit_transport_output_nodes_test
  unit_node_metadata_sender_test
  unit_nodes_test
  unit_plugin_failure_messages_test
  unit_video_sender_test
)

if (SIMA_WITH_OPENCV AND OPENCV_FOUND)
  list(APPEND UNIT_TESTS unit_tensor_from_cv_mat_exactness_test)
endif()

foreach(test ${UNIT_TESTS})
  add_pipeline_test(
    ${test}
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_testing/${test}.cpp
  )

  if (NOT test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
    set_property(TEST ${test} APPEND PROPERTY LABELS "unit")
  endif()
endforeach()

add_pipeline_test(
  unit_builder_test
  ${CMAKE_CURRENT_SOURCE_DIR}/unit_testing/unit_builder_test.cpp
)

if (NOT unit_builder_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST unit_builder_test APPEND PROPERTY LABELS "unit")
endif()

if (TARGET unit_power_telemetry_test)
  target_include_directories(unit_power_telemetry_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
endif()

if (TARGET unit_plugin_failure_messages_test)
  set_property(TEST unit_plugin_failure_messages_test APPEND PROPERTY
    ENVIRONMENT "GST_REGISTRY_1_0=${CMAKE_BINARY_DIR}/sima_gst_registry_unit_plugin_failure_messages_test_v2.bin")
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
    set_property(TEST ${test} APPEND PROPERTY LABELS "unit")
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
  output_appsink_policy_test
  gst_data_adapter_runtime_regression_test
  preproc_standalone_test
  session_naming_pipeline_integration_test
  session_rtsp_lifecycle_regression_test
  session_validate_report_regression_test
  stage_routing_regression_test
  metadata_sender_udp_integration_test
  udp_output_loopback_integration_test
)

foreach(test ${E2E_TESTS})
  add_pipeline_test(
    ${test}
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_testing/${test}.cpp
  )

  if (NOT test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
    set_property(TEST ${test} APPEND PROPERTY LABELS "e2e")
  endif()
endforeach()

add_pipeline_test_long(
  preproc_build_input_repro_test
  ${CMAKE_CURRENT_SOURCE_DIR}/e2e_testing/preproc_build_input_repro_test.cpp
)

if (NOT preproc_build_input_repro_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST preproc_build_input_repro_test APPEND PROPERTY LABELS "e2e")
endif()

add_pipeline_test_long(
  mnist_preproc_channels1_accuracy_test
  e2e_testing/mnist_preproc_channels1_accuracy_test.cpp
)

if (NOT mnist_preproc_channels1_accuracy_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST mnist_preproc_channels1_accuracy_test APPEND PROPERTY LABELS "e2e;modelzoo;long")
endif()

add_pipeline_test_long(
  mnist_async_consumer_visibility_control_test
  e2e_testing/mnist_async_consumer_visibility_control_test.cpp
)

if (NOT mnist_async_consumer_visibility_control_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST mnist_async_consumer_visibility_control_test APPEND PROPERTY LABELS "e2e;modelzoo;long;perf")
endif()

add_pipeline_test_long(
  modelzoo_bulk_models_test
  e2e_testing/modelzoo_bulk_models_test.cpp
)

if (NOT modelzoo_bulk_models_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST modelzoo_bulk_models_test APPEND PROPERTY LABELS "e2e;modelzoo;long")
endif()

add_pipeline_test_long(
  genai_llm_run_test
  e2e_testing/genai_llm_run_test.cpp
)

if (NOT genai_llm_run_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST genai_llm_run_test APPEND PROPERTY LABELS "e2e;genai;llima;long")
  set_property(TEST genai_llm_run_test APPEND PROPERTY RESOURCE_LOCK "dispatcher")
  set_tests_properties(genai_llm_run_test PROPERTIES DISABLED TRUE)
endif()

add_pipeline_test_long(
  genai_vlm_run_test
  SOURCES e2e_testing/genai_vlm_run_test.cpp
  ARGS ${CMAKE_SOURCE_DIR}
)

if (NOT genai_vlm_run_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST genai_vlm_run_test APPEND PROPERTY LABELS "e2e;genai;llima;vlm;long")
  set_property(TEST genai_vlm_run_test APPEND PROPERTY RESOURCE_LOCK "dispatcher")
  set_tests_properties(genai_vlm_run_test PROPERTIES DISABLED TRUE)
endif()

add_pipeline_test_long(
  genai_graph_llm_run_test
  e2e_testing/genai_graph_llm_run_test.cpp
)

if (NOT genai_graph_llm_run_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST genai_graph_llm_run_test APPEND PROPERTY LABELS "e2e;genai;llima;graph;long")
  set_property(TEST genai_graph_llm_run_test APPEND PROPERTY RESOURCE_LOCK "dispatcher")
  set_tests_properties(genai_graph_llm_run_test PROPERTIES DISABLED TRUE)
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
  set_property(TEST preproc_dynamic_renegotiate_test APPEND PROPERTY LABELS "e2e")
endif()

# ------------------------------------------------------------------------------
# Stress tests
# ------------------------------------------------------------------------------
set(STRESS_TESTS
  stress_graph_strict_sync_join_race_test
  stress_udp_metadata_burst_test
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
    set_property(TEST ${test} APPEND PROPERTY LABELS "stress")
  endif()
endforeach()

# ------------------------------------------------------------------------------
# Security tests (MPK / archive / JSON abuse)
# ------------------------------------------------------------------------------
set(SECURITY_TESTS
  security_modelpack_matrix_test
)

foreach(test ${SECURITY_TESTS})
  add_pipeline_test(
    ${test}
    ${CMAKE_CURRENT_SOURCE_DIR}/security/${test}.cpp
  )

  if (NOT test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
    set_property(TEST ${test} APPEND PROPERTY LABELS "security")
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
  set_property(TEST file_read_test APPEND PROPERTY LABELS "e2e")
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
      unit_modalix_contract_preflight_test
    unit_run_api_matrix_test
    unit_contracts_test
    unit_detection_types_bbox_test
    unit_gst_data_adapter_edge_test
    unit_nodes_test
    unit_outputspec_test
      unit_model_input_spec_contract_test
    unit_model_output_spec_contract_test
    unit_model_metadata_test
    unit_model_route_planner_test
    unit_model_stage_fragments_test
    session_naming_pipeline_integration_test
    gst_data_adapter_runtime_regression_test
    unit_modelpack_extract_test
    unit_pipeline_sequence_parse_test
    unit_mpk_loader_test
    unit_mpk_pipeline_adapter_test
    session_rtsp_lifecycle_regression_test
    session_validate_report_regression_test
    security_modelpack_matrix_test)
  if (NOT test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
    set_property(TEST ${test} APPEND PROPERTY LABELS "crash-gate;regression")
  endif()
endforeach()

# ------------------------------------------------------------------------------
# Runtime-core labels
# ------------------------------------------------------------------------------
foreach(test
    unit_run_api_matrix_test
    unit_model_infer_output_name_test
    unit_model_input_spec_contract_test
    unit_model_output_spec_contract_test
    unit_model_metadata_test
    unit_model_stage_fragments_test
    unit_session_io_contract_test
    unit_session_naming_transform_test
    unit_pipeline_internal_build_test
    unit_gst_data_adapter_edge_test
    unit_caps_bridge_test
    unit_simaai_guard_test
    unit_tensor_conversion_policy_test
    unit_tensor_image_mapping_test
    unit_detection_types_bbox_test
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
    unit_run_api_matrix_test
    unit_session_io_contract_test
    unit_graph_compiler_topology_test
    unit_graph_compiler_spec_validation_test
        unit_modelpack_extract_test
    unit_pipeline_sequence_parse_test
    unit_mpk_loader_test
    security_modelpack_matrix_test
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
  set_property(TEST async_yolov8_tput_test APPEND PROPERTY LABELS "e2e;obj_detection")
endif()

add_pipeline_test(
  sync_yolov8_test
  e2e_pipelines/obj_detection/sync_yolov8_test.cpp
  e2e_pipelines/e2e_utils.cpp
  e2e_pipelines/obj_detection/obj_detection_utils.cpp
  ARGS ${CMAKE_SOURCE_DIR}
)

if (NOT sync_yolov8_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST sync_yolov8_test APPEND PROPERTY LABELS "e2e;obj_detection")
endif()

add_pipeline_test(
  sync_yolov8_test_detessdequant
  e2e_pipelines/obj_detection/sync_yolov8_test_detessdequant.cpp
  e2e_pipelines/e2e_utils.cpp
  e2e_pipelines/obj_detection/obj_detection_utils.cpp
  ARGS ${CMAKE_SOURCE_DIR}
)

if (NOT sync_yolov8_test_detessdequant IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST sync_yolov8_test_detessdequant APPEND PROPERTY LABELS "e2e;obj_detection")
endif()

add_pipeline_test(
  async_yolov11_pose_bf16_tput_test
  e2e_pipelines/obj_detection/async_yolov11_pose_bf16_tput_test.cpp
  e2e_pipelines/e2e_utils.cpp
  e2e_pipelines/obj_detection/obj_detection_utils.cpp
  ARGS ${CMAKE_SOURCE_DIR}
)

if (NOT async_yolov11_pose_bf16_tput_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST async_yolov11_pose_bf16_tput_test APPEND PROPERTY LABELS "e2e;obj_detection")
  set_tests_properties(async_yolov11_pose_bf16_tput_test PROPERTIES DISABLED TRUE)
endif()

add_pipeline_test(
  yolov8_variant_route_matrix_test
  e2e_pipelines/obj_detection/yolov8_variant_route_matrix_test.cpp
  e2e_pipelines/e2e_utils.cpp
  e2e_pipelines/obj_detection/obj_detection_utils.cpp
)

if (NOT yolov8_variant_route_matrix_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST yolov8_variant_route_matrix_test APPEND PROPERTY LABELS "e2e;obj_detection")
endif()

add_pipeline_test(
  preproc_yolov8_matrix_test
  e2e_pipelines/obj_detection/preproc_yolov8_matrix_test.cpp
  e2e_pipelines/e2e_utils.cpp
  e2e_pipelines/obj_detection/obj_detection_utils.cpp
  ARGS ${CMAKE_SOURCE_DIR}
)

if (NOT preproc_yolov8_matrix_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST preproc_yolov8_matrix_test APPEND PROPERTY LABELS "e2e;obj_detection")
endif()

add_pipeline_test(
  graphpipes_yolo_metadata_sender_16stream_test
  e2e_pipelines/obj_detection/graphpipes_yolo_metadata_sender_16stream_test.cpp
  e2e_pipelines/e2e_utils.cpp
  e2e_pipelines/obj_detection/obj_detection_utils.cpp
  ARGS ${CMAKE_SOURCE_DIR}
)

if (NOT graphpipes_yolo_metadata_sender_16stream_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST graphpipes_yolo_metadata_sender_16stream_test APPEND PROPERTY LABELS "e2e;obj_detection")
endif()

# ------------------------------------------------------------------------------
# E2E pipelines – image classification
# ------------------------------------------------------------------------------
add_pipeline_test(
  model_resnet50_multi_test
  e2e_pipelines/img_classification/model_resnet50_multi_test.cpp
)

if (NOT model_resnet50_multi_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST model_resnet50_multi_test APPEND PROPERTY LABELS "e2e;img_classification")
endif()

add_pipeline_test(
  model_unique_names_test
  e2e_pipelines/img_classification/model_unique_names_test.cpp
)

if (NOT model_unique_names_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST model_unique_names_test APPEND PROPERTY LABELS "e2e;img_classification")
endif()

if (NOT resnet50_modelpack_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  add_executable(
    resnet50_modelpack_test
    e2e_pipelines/img_classification/resnet50_modelpack_test.cpp
  )
  add_dependencies(resnet50_modelpack_test prepare_test_fixtures)
  target_link_libraries(resnet50_modelpack_test PRIVATE sima_neat_internal)
  target_include_directories(resnet50_modelpack_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/e2e_pipelines
      ${CMAKE_CURRENT_SOURCE_DIR}/e2e_pipelines/obj_detection
  )
  list(APPEND PIPELINESESSION_TEST_TARGETS resnet50_modelpack_test)

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

  foreach(test resnet50_modelpack_direct resnet50_modelpack_jpeg resnet50_modelpack_rtsp)
    set_property(TEST ${test} APPEND PROPERTY LABELS "e2e;img_classification")
    set_property(TEST ${test} APPEND PROPERTY RESOURCE_LOCK "dispatcher")
  endforeach()
  append_pipeline_test_env(resnet50_modelpack_direct)
  append_pipeline_test_env(resnet50_modelpack_jpeg)
  append_pipeline_test_env(resnet50_modelpack_rtsp)
  set_property(TEST resnet50_modelpack_direct
    APPEND PROPERTY ENVIRONMENT "SIMA_RTSP_PORT_BASE=8557"
  )
  set_property(TEST resnet50_modelpack_jpeg
    APPEND PROPERTY ENVIRONMENT "SIMA_RTSP_PORT_BASE=8557"
  )
  set_property(TEST resnet50_modelpack_rtsp APPEND PROPERTY RESOURCE_LOCK "rtsp")
  set_property(TEST resnet50_modelpack_rtsp
    APPEND PROPERTY ENVIRONMENT "SIMA_RTSP_PORT_BASE=8567"
  )
endif()

add_pipeline_test_long(
  resnet50_neatargmax_test
  e2e_pipelines/img_classification/resnet50_neatargmax_test.cpp
)

if (NOT resnet50_neatargmax_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST resnet50_neatargmax_test APPEND PROPERTY LABELS "e2e;img_classification;long")
  set_tests_properties(resnet50_neatargmax_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

add_pipeline_test(
  rtsp_pull_frame_test
  e2e_pipelines/img_classification/rtsp_pull_frame_test.cpp
  ARGS --local --image ${CMAKE_SOURCE_DIR}/test.jpg
)

if (NOT rtsp_pull_frame_test IN_LIST PIPELINESESSION_TEST_BLACKLIST)
  set_property(TEST rtsp_pull_frame_test APPEND PROPERTY LABELS "e2e;img_classification")
  set_property(TEST rtsp_pull_frame_test APPEND PROPERTY RESOURCE_LOCK "rtsp")
endif()

add_executable(
  tess_a65_vs_ev74_bench
  e2e_testing/tess_a65_vs_ev74_bench.cpp
)
add_dependencies(tess_a65_vs_ev74_bench prepare_test_fixtures)
target_link_libraries(
  tess_a65_vs_ev74_bench
  PRIVATE sima_neat_internal
)
target_include_directories(
  tess_a65_vs_ev74_bench
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_pipelines
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_pipelines/obj_detection
)

install(
  TARGETS ${PIPELINESESSION_TEST_TARGETS}
  RUNTIME DESTINATION lib/sima-neat/tests
  COMPONENT extras
)

# Source-tree image fixture used by several pipeline tests (test.jpg). Ship it
# via the extras package so installed-from-deb test runs find it after the
# build-time CMAKE_SOURCE_DIR/test.jpg path is rebased by CI to
# ${prefix}/share/sima-neat/test-assets/test.jpg.
install(
  FILES "${CMAKE_SOURCE_DIR}/test.jpg"
  DESTINATION share/sima-neat/test-assets
  COMPONENT extras
)

# MPK + decoder fixtures used by the unit_mpk_loader / unit_modelpack_extract
# / security_modelpack_matrix / unit_pipeline_sequence_parse suites. Without
# this, tests on the CI runner (and any deb-installed environment) call
# MpKLoader::inspect/extract against a non-existent .mpk path and trip the
# "archive path does not exist" InvalidArchive branch — masking the real
# error-class assertions the tests are written for.
install(
  DIRECTORY "${CMAKE_SOURCE_DIR}/tests/assets/mpk/"
  DESTINATION lib/sima-neat/test-assets/mpk
  COMPONENT extras
)
install(
  DIRECTORY "${CMAKE_SOURCE_DIR}/tests/assets/decoder/"
  DESTINATION lib/sima-neat/test-assets/decoder
  COMPONENT extras
)
install(
  FILES "${CMAKE_BINARY_DIR}/test-fixtures/runtime_manifest_installed.json"
  DESTINATION lib/sima-neat/test-fixtures
  COMPONENT extras
  RENAME runtime_manifest.json
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION share/sima-neat/tests
  COMPONENT extras
  FILES_MATCHING
    PATTERN "*.cpp"
    PATTERN "*.h"
)

# Install built test binaries from the build tree (for extras packaging)
install(
  DIRECTORY ${CMAKE_BINARY_DIR}/tests/
  DESTINATION lib/sima-neat/tests
  COMPONENT extras
  USE_SOURCE_PERMISSIONS
  FILES_MATCHING
    PATTERN "*"
)
