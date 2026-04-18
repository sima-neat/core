if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.16)
  project(SimaNeatTutorials LANGUAGES CXX)
  enable_testing()

  find_package(SimaNeat REQUIRED CONFIG)
  set(SIMANEAT_TUTORIAL_LINK_TARGET SimaNeat::sima_neat)
else()
  set(SIMANEAT_TUTORIAL_LINK_TARGET sima_neat)
endif()

# Tutorial executables and CTest registration.

include(CMakeParseArguments)

set(PIPELINESESSION_TUTORIAL_TARGETS "")

function(add_tutorial exe source)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs LABELS)
  cmake_parse_arguments(AT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_executable(${exe} ${source})
  target_link_libraries(${exe} PRIVATE ${SIMANEAT_TUTORIAL_LINK_TARGET})
  target_include_directories(${exe} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_test(NAME ${exe} COMMAND ${exe})
  set_tests_properties(${exe} PROPERTIES RESOURCE_LOCK "dispatcher")

  set(label_list "tutorial")
  if (AT_LABELS)
    list(APPEND label_list ${AT_LABELS})
  endif()
  set_tests_properties(${exe} PROPERTIES LABELS "${label_list}")

  list(APPEND PIPELINESESSION_TUTORIAL_TARGETS ${exe})
endfunction()

# Tutorials are added as they are implemented.
add_tutorial(tutorial_0001_quickstart_image_to_tensor
  tutorial_0001_quickstart_image_to_tensor.cpp
  LABELS asset
)

add_tutorial(tutorial_0002_pipeline_session_build_run
  tutorial_0002_pipeline_session_build_run.cpp
  LABELS fast
)

add_tutorial(tutorial_0003_nodes_and_groups
  tutorial_0003_nodes_and_groups.cpp
  LABELS fast
)

add_tutorial(tutorial_0004_samples_and_neattensor
  tutorial_0004_samples_and_neattensor.cpp
  LABELS fast
)

add_tutorial(tutorial_0005_output_tensor_options
  tutorial_0005_output_tensor_options.cpp
  LABELS fast
)

add_tutorial(tutorial_0006_async_push_pull
  tutorial_0006_async_push_pull.cpp
  LABELS fast
)

add_tutorial(tutorial_0007_caps_negotiation
  tutorial_0007_caps_negotiation.cpp
  LABELS fast
)

add_tutorial(tutorial_0008_input_groups
  tutorial_0008_input_groups.cpp
  LABELS asset
)

add_tutorial(tutorial_0009_pipeline_save_load
  tutorial_0009_pipeline_save_load.cpp
  LABELS fast
)

add_tutorial(tutorial_0010_diagnostics_debug
  tutorial_0010_diagnostics_debug.cpp
  LABELS fast
)

add_tutorial(tutorial_0011_mpk_yolov8_pipeline
  tutorial_0011_mpk_yolov8_pipeline.cpp
  LABELS mpk
)

add_tutorial(tutorial_0012_stage_api
  tutorial_0012_stage_api.cpp
  LABELS mpk
)

add_tutorial(tutorial_0013_model_resnet50
  tutorial_0013_model_resnet50.cpp
  LABELS mpk
)

add_tutorial(tutorial_0014_rtsp_server
  tutorial_0014_rtsp_server.cpp
  LABELS rtsp
)

add_tutorial(tutorial_0015_multi_stream_multi_output
  tutorial_0015_multi_stream_multi_output.cpp
  LABELS fast
)

add_tutorial(tutorial_0016_data_formats_and_mapping
  tutorial_0016_data_formats_and_mapping.cpp
  LABELS fast
)

add_tutorial(tutorial_0017_neatmodel_end_to_end
  tutorial_0017_neatmodel_end_to_end.cpp
  LABELS mpk
)

add_tutorial(tutorial_0018_async_queue_tuning
  tutorial_0018_async_queue_tuning.cpp
  LABELS fast
)

add_tutorial(tutorial_0019_multi_output_bundle
  tutorial_0019_multi_output_bundle.cpp
  LABELS fast
)

add_tutorial(tutorial_0020_debugging_workflow
  tutorial_0020_debugging_workflow.cpp
  LABELS fast
)

add_tutorial(tutorial_0021_output_appsink_policies
  tutorial_0021_output_appsink_policies.cpp
  LABELS fast
)

add_tutorial(tutorial_0022_inputappsrc_advanced_caps
  tutorial_0022_inputappsrc_advanced_caps.cpp
  LABELS fast
)

add_tutorial(tutorial_0023_builder_graph_and_contracts
  tutorial_0023_builder_graph_and_contracts.cpp
  LABELS fast
)

add_tutorial(tutorial_0024_hybrid_graph
  tutorial_0024_hybrid_graph.cpp
  LABELS fast
)

add_tutorial(tutorial_0025_hybrid_multistream
  tutorial_0025_hybrid_multistream.cpp
  LABELS fast
)

add_tutorial(tutorial_0026_send_and_receive_via_pcie
  tutorial_0026_send_and_receive_via_pcie.cpp
  LABELS pcie
)

add_tutorial(tutorial_0027_run_e2e_application_via_pcie
  tutorial_0027_run_e2e_application_via_pcie.cpp
  LABELS pcie
)

# New fast-path tutorial track (paired with Python scripts under tutorials/00x_*/).
add_tutorial(tutorial_v2_001_run_your_first_model
  001_run_your_first_model/run_your_first_model.cpp
  LABELS mpk
)

add_tutorial(tutorial_v2_002_run_inference_async
  002_run_inference_async/run_inference_async.cpp
  LABELS fast
)

add_tutorial(tutorial_v2_003_build_inference_pipeline
  003_build_inference_pipeline/build_inference_pipeline.cpp
  LABELS fast
)

add_tutorial(tutorial_v2_004_configure_model_options
  004_configure_model_options/configure_model_options.cpp
  LABELS mpk
)

add_tutorial(tutorial_v2_005_preprocess_images
  005_preprocess_images/preprocess_images.cpp
  LABELS mpk
)

add_tutorial(tutorial_v2_006_read_detection_boxes
  006_read_detection_boxes/read_detection_boxes.cpp
  LABELS mpk
)

add_tutorial(tutorial_v2_007_plug_model_into_pipeline
  007_plug_model_into_pipeline/plug_model_into_pipeline.cpp
  LABELS fast
)

add_tutorial(tutorial_v2_008_pass_numpy_to_model
  008_pass_numpy_to_model/pass_numpy_to_model.cpp
  LABELS fast
)

add_tutorial(tutorial_v2_009_feed_multi_input_model
  009_feed_multi_input_model/feed_multi_input_model.cpp
  LABELS fast
)

add_tutorial(tutorial_v2_010_interpret_model_output
  010_interpret_model_output/interpret_model_output.cpp
  LABELS fast
)

add_tutorial(tutorial_v2_011_diagnose_a_pipeline
  011_diagnose_a_pipeline/diagnose_a_pipeline.cpp
  LABELS fast
)

add_tutorial(tutorial_v2_012_detect_objects_with_yolov8
  012_detect_objects_with_yolov8/detect_objects_with_yolov8.cpp
  LABELS mpk
)

add_tutorial(tutorial_v2_013_classify_images_with_resnet50
  013_classify_images_with_resnet50/classify_images_with_resnet50.cpp
  LABELS mpk
)

add_tutorial(tutorial_v2_014_build_a_custom_data_graph
  014_build_a_custom_data_graph/build_a_custom_data_graph.cpp
  LABELS fast
)

add_tutorial(tutorial_v2_015_embed_model_inside_graph
  015_embed_model_inside_graph/embed_model_inside_graph.cpp
  LABELS mpk
)

add_tutorial(tutorial_v2_016_run_multiple_streams
  016_run_multiple_streams/run_multiple_streams.cpp
  LABELS fast
)

add_tutorial(tutorial_v2_017_tune_throughput_and_queues
  017_tune_throughput_and_queues/tune_throughput_and_queues.cpp
  LABELS fast
)

add_tutorial(tutorial_v2_018_build_production_pipeline
  018_build_production_pipeline/build_production_pipeline.cpp
  LABELS fast
)

install(
  TARGETS ${PIPELINESESSION_TUTORIAL_TARGETS}
  RUNTIME DESTINATION lib/sima-neat/tutorials
  COMPONENT extras
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION share/sima-neat/tutorials
  COMPONENT extras
  FILES_MATCHING
    PATTERN "*.cpp"
    PATTERN "*.h"
    PATTERN "*.py"
    PATTERN "*.yaml"
    PATTERN "*.md"
    PATTERN "*.jpg"
    PATTERN "*.png"
)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/build.sh
  DESTINATION share/sima-neat/tutorials
  COMPONENT extras
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

# Install built tutorial binaries from the build tree (for extras packaging)
install(
  DIRECTORY ${CMAKE_BINARY_DIR}/tutorials/
  DESTINATION lib/sima-neat/tutorials
  COMPONENT extras
  USE_SOURCE_PERMISSIONS
  FILES_MATCHING
    PATTERN "*"
)
