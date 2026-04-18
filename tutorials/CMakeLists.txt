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

add_tutorial(tutorial_v2_019_run_yolo_on_rtsp_stream
  019_run_yolo_on_rtsp_stream/run_yolo_on_rtsp_stream.cpp
  LABELS asset
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
