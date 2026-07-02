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

set(SIMANEAT_TUTORIAL_MODEL_TARGET_FOLDER "/tmp" CACHE PATH
  "Directory passed to tutorials/build.sh --model-target-folder for Model Zoo downloads"
)

set(PIPELINESESSION_TUTORIAL_TARGETS "")

function(add_tutorial exe source)
  set(options NO_TEST)
  set(oneValueArgs MODEL)
  set(multiValueArgs ARGS LABELS)
  cmake_parse_arguments(AT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_executable(${exe} ${source})
  target_link_libraries(${exe} PRIVATE ${SIMANEAT_TUTORIAL_LINK_TARGET})
  target_include_directories(${exe} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

  set(test_args ${AT_ARGS})
  if (AT_MODEL)
    get_filename_component(model_base "${AT_MODEL}" NAME)
    list(APPEND test_args
      --model "${SIMANEAT_TUTORIAL_MODEL_TARGET_FOLDER}/${model_base}.tar.gz"
    )
  endif()

  set(label_list "tutorial")
  if (AT_LABELS)
    list(APPEND label_list ${AT_LABELS})
  endif()

  if (NOT AT_NO_TEST)
    if (AT_MODEL)
      add_test(NAME ${exe}
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_with_modelzoo.sh
          ${AT_MODEL}
          "${SIMANEAT_TUTORIAL_MODEL_TARGET_FOLDER}/${model_base}.tar.gz"
          "${SIMANEAT_TUTORIAL_MODEL_TARGET_FOLDER}"
          $<TARGET_FILE:${exe}>
          ${test_args}
      )
    else()
      add_test(NAME ${exe} COMMAND ${exe} ${test_args})
    endif()
    set_tests_properties(${exe} PROPERTIES
      LABELS "${label_list}"
      RESOURCE_LOCK "dispatcher"
    )
  endif()

  list(APPEND PIPELINESESSION_TUTORIAL_TARGETS ${exe})
endfunction()

# Tutorials are added as they are implemented.
add_tutorial(tutorial_001_run_your_first_model
  001_run_your_first_model/run_your_first_model.cpp
  MODEL resnet_50
  LABELS mpk
)

add_tutorial(tutorial_002_run_inference_async
  002_run_inference_async/run_inference_async.cpp
  MODEL resnet_50
  LABELS mpk
)

add_tutorial(tutorial_003_benchmark_your_model
  003_benchmark_your_model/benchmark_your_model.cpp
  MODEL resnet_50
  ARGS --samples 10
  LABELS mpk benchmark
)

add_tutorial(tutorial_004_build_inference_pipeline
  004_build_inference_pipeline/build_inference_pipeline.cpp
  LABELS fast
)

add_tutorial(tutorial_005_configure_model_options
  005_configure_model_options/configure_model_options.cpp
  MODEL yolo_v8s
  LABELS mpk
)

add_tutorial(tutorial_006_preprocess_images
  006_preprocess_images/preprocess_images.cpp
  MODEL resnet_50
  LABELS mpk
)

add_tutorial(tutorial_007_read_detection_boxes
  007_read_detection_boxes/read_detection_boxes.cpp
  MODEL yolo_v8s
  ARGS --image "${CMAKE_CURRENT_SOURCE_DIR}/assets/fronalpstock_1330.jpg"
  LABELS mpk
)

add_tutorial(tutorial_008_plug_model_into_pipeline
  008_plug_model_into_pipeline/plug_model_into_pipeline.cpp
  MODEL yolo_v8s
  LABELS mpk
)

add_tutorial(tutorial_009_pass_numpy_to_model
  009_pass_numpy_to_model/pass_numpy_to_model.cpp
  LABELS fast
)

add_tutorial(tutorial_010_feed_multi_input_model
  010_feed_multi_input_model/feed_multi_input_model.cpp
  LABELS fast
)

add_tutorial(tutorial_011_interpret_model_output
  011_interpret_model_output/interpret_model_output.cpp
  LABELS fast
)

add_tutorial(tutorial_012_diagnose_a_pipeline
  012_diagnose_a_pipeline/diagnose_a_pipeline.cpp
  LABELS fast
)

add_tutorial(tutorial_013_build_a_custom_data_graph
  013_build_a_custom_data_graph/build_a_custom_data_graph.cpp
  LABELS fast
)

add_tutorial(tutorial_014_embed_model_inside_graph
  014_embed_model_inside_graph/embed_model_inside_graph.cpp
  MODEL yolo_v8s
  LABELS mpk
)

add_tutorial(tutorial_015_run_multiple_streams
  015_run_multiple_streams/run_multiple_streams.cpp
  LABELS fast
)

add_tutorial(tutorial_016_tune_throughput_and_queues
  016_tune_throughput_and_queues/tune_throughput_and_queues.cpp
  LABELS fast
)

add_tutorial(tutorial_017_build_production_pipeline
  017_build_production_pipeline/build_production_pipeline.cpp
  MODEL resnet_50
  LABELS mpk
)

add_tutorial(tutorial_018_consume_rtsp_stream
  018_consume_rtsp_stream/consume_rtsp_stream.cpp
  NO_TEST
  LABELS asset
)
add_test(
  NAME tutorial_018_consume_rtsp_stream
  COMMAND /bin/sh -c [=[
url="${SIMANEAT_TEST_RTSP_H264_URL:-}"
if [ -z "${url}" ]; then
  urls="${SIMANEAT_TEST_RTSP_H264_URLS:-}"
  url="${urls%%[ ,;]*}"
fi
if [ -z "${url}" ]; then
  url="${SIMANEAT_APPS_TEST_RTSP_URL:-}"
fi
if [ -z "${url}" ]; then
  urls="${SIMANEAT_APPS_TEST_RTSP_URLS:-}"
  url="${urls%%[ ,;]*}"
fi
if [ -z "${url}" ]; then
  echo "Skipping tutorial_018_consume_rtsp_stream: set SIMANEAT_TEST_RTSP_H264_URL or SIMANEAT_TEST_RTSP_H264_URLS"
  exit 77
fi
out="$("$1" --url "${url}" --frames 10)"
printf '%s\n' "${out}"
if printf '%s\n' "${out}" | grep -q 'rtsp_timeout'; then
  echo "tutorial_018_consume_rtsp_stream: RTSP timeout while pulling frames" >&2
  exit 1
fi
shape_count="$(printf '%s\n' "${out}" | grep -cE '^frame=[0-9]+ shape=')"
if [ "${shape_count}" -ne 10 ]; then
  echo "tutorial_018_consume_rtsp_stream: expected 10 decoded frames, got ${shape_count}" >&2
  exit 1
fi
]=] sh $<TARGET_FILE:tutorial_018_consume_rtsp_stream>
)
set_tests_properties(tutorial_018_consume_rtsp_stream PROPERTIES
  LABELS "tutorial;asset;rtsp"
  RESOURCE_LOCK "dispatcher"
  SKIP_RETURN_CODE 77
)

add_tutorial(tutorial_019_run_an_llm
  019_run_an_llm/run_an_llm.cpp
  NO_TEST
  LABELS genai
)

add_tutorial(tutorial_020_run_a_vlm
  020_run_a_vlm/run_a_vlm.cpp
  NO_TEST
  LABELS genai image
)

add_tutorial(tutorial_021_serve_genai_models
  021_serve_genai_models/serve_genai_models.cpp
  NO_TEST
  LABELS genai server
)

add_tutorial(tutorial_022_compose_genai_into_graph
  022_compose_genai_into_graph/compose_genai_into_graph.cpp
  NO_TEST
  LABELS genai graph
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
    PATTERN "CMakeLists.txt"
)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/build.sh
  DESTINATION .
  COMPONENT extras
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/run_with_modelzoo.sh
  DESTINATION share/sima-neat/tutorials
  COMPONENT extras
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

set(SIMANEAT_TUTORIALS_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/../deps/manifest.json")
if (EXISTS "${SIMANEAT_TUTORIALS_MANIFEST}")
  install(
    FILES ${SIMANEAT_TUTORIALS_MANIFEST}
    DESTINATION share/sima-neat/deps
    COMPONENT extras
  )
endif()

# Install built tutorial binaries from the build tree (for extras packaging)
install(
  DIRECTORY ${CMAKE_BINARY_DIR}/tutorials/
  DESTINATION lib/sima-neat/tutorials
  COMPONENT extras
  USE_SOURCE_PERMISSIONS
  FILES_MATCHING
    PATTERN "*"
)
