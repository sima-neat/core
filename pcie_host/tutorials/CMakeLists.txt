if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.16)
  project(SimaPCIeHostTutorials LANGUAGES CXX)

  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  find_package(SimaPCIeHost REQUIRED CONFIG)
  set(SIMAPCIE_TUTORIAL_LINK_TARGET SimaPCIeHost::sima_neat_pcie_host)
else()
  set(SIMAPCIE_TUTORIAL_LINK_TARGET sima_neat_pcie_host)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(Threads REQUIRED)

set(SIMAPCIE_TUTORIAL_TARGETS "")

function(simapcie_add_tutorial name source)
  add_executable(${name} ${source})
  target_link_libraries(${name}
    PRIVATE
      ${SIMAPCIE_TUTORIAL_LINK_TARGET}
      opencv_core
      opencv_imgcodecs
      opencv_imgproc
      Threads::Threads)
  target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS})
  list(APPEND SIMAPCIE_TUTORIAL_TARGETS ${name})
  set(SIMAPCIE_TUTORIAL_TARGETS "${SIMAPCIE_TUTORIAL_TARGETS}" PARENT_SCOPE)
endfunction()

simapcie_add_tutorial(tutorial_024_run_tensor_mode
  024_run_pcie_inference_modes/run_tensor_mode.cpp)

simapcie_add_tutorial(tutorial_024_run_image_mode
  024_run_pcie_inference_modes/run_image_mode.cpp)

simapcie_add_tutorial(tutorial_024_run_image_boxdecode
  024_run_pcie_inference_modes/run_image_boxdecode.cpp)

simapcie_add_tutorial(tutorial_025_measure_pcie_detection_throughput
  025_measure_pcie_detection_throughput/measure_pcie_detection_throughput.cpp)

simapcie_add_tutorial(tutorial_026_run_models_on_multiple_pcie_queues
  026_run_models_on_multiple_pcie_queues/run_models_on_multiple_pcie_queues.cpp)

if (SIMAPCIE_TUTORIAL_TARGETS)
  install(
    TARGETS ${SIMAPCIE_TUTORIAL_TARGETS}
    RUNTIME DESTINATION lib/sima-pcie-host/tutorials
    COMPONENT PcieHostExtras)
endif()

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION share/sima-pcie-host/tutorials
  COMPONENT PcieHostExtras
  FILES_MATCHING
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.cpp"
    PATTERN "*.h"
    PATTERN "*.py"
    PATTERN "*.md"
    PATTERN "*.jpg"
    PATTERN "*.png"
    PATTERN "CMakeLists.txt")

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/build.sh
  DESTINATION .
  COMPONENT PcieHostExtras
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE)
