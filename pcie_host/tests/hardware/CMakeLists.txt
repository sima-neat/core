find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

set(SIMAPCIE_HARDWARE_TEST_INSTALL_DIR
  "lib/sima-pcie-host/tests")

function(simapcie_add_hardware_test name source)
  add_executable(${name} ${source})
  target_link_libraries(${name}
    PRIVATE
      sima_neat_pcie_host
      opencv_core
      opencv_imgcodecs
      opencv_imgproc)
  target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_compile_definitions(${name}
    PRIVATE
      DEFAULT_MODEL_PATH=""
      DEFAULT_SOURCE_IMAGE="")
  install(TARGETS ${name}
    RUNTIME DESTINATION "${SIMAPCIE_HARDWARE_TEST_INSTALL_DIR}"
    COMPONENT PcieHostExtras)
endfunction()

simapcie_add_hardware_test(test_metadata test_metadata.cpp)
simapcie_add_hardware_test(test_tensor_run test_tensor_run.cpp)
simapcie_add_hardware_test(test_tensor_perf test_tensor_perf.cpp)
simapcie_add_hardware_test(test_image_run test_image_run.cpp)
simapcie_add_hardware_test(test_image_boxdecode_run test_image_boxdecode_run.cpp)
simapcie_add_hardware_test(test_queue_blocker test_queue_blocker.cpp)

install(PROGRAMS test_tensor_parallel_queues.sh
  DESTINATION "${SIMAPCIE_HARDWARE_TEST_INSTALL_DIR}"
  COMPONENT PcieHostExtras)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/CTestTestfile.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/PcieHostExtrasCTestTestfile.cmake"
  @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/PcieHostExtrasCTestTestfile.cmake"
  RENAME CTestTestfile.cmake
  DESTINATION "${SIMAPCIE_HARDWARE_TEST_INSTALL_DIR}"
  COMPONENT PcieHostExtras)
