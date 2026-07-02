find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)

set(SIMAPCIE_HARDWARE_TEST_INSTALL_DIR
  "lib/sima-pcie-host/tests")
set(SIMAPCIE_HARDWARE_TEST_ASSET_INSTALL_DIR
  "${CMAKE_INSTALL_DATADIR}/sima-pcie-host/test-assets")
set(SIMAPCIE_HARDWARE_TEST_DEFAULT_IMAGE
  "../../../share/sima-pcie-host/test-assets/test_image.jpg")

function(simapcie_add_hardware_test name source)
  add_executable(${name} ${source})
  target_link_libraries(${name}
    PRIVATE
      sima_neat_pcie_host
      opencv_core
      opencv_imgcodecs)
  target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_compile_definitions(${name}
    PRIVATE
      DEFAULT_MODEL_PATH=""
      DEFAULT_SOURCE_IMAGE="${SIMAPCIE_HARDWARE_TEST_DEFAULT_IMAGE}")
  install(TARGETS ${name}
    RUNTIME DESTINATION "${SIMAPCIE_HARDWARE_TEST_INSTALL_DIR}"
    COMPONENT PcieHostExtras)
endfunction()

simapcie_add_hardware_test(test_metadata test_metadata.cpp)
simapcie_add_hardware_test(test_tensor_run test_tensor_run.cpp)
simapcie_add_hardware_test(test_image_run test_image_run.cpp)
simapcie_add_hardware_test(test_image_boxdecode_run test_image_boxdecode_run.cpp)
simapcie_add_hardware_test(test_queue_blocker test_queue_blocker.cpp)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/CTestTestfile.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/PcieHostExtrasCTestTestfile.cmake"
  @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/PcieHostExtrasCTestTestfile.cmake"
  RENAME CTestTestfile.cmake
  DESTINATION "${SIMAPCIE_HARDWARE_TEST_INSTALL_DIR}"
  COMPONENT PcieHostExtras)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/assets/test_image.jpg"
  DESTINATION "${SIMAPCIE_HARDWARE_TEST_ASSET_INSTALL_DIR}"
  COMPONENT PcieHostExtras)
