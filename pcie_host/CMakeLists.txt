cmake_minimum_required(VERSION 3.16)

set(SIMAPCIE_PACKAGE_VERSION "")
execute_process(
  COMMAND python3 -c "import json; print(json.load(open('../deps/manifest.json', encoding='utf-8'))['package-version'])"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE SIMAPCIE_PACKAGE_VERSION_RESULT
  OUTPUT_VARIABLE SIMAPCIE_PACKAGE_VERSION
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if (NOT SIMAPCIE_PACKAGE_VERSION_RESULT EQUAL 0 OR SIMAPCIE_PACKAGE_VERSION STREQUAL "")
  message(FATAL_ERROR "Failed to read package-version from ../deps/manifest.json")
endif()

project(SimaPCIeHost VERSION ${SIMAPCIE_PACKAGE_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "/usr" CACHE PATH "Install prefix" FORCE)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(SIMAPCIE_BUILD_TESTS "Build PCIe host unit tests" ON)
option(SIMAPCIE_BUILD_HARDWARE_TESTS "Build installable PCIe host hardware smoke tests" OFF)
option(SIMAPCIE_BUILD_PYTHON "Build PCIe host Python bindings via nanobind" OFF)
set(SIMAPCIE_NEATPCIEHOST_PLUGIN "" CACHE FILEPATH
  "Path to prebuilt libgstneatpciehost.so artifact to package")
set(SIMAPCIE_NEATPCIEHOST_INCLUDE_DIR "" CACHE PATH
  "Include directory supplied by the neatpciehost artifact/package")

if (DEFINED COPROC_BUILD_TESTS)
  set(SIMAPCIE_BUILD_TESTS "${COPROC_BUILD_TESTS}")
endif()
if (DEFINED COPROC_BUILD_HARDWARE_TESTS)
  set(SIMAPCIE_BUILD_HARDWARE_TESTS "${COPROC_BUILD_HARDWARE_TESTS}")
endif()
if (DEFINED COPROC_BUILD_PYTHON)
  set(SIMAPCIE_BUILD_PYTHON "${COPROC_BUILD_PYTHON}")
endif()

get_filename_component(SIMA_NEAT_CORE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET
  gstreamer-1.0
  gstreamer-app-1.0
  glib-2.0
)

set(SIMA_NEAT_PCIE_HOST_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/SimaPCIeHost.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/HostPcieChannel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ModelOptionsJsonWriter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/PcieModelFactsReader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/RemoteRuntime.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/SshRunner.cpp
  ${SIMA_NEAT_CORE_ROOT}/src/model/ModelArchiveLoader.cpp
  ${SIMA_NEAT_CORE_ROOT}/src/pipeline/internal/sima/MpkContract.cpp
)

add_library(sima_neat_pcie_host STATIC
  ${SIMA_NEAT_PCIE_HOST_SOURCES}
)

target_compile_definitions(sima_neat_pcie_host PRIVATE SIMA_NEAT_INTERNAL=1)
target_include_directories(sima_neat_pcie_host
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${SIMA_NEAT_CORE_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${SIMA_NEAT_CORE_ROOT}/src
    ${GST_INCLUDE_DIRS}
)
target_compile_options(sima_neat_pcie_host PRIVATE ${GST_CFLAGS_OTHER})
target_link_directories(sima_neat_pcie_host PRIVATE ${GST_LIBRARY_DIRS})
target_link_libraries(sima_neat_pcie_host PRIVATE ${GST_LIBRARIES})
if (NOT SIMAPCIE_NEATPCIEHOST_INCLUDE_DIR)
  message(FATAL_ERROR
    "SIMAPCIE_NEATPCIEHOST_INCLUDE_DIR is required and must contain "
    "gst/SimaTensorSetMetaAbi.h")
endif()
if (NOT EXISTS "${SIMAPCIE_NEATPCIEHOST_INCLUDE_DIR}/gst/SimaTensorSetMetaAbi.h")
  message(FATAL_ERROR
    "missing gst/SimaTensorSetMetaAbi.h under SIMAPCIE_NEATPCIEHOST_INCLUDE_DIR: "
    "${SIMAPCIE_NEATPCIEHOST_INCLUDE_DIR}")
endif()
target_include_directories(sima_neat_pcie_host PRIVATE
  ${SIMAPCIE_NEATPCIEHOST_INCLUDE_DIR}
)

install(TARGETS sima_neat_pcie_host
  EXPORT SimaPCIeHostTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/simaai/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/simaai
  COMPONENT Development
)

install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/pcie-setup.sh
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT Runtime
)

set(SimaPCIeHost_INSTALL_CMAKEDIR
  ${CMAKE_INSTALL_LIBDIR}/cmake/SimaPCIeHost
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SimaPCIeHostConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/SimaPCIeHostConfig.cmake
  INSTALL_DESTINATION ${SimaPCIeHost_INSTALL_CMAKEDIR}
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/SimaPCIeHostConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(
  EXPORT SimaPCIeHostTargets
  FILE SimaPCIeHostTargets.cmake
  NAMESPACE SimaPCIeHost::
  DESTINATION ${SimaPCIeHost_INSTALL_CMAKEDIR}
  COMPONENT Development
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/SimaPCIeHostConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/SimaPCIeHostConfigVersion.cmake
  DESTINATION ${SimaPCIeHost_INSTALL_CMAKEDIR}
  COMPONENT Development
)

if (SIMAPCIE_NEATPCIEHOST_PLUGIN)
  if (NOT EXISTS "${SIMAPCIE_NEATPCIEHOST_PLUGIN}")
    message(FATAL_ERROR
      "SIMAPCIE_NEATPCIEHOST_PLUGIN does not exist: ${SIMAPCIE_NEATPCIEHOST_PLUGIN}")
  endif()

  set(SIMAPCIE_HOST_PLUGIN_PRIVATE_DIR
    "${CMAKE_INSTALL_LIBDIR}/sima-pcie-host/gst-plugins")
  set(SIMAPCIE_GSTREAMER_PLUGIN_DIR
    "${CMAKE_INSTALL_LIBDIR}/gstreamer-1.0")

  install(FILES "${SIMAPCIE_NEATPCIEHOST_PLUGIN}"
    DESTINATION "${SIMAPCIE_HOST_PLUGIN_PRIVATE_DIR}"
    COMPONENT Runtime
  )

  install(CODE "
    file(MAKE_DIRECTORY \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${SIMAPCIE_GSTREAMER_PLUGIN_DIR}\")
    file(REMOVE \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${SIMAPCIE_GSTREAMER_PLUGIN_DIR}/libgstneatpciehost.so\")
    execute_process(
      COMMAND \"${CMAKE_COMMAND}\" -E create_symlink
              \"../sima-pcie-host/gst-plugins/libgstneatpciehost.so\"
              \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${SIMAPCIE_GSTREAMER_PLUGIN_DIR}/libgstneatpciehost.so\"
      RESULT_VARIABLE _simapcie_symlink_result)
    if (NOT _simapcie_symlink_result EQUAL 0)
      message(FATAL_ERROR \"failed to create libgstneatpciehost.so GStreamer symlink\")
    endif()
  " COMPONENT Runtime)
endif()

if (SIMAPCIE_BUILD_TESTS)
  enable_testing()

  add_executable(unit_pcie_host_options_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit_pcie_host_options_test.cpp
  )
  target_link_libraries(unit_pcie_host_options_test PRIVATE sima_neat_pcie_host)
  target_include_directories(unit_pcie_host_options_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

  add_executable(unit_pcie_host_caps_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit_pcie_host_caps_test.cpp
  )
  target_link_libraries(unit_pcie_host_caps_test PRIVATE sima_neat_pcie_host)
  target_include_directories(unit_pcie_host_caps_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GST_INCLUDE_DIRS}
  )

  add_executable(unit_pcie_host_lifecycle_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit_pcie_host_lifecycle_test.cpp
  )
  target_link_libraries(unit_pcie_host_lifecycle_test PRIVATE sima_neat_pcie_host)

  add_executable(unit_pcie_host_ssh_runner_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit_pcie_host_ssh_runner_test.cpp
  )
  target_link_libraries(unit_pcie_host_ssh_runner_test PRIVATE sima_neat_pcie_host)
  target_include_directories(unit_pcie_host_ssh_runner_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

  add_test(NAME unit_pcie_host_options_test COMMAND unit_pcie_host_options_test)
  add_test(NAME unit_pcie_host_caps_test COMMAND unit_pcie_host_caps_test)
  add_test(NAME unit_pcie_host_lifecycle_test COMMAND unit_pcie_host_lifecycle_test)
  add_test(NAME unit_pcie_host_ssh_runner_test COMMAND unit_pcie_host_ssh_runner_test)
endif()

if (SIMAPCIE_BUILD_HARDWARE_TESTS)
  add_subdirectory(tests/hardware)
endif()

if (SIMAPCIE_BUILD_PYTHON)
  add_subdirectory(python)
endif()

set(CPACK_GENERATOR "DEB")
set(CPACK_PACKAGE_NAME "sima-pcie-host")
set(CPACK_PACKAGE_VENDOR "SiMa.ai")
set(CPACK_PACKAGE_CONTACT "support@sima.ai")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "SiMa.ai")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Host-side NEAT PCIe coprocessor runtime")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS OFF)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_ALL Runtime Development)
set(CPACK_COMPONENT_RUNTIME_DISPLAY_NAME "SiMa PCIe host runtime")
set(CPACK_COMPONENT_DEVELOPMENT_DISPLAY_NAME "SiMa PCIe host development files")
set(CPACK_COMPONENT_DEVELOPMENT_DEPENDS Runtime)

set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME "sima-pcie-host")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_SECTION "libs")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS
  "openssh-client, libgstreamer1.0-0, libgstreamer-plugins-base1.0-0, libglib2.0-0"
)
set(CPACK_DEBIAN_RUNTIME_PACKAGE_RECOMMENDS "sshpass")
set(CPACK_DEBIAN_RUNTIME_DESCRIPTION
  "Host-side NEAT PCIe runtime, including the neatpciehost GStreamer plugin.")

set(CPACK_DEBIAN_DEVELOPMENT_PACKAGE_NAME "sima-pcie-host-dev")
set(CPACK_DEBIAN_DEVELOPMENT_PACKAGE_SECTION "libdevel")
set(CPACK_DEBIAN_DEVELOPMENT_PACKAGE_DEPENDS
  "sima-pcie-host (= ${PROJECT_VERSION}), pkg-config, libgstreamer1.0-dev, libgstreamer-plugins-base1.0-dev, libglib2.0-dev"
)
set(CPACK_DEBIAN_DEVELOPMENT_DESCRIPTION
  "Headers, static library, and CMake package files for SimaPCIeHost.")

include(CPack)
