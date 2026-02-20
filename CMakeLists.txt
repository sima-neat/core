cmake_minimum_required(VERSION 3.16)
project(SimaNeat VERSION 0.1.0 LANGUAGES CXX)

enable_testing()

# ------------------------------------------------------------------------------
# Global build settings
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_BUILD_RPATH "$ORIGIN/../neat-internals/gst-plugins")
set(CMAKE_INSTALL_RPATH "$ORIGIN/../neat-internals/gst-plugins")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Ensure installs without an explicit component are treated as "core"
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME core)

# ------------------------------------------------------------------------------
# Build options
# ------------------------------------------------------------------------------
option(SIMA_WITH_OPENCV "Enable OpenCV convenience APIs" ON)
option(SIMANEAT_BUILD_SAMPLES "Build example applications" ON)
option(SIMANEAT_BUILD_TUTORIALS "Build tutorial applications" ON)
option(SIMANEAT_BUILD_TESTS "Build test targets" ON)
option(SIMANEAT_BUILD_PYTHON "Build Python bindings via nanobind" OFF)
option(SIMANEAT_CPP_TIDY "Configure in analysis-only mode for clang-tidy" OFF)
option(SIMANEAT_STRICT_WARNINGS "Enable strict compiler warnings and treat them as errors" OFF)
option(SIMA_ENABLE_ASAN "Enable AddressSanitizer instrumentation" OFF)
option(SIMA_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer instrumentation" OFF)
option(SIMA_ENABLE_TSAN "Enable ThreadSanitizer instrumentation" OFF)
option(FUZZING "Build fuzzing harnesses and corpora runners" OFF)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Sanitizers.cmake)
simaneat_configure_sanitizers()

# ------------------------------------------------------------------------------
# Version from git (tag -> tag, else branch-hash)
# ------------------------------------------------------------------------------
set(SIMANEAT_VERSION "1.0.0")
execute_process(
  COMMAND git describe --tags --exact-match
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE GIT_TAG_RESULT
  OUTPUT_VARIABLE GIT_TAG
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (GIT_TAG_RESULT EQUAL 0 AND NOT GIT_TAG STREQUAL "")
  set(SIMANEAT_VERSION "${GIT_TAG}")
else()
  execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_BRANCH_RESULT
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_HASH_RESULT
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if (GIT_BRANCH_RESULT EQUAL 0 AND GIT_HASH_RESULT EQUAL 0
      AND NOT GIT_BRANCH STREQUAL "" AND NOT GIT_HASH STREQUAL "")
    set(SIMANEAT_VERSION "${GIT_BRANCH}-${GIT_HASH}")
  endif()
endif()

string(REGEX REPLACE "[^A-Za-z0-9.+~_-]" "-" SIMANEAT_VERSION "${SIMANEAT_VERSION}")

# Debian version must start with a digit. Also normalize common "v1.2.3" tags.
if (SIMANEAT_VERSION MATCHES "^v[0-9].*")
  string(REGEX REPLACE "^v" "" SIMANEAT_VERSION "${SIMANEAT_VERSION}")
endif()

if (NOT SIMANEAT_VERSION MATCHES "^[0-9].*")
  set(SIMANEAT_VERSION "0.0.0+${SIMANEAT_VERSION}")
endif()

if (SIMANEAT_STRICT_WARNINGS)
  if (MSVC)
    add_compile_options(/W4 /WX)
  else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    add_compile_options(-Wno-missing-field-initializers -Wno-unused-parameter -Wno-unused-but-set-variable)
    add_compile_options(-Wno-sign-compare -Wno-unused-function -Wno-switch)
    add_compile_options(-Wno-unused-variable -Wno-reorder -Wno-misleading-indentation)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT SIMANEAT_CPP_TIDY)
      add_compile_options(-Wno-restrict)
    endif()
  endif()
endif()

# ------------------------------------------------------------------------------
# Public header hygiene (no internal headers under include/)
# ------------------------------------------------------------------------------
file(GLOB_RECURSE SIMANEAT_PUBLIC_INTERNAL_HEADERS
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*/internal/*"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*/*/internal/*"
)

if (SIMANEAT_PUBLIC_INTERNAL_HEADERS)
  string(JOIN "\n  " SIMANEAT_PUBLIC_INTERNAL_HEADERS_STR ${SIMANEAT_PUBLIC_INTERNAL_HEADERS})
  message(FATAL_ERROR
    "Internal headers must not live under include/. Found:\n  ${SIMANEAT_PUBLIC_INTERNAL_HEADERS_STR}"
  )
endif()

function(simaneat_collect_forbidden_includes out_var)
  set(_hits "")
  foreach(_src IN LISTS ARGN)
    file(STRINGS "${_src}" _bad_lines
      REGEX "^[ \t]*#include[ \t]+\"([^\"]*internal/[^\"]*|tests/[^\"]*|e2e_pipelines/[^\"]*|test_utils\\.h)\"")
    foreach(_line IN LISTS _bad_lines)
      list(APPEND _hits "${_src}: ${_line}")
    endforeach()
  endforeach()
  set(${out_var} "${_hits}" PARENT_SCOPE)
endfunction()

# examples/tutorials must not consume internal or test-only headers
file(GLOB_RECURSE SIMANEAT_EXAMPLE_TUTORIAL_SOURCES
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/tutorials/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/tutorials/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/tutorials/*.cxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/tutorials/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/tutorials/*.hpp"
)
simaneat_collect_forbidden_includes(SIMANEAT_BOUNDARY_VIOLATIONS ${SIMANEAT_EXAMPLE_TUTORIAL_SOURCES})
if (SIMANEAT_BOUNDARY_VIOLATIONS)
  string(JOIN "\n  " SIMANEAT_BOUNDARY_VIOLATIONS_STR ${SIMANEAT_BOUNDARY_VIOLATIONS})
  message(FATAL_ERROR
    "examples/tutorials include test-only or internal headers. Fix includes:\n  ${SIMANEAT_BOUNDARY_VIOLATIONS_STR}"
  )
endif()

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)

if (SIMANEAT_CPP_TIDY)
  pkg_check_modules(GST QUIET
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-video-1.0
    gstreamer-sdp-1.0
    gstreamer-rtsp-server-1.0
    glib-2.0
  )
  if (NOT GST_FOUND)
    message(WARNING "GStreamer not found; configuring without GST includes/libs (analysis-only mode).")
  endif()
else()
  pkg_check_modules(GST REQUIRED
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-video-1.0
    gstreamer-sdp-1.0
    gstreamer-rtsp-server-1.0
    glib-2.0
  )
endif()

if (SIMANEAT_CPP_TIDY)
  pkg_check_modules(OPENCV QUIET opencv4)
  if (NOT OPENCV_FOUND)
    message(WARNING "OpenCV not found; configuring without OpenCV includes/libs (analysis-only mode).")
  endif()
else()
  pkg_check_modules(OPENCV REQUIRED opencv4)
endif()

find_library(SIMAAI_ALLOCATOR_LIB gstsimaallocator)
find_library(SIMAAI_BUFFERPOOL_LIB gstsimaaibufferpool)

# ------------------------------------------------------------------------------
# Library: sima_neat (FULL ORIGINAL SOURCE LIST)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Source blacklist (relative to src/)
# ------------------------------------------------------------------------------
set(SIMANEAT_SRC_BLACKLIST
)

# ------------------------------------------------------------------------------
# Collect SimaNeat sources automatically
# ------------------------------------------------------------------------------

file(GLOB_RECURSE SIMANEAT_ALL_SOURCES
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

# ------------------------------------------------------------------------------
# Apply blacklist
# ------------------------------------------------------------------------------
set(SIMANEAT_SOURCES ${SIMANEAT_ALL_SOURCES})

foreach(blacklisted ${SIMANEAT_SRC_BLACKLIST})
  set(abs_path "${CMAKE_CURRENT_SOURCE_DIR}/src/${blacklisted}")

  if (EXISTS "${abs_path}")
    message(STATUS "  [sima_neat] Blacklisting source: ${blacklisted}")
    list(REMOVE_ITEM SIMANEAT_SOURCES "${abs_path}")
  else()
    message(WARNING "  [sima_neat] Blacklist entry not found: ${blacklisted}")
  endif()
endforeach()

# ------------------------------------------------------------------------------
# Library: sima_neat
# ------------------------------------------------------------------------------
add_library(sima_neat STATIC
  ${SIMANEAT_SOURCES}
)

target_compile_definitions(sima_neat PRIVATE SIMA_NEAT_INTERNAL=1)

target_include_directories(sima_neat
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/pipeline>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/pipeline>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pipeline/internal
)

if (GST_FOUND)
  target_include_directories(sima_neat SYSTEM PUBLIC ${GST_INCLUDE_DIRS})
endif()
if (OPENCV_FOUND)
  target_include_directories(sima_neat SYSTEM PUBLIC ${OPENCV_INCLUDE_DIRS})
endif()

target_compile_options(sima_neat PRIVATE
)

if (GST_FOUND)
  target_compile_options(sima_neat PRIVATE ${GST_CFLAGS_OTHER})
endif()
if (OPENCV_FOUND)
  target_compile_options(sima_neat PRIVATE ${OPENCV_CFLAGS_OTHER})
endif()

if (OPENCV_FOUND)
  target_link_directories(sima_neat PRIVATE ${OPENCV_LIBRARY_DIRS})
endif()
if (GST_FOUND)
  target_link_directories(sima_neat PRIVATE ${GST_LIBRARY_DIRS})
endif()

target_link_libraries(sima_neat
  PUBLIC
)

if (GST_FOUND)
  target_link_libraries(sima_neat PUBLIC ${GST_LIBRARIES})
endif()
if (OPENCV_FOUND)
  target_link_libraries(sima_neat PUBLIC ${OPENCV_LIBRARIES})
endif()

if (SIMA_WITH_OPENCV)
  target_compile_definitions(sima_neat PUBLIC SIMA_WITH_OPENCV)
endif()

if (SIMAAI_ALLOCATOR_LIB)
  target_link_libraries(sima_neat PUBLIC ${SIMAAI_ALLOCATOR_LIB})
endif()

if (SIMAAI_BUFFERPOOL_LIB)
  target_link_libraries(sima_neat PUBLIC ${SIMAAI_BUFFERPOOL_LIB})
endif()

if (SIMAAI_ALLOCATOR_LIB AND SIMAAI_BUFFERPOOL_LIB)
  target_compile_definitions(sima_neat PUBLIC SIMA_HAS_SIMAAI_POOL=1)
endif()

# ------------------------------------------------------------------------------
# Internal-only helper target (tests/bench/tools)
# ------------------------------------------------------------------------------
add_library(sima_neat_internal INTERFACE)
target_link_libraries(sima_neat_internal INTERFACE sima_neat)
target_include_directories(sima_neat_internal INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/pipeline
)
target_compile_definitions(sima_neat_internal INTERFACE SIMA_NEAT_INTERNAL=1)

# ------------------------------------------------------------------------------
# Examples (optional, installable)
# ------------------------------------------------------------------------------
if (SIMANEAT_BUILD_SAMPLES)
  add_subdirectory(examples)
endif()

# ------------------------------------------------------------------------------
# Tutorials
# ------------------------------------------------------------------------------
if (SIMANEAT_BUILD_TUTORIALS)
  add_subdirectory(tutorials)
endif()

# ------------------------------------------------------------------------------
# Tests (internal only)
# ------------------------------------------------------------------------------
if (SIMANEAT_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# ------------------------------------------------------------------------------
# Python bindings (optional)
# ------------------------------------------------------------------------------
if (SIMANEAT_BUILD_PYTHON)
  add_subdirectory(python)
endif()


# ------------------------------------------------------------------------------
# Install / Package
# ------------------------------------------------------------------------------
install(
  TARGETS sima_neat
  EXPORT SimaNeatTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT core
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT core
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/neat-internals/gst-plugins/
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/sima-neat/gst-plugins
  COMPONENT core
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/distribution/skills/sima-neat/
  DESTINATION ${CMAKE_INSTALL_DATADIR}/sima-neat/skills/sima-neat
  COMPONENT core
)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/scripts/install_codex_skill.sh
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  RENAME sima-neat-install-skill
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
  COMPONENT core
)

set(SimaNeat_INSTALL_CMAKEDIR
  ${CMAKE_INSTALL_LIBDIR}/cmake/SimaNeat
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SimaNeatConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/SimaNeatConfig.cmake
  INSTALL_DESTINATION ${SimaNeat_INSTALL_CMAKEDIR}
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/SimaNeatConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(
  EXPORT SimaNeatTargets
  FILE SimaNeatTargets.cmake
  NAMESPACE SimaNeat::
  DESTINATION ${SimaNeat_INSTALL_CMAKEDIR}
  COMPONENT core
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/SimaNeatConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/SimaNeatConfigVersion.cmake
  DESTINATION ${SimaNeat_INSTALL_CMAKEDIR}
  COMPONENT core
)

# ------------------------------------------------------------------------------
# CPack (DEB)
# ------------------------------------------------------------------------------

set(CPACK_PACKAGE_NAME "sima-neat")
set(CPACK_PACKAGE_VENDOR "SiMa.ai")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "SiMa NEAT core static library and headers")

set(CPACK_COMPONENTS_ALL core extras)
set(CPACK_DEB_COMPONENT_INSTALL ON)

set(CPACK_COMPONENT_CORE_DESCRIPTION
    "SiMa NEAT core library and headers")
set(CPACK_COMPONENT_EXTRAS_DESCRIPTION
    "SiMa NEAT samples, tutorials, and tests")

set(CPACK_PACKAGE_VERSION "${SIMANEAT_VERSION}")

set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "SiMa.ai")

set(CPACK_DEBIAN_PACKAGE_SECTION "libs")
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")

set(CPACK_DEBIAN_CORE_PACKAGE_NAME "sima-neat")
set(CPACK_DEBIAN_EXTRAS_PACKAGE_NAME "sima-neat-extras")

set(CPACK_DEBIAN_CORE_PACKAGE_DEPENDS
  "libgstreamer1.0-0, libglib2.0-0, libopencv-core406"
)
set(CPACK_DEBIAN_EXTRAS_PACKAGE_DEPENDS
  "sima-neat (= ${CPACK_PACKAGE_VERSION})"
)

set(CPACK_INSTALL_CMAKE_PROJECTS
  "${CMAKE_BINARY_DIR};SimaNeat;ALL;/"
)

include(CPack)

# ------------------------------------------------------------------------------
# Docs (Doxygen)
# ------------------------------------------------------------------------------
find_package(Doxygen QUIET)
if (DOXYGEN_FOUND)
  set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/Doxyfile)
  add_custom_target(docs
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_IN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating Doxygen documentation"
    VERBATIM
  )
else()
  message(STATUS "Doxygen not found: 'docs' target will be unavailable.")
endif()
