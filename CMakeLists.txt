cmake_minimum_required(VERSION 3.16)
project(SimaNeat VERSION 0.3.0 LANGUAGES CXX)

enable_testing()

# ------------------------------------------------------------------------------
# Global build settings
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

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
option(SIMANEAT_BUILD_TUTORIALS "Build tutorial applications" ON)
option(SIMANEAT_BUILD_TESTS "Build test targets" ON)
option(SIMANEAT_BUILD_PYTHON "Build Python bindings via nanobind" OFF)
option(SIMANEAT_BUILD_DOCS "Configure documentation targets" OFF)
option(SIMANEAT_DOCS_ONLY_CONFIGURE "Relax runtime dependency checks for docs-only builds" OFF)
option(SIMANEAT_CPP_TIDY "Configure in analysis-only mode for clang-tidy" OFF)
option(SIMANEAT_STRICT_WARNINGS "Enable strict compiler warnings and treat them as errors" OFF)
option(SIMA_ENABLE_ASAN "Enable AddressSanitizer instrumentation" OFF)
option(SIMA_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer instrumentation" OFF)
option(SIMA_ENABLE_TSAN "Enable ThreadSanitizer instrumentation" OFF)
option(FUZZING "Build fuzzing harnesses and corpora runners" OFF)
option(SIMANEAT_USE_EXPORTED_NEAT_INTERNALS
  "Prefer the exported NeatInternals SDK package over ad hoc library/header discovery"
  ON)

set(SIMANEAT_REQUIRE_NEAT_RUNTIME_ARTIFACTS_DEFAULT ON)
if (SIMANEAT_CPP_TIDY OR SIMANEAT_DOCS_ONLY_CONFIGURE)
  set(SIMANEAT_REQUIRE_NEAT_RUNTIME_ARTIFACTS_DEFAULT OFF)
endif()
option(SIMANEAT_REQUIRE_NEAT_RUNTIME_ARTIFACTS
  "Require NEAT runtime libraries/headers during configure"
  ${SIMANEAT_REQUIRE_NEAT_RUNTIME_ARTIFACTS_DEFAULT})

set(SIMANEAT_REQUIRE_LLIMA_ARTIFACTS_DEFAULT ON)
if (SIMANEAT_CPP_TIDY OR SIMANEAT_DOCS_ONLY_CONFIGURE)
  set(SIMANEAT_REQUIRE_LLIMA_ARTIFACTS_DEFAULT OFF)
endif()
option(SIMANEAT_REQUIRE_LLIMA_ARTIFACTS
  "Require LLiMa runtime libraries/headers during configure"
  ${SIMANEAT_REQUIRE_LLIMA_ARTIFACTS_DEFAULT})
set(SIMANEAT_BUILDINFO_JSON "" CACHE FILEPATH
  "Optional package provenance JSON to install as share/sima-neat/buildinfo.json")

if (UNIX)
  set(SIMANEAT_SOURCE_FS_TYPE "")
  execute_process(
    COMMAND findmnt -T ${CMAKE_CURRENT_SOURCE_DIR} -n -o FSTYPE
    RESULT_VARIABLE SIMANEAT_FINDMNT_RESULT
    OUTPUT_VARIABLE SIMANEAT_SOURCE_FS_TYPE
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if (NOT SIMANEAT_FINDMNT_RESULT EQUAL 0 OR SIMANEAT_SOURCE_FS_TYPE STREQUAL "")
    if (APPLE)
      execute_process(
        COMMAND stat -f %T ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE SIMANEAT_STAT_RESULT
        OUTPUT_VARIABLE SIMANEAT_SOURCE_FS_TYPE
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    else()
      execute_process(
        COMMAND stat -f -c %T ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE SIMANEAT_STAT_RESULT
        OUTPUT_VARIABLE SIMANEAT_SOURCE_FS_TYPE
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    endif()
  endif()
  if (SIMANEAT_SOURCE_FS_TYPE MATCHES "^(cifs|nfs|nfs4|smb2|smb3|smbfs|fuse.sshfs)$")
    message(WARNING
      "SimaNeat is being configured from a ${SIMANEAT_SOURCE_FS_TYPE} source tree. "
      "Recursive source globs and large C++ builds can stall on remote mounts. "
      "Prefer ./build.sh, which can auto-stage a local shadow workspace via "
      "SIMANEAT_SHADOW_BUILD=auto."
    )
  endif()
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Sanitizers.cmake)
simaneat_configure_sanitizers()

# ------------------------------------------------------------------------------
# Versioning from deps/manifest.json:
# - package-version controls package/artifact identity.
# - platform-version controls dependency and CMake package compatibility.
# - abi-version controls the public C++ shared-library ABI and SONAME.
# Exact tags use the tag version; branch builds append branch/sha metadata.
# ------------------------------------------------------------------------------
set(SIMANEAT_PLATFORM_VERSION "")
set(SIMANEAT_PACKAGE_BASE_VERSION "")
set(SIMANEAT_ABI_VERSION "")
execute_process(
  COMMAND python3 -c "import json; data=json.load(open('deps/manifest.json', encoding='utf-8')); print(data['platform-version']); print(data['package-version']); print(data['abi-version'])"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE PLATFORM_VERSION_RESULT
  OUTPUT_VARIABLE SIMANEAT_MANIFEST_VERSIONS
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if (NOT PLATFORM_VERSION_RESULT EQUAL 0 OR SIMANEAT_MANIFEST_VERSIONS STREQUAL "")
  message(FATAL_ERROR
    "Failed to read package-version, platform-version, and abi-version from deps/manifest.json")
endif()
string(REPLACE "\n" ";" SIMANEAT_MANIFEST_VERSION_LIST "${SIMANEAT_MANIFEST_VERSIONS}")
list(GET SIMANEAT_MANIFEST_VERSION_LIST 0 SIMANEAT_PLATFORM_VERSION)
list(GET SIMANEAT_MANIFEST_VERSION_LIST 1 SIMANEAT_PACKAGE_BASE_VERSION)
list(GET SIMANEAT_MANIFEST_VERSION_LIST 2 SIMANEAT_ABI_VERSION)
if (NOT SIMANEAT_ABI_VERSION MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR
    "deps/manifest.json abi-version must be a positive integer; got '${SIMANEAT_ABI_VERSION}'")
endif()

set(SIMANEAT_VERSION "${SIMANEAT_PACKAGE_BASE_VERSION}")
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
  set(GIT_BRANCH "")
  set(GIT_HASH "")

  if ("$ENV{GITHUB_REF_TYPE}" STREQUAL "tag" AND NOT "$ENV{GITHUB_REF_NAME}" STREQUAL "")
    set(SIMANEAT_VERSION "$ENV{GITHUB_REF_NAME}")
  endif()

  execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_BRANCH_RESULT
    OUTPUT_VARIABLE GIT_BRANCH
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  execute_process(
    COMMAND git rev-parse --short=12 HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_HASH_RESULT
    OUTPUT_VARIABLE GIT_HASH
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if ((NOT GIT_BRANCH_RESULT EQUAL 0 OR GIT_BRANCH STREQUAL "" OR GIT_BRANCH STREQUAL "HEAD")
      AND (NOT "$ENV{GITHUB_HEAD_REF}" STREQUAL "" OR NOT "$ENV{GITHUB_REF_NAME}" STREQUAL ""))
    if (NOT "$ENV{GITHUB_HEAD_REF}" STREQUAL "")
      set(GIT_BRANCH "$ENV{GITHUB_HEAD_REF}")
    else()
      set(GIT_BRANCH "$ENV{GITHUB_REF_NAME}")
    endif()
    set(GIT_BRANCH_RESULT 0)
  endif()

  if ((NOT GIT_HASH_RESULT EQUAL 0 OR GIT_HASH STREQUAL "") AND NOT "$ENV{GITHUB_SHA}" STREQUAL "")
    string(SUBSTRING "$ENV{GITHUB_SHA}" 0 12 GIT_HASH)
    set(GIT_HASH_RESULT 0)
  endif()

  if (SIMANEAT_VERSION STREQUAL SIMANEAT_PACKAGE_BASE_VERSION
      AND GIT_BRANCH_RESULT EQUAL 0 AND GIT_HASH_RESULT EQUAL 0
      AND NOT GIT_BRANCH STREQUAL "" AND NOT GIT_HASH STREQUAL "")
    string(TOLOWER "${GIT_BRANCH}" GIT_BRANCH_VERSION)
    string(REGEX REPLACE "[^a-z0-9.+~-]+" "-" GIT_BRANCH_VERSION "${GIT_BRANCH_VERSION}")
    string(REGEX REPLACE "^-+|-+$" "" GIT_BRANCH_VERSION "${GIT_BRANCH_VERSION}")
    if (GIT_BRANCH_VERSION STREQUAL "")
      set(GIT_BRANCH_VERSION "branch")
    endif()
    set(SIMANEAT_VERSION "${SIMANEAT_PACKAGE_BASE_VERSION}+${GIT_BRANCH_VERSION}.${GIT_HASH}")
  endif()
endif()

string(REPLACE "." ";" SIMANEAT_PACKAGE_BASE_VERSION_PARTS "${SIMANEAT_PACKAGE_BASE_VERSION}")
list(GET SIMANEAT_PACKAGE_BASE_VERSION_PARTS 0 SIMANEAT_PACKAGE_BASE_VERSION_MAJOR)
list(GET SIMANEAT_PACKAGE_BASE_VERSION_PARTS 1 SIMANEAT_PACKAGE_BASE_VERSION_MINOR)
math(EXPR SIMANEAT_NEXT_PACKAGE_MINOR "${SIMANEAT_PACKAGE_BASE_VERSION_MINOR} + 1")
set(SIMANEAT_DEP_PACKAGE_MIN_VERSION "${SIMANEAT_PACKAGE_BASE_VERSION}")
set(SIMANEAT_DEP_PACKAGE_MAX_VERSION "${SIMANEAT_PACKAGE_BASE_VERSION_MAJOR}.${SIMANEAT_NEXT_PACKAGE_MINOR}.0")

string(REPLACE "." ";" SIMANEAT_PLATFORM_VERSION_PARTS "${SIMANEAT_PLATFORM_VERSION}")
list(GET SIMANEAT_PLATFORM_VERSION_PARTS 0 SIMANEAT_PLATFORM_VERSION_MAJOR)
list(GET SIMANEAT_PLATFORM_VERSION_PARTS 1 SIMANEAT_PLATFORM_VERSION_MINOR)
list(GET SIMANEAT_PLATFORM_VERSION_PARTS 2 SIMANEAT_PLATFORM_VERSION_PATCH)
if (SIMANEAT_PLATFORM_VERSION_PATCH GREATER 0)
  math(EXPR SIMANEAT_PREVIOUS_PLATFORM_PATCH "${SIMANEAT_PLATFORM_VERSION_PATCH} - 1")
else()
  set(SIMANEAT_PREVIOUS_PLATFORM_PATCH "${SIMANEAT_PLATFORM_VERSION_PATCH}")
endif()
math(EXPR SIMANEAT_NEXT_PLATFORM_PATCH "${SIMANEAT_PLATFORM_VERSION_PATCH} + 1")
set(SIMANEAT_PLATFORM_DEP_PACKAGE_MIN_VERSION
  "${SIMANEAT_PLATFORM_VERSION_MAJOR}.${SIMANEAT_PLATFORM_VERSION_MINOR}.${SIMANEAT_PREVIOUS_PLATFORM_PATCH}~")
set(SIMANEAT_PLATFORM_DEP_PACKAGE_MAX_VERSION
  "${SIMANEAT_PLATFORM_VERSION_MAJOR}.${SIMANEAT_PLATFORM_VERSION_MINOR}.${SIMANEAT_NEXT_PLATFORM_PATCH}")

# Debian package versions do not allow underscores. Keep branch-derived versions
# installable by normalizing every non-Debian-version character to '-'.
string(REGEX REPLACE "[^A-Za-z0-9.+~-]" "-" SIMANEAT_VERSION "${SIMANEAT_VERSION}")

# Debian version must start with a digit. Also normalize common "v1.2.3" tags.
if (SIMANEAT_VERSION MATCHES "^v[0-9].*")
  string(REGEX REPLACE "^v" "" SIMANEAT_VERSION "${SIMANEAT_VERSION}")
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
if (SIMANEAT_REQUIRE_LLIMA_ARTIFACTS)
  find_package(SimaLMM ${SIMANEAT_PLATFORM_VERSION} CONFIG REQUIRED)
else()
  # Core-only builds must not fail when LLiMa is absent, but if the package is
  # present we should build the real GenAI implementation instead of silently
  # shipping unavailable stubs.  This keeps local developer/CI environments
  # coherent without making LLiMa a hard dependency for normal Graph/Core builds.
  find_package(SimaLMM ${SIMANEAT_PLATFORM_VERSION} CONFIG QUIET)
endif()
if (SimaLMM_FOUND)
  message(STATUS "Using exported SimaLMM package from ${SimaLMM_DIR}")
else()
  message(STATUS
    "SimaLMM package not found; building GenAI/LLiMa APIs with unavailable stubs. "
    "Set SIMANEAT_REQUIRE_LLIMA_ARTIFACTS=ON to make this a configure error.")
endif()
find_path(SIMANEAT_JSONCPP_INCLUDE_DIR
  NAMES json/json.h
  PATHS
    "$ENV{SYSROOT}/usr/include/jsoncpp"
    "${CMAKE_SYSROOT}/usr/include/jsoncpp"
    "/usr/include/jsoncpp")
find_path(SIMANEAT_EIGEN3_INCLUDE_DIR
  NAMES Eigen/Core
  PATHS
    "$ENV{SYSROOT}/usr/include/eigen3"
    "${CMAKE_SYSROOT}/usr/include/eigen3"
    "${CMAKE_CURRENT_SOURCE_DIR}/../llima/third_party/eigen"
    "/usr/include/eigen3")
find_path(SIMANEAT_EIGEN_UNSUPPORTED_INCLUDE_DIR
  NAMES unsupported/Eigen/CXX11/Tensor
  PATHS
    "$ENV{SYSROOT}/usr/include/eigen3"
    "${CMAKE_SYSROOT}/usr/include/eigen3"
    "${CMAKE_CURRENT_SOURCE_DIR}/../llima/third_party/eigen"
    "/usr/include/eigen3")
if (SimaLMM_FOUND)
  pkg_check_modules(HTTPLIB REQUIRED cpp-httplib)
  find_library(SIMANEAT_HTTPLIB_LIBRARY
    NAMES cpp-httplib
    PATHS ${HTTPLIB_LIBRARY_DIRS}
    NO_DEFAULT_PATH)
  if (NOT SIMANEAT_HTTPLIB_LIBRARY)
    message(FATAL_ERROR "Could not find libcpp-httplib. Install libcpp-httplib-dev.")
  endif()
endif()

if (SIMANEAT_CPP_TIDY OR SIMANEAT_DOCS_ONLY_CONFIGURE)
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

if (SIMANEAT_CPP_TIDY OR SIMANEAT_DOCS_ONLY_CONFIGURE)
  pkg_check_modules(OPENCV QUIET opencv4)
  if (NOT OPENCV_FOUND)
    message(WARNING "OpenCV not found; configuring without OpenCV includes/libs (analysis-only mode).")
  endif()
else()
  pkg_check_modules(OPENCV REQUIRED opencv4)
endif()

set(SIMANEAT_USING_NEAT_INTERNALS_PACKAGE OFF)
if (SIMANEAT_USE_EXPORTED_NEAT_INTERNALS)
  find_package(NeatInternals ${SIMANEAT_PLATFORM_VERSION} CONFIG QUIET)
  if (NeatInternals_FOUND)
    set(SIMANEAT_USING_NEAT_INTERNALS_PACKAGE ON)
    message(STATUS "Using exported NeatInternals package from ${NeatInternals_DIR}")
  endif()
endif()

function(simaneat_imported_target_location out_var target_name)
  set(${out_var} "" PARENT_SCOPE)
  if (NOT TARGET ${target_name})
    return()
  endif()

  foreach(_prop
          IMPORTED_LOCATION_RELWITHDEBINFO
          IMPORTED_LOCATION_RELEASE
          IMPORTED_LOCATION_DEBUG
          IMPORTED_LOCATION_MINSIZEREL
          IMPORTED_LOCATION)
    get_target_property(_value ${target_name} ${_prop})
    if (_value AND NOT _value MATCHES "-NOTFOUND$")
      set(${out_var} "${_value}" PARENT_SCOPE)
      return()
    endif()
  endforeach()

  get_target_property(_configs ${target_name} IMPORTED_CONFIGURATIONS)
  foreach(_cfg IN LISTS _configs)
    string(TOUPPER "${_cfg}" _cfg_upper)
    get_target_property(_value ${target_name} "IMPORTED_LOCATION_${_cfg_upper}")
    if (_value AND NOT _value MATCHES "-NOTFOUND$")
      set(${out_var} "${_value}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

function(simaneat_find_exported_include_dir out_var target_name relative_header)
  set(${out_var} "" PARENT_SCOPE)
  if (NOT TARGET ${target_name})
    return()
  endif()

  get_target_property(_dirs ${target_name} INTERFACE_INCLUDE_DIRECTORIES)
  if (NOT _dirs OR _dirs MATCHES "-NOTFOUND$")
    return()
  endif()

  foreach(_dir IN LISTS _dirs)
    if (_dir MATCHES "^\\$<")
      continue()
    endif()
    if (EXISTS "${_dir}/${relative_header}")
      set(${out_var} "${_dir}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

function(simaneat_dependency_is_found out_var var_name)
  set(_found OFF)
  if (DEFINED ${var_name})
    set(_value "${${var_name}}")
    if (_value AND NOT _value MATCHES "-NOTFOUND$")
      set(_found ON)
    endif()
  endif()
  set(${out_var} ${_found} PARENT_SCOPE)
endfunction()

function(simaneat_require_dependency var_name error_message)
  simaneat_dependency_is_found(_found ${var_name})
  if (_found)
    return()
  endif()

  if (SIMANEAT_REQUIRE_NEAT_RUNTIME_ARTIFACTS)
    message(FATAL_ERROR "${error_message}")
  endif()

  message(STATUS
    "${error_message} Continuing because "
    "SIMANEAT_REQUIRE_NEAT_RUNTIME_ARTIFACTS=OFF.")
endfunction()

set(SIMANEAT_EXPORTED_NEAT_RUNTIME_DIR_CANDIDATES "")
if (SIMANEAT_USING_NEAT_INTERNALS_PACKAGE)
  foreach(_simaneat_internals_target IN ITEMS
      NeatInternals::neatprocessmlaruntime
      NeatInternals::PreparedRuntimeBridge
      NeatInternals::neattensorbufferfast
      NeatInternals::gstsimaaimeta)
    simaneat_imported_target_location(_simaneat_imported_runtime_location
      ${_simaneat_internals_target})
    if (_simaneat_imported_runtime_location)
      get_filename_component(_simaneat_imported_runtime_dir
        "${_simaneat_imported_runtime_location}" DIRECTORY)
      list(APPEND SIMANEAT_EXPORTED_NEAT_RUNTIME_DIR_CANDIDATES
        "${_simaneat_imported_runtime_dir}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES SIMANEAT_EXPORTED_NEAT_RUNTIME_DIR_CANDIDATES)
endif()

set(SIMANEAT_ELXR_SYSROOT /opt/toolchain/aarch64/modalix)
set(SIMANEAT_LOCAL_DEP_HEADER_ROOT
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/headers/usr/include)
set(SIMANEAT_INSTALLED_NEAT_PLUGIN_DIR_CANDIDATES
  ${SIMANEAT_ELXR_SYSROOT}/usr/lib/aarch64-linux-gnu/neat/gst-plugins
  ${SIMANEAT_ELXR_SYSROOT}/usr/lib/aarch64-linux-gnu/gstreamer-1.0
  ${SIMANEAT_ELXR_SYSROOT}/usr/lib/sima-neat/gst-plugins
  /usr/lib/aarch64-linux-gnu/neat/gst-plugins
  /lib/aarch64-linux-gnu/neat/gst-plugins
  /usr/lib/neat/gst-plugins
  /lib/neat/gst-plugins
)
set(SIMANEAT_INSTALLED_NEAT_RUNTIME_DIR_CANDIDATES
  ${SIMANEAT_ELXR_SYSROOT}/usr/lib/aarch64-linux-gnu/neat/runtime
  ${SIMANEAT_ELXR_SYSROOT}/usr/lib/aarch64-linux-gnu
  ${SIMANEAT_ELXR_SYSROOT}/usr/lib
  /usr/lib/aarch64-linux-gnu/neat/runtime
  /lib/aarch64-linux-gnu/neat/runtime
  /usr/lib/neat/runtime
  /lib/neat/runtime
)
set(SIMANEAT_INSTALLED_NEAT_INCLUDE_DIR_CANDIDATES
  ${SIMANEAT_LOCAL_DEP_HEADER_ROOT}/simaai
  ${SIMANEAT_ELXR_SYSROOT}/usr/include/simaai
  /usr/include/simaai
  /usr/local/include/simaai
  $ENV{HOME}/.local/include/simaai
)
set(SIMANEAT_REPO_NEAT_RUNTIME_DIR_CANDIDATES
  ${CMAKE_CURRENT_SOURCE_DIR}/../internals/build/core/prepared-runtime-bridge
  ${CMAKE_CURRENT_SOURCE_DIR}/../internals/build/core/processmla-runtime
  ${CMAKE_CURRENT_SOURCE_DIR}/../internals/build/core/metadata
  ${CMAKE_CURRENT_SOURCE_DIR}/../internals/build/core/utils
)
set(SIMANEAT_NEEDS_PREPARED_RUNTIME_BRIDGE OFF)
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/pipeline/internal/sima/PreparedRuntimeBuild.cpp")
  set(SIMANEAT_NEEDS_PREPARED_RUNTIME_BRIDGE ON)
endif()
set(SIMANEAT_NEAT_ALLOCATOR_HINTS
  ${SIMANEAT_INSTALLED_NEAT_PLUGIN_DIR_CANDIDATES}
)
list(APPEND SIMANEAT_NEAT_ALLOCATOR_HINTS
  /usr/lib/aarch64-linux-gnu/gstreamer-1.0
  /lib/aarch64-linux-gnu/gstreamer-1.0
  /usr/lib/gstreamer-1.0
  /lib/gstreamer-1.0
)

if (SIMANEAT_USING_NEAT_INTERNALS_PACKAGE)
  simaneat_imported_target_location(_simaneat_allocator_imported_location
    NeatInternals::Allocator)
  if (_simaneat_allocator_imported_location)
    set(SIMANEAT_NEAT_ALLOCATOR_LIB
      "${_simaneat_allocator_imported_location}"
      CACHE FILEPATH "NEAT allocator library from exported NeatInternals package" FORCE)
  endif()

  simaneat_imported_target_location(_simaneat_bufferpool_imported_location
    NeatInternals::BufferPool)
  if (_simaneat_bufferpool_imported_location)
    set(SIMANEAT_NEAT_BUFFERPOOL_LIB
      "${_simaneat_bufferpool_imported_location}"
      CACHE FILEPATH "NEAT buffer pool library from exported NeatInternals package" FORCE)
  endif()

  simaneat_imported_target_location(_simaneat_tensorbuffer_imported_location
    NeatInternals::TensorBuffer)
  if (_simaneat_tensorbuffer_imported_location)
    set(SIMANEAT_NEAT_TENSORBUFFER_LIB
      "${_simaneat_tensorbuffer_imported_location}"
      CACHE FILEPATH "NEAT tensor buffer library from exported NeatInternals package" FORCE)
  endif()

  if (SIMANEAT_NEEDS_PREPARED_RUNTIME_BRIDGE)
    simaneat_imported_target_location(_simaneat_bridge_imported_location
      NeatInternals::PreparedRuntimeBridge)
    if (_simaneat_bridge_imported_location)
      set(SIMANEAT_NEAT_PREPARED_RUNTIME_BRIDGE_LIB
        "${_simaneat_bridge_imported_location}"
        CACHE FILEPATH "Prepared-runtime bridge library from exported NeatInternals package" FORCE)
    endif()
  endif()

  simaneat_find_exported_include_dir(_simaneat_allocator_include_dir
    NeatInternals::Allocator
    gstsimaaiallocator.h)
  if (_simaneat_allocator_include_dir)
    set(SIMANEAT_NEAT_ALLOCATOR_INCLUDE_DIR
      "${_simaneat_allocator_include_dir}"
      CACHE PATH "NEAT allocator include directory from exported NeatInternals package" FORCE)
  endif()

  simaneat_find_exported_include_dir(_simaneat_tensorbuffer_include_dir
    NeatInternals::TensorBuffer
    gstsimaaitensorbuffer.h)
  if (_simaneat_tensorbuffer_include_dir)
    set(SIMANEAT_NEAT_TENSORBUFFER_INCLUDE_DIR
      "${_simaneat_tensorbuffer_include_dir}"
      CACHE PATH "NEAT tensor buffer include directory from exported NeatInternals package" FORCE)
  endif()

  simaneat_find_exported_include_dir(_simaneat_public_include_root
    NeatInternals::Headers
    neat/PreparedRuntimeBridge.h)
  if (_simaneat_public_include_root)
    set(SIMANEAT_NEAT_PUBLIC_INCLUDE_ROOT
      "${_simaneat_public_include_root}"
      CACHE PATH "NEAT public include root from exported NeatInternals package" FORCE)
  endif()
endif()

find_library(SIMANEAT_NEAT_ALLOCATOR_LIB
  NAMES gstneatallocator
  HINTS ${SIMANEAT_NEAT_ALLOCATOR_HINTS}
)
find_path(SIMANEAT_NEAT_ALLOCATOR_INCLUDE_DIR
  NAMES gstsimaaiallocator.h
  HINTS
    ${CMAKE_CURRENT_SOURCE_DIR}/../internals/core/allocator
    ${SIMANEAT_INSTALLED_NEAT_INCLUDE_DIR_CANDIDATES}
)

find_library(SIMANEAT_NEAT_BUFFERPOOL_LIB
  NAMES gstneatbufferpool
  HINTS ${SIMANEAT_NEAT_ALLOCATOR_HINTS}
)
set(SIMANEAT_NEAT_TENSORBUFFER_HINTS
  ${SIMANEAT_NEAT_ALLOCATOR_HINTS}
  $ENV{HOME}/.local/lib/gstreamer-1.0
  $ENV{HOME}/.local/lib/aarch64-linux-gnu/gstreamer-1.0
)

find_library(SIMANEAT_NEAT_TENSORBUFFER_LIB
  NAMES gstneattensorbuffer
  HINTS ${SIMANEAT_NEAT_TENSORBUFFER_HINTS}
)
find_library(SIMANEAT_SIMAAIMEM_LIB
  NAMES simaaimem
  HINTS
    /opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu
    /usr/lib/aarch64-linux-gnu
    /lib/aarch64-linux-gnu
    /usr/lib
    /lib
)
find_path(SIMANEAT_SIMAAIMEM_INCLUDE_ROOT
  NAMES simaai/simaai_memory.h
  HINTS
    ${SIMANEAT_LOCAL_DEP_HEADER_ROOT}
    /opt/toolchain/aarch64/modalix/usr/include
    /usr/include
    /usr/local/include
    $ENV{HOME}/.local/include
)
find_path(SIMANEAT_NEAT_TENSORBUFFER_INCLUDE_DIR
  NAMES gstsimaaitensorbuffer.h
  HINTS
    ${CMAKE_CURRENT_SOURCE_DIR}/../internals/core/tensorbuffer
    ${SIMANEAT_INSTALLED_NEAT_INCLUDE_DIR_CANDIDATES}
)

set(SIMANEAT_NEAT_PUBLIC_INCLUDE_ROOT
  "${SIMANEAT_NEAT_PUBLIC_INCLUDE_ROOT}"
  CACHE PATH
  "Root include directory that contains gst/, ev/, and neat/ headers exported by internals")
if (SIMANEAT_NEEDS_PREPARED_RUNTIME_BRIDGE AND NOT SIMANEAT_NEAT_PUBLIC_INCLUDE_ROOT)
  find_path(SIMANEAT_NEAT_PUBLIC_INCLUDE_ROOT
    NAMES gst/SimaPreparedRuntimeAbi.h
    HINTS
      ${SIMANEAT_LOCAL_DEP_HEADER_ROOT}
      /usr/include
      /usr/local/include
      $ENV{HOME}/.local/include
  )
endif()
if (SIMANEAT_NEEDS_PREPARED_RUNTIME_BRIDGE AND NOT SIMANEAT_NEAT_PUBLIC_INCLUDE_ROOT AND
    EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/gst/SimaPreparedRuntimeAbi.h" AND
    EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/ev/ev_tensor_abi.h" AND
    EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/neat/PreparedRuntimeBridge.h")
  set(SIMANEAT_NEAT_PUBLIC_INCLUDE_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    CACHE PATH
    "Root include directory that contains gst/, ev/, and neat/ headers exported by internals"
    FORCE)
endif()

set(SIMANEAT_NEAT_PREPARED_RUNTIME_BRIDGE_LIB
  "${SIMANEAT_NEAT_PREPARED_RUNTIME_BRIDGE_LIB}"
  CACHE FILEPATH
  "Path to the internals prepared-runtime bridge library")
if (SIMANEAT_NEEDS_PREPARED_RUNTIME_BRIDGE AND NOT SIMANEAT_NEAT_PREPARED_RUNTIME_BRIDGE_LIB)
  foreach(_simaneat_runtime_dir IN LISTS SIMANEAT_INSTALLED_NEAT_RUNTIME_DIR_CANDIDATES)
    if (EXISTS "${_simaneat_runtime_dir}/libneatpreparedruntimebridge.so")
      set(SIMANEAT_NEAT_PREPARED_RUNTIME_BRIDGE_LIB
        "${_simaneat_runtime_dir}/libneatpreparedruntimebridge.so"
        CACHE FILEPATH "Prepared-runtime bridge library from SDK/runtime directory" FORCE)
      break()
    endif()
  endforeach()
endif()
if (SIMANEAT_NEEDS_PREPARED_RUNTIME_BRIDGE AND NOT SIMANEAT_NEAT_PREPARED_RUNTIME_BRIDGE_LIB)
  find_library(SIMANEAT_NEAT_PREPARED_RUNTIME_BRIDGE_LIB
    NAMES neatpreparedruntimebridge
    HINTS
      ${CMAKE_CURRENT_SOURCE_DIR}/../internals/build/core/prepared-runtime-bridge
      ${SIMANEAT_INSTALLED_NEAT_RUNTIME_DIR_CANDIDATES}
      /usr/lib/aarch64-linux-gnu
      /lib/aarch64-linux-gnu
      /usr/lib
      /lib
  )
endif()

set(SIMANEAT_NEAT_PROCESSMLA_RUNTIME_LIB
  "${SIMANEAT_NEAT_PROCESSMLA_RUNTIME_LIB}"
  CACHE FILEPATH
  "Path to the NEAT processmla runtime library used by the prepared-runtime bridge")
if (SIMANEAT_NEEDS_PREPARED_RUNTIME_BRIDGE AND
    NOT SIMANEAT_USING_NEAT_INTERNALS_PACKAGE AND
    NOT SIMANEAT_NEAT_PROCESSMLA_RUNTIME_LIB)
  foreach(_simaneat_runtime_dir IN LISTS SIMANEAT_INSTALLED_NEAT_RUNTIME_DIR_CANDIDATES)
    if (EXISTS "${_simaneat_runtime_dir}/libneatprocessmlaruntime.so")
      set(SIMANEAT_NEAT_PROCESSMLA_RUNTIME_LIB
        "${_simaneat_runtime_dir}/libneatprocessmlaruntime.so"
        CACHE FILEPATH "ProcessMLA runtime library from SDK/runtime directory" FORCE)
      break()
    endif()
  endforeach()
endif()
if (SIMANEAT_NEEDS_PREPARED_RUNTIME_BRIDGE AND
    NOT SIMANEAT_USING_NEAT_INTERNALS_PACKAGE AND
    NOT SIMANEAT_NEAT_PROCESSMLA_RUNTIME_LIB)
  find_library(SIMANEAT_NEAT_PROCESSMLA_RUNTIME_LIB
    NAMES neatprocessmlaruntime
    HINTS
      ${CMAKE_CURRENT_SOURCE_DIR}/../internals/build/core/processmla-runtime
      ${SIMANEAT_INSTALLED_NEAT_RUNTIME_DIR_CANDIDATES}
      /usr/lib/aarch64-linux-gnu
      /lib/aarch64-linux-gnu
      /usr/lib
      /lib
  )
endif()

simaneat_require_dependency(SIMANEAT_NEAT_ALLOCATOR_LIB
  "Required NEAT allocator library not found: libgstneatallocator.so. Expected under a NEAT/GStreamer plugin directory.")
simaneat_require_dependency(SIMANEAT_NEAT_ALLOCATOR_INCLUDE_DIR
  "Required NEAT allocator headers not found: gstsimaaiallocator.h.")
simaneat_require_dependency(SIMANEAT_NEAT_BUFFERPOOL_LIB
  "Required NEAT buffer pool library not found: libgstneatbufferpool.so. Expected under a NEAT/GStreamer plugin directory.")
simaneat_require_dependency(SIMANEAT_NEAT_TENSORBUFFER_LIB
  "Required NEAT tensor buffer library not found: libgstneattensorbuffer.so. Expected under a NEAT/GStreamer plugin directory.")
simaneat_require_dependency(SIMANEAT_SIMAAIMEM_LIB
  "Required simaai memory library not found: libsimaaimem.so.")
simaneat_require_dependency(SIMANEAT_SIMAAIMEM_INCLUDE_ROOT
  "Required simaai memory headers not found: simaai/simaai_memory.h.")
simaneat_require_dependency(SIMANEAT_NEAT_TENSORBUFFER_INCLUDE_DIR
  "Required NEAT tensor buffer headers not found: gstsimaaitensorbuffer.h.")
if (SIMANEAT_NEEDS_PREPARED_RUNTIME_BRIDGE)
  simaneat_require_dependency(SIMANEAT_NEAT_PUBLIC_INCLUDE_ROOT
    "Prepared-runtime bridge headers not found. Expected an include root containing gst/SimaPreparedRuntimeAbi.h, ev/ev_tensor_abi.h, and neat/PreparedRuntimeBridge.h.")
  simaneat_dependency_is_found(_simaneat_has_public_include_root
    SIMANEAT_NEAT_PUBLIC_INCLUDE_ROOT)
  if (_simaneat_has_public_include_root AND NOT EXISTS
        "${SIMANEAT_NEAT_PUBLIC_INCLUDE_ROOT}/neat/PreparedRuntimeBridge.h")
    if (SIMANEAT_REQUIRE_NEAT_RUNTIME_ARTIFACTS)
      message(FATAL_ERROR
        "Prepared-runtime bridge API header not found under "
        "${SIMANEAT_NEAT_PUBLIC_INCLUDE_ROOT}/neat/PreparedRuntimeBridge.h"
      )
    endif()
    message(STATUS
      "Prepared-runtime bridge API header not found under "
      "${SIMANEAT_NEAT_PUBLIC_INCLUDE_ROOT}/neat/PreparedRuntimeBridge.h. "
      "Continuing because SIMANEAT_REQUIRE_NEAT_RUNTIME_ARTIFACTS=OFF.")
  endif()
  simaneat_require_dependency(SIMANEAT_NEAT_PREPARED_RUNTIME_BRIDGE_LIB
    "Prepared-runtime bridge library not found: libneatpreparedruntimebridge.so")
  if (NOT SIMANEAT_USING_NEAT_INTERNALS_PACKAGE)
    simaneat_require_dependency(SIMANEAT_NEAT_PROCESSMLA_RUNTIME_LIB
      "Required processmla runtime library not found: libneatprocessmlaruntime.so")
  endif()
endif()

set(SIMANEAT_SIMAAIMEM_TARGET_INCLUDE_ROOT "${SIMANEAT_SIMAAIMEM_INCLUDE_ROOT}")
simaneat_dependency_is_found(_simaneat_has_simaaimem_include_root
  SIMANEAT_SIMAAIMEM_INCLUDE_ROOT)
if (SIMANEAT_CPP_TIDY)
  set(SIMANEAT_TIDY_SIMAAIMEM_INCLUDE_ROOT
    "${CMAKE_BINARY_DIR}/tidy-include")
  file(MAKE_DIRECTORY "${SIMANEAT_TIDY_SIMAAIMEM_INCLUDE_ROOT}/simaai")
  if (_simaneat_has_simaaimem_include_root)
    set(_simaneat_tidy_simaaimem_header
      "#pragma once\n#include \"${SIMANEAT_SIMAAIMEM_INCLUDE_ROOT}/simaai/simaai_memory.h\"\n")
  else()
    set(_simaneat_tidy_simaaimem_header [=[
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct simaai_memory_t simaai_memory_t;

#define SIMAAI_MEM_TARGET_GENERIC (0)
#define SIMAAI_MEM_TARGET_OCM (1)
#define SIMAAI_MEM_TARGET_DMS0 (2)
#define SIMAAI_MEM_TARGET_DMS1 (3)
#define SIMAAI_MEM_TARGET_DMS2 (4)
#define SIMAAI_MEM_TARGET_DMS3 (5)
#define SIMAAI_MEM_TARGET_EV74 (SIMAAI_MEM_TARGET_GENERIC)
#define SIMAAI_MEM_TARGET_ALL (SIMAAI_MEM_TARGET_EV74)
#define SIMAAI_MEM_TARGET_UNKNOWN (SIMAAI_MEM_TARGET_ALL)

#define SIMAAI_MEM_FLAG_CACHED (1 << 0)
#define SIMAAI_MEM_FLAG_RDONLY (1 << 1)
#define SIMAAI_MEM_FLAG_DEFAULT (0x0)

#ifdef __cplusplus
}
#endif
]=])
  endif()
  file(WRITE
    "${SIMANEAT_TIDY_SIMAAIMEM_INCLUDE_ROOT}/simaai/simaai_memory.h"
    "${_simaneat_tidy_simaaimem_header}")
  set(SIMANEAT_SIMAAIMEM_TARGET_INCLUDE_ROOT
    "${SIMANEAT_TIDY_SIMAAIMEM_INCLUDE_ROOT}")
endif()

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
if (SimaLMM_FOUND)
  list(REMOVE_ITEM SIMANEAT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/genai/GenAIUnavailableStubs.cpp")
endif()
if (NOT SimaLMM_FOUND)
  # Keep lightweight public GenAI types buildable for API discovery, but do not compile
  # implementation TUs that require SimaLMM or call into LLiMa-backed model objects when the
  # LLiMa SDK/package is intentionally absent (clang-tidy and core-only CI).  The unavailable
  # stub TU above preserves linkability and reports a clear runtime diagnostic if those APIs are
  # used in a non-LLiMa build.
  list(REMOVE_ITEM SIMANEAT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/genai/ASRModel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/genai/GenAIModel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/genai/GraphFragments.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/genai/GenAIInternal.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/genai/GenAIServer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/genai/VisionLanguageModel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/genai/nodes/SpeechTranscriberNode.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/genai/nodes/VisionLanguageNode.cpp")
endif()

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
add_library(sima_neat_objects OBJECT
  ${SIMANEAT_SOURCES}
)
set_target_properties(sima_neat_objects PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_compile_definitions(sima_neat_objects PRIVATE SIMA_NEAT_INTERNAL=1)

if (NOT GIT_HASH)
  set(GIT_HASH "unknown")
endif()

add_library(sima_neat_common INTERFACE)

add_library(sima_neat SHARED
  $<TARGET_OBJECTS:sima_neat_objects>
)
set_target_properties(sima_neat PROPERTIES
  OUTPUT_NAME sima_neat
  # VERSION identifies the implementation/platform build. SOVERSION is the
  # independently versioned public C++ ABI that consumers record in DT_NEEDED.
  VERSION "${SIMANEAT_PLATFORM_VERSION}"
  SOVERSION "${SIMANEAT_ABI_VERSION}"
)

add_library(sima_neat_static STATIC
  $<TARGET_OBJECTS:sima_neat_objects>
)
set_target_properties(sima_neat_static PROPERTIES
  OUTPUT_NAME sima_neat
)

target_link_libraries(sima_neat PUBLIC sima_neat_common)
target_link_libraries(sima_neat_static PUBLIC sima_neat_common)
target_link_libraries(sima_neat_objects PRIVATE sima_neat_common)
add_custom_target(sima_neat_libraries DEPENDS sima_neat sima_neat_static)
target_compile_definitions(sima_neat_objects PRIVATE
  SIMA_NEAT_VERSION_STRING="${SIMANEAT_VERSION}"
  SIMA_NEAT_PLATFORM_VERSION_STRING="${SIMANEAT_PLATFORM_VERSION}"
  SIMA_NEAT_ABI_VERSION_STRING="${SIMANEAT_ABI_VERSION}"
  SIMANEAT_VERSION_STRING="${SIMANEAT_VERSION}"
  SIMANEAT_GIT_SHA="${GIT_HASH}"
)

option(SIMANEAT_ENABLE_EDGE_LTTNG_TRACEPOINTS
  "Emit framework-owned sima_neat_edge LTTng tracepoints from the graph runtime. Requires the SDK trace provider/library to export sima_neat_edge:message."
  OFF)
if (SIMANEAT_ENABLE_EDGE_LTTNG_TRACEPOINTS)
  target_compile_definitions(sima_neat_objects PRIVATE SIMA_NEAT_ENABLE_EDGE_LTTNG_TRACEPOINTS=1)
endif()

set(SIMANEAT_REPO_NEAT_INCLUDE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../internals/include")
set(SIMANEAT_REPO_NEAT_GST_INCLUDE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../internals/gst_plugins/include")

target_include_directories(sima_neat_common
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/pipeline>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/pipeline>
)
target_include_directories(sima_neat_objects
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pipeline/internal
    # Keep the source tree's public/internal bridge headers ahead of the
    # sibling internals checkout.  The sibling headers can lag while this repo
    # is under graph-contract development; source files in this target must see
    # the matching headers from core_graph_changes first.
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pipeline
)

foreach(_simaneat_llima_header_dep IN ITEMS
    "${SIMANEAT_JSONCPP_INCLUDE_DIR}"
    "${SIMANEAT_EIGEN3_INCLUDE_DIR}"
    "${SIMANEAT_EIGEN_UNSUPPORTED_INCLUDE_DIR}")
  if (_simaneat_llima_header_dep AND
      NOT _simaneat_llima_header_dep MATCHES "-NOTFOUND$")
    target_include_directories(sima_neat_objects SYSTEM PRIVATE "${_simaneat_llima_header_dep}")
  endif()
endforeach()

if (NOT SIMANEAT_USING_NEAT_INTERNALS_PACKAGE)
  foreach(_simaneat_neat_include_dir IN ITEMS
      "${SIMANEAT_REPO_NEAT_GST_INCLUDE_ROOT}"
      "${SIMANEAT_REPO_NEAT_INCLUDE_ROOT}"
      "${SIMANEAT_NEAT_ALLOCATOR_INCLUDE_DIR}"
      "${SIMANEAT_NEAT_TENSORBUFFER_INCLUDE_DIR}"
      "${SIMANEAT_SIMAAIMEM_TARGET_INCLUDE_ROOT}"
      "${SIMANEAT_NEAT_PUBLIC_INCLUDE_ROOT}")
    if (_simaneat_neat_include_dir AND
        NOT _simaneat_neat_include_dir MATCHES "-NOTFOUND$")
      target_include_directories(sima_neat_objects PRIVATE
        "${_simaneat_neat_include_dir}")
    endif()
  endforeach()
elseif (SIMANEAT_CPP_TIDY AND SIMANEAT_SIMAAIMEM_TARGET_INCLUDE_ROOT)
  target_include_directories(sima_neat_objects PRIVATE
    "${SIMANEAT_SIMAAIMEM_TARGET_INCLUDE_ROOT}")
endif()

# GStreamer/OpenCV headers are part of the public surface (e.g. pipeline/Graph.h),
# but their absolute include paths are build-host specific (the SDK cross sysroot
# under /opt/toolchain/... vs a DevKit's native /usr). Use them to build Neat
# itself via $<BUILD_INTERFACE:...>, but do NOT bake them into the exported
# target -- SimaNeatConfig.cmake re-discovers them with pkg-config on the
# consuming machine so the installed package is relocatable (it works both when
# cross-compiling in the Neat SDK and when building natively on the DevKit).
if (GST_FOUND)
  target_include_directories(sima_neat_common SYSTEM INTERFACE "$<BUILD_INTERFACE:${GST_INCLUDE_DIRS}>")
endif()
if (OPENCV_FOUND)
  target_include_directories(sima_neat_common SYSTEM INTERFACE "$<BUILD_INTERFACE:${OPENCV_INCLUDE_DIRS}>")
endif()

if (GST_FOUND)
  target_compile_options(sima_neat_objects PRIVATE ${GST_CFLAGS_OTHER})
endif()
if (OPENCV_FOUND)
  target_compile_options(sima_neat_objects PRIVATE ${OPENCV_CFLAGS_OTHER})
endif()
if (SimaLMM_FOUND)
  target_compile_options(sima_neat_objects PRIVATE ${HTTPLIB_CFLAGS_OTHER})
endif()

if (OPENCV_FOUND)
  target_link_directories(sima_neat PRIVATE ${OPENCV_LIBRARY_DIRS})
  target_link_directories(sima_neat_static PRIVATE ${OPENCV_LIBRARY_DIRS})
endif()
if (GST_FOUND)
  target_link_directories(sima_neat PRIVATE ${GST_LIBRARY_DIRS})
  target_link_directories(sima_neat_static PRIVATE ${GST_LIBRARY_DIRS})
endif()

if (SimaLMM_FOUND)
  target_link_libraries(sima_neat_common
    INTERFACE
      SimaLMM::sima_lmm_runtime
  )
endif()

if (SimaLMM_FOUND)
  target_include_directories(sima_neat_objects SYSTEM PRIVATE ${HTTPLIB_INCLUDE_DIRS})
  target_link_libraries(sima_neat PRIVATE ${SIMANEAT_HTTPLIB_LIBRARY})
  target_link_libraries(sima_neat_static PRIVATE ${SIMANEAT_HTTPLIB_LIBRARY})
endif()

if (GST_FOUND)
  target_link_libraries(sima_neat_common INTERFACE ${GST_LIBRARIES})
endif()
if (OPENCV_FOUND)
  target_link_libraries(sima_neat_common INTERFACE ${OPENCV_LIBRARIES})
endif()

if (SIMA_WITH_OPENCV)
  target_compile_definitions(sima_neat_common INTERFACE SIMA_WITH_OPENCV)
endif()

if (SIMANEAT_USING_NEAT_INTERNALS_PACKAGE)
  target_link_libraries(sima_neat_common INTERFACE
    NeatInternals::Allocator
    NeatInternals::BufferPool
    NeatInternals::TensorBuffer)
else()
  if (SIMANEAT_NEAT_ALLOCATOR_LIB)
    target_link_libraries(sima_neat_common INTERFACE ${SIMANEAT_NEAT_ALLOCATOR_LIB})
  endif()
  if (SIMANEAT_NEAT_BUFFERPOOL_LIB)
    target_link_libraries(sima_neat_common INTERFACE ${SIMANEAT_NEAT_BUFFERPOOL_LIB})
  endif()
  if (SIMANEAT_NEAT_TENSORBUFFER_LIB)
    target_link_libraries(sima_neat_common INTERFACE ${SIMANEAT_NEAT_TENSORBUFFER_LIB})
  endif()
endif()
if (SIMANEAT_SIMAAIMEM_LIB)
  target_link_libraries(sima_neat_common INTERFACE
    "$<BUILD_INTERFACE:${SIMANEAT_SIMAAIMEM_LIB}>"
    "$<INSTALL_INTERFACE:simaaimem>")
endif()
if (SIMANEAT_NEEDS_PREPARED_RUNTIME_BRIDGE)
  if (SIMANEAT_USING_NEAT_INTERNALS_PACKAGE)
    target_link_libraries(sima_neat_common INTERFACE NeatInternals::PreparedRuntimeBridge)
  else()
    if (SIMANEAT_NEAT_PREPARED_RUNTIME_BRIDGE_LIB)
      target_link_libraries(sima_neat_common INTERFACE
        ${SIMANEAT_NEAT_PREPARED_RUNTIME_BRIDGE_LIB})
    endif()
    if (SIMANEAT_NEAT_PROCESSMLA_RUNTIME_LIB)
      target_link_libraries(sima_neat_common INTERFACE
        ${SIMANEAT_NEAT_PROCESSMLA_RUNTIME_LIB})
    endif()
  endif()
endif()

target_compile_definitions(sima_neat_common INTERFACE SIMA_HAS_SIMAAI_POOL=1)

set(SIMANEAT_NEAT_ALLOCATOR_DIR "")
if (SIMANEAT_NEAT_ALLOCATOR_LIB)
  get_filename_component(SIMANEAT_NEAT_ALLOCATOR_DIR
    "${SIMANEAT_NEAT_ALLOCATOR_LIB}" DIRECTORY)
endif()
set(SIMANEAT_NEAT_BUFFERPOOL_DIR "")
if (SIMANEAT_NEAT_BUFFERPOOL_LIB)
  get_filename_component(SIMANEAT_NEAT_BUFFERPOOL_DIR
    "${SIMANEAT_NEAT_BUFFERPOOL_LIB}" DIRECTORY)
endif()
set(SIMANEAT_NEAT_TENSORBUFFER_DIR "")
if (SIMANEAT_NEAT_TENSORBUFFER_LIB)
  get_filename_component(SIMANEAT_NEAT_TENSORBUFFER_DIR
    "${SIMANEAT_NEAT_TENSORBUFFER_LIB}" DIRECTORY)
endif()
set(SIMANEAT_CTEST_NEAT_PLUGIN_DIR "" CACHE PATH
  "NEAT plugin directory used by core CTest/tutorial executions")
if (NOT SIMANEAT_CTEST_NEAT_PLUGIN_DIR AND DEFINED ENV{SIMA_GST_PLUGIN_DIR} AND
    NOT "$ENV{SIMA_GST_PLUGIN_DIR}" STREQUAL "")
  set(SIMANEAT_CTEST_NEAT_PLUGIN_DIR "$ENV{SIMA_GST_PLUGIN_DIR}" CACHE PATH
    "NEAT plugin directory used by core CTest/tutorial executions" FORCE)
endif()
if (NOT SIMANEAT_CTEST_NEAT_PLUGIN_DIR)
  set(SIMANEAT_CTEST_NEAT_PLUGIN_DIR "${SIMANEAT_NEAT_ALLOCATOR_DIR}" CACHE PATH
    "NEAT plugin directory used by core CTest/tutorial executions" FORCE)
endif()
set(_SIMANEAT_CTEST_NEAT_LD_LIBRARY_DIR_CANDIDATES
  "${SIMANEAT_CTEST_NEAT_PLUGIN_DIR}"
  "${SIMANEAT_NEAT_ALLOCATOR_DIR}"
  "${SIMANEAT_NEAT_BUFFERPOOL_DIR}"
  "${SIMANEAT_NEAT_TENSORBUFFER_DIR}"
  ${SIMANEAT_EXPORTED_NEAT_RUNTIME_DIR_CANDIDATES}
  ${SIMANEAT_REPO_NEAT_RUNTIME_DIR_CANDIDATES}
  ${SIMANEAT_INSTALLED_NEAT_RUNTIME_DIR_CANDIDATES}
)
set(SIMANEAT_CTEST_NEAT_LD_LIBRARY_DIRS "")
foreach(_simaneat_runtime_dir IN LISTS _SIMANEAT_CTEST_NEAT_LD_LIBRARY_DIR_CANDIDATES)
  if (_simaneat_runtime_dir AND EXISTS "${_simaneat_runtime_dir}")
    list(APPEND SIMANEAT_CTEST_NEAT_LD_LIBRARY_DIRS "${_simaneat_runtime_dir}")
  endif()
endforeach()
if (DEFINED ENV{LD_LIBRARY_PATH} AND NOT "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
  list(APPEND SIMANEAT_CTEST_NEAT_LD_LIBRARY_DIRS "$ENV{LD_LIBRARY_PATH}")
endif()
list(REMOVE_DUPLICATES SIMANEAT_CTEST_NEAT_LD_LIBRARY_DIRS)
string(JOIN ":" SIMANEAT_CTEST_NEAT_LD_LIBRARY_PATH ${SIMANEAT_CTEST_NEAT_LD_LIBRARY_DIRS})

set(SIMANEAT_TEST_RUNTIME_MANIFEST_DIR "${CMAKE_BINARY_DIR}/test-fixtures")
set(SIMANEAT_TEST_RUNTIME_MANIFEST "${SIMANEAT_TEST_RUNTIME_MANIFEST_DIR}/runtime_manifest.json")
file(MAKE_DIRECTORY "${SIMANEAT_TEST_RUNTIME_MANIFEST_DIR}")

file(RELATIVE_PATH SIMANEAT_TEST_RUNTIME_SOURCE_ROOT_REL
  "${CMAKE_BINARY_DIR}"
  "${CMAKE_SOURCE_DIR}")
file(RELATIVE_PATH SIMANEAT_TEST_RUNTIME_REPO_TMP_ROOT_REL
  "${CMAKE_BINARY_DIR}"
  "${CMAKE_SOURCE_DIR}/tmp")
file(RELATIVE_PATH SIMANEAT_TEST_RUNTIME_MODEL_ARCHIVE_FIXTURE_ROOT_REL
  "${CMAKE_BINARY_DIR}"
  "${CMAKE_BINARY_DIR}/test-assets/model-archive")
file(RELATIVE_PATH SIMANEAT_TEST_RUNTIME_MODEL_ARCHIVE_FIXTURE_MANIFEST_REL
  "${CMAKE_BINARY_DIR}"
  "${CMAKE_BINARY_DIR}/test-assets/model-archive/fixtures_manifest.json")
file(RELATIVE_PATH SIMANEAT_TEST_RUNTIME_MPK_FIXTURE_ROOT_REL
  "${CMAKE_BINARY_DIR}"
  "${CMAKE_SOURCE_DIR}/tests/assets/mpk")
file(RELATIVE_PATH SIMANEAT_TEST_RUNTIME_MPK_FIXTURE_MANIFEST_REL
  "${CMAKE_BINARY_DIR}"
  "${CMAKE_SOURCE_DIR}/tests/assets/mpk/fixtures_manifest.json")
file(RELATIVE_PATH SIMANEAT_TEST_RUNTIME_DECODER_FIXTURE_REL
  "${CMAKE_BINARY_DIR}"
  "${CMAKE_SOURCE_DIR}/tests/assets/decoder/dynamic_caps.h264")
file(RELATIVE_PATH SIMANEAT_TEST_RUNTIME_CODEC_PERF_H264_FIXTURE_REL
  "${CMAKE_BINARY_DIR}"
  "${CMAKE_BINARY_DIR}/test-assets/codec-perf/h264_1280x720_30fps_no_sei.h264")
file(RELATIVE_PATH SIMANEAT_TEST_RUNTIME_CODEC_PERF_H265_FIXTURE_REL
  "${CMAKE_BINARY_DIR}"
  "${CMAKE_BINARY_DIR}/test-assets/codec-perf/h265_1280x720_30fps_no_sei.h265")
file(RELATIVE_PATH SIMANEAT_TEST_RUNTIME_SHARED_TEST_ASSET_ROOT_REL
  "${CMAKE_BINARY_DIR}"
  "${CMAKE_SOURCE_DIR}")

set(SIMANEAT_TEST_RUNTIME_MANIFEST_CONTENT [=[
{
  "format_version": 1,
  "generator": "core/CMakeLists.txt",
  "source_root_rel": "@SIMANEAT_TEST_RUNTIME_SOURCE_ROOT_REL@",
  "repo_tmp_root_rel": "@SIMANEAT_TEST_RUNTIME_REPO_TMP_ROOT_REL@",
  "model_archive_fixture_root_rel": "@SIMANEAT_TEST_RUNTIME_MODEL_ARCHIVE_FIXTURE_ROOT_REL@",
  "model_archive_fixture_manifest_rel": "@SIMANEAT_TEST_RUNTIME_MODEL_ARCHIVE_FIXTURE_MANIFEST_REL@",
  "mpk_fixture_root_rel": "@SIMANEAT_TEST_RUNTIME_MPK_FIXTURE_ROOT_REL@",
  "mpk_fixture_manifest_rel": "@SIMANEAT_TEST_RUNTIME_MPK_FIXTURE_MANIFEST_REL@",
  "decoder_fixture_rel": "@SIMANEAT_TEST_RUNTIME_DECODER_FIXTURE_REL@",
  "codec_perf_h264_fixture_rel": "@SIMANEAT_TEST_RUNTIME_CODEC_PERF_H264_FIXTURE_REL@",
  "codec_perf_h265_fixture_rel": "@SIMANEAT_TEST_RUNTIME_CODEC_PERF_H265_FIXTURE_REL@",
  "shared_test_asset_root_rel": "@SIMANEAT_TEST_RUNTIME_SHARED_TEST_ASSET_ROOT_REL@"
}
]=])
string(CONFIGURE "${SIMANEAT_TEST_RUNTIME_MANIFEST_CONTENT}"
  SIMANEAT_TEST_RUNTIME_MANIFEST_CONTENT
  @ONLY)
file(WRITE "${SIMANEAT_TEST_RUNTIME_MANIFEST}" "${SIMANEAT_TEST_RUNTIME_MANIFEST_CONTENT}")

# Companion manifest for the *installed* extras deb layout. The build-tree
# manifest above carries CMAKE_BINARY_DIR-relative paths to the source-tree
# fixture roots, which only exist on the build host. After `cpack` packages
# the extras component, tests run from /usr/lib/sima-neat/tests/ and need a
# manifest whose paths resolve into the matching /usr/lib/sima-neat/
# test-assets/ tree shipped by the install rules in tests/CMakeLists.txt.
# find_runtime_manifest_path() walks up from the test binary, so we land the
# manifest at lib/sima-neat/test-fixtures/runtime_manifest.json — first
# walk-up hit from lib/sima-neat/tests/ — and keep every fixture under
# lib/sima-neat/test-assets/.
set(SIMANEAT_TEST_RUNTIME_MANIFEST_INSTALLED
  "${SIMANEAT_TEST_RUNTIME_MANIFEST_DIR}/runtime_manifest_installed.json")
set(SIMANEAT_TEST_RUNTIME_MANIFEST_INSTALLED_CONTENT [=[
{
  "format_version": 1,
  "generator": "core/CMakeLists.txt (installed-layout variant)",
  "source_root_rel": "",
  "repo_tmp_root_rel": "",
  "model_archive_fixture_root_rel": "test-assets/model-archive",
  "model_archive_fixture_manifest_rel": "test-assets/model-archive/fixtures_manifest.json",
  "mpk_fixture_root_rel": "test-assets/mpk",
  "mpk_fixture_manifest_rel": "test-assets/mpk/fixtures_manifest.json",
  "decoder_fixture_rel": "test-assets/decoder/dynamic_caps.h264",
  "codec_perf_h264_fixture_rel": "test-assets/codec-perf/h264_1280x720_30fps_no_sei.h264",
  "codec_perf_h265_fixture_rel": "test-assets/codec-perf/h265_1280x720_30fps_no_sei.h265",
  "shared_test_asset_root_rel": "../../share/sima-neat/test-assets"
}
]=])
file(WRITE "${SIMANEAT_TEST_RUNTIME_MANIFEST_INSTALLED}"
  "${SIMANEAT_TEST_RUNTIME_MANIFEST_INSTALLED_CONTENT}")

option(SIMANEAT_CTEST_UNIQUE_GST_REGISTRY
  "Use an isolated GStreamer registry file per CTest test. This preserves the legacy maximum-isolation behavior but makes installed DevKit CI much slower."
  OFF)

function(simaneat_append_ctest_env test_name)
  set(_plugin_dir "${SIMANEAT_CTEST_NEAT_PLUGIN_DIR}")
  set(_ld_library_path "${SIMANEAT_CTEST_NEAT_LD_LIBRARY_PATH}")
  if (DEFINED SIMA_TEST_NEAT_PLUGIN_DIR AND NOT "${SIMA_TEST_NEAT_PLUGIN_DIR}" STREQUAL "")
    set(_plugin_dir "${SIMA_TEST_NEAT_PLUGIN_DIR}")
  endif()
  if (DEFINED SIMA_TEST_NEAT_LD_LIBRARY_PATH AND
      NOT "${SIMA_TEST_NEAT_LD_LIBRARY_PATH}" STREQUAL "")
    set(_ld_library_path "${SIMA_TEST_NEAT_LD_LIBRARY_PATH}")
  endif()

  # Default to the shared, content-addressed registry cache created by GstInit.cpp.
  # That cache key includes the selected NEAT plugin directory and filtered system
  # plugin set, so tests still avoid stale plugin metadata without forcing every
  # process to rescan all plugins. Keep an escape hatch for tests/debug sessions
  # that need the pre-existing one-registry-file-per-test isolation.
  if (SIMANEAT_CTEST_UNIQUE_GST_REGISTRY)
    set(_gst_registry "${CMAKE_BINARY_DIR}/sima_gst_registry_${test_name}.bin")
    set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "GST_REGISTRY=${_gst_registry}")
    set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "GST_REGISTRY_1_0=${_gst_registry}")
    set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT
      "GST_REGISTRY_REUSE_PLUGIN_SCANNER=no")
    set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "SIMA_GST_UNIQUE_REGISTRY=1")
  endif()
  if (NOT "${_plugin_dir}" STREQUAL "")
    set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "GST_PLUGIN_PATH=${_plugin_dir}")
    set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "GST_PLUGIN_PATH_1_0=${_plugin_dir}")
    set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "GST_PLUGIN_SYSTEM_PATH_1_0=")
    set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "SIMA_GST_PLUGIN_DIR=${_plugin_dir}")
  endif()
  if (NOT "${_ld_library_path}" STREQUAL "")
    set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "LD_LIBRARY_PATH=${_ld_library_path}")
  endif()
  set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "SIMA_CLI_CHECK_FOR_UPDATE=0")
endfunction()

# ------------------------------------------------------------------------------
# Internal-only helper targets (tests/bench/tools/bindings)
# ------------------------------------------------------------------------------
add_library(sima_neat_internal_headers INTERFACE)
target_include_directories(sima_neat_internal_headers INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/pipeline
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/pipeline
)
if (NOT SIMANEAT_USING_NEAT_INTERNALS_PACKAGE)
  foreach(_simaneat_neat_internal_include_dir IN ITEMS
      "${SIMANEAT_REPO_NEAT_GST_INCLUDE_ROOT}"
      "${SIMANEAT_REPO_NEAT_INCLUDE_ROOT}"
      "${SIMANEAT_NEAT_ALLOCATOR_INCLUDE_DIR}"
      "${SIMANEAT_NEAT_TENSORBUFFER_INCLUDE_DIR}"
      "${SIMANEAT_SIMAAIMEM_TARGET_INCLUDE_ROOT}"
      "${SIMANEAT_NEAT_PUBLIC_INCLUDE_ROOT}")
    if (_simaneat_neat_internal_include_dir AND
        NOT _simaneat_neat_internal_include_dir MATCHES "-NOTFOUND$")
      target_include_directories(sima_neat_internal_headers INTERFACE
        "${_simaneat_neat_internal_include_dir}")
    endif()
  endforeach()
endif()
target_compile_definitions(sima_neat_internal_headers INTERFACE SIMA_NEAT_INTERNAL=1)

add_library(sima_neat_internal INTERFACE)
target_link_libraries(sima_neat_internal INTERFACE
  sima_neat
  sima_neat_internal_headers
)

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
  TARGETS sima_neat sima_neat_static sima_neat_common
  EXPORT SimaNeatTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT dev
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT core
    NAMELINK_COMPONENT dev
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT core
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT dev
  # Low-level actor-runtime graph headers are compiler/runtime internals.  The
  # public application surface is simaai::neat::Graph / Run via neat.h; do not
  # install a second public Graph concept.
  PATTERN "graph" EXCLUDE
  PATTERN "graph/*" EXCLUDE
  PATTERN "genai/nodes" EXCLUDE
  PATTERN "genai/nodes/*" EXCLUDE
  PATTERN "pipeline/internal" EXCLUDE
  PATTERN "pipeline/internal/*" EXCLUDE
  PATTERN "ProcessMlaRuntimeConfig.h" EXCLUDE
  PATTERN "SimaPluginStaticManifestAbi.h" EXCLUDE
  PATTERN "SimaPreparedRuntimeAbi.h" EXCLUDE
  PATTERN "SimaTensorSetMetaAbi.h" EXCLUDE
  PATTERN "ev_tensor_abi.h" EXCLUDE
  PATTERN "PreparedRuntimeBridge.h" EXCLUDE
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/distribution/skills/sima-neat/
  DESTINATION ${CMAKE_INSTALL_DATADIR}/sima-neat/skills/sima-neat
  COMPONENT dev
)

if (SIMANEAT_BUILDINFO_JSON)
  if (NOT EXISTS "${SIMANEAT_BUILDINFO_JSON}")
    message(FATAL_ERROR "SIMANEAT_BUILDINFO_JSON does not exist: ${SIMANEAT_BUILDINFO_JSON}")
  endif()
  install(
    FILES ${SIMANEAT_BUILDINFO_JSON}
    DESTINATION ${CMAKE_INSTALL_DATADIR}/sima-neat
    RENAME buildinfo.json
    COMPONENT core
  )
endif()

install(
  FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/fix_devkit_runtime.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/neat
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
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
  COMPONENT dev
)

set(SimaNeat_INSTALL_CMAKEDIR
  ${CMAKE_INSTALL_LIBDIR}/cmake/SimaNeat
)

set(SIMANEAT_CONFIG_WITH_LLIMA FALSE)
if (SimaLMM_FOUND)
  set(SIMANEAT_CONFIG_WITH_LLIMA TRUE)
endif()

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SimaNeatConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/SimaNeatConfig.cmake
  INSTALL_DESTINATION ${SimaNeat_INSTALL_CMAKEDIR}
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/SimaNeatConfigVersion.cmake
  VERSION ${SIMANEAT_PLATFORM_VERSION}
  COMPATIBILITY SameMajorVersion
)

# Export build-tree package targets as well so external helper binaries built
# against the active build tree do not silently fall back to ad hoc archive
# imports that can drift out of sync with the current plugin/runtime build.
export(
  EXPORT SimaNeatTargets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/SimaNeatTargets.cmake
  NAMESPACE SimaNeat::
)

install(
  EXPORT SimaNeatTargets
  FILE SimaNeatTargets.cmake
  NAMESPACE SimaNeat::
  DESTINATION ${SimaNeat_INSTALL_CMAKEDIR}
  COMPONENT dev
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/SimaNeatConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/SimaNeatConfigVersion.cmake
  DESTINATION ${SimaNeat_INSTALL_CMAKEDIR}
  COMPONENT dev
)

# ------------------------------------------------------------------------------
# CPack (DEB)
# ------------------------------------------------------------------------------

set(CPACK_PACKAGE_NAME "sima-neat")
set(CPACK_PACKAGE_VENDOR "SiMa.ai")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "SiMa NEAT core runtime and development packages")

set(CPACK_COMPONENTS_ALL core dev)
set(CPACK_DEB_COMPONENT_INSTALL ON)

set(CPACK_COMPONENT_CORE_DESCRIPTION
    "SiMa NEAT runtime library and command-line helpers")
set(CPACK_COMPONENT_DEV_DESCRIPTION
    "SiMa NEAT C++ development headers and CMake package")

set(CPACK_PACKAGE_VERSION "${SIMANEAT_VERSION}")

set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "SiMa.ai")

set(CPACK_DEBIAN_PACKAGE_SECTION "libs")
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")

set(CPACK_DEBIAN_CORE_PACKAGE_NAME "sima-neat")
set(CPACK_DEBIAN_DEV_PACKAGE_NAME "sima-neat-dev")

set(SIMANEAT_CORE_DEBIAN_DEPENDS
  "libgstreamer1.0-0"
  "libgstreamer-plugins-base1.0-0"
  "libgstrtspserver-1.0-0"
  "libglib2.0-0"
  "libopencv-core406"
  "libopencv-imgproc406"
  "libopencv-imgcodecs406"
  "neat-runtime (>= ${SIMANEAT_DEP_PACKAGE_MIN_VERSION})"
  "neat-runtime (<< ${SIMANEAT_DEP_PACKAGE_MAX_VERSION})"
  "neat-gst-plugins (>= ${SIMANEAT_DEP_PACKAGE_MIN_VERSION})"
  "neat-gst-plugins (<< ${SIMANEAT_DEP_PACKAGE_MAX_VERSION})"
)
if (SimaLMM_FOUND)
  list(APPEND SIMANEAT_CORE_DEBIAN_DEPENDS
    "sima-lmm-core (>= ${SIMANEAT_DEP_PACKAGE_MIN_VERSION})"
    "sima-lmm-core (<< ${SIMANEAT_DEP_PACKAGE_MAX_VERSION})"
    "libcpp-httplib0.11")
endif()
string(JOIN ", " CPACK_DEBIAN_CORE_PACKAGE_DEPENDS ${SIMANEAT_CORE_DEBIAN_DEPENDS})
set(SIMANEAT_DEV_DEBIAN_DEPENDS
  "sima-neat (= ${CPACK_PACKAGE_VERSION})"
  "pkg-config"
  "simaai-memory-lib-dev (>= ${SIMANEAT_PLATFORM_DEP_PACKAGE_MIN_VERSION})"
  "simaai-memory-lib-dev (<< ${SIMANEAT_PLATFORM_DEP_PACKAGE_MAX_VERSION})"
  "libgstreamer1.0-dev"
  "libgstreamer-plugins-base1.0-dev"
  "libgstrtspserver-1.0-dev"
  "libglib2.0-dev"
  "libopencv-dev"
  "nlohmann-json3-dev"
  "neat-internals-dev (>= ${SIMANEAT_DEP_PACKAGE_MIN_VERSION})"
  "neat-internals-dev (<< ${SIMANEAT_DEP_PACKAGE_MAX_VERSION})"
)
if (SimaLMM_FOUND)
  list(APPEND SIMANEAT_DEV_DEBIAN_DEPENDS
    "sima-lmm-dev (>= ${SIMANEAT_DEP_PACKAGE_MIN_VERSION})"
    "sima-lmm-dev (<< ${SIMANEAT_DEP_PACKAGE_MAX_VERSION})")
endif()
string(JOIN ", " CPACK_DEBIAN_DEV_PACKAGE_DEPENDS ${SIMANEAT_DEV_DEBIAN_DEPENDS})
set(CPACK_INSTALL_CMAKE_PROJECTS
  "${CMAKE_BINARY_DIR};SimaNeat;ALL;/"
)

include(CPack)

# ------------------------------------------------------------------------------
# Docs (Doxygen)
# ------------------------------------------------------------------------------
if (SIMANEAT_BUILD_DOCS)
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
endif()
