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
option(SIMANEAT_USE_EXPORTED_NEAT_INTERNALS
  "Prefer the exported NeatInternals SDK package over ad hoc library/header discovery"
  ON)

set(SIMANEAT_REQUIRE_NEAT_RUNTIME_ARTIFACTS_DEFAULT ON)
if (SIMANEAT_CPP_TIDY)
  set(SIMANEAT_REQUIRE_NEAT_RUNTIME_ARTIFACTS_DEFAULT OFF)
endif()
option(SIMANEAT_REQUIRE_NEAT_RUNTIME_ARTIFACTS
  "Require NEAT runtime libraries/headers during configure"
  ${SIMANEAT_REQUIRE_NEAT_RUNTIME_ARTIFACTS_DEFAULT})

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

# Debian package versions do not allow underscores. Keep branch-derived versions
# installable by normalizing every non-Debian-version character to '-'.
string(REGEX REPLACE "[^A-Za-z0-9.+~-]" "-" SIMANEAT_VERSION "${SIMANEAT_VERSION}")

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

set(SIMANEAT_USING_NEAT_INTERNALS_PACKAGE OFF)
if (SIMANEAT_USE_EXPORTED_NEAT_INTERNALS)
  find_package(NeatInternals CONFIG QUIET)
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
      NeatInternals::simaaineatprofiler
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
  /usr/lib/aarch64-linux-gnu/neat/runtime
  /lib/aarch64-linux-gnu/neat/runtime
  /usr/lib/neat/runtime
  /lib/neat/runtime
)
set(SIMANEAT_INSTALLED_NEAT_INCLUDE_DIR_CANDIDATES
  ${SIMANEAT_ELXR_SYSROOT}/usr/include/simaai
  /usr/include/simaai
  /usr/local/include/simaai
  $ENV{HOME}/.local/include/simaai
)
set(SIMANEAT_REPO_NEAT_RUNTIME_DIR_CANDIDATES
  ${CMAKE_CURRENT_SOURCE_DIR}/../internals/build/core/prepared-runtime-bridge
  ${CMAKE_CURRENT_SOURCE_DIR}/../internals/build/core/processmla-runtime
  ${CMAKE_CURRENT_SOURCE_DIR}/../internals/build/core/metadata
  ${CMAKE_CURRENT_SOURCE_DIR}/../internals/build/core/profiler
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
if (SIMANEAT_CPP_TIDY AND _simaneat_has_simaaimem_include_root)
  set(SIMANEAT_TIDY_SIMAAIMEM_INCLUDE_ROOT
    "${CMAKE_BINARY_DIR}/tidy-include")
  file(MAKE_DIRECTORY "${SIMANEAT_TIDY_SIMAAIMEM_INCLUDE_ROOT}/simaai")
  file(WRITE
    "${SIMANEAT_TIDY_SIMAAIMEM_INCLUDE_ROOT}/simaai/simaai_memory.h"
    "#pragma once\n#include \"${SIMANEAT_SIMAAIMEM_INCLUDE_ROOT}/simaai/simaai_memory.h\"\n")
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
  ${CMAKE_CURRENT_SOURCE_DIR}/src/pipeline/session/internal/SessionTestHooks.cpp
)

target_compile_definitions(sima_neat PRIVATE SIMA_NEAT_INTERNAL=1)

set(SIMANEAT_REPO_NEAT_INCLUDE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../internals/include")
set(SIMANEAT_REPO_NEAT_GST_INCLUDE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../internals/gst_plugins/include")

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
      target_include_directories(sima_neat PRIVATE
        "${_simaneat_neat_include_dir}")
    endif()
  endforeach()
endif()

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

if (SIMANEAT_USING_NEAT_INTERNALS_PACKAGE)
  target_link_libraries(sima_neat PUBLIC
    NeatInternals::Allocator
    NeatInternals::BufferPool
    NeatInternals::TensorBuffer)
else()
  if (SIMANEAT_NEAT_ALLOCATOR_LIB)
    target_link_libraries(sima_neat PUBLIC ${SIMANEAT_NEAT_ALLOCATOR_LIB})
  endif()
  if (SIMANEAT_NEAT_BUFFERPOOL_LIB)
    target_link_libraries(sima_neat PUBLIC ${SIMANEAT_NEAT_BUFFERPOOL_LIB})
  endif()
  if (SIMANEAT_NEAT_TENSORBUFFER_LIB)
    target_link_libraries(sima_neat PUBLIC ${SIMANEAT_NEAT_TENSORBUFFER_LIB})
  endif()
endif()
if (SIMANEAT_SIMAAIMEM_LIB)
  target_link_libraries(sima_neat PUBLIC ${SIMANEAT_SIMAAIMEM_LIB})
endif()
if (SIMANEAT_NEEDS_PREPARED_RUNTIME_BRIDGE)
  if (SIMANEAT_USING_NEAT_INTERNALS_PACKAGE)
    target_link_libraries(sima_neat PUBLIC NeatInternals::PreparedRuntimeBridge)
  else()
    if (SIMANEAT_NEAT_PREPARED_RUNTIME_BRIDGE_LIB)
      target_link_libraries(sima_neat PUBLIC
        ${SIMANEAT_NEAT_PREPARED_RUNTIME_BRIDGE_LIB})
    endif()
    if (SIMANEAT_NEAT_PROCESSMLA_RUNTIME_LIB)
      target_link_libraries(sima_neat PUBLIC
        ${SIMANEAT_NEAT_PROCESSMLA_RUNTIME_LIB})
    endif()
  endif()
endif()

# Per-kernel latency profiler event ring (libsimaaineatprofiler.so).  Provides
# the cross-shared-library event ring that LatencyProfiler drains and that the
# dispatcher / processcvu plugin emit into.  Optional: gracefully skipped when
# the .so cannot be found (no profiler support, but builds keep working).
set(SIMANEAT_NEAT_PROFILER_LIB
  "${SIMANEAT_NEAT_PROFILER_LIB}"
  CACHE FILEPATH
  "Path to libsimaaineatprofiler.so (per-kernel latency profiler event ring)")
if (NOT SIMANEAT_NEAT_PROFILER_LIB)
  find_library(SIMANEAT_NEAT_PROFILER_LIB
    NAMES simaaineatprofiler
    HINTS
      ${CMAKE_CURRENT_SOURCE_DIR}/../internals/build/core/profiler
      ${SIMANEAT_INSTALLED_NEAT_RUNTIME_DIR_CANDIDATES}
      /usr/lib/aarch64-linux-gnu
      /lib/aarch64-linux-gnu
      /usr/lib
      /lib
  )
endif()
if (SIMANEAT_NEAT_PROFILER_LIB)
  target_link_libraries(sima_neat PUBLIC ${SIMANEAT_NEAT_PROFILER_LIB})
  target_include_directories(sima_neat PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../internals/core/profiler>)
  target_compile_definitions(sima_neat PUBLIC SIMA_HAS_NEAT_PROFILER=1)
else()
  message(STATUS
    "libsimaaineatprofiler.so not found: per-kernel latency profiler "
    "will be unavailable. Build internals/core/profiler first to enable.")
endif()

target_compile_definitions(sima_neat PUBLIC SIMA_HAS_SIMAAI_POOL=1)

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
file(RELATIVE_PATH SIMANEAT_TEST_RUNTIME_MPK_FIXTURE_ROOT_REL
  "${CMAKE_BINARY_DIR}"
  "${CMAKE_SOURCE_DIR}/tests/assets/mpk")
file(RELATIVE_PATH SIMANEAT_TEST_RUNTIME_MPK_FIXTURE_MANIFEST_REL
  "${CMAKE_BINARY_DIR}"
  "${CMAKE_SOURCE_DIR}/tests/assets/mpk/fixtures_manifest.json")
file(RELATIVE_PATH SIMANEAT_TEST_RUNTIME_DECODER_FIXTURE_REL
  "${CMAKE_BINARY_DIR}"
  "${CMAKE_SOURCE_DIR}/tests/assets/decoder/dynamic_caps.h264")

set(SIMANEAT_TEST_RUNTIME_MANIFEST_CONTENT [=[
{
  "format_version": 1,
  "generator": "core/CMakeLists.txt",
  "source_root_rel": "@SIMANEAT_TEST_RUNTIME_SOURCE_ROOT_REL@",
  "repo_tmp_root_rel": "@SIMANEAT_TEST_RUNTIME_REPO_TMP_ROOT_REL@",
  "mpk_fixture_root_rel": "@SIMANEAT_TEST_RUNTIME_MPK_FIXTURE_ROOT_REL@",
  "mpk_fixture_manifest_rel": "@SIMANEAT_TEST_RUNTIME_MPK_FIXTURE_MANIFEST_REL@",
  "decoder_fixture_rel": "@SIMANEAT_TEST_RUNTIME_DECODER_FIXTURE_REL@"
}
]=])
string(CONFIGURE "${SIMANEAT_TEST_RUNTIME_MANIFEST_CONTENT}"
  SIMANEAT_TEST_RUNTIME_MANIFEST_CONTENT
  @ONLY)
file(WRITE "${SIMANEAT_TEST_RUNTIME_MANIFEST}" "${SIMANEAT_TEST_RUNTIME_MANIFEST_CONTENT}")

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

  set(_gst_registry "${CMAKE_BINARY_DIR}/sima_gst_registry_${test_name}.bin")
  set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "GST_REGISTRY=${_gst_registry}")
  set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "GST_REGISTRY_1_0=${_gst_registry}")
  set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "GST_REGISTRY_REUSE_PLUGIN_SCANNER=no")
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

if (SIMANEAT_NEAT_ALLOCATOR_DIR)
  target_link_options(sima_neat PUBLIC "-Wl,-rpath,${SIMANEAT_NEAT_ALLOCATOR_DIR}")
endif()
if (SIMANEAT_NEAT_BUFFERPOOL_DIR AND
    NOT SIMANEAT_NEAT_BUFFERPOOL_DIR STREQUAL SIMANEAT_NEAT_ALLOCATOR_DIR)
  target_link_options(sima_neat PUBLIC "-Wl,-rpath,${SIMANEAT_NEAT_BUFFERPOOL_DIR}")
endif()
if (SIMANEAT_NEAT_TENSORBUFFER_DIR AND
    NOT SIMANEAT_NEAT_TENSORBUFFER_DIR STREQUAL SIMANEAT_NEAT_ALLOCATOR_DIR AND
    NOT SIMANEAT_NEAT_TENSORBUFFER_DIR STREQUAL SIMANEAT_NEAT_BUFFERPOOL_DIR)
  target_link_options(sima_neat PUBLIC "-Wl,-rpath,${SIMANEAT_NEAT_TENSORBUFFER_DIR}")
endif()

# ------------------------------------------------------------------------------
# Internal-only helper target (tests/bench/tools)
# ------------------------------------------------------------------------------
add_library(sima_neat_internal INTERFACE)
target_link_libraries(sima_neat_internal INTERFACE sima_neat)
target_include_directories(sima_neat_internal INTERFACE
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
      target_include_directories(sima_neat_internal INTERFACE
        "${_simaneat_neat_internal_include_dir}")
    endif()
  endforeach()
endif()
target_compile_definitions(sima_neat_internal INTERFACE SIMA_NEAT_INTERNAL=1)

# ------------------------------------------------------------------------------
# Examples (optional, installable)
# ------------------------------------------------------------------------------
if (SIMANEAT_BUILD_SAMPLES AND
    EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
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

set(CPACK_COMPONENTS_ALL core)
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
  "libgstreamer1.0-0, libgstrtspserver-1.0-0, libglib2.0-0, libopencv-core406, neat-internals-dev, neat-gst-plugins"
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
