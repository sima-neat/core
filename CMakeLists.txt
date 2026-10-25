cmake_minimum_required(VERSION 3.16)

# Versioning from deps/manifest.json:
# - package-version controls package/artifact and CMake project identity.
# - platform-version records the selected platform environment.
# - modelzoo-version selects compatible Model Zoo assets and defaults to
#   platform-version when omitted.
# - abi-version controls the public C++ shared-library ABI and SONAME.
set(SIMANEAT_PLATFORM_VERSION "")
set(SIMANEAT_MODELZOO_VERSION "")
set(SIMANEAT_PACKAGE_BASE_VERSION "")
set(SIMANEAT_ABI_VERSION "")
execute_process(
  COMMAND python3 -c "import json; data=json.load(open('deps/manifest.json', encoding='utf-8')); platform=data['platform-version']; print(platform); print(data.get('modelzoo-version') or platform); print(data['package-version']); print(data['abi-version'])"
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  RESULT_VARIABLE SIMANEAT_MANIFEST_RESULT
  OUTPUT_VARIABLE SIMANEAT_MANIFEST_VERSIONS
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if (NOT SIMANEAT_MANIFEST_RESULT EQUAL 0 OR SIMANEAT_MANIFEST_VERSIONS STREQUAL "")
  message(FATAL_ERROR
    "Failed to read package-version, platform-version, modelzoo-version, and abi-version from deps/manifest.json")
endif()
string(REPLACE "\n" ";" SIMANEAT_MANIFEST_VERSION_LIST "${SIMANEAT_MANIFEST_VERSIONS}")
list(GET SIMANEAT_MANIFEST_VERSION_LIST 0 SIMANEAT_PLATFORM_VERSION)
list(GET SIMANEAT_MANIFEST_VERSION_LIST 1 SIMANEAT_MODELZOO_VERSION)
list(GET SIMANEAT_MANIFEST_VERSION_LIST 2 SIMANEAT_PACKAGE_BASE_VERSION)
list(GET SIMANEAT_MANIFEST_VERSION_LIST 3 SIMANEAT_ABI_VERSION)
if (NOT SIMANEAT_ABI_VERSION MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR
    "deps/manifest.json abi-version must be a positive integer; got '${SIMANEAT_ABI_VERSION}'")
endif()

project(SimaNeat VERSION "${SIMANEAT_PACKAGE_BASE_VERSION}" LANGUAGES CXX)

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
if(DEFINED SIMA_CORE_DEMAND_COHERENCY_V1)
  message(FATAL_ERROR "SIMA_CORE_DEMAND_COHERENCY_V1 was removed; use a fresh build without this option. Scheduler always owns prepared coherency.")
endif()
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
  "Deprecated compatibility cache option; ignored (Core requires Scheduler frontend)"
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
# Build/package versioning:
# Exact tags use the tag version; branch builds append branch/sha metadata.
# ------------------------------------------------------------------------------
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

# The existing docs-only entry point reads source headers; do not configure
# runtime libraries, test executables or installable packages without providers.
if (SIMANEAT_DOCS_ONLY_CONFIGURE)
  return()
endif()

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)
if (SIMANEAT_REQUIRE_LLIMA_ARTIFACTS)
  find_package(SimaLMM CONFIG REQUIRED)
else()
  # Core-only builds must not fail when LLiMa is absent, but if the package is
  # present we should build the real GenAI implementation instead of silently
  # shipping unavailable stubs.  This keeps local developer/CI environments
  # coherent without making LLiMa a hard dependency for normal Graph/Core builds.
  find_package(SimaLMM CONFIG QUIET)
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
# Model::benchmark() talks to the Sentinel daemon over HTTP on a Unix socket.
if (SIMANEAT_CPP_TIDY OR SIMANEAT_DOCS_ONLY_CONFIGURE)
  pkg_check_modules(HTTPLIB QUIET IMPORTED_TARGET cpp-httplib)
else()
  pkg_check_modules(HTTPLIB REQUIRED IMPORTED_TARGET cpp-httplib)
endif()

if (SIMANEAT_CPP_TIDY OR SIMANEAT_DOCS_ONLY_CONFIGURE)
  pkg_check_modules(GST QUIET
    gstreamer-1.0
    gstreamer-allocators-1.0
    gstreamer-app-1.0
    gstreamer-video-1.0
    gstreamer-allocators-1.0
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
    gstreamer-allocators-1.0
    gstreamer-app-1.0
    gstreamer-video-1.0
    gstreamer-allocators-1.0
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

# The canonical Core implementation consumes only Scheduler's compiled client
# facade. Producer-owned runtime/plugin directories are for packaging/tests;
# no sibling/SDK lower-header or ad hoc library fallback is admitted.
find_package(SimaScheduler CONFIG REQUIRED COMPONENTS frontend)
include(cmake/SchedulerRenderCapability.cmake)
include(cmake/SchedulerInstallRpath.cmake)
simaneat_scheduler_install_rpath(SIMANEAT_SCHEDULER_INSTALL_RPATH)
set(SIMANEAT_EXPORTED_NEAT_RUNTIME_DIR_CANDIDATES "${SimaScheduler_RUNTIME_DIR}")
set(SIMANEAT_NEAT_ALLOCATOR_DIR "${SimaScheduler_PLUGIN_DIR}")
set(SIMANEAT_NEAT_BUFFERPOOL_DIR "${SimaScheduler_PLUGIN_DIR}")
set(SIMANEAT_NEAT_TENSORBUFFER_DIR "${SimaScheduler_PLUGIN_DIR}")

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
if (NOT SIMANEAT_PLATFORM_VERSION MATCHES "^2[.]1[.]")
  target_compile_definitions(sima_neat_objects PRIVATE SIMANEAT_DIRECT_DRIVER_PROFILE=1)
endif()


if (NOT GIT_HASH)
  set(GIT_HASH "unknown")
endif()

add_library(sima_neat_common INTERFACE)
target_compile_features(sima_neat_common INTERFACE cxx_std_20)
find_package(nlohmann_json CONFIG REQUIRED)
include(cmake/CheckLLiMaClient.cmake)
target_link_libraries(sima_neat_common INTERFACE nlohmann_json::nlohmann_json)

add_library(sima_neat SHARED
  $<TARGET_OBJECTS:sima_neat_objects>
)
set_target_properties(sima_neat PROPERTIES
  OUTPUT_NAME sima_neat
  # VERSION identifies the implementation/platform build. SOVERSION is the
  # independently versioned public C++ ABI that consumers record in DT_NEEDED.
  VERSION "${SIMANEAT_PLATFORM_VERSION}"
  SOVERSION "${SIMANEAT_ABI_VERSION}"
  # Core and its selected frontend may use different install libdirs. Keep the
  # composed prefix relocatable without advertising plugin paths globally.
  INSTALL_RPATH "${SIMANEAT_SCHEDULER_INSTALL_RPATH}"
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
foreach(_target sima_neat_objects sima_neat sima_neat_static)
  target_link_libraries(${_target} PRIVATE
    SimaScheduler::frontend)
endforeach()
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
    # Transport ABI headers forward to the matched Runtime package; Core owns
    # only the graph-level projections, never another ABI definition.
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

# Resolve public dependency paths on the consuming machine.
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

if (OPENCV_FOUND)
  target_link_directories(sima_neat PRIVATE ${OPENCV_LIBRARY_DIRS})
  target_link_directories(sima_neat_static PRIVATE ${OPENCV_LIBRARY_DIRS})
endif()
if (GST_FOUND)
  target_link_directories(sima_neat PRIVATE ${GST_LIBRARY_DIRS})
  target_link_directories(sima_neat_static PRIVATE ${GST_LIBRARY_DIRS})
endif()

if (SIMANEAT_CPP_TIDY OR SIMANEAT_DOCS_ONLY_CONFIGURE)
  pkg_check_modules(YAML QUIET IMPORTED_TARGET yaml-cpp)
else()
  pkg_check_modules(YAML REQUIRED IMPORTED_TARGET yaml-cpp)
endif()
set(SIMANEAT_PRIVATE_LIBRARIES ZLIB::ZLIB)
if (YAML_FOUND)
  target_link_libraries(sima_neat_objects PRIVATE PkgConfig::YAML)
  list(APPEND SIMANEAT_PRIVATE_LIBRARIES PkgConfig::YAML)
endif()
find_package(ZLIB REQUIRED)
if (HTTPLIB_FOUND)
  list(APPEND SIMANEAT_PRIVATE_LIBRARIES PkgConfig::HTTPLIB)
endif()
if (SimaLMM_FOUND)
  list(APPEND SIMANEAT_PRIVATE_LIBRARIES SimaLMM::sima_lmm_runtime)
endif()

if (GST_FOUND)
  target_link_libraries(sima_neat_common INTERFACE ${GST_LIBRARIES})
endif()
if (OPENCV_FOUND)
  target_link_libraries(sima_neat_common INTERFACE opencv_core opencv_imgproc opencv_imgcodecs)
endif()

if (SIMA_WITH_OPENCV)
  target_compile_definitions(sima_neat_common INTERFACE SIMA_WITH_OPENCV)
endif()

foreach(_target sima_neat_objects sima_neat sima_neat_static)
  target_link_libraries(${_target} PRIVATE ${SIMANEAT_PRIVATE_LIBRARIES})
endforeach()

target_compile_definitions(sima_neat_common INTERFACE SIMA_HAS_SIMAAI_POOL=1)

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
# Internal contracts expose canonical preparation types. Propagate the installed
# facade's usage requirements; public Core headers remain dependency-isolated.
target_link_libraries(sima_neat_internal_headers INTERFACE
  ${SIMANEAT_PRIVATE_LIBRARIES} SimaScheduler::frontend
  nlohmann_json::nlohmann_json)
target_include_directories(sima_neat_internal_headers INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/pipeline
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/pipeline
)
target_compile_definitions(sima_neat_internal_headers INTERFACE SIMA_NEAT_INTERNAL=1)

add_library(sima_neat_internal INTERFACE)
target_link_libraries(sima_neat_internal INTERFACE
  sima_neat
  sima_neat_internal_headers
)

# Phase-1 offline structural admission tool. It shares the exact pure plan
# compiler used by ModelPack and never opens an accelerator device.
add_executable(neat-dmabuf-plan-audit
  tools/dmabuf_plan_audit.cpp
)
target_link_libraries(neat-dmabuf-plan-audit PRIVATE
  sima_neat
  sima_neat_internal_headers
)
install(
  TARGETS neat-dmabuf-plan-audit
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT core
)

# ------------------------------------------------------------------------------
# Card-side PCIe runtime helpers
# ------------------------------------------------------------------------------
if (SIMANEAT_REQUIRE_LLIMA_ARTIFACTS)
  find_path(SIMANEAT_SVC_INCLUDE_DIR simaai_svc.h PATH_SUFFIXES simaai REQUIRED)
  add_executable(neat-pcie-genai-worker
    tools/pcie_genai_worker.cpp
    shared/pcie_genai/Service.cpp
  )
  target_link_libraries(neat-pcie-genai-worker PRIVATE sima_neat_internal ${CMAKE_DL_LIBS})
  target_include_directories(neat-pcie-genai-worker PRIVATE
    ${SIMANEAT_SVC_INCLUDE_DIR} shared/pcie_genai tools)
  target_compile_definitions(neat-pcie-genai-worker PRIVATE SIMA_WITH_OPENCV=1)
  add_dependencies(sima_neat_libraries neat-pcie-genai-worker)
  install(TARGETS neat-pcie-genai-worker RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT core)
endif()
add_executable(pcie-pipeline-builder
  tools/pcie_pipeline_builder.cpp
)
target_link_libraries(pcie-pipeline-builder PRIVATE
  sima_neat_internal
  ${GST_LIBRARIES}
)
target_include_directories(pcie-pipeline-builder PRIVATE
  ${GST_INCLUDE_DIRS}
)
target_compile_options(pcie-pipeline-builder PRIVATE
  ${GST_CFLAGS_OTHER}
)
add_dependencies(sima_neat_libraries pcie-pipeline-builder)
install(TARGETS pcie-pipeline-builder
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT core
)

# ------------------------------------------------------------------------------
# Tutorials
# ------------------------------------------------------------------------------
include(cmake/NeatGuardedTests.cmake)
if (SIMANEAT_BUILD_TUTORIALS)
  add_subdirectory(tutorials)
endif()

# ------------------------------------------------------------------------------
# Tests (internal only)
# ------------------------------------------------------------------------------
if (SIMANEAT_BUILD_TESTS)
  find_package(SimaScheduler CONFIG REQUIRED COMPONENTS ClientTestSupport)
  target_link_libraries(sima_neat_internal_headers INTERFACE SimaScheduler::ClientTestSupport)
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
  TARGETS sima_neat sima_neat_common
  EXPORT SimaNeatTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT dev
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT core
    NAMELINK_COMPONENT dev
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT core
)

install(TARGETS sima_neat_static
  EXPORT SimaNeatStaticTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT dev
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
  PATTERN "SimaPluginStaticManifestAbi.h" EXCLUDE
  PATTERN "SimaTensorSetMetaAbi.h" EXCLUDE
  # Scheduler owns this installed path; never restore a Core implementation.
  PATTERN "SimaaiGstErrors.h" EXCLUDE
  PATTERN "ev_tensor_abi.h" EXCLUDE
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
  VERSION ${SIMANEAT_PACKAGE_BASE_VERSION}
  COMPATIBILITY SameMajorVersion
)

foreach(_export SimaNeatTargets SimaNeatStaticTargets)
  export(EXPORT ${_export}
    FILE ${CMAKE_CURRENT_BINARY_DIR}/${_export}.cmake
    NAMESPACE SimaNeat::)
  install(EXPORT ${_export}
    FILE ${_export}.cmake
    NAMESPACE SimaNeat::
    DESTINATION ${SimaNeat_INSTALL_CMAKEDIR}
    COMPONENT dev)
endforeach()

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

# Source-first preparation verifies the complete selected DEBs and supplies
# their actual control versions. Never derive dependency pins from Git hashes.
# Direct CMake builds retain their existing installed-package dependency names.
set(_simaneat_runtime_dependency "simaai-scheduler")
if (SIMANEAT_SCHEDULER_DEBIAN_VERSION)
  set(_simaneat_runtime_dependency "simaai-scheduler (= ${SIMANEAT_SCHEDULER_DEBIAN_VERSION})")
endif()
set(SIMANEAT_CORE_DEBIAN_DEPENDS
  "libgstreamer1.0-0"
  "libgstreamer-plugins-base1.0-0"
  "libgstrtspserver-1.0-0"
  "libglib2.0-0"
  "libopencv-core410"
  "libopencv-imgproc410"
  "libopencv-imgcodecs410"
  # libsima_neat.so links libz directly to inflate model archives in process.
  "zlib1g"
  "${_simaneat_runtime_dependency}"
  "libyaml-cpp0.8"
  "libcpp-httplib0.18"
)
if (SimaLMM_FOUND)
  list(APPEND SIMANEAT_CORE_DEBIAN_DEPENDS "sima-lmm-core")
endif()
string(JOIN ", " CPACK_DEBIAN_CORE_PACKAGE_DEPENDS ${SIMANEAT_CORE_DEBIAN_DEPENDS})
set(SIMANEAT_DEV_DEBIAN_DEPENDS
  "${_simaneat_runtime_dependency}"
  "sima-neat (= ${CPACK_PACKAGE_VERSION})"
  "pkg-config"
  "libgstreamer1.0-dev"
  "libgstreamer-plugins-base1.0-dev"
  "libgstrtspserver-1.0-dev"
  "libglib2.0-dev"
  "libopencv-dev"
  "nlohmann-json3-dev"
  "libyaml-cpp-dev"
)
string(JOIN ", " CPACK_DEBIAN_DEV_PACKAGE_DEPENDS ${SIMANEAT_DEV_DEBIAN_DEPENDS})
set(CPACK_INSTALL_CMAKE_PROJECTS
  "${CMAKE_BINARY_DIR};SimaNeat;ALL;/"
)

include(CPack)
