cmake_minimum_required(VERSION 3.16)

include(FetchContent)

# nanobind expects CMake's unified FindPython package/variables
# (Python::Module + Python_INCLUDE_DIRS).
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG QUIET)

if (NOT nanobind_FOUND)
  FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.4.0
    GIT_SHALLOW TRUE
    GIT_SUBMODULES_RECURSE TRUE
  )
  FetchContent_MakeAvailable(nanobind)
endif()

set(PYNEAT_BINDING_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/module.cpp
)

nanobind_add_module(_pyneat_core ${PYNEAT_BINDING_SOURCES})
# Python bindings include public node headers whose layouts contain internal-only
# fields when the core library is built with SIMA_NEAT_INTERNAL. Link through the
# internal interface target so _pyneat_core sees the same compile definitions and
# include paths as sima_neat, avoiding C++ ABI/layout mismatches.
target_link_libraries(_pyneat_core PRIVATE sima_neat_internal)
target_include_directories(_pyneat_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
if (DEFINED SIMANEAT_VERSION)
  target_compile_definitions(_pyneat_core PRIVATE PYNEAT_VERSION="${SIMANEAT_VERSION}")
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  if (TARGET nanobind-static)
    # Upstream nanobind uses a GNU/Clang extension that trips this project's
    # global -Wpedantic/-Werror policy.
    target_compile_options(nanobind-static PRIVATE -Wno-pedantic -Wno-error=pedantic)
  endif()
  if (TARGET nanobind)
    target_compile_options(nanobind PRIVATE -Wno-pedantic -Wno-error=pedantic)
  endif()
  target_compile_options(_pyneat_core PRIVATE -Wno-pedantic -Wno-error=pedantic)
endif()

# Cross-builds may need an explicit extension suffix so target Python can import
# the module even when host Python drives configuration.
if (DEFINED PYNEAT_EXT_SUFFIX AND NOT PYNEAT_EXT_SUFFIX STREQUAL "")
  set_target_properties(_pyneat_core PROPERTIES SUFFIX "${PYNEAT_EXT_SUFFIX}")
endif()

if (DEFINED SKBUILD_PLATLIB_DIR)
  set(PYNEAT_INSTALL_DIR "${SKBUILD_PLATLIB_DIR}/pyneat")
else()
  set(PYNEAT_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/python3/dist-packages/pyneat")
endif()

install(
  TARGETS _pyneat_core
  LIBRARY DESTINATION ${PYNEAT_INSTALL_DIR}
  COMPONENT core
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/pyneat/
  DESTINATION ${PYNEAT_INSTALL_DIR}
  COMPONENT core
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "py.typed"
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/
  DESTINATION share/sima-neat/python/tests
  COMPONENT extras
  FILES_MATCHING
  PATTERN "*.py"
)
