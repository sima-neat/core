cmake_minimum_required(VERSION 3.16)

include(FetchContent)

if (DEFINED Python_INCLUDE_DIR AND NOT Python_INCLUDE_DIR STREQUAL "")
  find_package(Python 3.8 REQUIRED COMPONENTS Interpreter)
  set(Python_INCLUDE_DIRS "${Python_INCLUDE_DIR}")
  if (NOT TARGET Python::Module)
    add_library(Python::Module INTERFACE IMPORTED)
    set_target_properties(Python::Module PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${Python_INCLUDE_DIR}")
  endif()
else()
  find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
endif()

find_package(nanobind CONFIG QUIET)

if (NOT nanobind_FOUND)
  FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.5.0
    GIT_SHALLOW TRUE
    GIT_SUBMODULES_RECURSE TRUE
  )
  FetchContent_MakeAvailable(nanobind)
endif()

nanobind_add_module(_pyneatpcie_core
  ${CMAKE_CURRENT_SOURCE_DIR}/src/module.cpp
)

target_link_libraries(_pyneatpcie_core PRIVATE sima_neat_pcie_host)
target_include_directories(_pyneatpcie_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set(SIMAPCIE_PYTHON_RPATH_DIRS
  "$ORIGIN"
  "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/sima-pcie-host/gst-plugins"
  "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/gstreamer-1.0"
)
set_target_properties(_pyneatpcie_core PROPERTIES
  BUILD_RPATH "${SIMAPCIE_PYTHON_RPATH_DIRS}"
  INSTALL_RPATH "${SIMAPCIE_PYTHON_RPATH_DIRS}"
)

target_compile_definitions(_pyneatpcie_core PRIVATE
  PYNEATPCIE_VERSION="${PROJECT_VERSION}"
)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  if (TARGET nanobind-static)
    target_compile_options(nanobind-static PRIVATE -Wno-pedantic -Wno-error=pedantic)
  endif()
  if (TARGET nanobind)
    target_compile_options(nanobind PRIVATE -Wno-pedantic -Wno-error=pedantic)
  endif()
  target_compile_options(_pyneatpcie_core PRIVATE -Wno-pedantic -Wno-error=pedantic)
endif()

if (DEFINED SIMAPCIEHOST_EXT_SUFFIX AND NOT SIMAPCIEHOST_EXT_SUFFIX STREQUAL "")
  set_target_properties(_pyneatpcie_core PROPERTIES SUFFIX "${SIMAPCIEHOST_EXT_SUFFIX}")
endif()

install(
  TARGETS _pyneatpcie_core
  LIBRARY DESTINATION pyneatpcie
  COMPONENT PcieHostPython
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/pyneatpcie/
  DESTINATION pyneatpcie
  COMPONENT PcieHostPython
  FILES_MATCHING
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.py"
  PATTERN "py.typed"
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/
  DESTINATION tests
  COMPONENT PcieHostPython
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)
