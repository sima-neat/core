cmake_minimum_required(VERSION 3.16)
project(SimaNeatInstallSmoke LANGUAGES CXX)

find_package(SimaNeat CONFIG REQUIRED)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../../deps/manifest.json"
  SIMANEAT_SMOKE_ABI_LINES REGEX "\"abi-version\"[ \t]*:"
)
list(LENGTH SIMANEAT_SMOKE_ABI_LINES SIMANEAT_SMOKE_ABI_LINE_COUNT)
if (NOT SIMANEAT_SMOKE_ABI_LINE_COUNT EQUAL 1 OR
    NOT SIMANEAT_SMOKE_ABI_LINES MATCHES "^[ \t]*\"abi-version\"[ \t]*:[ \t]*\"([1-9][0-9]*)\"[ \t]*,?[ \t]*$")
  message(FATAL_ERROR "Cannot read the expected Core ABI from deps/manifest.json")
endif()
set(SIMANEAT_SMOKE_EXPECTED_ABI "${CMAKE_MATCH_1}")
if (NOT SimaNeat_ABI_VERSION STREQUAL SIMANEAT_SMOKE_EXPECTED_ABI)
  message(FATAL_ERROR
    "Expected SimaNeat public ABI ${SIMANEAT_SMOKE_EXPECTED_ABI}, got '${SimaNeat_ABI_VERSION}'")
endif()

add_executable(install_smoke_app main.cpp)
target_link_libraries(install_smoke_app PRIVATE SimaNeat::sima_neat)
option(SIMANEAT_SMOKE_STATIC "Test the optional static component" OFF)
if (SIMANEAT_SMOKE_STATIC)
  find_package(SimaNeat CONFIG REQUIRED COMPONENTS Static)
  add_executable(install_smoke_static_app main.cpp)
  target_link_libraries(install_smoke_static_app PRIVATE SimaNeat::sima_neat_static)
endif()
