cmake_minimum_required(VERSION 3.16)
project(SimaNeatInstallSmoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SimaNeat REQUIRED)
if (NOT SimaNeat_ABI_VERSION STREQUAL "5")
  message(FATAL_ERROR
    "Expected SimaNeat public ABI 5, got '${SimaNeat_ABI_VERSION}'")
endif()

add_executable(install_smoke_app main.cpp)
target_link_libraries(install_smoke_app PRIVATE SimaNeat::sima_neat)

# The static target carries its dependencies in its link interface instead of absorbing them, so it
# is the one that exposes an unresolvable imported target or a build-host absolute path in the
# exported config. Linking only the shared target leaves both failures untested.
add_executable(install_smoke_static_app main.cpp)
target_link_libraries(install_smoke_static_app PRIVATE SimaNeat::sima_neat_static)
