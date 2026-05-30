# Phase-0 coherency verification ladder — self-contained (no sima_neat_internal
# dependency), so it builds and runs in host CI and on the Modalix devkit alike.
#
#   cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
#
# To fold into the main tree, add_subdirectory(coherency) from tests/CMakeLists.txt.
cmake_minimum_required(VERSION 3.16)
project(neat_coherency_phase0 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)

# Host-runnable suite: Levels 0,1,3,5 (logic + optimality + fuzzer).
add_executable(coherency_host_suite segment_coherency.cpp coherency_host_suite.cpp)
target_compile_options(coherency_host_suite PRIVATE -Wall -Wextra)
target_link_libraries(coherency_host_suite PRIVATE Threads::Threads)

# Negative control: same suite with the invalidate disabled — MUST fail. Proves
# the hazard/optimality tests have teeth. (ctest WILL_FAIL inverts the verdict.)
add_executable(coherency_host_suite_broken segment_coherency.cpp coherency_host_suite.cpp)
target_compile_definitions(coherency_host_suite_broken PRIVATE SEGCOH_BREAK_INVALIDATE)
target_link_libraries(coherency_host_suite_broken PRIVATE Threads::Threads)

# Devkit suite: Levels 2,4 (+ L6 differential hook) against real cacheable DDR and
# real dc civac/cvac. Skips (exit 77) off-target; compiles everywhere.
add_executable(coherency_devkit_suite segment_coherency.cpp coherency_devkit_suite.cpp)
target_link_libraries(coherency_devkit_suite PRIVATE Threads::Threads ${CMAKE_DL_LIBS})

enable_testing()
add_test(NAME coherency_host_suite COMMAND coherency_host_suite)

add_test(NAME coherency_negative_control COMMAND coherency_host_suite_broken)
set_tests_properties(coherency_negative_control PROPERTIES WILL_FAIL TRUE)

add_test(NAME coherency_devkit_suite COMMAND coherency_devkit_suite)
set_tests_properties(coherency_devkit_suite PROPERTIES SKIP_RETURN_CODE 77)
