cmake_minimum_required(VERSION 3.20)
project(CoreInstalledStartupConsumer LANGUAGES CXX)
find_package(SimaNeat CONFIG REQUIRED)
find_package(SimaScheduler CONFIG REQUIRED COMPONENTS frontend)
set(SIMA_STARTUP_PLUGIN_DIR "" CACHE PATH "Installed matching Scheduler plugin directory")
if(NOT SIMA_STARTUP_PLUGIN_DIR)
  message(FATAL_ERROR "Set SIMA_STARTUP_PLUGIN_DIR to the installed cohort, not build-tree plugins")
endif()
enable_testing()
foreach(consumer core_startup_consumer core_startup_missing_runner)
  add_executable(${consumer} startup_consumer.cpp)
  target_compile_features(${consumer} PRIVATE cxx_std_20)
  target_link_libraries(${consumer} PRIVATE SimaNeat::sima_neat SimaScheduler::frontend)
endforeach()
target_sources(core_startup_missing_runner PRIVATE missing_runner.cpp)
target_link_options(core_startup_missing_runner PRIVATE -Wl,--export-dynamic)
add_test(NAME installed_core_startup COMMAND core_startup_consumer positive "${SIMA_STARTUP_PLUGIN_DIR}")
add_test(NAME installed_core_startup_missing_mla COMMAND core_startup_consumer missing-mla "${SIMA_STARTUP_PLUGIN_DIR}")
add_test(NAME installed_core_startup_missing_runner COMMAND core_startup_missing_runner missing-runner "${SIMA_STARTUP_PLUGIN_DIR}")
set_tests_properties(installed_core_startup installed_core_startup_missing_mla
  installed_core_startup_missing_runner PROPERTIES TIMEOUT 30 LABELS "strict;installed-consumer;startup")

# The copied discovery fixture retains the installed provider's dependencies.
# Set loader paths before exec: changing LD_LIBRARY_PATH inside main does not
# establish the dynamic loader's search path for the relocated plugin.
get_filename_component(_startup_neat_dir "${SIMA_STARTUP_PLUGIN_DIR}" DIRECTORY)
set_tests_properties(installed_core_startup installed_core_startup_missing_mla
  installed_core_startup_missing_runner PROPERTIES
  ENVIRONMENT "LD_LIBRARY_PATH=${SIMA_STARTUP_PLUGIN_DIR}:${_startup_neat_dir}/runtime:$ENV{LD_LIBRARY_PATH}")
