# Single registration adapter for tests and tutorials. This only delegates to
# the existing persistent supervisor; it never creates hardware admission.
include_guard(GLOBAL)
include(CMakeParseArguments)
option(SIMANEAT_GUARDED_HARDWARE_TESTS
  "Run native test targets through a persistent admitted hardware supervisor" OFF)
set(SIMANEAT_HARDWARE_SUPERVISOR_SOURCE
  "${CMAKE_CURRENT_LIST_DIR}/../tools/hardware_test_supervisor.py")
set(SIMANEAT_GUARDED_TESTS_MODULE "${CMAKE_CURRENT_LIST_FILE}")
if (SIMANEAT_GUARDED_HARDWARE_TESTS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

function(simaneat_add_test)
  # PARSE_ARGV preserves semicolons in shell script arguments.
  cmake_parse_arguments(PARSE_ARGV 0 GUARD "NATIVE_COMMAND" "NAME" "COMMAND")
  list(GET GUARD_COMMAND 0 executable)
  if (SIMANEAT_GUARDED_HARDWARE_TESTS AND (TARGET "${executable}" OR GUARD_NATIVE_COMMAND))
    if (TARGET "${executable}")
      list(REMOVE_AT GUARD_COMMAND 0)
      list(PREPEND GUARD_COMMAND "$<TARGET_FILE:${executable}>")
    endif()
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tools")
    configure_file("${SIMANEAT_HARDWARE_SUPERVISOR_SOURCE}"
      "${CMAKE_CURRENT_BINARY_DIR}/tools/hardware_test_supervisor.py" COPYONLY)
    add_test(NAME "${GUARD_NAME}" COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_BINARY_DIR}/tools/hardware_test_supervisor.py" ctest-proxy
      --test-name "${GUARD_NAME}" -- ${GUARD_COMMAND})
  else()
    add_test(NAME "${GUARD_NAME}" COMMAND ${GUARD_COMMAND})
  endif()
endfunction()
