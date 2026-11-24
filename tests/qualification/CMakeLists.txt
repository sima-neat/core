# The qualification harness of the Dataflow backend (README.md): per catalogue package and option
# view, the Dataflow backend gives outputs byte-equal to GStreamer's or refuses at build with the one
# not-carried error. Test-side only.
#
#   neat_qualification_census   the planner's verdict per component and view (T0; CPU only)
#   neat_qualification_compare  one backend's records of one component (T1-T2; the board)
#   qualify.py                  runs both, compares, judges against expected.json
#
# ctests: qualification_selftest (the judge's own rules, no device, no package) always; with
# SIMANEAT_QUALIFICATION_CORPUS (a directory of <component id>.tar.gz, full or census-only packages:
# make_census_corpus.py) also qualification_census, the T0 census against expected.json.

if (NOT SIMANEAT_DATAFLOW_SOURCE_DIR)
  return()
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)

add_executable(neat_qualification_census ${CMAKE_CURRENT_SOURCE_DIR}/census.cpp)
target_link_libraries(neat_qualification_census PRIVATE sima_neat sima_neat_internal_headers)

# The comparator uses Core's public API only.
add_executable(neat_qualification_compare ${CMAKE_CURRENT_SOURCE_DIR}/compare.cpp)
target_link_libraries(neat_qualification_compare PRIVATE sima_neat)

if (Python3_Interpreter_FOUND)
  simaneat_add_test(NAME qualification_selftest
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qualify.py selftest)
  set_property(TEST qualification_selftest APPEND PROPERTY LABELS "unit;dataflow;qualification")

  set(SIMANEAT_QUALIFICATION_CORPUS "" CACHE PATH
    "Directory of catalogue component packages (<id>.tar.gz) for the qualification census")
  if (SIMANEAT_QUALIFICATION_CORPUS)
    simaneat_add_test(NAME qualification_census
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qualify.py census
        --tool $<TARGET_FILE:neat_qualification_census>
        --corpus ${SIMANEAT_QUALIFICATION_CORPUS}
        --out ${CMAKE_CURRENT_BINARY_DIR}/census)
    simaneat_append_ctest_env(qualification_census)
    set_tests_properties(qualification_census PROPERTIES TIMEOUT 3600)
    set_property(TEST qualification_census APPEND PROPERTY LABELS "dataflow;qualification")
  endif()
endif()
