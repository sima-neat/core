# ------------------------------------------------------------------------------
# SimaNeat examples
# ------------------------------------------------------------------------------

message(STATUS "Configuring SimaNeat examples")
set(SIMANEAT_EXAMPLE_TARGETS "")

# ------------------------------------------------------------------------------
# Blacklist (filenames only, no paths)
# ------------------------------------------------------------------------------
set(SIMANEAT_EXAMPLE_BLACKLIST
  # resnet50_debug_overloads.cpp
)

# ------------------------------------------------------------------------------
# Common sources shared by examples
# ------------------------------------------------------------------------------
set(EXAMPLE_COMMON_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/example_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/graphpipes_metadata_receiver_helpers.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/support/obj_detection_utils.cpp
)

# ------------------------------------------------------------------------------
# Collect all example .cpp files
# ------------------------------------------------------------------------------
file(GLOB EXAMPLE_SOURCES
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# ------------------------------------------------------------------------------
# Remove blacklisted files and common sources
# ------------------------------------------------------------------------------
foreach(src ${EXAMPLE_SOURCES})
  get_filename_component(fname ${src} NAME)

  # Skip blacklisted files
  if (fname IN_LIST SIMANEAT_EXAMPLE_BLACKLIST)
    message(STATUS "  [examples] Skipping blacklisted file: ${fname}")
    list(REMOVE_ITEM EXAMPLE_SOURCES ${src})
  endif()

  # Skip shared utility file (compiled separately)
  if (fname STREQUAL "example_utils.cpp")
    list(REMOVE_ITEM EXAMPLE_SOURCES ${src})
  endif()

  if (fname STREQUAL "graphpipes_metadata_receiver_helpers.cpp")
    list(REMOVE_ITEM EXAMPLE_SOURCES ${src})
  endif()

endforeach()

# ------------------------------------------------------------------------------
# Build one executable per remaining source file
# ------------------------------------------------------------------------------
foreach(src ${EXAMPLE_SOURCES})
  get_filename_component(exe ${src} NAME_WE)

  add_executable(${exe}
    ${src}
    ${EXAMPLE_COMMON_SOURCES}
  )

  target_link_libraries(${exe}
    PRIVATE sima_neat
  )

  target_include_directories(${exe}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
  )

  list(APPEND SIMANEAT_EXAMPLE_TARGETS ${exe})

  message(STATUS "  [examples] Added example: ${exe}")
endforeach()



# ------------------------------------------------------------------------------
# ANPR Demo
# ------------------------------------------------------------------------------
add_executable(anpr_demo
  ${CMAKE_CURRENT_SOURCE_DIR}/e2e_pipelines/anpr/src/main_anpr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/e2e_pipelines/anpr/src/anpr_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/example_utils.cpp
  ${CMAKE_SOURCE_DIR}/tests/e2e_pipelines/obj_detection/obj_detection_utils.cpp
)

target_link_libraries(anpr_demo PRIVATE sima_neat)

target_include_directories(anpr_demo PRIVATE
  ${CMAKE_SOURCE_DIR}/tests
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/e2e_pipelines/anpr/src
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Keep build/anpr_demo synchronized with build/examples/anpr_demo.
add_custom_command(TARGET anpr_demo POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:anpr_demo>
          ${CMAKE_BINARY_DIR}/anpr_demo
)

list(APPEND PIPELINESESSION_EXAMPLE_TARGETS anpr_demo)

install(
    TARGETS ${SIMANEAT_EXAMPLE_TARGETS}
    RUNTIME DESTINATION lib/sima-neat/examples
    COMPONENT extras
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION share/sima-neat/examples
    COMPONENT extras
    FILES_MATCHING
      PATTERN "*.cpp"
      PATTERN "*.h"
)

# Install built example binaries from the build tree (for extras packaging)
install(
    DIRECTORY ${CMAKE_BINARY_DIR}/examples/
    DESTINATION lib/sima-neat/examples
    COMPONENT extras
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING
      PATTERN "*"
)
