# Verify the actual ELF contract, not only CMake target metadata.  Workload
# priority enlarged exported C++ option structs, so an ABI-4 consumer must ask
# the dynamic loader for libsima_neat.so.4 and fail cleanly rather than bind to
# ABI 5 by accident.
if(NOT DEFINED NEAT_LIBRARY OR NOT EXISTS "${NEAT_LIBRARY}")
  message(FATAL_ERROR "NEAT_LIBRARY does not name a built shared library")
endif()
if(NOT DEFINED READELF OR NOT EXISTS "${READELF}")
  message(FATAL_ERROR "READELF is unavailable")
endif()

execute_process(
  COMMAND "${READELF}" -d "${NEAT_LIBRARY}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE dynamic
  ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "readelf failed: ${error}")
endif()
if(NOT dynamic MATCHES "SONAME[^\n]*libsima_neat\\.so\\.5")
  message(FATAL_ERROR "ABI-5 SONAME is absent:\n${dynamic}")
endif()
if(dynamic MATCHES "SONAME[^\n]*libsima_neat\\.so\\.4")
  message(FATAL_ERROR "stale ABI-4 SONAME remains:\n${dynamic}")
endif()
if(DEFINED NEAT_LINK_DIR AND
   EXISTS "${NEAT_LINK_DIR}/libsima_neat.so.4")
  message(FATAL_ERROR
    "stale libsima_neat.so.4 exists beside the ABI-5 implementation")
endif()
