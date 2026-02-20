# Central sanitizer configuration for SimaNeat.
# Use via:
#   option(SIMA_ENABLE_ASAN ...)
#   option(SIMA_ENABLE_UBSAN ...)
#   option(SIMA_ENABLE_TSAN ...)
#   include(cmake/Sanitizers.cmake)
#   simaneat_configure_sanitizers()

function(simaneat_configure_sanitizers)
  if (NOT (SIMA_ENABLE_ASAN OR SIMA_ENABLE_UBSAN OR SIMA_ENABLE_TSAN))
    return()
  endif()

  if (MSVC)
    message(FATAL_ERROR "Sanitizers are not configured for MSVC in this project.")
  endif()

  if (SIMA_ENABLE_TSAN AND SIMA_ENABLE_ASAN)
    message(FATAL_ERROR "SIMA_ENABLE_TSAN cannot be combined with SIMA_ENABLE_ASAN.")
  endif()

  if (SIMA_ENABLE_TSAN AND SIMA_ENABLE_UBSAN)
    message(STATUS "SIMA_ENABLE_TSAN with SIMA_ENABLE_UBSAN may be noisy; prefer separate runs.")
  endif()

  if (NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR
      "Sanitizers require GNU or Clang toolchains. Current compiler: ${CMAKE_CXX_COMPILER_ID}")
  endif()

  set(SIMANEAT_SANITIZER_FLAGS "")

  if (SIMA_ENABLE_ASAN)
    list(APPEND SIMANEAT_SANITIZER_FLAGS
      -fsanitize=address
      -fno-omit-frame-pointer)
  endif()

  if (SIMA_ENABLE_UBSAN)
    list(APPEND SIMANEAT_SANITIZER_FLAGS
      -fsanitize=undefined
      -fno-sanitize-recover=all)
  endif()

  if (SIMA_ENABLE_TSAN)
    list(APPEND SIMANEAT_SANITIZER_FLAGS
      -fsanitize=thread
      -fno-omit-frame-pointer)
  endif()

  if (NOT SIMANEAT_SANITIZER_FLAGS)
    return()
  endif()

  add_compile_options(${SIMANEAT_SANITIZER_FLAGS})
  add_link_options(${SIMANEAT_SANITIZER_FLAGS})

  # Sanitizer builds should retain symbols and predictable optimization.
  add_compile_options(-g -O1)

  message(STATUS "Sanitizers enabled: ${SIMANEAT_SANITIZER_FLAGS}")
endfunction()
