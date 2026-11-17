# JSON-bearing LLiMa methods expose their actual JSON C++ ABI. Check the real
# installed headers and DSO, rather than assuming a broad package minimum works.
if(SimaLMM_FOUND)
  include(CheckCXXSourceCompiles)
  include(CMakePushCheckState)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_INCLUDES ${OPENCV_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES nlohmann_json::nlohmann_json SimaLMM::sima_lmm_runtime)
  set(_neat_try_compile_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
  set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
  # Re-evaluate after a dependency/header change in an existing build directory.
  unset(SIMANEAT_LLIMA_CLIENT_LINKS CACHE)
  check_cxx_source_compiles([=[
#include <nlohmann/json.hpp>
#include <sima_lmm/chat.hpp>
// This executable is linked, never run. Referencing both methods from main
// keeps --gc-sections/LTO from discarding the ABI check as unused code.
int main(int argc, char** argv) {
  if (argc > 1) {
    auto& chat = *reinterpret_cast<simaai::llima::Chat*>(argv[1]);
    chat.set_messages(nlohmann::ordered_json::array());
    chat.set_tools(nlohmann::ordered_json::array());
  }
  return 0;
}
]=] SIMANEAT_LLIMA_CLIENT_LINKS)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_neat_try_compile_type}")
  unset(_neat_try_compile_type)
  cmake_pop_check_state()
  if(NOT SIMANEAT_LLIMA_CLIENT_LINKS)
    message(FATAL_ERROR
      "LLiMa client header/link check failed. Use the JSON development package "
      "matching the selected LLiMa C++ ABI and complete its runtime dependencies; "
      "see CMakeConfigureLog.yaml. No GenAI stub fallback is allowed for a found provider.")
  endif()
endif()
