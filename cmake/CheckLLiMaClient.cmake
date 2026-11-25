# JSON-bearing LLiMa methods expose their actual JSON C++ ABI. Check the real
# installed headers and DSO, rather than assuming a broad package minimum works.
if(SimaLMM_FOUND)
  include(CheckCXXSourceCompiles)
  include(CMakePushCheckState)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_INCLUDES ${OPENCV_INCLUDE_DIRS})
  # Match Core's real platform include context, including the Scheduler headers.
  set(CMAKE_REQUIRED_LIBRARIES
    nlohmann_json::nlohmann_json SimaLMM::sima_lmm_runtime SimaScheduler::frontend)
  set(_neat_try_compile_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
  set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
  # Re-evaluate after a dependency/header change in an existing build directory.
  unset(SIMANEAT_LLIMA_CLIENT_LINKS CACHE)
  check_cxx_source_compiles([=[
#include <nlohmann/json.hpp>
#include <sima_lmm/chat.hpp>
#include <sima_lmm/vision_language_model.hpp>
// This executable is linked, never run. Referencing the public chat and generation methods from main
// keeps --gc-sections/LTO from discarding the ABI check as unused code.
int main(int argc, char** argv) {
  if (argc > 1) {
    auto& chat = *reinterpret_cast<simaai::llima::Chat*>(argv[1]);
    chat.set_messages(nlohmann::ordered_json::array());
    chat.set_tools(nlohmann::ordered_json::array());
    auto& model = *reinterpret_cast<simaai::llima::VisionLanguageModel*>(argv[0]);
    simaai::llima::GenerationOptions options;
    options.cache_id = "probe";
    auto result = model.generate(chat, options);
    model.remove_kv_cache("probe");
    model.clear_kv_caches();
    simaai::llima::VisionLanguageModel candidate(argv[0], simaai::llima::ModelOptions{2});
    return result.token_ids.empty() && candidate.kv_cache_count() == 0 &&
            candidate.kv_cache_bytes_per_slot() > 0;
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
      "matching the selected LLiMa C++ ABI, including named KV cache support, "
      "and complete its runtime dependencies; "
      "see CMakeConfigureLog.yaml. No GenAI stub fallback is allowed for a found provider.")
  endif()
endif()
