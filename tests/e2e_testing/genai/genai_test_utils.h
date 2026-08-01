#pragma once

#include "test_utils.h"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

namespace simaai::neat::test {

inline constexpr const char* kLlimaModelsPathEnv = "LLIMA_MODELS_PATH";
inline constexpr const char* kDefaultLlimaModelsPath = "/media/nvme/llima/models";
inline constexpr const char* kDefaultTextModelName = "Qwen2.5-0.5B-Instruct-GPTQ-a16w4";
inline constexpr const char* kDefaultVlmModelName = "LFM2.5-VL-450M-a16w4";
inline constexpr const char* kDefaultAsrModelName = "whisper-small-a16w8";

inline std::string trim_env_value(const char* value) {
  if (value == nullptr) {
    return {};
  }

  std::string out(value);
  const auto first = out.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = out.find_last_not_of(" \t\r\n");
  return out.substr(first, last - first + 1);
}

inline std::filesystem::path resolve_genai_model_dir(const char* model_env,
                                                     const char* default_model_name,
                                                     const char* model_kind,
                                                     const char* expected_config) {
  std::string model_name = trim_env_value(std::getenv(model_env));
  if (model_name.empty()) {
    model_name = default_model_name;
  }
  if (model_name.empty()) {
    skip_long_test_exception(std::string("set ") + model_env + " to a " + model_kind +
                             " model directory name");
  }

  if (model_name[0] == '/' || model_name.find('/') != std::string::npos ||
      model_name.find("..") != std::string::npos) {
    skip_long_test_exception(std::string(model_env) +
                             " must be a model directory name under LLIMA_MODELS_PATH, not a "
                             "path or Hugging Face repo id: " +
                             model_name);
  }

  std::string model_root = trim_env_value(std::getenv(kLlimaModelsPathEnv));
  if (model_root.empty()) {
    model_root = kDefaultLlimaModelsPath;
  }
  const std::filesystem::path model_dir = std::filesystem::path(model_root) / model_name;
  std::error_code ec;
  if (!std::filesystem::is_regular_file(model_dir / expected_config, ec) || ec) {
    skip_long_test_exception(std::string(model_env) + " resolves to a missing or invalid " +
                             model_kind + " model directory: " + model_dir.string() +
                             " (expected " + expected_config + ")");
  }
  return model_dir;
}

} // namespace simaai::neat::test
