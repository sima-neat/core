#pragma once
#include "genai/GenAIValueTypes.h"
#include <filesystem>
#include <optional>

namespace simaai::neat::genai::internal {
struct ModelDirectoryInfo {
  std::filesystem::path package_root;
  std::filesystem::path root;
  std::optional<std::filesystem::path> draft_root;
  GenAITask task = GenAITask::VisionLanguage;
  bool accepts_text = false;
  bool accepts_image = false;
  bool accepts_audio = false;
};
ModelDirectoryInfo inspect_model_directory(const std::filesystem::path& model_dir);
bool has_vision_capability(const nlohmann::json& config);
std::optional<bool> speculative_role(const nlohmann::json& config);
} // namespace simaai::neat::genai::internal
