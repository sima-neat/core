#pragma once

#include "genai/GenAITypes.h"

#include <filesystem>
#include <string>
#include <vector>

namespace simaai::neat::genai::internal {

struct ModelDirectoryInfo {
  std::filesystem::path root;
  GenAITask task = GenAITask::VisionLanguage;
  bool accepts_text = false;
  bool accepts_image = false;
  bool accepts_audio = false;
};

ModelDirectoryInfo inspect_model_directory(const std::filesystem::path& model_dir);
std::string model_id_from_path(const std::filesystem::path& path);
std::vector<ChatMessage> build_text_messages(const GenerationRequest& request);
void validate_text_generation_request(const GenerationRequest& request);
void validate_asr_generation_request(const GenerationRequest& request);
void ensure_llima_runtime_connected();
void disconnect_llima_runtime();

} // namespace simaai::neat::genai::internal
