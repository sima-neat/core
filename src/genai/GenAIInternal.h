#pragma once

#include "genai/GenAITypes.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace simaai::llima {
class FileProvider;
}

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

struct ModelLoadContext {
  ModelDirectoryInfo info;
  std::shared_ptr<simaai::llima::FileProvider> files;
};

ModelLoadContext local_model_context(const std::filesystem::path& root);
ModelLoadContext provider_model_context(const std::filesystem::path& root,
                                        std::shared_ptr<simaai::llima::FileProvider> files);
struct ModelAccess {
  static GenAIModel create(ModelLoadContext context);
  static VisionLanguageModel vision(ModelLoadContext context);
  static ASRModel asr(ModelLoadContext context);
};

ModelDirectoryInfo inspect_model_directory(const std::filesystem::path& model_dir);
std::string model_id_from_path(const std::filesystem::path& path);
std::vector<ChatMessage> build_text_messages(const GenerationRequest& request);
void validate_text_generation_request(const GenerationRequest& request);
bool tool_calls_enabled(const GenerationRequest& request);
void validate_asr_generation_request(const GenerationRequest& request);
void ensure_llima_runtime_connected();

} // namespace simaai::neat::genai::internal
