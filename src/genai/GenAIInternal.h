#pragma once

#include "genai/GenAITypes.h"
#include "genai/ModelDirectory.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace simaai::llima {
class FileProvider;
}

namespace simaai::neat::genai::internal {

struct ModelLoadContext {
  ModelDirectoryInfo info;
  std::shared_ptr<simaai::llima::FileProvider> files;
  std::shared_ptr<simaai::llima::FileProvider> draft_files;
};

ModelLoadContext local_model_context(const std::filesystem::path& root);
ModelLoadContext
provider_model_context(const std::filesystem::path& root,
                       std::shared_ptr<simaai::llima::FileProvider> files,
                       std::shared_ptr<simaai::llima::FileProvider> draft_files = {});
struct ModelAccess {
  static GenAIModel create(ModelLoadContext context);
  static VisionLanguageModel vision(ModelLoadContext context);
  static ASRModel asr(ModelLoadContext context);
};

std::string model_id_from_path(const std::filesystem::path& path);
std::vector<ChatMessage> build_text_messages(const GenerationRequest& request);
void validate_text_generation_request(const GenerationRequest& request);
bool tool_calls_enabled(const GenerationRequest& request);
bool valid_reasoning_effort(const std::string& effort);
void validate_asr_generation_request(const GenerationRequest& request);
void ensure_llima_runtime_connected();

} // namespace simaai::neat::genai::internal
