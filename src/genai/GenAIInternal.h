#pragma once

#include "genai/GenAITypes.h"
#include "genai/ModelDirectory.h"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace simaai::llima {
class FileProvider;
}

namespace simaai::neat::genai::internal {

class ContextLengthExceeded : public std::invalid_argument {
public:
  ContextLengthExceeded(std::size_t prompt_tokens, std::size_t context_tokens)
      : std::invalid_argument("Prompt uses " + std::to_string(prompt_tokens) +
                              " tokens; model context is " + std::to_string(context_tokens) +
                              ". Start a new conversation or shorten the input.") {}
};

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
