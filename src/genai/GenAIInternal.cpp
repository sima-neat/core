#include "genai/GenAIInternal.h"

#include <sima_lmm/image_processor.hpp>
#include <sima_lmm/file_provider.hpp>
#include <sima_lmm/mla_model.hpp>
#include <sima_lmm/setup.hpp>
#include <sima_lmm/utils.hpp>

#include <spdlog/spdlog.h>

#include <mutex>

namespace simaai::neat::genai::internal {
ModelLoadContext provider_model_context(const std::filesystem::path& root,
                                        std::shared_ptr<simaai::llima::FileProvider> files,
                                        std::shared_ptr<simaai::llima::FileProvider> draft_files) {
  if (!files)
    throw std::invalid_argument("GenAI asset provider is required");
  const bool vlm = files->exists("devkit/vlm_config.json");
  const bool asr = files->exists("devkit/whisper_config.json");
  if (vlm == asr)
    throw std::runtime_error(
        "Model must contain exactly one devkit/vlm_config.json or devkit/whisper_config.json");
  const auto config = nlohmann::json::parse(
      *files->open_stream(vlm ? "devkit/vlm_config.json" : "devkit/whisper_config.json"));
  const auto role = vlm ? speculative_role(config) : std::nullopt;
  if (role.has_value() != bool(draft_files) || role.value_or(false))
    throw std::invalid_argument(
        "Speculative-decoding package must contain one target and one draft model");
  ModelDirectoryInfo info;
  info.package_root = info.root = root;
  if (draft_files) {
    if (!draft_files->exists("devkit/vlm_config.json") ||
        draft_files->exists("devkit/whisper_config.json") ||
        speculative_role(nlohmann::json::parse(
            *draft_files->open_stream("devkit/vlm_config.json"))) != std::optional<bool>(true))
      throw std::invalid_argument("Speculative-decoding draft must declare is_draft=true");
    info.draft_root = draft_files->reserve("devkit/vlm_config.json").parent_path().parent_path();
  }
  info.task = vlm ? GenAITask::VisionLanguage : GenAITask::ASR;
  info.accepts_text = vlm;
  info.accepts_image = vlm && has_vision_capability(config);
  info.accepts_audio = asr;
  return {std::move(info), std::move(files), std::move(draft_files)};
}

ModelLoadContext local_model_context(const std::filesystem::path& root) {
  auto info = inspect_model_directory(root);
  auto files = std::make_shared<simaai::llima::DiskFileProvider>(info.root);
  auto draft_files = info.draft_root
                         ? std::make_shared<simaai::llima::DiskFileProvider>(*info.draft_root)
                         : nullptr;
  return {std::move(info), std::move(files), std::move(draft_files)};
}

std::string model_id_from_path(const std::filesystem::path& path) {
  const auto name = path.filename().string();
  return name.empty() ? path.string() : name;
}

void ensure_llima_runtime_connected() {
  static std::once_flag once;
  std::call_once(once, [] {
    simaai::llima::set_log_level(spdlog::level::warn);
    simaai::llima::connect_mla_rt({});
    simaai::llima::MLAModelWithBuffer::read_env_vars();
    simaai::llima::ImageProcessor::read_env_vars();
    simaai::llima::initialize_default_sample_files();
  });
}

} // namespace simaai::neat::genai::internal
