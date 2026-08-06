#include "genai/GenAIInternal.h"

#include <sima_lmm/image_processor.hpp>
#include <sima_lmm/mla_model.hpp>
#include <sima_lmm/setup.hpp>
#include <sima_lmm/utils.hpp>

#include <spdlog/spdlog.h>

#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>

namespace simaai::neat::genai::internal {
namespace {

bool is_existing_directory(const std::filesystem::path& path) {
  std::error_code ec;
  return std::filesystem::is_directory(path, ec);
}

bool is_existing_regular_file(const std::filesystem::path& path) {
  std::error_code ec;
  return std::filesystem::is_regular_file(path, ec);
}

bool has_vision_capability(const nlohmann::json& config) {
  return config.contains("vm_cfg") && !config.at("vm_cfg").is_null() && config.contains("mm_cfg") &&
         !config.at("mm_cfg").is_null() && config.contains("vision_model_name") &&
         config.at("vision_model_name").is_string() &&
         !config.at("vision_model_name").get<std::string>().empty();
}

nlohmann::json parse_json_file(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Unable to open GenAI model config: " + path.string());
  }
  try {
    return nlohmann::json::parse(in);
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("Malformed GenAI model config " + path.string() + ": " + e.what());
  }
}

std::optional<bool> speculative_role(const std::filesystem::path& config_path) {
  const auto config = parse_json_file(config_path);
  const auto spec = config.value("lm_cfg", nlohmann::json::object())
                        .value("speculative_decoding_cfg", nlohmann::json{});
  if (spec.is_null()) {
    return std::nullopt;
  }
  return spec.value("is_draft", false);
}

std::optional<std::filesystem::path> resolve_draft_model(std::filesystem::path& model_root) {
  for (const auto& runtime_root : {model_root, model_root / "sima_files"}) {
    const auto config_path = runtime_root / "devkit" / "vlm_config.json";
    if (!is_existing_regular_file(config_path)) {
      continue;
    }
    if (speculative_role(config_path).has_value()) {
      throw std::runtime_error(
          model_root.string() +
          " is part of a speculative-decoding pair; pass its parent directory so both the "
          "target and draft models are loaded together");
    }
    model_root = runtime_root;
    return std::nullopt;
  }

  std::optional<std::filesystem::path> target;
  std::optional<std::filesystem::path> draft;
  bool target_is_speculative = false;
  for (const auto& entry : std::filesystem::directory_iterator(model_root)) {
    if (!entry.is_directory()) {
      continue;
    }
    const auto runtime_root = entry.path();
    const auto config_path = runtime_root / "devkit" / "vlm_config.json";
    if (!is_existing_regular_file(config_path)) {
      continue;
    }

    const auto role = speculative_role(config_path);
    const bool is_draft = role.value_or(false);
    auto& path = is_draft ? draft : target;
    if (path.has_value()) {
      throw std::runtime_error(std::string("Multiple ") + (is_draft ? "draft" : "target") +
                               " models found under " + model_root.string());
    }
    path = runtime_root;
    if (!is_draft) {
      target_is_speculative = role.has_value();
    }
  }

  if (!target.has_value() && !draft.has_value()) {
    return std::nullopt;
  }
  if (!target.has_value()) {
    throw std::runtime_error("Speculative-decoding package missing target model: " +
                             model_root.string());
  }
  if (target_is_speculative != draft.has_value()) {
    throw std::runtime_error("Speculative-decoding package must contain one target and one draft "
                             "model: " +
                             model_root.string());
  }
  model_root = *target;
  return draft;
}

} // namespace

ModelDirectoryInfo inspect_model_directory(const std::filesystem::path& model_dir) {
  std::error_code ec;
  const std::filesystem::path canonical = std::filesystem::weakly_canonical(model_dir, ec);
  const std::filesystem::path package_root = ec ? std::filesystem::absolute(model_dir) : canonical;

  if (!is_existing_directory(package_root)) {
    throw std::runtime_error("GenAI model directory does not exist: " + package_root.string());
  }

  auto normalized = package_root;
  const auto draft_root = resolve_draft_model(normalized);

  const auto devkit_dir = normalized / "devkit";
  if (!is_existing_directory(devkit_dir)) {
    throw std::runtime_error("GenAI model directory missing devkit/: " + normalized.string());
  }

  const auto elf_dir = normalized / "elf_files";
  if (!is_existing_directory(elf_dir)) {
    throw std::runtime_error("GenAI model directory missing elf_files/: " + normalized.string());
  }

  const auto vlm_config = devkit_dir / "vlm_config.json";
  const auto whisper_config = devkit_dir / "whisper_config.json";
  const bool has_vlm_config = is_existing_regular_file(vlm_config);
  const bool has_whisper_config = is_existing_regular_file(whisper_config);

  if (has_vlm_config == has_whisper_config) {
    throw std::runtime_error(
        has_vlm_config
            ? "GenAI model directory has both vlm_config.json and whisper_config.json: " +
                  normalized.string()
            : "GenAI model directory missing vlm_config.json or whisper_config.json: " +
                  normalized.string());
  }

  if (has_vlm_config) {
    const nlohmann::json config = parse_json_file(vlm_config);
    ModelDirectoryInfo info;
    info.package_root = package_root;
    info.root = normalized;
    info.draft_root = draft_root;
    info.task = GenAITask::VisionLanguage;
    info.accepts_text = true;
    info.accepts_image = has_vision_capability(config);
    return info;
  }

  (void)parse_json_file(whisper_config);
  ModelDirectoryInfo info;
  info.package_root = package_root;
  info.root = normalized;
  info.task = GenAITask::ASR;
  info.accepts_audio = true;
  return info;
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
