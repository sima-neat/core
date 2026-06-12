#include "genai/GenAIInternal.h"

#include <sima_lmm/image_processor.hpp>
#include <sima_lmm/mla_model.hpp>
#include <sima_lmm/setup.hpp>
#include <sima_lmm/utils.hpp>

#include <spdlog/spdlog.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <mutex>
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

} // namespace

ModelDirectoryInfo inspect_model_directory(const std::filesystem::path& model_dir) {
  std::error_code ec;
  const std::filesystem::path root = std::filesystem::weakly_canonical(model_dir, ec);
  const std::filesystem::path normalized = ec ? std::filesystem::absolute(model_dir) : root;

  if (!is_existing_directory(normalized)) {
    throw std::runtime_error("GenAI model directory does not exist: " + normalized.string());
  }

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
    info.root = normalized;
    info.task = GenAITask::VisionLanguage;
    info.accepts_text = true;
    info.accepts_image = has_vision_capability(config);
    return info;
  }

  (void)parse_json_file(whisper_config);
  ModelDirectoryInfo info;
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

void disconnect_llima_runtime() {
  simaai::llima::disconnect_mla_rt();
}

} // namespace simaai::neat::genai::internal
