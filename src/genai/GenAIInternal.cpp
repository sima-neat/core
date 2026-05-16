#include "genai/GenAIInternal.h"

#include <fstream>
#include <nlohmann/json.hpp>
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
  return config.contains("vm_cfg") && !config.at("vm_cfg").is_null() &&
         config.contains("mm_cfg") && !config.at("mm_cfg").is_null() &&
         config.contains("vision_model_name") && config.at("vision_model_name").is_string() &&
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
        has_vlm_config ? "GenAI model directory has both vlm_config.json and whisper_config.json: " +
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

std::vector<ChatMessage> build_text_messages(const GenerationRequest& request) {
  validate_text_generation_request(request);
  if (request.formatted_prompt.has_value()) {
    throw std::logic_error("GenerationRequest::formatted_prompt is not implemented yet");
  }

  if (!request.messages.empty()) {
    return request.messages;
  }

  std::vector<ChatMessage> messages;
  if (request.system_prompt.has_value()) {
    messages.push_back(ChatMessage{.role = "system", .content = *request.system_prompt});
  }
  messages.push_back(ChatMessage{.role = "user", .content = *request.prompt});
  return messages;
}

void validate_text_generation_request(const GenerationRequest& request) {
  const int text_source_count = (request.prompt.has_value() ? 1 : 0) +
                                (request.messages.empty() ? 0 : 1) +
                                (request.formatted_prompt.has_value() ? 1 : 0);
  if (text_source_count == 0) {
    throw std::runtime_error("GenerationRequest requires prompt, messages, or formatted_prompt");
  }
  if (text_source_count > 1) {
    throw std::runtime_error(
        "GenerationRequest accepts exactly one of prompt, messages, or formatted_prompt");
  }
}

} // namespace simaai::neat::genai::internal
