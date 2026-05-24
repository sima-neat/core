#include "genai/GenAIInternal.h"

#include <stdexcept>

namespace simaai::neat::genai::internal {

std::vector<ChatMessage> build_text_messages(const GenerationRequest& request) {
  validate_text_generation_request(request);

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
  const int text_source_count =
      (request.prompt.has_value() ? 1 : 0) + (request.messages.empty() ? 0 : 1);
  if (text_source_count == 0) {
    throw std::runtime_error("GenerationRequest requires prompt or messages");
  }
  if (text_source_count > 1) {
    throw std::runtime_error("GenerationRequest accepts exactly one of prompt or messages");
  }
  if (!request.messages.empty() && request.system_prompt.has_value()) {
    throw std::runtime_error("GenerationRequest::system_prompt is valid only with prompt");
  }
  if (!request.messages.empty() && (!request.images.empty() || request.use_cached_images)) {
    throw std::runtime_error(
        "GenerationRequest top-level images are valid only with prompt; attach images to "
        "ChatMessage when using messages");
  }
  if (!request.images.empty() && request.use_cached_images) {
    throw std::runtime_error("GenerationRequest cannot combine direct images with cached images");
  }
  if (request.audio.has_value() || request.audio_file.has_value()) {
    throw std::runtime_error(
        "GenerationRequest audio fields are not valid for VisionLanguageModel");
  }

  std::size_t cached_image_uses = request.use_cached_images ? 1U : 0U;
  for (const auto& message : request.messages) {
    if (!message.images.empty() && message.use_cached_images) {
      throw std::runtime_error("ChatMessage cannot combine direct images with cached images");
    }
    if (message.use_cached_images) {
      ++cached_image_uses;
    }
  }
  if (cached_image_uses > 1U) {
    throw std::runtime_error("GenerationRequest accepts at most one cached image insertion point");
  }
}

void validate_asr_generation_request(const GenerationRequest& request) {
  const int audio_source_count =
      (request.audio.has_value() ? 1 : 0) + (request.audio_file.has_value() ? 1 : 0);
  if (audio_source_count == 0) {
    throw std::runtime_error("GenerationRequest requires audio or audio_file for ASR");
  }
  if (audio_source_count > 1) {
    throw std::runtime_error("GenerationRequest accepts exactly one of audio or audio_file");
  }
  if (request.prompt.has_value() || request.system_prompt.has_value() ||
      !request.messages.empty() || !request.tools.empty() || !request.tool_choice.is_null()) {
    throw std::runtime_error("GenerationRequest text fields are not valid for ASRModel");
  }
  if (!request.images.empty() || request.use_cached_images) {
    throw std::runtime_error("GenerationRequest image fields are not valid for ASRModel");
  }
}

} // namespace simaai::neat::genai::internal
