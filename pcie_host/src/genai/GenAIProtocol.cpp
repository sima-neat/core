#include "genai/GenAIProtocol.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <stdexcept>

namespace simaai::neat::pcie::genai::internal {

namespace {

using ordered_json = nlohmann::ordered_json;

// Replace invalid UTF-8 instead of throwing, so one bad byte in user text can
// never lose the whole message.
std::string dump(const ordered_json& j) {
  return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

nlohmann::json parse_object(std::string_view payload, const char* tag) {
  nlohmann::json j = nlohmann::json::parse(payload, nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded() || !j.is_object()) {
    throw std::runtime_error(std::string("malformed ") + tag + " payload");
  }
  return j;
}

std::string string_or_empty(const nlohmann::json& j, const char* key) {
  return j.contains(key) && j[key].is_string() ? j[key].get<std::string>() : std::string();
}

double number_or_zero(const nlohmann::json& j, const char* key) {
  return j.contains(key) && j[key].is_number() ? j[key].get<double>() : 0.0;
}

bool bool_or_false(const nlohmann::json& j, const char* key) {
  return j.contains(key) && j[key].is_boolean() && j[key].get<bool>();
}

} // namespace

std::string encode_prompt(const std::string& id, const GenerationRequest& request,
                          const std::vector<std::string>& image_names) {
  if (!request.images.empty()) {
    throw std::invalid_argument(
        "PCIe GenAI does not send pixel images: set GenerationRequest.image_files to image "
        "paths instead");
  }
  if (request.audio.has_value() || request.audio_file.has_value()) {
    throw std::invalid_argument("PCIe GenAI is text/image only: audio is not supported");
  }
  if (!request.messages.empty()) {
    throw std::invalid_argument(
        "PCIe GenAI takes a single prompt for now: GenerationRequest.messages is not supported");
  }
  if (!request.tools.empty()) {
    throw std::invalid_argument("PCIe GenAI does not support tools yet");
  }
  if (!request.prompt.has_value() || request.prompt->empty()) {
    throw std::invalid_argument("PCIe GenAI needs a non-empty GenerationRequest.prompt");
  }
  ordered_json j;
  j["id"] = id;
  j["prompt"] = *request.prompt;
  if (request.system_prompt.has_value()) {
    j["system_prompt"] = *request.system_prompt;
  }
  if (!image_names.empty()) {
    j["images"] = image_names;
  }
  if (request.max_new_tokens > 0) {
    j["max_new_tokens"] = request.max_new_tokens;
  }
  j["enable_thinking"] = request.enable_thinking;
  return dump(j);
}

std::string encode_cancel(const std::string& id) {
  ordered_json j;
  j["id"] = id;
  return dump(j);
}

std::string encode_chat(const std::string& id, ChatOp op,
                        const std::optional<std::string>& system_prompt, bool enable_thinking) {
  ordered_json j;
  j["id"] = id;
  if (op == ChatOp::Print) {
    j["op"] = "print";
    return dump(j);
  }
  j["op"] = "reset";
  if (system_prompt.has_value()) {
    j["system_prompt"] = *system_prompt;
  }
  j["enable_thinking"] = enable_thinking;
  return dump(j);
}

MetricEvent parse_metric(std::string_view payload) {
  const nlohmann::json j = parse_object(payload, kTagMetrics);
  return MetricEvent{string_or_empty(j, "type"), number_or_zero(j, "value")};
}

FinalEvent parse_final(std::string_view payload) {
  const nlohmann::json j = parse_object(payload, kTagFinal);
  FinalEvent f;
  f.id = string_or_empty(j, "id");
  f.finish_reason = string_or_empty(j, "finish_reason");
  if (j.contains("generated_tokens") && j["generated_tokens"].is_number_unsigned()) {
    f.generated_tokens = j["generated_tokens"].get<std::uint32_t>();
  }
  f.ttft_s = number_or_zero(j, "ttft");
  f.tps = number_or_zero(j, "tps");
  f.history_cleared = bool_or_false(j, "history_cleared");
  return f;
}

TokenEvent parse_token(std::string_view payload) {
  TokenEvent event;
  const std::string_view::size_type nl = payload.find('\n');
  if (nl != std::string_view::npos) {
    const std::string_view prefix = payload.substr(0, nl);
    const bool numeric =
        !prefix.empty() && std::all_of(prefix.begin(), prefix.end(),
                                       [](unsigned char c) { return std::isdigit(c) != 0; });
    if (numeric) {
      std::uint64_t seq = 0;
      const auto result = std::from_chars(prefix.data(), prefix.data() + prefix.size(), seq);
      if (result.ec == std::errc()) {
        event.has_seq = true;
        event.seq = seq;
        event.text = std::string(payload.substr(nl + 1));
        return event;
      }
    }
  }
  // Old-style / no prefix: the whole payload is the token text.
  event.text = std::string(payload);
  return event;
}

ErrorEvent parse_error(std::string_view payload) {
  const nlohmann::json j = parse_object(payload, kTagError);
  return ErrorEvent{string_or_empty(j, "id"), string_or_empty(j, "message"),
                    bool_or_false(j, "history_cleared")};
}

ReplyEvent parse_reply(std::string_view payload) {
  const nlohmann::json j = parse_object(payload, kTagReply);
  return ReplyEvent{string_or_empty(j, "id"), bool_or_false(j, "ok"), string_or_empty(j, "text")};
}

} // namespace simaai::neat::pcie::genai::internal
