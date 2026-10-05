#pragma once
#include "genai/GenAIValueTypes.h"
#include <chrono>
#include <filesystem>
#include <optional>
#include <stdexcept>

namespace simaai::neat::pcie::genai::wire {
using Json = simaai::neat::genai::Json;
inline constexpr int version = 2;
inline constexpr std::size_t event_window = 32;
inline constexpr std::size_t max_message_bytes = 256 * 1024;
inline std::string relative_name(const std::string& name) {
  std::filesystem::path p(name);
  if (name.empty() || p.is_absolute() || name.find('\0') != std::string::npos ||
      name.find('\\') != std::string::npos)
    throw std::invalid_argument("Expected relative asset name");
  for (const auto& part : p)
    if (part == ".." || part == ".")
      throw std::invalid_argument("Unsafe asset name");
  return p.generic_string();
}
inline void validate_session(const std::string& session) {
  if (session.size() != 24 || session.find_first_not_of("0123456789abcdef") != std::string::npos)
    throw std::invalid_argument("Invalid GenAI session ID");
}
inline std::string tag(const std::string& session, bool reply) {
  validate_session(session);
  return "ng." + session + (reply ? ".r" : ".c");
}
inline std::string asset_tag(const std::string& session, bool reply) {
  validate_session(session);
  return "ng." + session + (reply ? ".f" : ".a");
}
inline Json envelope(const std::string& session, uint64_t request, const char* kind) {
  return {{"v", version}, {"session", session}, {"request", request}, {"kind", kind}};
}
inline Json parse(const std::string& text, const std::string& session) {
  if (text.size() > max_message_bytes)
    throw std::length_error("GenAI message limit exceeded");
  auto j = Json::parse(text);
  if (j.at("v").get<int>() != version || j.at("session").get<std::string>() != session)
    throw std::runtime_error("GenAI protocol/session mismatch");
  if (!j.at("request").is_number_unsigned())
    throw std::invalid_argument("Request ID must be an unsigned integer");
  (void)j.at("kind").get<std::string>();
  return j;
}
// One outstanding asset per provider. Callbacks allow the exchange to be tested
// without opening a daemon or a PCIe queue.
template <class Send, class Receive, class Cancelled>
std::optional<uint64_t> request_asset(const std::string& session, uint64_t id,
                                      const std::string& name, int timeout_ms, Send send,
                                      Receive receive, Cancelled cancelled) {
  using Clock = std::chrono::steady_clock;
  auto request = envelope(session, id, "asset");
  request["name"] = relative_name(name);
  const auto text = request.dump();
  const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
  auto sent = Clock::time_point{};
  while (Clock::now() < deadline) {
    if (cancelled())
      throw std::runtime_error("Model asset transfer cancelled");
    if (Clock::now() - sent >= std::chrono::seconds(1)) {
      send(text);
      sent = Clock::now();
    }
    const auto payload = receive(100);
    if (!payload)
      continue;
    const Json reply = parse(*payload, session);
    const uint64_t received = reply.at("request");
    if (received < id)
      continue;
    if (received != id || reply.at("kind") != "asset")
      throw std::runtime_error("Unexpected model asset reply");
    if (reply.contains("error"))
      throw std::runtime_error(reply.at("error").get<std::string>());
    if (!reply.at("found").get<bool>())
      return std::nullopt;
    return reply.at("bytes").get<uint64_t>();
  }
  throw std::runtime_error("Model asset transfer timed out: " + name);
}
inline Json encode(const simaai::neat::genai::TokenSample& s) {
  Json j = {{"text", s.text},
            {"reasoning", s.reasoning},
            {"final", s.is_final},
            {"finish_reason", s.finish_reason},
            {"language", s.language},
            {"tool_calls", s.tool_calls},
            {"tokens", s.metrics.generated_tokens},
            {"ttft", s.metrics.time_to_first_token_s},
            {"tps", s.metrics.tokens_per_second}};
  if (s.no_speech_prob)
    j["no_speech_prob"] = *s.no_speech_prob;
  if (s.avg_logprob)
    j["avg_logprob"] = *s.avg_logprob;
  return j;
}
inline simaai::neat::genai::TokenSample decode(const Json& j) {
  simaai::neat::genai::TokenSample s;
  s.text = j.at("text");
  s.reasoning = j.at("reasoning");
  s.is_final = j.at("final");
  s.finish_reason = j.at("finish_reason");
  s.language = j.at("language");
  s.tool_calls = j.at("tool_calls");
  s.metrics.generated_tokens = j.at("tokens");
  s.metrics.time_to_first_token_s = j.at("ttft");
  s.metrics.tokens_per_second = j.at("tps");
  if (j.contains("no_speech_prob"))
    s.no_speech_prob = j.at("no_speech_prob").get<float>();
  if (j.contains("avg_logprob"))
    s.avg_logprob = j.at("avg_logprob").get<float>();
  return s;
}
} // namespace simaai::neat::pcie::genai::wire
