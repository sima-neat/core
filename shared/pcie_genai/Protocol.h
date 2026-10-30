#pragma once
#include "genai/GenAIValueTypes.h"
#include <chrono>
#include <filesystem>
#include <optional>
#include <stdexcept>

namespace simaai::neat::pcie::genai::wire {
using Json = simaai::neat::genai::Json;
// Revision 3 carries reasoning_effort; older workers must not silently ignore it.
inline constexpr int version = 3;
inline constexpr std::size_t event_window = 32;
inline constexpr std::size_t max_message_bytes = 256 * 1024;
inline constexpr auto heartbeat_interval = std::chrono::seconds(1);
inline constexpr auto request_retry_interval = std::chrono::milliseconds(500);
inline constexpr auto worker_response_timeout = std::chrono::seconds(15);
inline constexpr auto worker_lease_timeout = std::chrono::seconds(30);
inline constexpr auto event_ack_timeout = std::chrono::seconds(30);
inline constexpr auto event_resend_interval = std::chrono::milliseconds(200);
inline constexpr int receive_poll_timeout_ms = 100;
// Preserve the direct request default when omitted. This is wire decoding only;
// the direct model validator remains the authority for supported effort values.
inline void decode_reasoning_effort(const Json& request, std::string& effort) {
  if (request.contains("reasoning_effort"))
    effort = request.at("reasoning_effort").get<std::string>();
}
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
inline void validate_received_asset(const std::filesystem::path& path, uint64_t expected_bytes,
                                    const std::string& name) {
  if (!std::filesystem::exists(path))
    throw std::runtime_error("Transferred model asset missing at expected path: " + path.string());
  if (!std::filesystem::is_regular_file(path))
    throw std::runtime_error("Transferred model asset is not a regular file: " + path.string());
  const auto actual_bytes = std::filesystem::file_size(path);
  if (actual_bytes != expected_bytes)
    throw std::runtime_error("Model asset transfer size mismatch: " + name + " (expected " +
                             std::to_string(expected_bytes) + ", received " +
                             std::to_string(actual_bytes) + " bytes at " + path.string() + ")");
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
