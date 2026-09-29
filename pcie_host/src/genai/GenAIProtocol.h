/**
 * @file
 * @brief The genai.* wire protocol between the host and pcie-genai-backend.
 *
 * Tag names and JSON payloads that SvcTransport sends and reads.
 * Rule: this must match the card copy (LLiMa pcie_backend/genai_protocol.hpp);
 * the tests on both sides check the same golden strings.
 * final and error carry the request id; token and metrics do not.
 */
#pragma once

#include "genai/GenAITypes.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace simaai::neat::pcie::genai::internal {

using simaai::neat::genai::GenerationRequest;

// The eight simaai_svc tags between the host and pcie-genai-backend. They must
// match the card side (LLiMa sima_lmm/devkit/pcie_backend/genai_protocol.hpp).
inline constexpr const char* kTagPrompt = "genai.prompt";   // host -> card
inline constexpr const char* kTagCancel = "genai.cancel";   // host -> card
inline constexpr const char* kTagToken = "genai.token";     // card -> host, raw text
inline constexpr const char* kTagMetrics = "genai.metrics"; // card -> host
inline constexpr const char* kTagFinal = "genai.final";     // card -> host, ends a run
inline constexpr const char* kTagError = "genai.error";     // card -> host, ends a run
inline constexpr const char* kTagChat = "genai.chat";       // host -> card, JSON
inline constexpr const char* kTagReply = "genai.reply";     // card -> host, JSON

/// JSON prompt payload. @p image_names (names relative to the card's data serve
/// root, in order) are written as the "images" list when not empty. Throws
/// std::invalid_argument for anything the card cannot do (no prompt, pixel
/// images, audio, multi-turn messages, tools).
std::string encode_prompt(const std::string& id, const GenerationRequest& request,
                          const std::vector<std::string>& image_names = {});

/// JSON cancel payload: {"id": ...}.
std::string encode_cancel(const std::string& id);

enum class ChatOp { Reset, Print };

/// JSON genai.chat payload. Reset: {"id","op":"reset",["system_prompt"],"enable_thinking"}
/// (system_prompt: nullopt = model default, "" = none). Print: {"id","op":"print"}.
std::string encode_chat(const std::string& id, ChatOp op,
                        const std::optional<std::string>& system_prompt = std::nullopt,
                        bool enable_thinking = false);

/// The card's genai.reply to a genai.chat: ok, plus text (print: the history JSON).
struct ReplyEvent {
  std::string id;
  bool ok = false;
  std::string text;
};

struct MetricEvent {
  std::string type;
  double value = 0.0;
};

// A genai.token payload from the card: raw text with a per-run sequence
// number in front ("<seq>\n<text>"). has_seq is false for an old-style
// payload that has no numeric prefix; then text is the whole payload.
struct TokenEvent {
  bool has_seq = false;
  std::uint64_t seq = 0;
  std::string text;
};

// Never throws: a payload with no numeric "<seq>\n" prefix parses as all text.
TokenEvent parse_token(std::string_view payload);

struct FinalEvent {
  std::string id;
  std::string finish_reason;
  std::uint32_t generated_tokens = 0;
  double ttft_s = 0.0;
  double tps = 0.0;
  bool history_cleared = false; // the card cleared its conversation (absent = false)
};

struct ErrorEvent {
  std::string id;
  std::string message;
  bool history_cleared = false; // the card cleared its conversation (absent = false)
};

// Each throws std::runtime_error if the payload is not a JSON object.
MetricEvent parse_metric(std::string_view payload);
FinalEvent parse_final(std::string_view payload);
ErrorEvent parse_error(std::string_view payload);
ReplyEvent parse_reply(std::string_view payload);

} // namespace simaai::neat::pcie::genai::internal
