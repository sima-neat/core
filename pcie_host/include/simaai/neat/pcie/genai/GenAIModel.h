/**
 * @file
 * @brief Host-side GenAI model handle that runs a text LLM on a Modalix card.
 *
 * This mirrors the on-card Core API simaai::neat::genai::GenAIModel, but the
 * work happens on a PCIe-attached card: the host never links the LLiMa/MLA
 * runtime. It reuses Core's GenAI value types (request/result/stream) so callers
 * write the same code whether the model runs locally or across PCIe.
 */
#pragma once

#include "genai/GenAITypes.h"
#include "simaai/neat/pcie/genai/ChatTypes.h"

#include <memory>
#include <optional>
#include <string>

namespace simaai::neat::pcie::genai {

// Reuse Core's GenAI vocabulary rather than redefining it.
using simaai::neat::genai::GenAITask;
using simaai::neat::genai::GenerationMetrics;
using simaai::neat::genai::GenerationRequest;
using simaai::neat::genai::GenerationResult;
using simaai::neat::genai::GenerationStream;
using simaai::neat::genai::TokenSample;

namespace internal {
class Transport;
struct GenAIModelAccess;
} // namespace internal

class GenAIModel {
public:
  ~GenAIModel();

  GenAIModel(GenAIModel&&) noexcept;
  GenAIModel& operator=(GenAIModel&&) noexcept;

  GenAIModel(const GenAIModel&) = delete;
  GenAIModel& operator=(const GenAIModel&) = delete;

  /// Task family. The current host slice serves text LLMs (VisionLanguage).
  GenAITask task() const;
  bool accepts_text() const;
  bool accepts_image() const;
  bool accepts_audio() const;
  std::string model_id() const;

  /// Run to completion and return the full text plus final metrics.
  ///
  /// Unlike the on-card GenAIModel, the card keeps the conversation:
  /// each run()/stream() adds its question and answer to a history on the card,
  /// and later requests see it, until reset_chat(). For independent requests,
  /// call reset_chat() before each one. A request whose system_prompt or
  /// enable_thinking differs from the last one also starts a new conversation.
  /// Not thread-safe: run one request (or chat call) at a time.
  GenerationResult run(const GenerationRequest& request);
  /// Stream tokens as the card produces them. Cancel the stream to stop early.
  /// Keeps the conversation on the card, like run().
  GenerationStream stream(const GenerationRequest& request);

  /// Clear the conversation on the card and set its system prompt and thinking
  /// mode. system_prompt: nullopt = the model's default, "" = no system prompt.
  ChatReply reset_chat(const std::optional<std::string>& system_prompt, bool enable_thinking);
  /// The conversation on the card as JSON, with host image paths.
  ChatReply chat_history();
  /// True if the card cleared the conversation during the last run/stream
  /// (Ctrl-C, empty answer, or an error). Read after the stream has ended.
  bool last_run_cleared_history() const;

private:
  class Impl;
  explicit GenAIModel(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;

  friend struct internal::GenAIModelAccess;
};

} // namespace simaai::neat::pcie::genai
