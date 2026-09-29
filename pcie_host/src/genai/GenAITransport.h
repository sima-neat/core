#pragma once

#include "genai/GenAITypes.h"
#include "simaai/neat/pcie/genai/ChatTypes.h"

#include <functional>
#include <optional>
#include <string>

namespace simaai::neat::pcie::genai::internal {

using simaai::neat::genai::GenerationRequest;
using simaai::neat::genai::TokenSample;
using simaai::neat::pcie::genai::ChatReply;

/**
 * @brief Card-facing seam for a single GenAI model.
 *
 * The host GenAIModel owns one Transport and knows nothing about how tokens
 * reach it. The production implementation drives the card over the simaai_svc
 * data plane; tests substitute a scripted fake. Keeping this an interface is
 * what keeps LLiMa/MLA out of the host package: the model never links a runtime,
 * it only talks to a Transport.
 */
class Transport {
public:
  virtual ~Transport() = default;

  /// Model identity reported to callers (mirrors Core GenAIModel::model_id()).
  virtual std::string model_id() const = 0;

  /**
   * @brief Drive one generation to completion.
   *
   * Runs on the GenerationStream worker thread. Deliver each token through
   * @p emit; deliver exactly one sample with is_final=true when generation ends.
   * Stop early when @p is_cancelled returns true. Implementations must not throw
   * across this boundary for expected end conditions.
   */
  virtual void generate(const GenerationRequest& request, const std::function<bool()>& is_cancelled,
                        const std::function<void(const TokenSample&)>& emit) = 0;

  /// Asynchronous cancel signal raised when the caller cancels the stream.
  virtual void cancel() = 0;

  /// Clear the card's conversation and set its system prompt and thinking mode
  /// (system_prompt: nullopt = model default, "" = none).
  virtual ChatReply reset_chat(const std::optional<std::string>& system_prompt,
                               bool enable_thinking) {
    (void)system_prompt;
    (void)enable_thinking;
    return ChatReply{false, "this connection keeps no chat history"};
  }

  /// The card's conversation as JSON, with host image paths.
  virtual ChatReply chat_history() {
    return ChatReply{false, "this connection keeps no chat history"};
  }

  /// True if the card cleared its conversation during the last generate()
  /// (cancel, empty answer, or an error). Read after the run has ended.
  virtual bool last_run_cleared_history() const {
    return false;
  }
};

} // namespace simaai::neat::pcie::genai::internal
