/**
 * @file
 * @brief The GenAI Transport that talks to pcie-genai-backend over simaai_svc.
 *
 * Chain: pcie-genai CLI -> GenAIModel -> SvcTransport -> SvcClient ->
 * simaai_mla_daemon -> PCIe -> card. SvcTransport runs one prompt at a time,
 * matches the card's final/error to the run by id, and never hangs: every
 * wait (idle, cancel, leftover drain) has a time limit.
 */
#pragma once

#include "genai/GenAIProtocol.h"
#include "genai/GenAITransport.h"
#include "genai/SvcClient.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::pcie::genai::internal {

struct SvcTransportOptions {
  /// Reported by model_id(); the model subfolder under the host "models" serve root.
  std::string model_id;
  /// One recv() wait. Also how often a cancel request is noticed.
  int recv_timeout_ms = 200;
  /// No event at all for this long means the card is gone.
  int idle_timeout_ms = 120000;
  /// After sending genai.cancel, how long to wait for the card's final.
  int cancel_timeout_ms = 10000;
  /// Absolute host path of the image stage dir (a folder named "pcie-genai"
  /// under the daemon's "data" serve root). Empty = images not available; a
  /// request with image_files is then refused.
  std::string image_stage_dir;
  /// How long a chat command (genai.chat) waits for the card's genai.reply.
  int chat_timeout_ms = 10000;
};

/**
 * @brief The production Transport: drives pcie-genai-backend over simaai_svc.
 *
 * generate() sends genai.prompt and turns genai.token / genai.metrics /
 * genai.final / genai.error into TokenSamples for GenAIModel. It subscribes in
 * the constructor, before any prompt can be sent, because a notification that
 * reaches no listener is dropped.
 */
class SvcTransport final : public Transport {
public:
  SvcTransport(std::unique_ptr<SvcClient> client, SvcTransportOptions options);

  std::string model_id() const override;
  void generate(const GenerationRequest& request,
                const std::vector<std::filesystem::path>& image_files,
                const std::function<bool()>& is_cancelled,
                const std::function<void(const TokenSample&)>& emit) override;
  /// Thread-safe: only sets a flag; generate() sends genai.cancel on its own thread.
  void cancel() override;

  /// Chat commands. Not thread-safe with generate(): call between runs.
  ChatReply reset_chat(const std::optional<std::string>& system_prompt,
                       bool enable_thinking) override;
  ChatReply chat_history() override;
  bool last_run_cleared_history() const override;
  std::uint32_t last_run_dropped_events() const override;

private:
  std::string next_request_id();
  void drain_abandoned_run();
  ChatReply chat_op(ChatOp op, const std::optional<std::string>& system_prompt,
                    bool enable_thinking);

  std::unique_ptr<SvcClient> client_;
  SvcTransportOptions options_;
  std::atomic<bool> cancel_requested_{false};
  std::string abandoned_id_;
  // Per-run token gap tracking. The card stamps each genai.token with a
  // sequence number; a jump means notifications were dropped in transit.
  std::uint64_t expected_token_seq_ = 0;
  std::uint32_t dropped_events_ = 0;
  // Did the card clear its conversation in the last run?
  bool last_history_cleared_ = false;
  // Sent image name -> host path, so print history shows the user's paths.
  // Emptied whenever the card's history is cleared.
  std::map<std::string, std::string> sent_images_;
};

} // namespace simaai::neat::pcie::genai::internal
