#pragma once
#include "genai/GenAIValueTypes.h"
#include "simaai/neat/pcie/Model.h"
#include <filesystem>
#include <memory>
#include <iterator>

namespace simaai::neat::pcie::genai {
using simaai::neat::genai::ASRTask;
using simaai::neat::genai::GenAITask;
using simaai::neat::genai::GenerationMetrics;
using simaai::neat::genai::GenerationResult;
using simaai::neat::genai::Json;
using simaai::neat::genai::TokenSample;

/// GenAI sessions are independent of vision pipeline queues.
struct ConnectionOptions {
  int card_id = 0;
  /// Optional address override; when empty, card N uses 10.0.N.2.
  std::string card_host;
  std::string user = "root";
  std::string ssh_key;
  std::string media_serve_root = "data";
  std::filesystem::path media_directory = "/srv/simaai/data";
  /// Card-side temporary asset directory; must match the platform receive location.
  std::filesystem::path card_receive_directory = "/srv/simaai/incoming";
  int startup_timeout_ms = 900000;
  int request_timeout_ms = 900000;
};

struct ChatMessage {
  std::string role;
  std::string content;
  std::vector<Tensor> images; ///< UInt8 HWC RGB tensors.
  Json tool_calls = Json::array();
  std::optional<std::string> tool_call_id;
  std::optional<std::string> name;
};

struct GenerationRequest {
  std::optional<std::string> prompt;
  std::optional<std::string> system_prompt;
  std::vector<ChatMessage> messages;
  std::vector<Tensor> images;  ///< UInt8 HWC RGB tensors.
  std::optional<Tensor> audio; ///< Float32 mono samples, shape [N].
  uint32_t sample_rate = 16000;
  std::optional<std::filesystem::path> audio_file;
  std::string language = "auto";
  ASRTask asr_task = ASRTask::Transcribe;
  uint32_t max_new_tokens = 0;
  bool enable_thinking = false;
  Json tools = Json::array();
  Json tool_choice = nullptr;
};

namespace internal {
struct StreamState;
}
class GenerationStream {
public:
  ~GenerationStream();
  GenerationStream(GenerationStream&&) noexcept;
  GenerationStream& operator=(GenerationStream&&) noexcept;
  GenerationStream(const GenerationStream&) = delete;
  GenerationStream& operator=(const GenerationStream&) = delete;
  std::optional<TokenSample> next();
  void cancel();
  class iterator {
  public:
    explicit iterator(GenerationStream* stream = nullptr) : stream_(stream) {
      advance();
    }
    const TokenSample& operator*() const {
      return *sample_;
    }
    const TokenSample* operator->() const {
      return &*sample_;
    }
    iterator& operator++() {
      advance();
      return *this;
    }
    void operator++(int) {
      advance();
    }
    bool operator==(std::default_sentinel_t) const {
      return !sample_;
    }

  private:
    void advance() {
      if (stream_)
        sample_ = stream_->next();
    }
    GenerationStream* stream_;
    std::optional<TokenSample> sample_;
  };
  iterator begin() {
    return iterator(this);
  }
  std::default_sentinel_t end() const {
    return {};
  }

private:
  explicit GenerationStream(std::shared_ptr<internal::StreamState> state);
  std::shared_ptr<internal::StreamState> state_;
  friend class GenAIModel;
};

/// One remote Core worker per handle. Requests carry explicit conversation
/// history; independent prompts do not inherit previous requests.
class GenAIModel {
public:
  /// Accepts a host model directory, absolute or relative to the current working
  /// directory. Model assets load on demand over PCIe.
  explicit GenAIModel(std::string model, ConnectionOptions connection = {});
  ~GenAIModel();
  GenAIModel(GenAIModel&&) noexcept;
  GenAIModel& operator=(GenAIModel&&) noexcept;
  GenAIModel(const GenAIModel&) = delete;
  GenAIModel& operator=(const GenAIModel&) = delete;
  GenAITask task() const;
  bool accepts_text() const;
  bool accepts_image() const;
  bool accepts_audio() const;
  std::string model_id() const;
  GenerationResult run(const GenerationRequest& request);
  GenerationStream stream(const GenerationRequest& request);
  void close();

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
} // namespace simaai::neat::pcie::genai
