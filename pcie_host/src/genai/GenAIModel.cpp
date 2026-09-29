#include "simaai/neat/pcie/genai/GenAIModel.h"

#include "genai/GenAIModelInternal.h"
#include "genai/GenAITransport.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace simaai::neat::pcie::genai {

class GenAIModel::Impl {
public:
  explicit Impl(std::unique_ptr<internal::Transport> transport) : transport_(std::move(transport)) {
    if (!transport_) {
      throw std::invalid_argument("GenAIModel requires a transport");
    }
  }

  internal::Transport& transport() const {
    return *transport_;
  }

private:
  std::unique_ptr<internal::Transport> transport_;
};

namespace internal {

// Bridges the private GenAIModel constructor for internal factories without
// widening the public API. Only this struct is a friend of GenAIModel.
struct GenAIModelAccess {
  static GenAIModel create(std::unique_ptr<Transport> transport) {
    return GenAIModel(std::make_unique<GenAIModel::Impl>(std::move(transport)));
  }
};

GenAIModel make_model_with_transport(std::unique_ptr<Transport> transport) {
  return GenAIModelAccess::create(std::move(transport));
}

} // namespace internal

GenAIModel::GenAIModel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

GenAIModel::~GenAIModel() = default;
GenAIModel::GenAIModel(GenAIModel&&) noexcept = default;
GenAIModel& GenAIModel::operator=(GenAIModel&&) noexcept = default;

GenAITask GenAIModel::task() const {
  return GenAITask::VisionLanguage;
}

bool GenAIModel::accepts_text() const {
  return true;
}

bool GenAIModel::accepts_image() const {
  // The connection can carry an image; whether the loaded model actually
  // accepts one is decided on the card (pcie-genai-backend replies with an
  // error for a text-only model).
  return true;
}

bool GenAIModel::accepts_audio() const {
  return false;
}

std::string GenAIModel::model_id() const {
  return impl_->transport().model_id();
}

GenerationStream GenAIModel::stream(const GenerationRequest& request) {
  internal::Transport& transport = impl_->transport();
  GenerationRequest request_copy = request;
  return GenerationStream::make(
      [&transport, request_copy = std::move(request_copy)](GenerationStream::Producer& producer) {
        transport.generate(
            request_copy, [&producer] { return producer.cancelled(); },
            [&producer](const TokenSample& sample) {
              if (!sample.is_final) {
                producer.push(sample);
                return;
              }
              producer.record_metric("ttft", sample.metrics.time_to_first_token_s);
              producer.record_metric("tps", sample.metrics.tokens_per_second);
              producer.record_metric("dropped_events", sample.metrics.dropped_events);
              producer.finish(sample.finish_reason, sample.metrics.generated_tokens);
            });
      },
      [&transport] { transport.cancel(); });
}

ChatReply GenAIModel::reset_chat(const std::optional<std::string>& system_prompt,
                                 bool enable_thinking) {
  return impl_->transport().reset_chat(system_prompt, enable_thinking);
}

ChatReply GenAIModel::chat_history() {
  return impl_->transport().chat_history();
}

bool GenAIModel::last_run_cleared_history() const {
  return impl_->transport().last_run_cleared_history();
}

GenerationResult GenAIModel::run(const GenerationRequest& request) {
  GenerationStream token_stream = stream(request);
  GenerationResult result;
  while (const std::optional<TokenSample> sample = token_stream.next()) {
    if (sample->is_final) {
      result.metrics = sample->metrics;
      result.finish_reason = sample->finish_reason;
    } else {
      result.text += sample->text;
    }
  }
  return result;
}

} // namespace simaai::neat::pcie::genai
