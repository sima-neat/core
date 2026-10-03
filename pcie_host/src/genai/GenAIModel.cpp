#include "simaai/neat/pcie/genai/GenAIModel.h"

#include "genai/GenAIModelInternal.h"
#include "genai/GenAITransport.h"

#include <memory>
#include <mutex>
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

  // The transport has ONE cancel flag for whatever runs now. So only the newest
  // stream, while it runs, may raise it: a cancel from an older stream (a
  // drained handle the caller kept, or its destructor) would stop the next
  // answer. A stream is made active in stream() and gives that up when its
  // generate() returns.
  void activate(const void* stream) {
    std::lock_guard<std::mutex> lock(active_mutex_);
    active_stream_ = stream;
  }
  void deactivate(const void* stream) {
    std::lock_guard<std::mutex> lock(active_mutex_);
    if (active_stream_ == stream) {
      active_stream_ = nullptr;
    }
  }
  void cancel_if_active(const void* stream) {
    std::lock_guard<std::mutex> lock(active_mutex_);
    if (active_stream_ == stream) {
      transport_->cancel();
    }
  }

private:
  std::unique_ptr<internal::Transport> transport_;
  std::mutex active_mutex_;
  const void* active_stream_ = nullptr;
};

namespace internal {

// Bridges the private GenAIModel constructor for internal factories without
// widening the public API. Only this struct is a friend of GenAIModel.
struct GenAIModelAccess {
  static GenAIModel create(std::unique_ptr<Transport> transport) {
    return GenAIModel(std::make_shared<GenAIModel::Impl>(std::move(transport)));
  }
};

GenAIModel make_model_with_transport(std::unique_ptr<Transport> transport) {
  return GenAIModelAccess::create(std::move(transport));
}

} // namespace internal

GenAIModel::GenAIModel(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

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

GenerationStream GenAIModel::stream(const GenerationRequest& request,
                                    const PcieRequestOptions& options) {
  // Both callbacks share ownership of Impl, so the transport outlives the model
  // if the caller destroys or replaces it before the stream ends. `token` only
  // names this stream for the active-stream check (see Impl::activate).
  auto token = std::make_shared<char>();
  impl_->activate(token.get());
  return GenerationStream::make(
      [impl = impl_, request, image_files = options.image_files,
       token](GenerationStream::Producer& producer) {
        struct Deactivate {
          Impl& impl;
          const void* stream;
          ~Deactivate() {
            impl.deactivate(stream);
          }
        } deactivate{*impl, token.get()};
        impl->transport().generate(
            request, image_files, [&producer] { return producer.cancelled(); },
            [&producer](const TokenSample& sample) {
              if (!sample.is_final) {
                producer.push(sample);
                return;
              }
              producer.record_metric("ttft", sample.metrics.time_to_first_token_s);
              producer.record_metric("tps", sample.metrics.tokens_per_second);
              producer.finish(sample.finish_reason, sample.metrics.generated_tokens);
            });
      },
      [impl = impl_, token] { impl->cancel_if_active(token.get()); });
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

std::uint32_t GenAIModel::last_run_dropped_events() const {
  return impl_->transport().last_run_dropped_events();
}

GenerationResult GenAIModel::run(const GenerationRequest& request,
                                 const PcieRequestOptions& options) {
  GenerationStream token_stream = stream(request, options);
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
