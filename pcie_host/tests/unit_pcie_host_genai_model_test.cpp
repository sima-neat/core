#include "genai/GenAIModelInternal.h"
#include "genai/GenAITransport.h"

#include "simaai/neat/pcie/genai/GenAIModel.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace pgenai = simaai::neat::pcie::genai;
namespace pgenai_internal = simaai::neat::pcie::genai::internal;

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

/// A transport that plays back a fixed script of tokens. It records whether it
/// was cancelled so the cancel path can be checked without a card.
class FakeTransport final : public pgenai_internal::Transport {
public:
  std::string model_id() const override {
    return "fake-text-model";
  }

  void generate(const pgenai::GenerationRequest& request, const std::function<bool()>& is_cancelled,
                const std::function<void(const pgenai::TokenSample&)>& emit) override {
    last_prompt_ = request.prompt.value_or("");
    if (is_cancelled()) {
      return;
    }
    pgenai::TokenSample a;
    a.text = "Hel";
    emit(a);
    pgenai::TokenSample b;
    b.text = "lo";
    emit(b);
    pgenai::TokenSample final;
    final.is_final = true;
    final.finish_reason = "stop";
    final.metrics.generated_tokens = 2;
    final.metrics.tokens_per_second = 12.5;
    final.metrics.dropped_events = final_dropped_events;
    emit(final);
  }

  void cancel() override {
    cancel_called = true;
  }

  pgenai::ChatReply reset_chat(const std::optional<std::string>& system_prompt,
                               bool enable_thinking) override {
    last_reset_system = system_prompt;
    last_reset_thinking = enable_thinking;
    return pgenai::ChatReply{true, ""};
  }
  pgenai::ChatReply chat_history() override {
    return pgenai::ChatReply{true, "[]"};
  }
  bool last_run_cleared_history() const override {
    return cleared;
  }

  std::optional<std::string> last_reset_system;
  bool last_reset_thinking = false;
  bool cleared = false;

  std::atomic<bool> cancel_called{false};
  std::string last_prompt_;
  std::uint32_t final_dropped_events = 0;
};

/// State a LifetimeTransport shares with the test. It lives outside the transport,
/// so the test can still read it after the transport is gone.
struct LifetimeState {
  std::atomic<bool> destroyed{false};
  std::atomic<bool> in_generate{false};
  std::atomic<bool> release{false};
  std::atomic<bool> cancelled{false};
};

/// A transport whose generate() waits until the test releases it. It records
/// its own destruction, so the test can check how long it stays alive.
class LifetimeTransport final : public pgenai_internal::Transport {
public:
  explicit LifetimeTransport(std::shared_ptr<LifetimeState> state) : state_(std::move(state)) {}
  ~LifetimeTransport() override {
    state_->destroyed = true;
  }

  std::string model_id() const override {
    return "lifetime-model";
  }

  void generate(const pgenai::GenerationRequest&, const std::function<bool()>&,
                const std::function<void(const pgenai::TokenSample&)>& emit) override {
    state_->in_generate = true;
    while (!state_->release.load() && !state_->cancelled.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    pgenai::TokenSample final;
    final.is_final = true;
    final.finish_reason = "stop";
    emit(final);
  }

  void cancel() override {
    state_->cancelled = true;
  }

private:
  std::shared_ptr<LifetimeState> state_;
};

} // namespace

int main() {
  try {
    // Capabilities: the first host GenAI slice is text-only.
    {
      pgenai::GenAIModel model =
          pgenai_internal::make_model_with_transport(std::make_unique<FakeTransport>());
      require(model.accepts_text(), "host GenAI model must accept text");
      require(model.accepts_image(), "the PCIe connection can carry an image");
      require(!model.accepts_audio(), "text-only model must not accept audio");
      require(model.task() == pgenai::GenAITask::VisionLanguage,
              "text LLM reports the VisionLanguage task family");
      require(model.model_id() == "fake-text-model", "model_id passes through from the transport");
    }

    // run(): the model concatenates token text and reports final metrics.
    {
      pgenai::GenAIModel model =
          pgenai_internal::make_model_with_transport(std::make_unique<FakeTransport>());
      pgenai::GenerationRequest request;
      request.prompt = "hi";
      const pgenai::GenerationResult result = model.run(request);
      require(result.text == "Hello", "run() must concatenate the token text");
      require(result.finish_reason == "stop", "run() must carry the final finish reason");
      require(result.metrics.generated_tokens == 2, "run() must carry the final token count");
    }

    // stream(): tokens arrive in order followed by exactly one final sample.
    {
      pgenai::GenAIModel model =
          pgenai_internal::make_model_with_transport(std::make_unique<FakeTransport>());
      pgenai::GenerationRequest request;
      request.prompt = "hi";
      pgenai::GenerationStream stream = model.stream(request);
      std::string text;
      int finals = 0;
      while (const std::optional<pgenai::TokenSample> sample = stream.next()) {
        if (sample->is_final) {
          ++finals;
        } else {
          text += sample->text;
        }
      }
      require(text == "Hello", "stream() must deliver token text in order");
      require(finals == 1, "stream() must deliver exactly one final sample");
    }

    // cancel(): cancelling the stream signals the transport.
    {
      auto transport = std::make_unique<FakeTransport>();
      FakeTransport* observer = transport.get();
      pgenai::GenAIModel model = pgenai_internal::make_model_with_transport(std::move(transport));
      pgenai::GenerationRequest request;
      request.prompt = "hi";
      pgenai::GenerationStream stream = model.stream(request);
      stream.cancel();
      while (stream.next()) {
      }
      require(observer->cancel_called.load(), "stream cancel must reach the transport");
    }

    // The final sample keeps the transport's dropped_events. An earlier version lost it
    // here, so the CLI always printed "dropped 0".
    {
      auto fake = std::make_unique<FakeTransport>();
      fake->final_dropped_events = 3;
      pgenai::GenAIModel model = pgenai_internal::make_model_with_transport(std::move(fake));
      pgenai::GenerationRequest r;
      r.prompt = "hi";
      pgenai::GenerationStream s = model.stream(r);
      std::optional<pgenai::TokenSample> last;
      while (std::optional<pgenai::TokenSample> x = s.next()) {
        last = x;
      }
      require(last.has_value() && last->is_final, "the stream ends with a final");
      require(last->metrics.dropped_events == 3, "dropped_events reaches the stream's final");
    }

    // A stream keeps the transport alive after its model is gone. An earlier version
    // held only a reference, so the worker and the cancel call used freed memory.
    {
      auto state = std::make_shared<LifetimeState>();
      std::optional<pgenai::GenerationStream> stream;
      {
        pgenai::GenAIModel model =
            pgenai_internal::make_model_with_transport(std::make_unique<LifetimeTransport>(state));
        pgenai::GenerationRequest r;
        r.prompt = "hi";
        stream.emplace(model.stream(r));
        while (!state->in_generate.load()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      }
      require(!state->destroyed.load(), "a live stream keeps the transport alive");
      stream->cancel();
      require(state->cancelled.load(), "cancel reaches the transport after the model is gone");
      while (stream->next()) {
      }
      require(!state->destroyed.load(), "a drained stream still owns the transport");
      stream.reset();
      require(state->destroyed.load(), "the transport is freed with the last stream");
    }

    // Chat operations are forwarded to the transport.
    {
      auto fake = std::make_unique<FakeTransport>();
      FakeTransport* f = fake.get();
      f->cleared = true;
      pgenai::GenAIModel model = pgenai_internal::make_model_with_transport(std::move(fake));
      require(model.reset_chat(std::string(""), true).ok, "reset_chat forwarded");
      require(f->last_reset_system == std::string("") && f->last_reset_thinking,
              "reset_chat passes its arguments");
      require(model.chat_history().text == "[]", "chat_history forwarded");
      require(model.last_run_cleared_history(), "last_run_cleared_history forwarded");
    }

    std::cout << "[PASS] host GenAI model over transport seam\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
