#include "simaai/neat/pcie/genai/GenAIModel.h"
#include "genai/RemoteSession.h"
#include "genai/MediaStage.h"
#include "genai/ModelAssets.h"
#include "genai/ModelDirectory.h"
#include "Protocol.h"
#include "Service.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace simaai::neat::pcie::genai {
using Clock = std::chrono::steady_clock;
namespace internal {
// Keep synchronous results equivalent to consuming every streaming sample.
void append_sample(GenerationResult& result, const TokenSample& sample) {
  result.text += sample.text;
  result.reasoning += sample.reasoning;
  for (const auto& call : sample.tool_calls)
    result.tool_calls.push_back(call);
  if (sample.is_final) {
    result.metrics = sample.metrics;
    result.finish_reason = sample.finish_reason;
    result.language = sample.language;
    result.no_speech_prob = sample.no_speech_prob;
    result.avg_logprob = sample.avg_logprob;
  }
}

struct StreamState {
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<TokenSample> samples;
  std::exception_ptr error;
  std::atomic<bool> cancelled{false};
  bool done = false;
  void fail(std::exception_ptr e) {
    std::lock_guard lock(mutex);
    error = e;
    done = true;
    changed.notify_all();
  }
};
} // namespace internal
GenerationStream::GenerationStream(std::shared_ptr<internal::StreamState> state)
    : state_(std::move(state)) {}
GenerationStream::~GenerationStream() {
  cancel();
}
GenerationStream::GenerationStream(GenerationStream&&) noexcept = default;
GenerationStream& GenerationStream::operator=(GenerationStream&& other) noexcept {
  if (this != &other) {
    cancel();
    state_ = std::move(other.state_);
  }
  return *this;
}
void GenerationStream::cancel() {
  if (state_)
    state_->cancelled = true;
}
std::optional<TokenSample> GenerationStream::next() {
  if (!state_)
    throw std::logic_error("Moved-from GenAI stream");
  std::unique_lock lock(state_->mutex);
  state_->changed.wait(lock, [&] { return state_->done || !state_->samples.empty(); });
  if (state_->error)
    std::rethrow_exception(state_->error);
  if (state_->samples.empty())
    return std::nullopt;
  auto sample = std::move(state_->samples.front());
  state_->samples.pop_front();
  return sample;
}

struct GenAIModel::Impl {
  ConnectionOptions options;
  std::string model;
  simaai::neat::genai::internal::ModelDirectoryInfo model_layout;
  internal::RemoteSession remote;
  wire::Service service;
  internal::ModelAssets assets;
  Json capabilities;
  std::mutex mutex, close_mutex;
  std::thread receiver;
  std::atomic<bool> stop{false};
  std::exception_ptr failure;
  std::shared_ptr<internal::StreamState> active;
  std::unique_ptr<internal::MediaStage> media;
  std::optional<Json> pending;
  uint64_t id = 0, sequence = 0;
  Clock::time_point deadline;

  Impl(std::string name, ConnectionOptions connection)
      : options(std::move(connection)), model(internal::host_model_path(name).string()),
        model_layout(simaai::neat::genai::internal::inspect_model_directory(model)),
        remote(options), service(options.card_id), assets(model, remote.id(), options.card_id) {
    service.subscribe(wire::tag(remote.id(), true));
    const auto target = model_layout.root.lexically_relative(model);
    remote.start(target == "." ? "" : target.generic_string(),
                 model_layout.draft_root
                     ? model_layout.draft_root->lexically_relative(model).generic_string()
                     : "");
    const auto until = Clock::now() + std::chrono::milliseconds(options.startup_timeout_ms);
    auto sent = Clock::time_point{};
    while (Clock::now() < until) {
      assets.check();
      if (Clock::now() - sent >= std::chrono::seconds(1)) {
        send(wire::envelope(remote.id(), 0, "hello"));
        sent = Clock::now();
      }
      if (auto payload = service.receive(100)) {
        auto j = wire::parse(*payload, remote.id());
        if (j.at("kind") != "hello")
          continue;
        if (j.contains("error"))
          throw std::runtime_error(j.at("error").get<std::string>());
        if (j.contains("capabilities")) {
          capabilities = j.at("capabilities");
          break;
        }
      }
    }
    if (capabilities.is_null())
      throw std::runtime_error("GenAI worker readiness timed out");
    receiver = std::thread([this] { receive(); });
  }
  ~Impl() {
    try {
      close();
    } catch (...) {
    }
  }
  void send(const Json& j) {
    auto text = j.dump();
    if (text.size() > wire::max_message_bytes)
      throw std::length_error("GenAI message exceeds limit");
    service.send(wire::tag(remote.id(), false), text);
  }
  void close() {
    std::lock_guard closing(close_mutex);
    stop = true;
    if (receiver.joinable())
      receiver.join();
    assets.close();
    remote.stop();
    std::lock_guard lock(mutex);
    if (active)
      active->fail(std::make_exception_ptr(std::runtime_error("GenAI model closed")));
    active.reset();
    media.reset();
    pending.reset();
  }
  std::shared_ptr<internal::StreamState> start(const GenerationRequest& request) {
    std::lock_guard lock(mutex);
    if (stop)
      throw std::logic_error("GenAI model is closed");
    if (failure)
      std::rethrow_exception(failure);
    if (active)
      throw std::logic_error("GenAI model already has an active request");
    if (id == UINT64_MAX)
      throw std::overflow_error("GenAI request IDs exhausted; reopen model");
    const auto request_id = id + 1;
    auto staged = std::make_unique<internal::MediaStage>(options, remote.id(), request_id);
    auto message = wire::envelope(remote.id(), request_id, "generate");
    message["previous_request"] = id;
    message["previous_sequence"] = sequence;
    message["body"] = staged->encode(request);
    if (message.dump().size() > wire::max_message_bytes)
      throw std::length_error("Request exceeds message limit");
    auto state = std::make_shared<internal::StreamState>();
    id = request_id;
    sequence = 0;
    deadline = Clock::now() + std::chrono::milliseconds(options.request_timeout_ms);
    media = std::move(staged);
    pending = std::move(message);
    active = state;
    return state;
  }
  void receive() {
    auto heartbeat = Clock::time_point{}, retry = Clock::time_point{}, last_reply = Clock::now();
    try {
      while (!stop) {
        assets.check();
        const auto now = Clock::now();
        if (now - heartbeat >= std::chrono::seconds(1)) {
          // hello doubles as a lease renewal with a reply, including during long prefill.
          send(wire::envelope(remote.id(), 0, "hello"));
          heartbeat = now;
        }
        {
          std::lock_guard lock(mutex);
          if (active) {
            if (now > deadline)
              throw std::runtime_error("GenAI request timeout");
            if (now - retry >= std::chrono::milliseconds(500)) {
              if (pending)
                send(*pending);
              if (active->cancelled)
                send(wire::envelope(remote.id(), id, "cancel")); // same ID; worker deduplicates
              retry = now;
            }
          }
        }
        if (now - last_reply > std::chrono::seconds(15))
          throw std::runtime_error("GenAI worker stopped responding");
        auto payload = service.receive(100);
        if (!payload)
          continue;
        auto j = wire::parse(*payload, remote.id());
        last_reply = Clock::now();
        const std::string kind = j.at("kind");
        if (kind == "hello") {
          if (j.contains("error"))
            throw std::runtime_error(j.at("error").get<std::string>());
          continue;
        }
        std::lock_guard lock(mutex);
        const uint64_t received_id = j.at("request");
        if (received_id > id)
          throw std::runtime_error("Unknown GenAI result request");
        if (received_id < id)
          continue;
        if (kind == "rejected")
          throw std::runtime_error(j.at("message").get<std::string>());
        if (kind != "sample" && kind != "error")
          throw std::runtime_error("Unknown GenAI event");
        const uint64_t seq = j.at("sequence");
        if (seq <= sequence) {
          auto ack = wire::envelope(remote.id(), id, "ack");
          ack["sequence"] = sequence;
          send(ack);
          continue;
        }
        if (!active)
          throw std::runtime_error("GenAI event after terminal result");
        if (seq != sequence + 1) {
          // Retained events are resent in order; acknowledge only the contiguous prefix.
          auto ack = wire::envelope(remote.id(), id, "ack");
          ack["sequence"] = sequence;
          send(ack);
          continue;
        }
        pending.reset();
        bool final = false;
        if (kind == "error") {
          active->fail(
              std::make_exception_ptr(std::runtime_error(j.at("message").get<std::string>())));
          final = true;
        } else {
          auto sample = wire::decode(j.at("body"));
          final = sample.is_final;
          std::lock_guard state_lock(active->mutex);
          if (active->samples.size() >= wire::event_window && !active->cancelled)
            throw std::runtime_error("GenAI stream consumer too slow; session stopped");
          // Cancellation drops token deltas, not the terminal result needed by drainers.
          if (!active->cancelled || final)
            active->samples.push_back(std::move(sample));
          active->done = final;
          active->changed.notify_all();
        }
        sequence = seq;
        auto ack = wire::envelope(remote.id(), id, "ack");
        ack["sequence"] = sequence;
        send(ack);
        if (final) {
          active.reset();
          media.reset();
        }
      }
      send(wire::envelope(remote.id(), 0, "close"));
    } catch (...) {
      std::lock_guard lock(mutex);
      failure = std::current_exception();
      if (active)
        active->fail(failure);
      // No more heartbeats: the worker lease expires if close cannot reach it.
    }
  }
};
GenAIModel::GenAIModel(std::string model, ConnectionOptions connection)
    : impl_(std::make_shared<Impl>(std::move(model), std::move(connection))) {}
GenAIModel::~GenAIModel() {
  try {
    close();
  } catch (...) {
  }
}
GenAIModel::GenAIModel(GenAIModel&&) noexcept = default;
GenAIModel& GenAIModel::operator=(GenAIModel&& other) noexcept {
  if (this != &other) {
    try {
      close();
    } catch (...) {
    }
    impl_ = std::move(other.impl_);
  }
  return *this;
}
GenAITask GenAIModel::task() const {
  return accepts_audio() ? GenAITask::ASR : GenAITask::VisionLanguage;
}
bool GenAIModel::accepts_text() const {
  return impl_->capabilities.at("text");
}
bool GenAIModel::accepts_image() const {
  return impl_->capabilities.at("image");
}
bool GenAIModel::accepts_audio() const {
  return impl_->capabilities.at("audio");
}
std::string GenAIModel::model_id() const {
  return impl_->model;
}
GenerationStream GenAIModel::stream(const GenerationRequest& request) {
  if (!impl_)
    throw std::logic_error("Moved-from GenAI model");
  return GenerationStream(impl_->start(request));
}
void GenAIModel::close() {
  if (impl_)
    impl_->close();
}
GenerationResult GenAIModel::run(const GenerationRequest& request) {
  auto output = stream(request);
  GenerationResult result;
  while (auto sample = output.next())
    internal::append_sample(result, *sample);
  return result;
}
} // namespace simaai::neat::pcie::genai
