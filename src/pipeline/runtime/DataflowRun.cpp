// The Dataflow backend's Run state (H0 plan 3.7): Core's push, try_push, pull, close_input and
// stop on one Session with one Feed, one Fetch and one link. Built only with
// SIMANEAT_DATAFLOW_SOURCE_DIR.
//
// Verbs (WORKER.md 18.1):
//   push      alloc(Acquire::Block) + copy + push_waiting(kNoDeadline): waits for room on the link
//             whatever its policy and never drops (RunPush.cpp:402-411). With OutputOptions::drop
//             the terminal queue is lossy for every push, as appsink drop=true is:
//             push(kNoDeadline).
//   try_push  alloc(Acquire::Fail) + copy + push(passed deadline): the link's policy applies;
//             DroppedIncoming is false and a drop, DroppedOldest true and a drop, a full Block link
//             false and no drop (unit_run_overflow_policy_parity_test.cpp:129-163).
//   pull      -1 is kNoDeadline, 0 a passed deadline (a poll), t > 0 now + t ms. The pulled Tensor
//             is CpuExternal over the pool slot, its holder the Message, read-only.
//   close_input  Session::close_input(kNoDeadline): cancels a parked push, keeps what is queued.
//   stop      Session::stop(now + 2 s), once more on Timeout (RunLifecycle.cpp:94-95).
// The copy into the slot runs on the pushing thread, inside push: when push returns, the caller's
// buffer is free (GStreamer retains the tensor and copies later on Core's input thread).
#if defined(SIMANEAT_WITH_DATAFLOW)

#include "pipeline/runtime/DataflowRun.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/DataflowErrorMap.h"
#include "pipeline/internal/DataflowRuntime.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/InputStream.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/SampleProjection.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/runtime/DataflowGraphCompiler.h"
#include "pipeline/runtime/RunCore.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core/mat.hpp>
#endif

namespace simaai::neat::runtime {

namespace {

namespace df = pipeline_internal::df;
namespace error_util = pipeline_internal::error_util;

// Core's millisecond timeouts at the call's entry (session.h, "Deadlines").
df::Deadline deadline_after(int timeout_ms) {
  if (timeout_ms < 0) {
    return df::kNoDeadline;
  }
  if (timeout_ms == 0) {
    return df::Deadline{};
  }
  return std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
}

std::string names_text(const std::vector<std::string>& names) {
  std::string out = "[";
  for (std::size_t i = 0; i < names.size(); ++i) {
    out += (i == 0 ? "" : ", ") + names[i];
  }
  return out + "]";
}

[[noreturn]] void throw_input_shape(const std::string& message) {
  error_util::throw_session_error(error_codes::kInputShape, message);
}

// The running Session and what was bound with it.
struct Bound {
  Bound(DataflowInputBinding b, df::Session s, std::uint32_t feed_op, std::uint32_t fetch_op)
      : binding(std::move(b)), session(std::move(s)), feed(df::OperatorRef<df::Feed>{feed_op}),
        fetch(df::OperatorRef<df::Fetch>{fetch_op}) {}
  DataflowInputBinding binding;
  df::Session session;
  df::FeedRef feed;
  df::FetchRef fetch;
};

class DataflowRun final : public DataflowRunState {
public:
  DataflowRun(RunCore& core, DataflowRunConfig config)
      : core_(core), config_(std::move(config)), streams_(config_.default_stream) {
    stream_options_.max_input_bytes = config_.run.advanced.max_input_bytes;
  }

  // The Session is destroyed with the Run state, after RunCore::close ran stop (A2: a destructor
  // after a stop that timed out only polls). Messages still held by pulled tensors keep their
  // pool alive on the runtime's retirement list.
  ~DataflowRun() override = default;

  // A seed, or a declaration that gives the spec, binds now; anything else at the first push.
  void bind_at_build(const Tensor* seed) {
    std::string why;
    std::optional<DataflowInputBinding> binding = bind_dataflow_input(config_.input, seed, &why);
    if (!why.empty()) {
      throw_input_shape("Graph::build: " + why + " (Dataflow backend)");
    }
    if (binding) {
      std::lock_guard<std::mutex> lock(mu_);
      bind_locked(std::move(*binding));
    }
  }

  bool push(std::optional<std::string_view> input, const TensorList& inputs, bool block,
            const char* where) override {
    check_input_name(input, where);
    if (inputs.size() != 1U) {
      throw_input_shape(std::string(where) + ": a TensorList of " + std::to_string(inputs.size()) +
                        " tensors (a record) is not supported on the Dataflow backend yet");
    }
    return push_tensor(inputs.front(), nullptr, block, where);
  }

  bool push(std::optional<std::string_view> input, const Sample& msgs, bool block,
            const char* where) override {
    check_input_name(input, where);
    enforce_transferable(msgs, where);
    if (msgs.kind == SampleKind::Bundle) {
      // Core pushes a Bundle as one item per field, stopping at the first refusal
      // (RunPush.cpp:1082-1086).
      for (const Sample& field : msgs.fields) {
        if (!push_item(field, block, where)) {
          return false;
        }
      }
      return true;
    }
    return push_item(msgs, block, where);
  }

  bool push(std::optional<std::string_view> input, const std::vector<cv::Mat>& inputs, bool block,
            const char* where) override {
#if defined(SIMA_WITH_OPENCV)
    check_input_name(input, where);
    if (inputs.size() != 1U) {
      throw std::runtime_error(error_util::decorate_error(
          error_codes::kRuntimePull,
          std::string(where) + ": raw-image ingress supports exactly one cv::Mat per inference "
                               "item (the Dataflow backend carries one tensor per push)"));
    }
    validate_public_input(inputs.front(), config_.input, stream_options_, where);
    // A CPU view of the Mat, whatever the Input's memory policy: the push copies it into the slot.
    InputOptions view_options = config_.input;
    view_options.memory_policy = InputMemoryPolicy::SystemMemory;
    return push_tensor(tensor_from_cv_mat(inputs.front(), view_options, where), nullptr, block,
                       where);
#else
    (void)input;
    (void)inputs;
    (void)block;
    throw std::runtime_error(std::string(where) + ": this build has no OpenCV");
#endif
  }

  bool push_holder(const std::shared_ptr<void>&, bool, const char* where) override {
    error_util::throw_session_error(
        error_codes::kPipelineShape,
        std::string(where) + ": GStreamer buffer holders do not exist on the Dataflow backend", {},
        "Push the Tensor or Sample instead, or build this Run with RunOptions::backend = "
        "RunBackend::GStreamer.");
  }

  PullStatus pull(std::optional<std::string_view> output, int timeout_ms, Sample& out,
                  PullError* err) override {
    error_util::set_pull_error(err, "", "");
    const std::string where =
        output ? "Run::pull(\"" + std::string(*output) + "\")" : std::string("Run::pull");
    if (output) {
      if (output->empty()) {
        error_util::set_pull_error(err, error_codes::kRuntimePull,
                                   "Run::pull(name): output name is empty");
        return PullStatus::Error;
      }
      if (std::find(config_.output_names.begin(), config_.output_names.end(), *output) ==
          config_.output_names.end()) {
        error_util::set_pull_error(
            err, error_codes::kRuntimePull,
            where + ": unknown output. Available outputs: " + names_text(config_.output_names));
        return PullStatus::Error;
      }
    }
    polls_.fetch_add(1, std::memory_order_relaxed);
    const df::Deadline deadline = deadline_after(timeout_ms);
    Bound* bound = wait_bound(deadline);
    if (!bound) {
      // Nothing was pushed yet: a pull waits for the first push in Core (WORKER.md 18.1).
      if (stopped_.load(std::memory_order_acquire) ||
          input_closed_.load(std::memory_order_acquire)) {
        error_util::set_pull_error(err, error_codes::kRuntimePull, where + ": stream closed");
        return PullStatus::Closed;
      }
      error_util::set_pull_error(err, error_codes::kRuntimePull,
                                 where + ": timeout waiting for output");
      return PullStatus::Timeout;
    }

    df::Result<df::Message> pulled = bound->session.pull(bound->fetch, deadline);
    if (const df::Error* error = df::failure(pulled)) {
      const pipeline_internal::DataflowErrorMapping mapping =
          pipeline_internal::map_dataflow_error(*error);
      if (mapping.status == PullStatus::Timeout) {
        error_util::set_pull_error(err, error_codes::kRuntimePull,
                                   where + ": timeout waiting for output");
        return PullStatus::Timeout;
      }
      if (mapping.status == PullStatus::Closed) {
        error_util::set_pull_error(err, error_codes::kRuntimePull, where + ": stream closed");
        // Closed and drained after close_input: the Run stops itself, as the GStreamer Run does
        // at its end of stream (RunPull.cpp:798-810).
        if (input_closed_.load(std::memory_order_acquire) &&
            !stopped_.load(std::memory_order_acquire)) {
          core_.stop();
        }
        return PullStatus::Closed;
      }
      const std::string text = pipeline_internal::dataflow_error_text(*error, where);
      error_util::set_pull_error(err, mapping.code, text,
                                 error_util::make_report(mapping.code, text));
      if (mapping.terminal) {
        std::lock_guard<std::mutex> lock(error_mu_);
        PullError latched;
        latched.code = mapping.code;
        latched.message = text;
        latched.report = error_util::make_report(mapping.code, text);
        last_error_ = std::move(latched);
      }
      return PullStatus::Error;
    }

    out = make_sample(bound->binding, std::get<df::Message>(std::move(pulled)));
    core_.outputs_pulled.fetch_add(1, std::memory_order_relaxed);
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> timing(core_.latency_mu);
    if (!core_.pull_timing_init) {
      core_.first_pull_at = now;
      core_.pull_timing_init = true;
    }
    core_.last_pull_at = now;
    return PullStatus::Ok;
  }

  void close_input() override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      input_closed_.store(true, std::memory_order_release);
    }
    cv_.notify_all();
    if (Bound* bound = bound_.load(std::memory_order_acquire)) {
      // Never waits behind a parked push: that push returns FR3 (A1, A2). Queued messages stay.
      (void)bound->session.close_input(bound->feed, df::kNoDeadline);
    }
  }

  void stop() override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (stopped_.load(std::memory_order_relaxed)) {
        return;
      }
      stopped_.store(true, std::memory_order_release);
    }
    cv_.notify_all();
    Bound* bound = bound_.load(std::memory_order_acquire);
    if (!bound) {
      return;
    }
    // Core's stream-stop budget, waited twice (RunLifecycle.cpp:94-95, 148-182).
    const auto budget = std::chrono::milliseconds(
        std::max(1, pipeline_internal::env_int("SIMA_PIPELINE_STREAM_STOP_TIMEOUT_MS", 2000)));
    df::Status status = bound->session.stop(std::chrono::steady_clock::now() + budget);
    if (status && status->code == df::ErrorCode::Timeout) {
      status = bound->session.stop(std::chrono::steady_clock::now() + budget);
    }
    if (status) {
      std::fprintf(stderr,
                   "[WARN] Run::stop: %s; the Dataflow session keeps its resources until the "
                   "calls inside it leave\n",
                   status->message);
    }
  }

  bool running() const override {
    return !stopped_.load(std::memory_order_acquire);
  }

  std::vector<std::string> input_names() const override {
    return config_.input_names;
  }

  std::vector<std::string> output_names() const override {
    return config_.output_names;
  }

  std::string last_error() const override {
    std::lock_guard<std::mutex> lock(error_mu_);
    return last_error_ ? last_error_->message : std::string{};
  }

  std::optional<PullError> last_error_detail() const override {
    std::lock_guard<std::mutex> lock(error_mu_);
    return last_error_;
  }

  InputStreamStats input_stats() const override {
    InputStreamStats out;
    out.pull_count = core_.outputs_pulled.load(std::memory_order_relaxed);
    out.poll_count = polls_.load(std::memory_order_relaxed);
    if (Bound* bound = bound_.load(std::memory_order_acquire)) {
      const df::SessionReport report = bound->session.report();
      for (const df::FeedReport& feed : report.feeds) {
        out.push_count += feed.pushes;
        out.push_failures += feed.push_failures + feed.backpressure;
      }
      out.dropped_frames = report.dropped_frames;
    }
    return out;
  }

private:
  void check_input_name(std::optional<std::string_view> input, const char* where) const {
    if (!input) {
      return;
    }
    if (input->empty()) {
      throw std::runtime_error(error_util::decorate_error(
          error_codes::kRuntimePull, std::string(where) + ": input name is empty"));
    }
    if (std::find(config_.input_names.begin(), config_.input_names.end(), *input) ==
        config_.input_names.end()) {
      throw std::runtime_error(error_util::decorate_error(
          error_codes::kRuntimePull,
          "Run::push(\"" + std::string(*input) +
              "\"): unknown input. Available inputs: " + names_text(config_.input_names)));
    }
  }

  // Core's cross-Run rule for frames still owned by another graph's producer (RunPush.cpp:39-50):
  // the same check and guidance as on GStreamer. Tensors of any other storage, CPU or DMA-heap,
  // are copied into the slot below.
  static void enforce_transferable(const Sample& msg, const char* where) {
    if (pipeline_internal::env_bool("SIMA_ALLOW_CROSS_RUN_GSTSAMPLE_PUSH", false) ||
        !pipeline_internal::sample_has_device_gstsample_producer_lifetime(msg, false)) {
      return;
    }
    std::string reason;
    if (!pipeline_internal::sample_has_transferable_zero_copy_loan(msg, &reason)) {
      throw std::runtime_error(pipeline_internal::cross_run_zero_copy_sample_error(where) +
                               (reason.empty() ? std::string{} : " Reason: " + reason + "."));
    }
  }

  bool push_item(const Sample& item, bool block, const char* where) {
    if (sample_has_tensor_list(item)) {
      if (item.tensors.size() != 1U) {
        throw_input_shape(std::string(where) + ": a Sample of " +
                          std::to_string(item.tensors.size()) +
                          " tensors (a record) is not supported on the Dataflow backend yet");
      }
      return push_tensor(item.tensors.front(), &item, block, where);
    }
    if (item.tensor.has_value()) {
      return push_tensor(*item.tensor, &item, block, where);
    }
    throw_input_shape(std::string(where) + ": the Sample carries no tensor");
  }

  // The first push binds the input: its tensor's spec, under the declaration (WORKER.md 18.1).
  Bound* bind_first(const Tensor& tensor, const char* where) {
    std::lock_guard<std::mutex> lock(mu_);
    if (Bound* bound = bound_.load(std::memory_order_acquire)) {
      return bound;
    }
    if (stopped_.load(std::memory_order_relaxed) || input_closed_.load(std::memory_order_relaxed)) {
      return nullptr;
    }
    std::string why;
    std::optional<DataflowInputBinding> binding = bind_dataflow_input(config_.input, &tensor, &why);
    if (!binding) {
      throw_input_shape(std::string(where) + ": " +
                        (why.empty() ? std::string("the input cannot be bound") : why) +
                        " (Dataflow backend)");
    }
    return bind_locked(std::move(*binding));
  }

  Bound* bind_locked(DataflowInputBinding binding) {
    DataflowNet net = compile_dataflow_net(config_, binding, streams_.default_id());
    df::Result<df::Session> session = std::move(net.plan).instantiate(df::SessionOptions{});
    if (const df::Error* error = df::failure(session)) {
      pipeline_internal::throw_dataflow_error(*error, "Graph::build");
    }
    auto bound = std::make_unique<Bound>(
        std::move(binding), std::get<df::Session>(std::move(session)), net.feed, net.fetch);
    if (const df::Status status = bound->session.start()) {
      pipeline_internal::throw_dataflow_error(*status, "Graph::build");
    }
    bound_storage_ = std::move(bound);
    bound_.store(bound_storage_.get(), std::memory_order_release);
    cv_.notify_all();
    return bound_storage_.get();
  }

  Bound* wait_bound(df::Deadline deadline) {
    if (Bound* bound = bound_.load(std::memory_order_acquire)) {
      return bound;
    }
    std::unique_lock<std::mutex> lock(mu_);
    const auto ready = [&] {
      return bound_.load(std::memory_order_acquire) != nullptr ||
             stopped_.load(std::memory_order_relaxed) ||
             input_closed_.load(std::memory_order_relaxed);
    };
    if (deadline == df::kNoDeadline) {
      cv_.wait(lock, ready);
    } else {
      cv_.wait_until(lock, deadline, ready);
    }
    return bound_.load(std::memory_order_acquire);
  }

  bool push_tensor(const Tensor& tensor, const Sample* meta, bool block, const char* where) {
    // Core's own push-time checks and texts first (RunPush.cpp:655, InputStreamUtil.cpp:2146-2190).
    validate_public_input(tensor, config_.input, stream_options_, where);
    if (tensor.storage && tensor.storage->kind == StorageKind::GstSample) {
      enforce_transferable(sample_from_tensors(TensorList{tensor}), where);
    }
    Bound* bound = bound_.load(std::memory_order_acquire);
    if (!bound) {
      if (stopped_.load(std::memory_order_acquire) ||
          input_closed_.load(std::memory_order_acquire)) {
        return false;
      }
      bound = bind_first(tensor, where);
      if (!bound) {
        return false;
      }
    }
    if (const std::string why = dataflow_binding_mismatch(bound->binding, tensor); !why.empty()) {
      throw_input_shape(std::string(where) + ": " + why);
    }

    df::Result<df::WriteLoan> allocated =
        bound->session.alloc(bound->feed, block ? df::Acquire::Block : df::Acquire::Fail);
    if (const df::Error* error = df::failure(allocated)) {
      return refused(*error, block, where);
    }
    df::WriteLoan loan = std::get<df::WriteLoan>(std::move(allocated));
    const df::Extent extent = copy_into(bound->binding, tensor, loan.bytes(), where);

    const pipeline_internal::ProjectedStamp identity =
        meta ? pipeline_internal::project_push_identity(*meta, streams_)
             : pipeline_internal::ProjectedStamp{.stream = streams_.default_id(),
                                                 .seq = 0,
                                                 .frame_id = -1,
                                                 .source_time_ns = std::nullopt};
    df::Stamp stamp;
    stamp.stream = df::StreamId{identity.stream};
    stamp.frame_id = identity.frame_id >= 0 ? identity.frame_id : df::kNoFrameId;
    if (identity.source_time_ns) {
      // Media time, not this host's clock: a foreign domain, never compared with publish_time.
      stamp.source_time = df::TimePoint{*identity.source_time_ns, df::ClockId::SensorLocal};
    }

    df::Result<df::PublishResult> pushed =
        !block ? bound->session.push(bound->feed, std::move(loan), stamp, extent, df::Deadline{})
        : config_.terminal_drop
            ? bound->session.push(bound->feed, std::move(loan), stamp, extent, df::kNoDeadline)
            : bound->session.push_waiting(bound->feed, std::move(loan), stamp, extent,
                                          df::kNoDeadline);
    if (const df::Error* error = df::failure(pushed)) {
      return refused(*error, block, where);
    }
    return published(std::get<df::PublishResult>(pushed));
  }

  // The bytes of `tensor` into the slot, at the binding's layout: one tight copy for a dense
  // tensor, row by row at the bound's pitches for NV12 and I420 planes.
  static df::Extent copy_into(const DataflowInputBinding& binding, const Tensor& tensor,
                              std::span<std::byte> slot, const char* where) {
    auto* dst = reinterpret_cast<std::uint8_t*>(slot.data());
    df::Extent extent;
    if (binding.planes == DataflowInputBinding::Planes::None) {
      extent.rank = static_cast<std::uint8_t>(tensor.shape.size());
      for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
        extent.dims[i] = tensor.shape[i];
      }
      if (!tensor.copy_dense_bytes_tight_to(dst, slot.size())) {
        throw_input_shape(std::string(where) +
                          ": the tensor's bytes could not be read for the copy (Dataflow backend)");
      }
      return extent;
    }
    const auto& planar =
        std::get<df::Planar>(std::get<df::TensorSpec>(binding.spec.payload).layout);
    const auto copy_rows = [&](std::size_t plane, const std::uint8_t* src, std::int64_t src_stride,
                               std::int64_t rows, std::int64_t row_bytes) {
      std::uint8_t* out = dst + planar.plane[plane].offset_bytes;
      for (std::int64_t r = 0; r < rows; ++r) {
        std::memcpy(out + r * planar.plane[plane].stride_bytes, src + r * src_stride,
                    static_cast<std::size_t>(row_bytes));
      }
    };
    std::int64_t h = 0;
    std::int64_t w = 0;
    if (binding.planes == DataflowInputBinding::Planes::Nv12) {
      auto mapped = tensor.map_nv12_read();
      if (!mapped) {
        throw_input_shape(std::string(where) + ": the NV12 tensor could not be mapped");
      }
      const Nv12View& v = mapped->view;
      h = v.height;
      w = v.width;
      copy_rows(0, v.y, v.y_stride, h, w);
      copy_rows(1, v.uv, v.uv_stride, h / 2, w);
    } else {
      auto mapped = tensor.map_i420_read();
      if (!mapped) {
        throw_input_shape(std::string(where) + ": the I420 tensor could not be mapped");
      }
      const I420View& v = mapped->view;
      h = v.height;
      w = v.width;
      copy_rows(0, v.y, v.y_stride, h, w);
      copy_rows(1, v.u, v.u_stride, h / 2, w / 2);
      copy_rows(2, v.v, v.v_stride, h / 2, w / 2);
    }
    extent.rank = 3;
    extent.dims[0] = h;
    extent.dims[1] = w;
    extent.dims[2] = 3;
    return extent;
  }

  // A push the Session refused: false, a drop where Core counts one, or Core's error.
  bool refused(const df::Error& error, bool block, const char* where) {
    switch (error.code) {
    case df::ErrorCode::Closed:       // FR3 (close_input), SR1 (stopped), a closed pool or port
    case df::ErrorCode::Backpressure: // FR2: a full Block link at a passed deadline (try_push)
    case df::ErrorCode::Timeout:
      return false;
    case df::ErrorCode::ResourceExhausted:
      // FR7: no free slot for try_push; the application holds every pulled slot. A drop for the
      // dropping policies, as a full queue is (RunPush.cpp:416-424).
      if (!block && (config_.terminal_drop || config_.link_policy != OverflowPolicy::Block)) {
        core_.inputs_dropped.fetch_add(1, std::memory_order_relaxed);
      }
      return false;
    default:
      pipeline_internal::throw_dataflow_error(error, where);
    }
  }

  bool published(const df::PublishResult& result) {
    if (!result.published() || result.size() == 0U) {
      return false; // the port closed under a racing stop
    }
    switch (result.outcome(0)) {
    case df::Link::Offer::Accepted:
      count_published();
      return true;
    case df::Link::Offer::DroppedOldest:
      // Evicted to make room: KeepLatest's input drop, or the terminal queue's output drop.
      count_published();
      (config_.terminal_drop ? core_.outputs_dropped : core_.inputs_dropped)
          .fetch_add(1, std::memory_order_relaxed);
      return true;
    case df::Link::Offer::DroppedIncoming:
      core_.inputs_dropped.fetch_add(1, std::memory_order_relaxed);
      return false;
    case df::Link::Offer::Backpressure:
    case df::Link::Offer::Failed:
    case df::Link::Offer::Closed:
      return false;
    }
    return false;
  }

  void count_published() {
    core_.inputs_enqueued.fetch_add(1, std::memory_order_relaxed);
    core_.inputs_pushed.fetch_add(1, std::memory_order_relaxed);
    core_.outputs_ready.fetch_add(1, std::memory_order_relaxed);
  }

  // The pulled Sample, shaped as the GStreamer Input -> Output Run returns it
  // (InputStreamPull.cpp:830-884): a TensorSet of one tensor, media type and payload tag from the
  // tensor, identity from the stamp (SampleProjection).
  Sample make_sample(const DataflowInputBinding& binding, df::Message message) const {
    const df::Stamp stamp = message.stamp();
    const df::Extent extent = message.extent();
    const std::span<const std::byte> bytes = message.bytes();

    Tensor tensor;
    const bool owned = config_.run.output_memory == OutputMemory::Owned;
    if (owned) {
      // OutputMemory::Owned: an independent CPU copy; the slot returns to the pool now.
      tensor.storage = make_cpu_owned_storage(bytes.size());
      if (!bytes.empty()) {
        std::memcpy(tensor.storage->data, bytes.data(), bytes.size());
      }
      message = df::Message{};
    } else {
      // A lease on the slot: the Message is the holder, released with the last copy of the tensor.
      auto* data = const_cast<std::byte*>(bytes.data());
      const std::size_t size = bytes.size();
      tensor.storage = make_cpu_external_storage(
          data, size, std::make_shared<df::Message>(std::move(message)), true);
    }
    tensor.dtype = binding.dtype;
    tensor.layout = binding.layout;
    tensor.axis_semantics = binding.axis_semantics;
    tensor.device = {DeviceType::CPU, 0};
    tensor.semantic = binding.semantic;
    tensor.read_only = true;
    if (binding.planes == DataflowInputBinding::Planes::None) {
      tensor.shape.assign(extent.dims.begin(), extent.dims.begin() + extent.rank);
    } else {
      const auto& planar =
          std::get<df::Planar>(std::get<df::TensorSpec>(binding.spec.payload).layout);
      const std::int64_t h = extent.dims[0];
      const std::int64_t w = extent.dims[1];
      tensor.shape = {h, w};
      tensor.layout = TensorLayout::HW;
      tensor.strides_bytes = {planar.plane[0].stride_bytes, 1};
      const auto plane = [&](PlaneRole role, std::size_t p, std::int64_t rows, std::int64_t cols) {
        Plane out;
        out.role = role;
        out.shape = {rows, cols};
        out.strides_bytes = {planar.plane[p].stride_bytes, 1};
        out.byte_offset = planar.plane[p].offset_bytes;
        return out;
      };
      tensor.planes.push_back(plane(PlaneRole::Y, 0, h, w));
      if (binding.planes == DataflowInputBinding::Planes::Nv12) {
        tensor.planes.push_back(plane(PlaneRole::UV, 1, h / 2, w));
      } else {
        tensor.planes.push_back(plane(PlaneRole::U, 1, h / 2, w / 2));
        tensor.planes.push_back(plane(PlaneRole::V, 2, h / 2, w / 2));
      }
    }

    Sample out;
    out.kind = SampleKind::TensorSet;
    out.owned = owned;
    const bool image = binding.semantic.image.has_value();
    out.media_type = image ? "video/x-raw" : "application/vnd.simaai.tensor";
    out.payload_type = image ? PayloadType::Image : PayloadType::Tensor;
    if (image) {
      out.payload_tag = pipeline_internal::pixel_format_token(binding.semantic.image->format);
      out.format = out.payload_tag;
    }
    out.tensors.push_back(std::move(tensor));
    pipeline_internal::ProjectedStamp identity;
    identity.stream = stamp.stream.value;
    identity.seq = stamp.seq;
    identity.frame_id = stamp.frame_id;
    if (stamp.source_time.valid()) {
      identity.source_time_ns = stamp.source_time.ns;
    }
    pipeline_internal::apply_pulled_identity(identity, streams_, out);
    return out;
  }

  RunCore& core_;
  DataflowRunConfig config_;
  InputStreamOptions stream_options_;
  pipeline_internal::StreamNames streams_;

  // Binding: written once under mu_, read lock-free through bound_ (release/acquire).
  std::mutex mu_;
  std::condition_variable cv_;
  std::unique_ptr<Bound> bound_storage_;
  std::atomic<Bound*> bound_{nullptr};
  std::atomic<bool> input_closed_{false};
  std::atomic<bool> stopped_{false};
  std::atomic<std::uint64_t> polls_{0};

  mutable std::mutex error_mu_;
  std::optional<PullError> last_error_;
};

} // namespace

std::unique_ptr<DataflowRunState> make_dataflow_run_state(RunCore& core, DataflowRunConfig config,
                                                          const Tensor* seed) {
  auto run = std::make_unique<DataflowRun>(core, std::move(config));
  run->bind_at_build(seed);
  return run;
}

} // namespace simaai::neat::runtime

#endif // SIMANEAT_WITH_DATAFLOW
