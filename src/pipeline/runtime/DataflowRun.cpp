// The Dataflow backend's Run state (H0 plan 3.7; H0 step 10, H0c): Core's push, try_push, pull,
// close_input and stop over one DataflowInputPort per public Input (its Session: one Feed, and one
// link and one Fetch per Output the input feeds). Built only with SIMANEAT_DATAFLOW_SOURCE_DIR.
// This file routes endpoint names to ports, orders the Run-wide close and stop, latches the
// terminal error, sums the statistics and maps what a port answered a push to Core's return value
// and counters (push_verdict). The slot, the binding and the tensor's bytes are the port's
// (DataflowInputPort.cpp), the input's format DataflowInputBinding's, a message's tensors
// DataflowTensorCodec.h's.
//
// Endpoints: push(name) and pull(name) select the input or output by its public name; the unnamed
// verbs reach the only input or output, and with several fail as GStreamer's graph path does ("no
// unambiguous default input/output"). Each input binds by its declaration, the seed (a graph of
// one input), or its own first push; a pull waits for its own input's binding. In a Run of several
// inputs, Sample::orig_input_seq is the Run's push counter, as on GStreamer's graph path;
// input_seq and Stamp::seq count each input's pushes.
//
// Verbs (WORKER.md 18.1):
//   push      waits for room on each link of the input and never drops (RunPush.cpp:402-411); an
//             Output with OutputOptions::drop is lossy for every push, as appsink drop=true is.
//   try_push  each link's policy applies: DroppedIncoming is false and a drop, DroppedOldest true
//             and a drop, a full Block link false and no drop
//             (unit_run_overflow_policy_parity_test.cpp:129-163).
//   pull      -1 is kNoDeadline, 0 a passed deadline (a poll), t > 0 now + t ms. The pulled Tensor
//             is CpuExternal over the pool slot, its holder the Message, read-only; a wrapped
//             message's is the pushed storage itself, and the slot returns at the pull. The
//             Outputs of a fan-out lease the one slot each message has: no copy per Output.
//             From a device input (H0b: memory_policy Ev74 or Dms0) it is device memory, a
//             GstSample over the slot's DMA-BUF, read-only as well.
// A device input's push and the tensors of its pulled messages are DataflowDeviceMemory.cpp's
// (dataflow_device::push, pulled_sample, pulled_tensors): another translation unit, reached here
// by one branch each, so that what this file compiles for a host input stays what it was.
//   close_input  Session::close_input(kNoDeadline) of every input: cancels a parked push, keeps
//             what is queued.
//   stop      Session::stop(now + 2 s) of every input, once more on Timeout (RunLifecycle.cpp
//             94-95).
// A pushed TensorList's first tensor's stamp proposes the message's stream, frame_id, source_time
// and flags; every pulled tensor carries the message's Stamp in its envelope.
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
#include "pipeline/runtime/DataflowInputPort.h"
#include "pipeline/runtime/DataflowTensorCodec.h"
#include "pipeline/runtime/RunCore.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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

namespace codec = dataflow_codec;
namespace error_util = pipeline_internal::error_util;
using codec::throw_input_shape;

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

// ---- a push's outcome --------------------------------------------------------------------------

// What decides the outcome of a push besides the Session's answer.
struct PushPolicy {
  bool block = false;       ///< push (waits, never drops); false: try_push
  bool lossy_block = false; ///< an Output of the input has OutputOptions::drop
  bool fan_out = false;     ///< the input feeds several Outputs (several links)
  OverflowPolicy link_policy = OverflowPolicy::Block; ///< DataflowRunConfig::link_policy
};

// Core's return value and counter deltas for one push.
struct PushVerdict {
  bool accepted = false; ///< push's return; one input enqueued and pushed
  bool error = false;    ///< the Session's error is Core's error: thrown
  std::uint32_t outputs_ready = 0;
  std::uint32_t inputs_dropped = 0;
  std::uint32_t outputs_dropped = 0;
};

// The one mapping from what the Session answered a push to what Core returns and counts.
//   Unbound  a closed input or a stopped Run before any binding: false.
//   Alloc    the slot was refused. Under try_push, Backpressure is the runtime's FR14: every link
//            of the Feed full, so the push would be refused, and alloc refused it before any slot
//            or copy, counted by the runtime as that push counts it. Core does what it does when
//            the push refused: false, and on DropIncoming its input drop. KeepLatest and the
//            terminal drop are DropOldest links, which never refuse: a fan-out with one of them
//            never sees FR14. Every other refusal is the push's:
//   Publish, refused: Closed (FR3 close_input, SR1 stopped, a closed pool or port), Backpressure
//            (FR2: a full Block link at a passed deadline) and Timeout are false.
//            ResourceExhausted is FR7, no free slot for try_push (the application holds every
//            pulled slot): a drop for the dropping policies, as a full queue is
//            (RunPush.cpp:416-424). Anything else is an error.
//   Publish, answered: one link: Accepted; DroppedOldest, accepted with a drop (evicted to make
//            room: KeepLatest's input drop, or the terminal queue's output drop); DroppedIncoming,
//            false and an input drop. A fan-out: each link answered for its own Output. The push
//            is accepted when one link took the message (each that did has one more item ready),
//            and an eviction, or a refusal while another link took it, is that Output's drop. When
//            no link took it the push is refused, and a drop of the input when a DropIncoming link
//            refused it.
PushVerdict push_verdict(const DataflowPushOutcome& outcome, const PushPolicy& policy) noexcept {
  PushVerdict v;
  if (outcome.stage == DataflowPushOutcome::Stage::Unbound) {
    return v;
  }
  if (const df::Error* error = df::failure(outcome.answer)) {
    if (outcome.stage == DataflowPushOutcome::Stage::Alloc && !policy.block &&
        error->code == df::ErrorCode::Backpressure) {
      v.inputs_dropped =
          !policy.lossy_block && policy.link_policy == OverflowPolicy::DropIncoming ? 1U : 0U;
      return v;
    }
    switch (error->code) {
    case df::ErrorCode::Closed:
    case df::ErrorCode::Backpressure:
    case df::ErrorCode::Timeout:
      return v;
    case df::ErrorCode::ResourceExhausted:
      v.inputs_dropped =
          !policy.block && (policy.lossy_block || policy.link_policy != OverflowPolicy::Block) ? 1U
                                                                                               : 0U;
      return v;
    default:
      v.error = true;
      return v;
    }
  }
  const df::PublishResult& result = std::get<df::PublishResult>(outcome.answer);
  if (!result.published() || result.size() == 0U) {
    return v; // the port closed under a racing stop
  }
  if (policy.fan_out) {
    std::uint32_t evicted = 0;
    std::uint32_t refused_incoming = 0;
    for (std::uint32_t k = 0; k < result.size(); ++k) {
      const df::Link::Offer offer = result.outcome(k);
      evicted += offer == df::Link::Offer::DroppedOldest ? 1U : 0U;
      refused_incoming += offer == df::Link::Offer::DroppedIncoming ? 1U : 0U;
    }
    if (result.accepted() == 0U) {
      v.inputs_dropped = refused_incoming != 0U ? 1U : 0U;
      return v;
    }
    v.accepted = true;
    v.outputs_ready = result.accepted();
    v.outputs_dropped = evicted + refused_incoming;
    return v;
  }
  switch (result.outcome(0)) {
  case df::Link::Offer::Accepted:
    v.accepted = true;
    v.outputs_ready = 1;
    return v;
  case df::Link::Offer::DroppedOldest:
    v.accepted = true;
    v.outputs_ready = 1;
    (policy.lossy_block ? v.outputs_dropped : v.inputs_dropped) = 1;
    return v;
  case df::Link::Offer::DroppedIncoming:
    v.inputs_dropped = 1;
    return v;
  case df::Link::Offer::Backpressure:
  case df::Link::Offer::Failed:
  case df::Link::Offer::Closed:
    return v;
  }
  return v;
}

// Where a public Output is read: its Input's port and its link there.
struct OutputSlot {
  std::size_t port = 0;
  std::size_t link = 0;
};

constexpr std::size_t kNoEndpoint = static_cast<std::size_t>(-1);

class DataflowRun final : public DataflowRunState {
public:
  DataflowRun(RunCore& core, DataflowRunConfig config)
      : core_(core), config_(std::move(config)), streams_(config_.inputs.front().default_stream) {
    stream_options_.max_input_bytes = config_.run.advanced.max_input_bytes;
  }

  ~DataflowRun() override = default;

  // One port per public Input, and where each Output is read (outside the constructor, whose
  // body then cannot throw).
  void add_ports() {
    const DataflowPortContext context{
        core_,    config_,  stream_options_,
        streams_, stopped_, config_.inputs.size() > 1U ? &run_seq_ : nullptr};
    for (std::size_t i = 0; i < config_.inputs.size(); ++i) {
      ports_.push_back(std::make_unique<DataflowInputPort>(context, i));
      const std::vector<std::size_t>& outputs = config_.inputs[i].outputs;
      for (std::size_t k = 0; k < outputs.size(); ++k) {
        if (slots_.size() <= outputs[k]) {
          slots_.resize(outputs[k] + 1U);
        }
        slots_[outputs[k]] = OutputSlot{i, k};
      }
    }
    // The unnamed push and pull reach the only input and the only output; with several there is
    // no default, as on GStreamer's graph path (plan.default_input, plan.default_output).
    default_input_ = ports_.size() == 1U ? ports_.front().get() : nullptr;
    default_output_ = config_.outputs.size() == 1U ? 0U : kNoEndpoint;
  }

  // A seed (one input only: the build checked it), or a declaration that gives the spec, binds now;
  // every other input binds at its own first push.
  void bind_at_build(std::span<const Tensor> seed) {
    for (const auto& port : ports_) {
      port->bind_at_build(seed);
    }
  }

  bool push(std::optional<std::string_view> input, const TensorList& inputs, bool block,
            const char* where) override {
    DataflowInputPort& port = input_for_push(input, where);
    if (inputs.empty()) {
      throw_input_shape(std::string(where) + ": an empty tensor list");
    }
    // The first tensor's stamp proposes the item's identity; a tensor without a stamp proposes
    // nothing (the default stream, no frame_id), as develop's TensorList push.
    const Stamp& proposal = inputs.front().stamp();
    if (std::optional<TensorList> fields = pipeline_internal::list_record_fields(inputs, where)) {
      return push_item(port, *fields, proposal, block, where);
    }
    return push_item(port, inputs, proposal, block, where, /*commit_allocation=*/block);
  }

  bool push(std::optional<std::string_view> input, const Sample& msgs, bool block,
            const char* where) override {
    DataflowInputPort& port = input_for_push(input, where);
    DataflowInputPort::enforce_transferable(msgs, where);
    if (msgs.kind == SampleKind::Bundle) {
      // Core pushes a Bundle as one item per field, stopping at the first refusal
      // (RunPush.cpp:1082-1086).
      for (const Sample& field : msgs.fields) {
        if (!push_sample_item(port, field, block, where)) {
          return false;
        }
      }
      return true;
    }
    return push_sample_item(port, msgs, block, where);
  }

  bool push(std::optional<std::string_view> input, const std::vector<cv::Mat>& inputs, bool block,
            const char* where) override {
#if defined(SIMA_WITH_OPENCV)
    DataflowInputPort& port = input_for_push(input, where);
    if (inputs.size() != 1U) {
      throw std::runtime_error(error_util::decorate_error(
          error_codes::kRuntimePull,
          std::string(where) + ": raw-image ingress supports exactly one cv::Mat per inference "
                               "item (the Dataflow backend carries one tensor per push)"));
    }
    validate_public_input(inputs.front(), port.config().input, stream_options_, where);
    // A CPU view of the Mat, whatever the Input's memory policy: the push copies it into the slot
    // (a device input's slot is the device memory), or wraps it into a host input when the Mat
    // owns its memory (its view's holder keeps a reference to it; a Mat over memory it does not
    // own keeps nothing alive).
    InputOptions view_options = port.config().input;
    view_options.memory_policy = InputMemoryPolicy::SystemMemory;
    const Tensor tensor = tensor_from_cv_mat(inputs.front(), view_options, where);
    // A device input wraps no CPU view; there `may_wrap` false marks an image, which GStreamer's
    // ingress copies into device memory without its CPU-tensor guard (a CV_32F matrix is a CPU
    // tensor to it, and to dataflow_device::push).
    const bool may_wrap =
        port.device() ? !tensor.semantic.image.has_value() : inputs.front().u != nullptr;
    return push_item(port, std::span<const Tensor>(&tensor, 1), Stamp{}, block, where, false, -1,
                     may_wrap);
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
    df::Message message;
    DataflowInputPort::Bound* bound = nullptr;
    const PullStatus status = take(output, timeout_ms, err, message, bound);
    if (status != PullStatus::Ok) {
      return status;
    }
    // A Run of several inputs: the Run's push counter, which the message carries.
    const std::int64_t orig = DataflowInputPort::orig_input_seq(message);
    // The caller's list buffer is reused: Run::pull(int, Sample&, PullError*) with one Sample
    // allocates no list per pull.
    TensorList tensors = std::move(out.tensors);
    tensors.clear();
    out =
        bound->device
            ? dataflow_device::pulled_sample(bound->binding, streams_, owned_output(),
                                             std::move(message), std::move(tensors))
            : codec::make_sample(bound->binding, bound->has_wraps.load(std::memory_order_relaxed),
                                 streams_, owned_output(), std::move(message), std::move(tensors));
    if (orig >= 0) {
      out.orig_input_seq = orig;
    }
    count_pulled();
    return PullStatus::Ok;
  }

  PullStatus pull(std::optional<std::string_view> output, int timeout_ms, TensorList& out,
                  PullError* err) override {
    df::Message message;
    DataflowInputPort::Bound* bound = nullptr;
    const PullStatus status = take(output, timeout_ms, err, message, bound);
    if (status != PullStatus::Ok) {
      return status;
    }
    if (bound->device) [[unlikely]] {
      dataflow_device::pulled_tensors(bound->binding, streams_, owned_output(), std::move(message),
                                      out);
    } else {
      codec::fill_tensors(bound->binding, bound->has_wraps.load(std::memory_order_relaxed),
                          streams_, owned_output(), std::move(message), out);
    }
    count_pulled();
    return PullStatus::Ok;
  }

  Tensor alloc(std::string_view input, const char* where) override {
    const std::string at = std::string(where) + "(\"" + std::string(input) + "\")";
    DataflowInputPort* port = find_input(input);
    if (!port) {
      throw std::runtime_error(error_util::decorate_error(
          error_codes::kRuntimePull,
          at + ": unknown input. Available inputs: " + dataflow_names_text(config_.input_names)));
    }
    return port->alloc(at);
  }

  // Every input of the Run (Run::close_input names none): each one's queued messages stay pullable,
  // then its Outputs are Closed.
  void close_input() override {
    for (const auto& port : ports_) {
      port->mark_closed();
    }
    for (const auto& port : ports_) {
      port->close_feed();
    }
  }

  void stop() override {
    if (stopped_.exchange(true, std::memory_order_acq_rel)) {
      return;
    }
    for (const auto& port : ports_) {
      port->wake();
    }
    trace_report();
    // Core's stream-stop budget, waited twice (RunLifecycle.cpp:94-95, 148-182), shared by the
    // Sessions: each stop wakes what is parked in its own Session, so they run one after the other
    // against one deadline, and the ones that timed out once more against a second one.
    const auto budget = std::chrono::milliseconds(
        std::max(1, pipeline_internal::env_int("SIMA_PIPELINE_STREAM_STOP_TIMEOUT_MS", 2000)));
    std::vector<DataflowInputPort*> pending;
    for (const auto& port : ports_) {
      pending.push_back(port.get());
    }
    for (int round = 0; round < 2 && !pending.empty(); ++round) {
      const auto deadline = std::chrono::steady_clock::now() + budget;
      std::vector<DataflowInputPort*> timed_out;
      for (DataflowInputPort* port : pending) {
        const df::Status status = port->stop(deadline);
        if (!status) {
          continue;
        }
        if (status->code == df::ErrorCode::Timeout && round == 0) {
          timed_out.push_back(port);
          continue;
        }
        std::fprintf(stderr,
                     "[WARN] Run::stop: %s; the Dataflow session keeps its resources until the "
                     "calls inside it leave\n",
                     status->message);
      }
      pending = std::move(timed_out);
    }
  }

  bool running() const override {
    return !stopped_.load(std::memory_order_acquire);
  }

  bool can_push() const override {
    return default_input_ != nullptr;
  }

  bool can_pull() const override {
    return default_output_ != kNoEndpoint;
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

  // Every input's Feed and every link, summed: the Run's input statistics.
  InputStreamStats input_stats() const override {
    InputStreamStats out;
    out.pull_count = core_.outputs_pulled.load(std::memory_order_relaxed);
    out.poll_count = polls_.load(std::memory_order_relaxed);
    for (const auto& port : ports_) {
      if (const std::optional<df::SessionReport> report = port->report()) {
        for (const df::FeedReport& feed : report->feeds) {
          out.push_count += feed.pushes;
          out.push_failures += feed.push_failures + feed.backpressure;
        }
        out.dropped_frames += report->dropped_frames;
      }
    }
    return out;
  }

private:
  bool owned_output() const noexcept {
    return config_.run.output_memory == OutputMemory::Owned;
  }

  DataflowInputPort* find_input(std::string_view name) const {
    for (const auto& port : ports_) {
      const std::vector<std::string>& names = port->config().names;
      if (std::find(names.begin(), names.end(), name) != names.end()) {
        return port.get();
      }
    }
    return nullptr;
  }

  // The input a push names, or the Run's only one for the unnamed push: GStreamer's checks and
  // texts (named_input_endpoint_or_throw, RunPush.cpp:216; RunCore::push_samples).
  DataflowInputPort& input_for_push(std::optional<std::string_view> input,
                                    const char* where) const {
    if (!input) {
      if (default_input_) {
        return *default_input_;
      }
      throw std::runtime_error(error_util::decorate_error(
          error_codes::kRuntimePull,
          "Run::push: graph has no unambiguous default input; use push(name, ...). Available "
          "inputs: " +
              dataflow_names_text(config_.input_names)));
    }
    if (input->empty()) {
      throw std::runtime_error(error_util::decorate_error(
          error_codes::kRuntimePull, std::string(where) + ": input name is empty"));
    }
    if (DataflowInputPort* port = find_input(*input)) {
      return *port;
    }
    throw std::runtime_error(error_util::decorate_error(
        error_codes::kRuntimePull,
        "Run::push(\"" + std::string(*input) +
            "\"): unknown input. Available inputs: " + dataflow_names_text(config_.input_names)));
  }

  // One item of a Sample push: its tensors (one, or several as a record) and its identity.
  bool push_sample_item(DataflowInputPort& port, const Sample& item, bool block,
                        const char* where) {
    const Stamp proposal = pipeline_internal::stamp_from_sample(item);
    // What GStreamer's graph ingress keeps as orig_input_seq without counting the push: a pushed
    // orig_input_seq, else a pushed input_seq (RunPush.cpp:276-287). Read in a Run of several
    // inputs only.
    const std::int64_t pushed_seq = item.orig_input_seq >= 0 ? item.orig_input_seq
                                    : item.input_seq >= 0    ? item.input_seq
                                                             : -1;
    if (sample_has_tensor_list(item)) {
      return push_item(port, item.tensors, proposal, block, where, false, pushed_seq);
    }
    if (item.tensor.has_value()) {
      return push_item(port, std::span<const Tensor>(&*item.tensor, 1), proposal, block, where,
                       false, pushed_seq);
    }
    throw_input_shape(std::string(where) + ": the Sample carries no tensor");
  }

  // One item into `port` (DataflowInputPort::push, inlined here once; a device input's is
  // device_push), and its outcome.
  bool push_item(DataflowInputPort& port, std::span<const Tensor> item, const Stamp& proposal,
                 bool block, const char* where, bool commit_allocation = false,
                 std::int64_t pushed_seq = -1, bool may_wrap = true) {
    const DataflowPushOutcome outcome =
        port.device()
            ? device_push(port, item, proposal, block, where, commit_allocation, pushed_seq,
                          may_wrap)
            : port.push(item, proposal, block, where, commit_allocation, pushed_seq, may_wrap);
    return settle(port, outcome, block, where);
  }

  // A device input's push (H0b). Its slot is device memory under a write epoch that the publish
  // ends; an end that fails (FR16, Unretired) published nothing and quarantined the slot: the
  // Run's terminal error, as a pull's is, latched here before settle throws it.
  [[gnu::noinline, gnu::cold]] DataflowPushOutcome
  device_push(DataflowInputPort& port, std::span<const Tensor> item, const Stamp& proposal,
              bool block, const char* where, bool commit_allocation, std::int64_t pushed_seq,
              bool may_wrap) {
    DataflowPushOutcome outcome = dataflow_device::push(port, item, proposal, block, where,
                                                        commit_allocation, pushed_seq, may_wrap);
    if (const df::Error* error = df::failure(outcome.answer);
        error && error->code == df::ErrorCode::Unretired) {
      const std::string text = pipeline_internal::dataflow_error_text(*error, where);
      PullError latched;
      latched.code = pipeline_internal::map_dataflow_error(*error).code;
      latched.message = text;
      latched.report = error_util::make_report(latched.code, text);
      std::lock_guard<std::mutex> lock(error_mu_);
      last_error_ = std::move(latched);
    }
    return outcome;
  }

  // What a port answered a push, as Core returns and counts it (push_verdict): the one place the
  // Run's push counters change.
  bool settle(const DataflowInputPort& port, const DataflowPushOutcome& outcome, bool block,
              const char* where) {
    const PushVerdict v = push_verdict(
        outcome, PushPolicy{block, port.lossy_block(), port.fan_out(), config_.link_policy});
    if (v.error) {
      pipeline_internal::throw_dataflow_error(*df::failure(outcome.answer), where);
    }
    if (v.accepted) {
      core_.inputs_enqueued.fetch_add(1, std::memory_order_relaxed);
      core_.inputs_pushed.fetch_add(1, std::memory_order_relaxed);
      core_.outputs_ready.fetch_add(v.outputs_ready, std::memory_order_relaxed);
    }
    if (v.inputs_dropped != 0U) {
      core_.inputs_dropped.fetch_add(v.inputs_dropped, std::memory_order_relaxed);
    }
    if (v.outputs_dropped != 0U) {
      core_.outputs_dropped.fetch_add(v.outputs_dropped, std::memory_order_relaxed);
    }
    return v.accepted;
  }

  // error_util::set_pull_error(err, "", ""), without building its arguments: no code, no message,
  // no report.
  static void clear_pull_error(PullError* err) {
    if (err) {
      err->code.clear();
      err->message.clear();
      err->report.reset();
    }
  }

  // A pull that found nothing in time. pull(0) is a non-blocking poll and leaves the error empty,
  // as GStreamer's Run does (RunPull.cpp:660-664): pull_samples() drains with it, and "nothing
  // more right now" must not read as a stall (unit_run_api_matrix_test's poll contract).
  static PullStatus timed_out(int timeout_ms, const std::string& where, PullError* err) {
    if (timeout_ms != 0) {
      error_util::set_pull_error(err, error_codes::kRuntimePull,
                                 where + ": timeout waiting for output");
    }
    return PullStatus::Timeout;
  }

  // The output a pull names, or the Run's only one for the unnamed pull; kNoEndpoint with the
  // error set (GStreamer's texts: RunCore::pull, RunCore::pull_named_output).
  std::size_t output_for_pull(std::optional<std::string_view> output, PullError* err) const {
    if (!output) {
      if (default_output_ == kNoEndpoint) {
        error_util::set_pull_error(
            err, error_codes::kRuntimePull,
            "Run::pull: graph has no unambiguous default output; use pull(name, ...). Available "
            "outputs: " +
                dataflow_names_text(config_.output_names));
      }
      return default_output_;
    }
    if (output->empty()) {
      error_util::set_pull_error(err, error_codes::kRuntimePull,
                                 "Run::pull(name): output name is empty");
      return kNoEndpoint;
    }
    for (std::size_t o = 0; o < config_.outputs.size(); ++o) {
      const std::vector<std::string>& names = config_.outputs[o].names;
      if (std::find(names.begin(), names.end(), *output) != names.end()) {
        return o;
      }
    }
    error_util::set_pull_error(
        err, error_codes::kRuntimePull,
        "Run::pull(\"" + std::string(*output) +
            "\"): unknown output. Available outputs: " + dataflow_names_text(config_.output_names));
    return kNoEndpoint;
  }

  // The next message of the output and the bound input it came through, or why there is none
  // (both pull verbs). The call's name for messages is built only when a message needs it.
  PullStatus take(std::optional<std::string_view> output, int timeout_ms, PullError* err,
                  df::Message& message, DataflowInputPort::Bound*& bound) {
    clear_pull_error(err);
    const auto where = [&output] {
      return output ? "Run::pull(\"" + std::string(*output) + "\")" : std::string("Run::pull");
    };
    const std::size_t o = output_for_pull(output, err);
    if (o == kNoEndpoint) {
      return PullStatus::Error;
    }
    const OutputSlot slot = slots_[o];
    DataflowInputPort* const port = ports_[slot.port].get();
    polls_.fetch_add(1, std::memory_order_relaxed);
    const df::Deadline deadline = deadline_after(timeout_ms);
    bound = port->wait_bound(deadline);
    if (!bound) {
      // Nothing was pushed into its input yet: a pull waits for that first push in Core
      // (WORKER.md 18.1).
      if (stopped_.load(std::memory_order_acquire) || port->closed()) {
        error_util::set_pull_error(err, error_codes::kRuntimePull, where() + ": stream closed");
        return PullStatus::Closed;
      }
      return timed_out(timeout_ms, where(), err);
    }

    df::Result<df::Message> pulled = DataflowInputPort::pull(*bound, slot.link, deadline);
    if (const df::Error* error = df::failure(pulled)) {
      const pipeline_internal::DataflowErrorMapping mapping =
          pipeline_internal::map_dataflow_error(*error);
      if (mapping.status == PullStatus::Timeout) {
        return timed_out(timeout_ms, where(), err);
      }
      if (mapping.status == PullStatus::Closed) {
        error_util::set_pull_error(err, error_codes::kRuntimePull, where() + ": stream closed");
        // Closed and drained after close_input: a Run of one output stops itself, as the GStreamer
        // Run does at its end of stream (RunPull.cpp:798-810). With several outputs the others
        // may still hold messages; GStreamer's graph path does not stop at a closed output either.
        if (config_.outputs.size() == 1U && port->closed() &&
            !stopped_.load(std::memory_order_acquire)) {
          core_.stop();
        }
        return PullStatus::Closed;
      }
      const std::string text = pipeline_internal::dataflow_error_text(*error, where());
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
    message = std::get<df::Message>(std::move(pulled));
    return PullStatus::Ok;
  }

  void count_pulled() {
    core_.outputs_pulled.fetch_add(1, std::memory_order_relaxed);
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> timing(core_.latency_mu);
    if (!core_.pull_timing_init) {
      core_.first_pull_at = now;
      core_.pull_timing_init = true;
    }
    core_.last_pull_at = now;
  }

  // With SIMA_NEAT_BACKEND set (the [NEAT] lines of DataflowBackendSelect.cpp), each input's and
  // each output's counters as the stop finds them: the per-endpoint report of the Run.
  void trace_report() const {
    if (pipeline_internal::env_str("SIMA_NEAT_BACKEND", "").empty()) {
      return;
    }
    for (const auto& port : ports_) {
      const DataflowInputConfig& input = port->config();
      std::string line = "[NEAT] Run::stop: input " + dataflow_names_text(input.names);
      const std::optional<df::SessionReport> report = port->report();
      if (!report) {
        std::fprintf(stderr, "%s: never bound\n", line.c_str());
        continue;
      }
      if (!report->feeds.empty()) {
        const df::FeedReport& feed = report->feeds.front();
        line += ": pushes " + std::to_string(feed.pushes) + ", push failures " +
                std::to_string(feed.push_failures) + ", backpressure " +
                std::to_string(feed.backpressure);
      }
      for (std::size_t k = 0; k < report->links.size() && k < input.outputs.size(); ++k) {
        const df::LinkReport& link = report->links[k];
        line += "; output " + dataflow_names_text(config_.outputs[input.outputs[k]].names) +
                ": accepted " + std::to_string(link.stats.accepted) + ", dropped oldest " +
                std::to_string(link.stats.dropped_oldest()) + ", dropped incoming " +
                std::to_string(link.stats.dropped_incoming()) + ", pending " +
                std::to_string(link.pending);
      }
      std::fprintf(stderr, "%s\n", line.c_str());
    }
  }

  RunCore& core_;
  DataflowRunConfig config_;
  InputStreamOptions stream_options_;
  pipeline_internal::StreamNames streams_;

  // One per public Input (config_.inputs order); where each public Output is read; the unnamed
  // push's and pull's endpoints when the graph has exactly one.
  std::vector<std::unique_ptr<DataflowInputPort>> ports_;
  std::vector<OutputSlot> slots_;
  DataflowInputPort* default_input_ = nullptr;
  std::size_t default_output_ = kNoEndpoint;

  std::atomic<bool> stopped_{false};
  std::atomic<std::uint64_t> polls_{0};
  // A Run of several inputs: the counter of its pushes, orig_input_seq.
  std::atomic<std::uint64_t> run_seq_{0};

  mutable std::mutex error_mu_;
  std::optional<PullError> last_error_;
};

} // namespace

std::unique_ptr<DataflowRunState> make_dataflow_run_state(RunCore& core, DataflowRunConfig config,
                                                          std::span<const Tensor> seed) {
  auto run = std::make_unique<DataflowRun>(core, std::move(config));
  run->add_ports();
  run->bind_at_build(seed);
  return run;
}

} // namespace simaai::neat::runtime

#endif // SIMANEAT_WITH_DATAFLOW
