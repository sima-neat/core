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
//             false and no drop (unit_run_overflow_policy_parity_test.cpp:129-163). A full link is
//             answered by alloc itself (the runtime's FR14, A3), before any slot or copy, and
//             counted by the runtime as that push: alloc_refused maps it to the same outcome.
//   pull      -1 is kNoDeadline, 0 a passed deadline (a poll), t > 0 now + t ms. The pulled Tensor
//             is CpuExternal over the pool slot, its holder the Message, read-only.
//   close_input  Session::close_input(kNoDeadline): cancels a parked push, keeps what is queued.
//   stop      Session::stop(now + 2 s), once more on Timeout (RunLifecycle.cpp:94-95).
// The copy into the slot runs on the pushing thread, inside push: when push returns, the caller's
// buffer is free (GStreamer retains the tensor and copies later on Core's input thread).
//
// An item is one tensor (a leaf: a TensorSpec port) or several (a record: a RecordSchema port with
// positional field names, H0 plan 3.5). The first item binds which, as it binds the shape.
//
// The TensorList verbs (Run::push(TensorList), pull(milliseconds) -> Result<TensorList>, alloc): a
// pushed list's first tensor's stamp proposes the message's stream, frame_id, source_time and
// flags; every pulled tensor carries the message's Stamp in its envelope. alloc() lends a pool slot
// as a writable Tensor; a blocking push of it alone in a list into this Run publishes the slot
// itself, without a copy (RunAllocation.h).
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
#include "pipeline/runtime/RunAllocation.h"
#include "pipeline/runtime/RunCore.h"

#include <algorithm>
#include <array>
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

// Core's public time and flag values are the runtime's, so a stamp converts field by field.
static_assert(static_cast<std::uint16_t>(ClockId::None) ==
                  static_cast<std::uint16_t>(df::ClockId::None) &&
              static_cast<std::uint16_t>(ClockId::Monotonic) ==
                  static_cast<std::uint16_t>(df::ClockId::Monotonic) &&
              static_cast<std::uint16_t>(ClockId::Tai) ==
                  static_cast<std::uint16_t>(df::ClockId::Tai) &&
              static_cast<std::uint16_t>(ClockId::SensorLocal) ==
                  static_cast<std::uint16_t>(df::ClockId::SensorLocal));
static_assert(static_cast<std::uint16_t>(StampFlags::Discontinuity) ==
                  static_cast<std::uint16_t>(df::StampFlags::Discontinuity) &&
              static_cast<std::uint16_t>(StampFlags::Gap) ==
                  static_cast<std::uint16_t>(df::StampFlags::Gap) &&
              static_cast<std::uint16_t>(StampFlags::EndOfStream) ==
                  static_cast<std::uint16_t>(df::StampFlags::EndOfStream) &&
              static_cast<std::uint16_t>(StampFlags::Degraded) ==
                  static_cast<std::uint16_t>(df::StampFlags::Degraded));

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

// The runtime's stamp for what a push proposes: the stream interned, the author's frame_id,
// source_time (any clock) and flags. emit writes seq and publish_time.
df::Stamp runtime_stamp(const Stamp& proposal, pipeline_internal::StreamNames& streams) {
  df::Stamp out;
  out.stream = df::StreamId{streams.intern(proposal.stream_name)};
  out.frame_id = proposal.frame_id >= 0 ? proposal.frame_id : df::kNoFrameId;
  if (proposal.source_time.valid()) {
    out.source_time = df::TimePoint{proposal.source_time.ns,
                                    static_cast<df::ClockId>(proposal.source_time.clock)};
  }
  out.flags = static_cast<df::StampFlags>(static_cast<std::uint16_t>(proposal.flags));
  return out;
}

// Core's Stamp of a pulled message.
void core_stamp(const df::Stamp& in, const pipeline_internal::StreamNames& streams, Stamp& out) {
  out.stream = in.stream.value;
  out.stream_name = streams.name(in.stream.value);
  out.seq = in.seq;
  out.frame_id = in.frame_id >= 0 ? in.frame_id : -1;
  out.source_time = TimePoint{in.source_time.ns, static_cast<ClockId>(in.source_time.clock)};
  out.publish_time = TimePoint{in.publish_time.ns, static_cast<ClockId>(in.publish_time.clock)};
  out.flags = static_cast<StampFlags>(static_cast<std::uint16_t>(in.flags));
  out.version = Stamp::kVersion;
}

// A CPU tensor's bytes for the copy into a slot, through its storage's own map, with its
// byte_offset applied: no allocation for CPU storage. Tensor::map (map_tensor_storage_raw) also
// wraps the storage and the mapping's guard in one keepalive, so that a Mapping may outlive the
// Tensor; the push copies while its caller holds the Tensor, so the storage's map is enough. Only
// CpuOwned and CpuExternal storage (whatever their map_fn does, a Run::alloc tensor's included);
// any other kind gives an empty Mapping and is read through the Tensor.
Mapping map_cpu_storage_read(const Tensor& tensor) {
  if (!tensor.storage || tensor.byte_offset < 0 ||
      (tensor.storage->kind != StorageKind::CpuOwned &&
       tensor.storage->kind != StorageKind::CpuExternal)) {
    return {};
  }
  // TensorBuffer::map without its keepalive (a copy of the holder, for a Mapping that outlives the
  // storage): this one ends inside the push.
  const TensorBuffer& storage = *tensor.storage;
  Mapping mapping;
  if (storage.map_fn) {
    mapping = storage.map_fn(MapMode::Read);
  } else {
    mapping.data = storage.data;
    mapping.size_bytes = storage.size_bytes;
  }
  const auto offset = static_cast<std::size_t>(tensor.byte_offset);
  if (!mapping.data || offset > mapping.size_bytes) {
    return {};
  }
  mapping.data = static_cast<std::uint8_t*>(mapping.data) + offset;
  mapping.size_bytes -= offset;
  return mapping;
}

// Tensor::copy_dense_bytes_tight_to, without its Mapping's allocation for a contiguous CPU tensor.
bool copy_dense_into(const Tensor& tensor, std::uint8_t* dst, std::size_t dst_size) {
  if (tensor.is_dense() && !tensor.has_tiled_storage() && !tensor.shape.empty() &&
      tensor.is_contiguous()) {
    const std::size_t bytes = tensor.dense_bytes_tight();
    if (bytes != 0U && dst_size >= bytes) {
      const Mapping mapping = map_cpu_storage_read(tensor);
      if (mapping.data && mapping.size_bytes >= bytes) {
        std::memcpy(dst, mapping.data, bytes);
        return true;
      }
    }
  }
  return tensor.copy_dense_bytes_tight_to(dst, dst_size);
}

// One plane of a CPU NV12 or I420 tensor: its rows as Tensor::map_nv12_read / map_i420_read give
// them, read through map_cpu_storage_read. False unless the plane exists with a {rows, cols} shape,
// a row stride of at least `cols` bytes (element stride 1 if given) and every row inside `mapping`.
struct PlaneRows {
  const std::uint8_t* data = nullptr;
  std::int64_t stride = 0;
};

bool cpu_plane_rows(const Tensor& tensor, const Mapping& mapping, PlaneRole role, std::int64_t rows,
                    std::int64_t cols, PlaneRows& out) {
  const Plane* plane = tensor.try_plane(role);
  if (!plane || plane->shape.size() < 2U || plane->shape[0] != rows || plane->shape[1] != cols ||
      plane->byte_offset < 0 ||
      (plane->strides_bytes.size() > 1U && plane->strides_bytes[1] != 1)) {
    return false;
  }
  const std::int64_t stride = plane->strides_bytes.empty() ? cols : plane->strides_bytes[0];
  if (rows <= 0 || cols <= 0 || stride < cols) {
    return false;
  }
  // end = offset + stride * (rows - 1) + cols, without overflow (all terms are positive).
  const auto size = static_cast<std::uint64_t>(mapping.size_bytes);
  const auto offset = static_cast<std::uint64_t>(plane->byte_offset);
  const auto ustride = static_cast<std::uint64_t>(stride);
  const auto span_rows = static_cast<std::uint64_t>(rows - 1);
  if (offset > size || (span_rows != 0U && ustride > (size - offset) / span_rows)) {
    return false;
  }
  const std::uint64_t end = offset + ustride * span_rows + static_cast<std::uint64_t>(cols);
  if (end > size) {
    return false;
  }
  out.data = static_cast<const std::uint8_t*>(mapping.data) + offset;
  out.stride = stride;
  return true;
}

// What validate_public_input and dataflow_binding_mismatch read from a leaf tensor
// (InputStreamUtil.cpp validate_declared_tensor_storage, describe_tensor_geometry_or_throw,
// validate_public_geometry; DataflowGraphCompiler.cpp dataflow_binding_mismatch), plus the rest of
// its image spec. Both are pure functions of these fields and of the Run's InputOptions and
// InputStreamOptions, which never change: a tensor whose geometry equals one that passed both
// passes both. Only CPU storage and no Semantic beyond an image spec are kept, so a tensor with
// anything else is always checked in full.
struct LeafGeometry {
  StorageKind kind = StorageKind::Unknown;
  std::size_t storage_bytes = 0;
  TensorDType dtype = TensorDType::UInt8;
  TensorLayout layout = TensorLayout::Unknown;
  std::int64_t byte_offset = 0;
  Device device{};
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> strides_bytes;
  std::vector<TensorAxisSemantic> axis_semantics;
  std::vector<Plane> planes;
  std::optional<ImageSpec> image;

  static bool keepable(const Tensor& t) noexcept {
    const Semantic& s = t.semantic;
    return t.storage &&
           (t.storage->kind == StorageKind::CpuOwned ||
            t.storage->kind == StorageKind::CpuExternal) &&
           !s.audio && !s.tokens && !s.text && !s.byte_stream && !s.tess && !s.encoded &&
           !s.quant && !s.detection && !s.feature && !s.preprocess;
  }

  static LeafGeometry of(const Tensor& t) {
    LeafGeometry g;
    g.kind = t.storage->kind;
    g.storage_bytes = t.storage->size_bytes;
    g.dtype = t.dtype;
    g.layout = t.layout;
    g.byte_offset = t.byte_offset;
    g.device = t.device;
    g.shape = t.shape;
    g.strides_bytes = t.strides_bytes;
    g.axis_semantics = t.axis_semantics;
    g.planes = t.planes;
    g.image = t.semantic.image;
    return g;
  }

  bool matches(const Tensor& t) const noexcept {
    if (!keepable(t) || t.storage->kind != kind || t.storage->size_bytes != storage_bytes ||
        t.dtype != dtype || t.layout != layout || t.byte_offset != byte_offset ||
        t.device.type != device.type || t.device.id != device.id || t.shape != shape ||
        t.strides_bytes != strides_bytes || t.axis_semantics != axis_semantics ||
        t.planes.size() != planes.size() || t.semantic.image.has_value() != image.has_value()) {
      return false;
    }
    if (image && (t.semantic.image->format != image->format ||
                  t.semantic.image->color_space != image->color_space)) {
      return false;
    }
    for (std::size_t i = 0; i < planes.size(); ++i) {
      const Plane& a = t.planes[i];
      const Plane& b = planes[i];
      if (a.role != b.role || a.byte_offset != b.byte_offset || a.shape != b.shape ||
          a.strides_bytes != b.strides_bytes) {
        return false;
      }
    }
    return true;
  }
};

df::Extent extent_of(const std::vector<std::int64_t>& shape) {
  df::Extent extent;
  extent.rank = static_cast<std::uint8_t>(shape.size());
  for (std::size_t i = 0; i < shape.size() && i < df::kMaxRank; ++i) {
    extent.dims[i] = shape[i];
  }
  return extent;
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

// What the tensors of one pulled message keep alive, in one allocation: the message (the lease on
// its slot) and the envelope every tensor of it shares (its Stamp). The tensors' storage holder
// and envelope are aliases into it; it holds no Tensor, so nothing points back at it.
struct PulledMessage {
  explicit PulledMessage(df::Message m) : message(std::move(m)) {}
  df::Message message;
  TensorEnvelope envelope;
};

// The backing of a Run::alloc Tensor on this backend: the loan of one slot of the input's pool.
struct SlotLoan {
  df::WriteLoan loan;
};

class DataflowRun final : public DataflowRunState {
public:
  DataflowRun(RunCore& core, DataflowRunConfig config)
      : core_(core), config_(std::move(config)), streams_(config_.default_stream) {
    stream_options_.max_input_bytes = config_.run.advanced.max_input_bytes;
  }

  // The Session is destroyed with the Run state, after RunCore::close ran stop (A2: a destructor
  // after a stop that timed out only polls). Messages still held by pulled tensors, and slots
  // still lent by alloc(), keep their pool alive on the runtime's retirement list.
  ~DataflowRun() override = default;

  // A seed, or a declaration that gives the spec, binds now; anything else at the first push.
  void bind_at_build(const TensorList& seed) {
    std::string why;
    std::optional<DataflowInputBinding> binding =
        seed.size() > 1U
            ? bind_dataflow_record(config_.input, seed, &why)
            : bind_dataflow_input(config_.input, seed.empty() ? nullptr : &seed.front(), &why);
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
    if (inputs.empty()) {
      throw_input_shape(std::string(where) + ": an empty tensor list");
    }
    // The first tensor's stamp proposes the item's identity; a tensor without a stamp proposes
    // nothing (the default stream, no frame_id), as develop's TensorList push.
    const Stamp& proposal = inputs.front().stamp();
    if (std::optional<TensorList> fields = pipeline_internal::list_record_fields(inputs, where)) {
      return push_item(*fields, proposal, block, where);
    }
    return push_item(inputs, proposal, block, where, /*commit_allocation=*/block);
  }

  bool push(std::optional<std::string_view> input, const Sample& msgs, bool block,
            const char* where) override {
    check_input_name(input, where);
    enforce_transferable(msgs, where);
    if (msgs.kind == SampleKind::Bundle) {
      // Core pushes a Bundle as one item per field, stopping at the first refusal
      // (RunPush.cpp:1082-1086).
      for (const Sample& field : msgs.fields) {
        if (!push_sample_item(field, block, where)) {
          return false;
        }
      }
      return true;
    }
    return push_sample_item(msgs, block, where);
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
    const Tensor tensor = tensor_from_cv_mat(inputs.front(), view_options, where);
    return push_item(std::span<const Tensor>(&tensor, 1), Stamp{}, block, where);
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
    Bound* bound = nullptr;
    const PullStatus status = take(output, timeout_ms, err, message, bound);
    if (status != PullStatus::Ok) {
      return status;
    }
    // The caller's list buffer is reused: Run::pull(int, Sample&, PullError*) with one Sample
    // allocates no list per pull.
    TensorList tensors = std::move(out.tensors);
    tensors.clear();
    out = make_sample(bound->binding, std::move(message), std::move(tensors));
    count_pulled();
    return PullStatus::Ok;
  }

  PullStatus pull(std::optional<std::string_view> output, int timeout_ms, TensorList& out,
                  PullError* err) override {
    df::Message message;
    Bound* bound = nullptr;
    const PullStatus status = take(output, timeout_ms, err, message, bound);
    if (status != PullStatus::Ok) {
      return status;
    }
    fill_tensors(bound->binding, std::move(message), out);
    count_pulled();
    return PullStatus::Ok;
  }

  Tensor alloc(std::string_view input, const char* where) override {
    const std::string at = std::string(where) + "(\"" + std::string(input) + "\")";
    if (std::find(config_.input_names.begin(), config_.input_names.end(), input) ==
        config_.input_names.end()) {
      throw std::runtime_error(error_util::decorate_error(
          error_codes::kRuntimePull,
          at + ": unknown input. Available inputs: " + names_text(config_.input_names)));
    }
    const std::vector<std::int64_t> shape = declared_alloc_shape(config_.input, at);
    Bound* bound = bound_.load(std::memory_order_acquire);
    if (!bound || stopped_.load(std::memory_order_acquire) ||
        input_closed_.load(std::memory_order_acquire)) {
      // A declared shape binds at build, so no binding means a closed input or a stopped Run.
      throw std::runtime_error(error_util::decorate_error(
          error_codes::kRuntimePull, at + ": the input is closed or the Run is stopped"));
    }
    df::Result<df::WriteLoan> allocated = bound->session.alloc(bound->feed, df::Acquire::Block);
    if (const df::Error* error = df::failure(allocated)) {
      if (error->code == df::ErrorCode::Closed) {
        throw std::runtime_error(error_util::decorate_error(
            error_codes::kRuntimePull, at + ": the input is closed or the Run is stopped"));
      }
      pipeline_internal::throw_dataflow_error(*error, at);
    }
    auto loan = std::make_shared<SlotLoan>();
    loan->loan = std::get<df::WriteLoan>(std::move(allocated));
    const std::span<std::byte> bytes = loan->loan.bytes();
    auto allocation = std::make_shared<RunAllocation>();
    allocation->run = core_.weak_from_this();
    allocation->data = bytes.data();
    allocation->size = bytes.size();
    allocation->backing = std::move(loan);
    return make_allocated_tensor(allocation, bound->binding.dtype, shape);
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

  // One item of a Sample push: its tensors (one, or several as a record) and its identity.
  bool push_sample_item(const Sample& item, bool block, const char* where) {
    const Stamp proposal = pipeline_internal::stamp_from_sample(item);
    if (sample_has_tensor_list(item)) {
      return push_item(item.tensors, proposal, block, where);
    }
    if (item.tensor.has_value()) {
      return push_item(std::span<const Tensor>(&*item.tensor, 1), proposal, block, where);
    }
    throw_input_shape(std::string(where) + ": the Sample carries no tensor");
  }

  // One item: Core's checks, the binding (the first item binds it), then the slot. With
  // `commit_allocation` (the blocking TensorList push only), a Tensor from this Run's alloc() alone
  // in the list is published as its slot; every other verb copies it like any CPU tensor, as the
  // GStreamer backend does.
  bool push_item(std::span<const Tensor> tensors, const Stamp& proposal, bool block,
                 const char* where, bool commit_allocation = false) {
    // Core's own push-time checks and texts first (RunPush.cpp:655, InputStreamUtil.cpp:2146-2190):
    // the declaration describes the item's representative (first) tensor, as on GStreamer.
    if (tensors.size() > 1U && !config_.input.shape.empty()) {
      throw_input_shape(std::string(where) + ": an item of " + std::to_string(tensors.size()) +
                        " tensors where one tensor of a declared shape is expected");
    }
    // A Tensor from this Run's alloc() that was pushed already has an empty storage: say so first.
    std::shared_ptr<RunAllocation> allocation =
        commit_allocation && tensors.size() == 1U ? allocation_of(tensors.front(), core_) : nullptr;
    if (allocation) {
      require_pushable(*allocation, where);
    }
    // A leaf with the geometry of the first leaf that passed every check below passes them again
    // (LeafGeometry): they are skipped for it.
    const LeafGeometry* known =
        tensors.size() == 1U ? validated_leaf_.load(std::memory_order_acquire) : nullptr;
    const bool validated = known && known->matches(tensors.front());
    if (!validated) {
      validate_public_input(tensors.front(), config_.input, stream_options_, where);
      for (const Tensor& tensor : tensors) {
        if (tensor.storage && tensor.storage->kind == StorageKind::GstSample) {
          enforce_transferable(sample_from_tensors(TensorList{tensor}), where);
        }
      }
    }
    Bound* bound = bound_.load(std::memory_order_acquire);
    if (!bound) {
      if (stopped_.load(std::memory_order_acquire) ||
          input_closed_.load(std::memory_order_acquire)) {
        return false;
      }
      bound = bind_first(tensors, where);
      if (!bound) {
        return false;
      }
    }
    const df::Stamp stamp = runtime_stamp(proposal, streams_);
    if (tensors.size() > 1U || bound->binding.is_record()) {
      if (const std::string why = dataflow_record_mismatch(bound->binding, tensors); !why.empty()) {
        throw_input_shape(std::string(where) + ": " + why);
      }
      return push_record(*bound, tensors, stamp, block, where);
    }
    const Tensor& tensor = tensors.front();
    if (!validated) {
      if (const std::string why = dataflow_binding_mismatch(bound->binding, tensor); !why.empty()) {
        throw_input_shape(std::string(where) + ": " + why);
      }
      remember_validated_leaf(tensor);
    }
    if (allocation) {
      return push_allocated(*bound, tensor, *allocation, stamp, block, where);
    }
    df::Result<df::WriteLoan> allocated =
        bound->session.alloc(bound->feed, block ? df::Acquire::Block : df::Acquire::Fail);
    if (const df::Error* error = df::failure(allocated)) {
      return alloc_refused(*error, block, where);
    }
    df::WriteLoan loan = std::get<df::WriteLoan>(std::move(allocated));
    const df::Extent extent = copy_into(bound->binding, tensor, loan.bytes(), where);
    return settle(publish(*bound, std::move(loan), stamp, extent, block), block, where);
  }

  // The first leaf that passed validate_public_input and the binding check, kept once (later
  // pushes read it without a lock).
  void remember_validated_leaf(const Tensor& tensor) {
    if (validated_leaf_.load(std::memory_order_acquire) || !LeafGeometry::keepable(tensor)) {
      return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (!validated_leaf_storage_) {
      validated_leaf_storage_ = std::make_unique<LeafGeometry>(LeafGeometry::of(tensor));
      validated_leaf_.store(validated_leaf_storage_.get(), std::memory_order_release);
    }
  }

  // A record: every field copied at its offset in one slot, one Extent per field.
  bool push_record(Bound& bound, std::span<const Tensor> fields, const df::Stamp& stamp, bool block,
                   const char* where) {
    df::Result<df::WriteLoan> allocated =
        bound.session.alloc(bound.feed, block ? df::Acquire::Block : df::Acquire::Fail);
    if (const df::Error* error = df::failure(allocated)) {
      return alloc_refused(*error, block, where);
    }
    df::WriteLoan loan = std::get<df::WriteLoan>(std::move(allocated));
    const std::span<std::byte> slot = loan.bytes();
    auto* dst = reinterpret_cast<std::uint8_t*>(slot.data());
    std::array<df::Extent, df::kMaxFields> extents{};
    for (std::size_t i = 0; i < fields.size(); ++i) {
      const DataflowFieldBinding& field = bound.binding.fields[i];
      extents[i] = extent_of(fields[i].shape);
      if (field.offset > slot.size() ||
          !copy_dense_into(fields[i], dst + field.offset, slot.size() - field.offset)) {
        throw_input_shape(std::string(where) + ": the bytes of field " + std::to_string(i) +
                          " could not be read for the copy (Dataflow backend)");
      }
    }
    const std::span<const df::Extent> extent_span(extents.data(), fields.size());
    df::Result<df::PublishResult> pushed =
        !block ? bound.session.push(bound.feed, std::move(loan), stamp, extent_span, df::Deadline{})
        : config_.terminal_drop
            ? bound.session.push(bound.feed, std::move(loan), stamp, extent_span, df::kNoDeadline)
            : bound.session.push_waiting(bound.feed, std::move(loan), stamp, extent_span,
                                         df::kNoDeadline);
    return settle(pushed, block, where);
  }

  // A Tensor from this Run's alloc(): its slot is published as it is, without a copy. The loan
  // goes back to the allocation when the Session refuses it with the loan intact (SR1, FR2, FR3,
  // FR5: stopped, no room, input closed), so the application may push it again or drop it.
  bool push_allocated(Bound& bound, const Tensor& tensor, RunAllocation& allocation,
                      const df::Stamp& stamp, bool block, const char* where) {
    begin_allocated_push(allocation, where);
    auto& box = *static_cast<SlotLoan*>(allocation.backing.get());
    df::WriteLoan loan = std::move(box.loan);
    const auto settle_allocation = [&] {
      if (loan) {
        box.loan = std::move(loan);
      }
      end_allocated_push(allocation, !box.loan);
    };
    df::Result<df::PublishResult> pushed;
    try {
      if (!loan || !tensor.is_dense() || tensor.byte_offset != 0 || !tensor.is_contiguous()) {
        throw_input_shape(std::string(where) +
                          ": a Tensor from Run::alloc must stay dense, contiguous and at byte "
                          "offset 0 to be pushed");
      }
      pushed = publish(bound, std::move(loan), stamp, extent_of(tensor.shape), block);
    } catch (...) {
      settle_allocation();
      throw;
    }
    settle_allocation();
    return settle(pushed, block, where);
  }

  df::Result<df::PublishResult> publish(Bound& bound, df::WriteLoan&& loan, const df::Stamp& stamp,
                                        const df::Extent& extent, bool block) {
    if (!block) {
      return bound.session.push(bound.feed, std::move(loan), stamp, extent, df::Deadline{});
    }
    if (config_.terminal_drop) {
      return bound.session.push(bound.feed, std::move(loan), stamp, extent, df::kNoDeadline);
    }
    return bound.session.push_waiting(bound.feed, std::move(loan), stamp, extent, df::kNoDeadline);
  }

  bool settle(const df::Result<df::PublishResult>& pushed, bool block, const char* where) {
    if (const df::Error* error = df::failure(pushed)) {
      return refused(*error, block, where);
    }
    return published(std::get<df::PublishResult>(pushed));
  }

  // The first push binds the input: its tensors' spec, under the declaration (WORKER.md 18.1).
  Bound* bind_first(std::span<const Tensor> tensors, const char* where) {
    std::lock_guard<std::mutex> lock(mu_);
    if (Bound* bound = bound_.load(std::memory_order_acquire)) {
      return bound;
    }
    if (stopped_.load(std::memory_order_relaxed) || input_closed_.load(std::memory_order_relaxed)) {
      return nullptr;
    }
    std::string why;
    std::optional<DataflowInputBinding> binding =
        tensors.size() > 1U ? bind_dataflow_record(config_.input, tensors, &why)
                            : bind_dataflow_input(config_.input, &tensors.front(), &why);
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

  // The next message of the output, or why there is none (both pull verbs). The call's name for
  // messages is built only when a message needs it.
  PullStatus take(std::optional<std::string_view> output, int timeout_ms, PullError* err,
                  df::Message& message, Bound*& bound) {
    clear_pull_error(err);
    const auto where = [&output] {
      return output ? "Run::pull(\"" + std::string(*output) + "\")" : std::string("Run::pull");
    };
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
            where() + ": unknown output. Available outputs: " + names_text(config_.output_names));
        return PullStatus::Error;
      }
    }
    polls_.fetch_add(1, std::memory_order_relaxed);
    const df::Deadline deadline = deadline_after(timeout_ms);
    bound = wait_bound(deadline);
    if (!bound) {
      // Nothing was pushed yet: a pull waits for the first push in Core (WORKER.md 18.1).
      if (stopped_.load(std::memory_order_acquire) ||
          input_closed_.load(std::memory_order_acquire)) {
        error_util::set_pull_error(err, error_codes::kRuntimePull, where() + ": stream closed");
        return PullStatus::Closed;
      }
      return timed_out(timeout_ms, where(), err);
    }

    df::Result<df::Message> pulled = bound->session.pull(bound->fetch, deadline);
    if (const df::Error* error = df::failure(pulled)) {
      const pipeline_internal::DataflowErrorMapping mapping =
          pipeline_internal::map_dataflow_error(*error);
      if (mapping.status == PullStatus::Timeout) {
        return timed_out(timeout_ms, where(), err);
      }
      if (mapping.status == PullStatus::Closed) {
        error_util::set_pull_error(err, error_codes::kRuntimePull, where() + ": stream closed");
        // Closed and drained after close_input: the Run stops itself, as the GStreamer Run does
        // at its end of stream (RunPull.cpp:798-810).
        if (input_closed_.load(std::memory_order_acquire) &&
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
      if (!copy_dense_into(tensor, dst, slot.size())) {
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
    // A CPU tensor's planes are copied without a Mapping allocation (copy_cpu_planes); any other
    // tensor is mapped through Tensor::map_nv12_read / map_i420_read.
    if (!copy_cpu_planes(binding, tensor, copy_rows, h, w)) {
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
    }
    extent.rank = 3;
    extent.dims[0] = h;
    extent.dims[1] = w;
    extent.dims[2] = 3;
    return extent;
  }

  // The planes of a CPU NV12 or I420 tensor into the slot (copy_rows), at the geometry
  // map_nv12_read / map_i420_read give a well-formed one (the Y plane's {h, w}, both even), read
  // through map_cpu_storage_read. False, with nothing copied, for any other tensor: the caller
  // then maps it through the Tensor, which reports what is wrong with it.
  template <class CopyRows>
  static bool copy_cpu_planes(const DataflowInputBinding& binding, const Tensor& tensor,
                              const CopyRows& copy_rows, std::int64_t& h, std::int64_t& w) {
    const bool nv12 = binding.planes == DataflowInputBinding::Planes::Nv12;
    if (tensor.dtype != TensorDType::UInt8 || !(nv12 ? tensor.is_nv12() : tensor.is_i420())) {
      return false;
    }
    const Plane* y_plane = tensor.try_plane(PlaneRole::Y);
    if (!y_plane || y_plane->shape.size() < 2U) {
      return false;
    }
    const std::int64_t rows = y_plane->shape[0];
    const std::int64_t cols = y_plane->shape[1];
    if (rows <= 0 || cols <= 0 || rows % 2 != 0 || cols % 2 != 0) {
      return false;
    }
    const Mapping mapping = map_cpu_storage_read(tensor);
    if (!mapping.data) {
      return false;
    }
    std::array<PlaneRows, 3> planes{};
    if (!cpu_plane_rows(tensor, mapping, PlaneRole::Y, rows, cols, planes[0])) {
      return false;
    }
    if (nv12) {
      if (!cpu_plane_rows(tensor, mapping, PlaneRole::UV, rows / 2, cols, planes[1])) {
        return false;
      }
      copy_rows(0, planes[0].data, planes[0].stride, rows, cols);
      copy_rows(1, planes[1].data, planes[1].stride, rows / 2, cols);
    } else {
      if (!cpu_plane_rows(tensor, mapping, PlaneRole::U, rows / 2, cols / 2, planes[1]) ||
          !cpu_plane_rows(tensor, mapping, PlaneRole::V, rows / 2, cols / 2, planes[2])) {
        return false;
      }
      copy_rows(0, planes[0].data, planes[0].stride, rows, cols);
      copy_rows(1, planes[1].data, planes[1].stride, rows / 2, cols / 2);
      copy_rows(2, planes[2].data, planes[2].stride, rows / 2, cols / 2);
    }
    h = rows;
    w = cols;
    return true;
  }

  // The slot the Session refused. Under Acquire::Fail (try_push), Backpressure is the runtime's
  // FR14: every link of the Feed full, so the push would be refused, and alloc refused it before
  // any slot or copy, counted by the runtime as that push counts it (a full Block link's FR2, a
  // DropIncoming link's drop). Core does what it does when the push refused: false, and on
  // DropIncoming its input drop (published()'s DroppedIncoming). KeepLatest and the terminal drop
  // are DropOldest links, which never refuse. Every other refusal, FR7 included, is refused()'s.
  bool alloc_refused(const df::Error& error, bool block, const char* where) {
    if (!block && error.code == df::ErrorCode::Backpressure) {
      if (!config_.terminal_drop && config_.link_policy == OverflowPolicy::DropIncoming) {
        core_.inputs_dropped.fetch_add(1, std::memory_order_relaxed);
      }
      return false;
    }
    return refused(error, block, where);
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

  // A dense tensor over `bytes` of a pulled message, with the Core metadata the binding kept,
  // written into `tensor` (a default Tensor, built where it is returned: a Tensor is about 1 KB,
  // and building it elsewhere and moving it costs as much as building it).
  static void fill_dense(Tensor& tensor, TensorDType dtype, TensorLayout layout,
                         const std::vector<TensorAxisSemantic>& axis_semantics,
                         const Semantic& semantic, std::span<const std::byte> bytes,
                         const df::Extent& extent, std::shared_ptr<void> holder) {
    tensor.storage = make_cpu_external_storage(const_cast<std::byte*>(bytes.data()), bytes.size(),
                                               std::move(holder), true);
    tensor.dtype = dtype;
    tensor.layout = layout;
    tensor.axis_semantics = axis_semantics;
    tensor.device = {DeviceType::CPU, 0};
    tensor.semantic = semantic;
    tensor.read_only = true;
    tensor.shape.assign(extent.dims.begin(), extent.dims.begin() + extent.rank);
  }

  static Tensor dense_tensor(TensorDType dtype, TensorLayout layout,
                             const std::vector<TensorAxisSemantic>& axis_semantics,
                             const Semantic& semantic, std::span<const std::byte> bytes,
                             const df::Extent& extent, std::shared_ptr<void> holder) {
    Tensor tensor;
    fill_dense(tensor, dtype, layout, axis_semantics, semantic, bytes, extent, std::move(holder));
    return tensor;
  }

  // The leaf tensor of a pulled message (dense or planar), over its slot (`holder` keeps it),
  // written into `tensor` (a default Tensor).
  static void fill_leaf(Tensor& tensor, const DataflowInputBinding& binding,
                        std::span<const std::byte> bytes, const df::Extent& extent,
                        std::shared_ptr<void> holder) {
    if (binding.planes == DataflowInputBinding::Planes::None) {
      fill_dense(tensor, binding.dtype, binding.layout, binding.axis_semantics, binding.semantic,
                 bytes, extent, std::move(holder));
      return;
    }
    tensor.storage = make_cpu_external_storage(const_cast<std::byte*>(bytes.data()), bytes.size(),
                                               std::move(holder), true);
    tensor.dtype = binding.dtype;
    tensor.axis_semantics = binding.axis_semantics;
    tensor.device = {DeviceType::CPU, 0};
    tensor.semantic = binding.semantic;
    tensor.read_only = true;
    const auto& planar =
        std::get<df::Planar>(std::get<df::TensorSpec>(binding.spec.payload).layout);
    const std::int64_t h = extent.dims[0];
    const std::int64_t w = extent.dims[1];
    tensor.shape = {h, w};
    tensor.layout = TensorLayout::HW;
    tensor.strides_bytes = {planar.plane[0].stride_bytes, 1};
    const auto plane = [&](PlaneRole role, std::size_t p, std::int64_t rows, std::int64_t cols) {
      Plane& out = tensor.planes.emplace_back();
      out.role = role;
      out.shape = {rows, cols};
      out.strides_bytes = {planar.plane[p].stride_bytes, 1};
      out.byte_offset = planar.plane[p].offset_bytes;
    };
    tensor.planes.reserve(binding.planes == DataflowInputBinding::Planes::Nv12 ? 2U : 3U);
    plane(PlaneRole::Y, 0, h, w);
    if (binding.planes == DataflowInputBinding::Planes::Nv12) {
      plane(PlaneRole::UV, 1, h / 2, w);
    } else {
      plane(PlaneRole::U, 1, h / 2, w / 2);
      plane(PlaneRole::V, 2, h / 2, w / 2);
    }
  }

  // An independent CPU copy of `bytes` (OutputMemory::Owned): the slot returns to the pool when
  // the message goes.
  static std::shared_ptr<TensorBuffer> owned_copy(std::span<const std::byte> bytes) {
    auto storage = make_cpu_owned_storage(bytes.size());
    if (!bytes.empty()) {
      std::memcpy(storage->data, bytes.data(), bytes.size());
    }
    return storage;
  }

  // The tensors of a record message, in schema order, each over its field of the slot.
  std::vector<Tensor> record_fields(const DataflowInputBinding& binding, const df::Message& message,
                                    const std::shared_ptr<void>& holder, bool owned) const {
    const df::RecordSchema& schema = *message.schema();
    const df::Bundle bundle(message);
    std::vector<Tensor> fields;
    fields.reserve(binding.fields.size());
    for (std::size_t i = 0; i < binding.fields.size(); ++i) {
      const DataflowFieldBinding& meta = binding.fields[i];
      const df::FieldView view =
          bundle.field(meta.offset, meta.extent_offset, schema.fields[i].spec);
      Tensor field = dense_tensor(meta.dtype, meta.layout, meta.axis_semantics, meta.semantic,
                                  view.bytes, *view.extent, owned ? nullptr : holder);
      if (owned) {
        field.storage = owned_copy(view.bytes);
      }
      fields.push_back(std::move(field));
    }
    return fields;
  }

  // The pulled Sample, shaped as the GStreamer Input -> Output Run returns it
  // (InputStreamPull.cpp:830-884): a TensorSet of the item's tensors, media type and payload tag
  // from the first tensor, identity from the stamp (SampleProjection). `tensors` (empty) lends its
  // buffer to the Sample's list.
  Sample make_sample(const DataflowInputBinding& binding, df::Message message,
                     TensorList tensors) const {
    Stamp stamp;
    core_stamp(message.stamp(), streams_, stamp);
    const bool owned = config_.run.output_memory == OutputMemory::Owned;
    Sample out;
    out.kind = SampleKind::TensorSet;
    out.owned = owned;
    out.tensors = std::move(tensors);
    const Semantic& first =
        binding.is_record() ? binding.fields.front().semantic : binding.semantic;
    const bool image = first.image.has_value();
    out.media_type = image ? "video/x-raw" : "application/vnd.simaai.tensor";
    out.payload_type = image ? PayloadType::Image : PayloadType::Tensor;
    if (image) {
      out.payload_tag = pipeline_internal::pixel_format_token(first.image->format);
      out.format = out.payload_tag;
    }
    if (binding.is_record()) {
      auto holder = owned ? nullptr : std::make_shared<df::Message>(message);
      out.tensors = record_fields(binding, message, holder, owned);
    } else {
      const std::span<const std::byte> bytes = message.bytes();
      const df::Extent extent = message.extent();
      Tensor& tensor = out.tensors.emplace_back();
      if (owned) {
        // OutputMemory::Owned: an independent CPU copy; the slot returns to the pool now.
        fill_leaf(tensor, binding, bytes, extent, nullptr);
        tensor.storage = owned_copy(bytes);
      } else {
        // A lease on the slot: the Message is the holder, released with the last copy of the
        // tensor.
        auto holder = std::make_shared<df::Message>(std::move(message));
        const std::span<const std::byte> leased = holder->bytes();
        fill_leaf(tensor, binding, leased, extent, std::move(holder));
      }
    }
    pipeline_internal::apply_pulled_stamp(stamp, out);
    return out;
  }

  // The pulled tensors (Run::pull(milliseconds)) into `out`: one for a leaf, the fields in schema
  // order for a record, every one carrying the message's Stamp. One allocation holds the message
  // and the envelope they share; the list (unless `out` has room), and each tensor's storage and
  // shape, are the only others. A leaf is built in its place in the list.
  void fill_tensors(const DataflowInputBinding& binding, df::Message message,
                    TensorList& out) const {
    const bool owned = config_.run.output_memory == OutputMemory::Owned;
    auto pulled = std::make_shared<PulledMessage>(std::move(message));
    core_stamp(pulled->message.stamp(), streams_, pulled->envelope.stamp);
    std::shared_ptr<const TensorEnvelope> envelope(pulled, &pulled->envelope);
    if (binding.is_record()) {
      std::shared_ptr<void> holder(pulled, &pulled->message);
      TensorList fields = record_fields(binding, pulled->message, holder, owned);
      for (Tensor& field : fields) {
        field.envelope = envelope;
      }
      if (owned) {
        pulled->message = df::Message{}; // the copies are made: the slot returns to the pool
      }
      out = std::move(fields);
      return;
    }
    const std::span<const std::byte> bytes = pulled->message.bytes();
    const df::Extent extent = pulled->message.extent();
    out.clear();
    Tensor& leaf = out.emplace_back();
    if (owned) {
      fill_leaf(leaf, binding, bytes, extent, nullptr);
      leaf.storage = owned_copy(bytes);
      pulled->message = df::Message{};
    } else {
      df::Message* const leased = &pulled->message;
      fill_leaf(leaf, binding, bytes, extent, std::shared_ptr<void>(std::move(pulled), leased));
    }
    leaf.envelope = std::move(envelope);
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

  // The validated leaf geometry: written once under mu_, read lock-free (release/acquire).
  std::unique_ptr<LeafGeometry> validated_leaf_storage_;
  std::atomic<const LeafGeometry*> validated_leaf_{nullptr};

  mutable std::mutex error_mu_;
  std::optional<PullError> last_error_;
};

} // namespace

std::unique_ptr<DataflowRunState> make_dataflow_run_state(RunCore& core, DataflowRunConfig config,
                                                          const TensorList& seed) {
  auto run = std::make_unique<DataflowRun>(core, std::move(config));
  run->bind_at_build(seed);
  return run;
}

} // namespace simaai::neat::runtime

#endif // SIMANEAT_WITH_DATAFLOW
