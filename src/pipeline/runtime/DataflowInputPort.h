#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// One public Input of a Run on the Dataflow backend: its Session, with one Feed and one
// link and one Fetch per Output the input feeds, bound once by the input's declaration, the seed
// or its first push, with its own mutex and condition variable (a bind, a close or a stop of one
// input wakes that input's waiting pulls only). Every push of the input goes through one funnel:
// take a slot, fill it (a copy, a wrap of the tensor's own storage, or the slot Run::alloc lent),
// publish it. The Run (DataflowRun.cpp) routes endpoint names to ports and maps what a port
// answers to Core's return value and counters.

#include "pipeline/Run.h"
#include "pipeline/internal/DataflowRuntime.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/SampleProjection.h"
#include "pipeline/runtime/DataflowDeviceMemory.h"
#include "pipeline/runtime/DataflowInputBinding.h"
#include "pipeline/runtime/DataflowRun.h"
#include "pipeline/runtime/DataflowTensorCodec.h"
#include "pipeline/runtime/RunAllocation.h"
#include "pipeline/runtime/RunCore.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace simaai::neat::runtime {

#pragma GCC visibility push(hidden)

namespace df = pipeline_internal::df;

// The times of a push that reached the publish, for the Run's input statistics
// (InputStreamStats): its slot (Session::alloc, waiting for a free one included), its bytes (the
// copy into the slot; nothing for a wrap), and the publish itself (Session::push or push_waiting,
// waiting for room on the links included). Nanoseconds.
struct DataflowPushTiming {
  std::int64_t alloc_ns = 0;
  std::int64_t copy_ns = 0;
  std::int64_t publish_ns = 0;
};

// What the Session answered a push: where it ended, and the runtime's own answer.
struct DataflowPushOutcome {
  enum class Stage : std::uint8_t {
    Unbound, ///< no binding, and the input is closed or the Run stopped: nothing was asked
    Alloc,   ///< Session::alloc refused the slot (`answer` holds the error)
    Publish, ///< Session::push or push_waiting answered (`answer`)
  };
  Stage stage = Stage::Unbound;
  df::Result<df::PublishResult> answer;
  DataflowPushTiming timing; ///< Stage::Publish only
};

// A model route's publish instants, for the latency of its outputs (RunStats, MeasureReport): the
// steady_clock instant at which each item entering the route was published, kept by the item's
// ordinal among the items the route takes. That ordinal is the seq the route's last operator
// stamps on the item's output: each operator publishes one output per item, in order, and every
// link behind the Feed's waits for room instead of dropping. An item the Feed's link refuses
// (DropIncoming, or a try_push refused before its slot) takes no ordinal; one it evicts
// (DropOldest, KeepLatest's try_push) gives its ordinal back to the items behind it.
// The pushing thread writes an item's instant before the publish (`claim`), so that the pull of
// its output always finds one, and the instant the publish returned after it (`settle`), which is
// the one the pull reads unless the route outran the pushing thread. With several threads pushing
// into one input at once an output may read its neighbour's instant, or none.
class DataflowPublishTimes {
public:
  // More items than a route holds between their publish and the pull of their outputs.
  static constexpr std::uint64_t kSize = 1024;

  // Before the publish: `ns` kept for the next ordinal.
  void claim(std::int64_t ns) noexcept {
    put(entered_.load(std::memory_order_relaxed) + 1U, ns);
  }
  // After the publish of the item `claim` was asked for, at `ns`: whether the item entered the
  // route, and whether its link evicted the oldest of the `depth` items it held.
  void settle(std::int64_t ns, bool entered, bool evicted, std::uint32_t depth) noexcept {
    if (!entered) {
      return; // the next claim takes the ordinal again
    }
    const std::uint64_t taken = entered_.fetch_add(1, std::memory_order_relaxed) + 1U;
    put(taken, ns);
    if (evicted && taken > depth) {
      // The evicted item never reaches the route: each item queued behind it takes the ordinal
      // before its own.
      for (std::uint64_t k = taken - depth; k < taken; ++k) {
        put(k, at_slot(k + 1U).ns.load(std::memory_order_relaxed));
      }
      entered_.fetch_sub(1, std::memory_order_relaxed);
    }
  }
  // The publish instant of the item whose output the route's last operator stamped `seq`.
  std::optional<std::int64_t> at(std::uint64_t seq) const noexcept {
    const Entry& entry = at_slot(seq);
    if (seq == 0U || entry.ordinal.load(std::memory_order_relaxed) != seq) {
      return std::nullopt;
    }
    return entry.ns.load(std::memory_order_relaxed);
  }

private:
  struct Entry {
    std::atomic<std::uint64_t> ordinal{0};
    std::atomic<std::int64_t> ns{0};
  };
  Entry& at_slot(std::uint64_t ordinal) noexcept {
    return ring_[ordinal & (kSize - 1U)];
  }
  const Entry& at_slot(std::uint64_t ordinal) const noexcept {
    return ring_[ordinal & (kSize - 1U)];
  }
  void put(std::uint64_t ordinal, std::int64_t ns) noexcept {
    Entry& entry = at_slot(ordinal);
    entry.ns.store(ns, std::memory_order_relaxed);
    entry.ordinal.store(ordinal, std::memory_order_relaxed);
  }
  std::array<Entry, kSize> ring_{};
  std::atomic<std::uint64_t> entered_{0};
};

// steady_clock's instant, as the nanoseconds DataflowPublishTimes keeps.
inline std::int64_t dataflow_now_ns() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// One item of a public push, as the Run hands it to its input: what DataflowInputPort::push,
// dataflow_device::push and dataflow_device::push_frame take.
struct DataflowPushItem {
  std::span<const Tensor> tensors; ///< one tensor (a leaf) or several (a record's fields)
  /// The first tensor's stamp, or a Sample's: proposes the message's stream, frame_id,
  /// source_time and flags.
  const Stamp& proposal;
  const char* where; ///< the call, for messages
  bool block;        ///< push: waits for room and never drops; false: try_push
  /// The blocking TensorList push: a Tensor from this Run's alloc() alone in the item is
  /// published as its slot. Every other verb copies it like any CPU tensor, as GStreamer does.
  bool commit_allocation = false;
  /// False: copied even when it could be wrapped (a cv::Mat over memory it does not own; on a
  /// device input an image cv::Mat, which GStreamer's ingress copies unasked).
  bool may_wrap = true;
  /// The item's own orig_input_seq (a Sample push), -1 for none: read in a Run of several
  /// inputs only, where the message carries it, or else the Run's push count, in its stamp's
  /// user word.
  std::int64_t pushed_seq = -1;
};

// What the ports of one Run share; it outlives them and does not change while they run.
struct DataflowPortContext {
  RunCore& core;
  const DataflowRunConfig& config;
  const InputStreamOptions& stream_options;
  pipeline_internal::StreamNames& streams;
  const std::atomic<bool>& stopped;    ///< Run::stop began
  std::atomic<std::uint64_t>* run_seq; ///< a Run of several inputs: its push counter; else null
};

class DataflowInputPort {
public:
  DataflowInputPort(const DataflowPortContext& run, std::size_t index);
  ~DataflowInputPort();
  DataflowInputPort(const DataflowInputPort&) = delete;
  DataflowInputPort& operator=(const DataflowInputPort&) = delete;

  const DataflowInputConfig& config() const noexcept {
    return config_;
  }
  /// A blocking push publishes with the policy push, which waits only on Block links, instead of
  /// the waiting push: some Output it feeds has OutputOptions::drop (DataflowGraphCompiler.cpp,
  /// derive_config, rules out the mixes the runtime cannot serve).
  bool lossy_block() const noexcept {
    return lossy_block_;
  }
  bool fan_out() const noexcept {
    return config_.outputs.size() > 1U;
  }
  /// A device input: InputOptions::memory_policy resolved to Ev74 or Dms0, so its port's
  /// slots are DMA-BUFs. Its pushes and pulls run in DataflowDeviceMemory.cpp.
  bool device() const noexcept {
    return device_;
  }
  /// The Run the input belongs to.
  const RunCore& core() const noexcept {
    return run_.core;
  }
  /// A model route's publish instants; null for an Input that feeds its Outputs directly, whose
  /// pulled message carries its own (the Feed's publish_time, which a Fetch reading it makes a
  /// steady_clock instant).
  const DataflowPublishTimes* publish_times() const noexcept {
    return publish_times_.get();
  }

  // Binds now when `seed` (one tensor: a leaf; several: a record), the declaration or, in front of
  // a model route, what the route's first stage takes gives the spec; otherwise the first push
  // binds. Throws NeatError when this backend cannot carry it.
  void bind_at_build(std::span<const Tensor> seed);

  // One item into the input: Core's checks, the binding (the input's first item binds it), then
  // the slot. A leaf is wrapped instead of copied when that is safe (DataflowInputPort.cpp,
  // "Copy or wrap") and `item.may_wrap`. Throws NeatError for an item the input does not take.
  // `device_port`: a constant at each call, true only in a device input's push
  // (dataflow_device::push, another translation unit); a wrap is then device memory of the port's
  // kind (dataflow_device::wrap_tensor) instead of CPU storage, and the Run's own push, a host
  // input's, is compiled without it.
  // `frame_word` (a device input's push only): not 0 on a model route whose frames change size,
  // where it is the message's stamp's user word (dataflow_frame_word). A constant 0 at every
  // other call, which is then compiled without it.
  // Inline (as publish, its funnel): the Run's push compiles the checks, the slot and the outcome
  // as one function.
  [[gnu::always_inline]] DataflowPushOutcome
  push(const DataflowPushItem& item, bool device_port = false, std::uint64_t frame_word = 0);

  // After a device input's push answered `outcome` (dataflow_device::push; defined in
  // DataflowDeviceMemory.cpp): the first device tensor (device storage) that passed the push's
  // checks has its geometry kept, and a push that repeats it skips those checks (device_known).
  void device_remember(std::span<const Tensor> item, const DataflowPushOutcome& outcome);
  // Before a device input's try_push (dataflow_device::push): the refusal Session::alloc gives a
  // device tensor of the validated geometry while the links are known to be full (Bound::full_at),
  // before any other work on the item; none otherwise. `pushed_seq` as push()'s.
  df::Status device_still_full(std::span<const Tensor> item, std::int64_t pushed_seq) noexcept;
  // A model route whose frames change size (DataflowFrameInput.cpp). After such a push
  // answered `outcome`: a device tensor that passed the checks and whose geometry is not kept
  // yet has it kept, beside the first one's (frames of a few sizes in turn each skip the checks
  // once their geometry was seen; device_known finds them).
  void device_remember_frame(std::span<const Tensor> item, const DataflowPushOutcome& outcome);
  // device_known for such a route's push: the validated geometry that `item` repeats, the first
  // one's or a further one's; nullptr otherwise. Asks Core's cross-Run rule as device_known does.
  const DataflowLeafGeometry* device_known_frame(std::span<const Tensor> item,
                                                 const char* where) const;
  // The push of one NV12 frame (`item`'s one tensor) into such a route's byte port, whose
  // message's stamp's user word is `frame_word`: Core's checks, the binding, then the slot backed
  // by the frame's own DMA-BUF when the frame is the port's bytes as it is
  // (dataflow_planar_frame_wraps), or its planes copied to the port's offsets. The funnel below
  // is not used: it carries a frame as the input's own spec states it, and this port's spec is
  // bytes.
  DataflowPushOutcome device_push_planes(const DataflowPushItem& item, std::uint64_t frame_word);

  // Run::alloc: a writable Tensor over a slot of the input's pool. `at` names the call.
  Tensor alloc(const std::string& at);

  // Core's cross-Run rule for frames still owned by another graph's producer
  // (allow_public_cross_run_gstsample_push).
  static void enforce_transferable(const Sample& sample, const char* where);

  struct Bound;
  // The bound input, once it is bound; waits for that until `deadline`, a close or a stop (then
  // nullptr). One load of the binding per pull: what follows reads the Bound.
  Bound* wait_bound(df::Deadline deadline) {
    Bound* b = bound();
    return b ? b : wait_bound_slow(deadline);
  }
  // The next message of the input's link `link` (DataflowInputConfig::outputs order).
  static df::Result<df::Message> pull(Bound& b, std::size_t link, df::Deadline deadline) noexcept {
    return b.session.pull(b.fetches[link], deadline);
  }
  // Sample::orig_input_seq of `message` in a Run of several inputs: its stamp's user word, which
  // the push set to the value plus one; -1 in a Run of one input, whose pushes leave the word 0.
  static std::int64_t orig_input_seq(const df::Message& message) noexcept {
    return static_cast<std::int64_t>(message.stamp().user) - 1;
  }

  bool closed() const noexcept {
    return closed_.load(std::memory_order_acquire);
  }
  // Run::close_input, in two steps over the Run's inputs: every input refuses pushes and wakes
  // its waiting pulls (mark_closed), then each Session closes its Feed (close_feed): a parked
  // push is cancelled, what is queued stays pullable.
  void mark_closed();
  void close_feed();
  // Run::stop set the Run's flag: wakes the pulls that wait for this input's binding.
  void wake();
  // Session::stop; nullopt when the input never bound or the stop finished.
  df::Status stop(df::Deadline deadline);
  // The Session's counters; nullopt when the input never bound.
  std::optional<df::SessionReport> report() const;
  // The Session's report as text (SessionReport::to_text, its operators named as the Plan names
  // them); empty when the input never bound.
  std::string report_text() const;

  // The running Session of the input and what was bound with it: one Feed, one Fetch per Output
  // the input feeds (link order: DataflowInputConfig::outputs). The Run reads `binding` and
  // `has_wraps` of it to build a pulled message's tensors.
  struct Bound {
    Bound(DataflowInputBinding b, df::Session s, std::uint32_t feed_op,
          const std::vector<std::uint32_t>& fetch_ops);
    ~Bound();
    DataflowInputBinding binding;
    df::Session session;
    df::FeedRef feed;
    std::vector<df::FetchRef> fetches;
    // A push may wrap (`wraps`): a tensor port, copy_input off. What a wrap must be:
    // `slot_bytes` long, aligned to `wrap_mask` + 1 (the Feed's port, port_spec.h slot_bytes and
    // slot_alignment), and, into a device port, device memory of the port's kind. Fixed at bind.
    bool wraps = false;
    /// The port's memory is a device's: a pulled message's tensors are built from its
    /// DMA-BUF (dataflow_device::pulled_sample, pulled_tensors). Fixed at bind.
    bool device = false;
    /// A model route whose first stage is the MLA, which reads a DMA-BUF of either device
    /// heap, or an A65 kernel, which reads either through the CPU's address: a wrap takes an EV74
    /// or an MLA tensor whatever the port's own kind (as GStreamer's input of the route does).
    /// False in front of a kernel on the EV74, which reaches linux,cma only: there a wrap is a
    /// tensor of the port's kind, and any other is copied. Fixed at bind.
    bool any_device_memory = false;
    std::size_t slot_bytes = 0;
    std::uintptr_t wrap_mask = 0;
    /// A push has wrapped: a pulled message may carry the pushed storage. Set, before its
    /// publish, by the first push that wraps; a pull's take orders that publish, and so this
    /// store, before it.
    std::atomic<bool> has_wraps{false};
    /// RunCore::outputs_pulled when alloc last refused a wrapping try_push because every link was
    /// full (FR14); kNoRefusal otherwise (still_full; a device input: device_full).
    static constexpr std::uint64_t kNoRefusal = ~std::uint64_t{0};
    std::atomic<std::uint64_t> full_at{kNoRefusal};
    /// A model route whose first stage is an A65 kernel, or whose frames change size: a wrap
    /// carries the CPU's address of the tensor's bytes, from this cache of the tensors pushed so
    /// far (DataflowDeviceMemory.h). Null on every other route. Fixed at bind.
    std::shared_ptr<dataflow_device::MappedWraps> mapped_wraps;
    /// What Plan::compile decided: the operators' names in the Session's report. Cold, last.
    df::PlanReport plan;
  };

private:
  struct SlotLoan;
  class LentSlot;

  Bound* bound() const noexcept {
    return bound_.load(std::memory_order_acquire);
  }
  Bound* wait_bound_slow(df::Deadline deadline);
  Bound* bind_first(std::span<const Tensor> item, const char* where);
  // `where` names the call (Graph::build, `at_build`, or the first push) in the errors of the
  // input's Plan and Session, and of a binding the route's model does not take.
  Bound* bind_locked(DataflowInputBinding binding, const char* where, bool at_build);
  df::Status still_full(Bound& bound);
  // A device input's validated geometry that `item` repeats, of a device tensor or of a CPU
  // tensor (DataflowDeviceMemory.cpp); nullptr otherwise. For a device tensor it asks Core's
  // cross-Run rule first (dataflow_device::enforce_transferable), which throws as the checks do.
  const DataflowLeafGeometry* device_known(std::span<const Tensor> item, const char* where) const;
  // A device input's try_push, before it prepares a slot (DataflowDeviceMemory.cpp): the refusal
  // Session::alloc gives now (FR14: every link full, kept in Bound::full_at), or none when it
  // would take a slot.
  df::Status device_full(Bound& bound) noexcept;
  // `tensors`: `item`'s, or the stage's view of them. `map_fn`: map_fn_type of the first tensor
  // when push asked it already, else null.
  [[gnu::always_inline]] DataflowPushOutcome publish(Bound& bound, const DataflowPushItem& item,
                                                     const df::Stamp& stamp,
                                                     RunAllocation* allocation, bool wrappable,
                                                     bool device_port, bool frame_port,
                                                     const std::type_info* map_fn);
  // The publish itself (`send` calls Session::push or push_waiting), timed: a push whose slot was
  // asked for at `alloc_start` and obtained at `alloc_end` (both 0 for a Run::alloc slot), its
  // bytes then written. A model route's publish instant is kept around it (publish_times_).
  template <class Send>
  [[gnu::always_inline]] DataflowPushOutcome publish_timed(Send&& send, std::int64_t alloc_start,
                                                           std::int64_t alloc_end);

  const DataflowPortContext run_;
  const std::size_t index_;
  const DataflowInputConfig& config_;
  const std::uint32_t stream_; ///< its default stream's id (StreamNames)
  bool lossy_block_ = false;
  const bool device_; ///< InputOptions::memory_policy resolved to Ev74 or Dms0
  /// A model route: the publish instants of the items it takes, and the depth of the Feed's link
  /// in front of its first operator.
  std::unique_ptr<DataflowPublishTimes> publish_times_;
  std::uint32_t feed_depth_ = 1;

  // The binding is written once under mu_ and read lock-free; cv_ wakes a pull that waits for it.
  std::mutex mu_;
  std::condition_variable cv_;
  std::unique_ptr<Bound> bound_storage_;
  std::atomic<Bound*> bound_{nullptr};
  std::atomic<bool> closed_{false};
  DataflowValidatedLeaf validated_;
  DataflowValidatedLeaf validated_device_; ///< a device input: of a device tensor
  /// A model route: the loaded model and the checked kernels of the input's stages
  /// (DataflowModelRoute.h), for this Run's operators. Null otherwise.
  std::shared_ptr<void> model_;
  /// A model route whose frames change size: the validated geometries of further device
  /// tensors, beside `validated_device_`'s first (device_remember_frame), as many in all as the
  /// route's frames table keeps geometries (DataflowFrames::kKept), so that frames of that many
  /// sizes in turn each pass the checks once. Each is written once, under mu_, and read
  /// lock-free.
  static constexpr std::size_t kValidatedFrames = 16;
  std::array<DataflowValidatedLeaf, kValidatedFrames - 1> validated_frames_;
};

// The backing of a Run::alloc Tensor on this backend: the loan of one slot of the input's pool.
struct DataflowInputPort::SlotLoan {
  df::WriteLoan loan;
};

// A Tensor from this Run's alloc() in a push: the loan of its slot leaves the allocation for the
// publish, and goes back when the push ends with the loan intact (an error, or SR1, FR2, FR3,
// FR5: stopped, no room, input closed), so the application may push it again or drop it.
class DataflowInputPort::LentSlot {
public:
  LentSlot(RunAllocation* allocation, df::WriteLoan& loan, const char* where)
      : allocation_(allocation), loan_(loan) {
    if (allocation_) {
      begin_allocated_push(*allocation_, where);
      loan_ = std::move(box().loan);
    }
  }
  ~LentSlot() {
    if (allocation_) {
      if (loan_) {
        box().loan = std::move(loan_);
      }
      end_allocated_push(*allocation_, !box().loan);
    }
  }
  LentSlot(const LentSlot&) = delete;
  LentSlot& operator=(const LentSlot&) = delete;

private:
  SlotLoan& box() const {
    return *static_cast<SlotLoan*>(allocation_->backing.get());
  }
  RunAllocation* allocation_;
  df::WriteLoan& loan_;
};

template <class Send>
inline DataflowPushOutcome DataflowInputPort::publish_timed(Send&& send, std::int64_t alloc_start,
                                                            std::int64_t alloc_end) {
  const std::int64_t start = dataflow_now_ns();
  if (publish_times_) {
    publish_times_->claim(start);
  }
  DataflowPushOutcome outcome{DataflowPushOutcome::Stage::Publish, send(), {}};
  const std::int64_t end = dataflow_now_ns();
  if (publish_times_) {
    bool entered = false;
    bool evicted = false;
    if (const auto* result = std::get_if<df::PublishResult>(&outcome.answer);
        result && result->published() && result->size() == 1U) {
      evicted = result->outcome(0) == df::Link::Offer::DroppedOldest;
      entered = evicted || result->outcome(0) == df::Link::Offer::Accepted;
    }
    publish_times_->settle(end, entered, evicted, feed_depth_);
  }
  outcome.timing.alloc_ns = alloc_end - alloc_start;
  outcome.timing.copy_ns = alloc_end == 0 ? 0 : start - alloc_end;
  outcome.timing.publish_ns = end - start;
  return outcome;
}

// The one path of every push: the slot, its bytes, the publish.
//   the slot   the loan of the allocation (a Tensor from Run::alloc), or one from the pool:
//              Acquire::Block for push, Acquire::Fail for try_push, backed by the tensor's own
//              storage when it wraps;
//   its bytes  nothing for a wrap or an allocation (they are the tensor's own); the tensor
//              copied at the binding's layout; a record's fields copied at their offsets;
//   publish    try_push: the policy push at a passed deadline. push: the waiting push, which
//              waits for room on every link and never drops; or, when an Output of the input has
//              OutputOptions::drop, the policy push without a deadline, which waits on the Block
//              links only and evicts on the DropOldest ones (appsink drop=true).
inline DataflowPushOutcome DataflowInputPort::publish(Bound& b, const DataflowPushItem& item,
                                                      const df::Stamp& stamp,
                                                      RunAllocation* allocation, bool wrappable,
                                                      bool device_port, bool frame_port,
                                                      const std::type_info* map_fn) {
  const bool block = item.block;
  const char* const where = item.where;
  df::WriteLoan loan;
  const LentSlot lent(allocation, loan, where);
  const Tensor& first = item.tensors.front();
  bool wrapped = false;
  std::int64_t alloc_start = 0;
  std::int64_t alloc_end = 0;
  if (allocation) {
    if (!loan || !first.is_dense() || first.byte_offset != 0 || !first.is_contiguous()) {
      dataflow_codec::throw_input_shape(
          std::string(where) + ": a Tensor from Run::alloc must stay dense, contiguous and at byte "
                               "offset 0 to be pushed");
    }
  } else {
    // A device port's try_push asks alloc first, every time (device_full): its wrap costs the
    // checks of the tensor's memory, which a refusal must not pay. With room, full_at is clear and
    // the question below is not asked.
    if (device_port && !block) {
      if (const df::Status refusal = device_full(b)) {
        return {DataflowPushOutcome::Stage::Alloc, *refusal};
      }
    }
    if (!block && b.full_at.load(std::memory_order_relaxed) ==
                      run_.core.outputs_pulled.load(std::memory_order_relaxed)) {
      if (const df::Status refusal = still_full(b)) {
        return {DataflowPushOutcome::Stage::Alloc, *refusal};
      }
    }
    // Wrap when safe. The runtime takes a wrap only with the slot (Session::alloc), so the wrap is
    // decided before alloc; a refusal of alloc (FR14, FR19, FR7, SR1, FR3) takes no slot and keeps
    // nothing: `wrap`, the only reference this call took, is released on return.
    // A device port takes device memory only: the tensor's own DMA-BUF; the bounded port of a
    // model route whose frames change size (`frame_port`, a constant at each call as
    // `device_port` is) a frame of any size up to its bound.
    df::Block wrap;
    wrapped =
        wrappable &&
        (device_port
             ? (frame_port
                    ? dataflow_device::wrap_frame(first, b.slot_bytes, b.wrap_mask, wrap,
                                                  b.mapped_wraps.get())
                    : dataflow_device::wrap_tensor(first, b.binding.spec, b.slot_bytes, b.wrap_mask,
                                                   wrap, b.any_device_memory, b.mapped_wraps.get()))
             : dataflow_codec::wrap_block(first, b.slot_bytes, b.wrap_mask,
                                          map_fn ? *map_fn : dataflow_codec::map_fn_type(first),
                                          wrap));
    const df::Acquire mode = block ? df::Acquire::Block : df::Acquire::Fail;
    alloc_start = dataflow_now_ns();
    df::Result<df::WriteLoan> allocated =
        b.session.alloc(b.feed, mode, df::kNoDeadline, wrapped ? &wrap : nullptr);
    if (const df::Error* error = df::failure(allocated)) {
      if (!wrapped || error->code != df::ErrorCode::InvalidArgument) {
        if (!block && wrapped && error->code == df::ErrorCode::Backpressure) {
          b.full_at.store(run_.core.outputs_pulled.load(std::memory_order_relaxed),
                          std::memory_order_relaxed);
        }
        return {DataflowPushOutcome::Stage::Alloc, *error};
      }
      // FR15: the runtime refused the wrap itself (dataflow_wrap_layout and wrap_block state its
      // rule, so this is not expected), with the slot returned and nothing kept. The copy.
      wrapped = false;
      allocated = b.session.alloc(b.feed, mode);
      if (const df::Error* again = df::failure(allocated)) {
        return {DataflowPushOutcome::Stage::Alloc, *again};
      }
    }
    alloc_end = dataflow_now_ns();
    loan = std::get<df::WriteLoan>(std::move(allocated));
  }
  const auto send = [&](const auto& extent) {
    return publish_timed(
        [&] {
          if (!block) {
            return b.session.push(b.feed, std::move(loan), stamp, extent, df::Deadline{});
          }
          if (lossy_block_) {
            return b.session.push(b.feed, std::move(loan), stamp, extent, df::kNoDeadline);
          }
          return b.session.push_waiting(b.feed, std::move(loan), stamp, extent, df::kNoDeadline);
        },
        alloc_start, alloc_end);
  };
  if (b.binding.is_record()) {
    std::array<df::Extent, df::kMaxFields> extents{};
    dataflow_codec::copy_record(b.binding, item.tensors, loan.bytes(), extents, where);
    return send(std::span<const df::Extent>(extents.data(), item.tensors.size()));
  }
  df::Extent extent;
  if (wrapped) {
    // The loan's bytes are the tensor's own: nothing to copy.
    extent = dataflow_codec::wrapped_extent(b.binding, first);
    if (!b.has_wraps.load(std::memory_order_relaxed)) {
      b.has_wraps.store(true, std::memory_order_relaxed);
    }
  } else if (allocation) {
    extent = dataflow_codec::extent_of(first.shape); // the slot is published as it is
  } else {
    extent = dataflow_codec::copy_into(b.binding, first, loan.bytes(), where);
  }
  return send(extent);
}

inline DataflowPushOutcome DataflowInputPort::push(const DataflowPushItem& pushed, bool device_port,
                                                   std::uint64_t frame_word) {
  const std::span<const Tensor> item = pushed.tensors;
  const char* const where = pushed.where;
  const InputOptions& declared = config_.input;
  // Core's own push-time checks and texts first (push_tensor_items, validate_declared_shape):
  // the declaration describes the item's representative (first) tensor, as on GStreamer.
  if (item.size() > 1U && !declared.shape.empty()) {
    dataflow_codec::throw_input_shape(std::string(where) + ": an item of " +
                                      std::to_string(item.size()) +
                                      " tensors where one tensor of a declared shape is expected");
  }
  // A Tensor from this Run's alloc() that was pushed already has an empty storage: say so first.
  // Its storage's map_fn type tells it; publish reads the same type for the wrap.
  const std::type_info* map_fn = nullptr;
  std::shared_ptr<RunAllocation> allocation;
  if (pushed.commit_allocation && item.size() == 1U) {
    map_fn = &dataflow_codec::map_fn_type(item.front());
    if (*map_fn == typeid(RunAllocationMapper)) {
      allocation = allocation_of(item.front(), run_.core);
    }
  }
  if (allocation) {
    require_pushable(*allocation, where);
  }
  // A leaf with the geometry of the first leaf that passed every check below passes them again
  // (DataflowValidatedLeaf): they are skipped for it. A device input also keeps a device tensor's
  // geometry, and asks the cross-Run rule, which is not geometry, of every push (device_known).
  const DataflowLeafGeometry* const known =
      device_port ? (frame_word != 0U ? device_known_frame(item, where) : device_known(item, where))
                  : validated_.find(item);
  if (!known) {
    validate_public_input(item.front(), declared, run_.stream_options, where);
    for (const Tensor& tensor : item) {
      if (tensor.storage && tensor.storage->kind == StorageKind::GstSample) {
        enforce_transferable(sample_from_tensors(TensorList{tensor}), where);
      }
    }
  }
  Bound* b = bound();
  if (!b) {
    if (run_.stopped.load(std::memory_order_acquire) || closed()) {
      return {};
    }
    b = bind_first(item, where);
    if (!b) {
      return {};
    }
  }
  df::Stamp stamp = dataflow_codec::runtime_stamp(pushed.proposal, run_.streams, stream_,
                                                  config_.model != nullptr);
  bool wrappable = false;
  if (item.size() > 1U || b->binding.is_record()) {
    if (const std::string why = dataflow_record_mismatch(b->binding, item); !why.empty()) {
      dataflow_codec::throw_input_shape(std::string(where) + ": " + why);
    }
  } else {
    const Tensor& tensor = item.front();
    if (!known) {
      if (const std::string why = dataflow_binding_mismatch(b->binding, tensor); !why.empty()) {
        dataflow_codec::throw_input_shape(std::string(where) + ": " + why);
      }
      if (validated_.empty() && DataflowLeafGeometry::keepable(tensor)) {
        std::lock_guard<std::mutex> lock(mu_);
        validated_.remember(tensor,
                            b->wraps && dataflow_wrap_layout(b->binding, b->slot_bytes, tensor));
      }
    }
    // Wrap when safe (the layout here, the storage in publish).
    wrappable =
        !allocation && pushed.may_wrap && b->wraps &&
        (known ? known->wrappable : dataflow_wrap_layout(b->binding, b->slot_bytes, tensor));
  }
  if (run_.run_seq) {
    // A Run of several inputs (GStreamer's graph ingress, stamp_public_graph_ingress_sample): the
    // pushed value, or the next of the Run's push counter, which its inputs share (one relaxed
    // increment; a refused push consumes a number, as there).
    stamp.user =
        1U + (pushed.pushed_seq >= 0 ? static_cast<std::uint64_t>(pushed.pushed_seq)
                                     : run_.run_seq->fetch_add(1, std::memory_order_relaxed));
  }
  if (device_port && frame_word != 0U) {
    // A model route (a Run of one input, so the word is not the Run's counter) whose frames
    // change size states the frame's geometry here, for its stages and its pull.
    stamp.user = frame_word;
  }
  return publish(*b, pushed, stamp, allocation.get(), wrappable, device_port,
                 device_port && frame_word != 0U, map_fn);
}

#pragma GCC visibility pop

} // namespace simaai::neat::runtime
