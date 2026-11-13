#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// One public Input of a Run on the Dataflow backend (H0c): its Session, with one Feed and one
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

// What the Session answered a push: where it ended, and the runtime's own answer.
struct DataflowPushOutcome {
  enum class Stage : std::uint8_t {
    Unbound, ///< no binding, and the input is closed or the Run stopped: nothing was asked
    Alloc,   ///< Session::alloc refused the slot (`answer` holds the error)
    Publish, ///< Session::push or push_waiting answered (`answer`)
  };
  Stage stage = Stage::Unbound;
  df::Result<df::PublishResult> answer;
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
  /// A device input (H0b): InputOptions::memory_policy resolved to Ev74 or Dms0, so its port's
  /// slots are DMA-BUFs. Its pushes and pulls run in DataflowDeviceMemory.cpp.
  bool device() const noexcept {
    return device_;
  }

  // Binds now when `seed` (one tensor: a leaf; several: a record) or the declaration gives the
  // spec; otherwise the first push binds. Throws NeatError when this backend cannot carry it.
  void bind_at_build(std::span<const Tensor> seed);

  // One item into the input: Core's checks, the binding (the input's first item binds it), then
  // the slot. With `commit_allocation` (the blocking TensorList push only), a Tensor from this
  // Run's alloc() alone in the item is published as its slot; every other verb copies it like any
  // CPU tensor, as the GStreamer backend does. `pushed_seq`: the item's own orig_input_seq (a
  // Sample push), -1 for none; read in a Run of several inputs only, where the message carries
  // it, or else the Run's push count, in its stamp's user word. A leaf is wrapped instead of
  // copied when that is safe (W2), unless `may_wrap` is false (a cv::Mat that does not own its
  // memory). Throws NeatError for an item the input does not take.
  // `device_port` (H0b): a constant at each call, true only in a device input's push
  // (dataflow_device::push, another translation unit); a wrap is then device memory of the port's
  // kind (dataflow_device::wrap_tensor) instead of CPU storage, and the Run's own push, a host
  // input's, is compiled without it.
  // Inline (as publish, its funnel): the Run's push compiles the checks, the slot and the outcome
  // as one function, as it did when they were one (0.1 to 0.3 us of a 3 us round trip on the A65).
  [[gnu::always_inline]] DataflowPushOutcome push(std::span<const Tensor> item,
                                                  const Stamp& proposal, bool block,
                                                  const char* where, bool commit_allocation = false,
                                                  std::int64_t pushed_seq = -1,
                                                  bool may_wrap = true, bool device_port = false);

  // Run::alloc: a writable Tensor over a slot of the input's pool. `at` names the call.
  Tensor alloc(const std::string& at);

  // Core's cross-Run rule for frames still owned by another graph's producer (RunPush.cpp:39-50).
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
    // W2. A push may wrap (`wraps`): a tensor port, copy_input off. What a wrap must be:
    // `slot_bytes` long, aligned to `wrap_mask` + 1 (the Feed's port, port_spec.h slot_bytes and
    // slot_alignment), and, into a device port (H0b), device memory of the port's kind. Fixed at
    // bind.
    bool wraps = false;
    /// The port's memory is a device's (H0b): a pulled message's tensors are built from its
    /// DMA-BUF (dataflow_device::pulled_sample, pulled_tensors). Fixed at bind.
    bool device = false;
    std::size_t slot_bytes = 0;
    std::uintptr_t wrap_mask = 0;
    /// A push has wrapped: a pulled message may carry the pushed storage (W2). Set, before its
    /// publish, by the first push that wraps; a pull's take orders that publish, and so this
    /// store, before it.
    std::atomic<bool> has_wraps{false};
    /// RunCore::outputs_pulled when alloc last refused a wrapping try_push because every link was
    /// full (FR14); kNoRefusal otherwise (still_full).
    static constexpr std::uint64_t kNoRefusal = ~std::uint64_t{0};
    std::atomic<std::uint64_t> full_at{kNoRefusal};
  };

private:
  struct SlotLoan;
  class LentSlot;

  Bound* bound() const noexcept {
    return bound_.load(std::memory_order_acquire);
  }
  Bound* wait_bound_slow(df::Deadline deadline);
  Bound* bind_first(std::span<const Tensor> item, const char* where);
  Bound* bind_locked(DataflowInputBinding binding);
  df::Status still_full(Bound& bound);
  [[gnu::always_inline]] DataflowPushOutcome publish(Bound& bound, std::span<const Tensor> item,
                                                     const df::Stamp& stamp, bool block,
                                                     const char* where, RunAllocation* allocation,
                                                     bool wrappable, bool device_port);

  const DataflowPortContext run_;
  const std::size_t index_;
  const DataflowInputConfig& config_;
  const std::uint32_t stream_; ///< its default stream's id (StreamNames)
  bool lossy_block_ = false;
  const bool device_; ///< InputOptions::memory_policy resolved to Ev74 or Dms0

  // The binding is written once under mu_ and read lock-free; cv_ wakes a pull that waits for it.
  std::mutex mu_;
  std::condition_variable cv_;
  std::unique_ptr<Bound> bound_storage_;
  std::atomic<Bound*> bound_{nullptr};
  std::atomic<bool> closed_{false};
  DataflowValidatedLeaf validated_;
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
inline DataflowPushOutcome DataflowInputPort::publish(Bound& b, std::span<const Tensor> item,
                                                      const df::Stamp& stamp, bool block,
                                                      const char* where, RunAllocation* allocation,
                                                      bool wrappable, bool device_port) {
  df::WriteLoan loan;
  const LentSlot lent(allocation, loan, where);
  const Tensor& first = item.front();
  bool wrapped = false;
  if (allocation) {
    if (!loan || !first.is_dense() || first.byte_offset != 0 || !first.is_contiguous()) {
      dataflow_codec::throw_input_shape(
          std::string(where) + ": a Tensor from Run::alloc must stay dense, contiguous and at byte "
                               "offset 0 to be pushed");
    }
  } else {
    if (!block && b.full_at.load(std::memory_order_relaxed) ==
                      run_.core.outputs_pulled.load(std::memory_order_relaxed)) {
      if (const df::Status refusal = still_full(b)) {
        return {DataflowPushOutcome::Stage::Alloc, *refusal};
      }
    }
    // W2: wrap when safe. The runtime takes a wrap only with the slot (Session::alloc), so the
    // wrap is decided before alloc; a refusal of alloc (A3's FR14, FR7, SR1, FR3) takes no slot and
    // keeps nothing: `wrap`, the only reference this call took, is released on return.
    // A device port (H0b) takes device memory only: the tensor's own DMA-BUF.
    df::Block wrap;
    wrapped = wrappable &&
              (device_port ? dataflow_device::wrap_tensor(first, b.binding.spec.memory,
                                                          b.slot_bytes, b.wrap_mask, wrap)
                           : dataflow_codec::wrap_block(first, b.slot_bytes, b.wrap_mask, wrap));
    const df::Acquire mode = block ? df::Acquire::Block : df::Acquire::Fail;
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
    loan = std::get<df::WriteLoan>(std::move(allocated));
  }
  const auto send = [&](const auto& extent) {
    if (!block) {
      return b.session.push(b.feed, std::move(loan), stamp, extent, df::Deadline{});
    }
    if (lossy_block_) {
      return b.session.push(b.feed, std::move(loan), stamp, extent, df::kNoDeadline);
    }
    return b.session.push_waiting(b.feed, std::move(loan), stamp, extent, df::kNoDeadline);
  };
  if (b.binding.is_record()) {
    std::array<df::Extent, df::kMaxFields> extents{};
    dataflow_codec::copy_record(b.binding, item, loan.bytes(), extents, where);
    return {DataflowPushOutcome::Stage::Publish,
            send(std::span<const df::Extent>(extents.data(), item.size()))};
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
  return {DataflowPushOutcome::Stage::Publish, send(extent)};
}

inline DataflowPushOutcome DataflowInputPort::push(std::span<const Tensor> item,
                                                   const Stamp& proposal, bool block,
                                                   const char* where, bool commit_allocation,
                                                   std::int64_t pushed_seq, bool may_wrap,
                                                   bool device_port) {
  const InputOptions& declared = config_.input;
  // Core's own push-time checks and texts first (RunPush.cpp:655, InputStreamUtil.cpp:2146-2190):
  // the declaration describes the item's representative (first) tensor, as on GStreamer.
  if (item.size() > 1U && !declared.shape.empty()) {
    dataflow_codec::throw_input_shape(std::string(where) + ": an item of " +
                                      std::to_string(item.size()) +
                                      " tensors where one tensor of a declared shape is expected");
  }
  // A Tensor from this Run's alloc() that was pushed already has an empty storage: say so first.
  const std::shared_ptr<RunAllocation> allocation =
      commit_allocation && item.size() == 1U ? allocation_of(item.front(), run_.core) : nullptr;
  if (allocation) {
    require_pushable(*allocation, where);
  }
  // A leaf with the geometry of the first leaf that passed every check below passes them again
  // (DataflowValidatedLeaf): they are skipped for it.
  const DataflowLeafGeometry* const known = validated_.find(item);
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
  df::Stamp stamp = dataflow_codec::runtime_stamp(proposal, run_.streams, stream_);
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
    // W2: wrap when safe (the layout here, the storage in publish).
    wrappable =
        !allocation && may_wrap && b->wraps &&
        (known ? known->wrappable : dataflow_wrap_layout(b->binding, b->slot_bytes, tensor));
  }
  if (run_.run_seq) {
    // A Run of several inputs (GStreamer's graph ingress, RunPush.cpp:276-287): the pushed value,
    // or the next of the Run's push counter, which its inputs share (one relaxed increment; a
    // refused push consumes a number, as there).
    stamp.user = 1U + (pushed_seq >= 0 ? static_cast<std::uint64_t>(pushed_seq)
                                       : run_.run_seq->fetch_add(1, std::memory_order_relaxed));
  }
  return publish(*b, item, stamp, block, where, allocation.get(), wrappable, device_port);
}

#pragma GCC visibility pop

} // namespace simaai::neat::runtime
