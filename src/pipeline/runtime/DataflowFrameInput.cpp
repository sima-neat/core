// A model route whose frames change size on the Dataflow backend (H1, step Y4): the push of a
// frame into its input (dataflow_device::push_frame, DataflowDeviceMemory.h), and the two
// functions of the input's port that only such a route calls. In a unit of their own: the device
// push's unit (DataflowDeviceMemory.cpp) is compiled as it was, and this one instantiates the
// port's push funnel a second time, for the frame's stamp word.
#if defined(SIMANEAT_WITH_DATAFLOW)

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/runtime/DataflowDeviceMemory.h"
#include "pipeline/runtime/DataflowInputPort.h"
#include "pipeline/runtime/DataflowModelRoute.h"
#include "pipeline/runtime/DataflowTensorCodec.h"

#include <sys/mman.h>
#include <unistd.h>

#include <array>
#include <memory>
#include <mutex>
#include <span>
#include <utility>

namespace simaai::neat::runtime::dataflow_device {

namespace {

// What a frame wrapped with its CPU address keeps: the pushed tensor's storage and Core's own
// read-only mapping of the frame's bytes in its DMA-BUF (not a map of its GStreamer memory,
// which would refuse the application's next write map of the tensor). The mapping ends with the
// last drop of the Block: the message's, or the operator's that kept the buffer.
struct FrameHold {
  std::shared_ptr<TensorBuffer> storage;
  void* base = nullptr;
  std::size_t length = 0;
  ~FrameHold() {
    if (base) {
      ::munmap(base, length);
    }
  }
};

// One mapping per pushed frame, found again by the tensor's storage, byte offset and bytes while
// somebody still holds it (the operator keeps the buffers it admitted). Sixteen frames in
// rotation are served without a map; a seventeenth replaces the oldest entry. An entry does not
// keep its mapping alive.
class FrameWraps {
public:
  std::shared_ptr<FrameHold> get(const std::shared_ptr<TensorBuffer>& storage,
                                 std::int64_t byte_offset, int fd, std::size_t fd_offset,
                                 std::size_t bytes, std::byte*& data) {
    static const std::size_t page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    const std::size_t lead = fd_offset % page;
    std::lock_guard<std::mutex> lock(mu_);
    for (Entry& entry : entries_) {
      if (entry.storage == storage.get() && entry.byte_offset == byte_offset &&
          entry.bytes == bytes) {
        if (std::shared_ptr<FrameHold> hold = entry.hold.lock()) {
          data = static_cast<std::byte*>(hold->base) + lead;
          return hold;
        }
      }
    }
    const std::size_t length = lead + bytes;
    void* const base =
        ::mmap(nullptr, length, PROT_READ, MAP_SHARED, fd, static_cast<off_t>(fd_offset - lead));
    if (base == MAP_FAILED) {
      return nullptr;
    }
    auto hold = std::make_shared<FrameHold>();
    hold->storage = storage;
    hold->base = base;
    hold->length = length;
    Entry& entry = entries_[next_++ % entries_.size()];
    entry.storage = storage.get();
    entry.byte_offset = byte_offset;
    entry.bytes = bytes;
    entry.hold = hold;
    data = static_cast<std::byte*>(base) + lead;
    return hold;
  }

private:
  struct Entry {
    const TensorBuffer* storage = nullptr; ///< compared only while `hold` lives, which keeps it
    std::int64_t byte_offset = 0;
    std::size_t bytes = 0;
    std::weak_ptr<FrameHold> hold;
  };
  std::mutex mu_;
  std::array<Entry, 16> entries_;
  std::size_t next_ = 0;
};

} // namespace

std::shared_ptr<void> make_frame_wraps() {
  return std::make_shared<FrameWraps>();
}

bool wrap_frame(const Tensor& t, std::size_t slot_bytes, std::uintptr_t align_mask, df::Block& out,
                void* wraps) {
  // The frame's own bytes (dataflow_wrap_layout let a dense frame with tight rows come here).
  const std::size_t bytes = t.dense_bytes_tight();
  if (!wraps || bytes == 0U || bytes > slot_bytes) {
    return false;
  }
  // wrap_tensor's Block for a tensor of exactly these bytes into an Ev74 port of fixed shape:
  // its conditions on the storage, the descriptor, the offset and its alignment.
  df::TensorSpec tensor;
  tensor.dtype = df::DType::UInt8;
  tensor.shape.rank = 1;
  tensor.shape.dims[0] = df::Dim(static_cast<std::int64_t>(bytes));
  tensor.layout = df::Dense{};
  df::PortSpec port;
  port.payload = tensor;
  port.memory = df::MemoryKind::Ev74;
  if (!wrap_tensor(t, port, bytes, align_mask, out)) {
    return false;
  }
  std::byte* data = nullptr;
  std::shared_ptr<FrameHold> hold = static_cast<FrameWraps*>(wraps)->get(
      t.storage, t.byte_offset, out.fd, out.fd_offset, bytes, data);
  if (!hold) {
    return false;
  }
  out.data = data;
  out.holder = std::move(hold);
  return true;
}

// H1: a push into a model route whose frames change size (DataflowModelRoute::frames): every
// push of one tensor into such a route comes here (its `input_shape` is empty, so the test
// that sends a tensor without the stage's batch axis out of line sends each of them). The
// frame's geometry is stated in the stamp's user word, which the route's stages and the pull
// read; a frame the route cannot run is refused before anything is asked of the Session. Then
// the device push as for any input: GStreamer's guard, the checks, the slot (the frame's own
// DMA-BUF when it is in EV74 memory, whatever its size), the publish.
DataflowPushOutcome push_frame(DataflowInputPort& port, const DataflowModelRoute& route,
                               const Tensor& tensor, const Stamp& proposal, bool block,
                               const char* where, bool commit_allocation, std::int64_t pushed_seq,
                               bool may_wrap) {
  const std::span<const Tensor> item(&tensor, 1);
  if (may_wrap && !(commit_allocation && allocation_of(tensor, port.core()))) {
    enforce_device_visible_tensors_or_throw(item, where);
  }
  const std::uint64_t word = dataflow_frame_word(route, tensor, where);
  if (route.planar_input) {
    const DataflowPushOutcome outcome =
        port.device_push_planes(tensor, proposal, block, where, word);
    port.device_remember_frame(item, outcome);
    return outcome;
  }
  // A frame without the kernel's batch axis is pushed as the kernel's tensor, a view of the
  // same storage.
  Tensor view;
  const Tensor& frame = dataflow_model_input_view(route, tensor, view);
  const std::span<const Tensor> pushed(&frame, 1);
  if (!block) {
    if (const df::Status refusal = port.device_still_full(pushed, pushed_seq)) {
      return {DataflowPushOutcome::Stage::Alloc, *refusal};
    }
  }
  const DataflowPushOutcome outcome =
      port.push(pushed, proposal, block, where, commit_allocation, pushed_seq, may_wrap,
                /*device_port=*/true, word);
  port.device_remember_frame(pushed, outcome);
  return outcome;
}

} // namespace simaai::neat::runtime::dataflow_device

namespace simaai::neat::runtime {

const DataflowLeafGeometry* DataflowInputPort::device_known_frame(std::span<const Tensor> item,
                                                                  const char* where) const {
  const DataflowLeafGeometry* known = validated_device_.find_device(item);
  for (std::size_t i = 0; !known && i < validated_frames_.size(); ++i) {
    known = validated_frames_[i].find_device(item);
  }
  if (!known) {
    return validated_.find(item);
  }
  dataflow_device::enforce_transferable(item, where);
  return known;
}

void DataflowInputPort::device_remember_frame(std::span<const Tensor> item,
                                              const DataflowPushOutcome& outcome) {
  Bound* const b = bound();
  if (!b || outcome.stage == DataflowPushOutcome::Stage::Unbound || item.size() != 1U ||
      b->binding.is_record() || !DataflowLeafGeometry::keepable_device(item.front()) ||
      validated_device_.find_device(item)) {
    return;
  }
  DataflowValidatedLeaf* free = validated_device_.empty() ? &validated_device_ : nullptr;
  for (DataflowValidatedLeaf& more : validated_frames_) {
    if (more.find_device(item)) {
      return;
    }
    if (!free && more.empty()) {
      free = &more;
    }
  }
  if (free) {
    std::lock_guard<std::mutex> lock(mu_);
    free->remember(item.front(),
                   b->wraps && dataflow_wrap_layout(b->binding, b->slot_bytes, item.front()));
  }
}

DataflowPushOutcome DataflowInputPort::device_push_planes(const Tensor& frame,
                                                          const Stamp& proposal, bool block,
                                                          const char* where,
                                                          std::uint64_t frame_word) {
  const std::span<const Tensor> item(&frame, 1);
  // Core's push-time checks, as push(): skipped for a device tensor whose geometry passed them.
  if (!device_known_frame(item, where)) {
    validate_public_input(frame, config_.input, run_.stream_options, where);
    if (frame.storage && frame.storage->kind == StorageKind::GstSample) {
      enforce_transferable(sample_from_tensors(TensorList{frame}), where);
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
  stamp.user = frame_word;
  const DataflowModelRoute& route = *config_.model;
  // The slot: the frame's own DMA-BUF, by its descriptor, when the frame is the port's bytes as
  // it is; otherwise a slot of the pool, the planes copied to the port's offsets (its write
  // epoch is the Session's, as for any copy).
  df::Block wrap;
  bool wrapped =
      b->wraps && dataflow_planar_frame_wraps(route, frame) &&
      dataflow_device::wrap_tensor(frame, b->binding.spec, b->slot_bytes, b->wrap_mask, wrap);
  const df::Acquire mode = block ? df::Acquire::Block : df::Acquire::Fail;
  df::Result<df::WriteLoan> allocated =
      b->session.alloc(b->feed, mode, df::kNoDeadline, wrapped ? &wrap : nullptr);
  if (const df::Error* error = df::failure(allocated)) {
    if (!wrapped || error->code != df::ErrorCode::InvalidArgument) {
      return {DataflowPushOutcome::Stage::Alloc, *error};
    }
    wrapped = false; // the runtime refused the wrap itself: the copy
    allocated = b->session.alloc(b->feed, mode);
    if (const df::Error* again = df::failure(allocated)) {
      return {DataflowPushOutcome::Stage::Alloc, *again};
    }
  }
  df::WriteLoan loan = std::get<df::WriteLoan>(std::move(allocated));
  if (wrapped) {
    if (!b->has_wraps.load(std::memory_order_relaxed)) {
      b->has_wraps.store(true, std::memory_order_relaxed);
    }
  } else if (!copy_dataflow_planar_frame(route, frame, loan.bytes())) {
    dataflow_codec::throw_input_shape(
        std::string(where) +
        ": the NV12 frame's planes could not be read for the copy (Dataflow backend)");
  }
  df::Extent extent;
  extent.rank = 1;
  extent.dims[0] = static_cast<std::int64_t>(b->slot_bytes);
  if (!block) {
    return {DataflowPushOutcome::Stage::Publish,
            b->session.push(b->feed, std::move(loan), stamp, extent, df::Deadline{})};
  }
  if (lossy_block_) {
    return {DataflowPushOutcome::Stage::Publish,
            b->session.push(b->feed, std::move(loan), stamp, extent, df::kNoDeadline)};
  }
  return {DataflowPushOutcome::Stage::Publish,
          b->session.push_waiting(b->feed, std::move(loan), stamp, extent, df::kNoDeadline)};
}

} // namespace simaai::neat::runtime

#endif // SIMANEAT_WITH_DATAFLOW
