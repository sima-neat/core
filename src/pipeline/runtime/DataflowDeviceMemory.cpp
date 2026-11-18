// Device memory on the Dataflow backend (DataflowDeviceMemory.h; H0b): the Blocks and storages of
// device tensors, and a device input's push and pulled tensors, which instantiate the port's push
// funnel and use the codec here so that DataflowRun.cpp compiles neither for a device.
#if defined(SIMANEAT_WITH_DATAFLOW)

#include "pipeline/runtime/DataflowDeviceMemory.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/SimaaiGstCompat.h"
#include "pipeline/internal/TensorTransfer.h"
#include "pipeline/runtime/DataflowInputPort.h"
#include "pipeline/runtime/DataflowModelRoute.h"
#include "pipeline/runtime/DataflowTensorCodec.h"

#include <gst/allocators/gstdmabuf.h>
#include <gst/allocators/gstfdmemory.h>
#include <gst/gst.h>

#include <sys/mman.h>
#include <unistd.h>

#include <array>
#include <mutex>
#include <optional>
#include <span>
#include <typeinfo>
#include <utility>
#include <variant>

namespace simaai::neat::runtime::dataflow_device {

namespace {

namespace codec = dataflow_codec;

// What a device message wrapped with a CPU address keeps (a port of bounded shape, wrap_tensor):
// the pushed tensor's storage and the read map its CPU address comes from. The pull takes
// `storage`; the map ends with the last drop of the message. A message wrapped by its descriptor
// alone keeps the pushed tensor's TensorBuffer itself, as a host wrap does, and no map.
struct WrapHold {
  std::shared_ptr<TensorBuffer> storage;
  std::int64_t byte_offset = 0;
  Mapping mapping;
};

// The map_fn of a pulled slot's storage: the slot's own mapping, for reading. One type, so that a
// push recognises a tensor that leases a slot of a device pool (D15) by it, as the codec's
// SlotLeaseMapper marks a host slot's lease. The pulled tensor is read-only: a CPU write needs the
// pool's write epoch, which only a push holds.
struct SlotMapper {
  std::shared_ptr<void> holder; ///< the GstSample: its memory keeps the Message
  void* data = nullptr;
  std::size_t size = 0;
  Mapping operator()(MapMode mode) const {
    Mapping mapping;
    if (mode == MapMode::Read) {
      mapping.data = data;
      mapping.size_bytes = size;
      mapping.keepalive = holder;
    }
    return mapping;
  }
};

Device device_of(df::MemoryKind kind) {
  return {kind == df::MemoryKind::Dms0 ? DeviceType::SIMA_MLA : DeviceType::SIMA_CVU, 0};
}

std::uint64_t target_flags_of(df::MemoryKind kind) {
  return static_cast<std::uint64_t>(kind == df::MemoryKind::Dms0 ? GST_SIMAAI_MEMORY_TARGET_DMS0
                                                                 : GST_SIMAAI_MEMORY_TARGET_EV74);
}

// The one GstMemory of the storage's GstSample when it is a standard DMA-BUF; nullptr otherwise.
GstMemory* dmabuf_memory(const TensorBuffer& storage) {
  auto* sample = static_cast<GstSample*>(storage.holder.get());
  GstBuffer* buffer = sample && GST_IS_SAMPLE(sample) ? gst_sample_get_buffer(sample) : nullptr;
  if (!buffer || gst_buffer_n_memory(buffer) != 1U) {
    return nullptr;
  }
  GstMemory* memory = gst_buffer_peek_memory(buffer, 0U);
  return memory && gst_is_dmabuf_memory(memory) ? memory : nullptr;
}

// A slot of a device pool as a TensorMemory::EV74 (MLA) tensor's storage.
std::shared_ptr<TensorBuffer> slot_storage(const df::Message& message, const df::Block& block) {
  // Process-wide and never freed: memories of it may outlive every Run.
  static GstAllocator* const allocator = gst_dmabuf_allocator_new();
  static const GQuark message_quark = g_quark_from_static_string("sima-neat-dataflow-slot");
  const std::span<const std::byte> bytes = message.bytes();
  GstMemory* memory = gst_dmabuf_allocator_alloc_with_flags(
      allocator, block.fd, block.fd_offset + bytes.size(), GST_FD_MEMORY_FLAG_DONT_CLOSE);
  GstBuffer* buffer = memory ? gst_buffer_new() : nullptr;
  if (!buffer) {
    if (memory) {
      gst_memory_unref(memory);
    }
    pipeline_internal::error_util::throw_session_error(
        error_codes::kMemoryAllocationFailed,
        "Run::pull: GStreamer gave no DMA-BUF memory for a device slot (Dataflow backend)");
  }
  if (block.fd_offset != 0U) {
    gst_memory_resize(memory, static_cast<gssize>(block.fd_offset), bytes.size());
  }
  // The memory, not the buffer, keeps the slot: a GstMemory can outlive the buffer it came in.
  gst_mini_object_set_qdata(GST_MINI_OBJECT_CAST(memory), message_quark, new df::Message(message),
                            [](gpointer kept) { delete static_cast<df::Message*>(kept); });
  gst_buffer_append_memory(buffer, memory);
  GstSample* sample = gst_sample_new(buffer, nullptr, nullptr, nullptr);
  gst_buffer_unref(buffer);
  std::shared_ptr<TensorBuffer> storage = pipeline_internal::make_driver_dmabuf_storage(
      sample, device_of(block.kind), target_flags_of(block.kind));
  gst_sample_unref(sample);
  if (!storage) {
    pipeline_internal::error_util::throw_session_error(
        error_codes::kMemoryAllocationFailed,
        "Run::pull: no storage for a device slot's DMA-BUF (Dataflow backend)");
  }
  // D5: the CPU reads the slot through the pool's own mapping. The slot was written by the CPU and
  // cleaned at its push, or (H1, a model stage's output) by the device, and the Worker invalidated
  // the CPU's view of it before it published the message: either way the view is current and a
  // read needs no epoch and no second mmap.
  storage->map_fn = SlotMapper{storage->holder, const_cast<std::byte*>(bytes.data()), bytes.size()};
  return storage;
}

} // namespace

// One read-only mapping per pushed tensor, found again by the tensor's storage and byte offset.
// Sixteen tensors in rotation are served without a map; a seventeenth replaces the oldest entry.
class CpuWraps {
public:
  // The mapping of `slot_bytes` at `fd_offset` of the DMA-BUF `fd`, which is `storage`'s, with
  // the tensor's first byte in `data`; null when the kernel gives none.
  [[gnu::noinline]] std::shared_ptr<WrapHold> get(const std::shared_ptr<TensorBuffer>& storage,
                                                  std::int64_t byte_offset, int fd,
                                                  std::size_t fd_offset, std::size_t slot_bytes,
                                                  std::byte*& data) {
    static const std::size_t page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    const std::size_t lead = fd_offset % page;
    std::lock_guard<std::mutex> lock(mu_);
    for (Entry& entry : entries_) {
      if (entry.storage == storage.get() && entry.byte_offset == byte_offset &&
          entry.bytes == slot_bytes) {
        if (std::shared_ptr<WrapHold> hold = entry.hold.lock()) {
          data = static_cast<std::byte*>(hold->mapping.data) + lead;
          return hold;
        }
      }
    }
    const std::size_t length = lead + slot_bytes;
    void* const base =
        ::mmap(nullptr, length, PROT_READ, MAP_SHARED, fd, static_cast<off_t>(fd_offset - lead));
    if (base == MAP_FAILED) {
      return nullptr;
    }
    auto hold = std::make_shared<WrapHold>();
    hold->storage = storage;
    hold->byte_offset = byte_offset;
    hold->mapping.data = base;
    hold->mapping.size_bytes = length;
    hold->mapping.unmap = [base, length] { ::munmap(base, length); };
    Entry& entry = entries_[next_++ % entries_.size()];
    entry.storage = storage.get();
    entry.byte_offset = byte_offset;
    entry.bytes = slot_bytes;
    entry.hold = hold;
    data = static_cast<std::byte*>(base) + lead;
    return hold;
  }

private:
  struct Entry {
    const TensorBuffer* storage = nullptr; ///< compared only while `hold` lives, which keeps it
    std::int64_t byte_offset = 0;
    std::size_t bytes = 0;
    std::weak_ptr<WrapHold> hold;
  };
  std::mutex mu_;
  std::array<Entry, 16> entries_;
  std::size_t next_ = 0;
};

std::shared_ptr<CpuWraps> make_cpu_wraps() {
  return std::make_shared<CpuWraps>();
}

bool wrap_tensor(const Tensor& t, const df::PortSpec& port_spec, std::size_t slot_bytes,
                 std::uintptr_t align_mask, df::Block& out, bool any_device_memory, CpuWraps* cpu) {
  const df::MemoryKind port = port_spec.memory;
  if (port == df::MemoryKind::Host || !t.storage || t.storage->kind != StorageKind::GstSample ||
      !t.storage->holder || t.byte_offset < 0) {
    return false;
  }
  const TensorBuffer& storage = *t.storage;
  const bool kind_fits = storage.device.type == device_of(port).type ||
                         (any_device_memory && (storage.device.type == DeviceType::SIMA_CVU ||
                                                storage.device.type == DeviceType::SIMA_MLA));
  if (!kind_fits || storage.map_fn.target_type() == typeid(SlotMapper) ||
      pipeline_internal::tensor_has_device_gstsample_producer_lifetime(t)) {
    return false;
  }
  GstMemory* memory = dmabuf_memory(storage);
  const int fd = memory ? gst_dmabuf_memory_get_fd(memory) : -1;
  if (fd < 0) {
    return false;
  }
  gsize memory_offset = 0;
  const gsize memory_size = gst_memory_get_sizes(memory, &memory_offset, nullptr);
  const auto offset = static_cast<std::size_t>(t.byte_offset);
  if (offset > memory_size || memory_size - offset < slot_bytes) {
    return false;
  }
  out.size = slot_bytes;
  out.fd = fd;
  out.fd_offset = static_cast<std::size_t>(memory_offset) + offset;
  out.kind = port;
  if (cpu) {
    // The first operator reads through the CPU's address: the tensor's own mapping (CpuWraps).
    // A DMA-BUF maps at a page, so the address is aligned as the offset is.
    if ((out.fd_offset & align_mask) != 0U) {
      return false;
    }
    std::byte* data = nullptr;
    std::shared_ptr<WrapHold> hold =
        cpu->get(t.storage, t.byte_offset, fd, out.fd_offset, slot_bytes, data);
    if (!hold) {
      return false;
    }
    out.data = data;
    out.holder = std::move(hold);
    return true;
  }
  const df::TensorSpec* const spec = std::get_if<df::TensorSpec>(&port_spec.payload);
  if (spec && spec->shape.fixed()) {
    // By the descriptor alone: nothing is mapped. The alignment asked of the CPU address is asked
    // of the offset in the DMA-BUF, which is where that address would be in any map of it.
    if ((out.fd_offset & align_mask) != 0U) {
      return false;
    }
    out.data = nullptr;
    out.holder = t.storage;
    return true;
  }
  // A port of bounded shape takes a wrap only with its CPU address (the runtime's rule): one read
  // map of the tensor's storage, kept for the message's life. A refusal ends it here.
  auto hold = std::make_shared<WrapHold>();
  hold->mapping = storage.map(MapMode::Read);
  if (!hold->mapping.data || hold->mapping.size_bytes < offset ||
      hold->mapping.size_bytes - offset < slot_bytes) {
    return false;
  }
  auto* data = static_cast<std::byte*>(hold->mapping.data) + offset;
  if ((reinterpret_cast<std::uintptr_t>(data) & align_mask) != 0U) {
    return false;
  }
  hold->storage = t.storage;
  hold->byte_offset = t.byte_offset;
  out.data = data;
  out.holder = std::move(hold);
  return true;
}

std::shared_ptr<TensorBuffer> pulled_storage(const df::Message& message, const df::Block& block,
                                             std::int64_t& byte_offset) {
  if (block.holder && !block.data) {
    // Wrapped by its descriptor (wrap_tensor): the holder is the pushed tensor's TensorBuffer, and
    // the tensor's first byte is where the Block's offset says, past the memory's own.
    auto storage = std::static_pointer_cast<TensorBuffer>(block.holder);
    GstMemory* memory = dmabuf_memory(*storage);
    gsize memory_offset = 0;
    if (memory) {
      gst_memory_get_sizes(memory, &memory_offset, nullptr);
    }
    if (!memory || block.fd_offset < memory_offset) {
      // The application replaced the storage's memory while the Run held it (against D16).
      pipeline_internal::error_util::throw_session_error(
          error_codes::kMemoryAllocationFailed,
          "Run::pull: the storage of a pushed device tensor changed while the Run held it "
          "(Dataflow backend)");
    }
    byte_offset = static_cast<std::int64_t>(block.fd_offset - memory_offset);
    return storage;
  }
  if (block.holder) {
    // Every wrap of a Feed is Core's own (DataflowInputPort), and a wrap with a descriptor and a
    // CPU address is wrap_tensor's WrapHold.
    const auto* hold = static_cast<const WrapHold*>(block.holder.get());
    byte_offset = hold->byte_offset;
    return hold->storage;
  }
  byte_offset = 0;
  return slot_storage(message, block);
}

namespace {

// H1: on a model route whose first stage is a kernel, a tensor without the leading unit axes of
// the stage's tensor is pushed as that tensor (GStreamer takes both): a view of the same storage
// with the stage's shape, pushed in its place. nullopt when `item` is not such a tensor (the push
// goes on with it and refuses it). Out of line and cold, so that a push of the stage's own shape,
// and every push on every other route, neither builds nor destroys a Tensor for it.
[[gnu::noinline, gnu::cold]] std::optional<DataflowPushOutcome>
push_as_stage_tensor(DataflowInputPort& port, const DataflowModelRoute& route,
                     std::span<const Tensor> item, const Stamp& proposal, bool block,
                     const char* where, bool commit_allocation, std::int64_t pushed_seq,
                     bool may_wrap) {
  if (route.frames) {
    return push_frame(port, route, item.front(), proposal, block, where, commit_allocation,
                      pushed_seq, may_wrap);
  }
  Tensor view;
  if (&dataflow_model_input_view(route, item.front(), view) != &view) {
    return std::nullopt;
  }
  return push(port, std::span<const Tensor>(&view, 1), proposal, block, where, commit_allocation,
              pushed_seq, may_wrap);
}

} // namespace

DataflowPushOutcome push(DataflowInputPort& port, std::span<const Tensor> item,
                         const Stamp& proposal, bool block, const char* where,
                         bool commit_allocation, std::int64_t pushed_seq, bool may_wrap) {
  if (const DataflowModelRoute* const route = port.config().model.get();
      route && route->kernel_input && item.size() == 1U &&
      item.front().shape.size() != route->input_shape.size()) [[unlikely]] {
    if (std::optional<DataflowPushOutcome> outcome = push_as_stage_tensor(
            port, *route, item, proposal, block, where, commit_allocation, pushed_seq, may_wrap)) {
      return *outcome;
    }
  }
  if (!block) {
    if (const df::Status refusal = port.device_still_full(item, pushed_seq)) {
      return {DataflowPushOutcome::Stage::Alloc, *refusal};
    }
  }
  // GStreamer's guard of a device-visible route, by its own function: a CPU-backed tensor is
  // refused unless SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY allows the copy into the slot. Not
  // asked for a Tensor from this Run's alloc() that the push publishes as its slot (the funnel's
  // own condition), nor for an image cv::Mat (`may_wrap` false), which GStreamer copies unasked.
  if (may_wrap &&
      !(commit_allocation && item.size() == 1U && allocation_of(item.front(), port.core()))) {
    enforce_device_visible_tensors_or_throw(item, where);
  }
  const DataflowPushOutcome outcome = port.push(item, proposal, block, where, commit_allocation,
                                                pushed_seq, may_wrap, /*device_port=*/true);
  port.device_remember(item, outcome);
  return outcome;
}

void enforce_transferable(std::span<const Tensor> item, const char* where) {
  for (const Tensor& tensor : item) {
    // One lookup of the storage's producer mark. The Sample that the rule reads is built only for
    // a tensor that has one: a frame of another graph, not Core's own TensorMemory::EV74 tensor.
    if (pipeline_internal::tensor_has_device_gstsample_producer_lifetime(tensor)) {
      DataflowInputPort::enforce_transferable(sample_from_tensors(TensorList{tensor}), where);
    }
  }
}

namespace {

// The storage of a pulled leaf: device memory, or with `owned` an independent CPU copy (the slot
// returns to the pool as the message goes).
std::shared_ptr<TensorBuffer> leaf_storage(bool owned, const df::Message& message,
                                           std::int64_t& byte_offset) {
  if (!owned) {
    return pulled_storage(message, message.block(), byte_offset);
  }
  const std::span<const std::byte> bytes = message.bytes();
  if (!bytes.empty()) {
    return codec::owned_copy(bytes);
  }
  // A tensor wrapped by its descriptor has no bytes mapped (the runtime's bytes() is empty): the
  // copy reads them through one read map of the pushed storage, which ends here. This is the map
  // a push took before; only an Owned pull pays it now.
  std::int64_t offset = 0;
  const std::shared_ptr<TensorBuffer> pushed = pulled_storage(message, message.block(), offset);
  const std::size_t size = df::used_bytes(*message.spec(), message.extent());
  const Mapping mapping = pushed->map(MapMode::Read);
  const auto first = static_cast<std::size_t>(offset);
  if (!mapping.data || mapping.size_bytes < first || mapping.size_bytes - first < size) {
    pipeline_internal::error_util::throw_session_error(
        error_codes::kMemoryAllocationFailed,
        "Run::pull: a pushed device tensor could not be mapped for the Owned copy (Dataflow "
        "backend)");
  }
  return codec::owned_copy({static_cast<const std::byte*>(mapping.data) + first, size});
}

} // namespace

void fill_stage_leaf(Tensor& tensor, const DataflowPulledTensor& pulled,
                     std::shared_ptr<TensorBuffer> storage, std::int64_t byte_offset) {
  if (storage->kind == StorageKind::GstSample) {
    // The output buffer as a GStreamer Run's appsink hands it out: a DMA-BUF carrier without a
    // device placement, its one segment named as the stage's publication names the allocation.
    storage->device = {DeviceType::CPU, 0};
    storage->sima_mem_target_flags = 0U;
    if (storage->sima_segments.size() == 1U) {
      storage->sima_segments.front().name = pulled.route.segment_name;
    }
  }
  tensor.storage = std::move(storage);
  tensor.dtype = pulled.dtype;
  tensor.layout = pulled.layout;
  tensor.shape = pulled.shape;
  tensor.strides_bytes = pulled.strides_bytes;
  tensor.byte_offset = byte_offset + pulled.byte_offset;
  tensor.device = {DeviceType::CPU, 0};
  tensor.read_only = true;
  tensor.route = pulled.route;
  tensor.semantic.quant = pulled.quant;
  tensor.semantic.tess.emplace().format = pulled.format;
}

Sample pulled_sample(const DataflowInputBinding& binding,
                     const pipeline_internal::StreamNames& streams, bool owned, df::Message message,
                     TensorList tensors) {
  Stamp stamp;
  codec::core_stamp(message.stamp(), streams, stamp);
  if (const DataflowPulledTensor* const pulled = binding.pulled) {
    Sample out;
    out.kind = SampleKind::TensorSet;
    out.owned = owned;
    out.tensors = std::move(tensors);
    out.media_type = "application/vnd.simaai.tensor";
    out.payload_type = PayloadType::Tensor;
    out.payload_tag = pulled->format;
    out.format = pulled->format;
    out.stream_label = pulled->sample_label;
    out.segment_name = pulled->sample_label;
    out.output_index = pulled->route.logical_index;
    out.logical_output_index = pulled->route.logical_index;
    out.route_slot = pulled->route.route_slot;
    std::int64_t offset = 0;
    std::shared_ptr<TensorBuffer> storage = leaf_storage(owned, message, offset);
    if (pulled->beyond_leaf) [[unlikely]] {
      // Behind an image preproc, or several tensors in the stage's one buffer
      // (DataflowModelRoute.cpp, so that this unit's code for every other route is as it was).
      fill_dataflow_stage_tensors(*pulled, storage, message, nullptr, out.tensors);
    } else {
      fill_stage_leaf(out.tensors.emplace_back(), *pulled, std::move(storage), offset);
    }
    pipeline_internal::apply_pulled_stamp(stamp, out);
    return out;
  }
  // Shaped as codec::make_sample shapes a host input's leaf.
  Sample out;
  out.kind = SampleKind::TensorSet;
  out.owned = owned;
  out.tensors = std::move(tensors);
  const bool image = binding.semantic.image.has_value();
  out.media_type = image ? "video/x-raw" : "application/vnd.simaai.tensor";
  out.payload_type = image ? PayloadType::Image : PayloadType::Tensor;
  if (image) {
    out.payload_tag = pipeline_internal::pixel_format_token(binding.semantic.image->format);
    out.format = out.payload_tag;
  }
  std::int64_t offset = 0;
  std::shared_ptr<TensorBuffer> storage = leaf_storage(owned, message, offset);
  codec::fill_leaf(out.tensors.emplace_back(), binding, message.extent(), std::move(storage),
                   offset);
  pipeline_internal::apply_pulled_stamp(stamp, out);
  return out;
}

void pulled_tensors(const DataflowInputBinding& binding,
                    const pipeline_internal::StreamNames& streams, bool owned, df::Message message,
                    TensorList& out) {
  auto envelope = std::make_shared<TensorEnvelope>();
  codec::core_stamp(message.stamp(), streams, envelope->stamp);
  std::int64_t offset = 0;
  std::shared_ptr<TensorBuffer> storage = leaf_storage(owned, message, offset);
  out.clear();
  if (const DataflowPulledTensor* const pulled = binding.pulled) {
    if (pulled->beyond_leaf) [[unlikely]] {
      fill_dataflow_stage_tensors(*pulled, storage, message, envelope, out);
      return;
    }
    Tensor& leaf = out.emplace_back();
    fill_stage_leaf(leaf, *pulled, std::move(storage), offset);
    leaf.envelope = std::move(envelope);
    return;
  }
  Tensor& leaf = out.emplace_back();
  codec::fill_leaf(leaf, binding, message.extent(), std::move(storage), offset);
  leaf.envelope = std::move(envelope);
}

} // namespace simaai::neat::runtime::dataflow_device

namespace simaai::neat::runtime {

// A device input's validated geometries: of the first device tensor (GstSample storage) that
// passed the checks, kept by device_remember, and of the first CPU tensor (pushed with the
// compatibility copy allowed), kept by the push itself. The geometry of a device tensor is not who
// produced its memory: Core's cross-Run rule is asked of every push that repeats it.
const DataflowLeafGeometry* DataflowInputPort::device_known(std::span<const Tensor> item,
                                                            const char* where) const {
  const DataflowLeafGeometry* const known = validated_device_.find_device(item);
  if (!known) {
    if (config_.model && config_.model->frame_input && item.size() == 1U) {
      // H1: in front of the image preproc, a tensor that is not the validated frame: another
      // size or ROI metadata is refused here, with what to do instead.
      refuse_dataflow_frame(*config_.model, item.front(), where);
    }
    return validated_.find(item);
  }
  dataflow_device::enforce_transferable(item, where);
  return known;
}

void DataflowInputPort::device_remember(std::span<const Tensor> item,
                                        const DataflowPushOutcome& outcome) {
  // The push reached the slot, so the item passed every check.
  Bound* const b = bound();
  if (b && outcome.stage != DataflowPushOutcome::Stage::Unbound && validated_device_.empty() &&
      item.size() == 1U && !b->binding.is_record() &&
      DataflowLeafGeometry::keepable_device(item.front())) {
    std::lock_guard<std::mutex> lock(mu_);
    validated_device_.remember(
        item.front(), b->wraps && dataflow_wrap_layout(b->binding, b->slot_bytes, item.front()));
  }
}

// A try_push that wraps a device tensor prepares the wrap (wrap_tensor: its checks, and on a port
// of bounded shape a map) before Session::alloc can refuse it, and a refused call must not pay
// that. The question a host input asks once it was refused
// (still_full: a slot taken and given back) would begin and end a write epoch on a device slot
// whenever there is room. So a device input asks with a Block that alloc cannot take: full links
// refuse it first, before any slot (FR14, counted as the refusal of this try_push, which it is);
// with room the slot is taken, the wrap refused (FR15: an empty Block has neither a CPU address
// nor a descriptor) and the slot returned,
// with no write epoch and nothing kept (block.h: a wrap is checked after the slot is taken, and a
// device port begins no epoch around a wrap). Any other refusal (FR7, a closed input, a stopped
// Session) is the one the slot's own alloc would give, and is the push's.
// full_at keeps what was learnt, as on a host input: a refusal by full links, until a pull.
df::Status DataflowInputPort::device_full(Bound& b) noexcept {
  const std::uint64_t pulled = run_.core.outputs_pulled.load(std::memory_order_relaxed);
  const df::Block none;
  const df::Result<df::WriteLoan> asked =
      b.session.alloc(b.feed, df::Acquire::Fail, df::kNoDeadline, &none);
  const df::Error* const refusal = df::failure(asked);
  if (!refusal || refusal->code == df::ErrorCode::InvalidArgument) {
    // Room: still_full, whose question takes a slot, is not asked after this.
    if (b.full_at.load(std::memory_order_relaxed) != Bound::kNoRefusal) {
      b.full_at.store(Bound::kNoRefusal, std::memory_order_relaxed);
    }
    return std::nullopt;
  }
  if (refusal->code == df::ErrorCode::Backpressure) {
    b.full_at.store(pulled, std::memory_order_relaxed);
  }
  return *refusal;
}

// The refusal of a try_push while the links are known to be full (full_at: alloc refused the last
// one and no pull was counted since), asked before the per-push checks of the tensor's memory
// (the device-visible guard, the cross-Run rule): a refused call does nothing with the tensor, so
// it needs none of them, and costs what a CPU tensor's refusal costs. Only for a device tensor of
// the validated geometry, whose other checks are skipped anyway: every other item is checked
// first, as before, and throws what it threw. None when the links are not known to be full, or
// have room after all: the push then runs in full.
df::Status DataflowInputPort::device_still_full(std::span<const Tensor> item,
                                                std::int64_t pushed_seq) noexcept {
  Bound* const b = bound();
  if (!b ||
      b->full_at.load(std::memory_order_relaxed) !=
          run_.core.outputs_pulled.load(std::memory_order_relaxed) ||
      !validated_device_.find_device(item)) {
    return std::nullopt;
  }
  const df::Status refusal = device_full(*b);
  if (refusal && run_.run_seq && pushed_seq < 0) {
    // A Run of several inputs: a refused push consumes a number of the Run's counter (push()).
    run_.run_seq->fetch_add(1, std::memory_order_relaxed);
  }
  return refusal;
}

} // namespace simaai::neat::runtime

#endif // SIMANEAT_WITH_DATAFLOW
