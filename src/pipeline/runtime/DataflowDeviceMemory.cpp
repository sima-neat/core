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
#include "pipeline/runtime/DataflowTensorCodec.h"

#include <gst/allocators/gstdmabuf.h>
#include <gst/allocators/gstfdmemory.h>
#include <gst/gst.h>

#include <span>
#include <typeinfo>
#include <utility>

namespace simaai::neat::runtime::dataflow_device {

namespace {

namespace codec = dataflow_codec;

// What a wrapped device message keeps: the pushed tensor's storage and the read map its CPU
// address comes from. The pull takes `storage`; the map ends with the last drop of the message.
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
  // cleaned at its push, so the view is current and a read needs no epoch and no second mmap.
  storage->map_fn = SlotMapper{storage->holder, const_cast<std::byte*>(bytes.data()), bytes.size()};
  return storage;
}

} // namespace

bool wrap_tensor(const Tensor& t, df::MemoryKind port, std::size_t slot_bytes,
                 std::uintptr_t align_mask, df::Block& out) {
  if (port == df::MemoryKind::Host || !t.storage || t.storage->kind != StorageKind::GstSample ||
      !t.storage->holder || t.byte_offset < 0) {
    return false;
  }
  const TensorBuffer& storage = *t.storage;
  if (storage.device.type != device_of(port).type ||
      storage.map_fn.target_type() == typeid(SlotMapper) ||
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
  // The CPU address of the message's bytes (an Owned pull copies from it): one read map of the
  // tensor's storage, kept for the message's life. A refusal ends it here.
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
  out.size = slot_bytes;
  out.fd = fd;
  out.fd_offset = static_cast<std::size_t>(memory_offset) + offset;
  out.kind = port;
  out.holder = std::move(hold);
  return true;
}

std::shared_ptr<TensorBuffer> pulled_storage(const df::Message& message, const df::Block& block,
                                             std::int64_t& byte_offset) {
  if (block.holder) {
    // Every wrap of a Feed is Core's own (DataflowInputPort), and a wrap with a descriptor is
    // wrap_tensor's.
    const auto* hold = static_cast<const WrapHold*>(block.holder.get());
    byte_offset = hold->byte_offset;
    return hold->storage;
  }
  byte_offset = 0;
  return slot_storage(message, block);
}

DataflowPushOutcome push(DataflowInputPort& port, std::span<const Tensor> item,
                         const Stamp& proposal, bool block, const char* where,
                         bool commit_allocation, std::int64_t pushed_seq, bool may_wrap) {
  // GStreamer's guard of a device-visible route, by its own function: a CPU-backed tensor is
  // refused unless SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY allows the copy into the slot. Not
  // asked for a Tensor from this Run's alloc() that the push publishes as its slot (the funnel's
  // own condition), nor for an image cv::Mat (`may_wrap` false), which GStreamer copies unasked.
  if (may_wrap &&
      !(commit_allocation && item.size() == 1U && allocation_of(item.front(), port.core()))) {
    enforce_device_visible_tensors_or_throw(item, where);
  }
  return port.push(item, proposal, block, where, commit_allocation, pushed_seq, may_wrap,
                   /*device_port=*/true);
}

namespace {

// The storage of a pulled leaf: device memory, or with `owned` an independent CPU copy (the slot
// returns to the pool as the message goes).
std::shared_ptr<TensorBuffer> leaf_storage(bool owned, const df::Message& message,
                                           std::int64_t& byte_offset) {
  return owned ? codec::owned_copy(message.bytes())
               : pulled_storage(message, message.block(), byte_offset);
}

} // namespace

Sample pulled_sample(const DataflowInputBinding& binding,
                     const pipeline_internal::StreamNames& streams, bool owned, df::Message message,
                     TensorList tensors) {
  Stamp stamp;
  codec::core_stamp(message.stamp(), streams, stamp);
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
  Tensor& leaf = out.emplace_back();
  codec::fill_leaf(leaf, binding, message.extent(), std::move(storage), offset);
  leaf.envelope = std::move(envelope);
}

} // namespace simaai::neat::runtime::dataflow_device

#endif // SIMANEAT_WITH_DATAFLOW
