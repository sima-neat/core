#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// Device memory on the Dataflow backend: Core's DMA-BUF tensors to and from the runtime's Blocks. A
// device input (InputOptions::memory_policy Ev74 or Dms0) has a pool of one DMA-BUF per slot.
//
// A device tensor on this backend is no GStreamer object. Its storage is StorageKind::DeviceHandle:
// one DMA-BUF of the runtime's memory, the Block its map_fn carries, of one of two kinds:
//   - a slot of a device pool, pulled (pulled_storage): read-only, the slot leased until the
//     storage goes;
//   - Core's own device memory (make_device_storage): what Tensor creation and the transfers place
//     in EV74 or MLA memory in a process on this backend, from the runtime's Runner-backed source.
// A device port backs its slot with such a tensor's DMA-BUF instead of copying it (wrap_tensor), as
// it does with a GstSample over one standard DMA-BUF (a tensor a GStreamer object gave the
// application, which this backend takes as it is but never makes).
//
// Everything a device input does differently is compiled in DataflowDeviceMemory.cpp: its push
// (the port's inline funnel, instantiated there with the device wrap) and the tensors of its
// pulled messages. The Run (DataflowRun.cpp) reaches them by one branch per verb, so the code it
// compiles for a host input does not change with what a device input needs.

#include "pipeline/TensorCore.h"
#include "pipeline/internal/DataflowRuntime.h"
#include "pipeline/internal/SampleProjection.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>

namespace simaai::neat::runtime {

#pragma GCC visibility push(hidden)

class DataflowInputPort;
struct DataflowInputBinding;
struct DataflowPushItem;
struct DataflowPushOutcome;
struct DataflowPulledTensor;
struct DataflowModelRoute;

namespace dataflow_device {

namespace df = pipeline_internal::df;

// Whether a storage of `kind` can be device memory a device port takes by its descriptor: this
// backend's (DeviceHandle) or a GStreamer object's (GstSample). device_memory says whether it is.
constexpr bool device_storage_kind(StorageKind kind) noexcept {
  return kind == StorageKind::DeviceHandle || kind == StorageKind::GstSample;
}

// The DMA-BUF of a device storage: its descriptor (borrowed: valid while the storage lives), the
// offset of the storage's first byte in it, and the storage's bytes from there.
struct DeviceMemory {
  int fd = -1;
  std::size_t fd_offset = 0;
  std::size_t size = 0;
};

// The DMA-BUF of `storage` when it is one a device port can take: Core's own device memory on this
// backend (make_device_storage), or a GstSample over one standard DMA-BUF memory. False for every
// other storage, a pulled slot's included (a wrap of it would pin a slot of its pool).
bool device_memory(const TensorBuffer& storage, DeviceMemory& out);

// Core's own device memory on this backend: `bytes` of `target`'s memory (SIMA_CVU: EV74 memory,
// linux,cma; SIMA_MLA: the MLA's DMS0 heap) from the runtime's Runner-backed source, filled by
// `fill` inside the CPU's write epoch, whose end (the cache clean) makes the bytes visible to a
// device before this returns. The storage: DeviceHandle, placed on `target` with its target flags,
// `bytes` long, no CPU pointer (`data` null, as a DMA-BUF carrier's); its map is the Runner's write
// epoch for a write (a second writer at once is refused: an empty Mapping) and the CPU's own view
// for a read, which is current because only the CPU writes this memory (a device reads it). Memory
// released by its last holder is kept for the next storage of the same size (at most 128
// allocations, 256 MiB). Throws NeatError when this build has no device memory, when the heap has
// none, or when the write epoch fails; std::exception from `fill` passes through.
std::shared_ptr<TensorBuffer>
make_device_storage(const Device& target, std::size_t bytes,
                    const std::function<void(std::byte* data, std::size_t size)>& fill);

// prewarm_runtime() in a process on this backend: the device memory's one-time setup that the first
// device tensor or device pool would do otherwise (the Runner's capability probe and the heap's
// first allocation), with one page allocated and freed. Nothing when this build has no device
// memory; a failure is left for that first use to report.
void prewarm_device_memory() noexcept;

// DataflowInputPort::push for a device input: the same checks, binding and publish, with a wrap
// that is device memory (wrap_tensor) where a host input's is CPU storage. Before them, GStreamer's
// guard of a device-visible route (enforce_device_visible_tensors_or_throw): a CPU-backed tensor,
// which would be copied into the slot, throws std::runtime_error with GStreamer's message unless
// SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY is set. A device-backed tensor is never refused, wrapped
// or copied; nor is a Tensor from this Run's alloc() that the push publishes (`commit_allocation`,
// alone in the item); nor an image cv::Mat, which GStreamer's ingress copies without asking:
// `may_wrap` false says the item is one (Run::push(cv::Mat) on a device input).
DataflowPushOutcome push(DataflowInputPort& port, const DataflowPushItem& item);

// A push of one tensor into a model route whose frames change size (`route.frames`), which
// dataflow_device::push hands every such push (DataflowFrameInput.cpp). The frame's geometry is
// stated in the stamp's user word, which the route's stages and the pull read
// (dataflow_frame_word); a frame the route cannot run is refused before anything is asked of the
// Session. Then the device push as for any input: GStreamer's guard, the checks, the slot (the
// frame's own DMA-BUF when it is in EV74 memory, whatever its size), the publish. An NV12 frame
// goes to the port's own push of planes (DataflowInputPort::device_push_planes). `item` is one
// tensor, the frame.
DataflowPushOutcome push_frame(DataflowInputPort& port, const DataflowModelRoute& route,
                               const DataflowPushItem& item);

// Core's own read-only mappings of the device tensors pushed into one input whose wraps carry the
// CPU's address of their bytes: a model route whose first stage is an A65 kernel (it reads
// through that address), and one whose frames change size (the runtime takes a wrap into a
// bounded device port only with it). A tensor pushed again is given the mapping it had, so the
// operator knows its buffer again (a wrapped Block is known by its address, descriptor, offset
// and holder) and nothing is mapped or admitted per frame. A mapping is an mmap of the tensor's
// DMA-BUF, not a map of its GStreamer memory: it lives as long as an operator keeps the buffer,
// and a held GstMemory map would refuse the application's next write map of the same tensor.
// Its holder keeps the tensor's TensorBuffer; it is the one holder type of every device wrap
// with a CPU address, which is how a pull finds the pushed storage again (pulled_storage).
// Sixteen tensors in rotation are served without a map; a seventeenth replaces the oldest entry.
// An entry does not keep its mapping alive (whoever holds the Block does); a dead one is replaced.
class MappedWraps;
std::shared_ptr<MappedWraps> make_mapped_wraps();

// A model route whose frames change size (DataflowFrameInput.cpp), in wrap_tensor's place in its
// push: the Block that backs a slot of the route's bounded Ev74 port with the DMA-BUF of the frame
// `t`, whatever the frame's size up to the port's (dataflow_wrap_layout: dense, rows not padded).
// It is wrap_tensor's Block for a tensor of the frame's own bytes, with the CPU's address of its
// first byte from `mapped`. Its size is the frame's, which the push's extent is held against;
// nothing is copied, and nothing is cleaned (an EV74 tensor's writer made its bytes visible when
// its map ended).
bool wrap_frame(const Tensor& t, std::size_t slot_bytes, std::uintptr_t align_mask, df::Block& out,
                MappedWraps* mapped);

// Core's cross-Run rule (DataflowInputPort::enforce_transferable) for the tensors of one item,
// asked of every push into a device input that skips the other checks for a device tensor of a
// validated geometry: a tensor whose storage a running graph's stream produced must carry a live
// zero-copy loan, or the push throws. Whether a tensor is such a one is read from the tensor
// itself; the Sample that the rule reads is built only when it is.
void enforce_transferable(std::span<const Tensor> item, const char* where);

// The Sample and the TensorList of a device input's pulled message, as the codec's make_sample
// and fill_tensors give a host input's: one leaf (a device port carries no record), over the
// message's device memory (pulled_storage), or over a CPU copy with `owned`
// (OutputMemory::Owned; of a tensor wrapped by its descriptor, read through one map of the pushed
// storage for the copy). The leaf is labelled as a GStreamer Run labels a pulled DMA-BUF tensor
// (Tensor::device CPU); the placement is its storage's. `tensors` (empty) lends its buffer to the
// Sample's list.
Sample pulled_sample(const DataflowInputBinding& binding,
                     const pipeline_internal::StreamNames& streams, bool owned, df::Message message,
                     TensorList tensors);
void pulled_tensors(const DataflowInputBinding& binding,
                    const pipeline_internal::StreamNames& streams, bool owned, df::Message message,
                    TensorList& out);

// One tensor of a model route's pulled message, the stage's published tensor `pulled` over
// `storage` (the stage's output buffer, from `byte_offset`), as InputStreamPull.cpp's
// apply_tensor_set_descriptor leaves a GStreamer Run's: the one dense leaf of a route whose last
// stage publishes one tensor. What a route publishes beyond that (DataflowPulledTensor::
// beyond_leaf: an image preproc's frame semantics, several tensors in one buffer) is
// fill_dataflow_stage_tensors' (DataflowModelRoute.h), which fills each tensor with this.
void fill_stage_leaf(Tensor& tensor, const DataflowPulledTensor& pulled,
                     std::shared_ptr<TensorBuffer> storage, std::int64_t byte_offset);

// Push. The Block that backs a slot of a device port (`port`: Ev74 or Dms0) with the DMA-BUF of
// `t`, whose layout is a whole slot's (dataflow_wrap_layout). True when all of these hold, and a
// copy into the slot otherwise:
//   - the storage has a DMA-BUF a device port takes (device_memory);
//   - it is the port's kind (an Ev74 port: placed on SIMA_CVU; a Dms0 port: on SIMA_MLA), or,
//     with `any_device_memory` (a model route, whose only consumer is the MLA: it reads a DMA-BUF
//     of either heap), placed on either;
//   - it is Core's own (Tensor creation, or a transfer), whose writer ended its epoch through the
//     Runner when its map ended: neither a tensor pulled from a Dataflow device pool (it leases a
//     slot of that pool: a wrap would pin it) nor a buffer a running graph produced (another
//     exporter's protocol, which this backend does not follow);
//   - `slot_bytes` of it lie in the memory from the tensor's first byte, and that byte's offset
//     in the DMA-BUF is aligned to `align_mask` + 1 (a DMA-BUF maps at a page, so its CPU address
//     in any map is aligned as its offset is).
// On a port of fixed shape the Block is the descriptor alone (no CPU address: the runtime takes
// such a wrap into a device port, and nothing is mapped at the push); its holder is the tensor's
// TensorBuffer, as a host wrap's, kept until the message's last drop. On a port of bounded shape
// the runtime wants the CPU address: one read map of the storage, whose address must be aligned
// too, kept with the storage until the message's last drop.
// With `mapped` the Block also carries the address of the tensor's first byte in a mapping from
// `mapped`, whatever the port's shape; its holder is that mapping.
bool wrap_tensor(const Tensor& t, const df::PortSpec& port, std::size_t slot_bytes,
                 std::uintptr_t align_mask, df::Block& out, bool any_device_memory = false,
                 MappedWraps* mapped = nullptr);

// Pull. The storage of the tensor of `message`, whose `block` (Message::block) has a descriptor:
//   - a wrap (the Block has a holder): the pushed tensor's storage itself, with the pushed
//     tensor's byte offset. A wrap by its descriptor alone (no CPU address) holds the
//     TensorBuffer itself; a wrap with a CPU address holds the mapping it came from (a read map
//     of the storage, on a bounded port, or one of MappedWraps), which ends when the last holder
//     drops it. A CPU read of the pulled tensor maps through that storage, as of any EV74 tensor;
//   - a slot of the device pool: a DeviceHandle storage of the slot's bytes, placed and segmented
//     as a TensorMemory::EV74 (MLA) tensor of that size is (its one segment `memory0`). It keeps a
//     copy of the Message, and so the slot, until the storage itself is freed: whatever maps the
//     storage has ended before the pool hands the slot out again. Its CPU read map is the slot's
//     own mapping, which the pool's writer left current; a write map is refused.
// Throws NeatError when the storage of a pushed device tensor changed while the Run held it.
std::shared_ptr<TensorBuffer> pulled_storage(const df::Message& message, const df::Block& block,
                                             std::int64_t& byte_offset);

} // namespace dataflow_device

#pragma GCC visibility pop

} // namespace simaai::neat::runtime
