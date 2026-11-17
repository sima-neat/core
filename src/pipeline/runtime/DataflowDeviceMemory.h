#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// Device memory on the Dataflow backend (H0b, agent-work/h0b/plan.md 3.2, 3.3, 7.3): Core's DMA-BUF
// tensors to and from the runtime's Blocks. A device input (InputOptions::memory_policy Ev74 or
// Dms0) has a pool of one DMA-BUF per slot. The pull of such a slot is a Tensor of GstSample
// storage over a GstDmaBufMemory of the slot's descriptor, which is what a TensorMemory::EV74 (MLA)
// tensor is, so Core and GStreamer code downstream use it as device memory; a push of such a tensor
// backs the slot with the tensor's own DMA-BUF instead of copying it.
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
#include <memory>
#include <span>

namespace simaai::neat::runtime {

#pragma GCC visibility push(hidden)

class DataflowInputPort;
struct DataflowInputBinding;
struct DataflowPushOutcome;

namespace dataflow_device {

namespace df = pipeline_internal::df;

// DataflowInputPort::push for a device input: the same checks, binding and publish, with a wrap
// that is device memory (wrap_tensor) where a host input's is CPU storage. Before them, GStreamer's
// guard of a device-visible route (enforce_device_visible_tensors_or_throw): a CPU-backed tensor,
// which would be copied into the slot, throws std::runtime_error with GStreamer's message unless
// SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY is set. A device-backed tensor is never refused, wrapped
// or copied; nor is a Tensor from this Run's alloc() that the push publishes (`commit_allocation`,
// alone in the item); nor an image cv::Mat, which GStreamer's ingress copies without asking:
// `may_wrap` false says the item is one (Run::push(cv::Mat) on a device input).
DataflowPushOutcome push(DataflowInputPort& port, std::span<const Tensor> item,
                         const Stamp& proposal, bool block, const char* where,
                         bool commit_allocation, std::int64_t pushed_seq, bool may_wrap);

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

// The CPU's view of the device tensors pushed into an input whose consumer reads through the
// CPU's address (a model route whose first stage is an A65 kernel, H1). A tensor pushed again is
// given the mapping it had, so the operator knows its buffer again (a wrapped Block is known by
// its address, descriptor, offset and holder) and nothing is mapped or admitted per frame. The
// mapping is Core's own read-only mmap of the tensor's DMA-BUF, not a map of its GStreamer
// memory: it lives as long as an operator keeps the buffer, and a held GstMemory map would refuse
// the application's next write map of the same tensor. An entry does not keep its mapping alive
// (whoever holds the Block does); a dead one is replaced.
class CpuWraps;
std::shared_ptr<CpuWraps> make_cpu_wraps();

// Push. The Block that backs a slot of a device port (`port`: Ev74 or Dms0) with the DMA-BUF of
// `t`, whose layout is a whole slot's (dataflow_wrap_layout). True when all of these hold, and a
// copy into the slot otherwise:
//   - the storage is a GstSample over one standard DMA-BUF memory, with a descriptor;
//   - it is the port's kind (an Ev74 port: placed on SIMA_CVU; a Dms0 port: on SIMA_MLA), or,
//     with `any_device_memory` (a model route, whose only consumer is the MLA: it reads a DMA-BUF
//     of either heap), placed on either;
//   - it is Core's own (Tensor creation, or a transfer), whose writer ended its epoch through the
//     Runner when its map ended: neither a tensor pulled from a Dataflow device pool (it leases a
//     slot of that pool, D15) nor a buffer a running graph produced (another exporter's protocol,
//     out of H0b);
//   - `slot_bytes` of it lie in the memory from the tensor's first byte, and that byte's offset
//     in the DMA-BUF is aligned to `align_mask` + 1 (a DMA-BUF maps at a page, so its CPU address
//     in any map is aligned as its offset is).
// On a port of fixed shape the Block is the descriptor alone (no CPU address: the runtime takes
// such a wrap into a device port, and nothing is mapped at the push); its holder is the tensor's
// TensorBuffer, as a host wrap's, kept until the message's last drop. On a port of bounded shape
// the runtime wants the CPU address: one read map of the storage, whose address must be aligned
// too, kept with the storage until the message's last drop.
// With `cpu` (a route whose first operator reads through the CPU's address) the Block also
// carries the address of the tensor's first byte in a mapping from `cpu` (CpuWraps), whatever the
// port's shape; its holder is that mapping, which keeps the tensor's TensorBuffer.
bool wrap_tensor(const Tensor& t, const df::PortSpec& port, std::size_t slot_bytes,
                 std::uintptr_t align_mask, df::Block& out, bool any_device_memory = false,
                 CpuWraps* cpu = nullptr);

// Pull. The storage of the tensor of `message`, whose `block` (Message::block) has a descriptor:
//   - a wrap (the Block has a holder): the pushed tensor's storage itself (D14), with the pushed
//     tensor's byte offset. Nothing was mapped for it unless the port's shape is bounded (then
//     the wrap's read map ends when the caller drops the message); a CPU read of the pulled
//     tensor maps through that storage, as of any EV74 tensor;
//   - a slot of the device pool: a GstSample over one GstDmaBufMemory of the slot's descriptor
//     (not closed with the memory: the pool owns it), placed as make_driver_dmabuf_storage places
//     a TensorMemory::EV74 (MLA) tensor. The memory keeps a copy of the Message, and so the slot,
//     until the memory itself is freed: whatever maps or imports that memory has ended before the
//     pool hands the slot out again (a retained read map would refuse the next write epoch). Its
//     CPU read map is the slot's own mapping, which the pool's writer left current.
// Throws NeatError when GStreamer gives no memory for the slot.
std::shared_ptr<TensorBuffer> pulled_storage(const df::Message& message, const df::Block& block,
                                             std::int64_t& byte_offset);

} // namespace dataflow_device

#pragma GCC visibility pop

} // namespace simaai::neat::runtime
