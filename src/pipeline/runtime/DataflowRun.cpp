// The Dataflow backend's Run state (H0 plan 3.7; H0 step 10, H0c): Core's push, try_push, pull,
// close_input and stop on one Session per public Input, each with one Feed and one link and one
// Fetch per Output the input feeds (a fan-out when there are several). Built only with
// SIMANEAT_DATAFLOW_SOURCE_DIR.
//
// Endpoints: push(name) and pull(name) select the input or output by its public name; the unnamed
// verbs reach the only input or output, and with several fail as GStreamer's graph path does ("no
// unambiguous default input/output"). Each input binds by its declaration, the seed (a graph of
// one input), or its own first push; a pull waits for its own input's binding. In a Run of several
// inputs, Sample::orig_input_seq is the Run's push counter, as on GStreamer's graph path
// (SlotSeqTable); input_seq and Stamp::seq count each input's pushes.
//
// Verbs (WORKER.md 18.1):
//   push      alloc(Acquire::Block) + copy or wrap + push_waiting(kNoDeadline): waits for room
//             on each link of the input whatever its policy and never drops
//             (RunPush.cpp:402-411). When an Output the input feeds has OutputOptions::drop, its
//             queue is lossy for every push, as appsink drop=true is: push(kNoDeadline), which
//             waits on the Block links (the other Outputs, under OverflowPolicy::Block, the one mix
//             of both the compiler accepts) and evicts on the DropOldest ones.
//   try_push  alloc(Acquire::Fail) + copy or wrap + push(passed deadline): each link's policy
//             applies; DroppedIncoming is false and a drop, DroppedOldest true and a drop, a full
//             Block link false and no drop (unit_run_overflow_policy_parity_test.cpp:129-163). A
//             push every link would refuse is answered by alloc itself (the runtime's FR14, A3),
//             before any slot, copy or wrap, and counted by the runtime as that push:
//             alloc_refused maps it to the same outcome. A fan-out is refused only when every link
//             refuses (a DropOldest link never does); one link taking it makes it accepted, and
//             each other link's refusal or eviction is its Output's drop.
//   pull      -1 is kNoDeadline, 0 a passed deadline (a poll), t > 0 now + t ms. The pulled Tensor
//             is CpuExternal over the pool slot, its holder the Message, read-only; a wrapped
//             message's is the pushed storage itself, and the slot returns at the pull. The
//             Outputs of a fan-out lease the one slot each message has: no copy per Output.
//   close_input  Session::close_input(kNoDeadline) of every input: cancels a parked push, keeps
//             what is queued.
//   stop      Session::stop(now + 2 s) of every input, once more on Timeout (RunLifecycle.cpp
//             94-95).
//
// Copy or wrap (W2, agent-work/h0b/plan.md 7.3, D14-D16). A push and a try_push of one tensor wrap
// its storage instead of copying it when that is safe (wrap_layout, wrap_block): CPU storage
// (CpuOwned, CpuExternal) with an owning holder, laid out exactly as a slot of the input's port and
// exactly a slot long (every frame of a fixed-shape input, a frame at a bounded input's bound),
// aligned to the port's slot alignment (64 bytes or more), neither pulled from a Dataflow Run nor a
// Run::alloc one (D15), into a Run of one input, with RunAdvancedOptions::copy_input off.
// Session::alloc then backs the slot with that storage (the runtime's Block, W1), keeping the
// pushed Tensor's TensorBuffer alive (one count on its own control block: no allocation) until the
// last pulled tensor or queued message using it is gone; the pull returns that TensorBuffer (D14).
// Anything else is copied into the slot on the pushing thread, inside push, as before: then the
// caller's buffer is free when push returns. Either way the application must not change a pushed
// tensor's bytes while a Tensor shares its storage (D16; GStreamer's push also retains the tensor
// and copies it later on Core's input thread); copy_input = true copies at the push. EV74 and
// other device tensors (GstSample storage) are copied: every Dataflow port is Host memory until
// H0b's device pools.
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
#include <typeinfo>
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

// The runtime's stamp for what a push proposes: the stream interned (an empty name is the input's
// default stream, `default_stream`), the author's frame_id, source_time (any clock) and flags. emit
// writes seq and publish_time.
df::Stamp runtime_stamp(const Stamp& proposal, pipeline_internal::StreamNames& streams,
                        std::uint32_t default_stream) {
  df::Stamp out;
  out.stream = df::StreamId{proposal.stream_name.empty() ? default_stream
                                                         : streams.intern(proposal.stream_name)};
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
  /// wrap_layout of this geometry against the input's binding (W2), computed once: a repeated
  /// geometry needs only wrap_block's per-storage checks.
  bool wrappable = false;

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

// The map_fn of a pulled tensor's storage while it leases a pool slot: the slot's bytes. One type
// for every lease, as RunAllocationMapper marks a Run::alloc Tensor, so a push recognises such a
// tensor by its map_fn's type and copies it instead of wrapping it (D15: a wrap would pin a slot
// of another pool, or of this one). The same size as make_cpu_external_storage's lambda: kept
// inside the std::function, no allocation.
struct SlotLeaseMapper {
  void* data = nullptr;
  std::size_t size = 0;
  Mapping operator()(MapMode /*mode*/) const {
    Mapping mapping;
    mapping.data = data;
    mapping.size_bytes = size;
    return mapping;
  }
};

// make_cpu_external_storage(bytes, holder) with a SlotLeaseMapper: a read-only CpuExternal view of
// a slot that `holder` (the Message) keeps leased.
std::shared_ptr<TensorBuffer> lease_storage(std::span<const std::byte> bytes,
                                            std::shared_ptr<void> holder) {
  auto storage = std::make_shared<TensorBuffer>();
  storage->kind = StorageKind::CpuExternal;
  storage->device = {DeviceType::CPU, 0};
  storage->size_bytes = bytes.size();
  storage->holder = std::move(holder);
  storage->data = const_cast<std::byte*>(bytes.data());
  storage->map_fn = SlotLeaseMapper{storage->data, bytes.size()};
  return storage;
}

// W2: whether a leaf's bytes are exactly a whole slot of `binding`'s port, so that the slot can be
// backed by the tensor's own storage. The runtime takes a wrap only of exactly slot_bytes (block.h:
// push then checks no extent against it), so the tensor's own bytes must be that size: every frame
// of a fixed-shape port, a frame at the bound of a bounded one (InputOptions::max_*); a smaller
// frame is copied. CPU storage that holds them from the tensor's first byte; a dense port's tensor
// dense and contiguous (a slot holds it tight); a planar port's (NV12, I420) frame at the bound,
// its planes at the offsets and pitches of the port's layout, which planar_layout packs back to
// back (DataflowGraphCompiler.cpp). Everything read here is part of LeafGeometry, so a repeated
// geometry reuses the answer; wrap_block checks the rest per push.
bool wrap_layout(const DataflowInputBinding& binding, std::size_t slot_bytes, const Tensor& t) {
  if (binding.is_record() || slot_bytes == 0U || !t.storage || t.byte_offset < 0 ||
      (t.storage->kind != StorageKind::CpuOwned && t.storage->kind != StorageKind::CpuExternal) ||
      t.has_tiled_storage()) {
    return false;
  }
  const auto offset = static_cast<std::size_t>(t.byte_offset);
  if (offset > t.storage->size_bytes || t.storage->size_bytes - offset < slot_bytes) {
    return false;
  }
  if (binding.planes == DataflowInputBinding::Planes::None) {
    return t.is_dense() && !t.shape.empty() &&
           (t.strides_bytes.empty() || t.strides_bytes.size() == t.shape.size()) &&
           t.is_contiguous() && t.dense_bytes_tight() == slot_bytes;
  }
  const auto& spec = std::get<df::TensorSpec>(binding.spec.payload);
  const auto& planar = std::get<df::Planar>(spec.layout);
  const bool nv12 = binding.planes == DataflowInputBinding::Planes::Nv12;
  if (t.dtype != TensorDType::UInt8 || !(nv12 ? t.is_nv12() : t.is_i420()) ||
      t.planes.size() != planar.planes) {
    return false;
  }
  const Plane* y = t.try_plane(PlaneRole::Y);
  if (!y || y->shape.size() < 2U) {
    return false;
  }
  const std::int64_t rows = y->shape[0];
  const std::int64_t cols = y->shape[1];
  if (rows != spec.shape.dims[0].max || cols != spec.shape.dims[1].max || rows % 2 != 0 ||
      cols % 2 != 0) {
    return false;
  }
  const auto at = [&](PlaneRole role, std::size_t p, std::int64_t r, std::int64_t c) {
    const Plane* plane = t.try_plane(role);
    return plane && plane->shape.size() >= 2U && plane->shape[0] == r && plane->shape[1] == c &&
           plane->byte_offset == planar.plane[p].offset_bytes &&
           (plane->strides_bytes.empty() ? c : plane->strides_bytes[0]) ==
               planar.plane[p].stride_bytes &&
           (plane->strides_bytes.size() < 2U || plane->strides_bytes[1] == 1);
  };
  if (!at(PlaneRole::Y, 0, rows, cols)) {
    return false;
  }
  return nv12 ? at(PlaneRole::UV, 1, rows / 2, cols)
              : at(PlaneRole::U, 1, rows / 2, cols / 2) && at(PlaneRole::V, 2, rows / 2, cols / 2);
}

// W2: the Block that backs a slot with the storage of `t` (wrap_layout held): its first byte,
// `slot_bytes` long, and as keep-alive the Tensor's own TensorBuffer, which keeps the storage's
// holder (one count on the shared_ptr's existing control block; nothing is allocated; an
// application can read the release from its storage's use_count, as from GStreamer's). False,
// and a copy instead, when the storage cannot be kept or must not be: no holder or no address
// (memory nobody owns), a lease on a Dataflow slot or a Run::alloc tensor (D15), or an address
// not aligned to the port's slot alignment (`align_mask` = slot_alignment - 1).
bool wrap_block(const Tensor& t, std::size_t slot_bytes, std::uintptr_t align_mask,
                df::Block& out) {
  const TensorBuffer& storage = *t.storage;
  if (!storage.holder || !storage.data) {
    return false;
  }
  const std::type_info& mapper = storage.map_fn.target_type(); // typeid(void) without a map_fn
  if (mapper == typeid(SlotLeaseMapper) || mapper == typeid(RunAllocationMapper)) {
    return false;
  }
  auto* data = static_cast<std::byte*>(storage.data) + t.byte_offset;
  if ((reinterpret_cast<std::uintptr_t>(data) & align_mask) != 0U) {
    return false;
  }
  out.data = data;
  out.size = slot_bytes;
  out.holder = t.storage;
  return true;
}

// The extent of a wrapped leaf: what copy_into reports for the same tensor.
df::Extent wrapped_extent(const DataflowInputBinding& binding, const Tensor& t) {
  if (binding.planes == DataflowInputBinding::Planes::None) {
    return extent_of(t.shape);
  }
  const Plane* y = t.try_plane(PlaneRole::Y);
  df::Extent extent;
  extent.rank = 3;
  extent.dims[0] = y->shape[0];
  extent.dims[1] = y->shape[1];
  extent.dims[2] = 3;
  return extent;
}

// Sample::orig_input_seq of the messages of one input in a Run of several inputs, by the slot each
// message occupies (H0c). GStreamer's graph path, which every Run of several inputs takes there,
// numbers the pushes of the whole Run in one counter and writes it to orig_input_seq, or keeps a
// pushed orig_input_seq (or input_seq) without counting the push (RunPush.cpp:276-287); the
// runtime's Stamp has no field for it and its seq is per Feed. A Run of one input has no table:
// there orig_input_seq is input_seq (seq - 1), as before.
// The pusher writes a message's entry while it holds the slot's loan, before the publish; a puller
// reads it while it holds a lease on the message, so the slot cannot carry another message in
// between, and the publish's release and the take's acquire order the write before the read. The
// keys are atomics, so a lookup may run beside the insertion of another slot's key; a slot's key
// is inserted once (the pool's slots are fixed), and the table holds twice the pool's slots.
class SlotSeqTable {
public:
  explicit SlotSeqTable(std::uint64_t slots) {
    unsigned bits = 4;
    while ((std::uint64_t{1} << bits) < 2U * slots && bits < 40U) {
      ++bits;
    }
    shift_ = 64U - bits;
    mask_ = (std::size_t{1} << bits) - 1U;
    cells_ = std::make_unique<Cell[]>(mask_ + 1U);
  }

  void set(const void* slot, std::int64_t seq) noexcept {
    for (std::size_t i = index(slot);; i = (i + 1U) & mask_) {
      const void* key = cells_[i].key.load(std::memory_order_relaxed);
      if (key == nullptr &&
          cells_[i].key.compare_exchange_strong(key, slot, std::memory_order_relaxed)) {
        key = slot;
      }
      if (key == slot) {
        cells_[i].value.store(seq, std::memory_order_relaxed);
        return;
      }
    }
  }

  // -1 for a slot the table has never seen.
  std::int64_t get(const void* slot) const noexcept {
    for (std::size_t n = 0, i = index(slot); n <= mask_; ++n, i = (i + 1U) & mask_) {
      const void* key = cells_[i].key.load(std::memory_order_relaxed);
      if (key == slot) {
        return cells_[i].value.load(std::memory_order_relaxed);
      }
      if (key == nullptr) {
        return -1;
      }
    }
    return -1;
  }

private:
  struct Cell {
    std::atomic<const void*> key{nullptr};
    std::atomic<std::int64_t> value{-1};
  };
  std::size_t index(const void* slot) const noexcept {
    // Fibonacci hashing: the high bits of the product.
    return static_cast<std::size_t>(
        (reinterpret_cast<std::uintptr_t>(slot) * std::uint64_t{0x9E3779B97F4A7C15}) >> shift_);
  }
  std::unique_ptr<Cell[]> cells_;
  std::size_t mask_ = 0;
  unsigned shift_ = 60;
};

// The running Session of one input and what was bound with it: one Feed, one Fetch per Output the
// input feeds (link order: DataflowInputConfig::outputs).
struct Bound {
  Bound(DataflowInputBinding b, df::Session s, std::uint32_t feed_op,
        const std::vector<std::uint32_t>& fetch_ops)
      : binding(std::move(b)), session(std::move(s)), feed(df::OperatorRef<df::Feed>{feed_op}) {
    fetches.reserve(fetch_ops.size());
    for (const std::uint32_t op : fetch_ops) {
      fetches.emplace_back(df::OperatorRef<df::Fetch>{op});
    }
  }
  DataflowInputBinding binding;
  df::Session session;
  df::FeedRef feed;
  std::vector<df::FetchRef> fetches;
  std::unique_ptr<SlotSeqTable> run_seqs; ///< a Run of several inputs only: orig_input_seq
  // W2. A push may wrap (`wraps`): a tensor port, copy_input off, and a Run of one input (in a
  // Run of several, orig_input_seq is keyed by the slot's own address, SlotSeqTable, which a
  // wrapped slot does not have: those copy). What a wrap must be: `slot_bytes` long, aligned to
  // `wrap_mask` + 1 (the Feed's port, port_spec.h slot_bytes and slot_alignment). Fixed at bind.
  bool wraps = false;
  std::size_t slot_bytes = 0;
  std::uintptr_t wrap_mask = 0;
  /// Set, before its publish, by the first push that wraps; a pull asks Message::block() only
  /// then (its take orders that publish, and so this store, before it).
  std::atomic<bool> has_wraps{false};
  /// RunCore::outputs_pulled when alloc last refused a wrapping try_push because every link was
  /// full (FR14); kNoRefusal otherwise (DataflowRun::still_full).
  static constexpr std::uint64_t kNoRefusal = ~std::uint64_t{0};
  std::atomic<std::uint64_t> full_at{kNoRefusal};
};

// One public Input of the Run (H0c): its Session, bound by the input's declaration, its seed or its
// first push, and that input's own state. The binding and the validated geometry are written once
// under the Run's mutex and read lock-free (release/acquire).
struct Component {
  std::size_t index = 0;                       ///< DataflowRunConfig::inputs
  const DataflowInputConfig* config = nullptr; ///< DataflowRunConfig::inputs[index]
  std::uint32_t stream = 0;                    ///< its default stream's id (StreamNames)
  /// A blocking push publishes with the policy push, which waits only on Block links, instead of
  /// the waiting push: some Output it feeds has OutputOptions::drop (DataflowGraphCompiler.cpp,
  /// derive_config, rules out the mixes the runtime cannot serve).
  bool lossy_block = false;
  bool fan_out = false; ///< it feeds several Outputs (several links)

  std::unique_ptr<Bound> bound_storage;
  std::atomic<Bound*> bound{nullptr};
  std::atomic<bool> input_closed{false};
  std::unique_ptr<LeafGeometry> validated_leaf_storage;
  std::atomic<const LeafGeometry*> validated_leaf{nullptr};
};

// Where a public Output is read: its Input's component and its link there.
struct OutputSlot {
  std::size_t component = 0;
  std::size_t link = 0;
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

constexpr std::size_t kNoEndpoint = static_cast<std::size_t>(-1);

class DataflowRun final : public DataflowRunState {
public:
  DataflowRun(RunCore& core, DataflowRunConfig config)
      : core_(core), config_(std::move(config)), streams_(config_.inputs.front().default_stream) {
    stream_options_.max_input_bytes = config_.run.advanced.max_input_bytes;
  }

  // Each Session is destroyed with the Run state, after RunCore::close ran stop (A2: a destructor
  // after a stop that timed out only polls). Messages still held by pulled tensors, and slots
  // still lent by alloc(), keep their pool alive on the runtime's retirement list.
  ~DataflowRun() override = default;

  // One component per public Input, and where each Output is read (outside the constructor, whose
  // body then cannot throw).
  void add_components() {
    for (std::size_t i = 0; i < config_.inputs.size(); ++i) {
      auto c = std::make_unique<Component>();
      c->index = i;
      c->config = &config_.inputs[i];
      c->stream = streams_.intern(c->config->default_stream);
      c->fan_out = c->config->outputs.size() > 1U;
      for (std::size_t k = 0; k < c->config->outputs.size(); ++k) {
        const std::size_t o = c->config->outputs[k];
        c->lossy_block = c->lossy_block || config_.outputs[o].terminal_drop;
        if (slots_.size() <= o) {
          slots_.resize(o + 1U);
        }
        slots_[o] = OutputSlot{i, k};
      }
      components_.push_back(std::move(c));
    }
    // The unnamed push and pull reach the only input and the only output; with several there is
    // no default, as on GStreamer's graph path (plan.default_input, plan.default_output).
    default_input_ = components_.size() == 1U ? components_.front().get() : nullptr;
    default_output_ = config_.outputs.size() == 1U ? 0U : kNoEndpoint;
  }

  // A seed (one input only: the build checked it), or a declaration that gives the spec, binds now;
  // every other input binds at its own first push.
  void bind_at_build(const TensorList& seed) {
    for (const auto& c : components_) {
      std::string why;
      const InputOptions& declared = c->config->input;
      std::optional<DataflowInputBinding> binding =
          seed.size() > 1U
              ? bind_dataflow_record(declared, seed, &why)
              : bind_dataflow_input(declared, seed.empty() ? nullptr : &seed.front(), &why);
      if (!why.empty()) {
        throw_input_shape("Graph::build: " + why + " (Dataflow backend)");
      }
      if (binding) {
        std::lock_guard<std::mutex> lock(mu_);
        bind_locked(*c, std::move(*binding));
      }
    }
  }

  bool push(std::optional<std::string_view> input, const TensorList& inputs, bool block,
            const char* where) override {
    Component& c = input_for_push(input, where);
    if (inputs.empty()) {
      throw_input_shape(std::string(where) + ": an empty tensor list");
    }
    // The first tensor's stamp proposes the item's identity; a tensor without a stamp proposes
    // nothing (the default stream, no frame_id), as develop's TensorList push.
    const Stamp& proposal = inputs.front().stamp();
    if (std::optional<TensorList> fields = pipeline_internal::list_record_fields(inputs, where)) {
      return push_item(c, *fields, proposal, block, where);
    }
    return push_item(c, inputs, proposal, block, where, /*commit_allocation=*/block);
  }

  bool push(std::optional<std::string_view> input, const Sample& msgs, bool block,
            const char* where) override {
    Component& c = input_for_push(input, where);
    enforce_transferable(msgs, where);
    if (msgs.kind == SampleKind::Bundle) {
      // Core pushes a Bundle as one item per field, stopping at the first refusal
      // (RunPush.cpp:1082-1086).
      for (const Sample& field : msgs.fields) {
        if (!push_sample_item(c, field, block, where)) {
          return false;
        }
      }
      return true;
    }
    return push_sample_item(c, msgs, block, where);
  }

  bool push(std::optional<std::string_view> input, const std::vector<cv::Mat>& inputs, bool block,
            const char* where) override {
#if defined(SIMA_WITH_OPENCV)
    Component& c = input_for_push(input, where);
    if (inputs.size() != 1U) {
      throw std::runtime_error(error_util::decorate_error(
          error_codes::kRuntimePull,
          std::string(where) + ": raw-image ingress supports exactly one cv::Mat per inference "
                               "item (the Dataflow backend carries one tensor per push)"));
    }
    validate_public_input(inputs.front(), c.config->input, stream_options_, where);
    // A CPU view of the Mat, whatever the Input's memory policy: the push copies it into the slot,
    // or wraps it when the Mat owns its memory (its view's holder keeps a reference to it; a Mat
    // over memory it does not own keeps nothing alive).
    InputOptions view_options = c.config->input;
    view_options.memory_policy = InputMemoryPolicy::SystemMemory;
    const Tensor tensor = tensor_from_cv_mat(inputs.front(), view_options, where);
    return push_item(c, std::span<const Tensor>(&tensor, 1), Stamp{}, block, where, false, -1,
                     inputs.front().u != nullptr);
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
    // A Run of several inputs: the Run's push counter (SlotSeqTable), read under the lease.
    const std::int64_t orig = bound->run_seqs ? bound->run_seqs->get(message.bytes().data()) : -1;
    // The caller's list buffer is reused: Run::pull(int, Sample&, PullError*) with one Sample
    // allocates no list per pull.
    TensorList tensors = std::move(out.tensors);
    tensors.clear();
    out = make_sample(*bound, std::move(message), std::move(tensors));
    if (orig >= 0) {
      out.orig_input_seq = orig;
    }
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
    fill_tensors(*bound, std::move(message), out);
    count_pulled();
    return PullStatus::Ok;
  }

  Tensor alloc(std::string_view input, const char* where) override {
    const std::string at = std::string(where) + "(\"" + std::string(input) + "\")";
    Component* c = find_input(input);
    if (!c) {
      throw std::runtime_error(error_util::decorate_error(
          error_codes::kRuntimePull,
          at + ": unknown input. Available inputs: " + names_text(config_.input_names)));
    }
    const std::vector<std::int64_t> shape = declared_alloc_shape(c->config->input, at);
    Bound* bound = c->bound.load(std::memory_order_acquire);
    if (!bound || stopped_.load(std::memory_order_acquire) ||
        c->input_closed.load(std::memory_order_acquire)) {
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

  // Every input of the Run (Run::close_input names none): each one's queued messages stay pullable,
  // then its Outputs are Closed.
  void close_input() override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      for (const auto& c : components_) {
        c->input_closed.store(true, std::memory_order_release);
      }
    }
    cv_.notify_all();
    for (const auto& c : components_) {
      if (Bound* bound = c->bound.load(std::memory_order_acquire)) {
        // Never waits behind a parked push: that push returns FR3 (A1, A2). Queued messages stay.
        (void)bound->session.close_input(bound->feed, df::kNoDeadline);
      }
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
    trace_report();
    // Core's stream-stop budget, waited twice (RunLifecycle.cpp:94-95, 148-182), shared by the
    // Sessions: each stop wakes what is parked in its own Session, so they run one after the other
    // against one deadline, and the ones that timed out once more against a second one.
    const auto budget = std::chrono::milliseconds(
        std::max(1, pipeline_internal::env_int("SIMA_PIPELINE_STREAM_STOP_TIMEOUT_MS", 2000)));
    std::vector<Bound*> pending;
    for (const auto& c : components_) {
      if (Bound* bound = c->bound.load(std::memory_order_acquire)) {
        pending.push_back(bound);
      }
    }
    for (int round = 0; round < 2 && !pending.empty(); ++round) {
      const auto deadline = std::chrono::steady_clock::now() + budget;
      std::vector<Bound*> timed_out;
      for (Bound* bound : pending) {
        const df::Status status = bound->session.stop(deadline);
        if (!status) {
          continue;
        }
        if (status->code == df::ErrorCode::Timeout && round == 0) {
          timed_out.push_back(bound);
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
    for (const auto& c : components_) {
      if (Bound* bound = c->bound.load(std::memory_order_acquire)) {
        const df::SessionReport report = bound->session.report();
        for (const df::FeedReport& feed : report.feeds) {
          out.push_count += feed.pushes;
          out.push_failures += feed.push_failures + feed.backpressure;
        }
        out.dropped_frames += report.dropped_frames;
      }
    }
    return out;
  }

private:
  Component* find_input(std::string_view name) const {
    for (const auto& c : components_) {
      const std::vector<std::string>& names = c->config->names;
      if (std::find(names.begin(), names.end(), name) != names.end()) {
        return c.get();
      }
    }
    return nullptr;
  }

  // The input a push names, or the Run's only one for the unnamed push: GStreamer's checks and
  // texts (named_input_endpoint_or_throw, RunPush.cpp:216; RunCore::push_samples).
  Component& input_for_push(std::optional<std::string_view> input, const char* where) const {
    if (!input) {
      if (default_input_) {
        return *default_input_;
      }
      throw std::runtime_error(error_util::decorate_error(
          error_codes::kRuntimePull,
          "Run::push: graph has no unambiguous default input; use push(name, ...). Available "
          "inputs: " +
              names_text(config_.input_names)));
    }
    if (input->empty()) {
      throw std::runtime_error(error_util::decorate_error(
          error_codes::kRuntimePull, std::string(where) + ": input name is empty"));
    }
    if (Component* c = find_input(*input)) {
      return *c;
    }
    throw std::runtime_error(error_util::decorate_error(
        error_codes::kRuntimePull,
        "Run::push(\"" + std::string(*input) +
            "\"): unknown input. Available inputs: " + names_text(config_.input_names)));
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
  bool push_sample_item(Component& c, const Sample& item, bool block, const char* where) {
    const Stamp proposal = pipeline_internal::stamp_from_sample(item);
    // What GStreamer's graph ingress keeps as orig_input_seq without counting the push: a pushed
    // orig_input_seq, else a pushed input_seq (RunPush.cpp:276-287). Read in a Run of several
    // inputs only (SlotSeqTable).
    const std::int64_t pushed_seq = item.orig_input_seq >= 0 ? item.orig_input_seq
                                    : item.input_seq >= 0    ? item.input_seq
                                                             : -1;
    if (sample_has_tensor_list(item)) {
      return push_item(c, item.tensors, proposal, block, where, false, pushed_seq);
    }
    if (item.tensor.has_value()) {
      return push_item(c, std::span<const Tensor>(&*item.tensor, 1), proposal, block, where, false,
                       pushed_seq);
    }
    throw_input_shape(std::string(where) + ": the Sample carries no tensor");
  }

  // One item into input `c`: Core's checks, the binding (the input's first item binds it), then
  // the slot. With `commit_allocation` (the blocking TensorList push only), a Tensor from this
  // Run's alloc() alone in the list is published as its slot; every other verb copies it like any
  // CPU tensor, as the GStreamer backend does. `pushed_seq`: the item's own orig_input_seq (a
  // Sample push), -1 for none; read in a Run of several inputs only. A leaf is wrapped instead of
  // copied when that is safe (W2: wrap_layout, wrap_block), unless `may_wrap` is false (a cv::Mat
  // that does not own its memory).
  bool push_item(Component& c, std::span<const Tensor> tensors, const Stamp& proposal, bool block,
                 const char* where, bool commit_allocation = false, std::int64_t pushed_seq = -1,
                 bool may_wrap = true) {
    const InputOptions& declared = c.config->input;
    // Core's own push-time checks and texts first (RunPush.cpp:655, InputStreamUtil.cpp:2146-2190):
    // the declaration describes the item's representative (first) tensor, as on GStreamer.
    if (tensors.size() > 1U && !declared.shape.empty()) {
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
        tensors.size() == 1U ? c.validated_leaf.load(std::memory_order_acquire) : nullptr;
    const bool validated = known && known->matches(tensors.front());
    if (!validated) {
      validate_public_input(tensors.front(), declared, stream_options_, where);
      for (const Tensor& tensor : tensors) {
        if (tensor.storage && tensor.storage->kind == StorageKind::GstSample) {
          enforce_transferable(sample_from_tensors(TensorList{tensor}), where);
        }
      }
    }
    Bound* bound = c.bound.load(std::memory_order_acquire);
    if (!bound) {
      if (stopped_.load(std::memory_order_acquire) ||
          c.input_closed.load(std::memory_order_acquire)) {
        return false;
      }
      bound = bind_first(c, tensors, where);
      if (!bound) {
        return false;
      }
    }
    const df::Stamp stamp = runtime_stamp(proposal, streams_, c.stream);
    if (tensors.size() > 1U || bound->binding.is_record()) {
      if (const std::string why = dataflow_record_mismatch(bound->binding, tensors); !why.empty()) {
        throw_input_shape(std::string(where) + ": " + why);
      }
      return push_record(c, *bound, tensors, stamp, run_seq(*bound, pushed_seq), block, where);
    }
    const Tensor& tensor = tensors.front();
    if (!validated) {
      if (const std::string why = dataflow_binding_mismatch(bound->binding, tensor); !why.empty()) {
        throw_input_shape(std::string(where) + ": " + why);
      }
      remember_validated_leaf(c, *bound, tensor);
    }
    const std::int64_t seq = run_seq(*bound, pushed_seq);
    if (allocation) {
      return push_allocated(c, *bound, tensor, *allocation, stamp, seq, block, where);
    }
    // A try_push into links that FR14 found full, with no pull since: alloc is asked first, without
    // a wrap (still_full).
    if (!block && bound->full_at.load(std::memory_order_relaxed) ==
                      core_.outputs_pulled.load(std::memory_order_relaxed)) {
      if (still_full(c, *bound, where)) {
        return false;
      }
    }
    // W2: wrap when safe. The runtime takes a wrap only with the slot (Session::alloc), so the
    // wrap is decided before alloc; a refusal of alloc (A3's FR14, FR7, SR1, FR3) takes no slot and
    // keeps nothing: `wrap`, the only reference this call took, is released on return.
    df::Block wrap;
    bool wrapped =
        may_wrap && bound->wraps &&
        (validated ? known->wrappable : wrap_layout(bound->binding, bound->slot_bytes, tensor)) &&
        wrap_block(tensor, bound->slot_bytes, bound->wrap_mask, wrap);
    const df::Acquire mode = block ? df::Acquire::Block : df::Acquire::Fail;
    df::Result<df::WriteLoan> allocated =
        bound->session.alloc(bound->feed, mode, df::kNoDeadline, wrapped ? &wrap : nullptr);
    if (const df::Error* error = df::failure(allocated)) {
      if (!wrapped || error->code != df::ErrorCode::InvalidArgument) {
        if (!block && wrapped && error->code == df::ErrorCode::Backpressure) {
          bound->full_at.store(core_.outputs_pulled.load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
        }
        return alloc_refused(c, *error, block, where);
      }
      // FR15: the runtime refused the wrap itself (wrap_layout and wrap_block state its rule, so
      // this is not expected), with the slot returned and nothing kept. The copy, as before W2.
      wrapped = false;
      allocated = bound->session.alloc(bound->feed, mode);
      if (const df::Error* again = df::failure(allocated)) {
        return alloc_refused(c, *again, block, where);
      }
    }
    df::WriteLoan loan = std::get<df::WriteLoan>(std::move(allocated));
    df::Extent extent;
    if (wrapped) {
      // The loan's bytes are the tensor's own: nothing to copy.
      extent = wrapped_extent(bound->binding, tensor);
      if (!bound->has_wraps.load(std::memory_order_relaxed)) {
        bound->has_wraps.store(true, std::memory_order_relaxed);
      }
    } else {
      if (bound->run_seqs) {
        bound->run_seqs->set(loan.bytes().data(), seq);
      }
      extent = copy_into(bound->binding, tensor, loan.bytes(), where);
    }
    return settle(c, publish(c, *bound, std::move(loan), stamp, extent, block), block, where);
  }

  // A try_push that would wrap pays the wrap decision (the storage's type query, the holder's copy
  // and release) before alloc can refuse it. So once FR14 has refused one (every link full:
  // Bound::full_at), and while no pull has returned since, the links are still full, and the next
  // try_push asks alloc first, without a wrap: its refusal (true, counted as alloc_refused counts
  // it) costs what it cost before W2. If alloc finds room after all (a pull is between its take
  // and its count), the slot goes back at once, before the push takes its own. Out of line: the
  // blocking push never comes here.
  [[gnu::noinline]] bool still_full(const Component& c, Bound& bound, const char* where) {
    const df::Result<df::WriteLoan> room = bound.session.alloc(bound.feed, df::Acquire::Fail);
    if (const df::Error* error = df::failure(room)) {
      (void)alloc_refused(c, *error, false, where);
      return true;
    }
    bound.full_at.store(Bound::kNoRefusal, std::memory_order_relaxed);
    return false;
  }

  // The first leaf of input `c` that passed validate_public_input and the binding check, kept once
  // (later pushes read it without a lock), with whether its geometry can be wrapped (W2).
  void remember_validated_leaf(Component& c, const Bound& bound, const Tensor& tensor) {
    if (c.validated_leaf.load(std::memory_order_acquire) || !LeafGeometry::keepable(tensor)) {
      return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (!c.validated_leaf_storage) {
      auto geometry = std::make_unique<LeafGeometry>(LeafGeometry::of(tensor));
      geometry->wrappable = bound.wraps && wrap_layout(bound.binding, bound.slot_bytes, tensor);
      c.validated_leaf_storage = std::move(geometry);
      c.validated_leaf.store(c.validated_leaf_storage.get(), std::memory_order_release);
    }
  }

  // The orig_input_seq of the item being pushed into a Run of several inputs: the pushed one, or
  // the next value of the Run's counter (one relaxed increment; the counter is shared by the
  // inputs). -1 in a Run of one input, which counts nothing here.
  std::int64_t run_seq(const Bound& bound, std::int64_t pushed_seq) noexcept {
    if (!bound.run_seqs) {
      return -1;
    }
    return pushed_seq >= 0
               ? pushed_seq
               : static_cast<std::int64_t>(run_seq_.fetch_add(1, std::memory_order_relaxed));
  }

  // A record: every field copied at its offset in one slot, one Extent per field.
  bool push_record(Component& c, Bound& bound, std::span<const Tensor> fields,
                   const df::Stamp& stamp, std::int64_t seq, bool block, const char* where) {
    df::Result<df::WriteLoan> allocated =
        bound.session.alloc(bound.feed, block ? df::Acquire::Block : df::Acquire::Fail);
    if (const df::Error* error = df::failure(allocated)) {
      return alloc_refused(c, *error, block, where);
    }
    df::WriteLoan loan = std::get<df::WriteLoan>(std::move(allocated));
    if (bound.run_seqs) {
      bound.run_seqs->set(loan.bytes().data(), seq);
    }
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
        : c.lossy_block
            ? bound.session.push(bound.feed, std::move(loan), stamp, extent_span, df::kNoDeadline)
            : bound.session.push_waiting(bound.feed, std::move(loan), stamp, extent_span,
                                         df::kNoDeadline);
    return settle(c, pushed, block, where);
  }

  // A Tensor from this Run's alloc(): its slot is published as it is, without a copy. The loan
  // goes back to the allocation when the Session refuses it with the loan intact (SR1, FR2, FR3,
  // FR5: stopped, no room, input closed), so the application may push it again or drop it.
  bool push_allocated(Component& c, Bound& bound, const Tensor& tensor, RunAllocation& allocation,
                      const df::Stamp& stamp, std::int64_t seq, bool block, const char* where) {
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
      if (bound.run_seqs) {
        bound.run_seqs->set(loan.bytes().data(), seq);
      }
      pushed = publish(c, bound, std::move(loan), stamp, extent_of(tensor.shape), block);
    } catch (...) {
      settle_allocation();
      throw;
    }
    settle_allocation();
    return settle(c, pushed, block, where);
  }

  // try_push: the policy push at a passed deadline. push: the waiting push, which waits for room on
  // every link and never drops; or, when an Output of the input has OutputOptions::drop, the
  // policy push without a deadline, which waits on the Block links only and evicts on the
  // DropOldest ones (appsink drop=true).
  df::Result<df::PublishResult> publish(const Component& c, Bound& bound, df::WriteLoan&& loan,
                                        const df::Stamp& stamp, const df::Extent& extent,
                                        bool block) {
    if (!block) {
      return bound.session.push(bound.feed, std::move(loan), stamp, extent, df::Deadline{});
    }
    if (c.lossy_block) {
      return bound.session.push(bound.feed, std::move(loan), stamp, extent, df::kNoDeadline);
    }
    return bound.session.push_waiting(bound.feed, std::move(loan), stamp, extent, df::kNoDeadline);
  }

  bool settle(const Component& c, const df::Result<df::PublishResult>& pushed, bool block,
              const char* where) {
    if (const df::Error* error = df::failure(pushed)) {
      return refused(c, *error, block, where);
    }
    return published(c, std::get<df::PublishResult>(pushed));
  }

  // The first push into input `c` binds it: its tensors' spec, under the declaration (WORKER.md
  // 18.1). Every other input binds by its own first push.
  Bound* bind_first(Component& c, std::span<const Tensor> tensors, const char* where) {
    std::lock_guard<std::mutex> lock(mu_);
    if (Bound* bound = c.bound.load(std::memory_order_acquire)) {
      return bound;
    }
    if (stopped_.load(std::memory_order_relaxed) ||
        c.input_closed.load(std::memory_order_relaxed)) {
      return nullptr;
    }
    std::string why;
    std::optional<DataflowInputBinding> binding =
        tensors.size() > 1U ? bind_dataflow_record(c.config->input, tensors, &why)
                            : bind_dataflow_input(c.config->input, &tensors.front(), &why);
    if (!binding) {
      throw_input_shape(std::string(where) + ": " +
                        (why.empty() ? std::string("the input cannot be bound") : why) +
                        " (Dataflow backend)");
    }
    return bind_locked(c, std::move(*binding));
  }

  Bound* bind_locked(Component& c, DataflowInputBinding binding) {
    DataflowNet net = compile_dataflow_net(config_, c.index, binding, c.stream);
    std::uint64_t slots = 0; // the Feed's pool: what a SlotSeqTable must hold
    for (const df::PlanPort& port : net.plan.report().ports) {
      if (port.op == net.feed) {
        slots = port.capacity;
      }
    }
    df::Result<df::Session> session = std::move(net.plan).instantiate(df::SessionOptions{});
    if (const df::Error* error = df::failure(session)) {
      pipeline_internal::throw_dataflow_error(*error, "Graph::build");
    }
    auto bound = std::make_unique<Bound>(
        std::move(binding), std::get<df::Session>(std::move(session)), net.feed, net.fetches);
    if (config_.inputs.size() > 1U) {
      bound->run_seqs = std::make_unique<SlotSeqTable>(slots);
    }
    if (!bound->binding.is_record()) {
      bound->slot_bytes = df::slot_bytes(bound->binding.spec);
      bound->wrap_mask = df::slot_alignment(bound->binding.spec) - 1U;
      bound->wraps = !config_.run.advanced.copy_input && !bound->run_seqs;
    }
    if (const df::Status status = bound->session.start()) {
      pipeline_internal::throw_dataflow_error(*status, "Graph::build");
    }
    c.bound_storage = std::move(bound);
    c.bound.store(c.bound_storage.get(), std::memory_order_release);
    cv_.notify_all();
    return c.bound_storage.get();
  }

  Bound* wait_bound(Component& c, df::Deadline deadline) {
    if (Bound* bound = c.bound.load(std::memory_order_acquire)) {
      return bound;
    }
    std::unique_lock<std::mutex> lock(mu_);
    const auto ready = [&] {
      return c.bound.load(std::memory_order_acquire) != nullptr ||
             stopped_.load(std::memory_order_relaxed) ||
             c.input_closed.load(std::memory_order_relaxed);
    };
    if (deadline == df::kNoDeadline) {
      cv_.wait(lock, ready);
    } else {
      cv_.wait_until(lock, deadline, ready);
    }
    return c.bound.load(std::memory_order_acquire);
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
                names_text(config_.output_names));
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
            "\"): unknown output. Available outputs: " + names_text(config_.output_names));
    return kNoEndpoint;
  }

  // The next message of the output, or why there is none (both pull verbs). The call's name for
  // messages is built only when a message needs it.
  PullStatus take(std::optional<std::string_view> output, int timeout_ms, PullError* err,
                  df::Message& message, Bound*& bound) {
    clear_pull_error(err);
    const auto where = [&output] {
      return output ? "Run::pull(\"" + std::string(*output) + "\")" : std::string("Run::pull");
    };
    const std::size_t o = output_for_pull(output, err);
    if (o == kNoEndpoint) {
      return PullStatus::Error;
    }
    const OutputSlot slot = slots_[o];
    Component& c = *components_[slot.component];
    polls_.fetch_add(1, std::memory_order_relaxed);
    const df::Deadline deadline = deadline_after(timeout_ms);
    bound = wait_bound(c, deadline);
    if (!bound) {
      // Nothing was pushed into its input yet: a pull waits for that first push in Core
      // (WORKER.md 18.1).
      if (stopped_.load(std::memory_order_acquire) ||
          c.input_closed.load(std::memory_order_acquire)) {
        error_util::set_pull_error(err, error_codes::kRuntimePull, where() + ": stream closed");
        return PullStatus::Closed;
      }
      return timed_out(timeout_ms, where(), err);
    }

    df::Result<df::Message> pulled = bound->session.pull(bound->fetches[slot.link], deadline);
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
        if (config_.outputs.size() == 1U && c.input_closed.load(std::memory_order_acquire) &&
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
  // are DropOldest links, which never refuse: a fan-out with one of them never sees FR14. Every
  // other refusal, FR7 included, is refused()'s.
  bool alloc_refused(const Component& c, const df::Error& error, bool block, const char* where) {
    if (!block && error.code == df::ErrorCode::Backpressure) {
      if (!c.lossy_block && config_.link_policy == OverflowPolicy::DropIncoming) {
        core_.inputs_dropped.fetch_add(1, std::memory_order_relaxed);
      }
      return false;
    }
    return refused(c, error, block, where);
  }

  // A push the Session refused: false, a drop where Core counts one, or Core's error.
  bool refused(const Component& c, const df::Error& error, bool block, const char* where) {
    switch (error.code) {
    case df::ErrorCode::Closed:       // FR3 (close_input), SR1 (stopped), a closed pool or port
    case df::ErrorCode::Backpressure: // FR2: a full Block link at a passed deadline (try_push)
    case df::ErrorCode::Timeout:
      return false;
    case df::ErrorCode::ResourceExhausted:
      // FR7: no free slot for try_push; the application holds every pulled slot. A drop for the
      // dropping policies, as a full queue is (RunPush.cpp:416-424).
      if (!block && (c.lossy_block || config_.link_policy != OverflowPolicy::Block)) {
        core_.inputs_dropped.fetch_add(1, std::memory_order_relaxed);
      }
      return false;
    default:
      pipeline_internal::throw_dataflow_error(error, where);
    }
  }

  bool published(const Component& c, const df::PublishResult& result) {
    if (!result.published() || result.size() == 0U) {
      return false; // the port closed under a racing stop
    }
    if (c.fan_out) {
      return published_fan_out(result);
    }
    switch (result.outcome(0)) {
    case df::Link::Offer::Accepted:
      count_published(1U);
      return true;
    case df::Link::Offer::DroppedOldest:
      // Evicted to make room: KeepLatest's input drop, or the terminal queue's output drop.
      count_published(1U);
      (c.lossy_block ? core_.outputs_dropped : core_.inputs_dropped)
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

  // A fan-out: each link answered for its own Output. The push is accepted when one link took the
  // message (each that did has one more item ready), and an eviction, or a refusal while another
  // link took it, is that Output's drop. When no link took it the push is refused, and a drop of
  // the input when a DropIncoming link refused it.
  bool published_fan_out(const df::PublishResult& result) {
    std::uint64_t evicted = 0;
    std::uint64_t refused_incoming = 0;
    for (std::uint32_t k = 0; k < result.size(); ++k) {
      const df::Link::Offer outcome = result.outcome(k);
      if (outcome == df::Link::Offer::DroppedOldest) {
        ++evicted;
      } else if (outcome == df::Link::Offer::DroppedIncoming) {
        ++refused_incoming;
      }
    }
    if (result.accepted() == 0U) {
      if (refused_incoming != 0U) {
        core_.inputs_dropped.fetch_add(1, std::memory_order_relaxed);
      }
      return false;
    }
    count_published(result.accepted());
    if (evicted + refused_incoming != 0U) {
      core_.outputs_dropped.fetch_add(evicted + refused_incoming, std::memory_order_relaxed);
    }
    return true;
  }

  // One input item published, ready on `outputs` Outputs.
  void count_published(std::uint32_t outputs) {
    core_.inputs_enqueued.fetch_add(1, std::memory_order_relaxed);
    core_.inputs_pushed.fetch_add(1, std::memory_order_relaxed);
    core_.outputs_ready.fetch_add(outputs, std::memory_order_relaxed);
  }

  // A dense tensor of a pulled message over `storage` from `byte_offset` (a lease on its slot,
  // lease_storage; an Owned copy; or the pushed storage of a wrapped message), with the Core
  // metadata the binding kept, written into `tensor` (a default Tensor, built where it is
  // returned: a Tensor is about 1 KB, and building it elsewhere and moving it costs as much as
  // building it).
  static void fill_dense(Tensor& tensor, TensorDType dtype, TensorLayout layout,
                         const std::vector<TensorAxisSemantic>& axis_semantics,
                         const Semantic& semantic, const df::Extent& extent,
                         std::shared_ptr<TensorBuffer> storage, std::int64_t byte_offset) {
    tensor.storage = std::move(storage);
    tensor.byte_offset = byte_offset;
    tensor.dtype = dtype;
    tensor.layout = layout;
    tensor.axis_semantics = axis_semantics;
    tensor.device = {DeviceType::CPU, 0};
    tensor.semantic = semantic;
    tensor.read_only = true;
    tensor.shape.assign(extent.dims.begin(), extent.dims.begin() + extent.rank);
  }

  // The leaf tensor of a pulled message (dense or planar) over `storage` from `byte_offset`,
  // written into `tensor` (a default Tensor).
  static void fill_leaf(Tensor& tensor, const DataflowInputBinding& binding,
                        const df::Extent& extent, std::shared_ptr<TensorBuffer> storage,
                        std::int64_t byte_offset) {
    if (binding.planes == DataflowInputBinding::Planes::None) {
      fill_dense(tensor, binding.dtype, binding.layout, binding.axis_semantics, binding.semantic,
                 extent, std::move(storage), byte_offset);
      return;
    }
    tensor.storage = std::move(storage);
    tensor.byte_offset = byte_offset;
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

  // W2: the pushed storage behind a wrapped message (D14), shared, with the offset of the
  // message's first byte in it; null for a message in a slot's own memory, which the pull leases.
  // Message::block() is asked only once the input has wrapped (Bound::has_wraps). Every wrap of a
  // Feed is Core's own (push_item), so its holder is the pushed Tensor's TensorBuffer; a storage
  // whose address or size no longer contains the message (the application changed the
  // TensorBuffer against D16) is not returned, and the slot is leased instead.
  static std::shared_ptr<TensorBuffer>
  wrapped_storage(const Bound& bound, const df::Message& message, std::int64_t& byte_offset) {
    if (!bound.has_wraps.load(std::memory_order_relaxed)) {
      return nullptr;
    }
    df::Block block = message.block();
    if (!block.holder) {
      return nullptr;
    }
    auto storage = std::static_pointer_cast<TensorBuffer>(std::move(block.holder));
    const std::span<const std::byte> bytes = message.bytes();
    const auto base = reinterpret_cast<std::uintptr_t>(storage->data);
    const auto first = reinterpret_cast<std::uintptr_t>(bytes.data());
    if (base == 0U || first < base || first - base > storage->size_bytes ||
        storage->size_bytes - (first - base) < bytes.size()) {
      return nullptr;
    }
    byte_offset = static_cast<std::int64_t>(first - base);
    return storage;
  }

  // The tensors of a record message, in schema order, each over its field of the slot (records
  // are never wrapped).
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
      Tensor field;
      fill_dense(field, meta.dtype, meta.layout, meta.axis_semantics, meta.semantic, *view.extent,
                 owned ? owned_copy(view.bytes) : lease_storage(view.bytes, holder), 0);
      fields.push_back(std::move(field));
    }
    return fields;
  }

  // The pulled Sample, shaped as the GStreamer Input -> Output Run returns it
  // (InputStreamPull.cpp:830-884): a TensorSet of the item's tensors, media type and payload tag
  // from the first tensor, identity from the stamp (SampleProjection). `tensors` (empty) lends its
  // buffer to the Sample's list.
  Sample make_sample(const Bound& bound, df::Message message, TensorList tensors) const {
    const DataflowInputBinding& binding = bound.binding;
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
      const df::Extent extent = message.extent();
      Tensor& tensor = out.tensors.emplace_back();
      std::int64_t offset = 0;
      if (owned) {
        // OutputMemory::Owned: an independent CPU copy; the slot returns to the pool now.
        fill_leaf(tensor, binding, extent, owned_copy(message.bytes()), 0);
      } else if (std::shared_ptr<TensorBuffer> pushed = wrapped_storage(bound, message, offset)) {
        // A wrapped message (W2): the pushed storage itself (D14); the slot returns as `message`
        // goes, at the return.
        fill_leaf(tensor, binding, extent, std::move(pushed), offset);
      } else {
        // A lease on the slot: the Message is the holder, released with the last copy of the
        // tensor.
        auto holder = std::make_shared<df::Message>(std::move(message));
        const std::span<const std::byte> leased = holder->bytes();
        fill_leaf(tensor, binding, extent, lease_storage(leased, std::move(holder)), 0);
      }
    }
    pipeline_internal::apply_pulled_stamp(stamp, out);
    return out;
  }

  // The pulled tensors (Run::pull(milliseconds)) into `out`: one for a leaf, the fields in schema
  // order for a record, every one carrying the message's Stamp. One allocation holds the message
  // and the envelope they share; the list (unless `out` has room), and each tensor's storage and
  // shape, are the only others. A leaf is built in its place in the list. A wrapped message (W2)
  // gives the pushed storage itself (D14) and its slot back at once: one allocation fewer (no
  // storage), the envelope alone in the other.
  void fill_tensors(const Bound& bound, df::Message message, TensorList& out) const {
    const DataflowInputBinding& binding = bound.binding;
    const bool owned = config_.run.output_memory == OutputMemory::Owned;
    std::int64_t offset = 0;
    std::shared_ptr<TensorBuffer> pushed =
        owned || binding.is_record() ? nullptr : wrapped_storage(bound, message, offset);
    std::shared_ptr<PulledMessage> pulled;
    std::shared_ptr<const TensorEnvelope> envelope;
    if (pushed) {
      auto own = std::make_shared<TensorEnvelope>();
      core_stamp(message.stamp(), streams_, own->stamp);
      envelope = std::move(own);
    } else {
      pulled = std::make_shared<PulledMessage>(std::move(message));
      core_stamp(pulled->message.stamp(), streams_, pulled->envelope.stamp);
      envelope = std::shared_ptr<const TensorEnvelope>(pulled, &pulled->envelope);
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
    }
    const df::Message& source = pushed ? message : pulled->message;
    const std::span<const std::byte> bytes = source.bytes();
    const df::Extent extent = source.extent();
    out.clear();
    Tensor& leaf = out.emplace_back();
    if (pushed) {
      // A wrapped message: the slot returns to the pool now; the bytes live with `pushed`.
      message = df::Message{};
      fill_leaf(leaf, binding, extent, std::move(pushed), offset);
    } else if (owned) {
      fill_leaf(leaf, binding, extent, owned_copy(bytes), 0);
      pulled->message = df::Message{};
    } else {
      df::Message* const leased = &pulled->message;
      fill_leaf(leaf, binding, extent,
                lease_storage(bytes, std::shared_ptr<void>(std::move(pulled), leased)), 0);
    }
    leaf.envelope = std::move(envelope);
  }

  // With SIMA_NEAT_BACKEND set (the [NEAT] lines of DataflowGraphCompiler.cpp), each input's and
  // each output's counters as the stop finds them: the per-endpoint report of the Run.
  void trace_report() const {
    if (pipeline_internal::env_str("SIMA_NEAT_BACKEND", "").empty()) {
      return;
    }
    for (const auto& c : components_) {
      Bound* bound = c->bound.load(std::memory_order_acquire);
      std::string line = "[NEAT] Run::stop: input " + names_text(c->config->names);
      if (!bound) {
        std::fprintf(stderr, "%s: never bound\n", line.c_str());
        continue;
      }
      const df::SessionReport report = bound->session.report();
      if (!report.feeds.empty()) {
        const df::FeedReport& feed = report.feeds.front();
        line += ": pushes " + std::to_string(feed.pushes) + ", push failures " +
                std::to_string(feed.push_failures) + ", backpressure " +
                std::to_string(feed.backpressure);
      }
      for (std::size_t k = 0; k < report.links.size() && k < c->config->outputs.size(); ++k) {
        const df::LinkReport& link = report.links[k];
        line += "; output " + names_text(config_.outputs[c->config->outputs[k]].names) +
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
  std::vector<std::unique_ptr<Component>> components_;
  std::vector<OutputSlot> slots_;
  Component* default_input_ = nullptr;
  std::size_t default_output_ = kNoEndpoint;

  // Bindings and the validated geometries are written under mu_; cv_ wakes a pull that waits for
  // its input's binding.
  std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<bool> stopped_{false};
  std::atomic<std::uint64_t> polls_{0};
  // A Run of several inputs: the counter of its pushes, orig_input_seq (SlotSeqTable).
  std::atomic<std::uint64_t> run_seq_{0};

  mutable std::mutex error_mu_;
  std::optional<PullError> last_error_;
};

} // namespace

std::unique_ptr<DataflowRunState> make_dataflow_run_state(RunCore& core, DataflowRunConfig config,
                                                          const TensorList& seed) {
  auto run = std::make_unique<DataflowRun>(core, std::move(config));
  run->add_components();
  run->bind_at_build(seed);
  return run;
}

} // namespace simaai::neat::runtime

#endif // SIMANEAT_WITH_DATAFLOW
