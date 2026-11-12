#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// The Dataflow backend's tensor codec: Core's Tensor, Sample and Stamp to and from the runtime's
// slot, Message and Stamp. Free functions without state: no Session, no Run; `static inline`, so
// that each of the two files that include this one inlines them as it would its own. The push side
// (DataflowInputPort.h) writes a tensor's bytes into a slot, or backs the slot with the tensor's
// own storage (W2); the pull side (DataflowRun.cpp) builds the Tensors or the Sample of a message.

#include "pipeline/ErrorCodes.h"
#include "pipeline/TensorCore.h"
#include "pipeline/internal/DataflowRuntime.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/SampleProjection.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/runtime/DataflowInputBinding.h"
#include "pipeline/runtime/RunAllocation.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <typeinfo>
#include <utility>
#include <variant>
#include <vector>

namespace simaai::neat::runtime::dataflow_codec {

#pragma GCC visibility push(hidden)

namespace df = pipeline_internal::df;

[[noreturn]] static inline void throw_input_shape(const std::string& message) {
  pipeline_internal::error_util::throw_session_error(error_codes::kInputShape, message);
}

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

// The runtime's stamp for what a push proposes: the stream interned (an empty name is the input's
// default stream, `default_stream`), the author's frame_id, source_time (any clock) and flags. emit
// writes seq and publish_time.
static inline df::Stamp runtime_stamp(const Stamp& proposal,
                                      pipeline_internal::StreamNames& streams,
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
static inline void core_stamp(const df::Stamp& in, const pipeline_internal::StreamNames& streams,
                              Stamp& out) {
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
static inline Mapping map_cpu_storage_read(const Tensor& tensor) {
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
static inline bool copy_dense_into(const Tensor& tensor, std::uint8_t* dst, std::size_t dst_size) {
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

static inline bool cpu_plane_rows(const Tensor& tensor, const Mapping& mapping, PlaneRole role,
                                  std::int64_t rows, std::int64_t cols, PlaneRows& out) {
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

static inline df::Extent extent_of(const std::vector<std::int64_t>& shape) {
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
static inline std::shared_ptr<TensorBuffer> lease_storage(std::span<const std::byte> bytes,
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

// W2: the Block that backs a slot with the storage of `t` (dataflow_wrap_layout held): its first
// byte, `slot_bytes` long, and as keep-alive the Tensor's own TensorBuffer, which keeps the
// storage's holder (one count on the shared_ptr's existing control block; nothing is allocated; an
// application can read the release from its storage's use_count, as from GStreamer's). False,
// and a copy instead, when the storage cannot be kept or must not be: no holder or no address
// (memory nobody owns), a lease on a Dataflow slot or a Run::alloc tensor (D15), or an address
// not aligned to the port's slot alignment (`align_mask` = slot_alignment - 1).
static inline bool wrap_block(const Tensor& t, std::size_t slot_bytes, std::uintptr_t align_mask,
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
static inline df::Extent wrapped_extent(const DataflowInputBinding& binding, const Tensor& t) {
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

// The planes of a CPU NV12 or I420 tensor into the slot (copy_rows), at the geometry
// map_nv12_read / map_i420_read give a well-formed one (the Y plane's {h, w}, both even), read
// through map_cpu_storage_read. False, with nothing copied, for any other tensor: the caller
// then maps it through the Tensor, which reports what is wrong with it.
template <class CopyRows>
static inline bool copy_cpu_planes(const DataflowInputBinding& binding, const Tensor& tensor,
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

// The bytes of `tensor` into the slot, at the binding's layout: one tight copy for a dense
// tensor, row by row at the bound's pitches for NV12 and I420 planes.
static inline df::Extent copy_into(const DataflowInputBinding& binding, const Tensor& tensor,
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
  const auto& planar = std::get<df::Planar>(std::get<df::TensorSpec>(binding.spec.payload).layout);
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

// A record's fields into the slot, each at its offset, one Extent per field (`extents`, as many
// as `fields`).
static inline void copy_record(const DataflowInputBinding& binding, std::span<const Tensor> fields,
                               std::span<std::byte> slot, std::span<df::Extent> extents,
                               const char* where) {
  auto* dst = reinterpret_cast<std::uint8_t*>(slot.data());
  for (std::size_t i = 0; i < fields.size(); ++i) {
    const DataflowFieldBinding& field = binding.fields[i];
    extents[i] = extent_of(fields[i].shape);
    if (field.offset > slot.size() ||
        !copy_dense_into(fields[i], dst + field.offset, slot.size() - field.offset)) {
      throw_input_shape(std::string(where) + ": the bytes of field " + std::to_string(i) +
                        " could not be read for the copy (Dataflow backend)");
    }
  }
}

// A dense tensor of a pulled message over `storage` from `byte_offset` (a lease on its slot,
// lease_storage; an Owned copy; or the pushed storage of a wrapped message), with the Core
// metadata the binding kept, written into `tensor` (a default Tensor, built where it is
// returned: a Tensor is about 1 KB, and building it elsewhere and moving it costs as much as
// building it).
static inline void fill_dense(Tensor& tensor, TensorDType dtype, TensorLayout layout,
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
static inline void fill_leaf(Tensor& tensor, const DataflowInputBinding& binding,
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
  const auto& planar = std::get<df::Planar>(std::get<df::TensorSpec>(binding.spec.payload).layout);
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
static inline std::shared_ptr<TensorBuffer> owned_copy(std::span<const std::byte> bytes) {
  auto storage = make_cpu_owned_storage(bytes.size());
  if (!bytes.empty()) {
    std::memcpy(storage->data, bytes.data(), bytes.size());
  }
  return storage;
}

// W2: the pushed storage behind a wrapped message (D14), shared, with the offset of the
// message's first byte in it; null for a message in a slot's own memory, which the pull leases.
// Message::block() is asked only once the input has wrapped (`has_wraps`). Every wrap of a Feed
// is Core's own (DataflowInputPort), so its holder is the pushed Tensor's TensorBuffer; a storage
// whose address or size no longer contains the message (the application changed the
// TensorBuffer against D16) is not returned, and the slot is leased instead.
static inline std::shared_ptr<TensorBuffer>
wrapped_storage(bool has_wraps, const df::Message& message, std::int64_t& byte_offset) {
  if (!has_wraps) {
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
static inline std::vector<Tensor> record_fields(const DataflowInputBinding& binding,
                                                const df::Message& message,
                                                const std::shared_ptr<void>& holder, bool owned) {
  const df::RecordSchema& schema = *message.schema();
  const df::Bundle bundle(message);
  std::vector<Tensor> fields;
  fields.reserve(binding.fields.size());
  for (std::size_t i = 0; i < binding.fields.size(); ++i) {
    const DataflowFieldBinding& meta = binding.fields[i];
    const df::FieldView view = bundle.field(meta.offset, meta.extent_offset, schema.fields[i].spec);
    Tensor field;
    fill_dense(field, meta.dtype, meta.layout, meta.axis_semantics, meta.semantic, *view.extent,
               owned ? owned_copy(view.bytes) : lease_storage(view.bytes, holder), 0);
    fields.push_back(std::move(field));
  }
  return fields;
}

// What the tensors of one pulled message keep alive, in one allocation: the message (the lease on
// its slot) and the envelope every tensor of it shares (its Stamp). The tensors' storage holder
// and envelope are aliases into it; it holds no Tensor, so nothing points back at it.
struct PulledMessage {
  explicit PulledMessage(df::Message m) : message(std::move(m)) {}
  df::Message message;
  TensorEnvelope envelope;
};

// The pulled Sample, shaped as the GStreamer Input -> Output Run returns it
// (InputStreamPull.cpp:830-884): a TensorSet of the item's tensors, media type and payload tag
// from the first tensor, identity from the stamp (SampleProjection). `tensors` (empty) lends its
// buffer to the Sample's list. `owned`: OutputMemory::Owned; `has_wraps`: the input has wrapped.
static inline Sample make_sample(const DataflowInputBinding& binding, bool has_wraps,
                                 const pipeline_internal::StreamNames& streams, bool owned,
                                 df::Message message, TensorList tensors) {
  Stamp stamp;
  core_stamp(message.stamp(), streams, stamp);
  Sample out;
  out.kind = SampleKind::TensorSet;
  out.owned = owned;
  out.tensors = std::move(tensors);
  const Semantic& first = binding.is_record() ? binding.fields.front().semantic : binding.semantic;
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
    } else if (std::shared_ptr<TensorBuffer> pushed = wrapped_storage(has_wraps, message, offset)) {
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
static inline void fill_tensors(const DataflowInputBinding& binding, bool has_wraps,
                                const pipeline_internal::StreamNames& streams, bool owned,
                                df::Message message, TensorList& out) {
  std::int64_t offset = 0;
  std::shared_ptr<TensorBuffer> pushed =
      owned || binding.is_record() ? nullptr : wrapped_storage(has_wraps, message, offset);
  std::shared_ptr<PulledMessage> pulled;
  std::shared_ptr<const TensorEnvelope> envelope;
  if (pushed) {
    auto own = std::make_shared<TensorEnvelope>();
    core_stamp(message.stamp(), streams, own->stamp);
    envelope = std::move(own);
  } else {
    pulled = std::make_shared<PulledMessage>(std::move(message));
    core_stamp(pulled->message.stamp(), streams, pulled->envelope.stamp);
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

#pragma GCC visibility pop

} // namespace simaai::neat::runtime::dataflow_codec
