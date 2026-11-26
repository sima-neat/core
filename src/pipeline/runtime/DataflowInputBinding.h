#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// The format contract of one public Input on the Dataflow backend: the port spec its declaration,
// its seed or its first push binds (the shape, its bounds, the format; a leaf or a record), why a
// tensor does not fit it, and the geometry of the first leaf that passed every push-time check,
// which lets a repeated geometry skip them. Pure functions of Core's InputOptions and Tensor; no
// Session, no slot.

#include "nodes/io/Input.h"
#include "pipeline/TensorCore.h"
#include "pipeline/internal/DataflowRuntime.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace simaai::neat::runtime {

#pragma GCC visibility push(hidden)

// One field of a record binding: the Core metadata of a dense field, and where it lives in the
// record's slot (port_spec.h's field_offset and field_extent_offset, resolved once at binding).
struct DataflowFieldBinding {
  TensorDType dtype = TensorDType::UInt8;
  TensorLayout layout = TensorLayout::Unknown;
  std::vector<TensorAxisSemantic> axis_semantics;
  Semantic semantic{};
  std::size_t rank = 0;
  std::size_t offset = 0;        ///< the field's bytes, from the slot start
  std::size_t extent_offset = 0; ///< the field's Extent in the record header
};

// Core's side of the bound input: the port spec a pushed tensor must fit, and the Core metadata
// (dtype, layout, semantic, planes) the runtime does not carry, restored on every pulled tensor.
// A leaf binds a TensorSpec; a record (an item of several tensors: TensorList, TensorSet, a
// positional Tensor::record) a RecordSchema with positional field names "0", "1", ... and one
// DataflowFieldBinding per field.
struct DataflowPulledTensor; // DataflowModelRoute.h

struct DataflowInputBinding {
  enum class Planes { None, Nv12, I420 };
  pipeline_internal::df::PortSpec spec;
  TensorDType dtype = TensorDType::UInt8;
  TensorLayout layout = TensorLayout::Unknown;
  std::vector<TensorAxisSemantic> axis_semantics;
  Semantic semantic{};
  Planes planes = Planes::None;
  std::size_t rank = 0; ///< Core rank: the spec's for a dense tensor, 2 ({H, W}) for planes
  std::vector<DataflowFieldBinding> fields; ///< a record's fields; empty for a leaf
  std::string text;                         ///< the spec as one line, for messages
  /// A model route: what a pulled message's tensor is, where the Fetch's port is the model
  /// stage's output and not this spec. The Run's configuration owns it. Null: the pushed tensor.
  const DataflowPulledTensor* pulled = nullptr;
  bool is_record() const noexcept {
    return !fields.empty();
  }
};

// The input's binding from its declaration and, when given, a tensor (a seed or the first push).
// nullopt with `why` set when this backend does not carry it: the fact about the item, in the
// present tense (DataflowNotCarried::fact, the input its subject). A declared shape
// (InputOptions::shape) binds from the declaration alone; each of its dynamic axes needs a bound
// (max_shape).
std::optional<DataflowInputBinding> bind_dataflow_input(const InputOptions& declared,
                                                        const Tensor* tensor, std::string* why);

// A record binding from the tensors of one item (2 or more): every field dense, bound by its
// tensor (dtype and shape fixed). nullopt with `why` set, as above, when this backend does not
// carry it.
std::optional<DataflowInputBinding> bind_dataflow_record(const InputOptions& declared,
                                                         std::span<const Tensor> fields,
                                                         std::string* why);

// The binding of a seed or a first item: a leaf for one tensor (or none: the declaration alone), a
// record for several.
inline std::optional<DataflowInputBinding>
bind_dataflow_item(const InputOptions& declared, std::span<const Tensor> item, std::string* why) {
  return item.size() > 1U
             ? bind_dataflow_record(declared, item, why)
             : bind_dataflow_input(declared, item.empty() ? nullptr : item.data(), why);
}

// Fixes every bounded axis of a dense leaf `binding` at its bound (a model's input has one size)
// and describes it again. "" when done; otherwise why the spec is not valid.
std::string fix_dataflow_binding_at_bound(DataflowInputBinding& binding);

// "" when `tensor` fits the leaf `binding`; otherwise why it does not. A record binding is
// dataflow_record_mismatch's.
std::string dataflow_binding_mismatch(const DataflowInputBinding& binding, const Tensor& tensor);

// "" when the item's `fields` fit the record `binding`; otherwise why they do not.
std::string dataflow_record_mismatch(const DataflowInputBinding& binding,
                                     std::span<const Tensor> fields);

// Whether a leaf's bytes are laid out exactly as a whole slot (`slot_bytes`) of `binding`'s port,
// so that the slot can be backed by the tensor's own storage: CPU storage for a host port, device
// storage (DeviceHandle or GstSample) for a device port. Reads only what DataflowLeafGeometry
// keeps, so a repeated geometry reuses the answer.
bool dataflow_wrap_layout(const DataflowInputBinding& binding, std::size_t slot_bytes,
                          const Tensor& tensor);

// What validate_public_input and dataflow_binding_mismatch read from a leaf tensor
// (validate_declared_tensor_storage, describe_tensor_geometry_or_throw, validate_public_geometry),
// plus the rest of its image spec. Both are pure functions of these
// fields and of the Run's InputOptions and InputStreamOptions, which never change: a tensor whose
// geometry equals one that passed both passes both. Only CPU storage and no Semantic beyond an
// image spec are kept, so a tensor with anything else is always checked in full.
// A device input also keeps the geometry of a device tensor (DeviceHandle or GstSample), in a cache
// of its own (keepable_device, matches_device): both functions read the same fields of it, the
// storage's kind and size among them. What they do not read is asked of every push of such a
// tensor: its memory, and whose it is (DataflowDeviceMemory.h).
struct DataflowLeafGeometry {
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
  /// dataflow_wrap_layout of this geometry against the input's binding, computed once.
  bool wrappable = false;

  static bool keepable(const Tensor& t) noexcept;
  static bool keepable_device(const Tensor& t) noexcept;
  static DataflowLeafGeometry of(const Tensor& t);
  bool matches(const Tensor& t) const noexcept;
  bool matches_device(const Tensor& t) const noexcept;
};

// Inline: every push of a leaf asks it. matches_device compares the same fields.
inline bool DataflowLeafGeometry::matches(const Tensor& t) const noexcept {
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

// The validated-geometry cache of one input: the first leaf that passed validate_public_input and
// the binding check. Written once, under the input's mutex; read lock-free (release/acquire).
class DataflowValidatedLeaf {
public:
  // The cached geometry when `item` is one leaf that repeats it; nullptr otherwise.
  const DataflowLeafGeometry* find(std::span<const Tensor> item) const noexcept {
    const DataflowLeafGeometry* known =
        item.size() == 1U ? leaf_.load(std::memory_order_acquire) : nullptr;
    return known && known->matches(item.front()) ? known : nullptr;
  }
  // The same of a device input's cache of a device tensor (matches_device).
  const DataflowLeafGeometry* find_device(std::span<const Tensor> item) const noexcept;
  bool empty() const noexcept {
    return leaf_.load(std::memory_order_acquire) == nullptr;
  }
  // Keeps `tensor`'s geometry if none is kept yet. The caller holds the input's mutex.
  void remember(const Tensor& tensor, bool wrappable);

private:
  std::unique_ptr<DataflowLeafGeometry> storage_;
  std::atomic<const DataflowLeafGeometry*> leaf_{nullptr};
};

#pragma GCC visibility pop

} // namespace simaai::neat::runtime
