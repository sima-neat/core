#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/GraphOptions.h"
#include "pipeline/Tensor.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/TensorUtil.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace simaai::neat {

struct OutputTensorOverrideEntry {
  std::vector<int64_t> shape;
  std::vector<int64_t> strides_bytes;
  int64_t byte_offset = 0;
  int memory_index = -1;
  int logical_output_index = -1;
  int route_slot = -1;
  TensorDType dtype = TensorDType::UInt8;
  TensorLayout layout = TensorLayout::Unknown;
  std::string format;
  std::string name;
  std::string segment_name;
};

struct OutputTensorOverride {
  std::vector<OutputTensorOverrideEntry> outputs;
};

inline int estimate_public_zero_copy_holder_arity(const OutputTensorOverride& override) {
  if (override.outputs.empty()) {
    return 1;
  }

  std::unordered_set<std::string> keys;
  bool all_keyed = true;
  for (std::size_t i = 0; i < override.outputs.size(); ++i) {
    const auto& entry = override.outputs[i];
    if (!entry.segment_name.empty()) {
      keys.insert("seg:" + entry.segment_name);
    } else if (entry.memory_index >= 0) {
      keys.insert("mem:" + std::to_string(entry.memory_index));
    } else if (entry.logical_output_index >= 0) {
      // Logical output index is not a physical-holder identity, but it is a
      // conservative fallback for static multi-output contracts when segment
      // metadata is unavailable.
      keys.insert("logical:" + std::to_string(entry.logical_output_index));
    } else {
      all_keyed = false;
      break;
    }
  }

  if (!all_keyed || keys.empty()) {
    return static_cast<int>(std::max<std::size_t>(1U, override.outputs.size()));
  }
  return static_cast<int>(std::max<std::size_t>(1U, keys.size()));
}

inline std::uint64_t
output_override_entry_physical_span_bytes(const OutputTensorOverrideEntry& entry) {
  const std::uint64_t elem_bytes =
      static_cast<std::uint64_t>(pipeline_internal::dtype_bytes(entry.dtype));
  if (elem_bytes == 0U || entry.shape.empty()) {
    return 0U;
  }
  if (!entry.strides_bytes.empty() && entry.strides_bytes.size() == entry.shape.size()) {
    std::uint64_t max_offset = 0U;
    for (std::size_t i = 0; i < entry.shape.size(); ++i) {
      const std::int64_t dim = entry.shape[i];
      const std::int64_t stride = entry.strides_bytes[i];
      if (dim <= 0 || stride < 0 || (dim > 1 && stride == 0)) {
        return 0U;
      }
      if (dim == 1) {
        continue;
      }
      const auto dim_minus_one = static_cast<std::uint64_t>(dim - 1);
      const auto stride_bytes = static_cast<std::uint64_t>(stride);
      if (stride_bytes > std::numeric_limits<std::uint64_t>::max() / dim_minus_one) {
        return 0U;
      }
      const std::uint64_t delta = dim_minus_one * stride_bytes;
      if (max_offset > std::numeric_limits<std::uint64_t>::max() - delta) {
        return 0U;
      }
      max_offset += delta;
    }
    if (max_offset > std::numeric_limits<std::uint64_t>::max() - elem_bytes) {
      return 0U;
    }
    return max_offset + elem_bytes;
  }

  std::uint64_t elements = 1U;
  for (const auto dim : entry.shape) {
    if (dim <= 0) {
      return 0U;
    }
    const auto u_dim = static_cast<std::uint64_t>(dim);
    if (elements > std::numeric_limits<std::uint64_t>::max() / u_dim) {
      return 0U;
    }
    elements *= u_dim;
  }
  if (elements > std::numeric_limits<std::uint64_t>::max() / elem_bytes) {
    return 0U;
  }
  return elements * elem_bytes;
}

inline simaai::neat::Tensor
apply_output_tensor_override_entry(const simaai::neat::Tensor& base,
                                   const OutputTensorOverrideEntry& entry, bool materialize_output);

inline void apply_output_tensor_override_route(simaai::neat::Tensor& tensor,
                                               const OutputTensorOverrideEntry& entry,
                                               std::size_t fallback_index) {
  const bool keep_resolved_segment_route = !entry.segment_name.empty() &&
                                           tensor.route.segment_name == entry.segment_name &&
                                           tensor.route.memory_index >= 0;
  const int resolved_memory_index = tensor.route.memory_index;
  const int resolved_physical_index = tensor.route.physical_index;
  tensor.route.logical_index = (entry.logical_output_index >= 0) ? entry.logical_output_index
                                                                 : static_cast<int>(fallback_index);
  tensor.route.route_slot = (entry.route_slot >= 0) ? entry.route_slot : tensor.route.route_slot;
  tensor.route.memory_index =
      keep_resolved_segment_route ? resolved_memory_index : entry.memory_index;
  tensor.route.physical_index =
      keep_resolved_segment_route
          ? resolved_physical_index
          : ((entry.memory_index >= 0) ? entry.memory_index : tensor.route.physical_index);
  tensor.route.physical_byte_offset = entry.byte_offset;
  tensor.route.segment_name = !entry.segment_name.empty()
                                  ? entry.segment_name
                                  : (!entry.name.empty() ? entry.name : tensor.route.segment_name);
  if (!entry.name.empty()) {
    tensor.route.name = entry.name;
  }
}

inline const simaai::neat::Tensor* select_output_tensor_override_base(
    const TensorList& tensors, const OutputTensorOverrideEntry& entry, std::size_t fallback_index) {
  if (tensors.empty()) {
    return nullptr;
  }
  if (entry.logical_output_index >= 0) {
    for (const auto& tensor : tensors) {
      if (tensor.route.logical_index == entry.logical_output_index) {
        return &tensor;
      }
    }
  }
  if (entry.memory_index >= 0) {
    for (const auto& tensor : tensors) {
      if (tensor.route.memory_index == entry.memory_index ||
          tensor.route.physical_index == entry.memory_index) {
        return &tensor;
      }
    }
  }
  if (fallback_index < tensors.size()) {
    return &tensors[fallback_index];
  }
  return &tensors.front();
}

inline Sample build_output_tensor_override_bundle(const Sample& canonical,
                                                  const Tensor& base_tensor,
                                                  const OutputTensorOverride& override,
                                                  bool materialize_output) {
  std::unordered_set<int> unique_memory_indices;
  for (const auto& entry : override.outputs) {
    if (entry.memory_index >= 0) {
      unique_memory_indices.insert(entry.memory_index);
    }
  }
  pipeline_internal::record_tensor_bundle_projection(override.outputs.size());
  pipeline_internal::record_tensor_packed_view_reuse(
      override.outputs.size(), unique_memory_indices.size(), materialize_output);

  Sample bundle;
  bundle.kind = SampleKind::TensorSet;
  bundle.owned = canonical.owned;
  bundle.caps_string = canonical.caps_string;
  bundle.payload_type = canonical.payload_type;
  bundle.media_type = canonical.media_type;
  bundle.payload_tag = canonical.payload_tag;
  bundle.format = canonical.format;
  bundle.frame_id = canonical.frame_id;
  bundle.stream_id = canonical.stream_id;
  bundle.stream_label = canonical.stream_label;
  bundle.output_index = canonical.output_index;
  bundle.logical_output_index = canonical.logical_output_index;
  bundle.memory_index = canonical.memory_index;
  bundle.route_slot = canonical.route_slot;
  bundle.segment_name = canonical.segment_name;
  bundle.input_seq = canonical.input_seq;
  bundle.orig_input_seq = canonical.orig_input_seq;
  bundle.pts_ns = canonical.pts_ns;
  bundle.dts_ns = canonical.dts_ns;
  bundle.duration_ns = canonical.duration_ns;

  bundle.tensors.reserve(override.outputs.size());
  for (size_t i = 0; i < override.outputs.size(); ++i) {
    const OutputTensorOverrideEntry& entry = override.outputs[i];
    Tensor base = base_tensor;
    base.byte_offset = 0;
    Tensor tensor = apply_output_tensor_override_entry(base, entry, materialize_output);
    apply_output_tensor_override_route(tensor, entry, i);
    if (tensor.route.segment_name.empty()) {
      tensor.route.segment_name = "output" + std::to_string(i);
    }
    if (tensor.route.name.empty()) {
      tensor.route.name = "output" + std::to_string(i);
    }
    bundle.tensors.emplace_back(std::move(tensor));
  }
  return bundle;
}

inline simaai::neat::Tensor
apply_output_tensor_override_entry(const simaai::neat::Tensor& base,
                                   const OutputTensorOverrideEntry& entry,
                                   bool materialize_output) {
  simaai::neat::Tensor out_source = base;
  if ((entry.memory_index >= 0 || !entry.segment_name.empty()) && base.storage &&
      base.storage->kind == simaai::neat::StorageKind::GstSample) {
    // Build the authoritative output segment view first, then apply shape/dtype/stride below.
    // For SimaaiSegmentMemory buffers the GstBuffer usually has one parent memory and multiple
    // named runtime segments.  The public output contract's segment_name is more specific than
    // memory_index (which may be a physical-output id, route id, or historical segment index), so
    // resolve by name first and use memory_index only as a fallback for legacy multi-GstMemory
    // buffers.
    out_source = pipeline_internal::tensor_view_from_sample_segment(
        base, entry.segment_name, entry.memory_index, /*keep_holder=*/true);
  }
  simaai::neat::Tensor out = std::move(out_source);
  out.shape = entry.shape;
  out.byte_offset = out.byte_offset + entry.byte_offset;
  out.dtype = entry.dtype;
  if (entry.layout != TensorLayout::Unknown) {
    out.layout = entry.layout;
  }
  out.planes.clear();
  const std::size_t elem_bytes = pipeline_internal::dtype_bytes(out.dtype);
  if (!entry.strides_bytes.empty() && entry.strides_bytes.size() == out.shape.size()) {
    out.strides_bytes = entry.strides_bytes;
  } else if (!out.shape.empty() && elem_bytes > 0) {
    out.strides_bytes = pipeline_internal::contiguous_strides_bytes(out.shape, elem_bytes);
  } else {
    out.strides_bytes.clear();
  }
  if (!entry.format.empty()) {
    if (!out.semantic.tess.has_value()) {
      simaai::neat::TessSpec tess;
      tess.format = entry.format;
      out.semantic.tess = tess;
    } else {
      out.semantic.tess->format = entry.format;
    }
  }
  if (materialize_output) {
    if (out.storage && out.storage->kind == simaai::neat::StorageKind::GstSample) {
      if (out.byte_offset < 0) {
        throw std::runtime_error("output override materialize: negative logical byte offset");
      }
      const std::uint64_t required_u64 = output_override_entry_physical_span_bytes(entry);
      if (required_u64 == 0U ||
          required_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("output override materialize: invalid logical output span");
      }
      const std::size_t required = static_cast<std::size_t>(required_u64);
      const simaai::neat::Mapping src = out.view_read(); // includes entry.byte_offset
      if (!src.data && required != 0U) {
        throw std::runtime_error("output override materialize: failed to map output segment");
      }
      if (src.size_bytes < required) {
        throw std::runtime_error("output override materialize: mapped output segment is smaller "
                                 "than the logical output span");
      }
      auto storage = simaai::neat::make_cpu_owned_storage(required);
      const simaai::neat::Mapping dst =
          storage ? storage->map(simaai::neat::MapMode::Write) : simaai::neat::Mapping{};
      if (!dst.data && required != 0U) {
        throw std::runtime_error("output override materialize: failed to allocate output copy");
      }
      if (dst.size_bytes < required) {
        throw std::runtime_error("output override materialize: output copy allocation too small");
      }
      if (required > 0U) {
        std::memcpy(dst.data, src.data, required);
      }
      out.storage = std::move(storage);
      out.byte_offset = 0;
      out.read_only = false;
      out.device = {simaai::neat::DeviceType::CPU, 0};
    } else {
      out = out.clone();
    }
  }
  return out;
}

inline void overlay_output_tensor_override_entry(simaai::neat::Tensor& out,
                                                 const OutputTensorOverrideEntry& entry) {
  out.shape = entry.shape;
  out.byte_offset = out.byte_offset + entry.byte_offset;
  out.dtype = entry.dtype;
  if (entry.layout != TensorLayout::Unknown) {
    out.layout = entry.layout;
  }
  out.planes.clear();
  const std::size_t elem_bytes = pipeline_internal::dtype_bytes(out.dtype);
  if (!entry.strides_bytes.empty() && entry.strides_bytes.size() == out.shape.size()) {
    out.strides_bytes = entry.strides_bytes;
  } else if (!out.shape.empty() && elem_bytes > 0) {
    out.strides_bytes = pipeline_internal::contiguous_strides_bytes(out.shape, elem_bytes);
  } else {
    out.strides_bytes.clear();
  }
  if (!entry.format.empty()) {
    if (!out.semantic.tess.has_value()) {
      simaai::neat::TessSpec tess;
      tess.format = entry.format;
      out.semantic.tess = tess;
    } else {
      out.semantic.tess->format = entry.format;
    }
  }
}

inline Sample apply_output_tensor_override(const Sample& base, const OutputTensorOverride& override,
                                           bool materialize_output) {
  if (override.outputs.empty())
    return base;
  const Sample canonical = pipeline_internal::canonicalize_tensor_transport_sample(base);
  if (sample_has_tensor_list(canonical)) {
    if (canonical.tensors.size() == 1U && override.outputs.size() > 1U &&
        !canonical.tensors.front().is_composite()) {
      return build_output_tensor_override_bundle(canonical, canonical.tensors.front(), override,
                                                 materialize_output);
    }
    Sample out = canonical;
    out.tensors.clear();
    out.tensors.reserve(override.outputs.size());
    for (std::size_t i = 0; i < override.outputs.size(); ++i) {
      const auto& entry = override.outputs[i];
      const Tensor* selected_base = select_output_tensor_override_base(canonical.tensors, entry, i);
      if (!selected_base) {
        continue;
      }
      Tensor base_tensor = *selected_base;
      // Public overrides are authoritative.  TensorSet metadata may describe a
      // downstream/internal view, so do not add its stale byte_offset to the
      // selected public contract's byte_offset.
      base_tensor.byte_offset = 0;
      Tensor tensor = apply_output_tensor_override_entry(base_tensor, entry, materialize_output);
      apply_output_tensor_override_route(tensor, entry, i);
      out.tensors.emplace_back(std::move(tensor));
    }
    out.kind = SampleKind::TensorSet;
    return out;
  }
  if (canonical.kind == SampleKind::Bundle) {
    Sample out = canonical;
    const std::size_t field_count = std::min(out.fields.size(), override.outputs.size());
    for (std::size_t i = 0; i < field_count; ++i) {
      const auto& entry = override.outputs[i];
      auto& field = out.fields[i];
      if (sample_has_tensor_list(field) && !field.tensors.empty()) {
        overlay_output_tensor_override_entry(field.tensors.front(), entry);
      }
      field.output_index =
          (entry.logical_output_index >= 0) ? entry.logical_output_index : static_cast<int>(i);
      field.logical_output_index = field.output_index;
      field.memory_index = entry.memory_index;
      field.route_slot = (entry.route_slot >= 0) ? entry.route_slot : field.route_slot;
      field.segment_name = !entry.segment_name.empty()
                               ? entry.segment_name
                               : (!entry.name.empty() ? entry.name : field.segment_name);
      if (!entry.name.empty()) {
        field.stream_label = entry.name;
      }
    }
    return out;
  }
  if (!sample_has_tensor_list(canonical) || canonical.tensors.size() != 1U)
    return canonical;
  if (canonical.tensors.front().is_composite())
    return canonical;

  if (override.outputs.size() == 1) {
    Sample out = canonical;
    const auto& entry = override.outputs[0];
    out.tensors.front() =
        apply_output_tensor_override_entry(canonical.tensors.front(), entry, materialize_output);
    apply_output_tensor_override_route(out.tensors.front(), entry, 0U);
    if (entry.logical_output_index >= 0) {
      out.output_index = entry.logical_output_index;
      out.logical_output_index = entry.logical_output_index;
    }
    out.memory_index = out.tensors.front().route.memory_index;
    out.route_slot = (entry.route_slot >= 0) ? entry.route_slot : out.route_slot;
    if (!entry.segment_name.empty()) {
      out.segment_name = entry.segment_name;
    } else if (!entry.name.empty()) {
      out.segment_name = entry.name;
    }
    if (!entry.name.empty()) {
      out.stream_label = entry.name;
    }
    return out;
  }

  return build_output_tensor_override_bundle(canonical, canonical.tensors.front(), override,
                                             materialize_output);
}

} // namespace simaai::neat
