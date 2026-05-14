#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/SessionOptions.h"
#include "pipeline/Tensor.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/TensorUtil.h"

#include <cstdint>
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

inline simaai::neat::Tensor
apply_output_tensor_override_entry(const simaai::neat::Tensor& base,
                                   const OutputTensorOverrideEntry& entry, bool materialize_output);

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
    Tensor tensor = apply_output_tensor_override_entry(base_tensor, entry, materialize_output);
    tensor.route.logical_index =
        (entry.logical_output_index >= 0) ? entry.logical_output_index : static_cast<int>(i);
    tensor.route.memory_index = entry.memory_index;
    tensor.route.physical_index =
        (entry.memory_index >= 0) ? entry.memory_index : tensor.route.physical_index;
    tensor.route.physical_byte_offset = entry.byte_offset;
    tensor.route.segment_name =
        !entry.segment_name.empty()
            ? entry.segment_name
            : (!entry.name.empty() ? entry.name : ("output" + std::to_string(i)));
    tensor.route.name = !entry.name.empty() ? entry.name : ("output" + std::to_string(i));
    bundle.tensors.emplace_back(std::move(tensor));
  }
  return bundle;
}

inline simaai::neat::Tensor
apply_output_tensor_override_entry(const simaai::neat::Tensor& base,
                                   const OutputTensorOverrideEntry& entry,
                                   bool materialize_output) {
  simaai::neat::Tensor out_source = base;
  if (entry.memory_index >= 0 && base.storage &&
      base.storage->kind == simaai::neat::StorageKind::GstSample) {
    out_source = materialize_output
                     ? pipeline_internal::copy_tensor_from_sample_memory(base, entry.memory_index,
                                                                         /*keep_holder=*/false)
                     : pipeline_internal::tensor_view_from_sample_memory(base, entry.memory_index,
                                                                         /*keep_holder=*/true);
  } else if (materialize_output) {
    out_source = base.clone();
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
    const std::size_t field_count = std::min(out.tensors.size(), override.outputs.size());
    for (std::size_t i = 0; i < field_count; ++i) {
      const auto& entry = override.outputs[i];
      auto& tensor = out.tensors[i];
      if (canonical.tensors.size() == 1U) {
        if (materialize_output) {
          tensor = apply_output_tensor_override_entry(canonical.tensors[i], entry, true);
        } else {
          overlay_output_tensor_override_entry(tensor, entry);
        }
      } else {
        // TensorSet metadata already describes the per-output physical layout.
        // In the multi-tensor case, the override only needs to patch routing/names.
        if (tensor.shape.empty() && !entry.shape.empty()) {
          tensor.shape = entry.shape;
        }
        if (tensor.layout == TensorLayout::Unknown && entry.layout != TensorLayout::Unknown) {
          tensor.layout = entry.layout;
        }
        if (tensor.strides_bytes.empty() && !tensor.shape.empty()) {
          const std::size_t elem_bytes = pipeline_internal::dtype_bytes(tensor.dtype);
          if (elem_bytes > 0U) {
            tensor.strides_bytes =
                pipeline_internal::contiguous_strides_bytes(tensor.shape, elem_bytes);
          }
        }
        if (!entry.format.empty()) {
          if (!tensor.semantic.tess.has_value()) {
            simaai::neat::TessSpec tess;
            tess.format = entry.format;
            tensor.semantic.tess = tess;
          } else {
            tensor.semantic.tess->format = entry.format;
          }
        }
      }
      tensor.route.logical_index =
          (entry.logical_output_index >= 0) ? entry.logical_output_index : static_cast<int>(i);
      tensor.route.memory_index = entry.memory_index;
      tensor.route.physical_index =
          (entry.memory_index >= 0) ? entry.memory_index : tensor.route.physical_index;
      tensor.route.physical_byte_offset = entry.byte_offset;
      tensor.route.segment_name =
          !entry.segment_name.empty()
              ? entry.segment_name
              : (!entry.name.empty() ? entry.name : tensor.route.segment_name);
      if (!entry.name.empty()) {
        tensor.route.name = entry.name;
      }
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
    out.kind = SampleKind::TensorSet;
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
    if (entry.logical_output_index >= 0) {
      out.output_index = entry.logical_output_index;
      out.logical_output_index = entry.logical_output_index;
    }
    out.memory_index = entry.memory_index;
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
