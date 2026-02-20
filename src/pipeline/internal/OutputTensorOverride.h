#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/SessionOptions.h"
#include "pipeline/Tensor.h"
#include "pipeline/internal/TensorMath.h"

#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat {

struct OutputTensorOverrideEntry {
  std::vector<int64_t> shape;
  int64_t byte_offset = 0;
  TensorDType dtype = TensorDType::UInt8;
  TensorLayout layout = TensorLayout::Unknown;
  std::string format;
  std::string name;
};

struct OutputTensorOverride {
  std::vector<OutputTensorOverrideEntry> outputs;
};

inline simaai::neat::Tensor
apply_output_tensor_override_entry(const simaai::neat::Tensor& base,
                                   const OutputTensorOverrideEntry& entry) {
  simaai::neat::Tensor out = base;
  out.shape = entry.shape;
  out.byte_offset = base.byte_offset + entry.byte_offset;
  out.dtype = entry.dtype;
  if (entry.layout != TensorLayout::Unknown) {
    out.layout = entry.layout;
  }
  out.planes.clear();
  const std::size_t elem_bytes = pipeline_internal::dtype_bytes(out.dtype);
  if (!out.shape.empty() && elem_bytes > 0) {
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

inline Sample apply_output_tensor_override(const Sample& base,
                                           const OutputTensorOverride& override) {
  if (override.outputs.empty())
    return base;
  if (base.kind != SampleKind::Tensor || !base.tensor.has_value())
    return base;
  if (base.tensor->is_composite())
    return base;

  if (override.outputs.size() == 1) {
    Sample out = base;
    out.tensor = apply_output_tensor_override_entry(*base.tensor, override.outputs[0]);
    return out;
  }

  Sample bundle;
  bundle.kind = SampleKind::Bundle;
  bundle.owned = base.owned;
  bundle.caps_string = base.caps_string;
  bundle.media_type = base.media_type;
  bundle.payload_tag = base.payload_tag;
  bundle.format = base.format;
  bundle.frame_id = base.frame_id;
  bundle.stream_id = base.stream_id;
  bundle.port_name = base.port_name;
  bundle.output_index = base.output_index;
  bundle.input_seq = base.input_seq;
  bundle.orig_input_seq = base.orig_input_seq;
  bundle.pts_ns = base.pts_ns;
  bundle.dts_ns = base.dts_ns;
  bundle.duration_ns = base.duration_ns;

  bundle.fields.reserve(override.outputs.size());
  for (size_t i = 0; i < override.outputs.size(); ++i) {
    const OutputTensorOverrideEntry& entry = override.outputs[i];
    Sample field;
    field.kind = SampleKind::Tensor;
    field.owned = base.owned;
    field.caps_string = base.caps_string;
    field.media_type = base.media_type;
    field.payload_tag = base.payload_tag;
    field.format = base.format;
    field.frame_id = base.frame_id;
    field.stream_id = base.stream_id;
    field.output_index = static_cast<int>(i);
    field.input_seq = base.input_seq;
    field.orig_input_seq = base.orig_input_seq;
    field.pts_ns = base.pts_ns;
    field.dts_ns = base.dts_ns;
    field.duration_ns = base.duration_ns;
    field.tensor = apply_output_tensor_override_entry(*base.tensor, entry);
    if (!entry.name.empty()) {
      field.port_name = entry.name;
    } else {
      field.port_name = "output" + std::to_string(i);
    }
    bundle.fields.emplace_back(std::move(field));
  }
  return bundle;
}

} // namespace simaai::neat
