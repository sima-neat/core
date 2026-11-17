#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/TensorCore.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/RenderedStageQueryTypes.h"

namespace simaai::neat::pipeline_internal {

// Wire caps belong to InputOptions. They must not rewrite the source view
// presented to admission or turn a padded/packed tensor into invented dense data.
inline Tensor stage_source_tensor_view(const Tensor& input) {
  return input;
}

// Legacy negotiation dimensions only: reuse the common layout-aware projection,
// retaining the full tensor shape, byte strides, axis semantics and carrier view.
inline stages::TensorDims stage_tensor_wire_dims(const Tensor& input) {
  const auto compat = tensor_compat_dims_from_shape(input.shape, input.layout);
  stages::TensorDims dims;
  dims.width = compat.width;
  dims.height = compat.height;
  dims.depth = compat.depth;
  if (input.is_composite() && !input.planes.empty()) {
    const auto& primary = input.planes.front();
    if (primary.shape.size() >= 2U) {
      if (dims.height <= 0) {
        dims.height = static_cast<int>(primary.shape[0]);
      }
      if (dims.width <= 0) {
        dims.width = static_cast<int>(primary.shape[1]);
      }
    }
  }
  return dims;
}

} // namespace simaai::neat::pipeline_internal
