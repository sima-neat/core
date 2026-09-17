#pragma once

#include "model/internal/ModelRouteRetarget.h"
#include "nodes/sima/Preproc.h"

namespace simaai::neat::internal {

struct ModelManagedPreprocMaxInputShape {
  int height = 0;
  int width = 0;
  int channels = 0;
};

inline ModelManagedPreprocMaxInputShape
model_managed_preproc_max_input_shape(const PreprocOptions& options) {
  ModelManagedPreprocMaxInputShape shape;
  if (!options.model_lineage) {
    return shape;
  }
  const auto& lineage_shape = options.model_lineage->preproc_max_input_shape;
  shape.height = PreprocOptions::shape_dim(lineage_shape, 0);
  shape.width = PreprocOptions::shape_dim(lineage_shape, 1);
  shape.channels = PreprocOptions::shape_channels(lineage_shape);
  return shape;
}

inline PreprocOptions
model_managed_preproc_static_envelope_options(const PreprocOptions& options) {
  PreprocOptions envelope = options;
  if (!options.model_managed_contract || !options.dynamic_input_dims ||
      !options.model_lineage) {
    return envelope;
  }

  const auto shape = model_managed_preproc_max_input_shape(options);
  if (shape.height > 0 && shape.width > 0 && shape.channels > 0) {
    envelope.set_input_shape({shape.height, shape.width, shape.channels});
  }
  return envelope;
}

} // namespace simaai::neat::internal
