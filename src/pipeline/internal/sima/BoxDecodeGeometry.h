#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif
#include "pipeline/internal/sima/BoxDecodeKernelRecipe.h"
#include <simaai/runtime/gstreamer/prepared_tensor_stage.h>
namespace simaai::neat::pipeline_internal::sima {
// Install only on the numerical BoxDecode member. Explicit original-size
// properties retain their precedence; otherwise each frame supplies geometry.
void attach_boxdecode_geometry_resolver(KernelStageRecipe&, std::size_t member,
                                        BoxDecodeGeometryOptions, bool original_size_from_metadata,
                                        bool require_resize_metadata);
} // namespace simaai::neat::pipeline_internal::sima
