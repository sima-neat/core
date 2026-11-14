#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif
#include "pipeline/internal/sima/BoxDecodeKernelRecipe.h"
#include <simaai/scheduler/client/stage.h>
namespace simaai::neat::pipeline_internal::sima {
// Install only on the numerical BoxDecode member. Explicit original-size
// properties retain their precedence; otherwise each frame supplies geometry.
void attach_boxdecode_geometry_resolver(KernelStageRecipe&, std::size_t member,
                                        BoxDecodeGeometryOptions, bool original_size_from_metadata,
                                        bool require_resize_metadata);
// For outputs sized by the original image (Segment Anything). The prepared plan keeps
// its image size; a frame whose metadata reports another size fails loudly.
void attach_boxdecode_fixed_image_guard(KernelStageRecipe&, std::size_t member, int original_width,
                                        int original_height);
} // namespace simaai::neat::pipeline_internal::sima
