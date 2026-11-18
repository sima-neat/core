#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif
#include "pipeline/internal/sima/BoxDecodeKernelRecipe.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include <string_view>
#include <simaai/scheduler/client/stage.h>
namespace simaai::neat::pipeline_internal::sima {
// Install only on the numerical BoxDecode member. Explicit original-size
// properties retain their precedence; otherwise each frame supplies geometry.
void attach_boxdecode_geometry_resolver(KernelStageRecipe&, std::size_t member,
                                        BoxDecodeGeometryOptions, bool original_size_from_metadata,
                                        bool require_resize_metadata);
// The numerical member's request for frames that all carry the same preprocess facts: an image
// of `original_width` x `original_height` (0, 0: the frames carry none) resized by `resize_mode`
// ("" when they do not say). It is what the resolver above selects for such a frame, by its
// rules and with its texts: the cold request when the size is explicit or equals the seed's,
// otherwise the request with the image size the "image_size" variant would give the kernel. For
// a backend that fixes the frame geometry when its input binds and makes one kernel for it. A
// complete-geometry profile (each frame's affine) has no such request.
::simaai::scheduler::client::Result<KernelStageRecipe::Request>
boxdecode_request_for_frames(const KernelStageRecipe&, int original_width, int original_height,
                             std::string_view resize_mode);
} // namespace simaai::neat::pipeline_internal::sima
