#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif
#include "pipeline/internal/sima/BoxDecodeKernelRecipe.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include <string_view>
#include <simaai/scheduler/client/stage.h>
#include <array>
#include <cstdint>
#include <string>
namespace simaai::neat::pipeline_internal::sima {
// The image a prepared Segment Anything plan was admitted for, as its preprocess
// facts published it: the original size, the resize mode, the model canvas and
// the scaled image's {x, y, width, height} on that canvas.
struct BoxDecodeFixedImage {
  int original_width = 0, original_height = 0;
  std::string resize_mode;
  int canvas_width = 0, canvas_height = 0;
  std::array<int, 4> placed{};
};
// Whether a model-to-original affine maps the scaled image's {x, y, width, height}
// on the canvas onto the whole original image, to within half a canvas pixel (below
// the resolution masks are decoded at): the one transform the Segment Anything mask
// kernel reproduces. A crop or ROI composed before the resize maps it onto part of
// the image instead.
bool boxdecode_affine_covers_image(const std::array<double, 6>& model_to_original,
                                   int original_width, int original_height,
                                   const std::array<std::int64_t, 4>& placed);
// Install only on the numerical BoxDecode member. Explicit original-size
// properties retain their precedence; otherwise each frame supplies geometry.
void attach_boxdecode_geometry_resolver(KernelStageRecipe&, std::size_t member,
                                        BoxDecodeGeometryOptions, bool original_size_from_metadata,
                                        bool require_resize_metadata);
// For outputs sized by the original image (Segment Anything). The prepared plan keeps
// its image and preprocessing. Every run needs at least one input with the image's
// facts (inputs without facts, such as prompts, are skipped); a frame whose facts report
// an ROI list, another size, resize mode, canvas or placement, or an affine that does not
// cover the whole image, fails loudly.
void attach_boxdecode_fixed_image_guard(KernelStageRecipe&, std::size_t member,
                                        BoxDecodeFixedImage);
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
