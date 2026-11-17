#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif
#include "pipeline/internal/sima/BoxDecodeKernelRecipe.h"
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
} // namespace simaai::neat::pipeline_internal::sima
