#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif
#include "pipeline/internal/sima/BoxDecodeKernelRecipe.h"
#include <simaai/scheduler/client/stage.h>
#include <array>
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
// Install only on the numerical BoxDecode member. Explicit original-size
// properties retain their precedence; otherwise each frame supplies geometry.
void attach_boxdecode_geometry_resolver(KernelStageRecipe&, std::size_t member,
                                        BoxDecodeGeometryOptions, bool original_size_from_metadata,
                                        bool require_resize_metadata);
// For outputs sized by the original image (Segment Anything). The prepared plan keeps
// its image and preprocessing; a frame whose metadata reports another size, resize
// mode, canvas or placement fails loudly.
void attach_boxdecode_fixed_image_guard(KernelStageRecipe&, std::size_t member,
                                        BoxDecodeFixedImage);
} // namespace simaai::neat::pipeline_internal::sima
