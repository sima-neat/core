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
#include <optional>
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
// One frame's image geometry as its preprocess facts state it: what the image preproc publishes
// with a frame, and what the resolver above reads from a buffer's compact facts.
struct BoxDecodeFrameFacts {
  int original_width = 0, original_height = 0; ///< 0, 0: the frames carry no facts
  int resized_width = 0, resized_height = 0;   ///< the model canvas the frame is resized onto
  std::string_view resize_mode;                ///< "" when the facts do not say
  bool roi_list = false;
  std::array<double, 6> affine{}; ///< model-to-original, row-major 2x3
};
// What the numerical member runs for frames that all carry the same preprocess facts: what the
// resolver above selects for one buffer with those facts, by its rules, in its order and with its
// texts. For a backend that fixes the frame geometry when its input binds and makes one kernel for
// it, or one per frame size.
struct BoxDecodeFrameGeometry {
  /// The resolver selects the stage's cold kernel as it is: the stage takes no geometry from its
  /// frames, or the frames' geometry is the cold seed's.
  bool cold = false;
  /// The variant the stage binds for a frame's geometry, "image_size" or "image_geometry" (a
  /// complete-geometry profile: each frame's affine as well); "" when it binds none. Its values:
  /// the original image's size, and for "image_geometry" the model-to-original transform (scale
  /// x, scale y, offset x, offset y) when the frame or the seed has one.
  std::string_view variant;
  std::array<int, 2> image_size{};
  std::optional<std::array<float, 4>> model_to_original;
  /// The cold request with those values in place of the seed's: the kernel made of it runs such
  /// frames as the variant of the cold kernel does.
  KernelStageRecipe::Request request;
};
::simaai::scheduler::client::Result<BoxDecodeFrameGeometry>
boxdecode_geometry_for_frames(const KernelStageRecipe&, const BoxDecodeFrameFacts&);
} // namespace simaai::neat::pipeline_internal::sima
