#pragma once

#include "nodes/sima/ImageTransform.h"

namespace simaai::neat {
class Preproc;
namespace internal {
enum class ImageTransformMode { Resize, ColorConvert, Normalization };
// Immutable application policy, never a session, producer loan or numerical implementation.
struct ImageTransformPolicy {
  ImageTransformMode mode = ImageTransformMode::Resize;
  ImageColorOptions color;
  ImageSurfaceLayout surface = ImageSurfaceLayout::Native;
  int max_input_width = 0;
  int max_input_height = 0;
  std::optional<std::array<unsigned char, 3>> yuv_padding;
  bool raw_output() const {
    return mode != ImageTransformMode::Normalization;
  }
  bool dynamic_input() const {
    return mode == ImageTransformMode::Resize;
  }
};
struct ImageTransformAccess {
  static const ImageTransformPolicy* policy(const Preproc&);
  static void set_policy(Preproc&, std::shared_ptr<const ImageTransformPolicy>);
};
} // namespace internal
} // namespace simaai::neat
