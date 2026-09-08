#pragma once

namespace simaai::neat {

/// Geometry of decoded instance masks.
enum class MaskSize { Native = 0, Source = 1, Fixed = 2 };
/// Native probabilities preserve information for application-side resizing.
enum class MaskOutput { Binary = 1, Probabilities = 2 };
struct MaskOptions {
  double threshold = 0.5;
  MaskSize size = MaskSize::Native;
  int width = 0;
  int height = 0;
  MaskOutput output = MaskOutput::Binary;
};

} // namespace simaai::neat
