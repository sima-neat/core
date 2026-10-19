#pragma once

#include "builder/Node.h"

#include <array>
#include <memory>
#include <optional>
#include <string>

namespace simaai::neat {
/// Raw UInt8 image formats. Numerical tensor dtypes are intentionally separate.
enum class ImageFormat { RGB, BGR, GRAY8, NV12, I420, NV16 };
enum class ResizeInterpolation { Nearest, Bilinear, Bicubic, BoxAverage };
enum class ResizeFit { Stretch, Contain };
enum class ImageColorMatrix { Bt601, Bt709 };
enum class ImageYuvRange { Full, Limited };
enum class ImageChromaSiting { Centered, Cosited };
enum class ImageSurfaceLayout {
  Native,      ///< Tight packed rows or 16-byte YUV pitches; preserve padding holes.
  EncoderNv12, ///< Encoder NV12 pitch/storage rows; initialize declared padding on EV every frame.
};

/// YUV color interpretation. These settings do not perform transfer-function conversion.
struct ImageColorOptions {
  ImageColorMatrix matrix = ImageColorMatrix::Bt601;
  ImageYuvRange range = ImageYuvRange::Full;
  std::array<ImageChromaSiting, 2> input_siting{ImageChromaSiting::Centered,
                                                ImageChromaSiting::Centered};
  std::array<ImageChromaSiting, 2> output_siting{ImageChromaSiting::Centered,
                                                 ImageChromaSiting::Centered};
};

/// Resize changes geometry only. Output is a raw image in the input format.
struct ResizeOptions {
  int width = 0;
  int height = 0;
  ResizeInterpolation interpolation = ResizeInterpolation::Bilinear;
  ResizeFit fit = ResizeFit::Stretch;
  int max_input_width = 0;  ///< Zero fixes capacity to the first prepared input width.
  int max_input_height = 0; ///< Zero fixes capacity to the first prepared input height.
  int pad_value = 0;
  std::optional<std::array<unsigned char, 3>> yuv_padding;
  ImageColorOptions color;
  ImageSurfaceLayout surface = ImageSurfaceLayout::Native;
  std::string element_name;
};

/// Color conversion preserves the fixed geometry negotiated for this run.
/// GRAY8 output requires GRAY8 input; RGB/GRAY-to-YUV requires full/centered color policy.
struct ColorConvertOptions {
  ImageFormat output_format = ImageFormat::RGB;
  ImageColorOptions color;
  ImageSurfaceLayout surface = ImageSurfaceLayout::Native;
  std::string element_name;
};

/// UInt8 RGB/BGR/GRAY8 to dense BF16; coefficients are in semantic RGB order.
/// Arithmetic follows Preproc: (x - float(mean * 255)) * float(1 / (stddev * 255)).
/// No resize, color conversion, tessellation or implicit CPU fallback is performed.
struct NormalizationOptions {
  std::array<float, 3> mean{0.0f, 0.0f, 0.0f};
  std::array<float, 3> stddev{1.0f, 1.0f, 1.0f};
  std::string element_name;
};
} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Named Preproc frontends; canonical Node::kind() remains "Preproc".
std::shared_ptr<Node> Resize(ResizeOptions options);
std::shared_ptr<Node> ColorConvert(ColorConvertOptions options = {});
std::shared_ptr<Node> Normalization(NormalizationOptions options = {});
} // namespace simaai::neat::nodes
