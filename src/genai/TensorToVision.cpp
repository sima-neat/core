#include "genai/TensorToVision.h"

#include <stdexcept>

namespace simaai::neat::genai::internal {

cv::Mat tensor_to_rgb_mat(const Tensor& image) {
  if (image.dtype != TensorDType::UInt8) {
    throw std::runtime_error("GenAI image tensor must have UInt8 dtype");
  }
  if (image.layout != TensorLayout::HWC) {
    throw std::runtime_error("GenAI image tensor must use HWC layout");
  }
  if (image.shape.size() != 3U || image.shape[2] != 3) {
    throw std::runtime_error("GenAI image tensor must have shape [H, W, 3]");
  }
  if (!image.semantic.image.has_value() ||
      image.semantic.image->format != ImageSpec::PixelFormat::RGB) {
    throw std::runtime_error("GenAI image tensor must declare RGB image semantics");
  }

  return image.to_cv_mat_copy(ImageSpec::PixelFormat::RGB);
}

} // namespace simaai::neat::genai::internal
