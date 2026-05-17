#pragma once

#include "pipeline/Tensor.h"

#include <opencv2/core/mat.hpp>

namespace simaai::neat::genai::internal {

cv::Mat tensor_to_rgb_mat(const Tensor& image);

} // namespace simaai::neat::genai::internal
