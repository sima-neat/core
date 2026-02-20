/**
 * @file
 * @ingroup tensors
 * @brief OpenCV helpers for Tensor.
 */
#pragma once

#include "pipeline/TensorCore.h"

#if defined(SIMA_WITH_OPENCV)
namespace simaai::neat {

Tensor from_cv_mat(const cv::Mat& mat, ImageSpec::PixelFormat fmt = ImageSpec::PixelFormat::BGR,
                   bool read_only = true);

} // namespace simaai::neat
#endif
