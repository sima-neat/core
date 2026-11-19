#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/StageRun.h"

namespace simaai::neat::stages::internal {

enum class PreprocOutput { Retained, ForImmediateMla };

Tensor preproc_tensor(const cv::Mat& input, const Model& model, PreprocOutput output,
                      ImageSpec::PixelFormat image_format = ImageSpec::PixelFormat::UNKNOWN);

} // namespace simaai::neat::stages::internal
