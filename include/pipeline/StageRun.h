#pragma once

#include "pipeline/DetectionTypes.h"
#include "pipeline/TensorCore.h"

#include <string>

namespace cv {
class Mat;
} // namespace cv

namespace simaai::neat {
class Model;

namespace stages {

struct BoxDecodeOptions {
  std::string decode_type = "";
  int original_width = 0;
  int original_height = 0;
  double detection_threshold = 0.0;
  double nms_iou_threshold = 0.0;
  int top_k = 0;
};

simaai::neat::Tensor Preproc(const cv::Mat& input, const simaai::neat::Model& model);
simaai::neat::Tensor Infer(const simaai::neat::Tensor& input, const simaai::neat::Model& model);
simaai::neat::Tensor MLA(const simaai::neat::Tensor& input, const simaai::neat::Model& model);
BoxDecodeResult BoxDecode(const simaai::neat::Tensor& input, const simaai::neat::Model& model,
                          const BoxDecodeOptions& opt = {});

} // namespace stages
} // namespace simaai::neat
