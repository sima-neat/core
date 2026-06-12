#include "pipeline/internal/InputStreamUtil.h"
#include "nodes/io/Input.h"

#include "test_main.h"

#if !defined(SIMA_WITH_OPENCV)
#error "unit_tensor_from_cv_mat_exactness_test requires SIMA_WITH_OPENCV"
#endif

#include <opencv2/core.hpp>

#include <string>
#include <vector>

RUN_TEST("unit_tensor_from_cv_mat_exactness_test", [] {
  using simaai::neat::InputOptions;

  InputOptions fp32_opt;
  fp32_opt.payload_type = simaai::neat::PayloadType::Tensor;
  fp32_opt.format = "FP32";

  cv::Mat fp32(4, 5, CV_32FC3, cv::Scalar(0.1f, 0.2f, 0.3f));
  simaai::neat::Tensor fp32_tensor =
      simaai::neat::tensor_from_cv_mat(fp32, fp32_opt, "unit_tensor_from_cv_mat_exactness_test");
  require(fp32_tensor.dtype == simaai::neat::TensorDType::Float32, "expected Float32 tensor");
  require(fp32_tensor.layout == simaai::neat::TensorLayout::HWC, "expected HWC layout");
  require(fp32_tensor.shape == std::vector<int64_t>({4, 5, 3}), "unexpected FP32 tensor shape");

  cv::Mat u8(4, 5, CV_8UC3, cv::Scalar(10, 20, 30));
  bool threw_u8 = false;
  try {
    (void)simaai::neat::tensor_from_cv_mat(u8, fp32_opt,
                                           "unit_tensor_from_cv_mat_exactness_test_u8");
  } catch (const std::exception& e) {
    threw_u8 = true;
    require(std::string(e.what()).find("must already be CV_32F") != std::string::npos,
            std::string("unexpected U8 rejection text: ") + e.what());
  }
  require(threw_u8, "expected U8 tensor cv::Mat ingress to fail");

  InputOptions bf16_opt;
  bf16_opt.payload_type = simaai::neat::PayloadType::Tensor;
  bf16_opt.format = "BF16";
  bool threw_bf16 = false;
  try {
    (void)simaai::neat::tensor_from_cv_mat(fp32, bf16_opt,
                                           "unit_tensor_from_cv_mat_exactness_test_bf16");
  } catch (const std::exception& e) {
    threw_bf16 = true;
    require(std::string(e.what()).find("supports FP32/EVXX_FLOAT32 only") != std::string::npos,
            std::string("unexpected BF16 rejection text: ") + e.what());
  }
  require(threw_bf16, "expected BF16 tensor cv::Mat ingress to fail");
});
