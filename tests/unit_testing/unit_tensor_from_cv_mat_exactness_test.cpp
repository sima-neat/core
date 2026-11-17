#include "gst/GstInit.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "nodes/io/Input.h"
#include "pipeline/internal/GstDataAdapter.h"

#include <gst/allocators/gstdmabuf.h>

#include "test_main.h"

#if !defined(SIMA_WITH_OPENCV)
#error "unit_tensor_from_cv_mat_exactness_test requires SIMA_WITH_OPENCV"
#endif

#include <opencv2/core.hpp>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

RUN_TEST(
    "unit_tensor_from_cv_mat_exactness_test", ([] {
      using namespace simaai::neat;
      gst_init_once();

      // Exercise the common converter used by connected graph build and push.
      // Check real transport backing, not only the Tensor's device label.
      for (const auto policy : {InputMemoryPolicy::Auto, InputMemoryPolicy::Ev74}) {
        for (const int type : {CV_8UC3, CV_8UC1, CV_32FC3}) {
          cv::Mat parent(5, 7, type);
          cv::randu(parent, cv::Scalar::all(0), cv::Scalar::all(100));
          cv::Mat roi = parent(cv::Rect(1, 1, 4, 3));
          const cv::Mat expected = roi.clone();
          InputOptions options;
          options.memory_policy = policy;
          Tensor owned = tensor_from_cv_mat(roi, options, "owning Mat ingress");
          require(owned.storage && owned.storage->kind == StorageKind::GstSample,
                  "default/EV74 Mat ingress must retain device-backed storage");
          require(owned.device.type == DeviceType::SIMA_CVU && owned.is_contiguous(),
                  "owning Mat ingress must publish a dense EV74-addressable tensor");
          require(owned.shape == (roi.channels() == 1 ? std::vector<int64_t>{3, 4}
                                                      : std::vector<int64_t>{3, 4, 3}),
                  "Mat ingress must preserve logical dimensions");
          require(owned.dtype ==
                      (roi.depth() == CV_32F ? TensorDType::Float32 : TensorDType::UInt8),
                  "Mat ingress must preserve dtype without conversion");
          std::string error;
          std::unique_ptr<GstBuffer, decltype(&gst_buffer_unref)> buffer(
              pipeline_internal::buffer_from_holder_if_gstsample(owned, &error), gst_buffer_unref);
          require(buffer && gst_buffer_n_memory(buffer.get()) == 1U &&
                      gst_is_dmabuf_memory(gst_buffer_peek_memory(buffer.get(), 0)),
                  "Mat ingress must carry actual DMA-BUF memory: " + error);
          roi.setTo(cv::Scalar::all(0));
          parent.release();
          roi.release();
          const auto mapping = owned.map_read();
          const auto bytes = expected.total() * expected.elemSize();
          require(mapping.data && mapping.size_bytes >= bytes &&
                      std::memcmp(mapping.data, expected.data, bytes) == 0,
                  "owning Mat ingress must copy ROI rows exactly and outlive its source");
        }
      }

      // Explicit host policy and the legacy pool opt-out remain borrowed views.
      for (const bool explicit_policy : {true, false}) {
        for (const int type : {CV_8UC3, CV_32FC3}) {
          InputOptions options;
          options.memory_policy =
              explicit_policy ? InputMemoryPolicy::SystemMemory : InputMemoryPolicy::Auto;
          options.use_simaai_pool = explicit_policy;
          cv::Mat parent(5, 7, type, cv::Scalar::all(1));
          cv::Mat roi = parent(cv::Rect(1, 1, 4, 3));
          const Tensor borrowed = tensor_from_cv_mat(roi, options, "host Mat ingress");
          require(borrowed.storage && borrowed.storage->kind == StorageKind::CpuExternal &&
                      borrowed.device.type == DeviceType::CPU && !borrowed.is_contiguous(),
                  "explicit host ingress must preserve the borrowed ROI storage");
          const auto mapping = borrowed.map_read();
          require(mapping.data == roi.data,
                  "host ingress must not silently allocate DMA-BUF storage");
        }
      }

      InputOptions fp32_opt;
      fp32_opt.payload_type = simaai::neat::PayloadType::Tensor;
      fp32_opt.format = "FP32";

      cv::Mat fp32(4, 5, CV_32FC3, cv::Scalar(0.1f, 0.2f, 0.3f));
      simaai::neat::Tensor fp32_tensor = simaai::neat::tensor_from_cv_mat(
          fp32, fp32_opt, "unit_tensor_from_cv_mat_exactness_test");
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
    }));
