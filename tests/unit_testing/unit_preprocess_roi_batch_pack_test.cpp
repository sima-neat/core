#include "nodes/io/Input.h"
#include "pipeline/TensorCore.h"
#include "test_main.h"
#include "test_utils.h"

#if !defined(SIMA_WITH_OPENCV)
#error "unit_preprocess_roi_batch_pack_test requires SIMA_WITH_OPENCV"
#endif

#include <opencv2/core.hpp>

#include <cstdint>
#include <exception>
#include <string>
#include <vector>

namespace simaai::neat::stages::internal {
Tensor make_cpu_packed_preproc_source_batch_for_stage(const std::vector<cv::Mat>& inputs,
                                                      const InputOptions& src_opt,
                                                      const PreprocessRuntimeMeta& runtime_meta);
void validate_preproc_roi_list_request_for_stage(const std::vector<cv::Mat>& inputs,
                                                 const std::vector<PreprocessRoi>& rois);
} // namespace simaai::neat::stages::internal

namespace {

void fill_roi_image(cv::Mat image, std::uint8_t base) {
  for (int y = 0; y < image.rows; ++y) {
    auto* row = image.ptr<std::uint8_t>(y);
    for (int x = 0; x < image.cols; ++x) {
      for (int c = 0; c < image.channels(); ++c) {
        row[x * image.channels() + c] = static_cast<std::uint8_t>(base + y * 20 + x * 3 + c);
      }
    }
  }
}

void require_contains(const std::exception& e, const std::string& needle) {
  require(std::string(e.what()).find(needle) != std::string::npos,
          std::string("unexpected exception: ") + e.what());
}

} // namespace

RUN_TEST("unit_preprocess_roi_batch_pack_test", ([] {
           constexpr std::uint8_t kPaddingSentinel = 0xeeU;
           cv::Mat backing0(2, 4, CV_8UC3,
                            cv::Scalar(kPaddingSentinel, kPaddingSentinel, kPaddingSentinel));
           cv::Mat backing1(2, 4, CV_8UC3,
                            cv::Scalar(kPaddingSentinel, kPaddingSentinel, kPaddingSentinel));
           cv::Mat img0 = backing0(cv::Rect(0, 0, 3, 2));
           cv::Mat img1 = backing1(cv::Rect(0, 0, 3, 2));
           require(!img0.isContinuous(), "pack test expects first ROI view to be non-contiguous");
           require(!img1.isContinuous(), "pack test expects second ROI view to be non-contiguous");
           fill_roi_image(img0, 10);
           fill_roi_image(img1, 80);

           simaai::neat::InputOptions src_opt;
           src_opt.payload_type = simaai::neat::PayloadType::Image;
           src_opt.format = "BGR";

           simaai::neat::PreprocessRuntimeMeta meta;
           meta.roi_list_enabled = true;
           meta.roi_input_batch_size = 2;
           meta.roi_source_width = 3;
           meta.roi_source_height = 2;
           meta.roi_source_stride_bytes = 9;
           // ROI output capacity is independent of source-image batch size:
           // three ROI outputs may legally reference two packed source frames.
           meta.rois = {{0, 0, 0, 3, 2}, {1, 0, 0, 3, 2}, {0, 1, 0, 2, 2}};

           simaai::neat::Tensor packed =
               simaai::neat::stages::internal::make_cpu_packed_preproc_source_batch_for_stage(
                   std::vector<cv::Mat>{img0, img1}, src_opt, meta);

           require(packed.shape == std::vector<int64_t>({2, 2, 3, 3}),
                   "packed tensor shape mismatch");
           require(packed.strides_bytes == std::vector<int64_t>({18, 9, 3, 1}),
                   "packed tensor strides mismatch");
           require(packed.axis_semantics.size() == 4U, "packed tensor missing axis semantics");
           require(packed.semantic.preprocess.has_value(),
                   "packed tensor missing preprocess metadata");
           require(packed.semantic.preprocess->roi_input_batch_size == 2,
                   "packed tensor preprocess batch size mismatch");
           require(packed.semantic.preprocess->rois.size() == 3U,
                   "packed tensor should preserve independent ROI output capacity");

           simaai::neat::Mapping mapping = packed.storage->map(simaai::neat::MapMode::Read);
           require(mapping.data != nullptr, "packed tensor map failed");
           const auto* bytes = static_cast<const std::uint8_t*>(mapping.data);
           const std::size_t frame_bytes = 18;
           const std::size_t row_bytes = 9;
           const std::vector<cv::Mat> refs{img0, img1};
           for (std::size_t n = 0; n < refs.size(); ++n) {
             for (int y = 0; y < refs[n].rows; ++y) {
               const auto* src = refs[n].ptr<std::uint8_t>(y);
               for (std::size_t b = 0; b < row_bytes; ++b) {
                 const std::uint8_t packed =
                     bytes[n * frame_bytes + static_cast<std::size_t>(y) * row_bytes + b];
                 require(packed == src[b], "packed tensor byte layout mismatch");
                 require(packed != kPaddingSentinel,
                         "packed tensor copied non-contiguous row padding bytes");
               }
             }
           }

           simaai::neat::stages::internal::validate_preproc_roi_list_request_for_stage(
               std::vector<cv::Mat>{img0, img1}, meta.rois);

           bool threw_bad_index = false;
           try {
             std::vector<simaai::neat::PreprocessRoi> bad_rois = {{2, 0, 0, 3, 2}};
             simaai::neat::stages::internal::validate_preproc_roi_list_request_for_stage(
                 std::vector<cv::Mat>{img0, img1}, bad_rois);
           } catch (const std::exception& e) {
             threw_bad_index = true;
             require_contains(e, "batch_index");
           }
           require(threw_bad_index, "expected out-of-range ROI batch_index to throw");

           bool threw_bad_extent = false;
           try {
             std::vector<simaai::neat::PreprocessRoi> bad_rois = {{0, 0, 0, 0, 2}};
             simaai::neat::stages::internal::validate_preproc_roi_list_request_for_stage(
                 std::vector<cv::Mat>{img0, img1}, bad_rois);
           } catch (const std::exception& e) {
             threw_bad_extent = true;
             require_contains(e, "width/height");
           }
           require(threw_bad_extent, "expected non-positive ROI extent to throw");

           bool threw_empty_inputs = false;
           try {
             std::vector<simaai::neat::PreprocessRoi> rois = {{0, 0, 0, 3, 2}};
             simaai::neat::stages::internal::validate_preproc_roi_list_request_for_stage(
                 std::vector<cv::Mat>{}, rois);
           } catch (const std::exception& e) {
             threw_empty_inputs = true;
             require_contains(e, "inputs must not be empty");
           }
           require(threw_empty_inputs, "expected ROI-list request with no source images to throw");

           bool threw_mismatch = false;
           try {
             cv::Mat wrong_shape(3, 3, CV_8UC3);
             (void)simaai::neat::stages::internal::make_cpu_packed_preproc_source_batch_for_stage(
                 std::vector<cv::Mat>{img0, wrong_shape}, src_opt, meta);
           } catch (const std::exception& e) {
             threw_mismatch = true;
             require_contains(e, "matching size");
           }
           require(threw_mismatch, "expected mismatched source image geometry to throw");
         }));
