#include "model/Model.h"
#include "pipeline/StageRun.h"
#include "pipeline/TensorAdapters.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/TensorUtil.h"

#include "e2e_pipelines/e2e_utils.h"
#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "test_utils.h"

#include <gst/gst.h>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {

simaai::neat::PreprocessRuntimeMeta require_preprocess_meta(const simaai::neat::Tensor& tensor,
                                                            const std::string& label) {
#if !SIMA_HAS_SIMAAI_POOL
  (void)tensor;
  skip_test_exception(label + ": simaai pool/meta unavailable");
  return {};
#else
  const auto holder = simaai::neat::pipeline_internal::holder_from_tensor(tensor);
  require(holder != nullptr, label + ": missing tensor holder");
  GstBuffer* buffer = simaai::neat::pipeline_internal::buffer_from_tensor_holder(holder);
  require(buffer != nullptr, label + ": missing GstBuffer");
  const auto meta = simaai::neat::read_simaai_preprocess_meta(buffer);
  gst_buffer_unref(buffer);
  require(meta.has_value(), label + ": missing preprocess metadata");
  return *meta;
#endif
}

simaai::neat::Tensor make_corrupt_meta_tensor_copy(const simaai::neat::Tensor& tensor,
                                                   const std::string& label) {
#if !SIMA_HAS_SIMAAI_POOL
  (void)tensor;
  skip_test_exception(label + ": simaai pool/meta unavailable");
  return {};
#else
  const auto holder = simaai::neat::pipeline_internal::holder_from_tensor(tensor);
  require(holder != nullptr, label + ": missing tensor holder");
  auto* sample = static_cast<GstSample*>(holder.get());
  require(sample != nullptr && GST_IS_SAMPLE(sample), label + ": holder is not GstSample");
  GstBuffer* src_buf = gst_sample_get_buffer(sample);
  require(src_buf != nullptr, label + ": source sample missing GstBuffer");
  const auto src_meta = simaai::neat::read_simaai_preprocess_meta(src_buf);
  require(src_meta.has_value(), label + ": source buffer missing preprocess metadata");

  // Keep original segmented memory backing; deep-copy can materialize plain
  // memory and break standalone boxdecode segmented-input contract.
  GstBuffer* copied_buf = gst_buffer_copy(src_buf);
  require(copied_buf != nullptr, label + ": failed to copy GstBuffer");

  auto corrupt = *src_meta;
  corrupt.original_width = 0;
  require(simaai::neat::write_simaai_preprocess_meta(copied_buf, corrupt),
          label + ": failed to write corrupt preprocess metadata");

  const GstStructure* src_info = gst_sample_get_info(sample);
  GstStructure* info_copy = src_info ? gst_structure_copy(src_info) : nullptr;
  GstSample* copied_sample = gst_sample_new(copied_buf, gst_sample_get_caps(sample),
                                            gst_sample_get_segment(sample), info_copy);
  gst_buffer_unref(copied_buf);
  require(copied_sample != nullptr, label + ": failed to build copied GstSample");
  auto copied_tensor = simaai::neat::from_gst_sample(copied_sample);
  gst_sample_unref(copied_sample);
  return copied_tensor;
#endif
}

} // namespace

int main(int argc, char** argv) {
  try {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    const std::string tar_gz = sima_yolov8_test::resolve_yolov8s_tar_or_skip(root);
    const cv::Mat img_bgr = sima_yolov8_test::load_people_image_or_skip(root);

    simaai::neat::Model::Options model_opt;
    model_opt.preprocess.kind = simaai::neat::InputKind::Image;
    model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
    model_opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
    model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
    model_opt.upstream_name = "decoder";
    simaai::neat::Model model(tar_gz, model_opt);

    cv::Mat frame0 = img_bgr.clone();
    cv::Mat frame1;
    cv::resize(img_bgr, frame1,
               cv::Size(std::max(64, img_bgr.cols / 2), std::max(64, img_bgr.rows / 2)));

    simaai::neat::stages::BoxDecodeOptions box_opt(simaai::neat::BoxDecodeType::YoloV8);
    box_opt.detection_threshold = 0.25;
    box_opt.nms_iou_threshold = 0.5;
    box_opt.top_k = 100;

    for (const auto& frame : {frame0, frame1}) {
      auto pre = simaai::neat::stages::Preproc(std::vector<cv::Mat>{frame}, model);
      auto infer = simaai::neat::stages::Infer(pre, model);
      require(!infer.empty(), "sync-meta: infer returned no tensors");
      const auto meta = require_preprocess_meta(infer.front(), "sync-meta");

      require(meta.original_width == frame.cols && meta.original_height == frame.rows,
              "sync-meta: original geometry mismatch");
      require(meta.affine_scale_x > 0.0 && meta.affine_scale_y > 0.0,
              "sync-meta: affine scales must be positive");

      (void)simaai::neat::stages::BoxDecode(
          simaai::neat::Sample{simaai::neat::sample_from_tensors(infer)}, model, box_opt);
    }

#if SIMA_HAS_SIMAAI_POOL
    // Negative case: corrupt required metadata and verify strict no-fallback failure.
    {
      auto pre = simaai::neat::stages::Preproc(std::vector<cv::Mat>{frame0}, model);
      auto infer = simaai::neat::stages::Infer(pre, model);
      require(!infer.empty(), "sync-meta-negative: infer returned no tensors");
      auto corrupt = make_corrupt_meta_tensor_copy(infer.front(), "sync-meta-negative");

      bool threw = false;
      try {
        (void)simaai::neat::stages::BoxDecode(
            simaai::neat::Sample{
                simaai::neat::sample_from_tensors(simaai::neat::TensorList{corrupt})},
            model, box_opt);
      } catch (const std::exception& e) {
        threw = true;
        const std::string msg = e.what();
        require_contains(msg, "preprocess metadata contract violation",
                         "sync-meta-negative: missing contract error wording");
        require_contains(msg, "no fallback allowed",
                         "sync-meta-negative: missing no-fallback wording");
      }
      require(threw, "sync-meta-negative: expected BoxDecode to fail on corrupt metadata");
    }
#endif

    std::cout << "[OK] yolov8_preprocess_meta_sync_test passed\n";
    return 0;
  } catch (const SkipTest& e) {
    std::cout << "[SKIP] " << e.what() << "\n";
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    if (is_dispatcher_unavailable(e.what())) {
      return skip_long_test("dispatcher unavailable");
    }
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
