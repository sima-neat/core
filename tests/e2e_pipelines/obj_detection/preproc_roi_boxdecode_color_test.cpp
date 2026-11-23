#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
/**
 * @example preproc_roi_boxdecode_color_test.cpp
 * Real Preproc ROI -> MLA -> SimaBoxDecode with RGB and BGR source frames (#1507).
 *
 * Detections of a full frame are the oracle: boxes decoded from an ROI of the same frame, which
 * Preproc letterboxes or stretches into the model canvas, must land on the same source-frame
 * pixels. The ROI moves between frames of one built graph, so the per-frame affine is exercised
 * with the real Preproc publisher, not with hand-built metadata.
 */
#include "model/Model.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/Graph.h"
#include "pipeline/StageRun.h"

#include "e2e_pipelines/e2e_utils.h"
#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "test_utils.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kTopK = 100;
constexpr int kTimeoutMs = 30000;
constexpr float kMinScore = 0.55f;
constexpr float kRoiMargin = 8.0f;

enum class Color { Bgr, Rgb };

const char* color_name(Color color) {
  return color == Color::Bgr ? "BGR" : "RGB";
}

struct Variant {
  Color color;
  bool stretch;
  float min_iou;
  int min_matches;
  std::string label() const {
    return std::string(color_name(color)) + (stretch ? "-stretch" : "-letterbox");
  }
};

simaai::neat::Model make_model(const std::string& tar_gz, const Variant& variant) {
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  opt.preprocess.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.color_convert.input_format = variant.color == Color::Bgr
                                                  ? simaai::neat::PreprocessColorFormat::BGR
                                                  : simaai::neat::PreprocessColorFormat::RGB;
  if (variant.stretch) {
    opt.preprocess.resize.enable = simaai::neat::AutoFlag::On;
    opt.preprocess.resize.width = 640;
    opt.preprocess.resize.height = 640;
    opt.preprocess.resize.mode = simaai::neat::ResizeMode::Stretch;
  }
  opt.decode_type = simaai::neat::BoxDecodeType::YoloV8;
  opt.score_threshold = kMinScore;
  opt.nms_iou_threshold = 0.5f;
  opt.top_k = kTopK;
  opt.upstream_name = "decoder";
  return simaai::neat::Model(tar_gz, opt);
}

std::vector<objdet::Box> boxes_from(const simaai::neat::Sample& decoded, const cv::Mat& frame,
                                    const std::string& label) {
  std::vector<uint8_t> payload;
  std::string err;
  require(objdet::extract_bbox_payload(decoded, payload, err), label + ": " + err);
  auto boxes = objdet::parse_boxes_lenient(payload, frame.cols, frame.rows, kTopK);
  boxes.erase(std::remove_if(boxes.begin(), boxes.end(),
                             [](const objdet::Box& b) { return b.score < kMinScore; }),
              boxes.end());
  return boxes;
}

// Preproc (optionally with one ROI) -> Infer, as the wire sample a BoxDecode graph consumes.
simaai::neat::Sample preprocess_and_infer(const cv::Mat& frame, const simaai::neat::Model& model,
                                          const simaai::neat::PreprocessRoi* roi) {
  const simaai::neat::TensorList pre =
      roi ? simaai::neat::stages::Preproc(std::vector<cv::Mat>{frame}, model,
                                          std::vector<simaai::neat::PreprocessRoi>{*roi})
          : simaai::neat::stages::Preproc(std::vector<cv::Mat>{frame}, model);
  require(pre.size() == 1U, "Preproc must return one tensor");
  const simaai::neat::Sample inferred = simaai::neat::stages::Infer(
      simaai::neat::Sample{simaai::neat::sample_from_tensors(pre)}, model);
  require(inferred.size() == 1U, "Infer should return exactly one sample");
  return inferred.front();
}

std::vector<objdet::Box> decode(const cv::Mat& frame, const simaai::neat::Model& model,
                                const simaai::neat::PreprocessRoi* roi, const std::string& label) {
  simaai::neat::stages::BoxDecodeOptions box_opt(simaai::neat::BoxDecodeType::YoloV8);
  box_opt.detection_threshold = kMinScore;
  box_opt.nms_iou_threshold = 0.5f;
  box_opt.top_k = kTopK;
  const auto inferred = preprocess_and_infer(frame, model, roi);
  const simaai::neat::Sample out =
      simaai::neat::stages::BoxDecode(simaai::neat::Sample{inferred}, model, box_opt);
  require(out.size() == 1U, label + ": BoxDecode should return one sample");
  return boxes_from(out.front(), frame, label);
}

bool inside(const objdet::Box& box, const simaai::neat::PreprocessRoi& roi, float margin) {
  return box.x1 >= roi.x - margin && box.y1 >= roi.y - margin &&
         box.x2 <= roi.x + roi.width + margin && box.y2 <= roi.y + roi.height + margin;
}

// Every full-frame box that lies inside the ROI must be found again, in source pixels; no ROI box
// may leave the ROI.
void check_roi(const std::vector<objdet::Box>& full, const std::vector<objdet::Box>& roi_boxes,
               const simaai::neat::PreprocessRoi& roi, const Variant& variant,
               const std::string& label) {
  int expected = 0, matched = 0;
  for (const auto& reference : full) {
    if (!inside(reference, roi, 0.0f))
      continue;
    ++expected;
    float best = 0.0f;
    for (const auto& box : roi_boxes)
      if (box.class_id == reference.class_id)
        best = std::max(best, objdet::box_iou_xyxy(reference.x1, reference.y1, reference.x2,
                                                   reference.y2, box.x1, box.y1, box.x2, box.y2));
    if (best >= variant.min_iou)
      ++matched;
    else
      std::cerr << "[WARN] " << label << ": no ROI box for full-frame "
                << objdet::format_box(reference) << " (best iou " << best << ")\n";
  }
  require(expected >= variant.min_matches,
          label + ": the ROI contains too few full-frame detections to compare");
  require(matched == expected, label + ": " + std::to_string(expected - matched) + " of " +
                                   std::to_string(expected) +
                                   " ROI detections differ from the full frame in source pixels");
  require(!roi_boxes.empty(), label + ": no detections from the ROI");
  for (const auto& box : roi_boxes)
    require(inside(box, roi, kRoiMargin),
            label + ": an ROI detection lies outside the ROI: " + objdet::format_box(box));
}

void run_variant(const std::string& tar_gz, const cv::Mat& bgr, const Variant& variant) {
  const std::string tag = variant.label();
  const simaai::neat::Model model = make_model(tar_gz, variant);
  const cv::Mat frame = [&] {
    if (variant.color == Color::Bgr)
      return bgr;
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
  }();
  const auto full = decode(frame, model, nullptr, tag + " full frame");
  require(!full.empty(), tag + ": the full frame has no detections");

  // Two ROIs of different aspect, each holding the same people as the full frame.
  const simaai::neat::PreprocessRoi first{0, 96, 16, 1056, 704};
  const simaai::neat::PreprocessRoi second{0, 16, 0, 1136, 720};
  require(first.x + first.width <= frame.cols && first.y + first.height <= frame.rows &&
              second.x + second.width <= frame.cols && second.y + second.height <= frame.rows,
          "the ROIs must lie inside the test image");

  // Stage route: Preproc with the ROI, then Infer and BoxDecode.
  check_roi(full, decode(frame, model, &first, tag + " ROI 1"), first, variant, tag + " ROI 1");
  check_roi(full, decode(frame, model, &second, tag + " ROI 2"), second, variant, tag + " ROI 2");

  // Graph route: one built Infer + BoxDecode graph decodes ROI frames that move between runs,
  // each frame with the affine its own Preproc published.
  simaai::neat::Graph graph;
  graph.add(simaai::neat::nodes::Input());
  graph.add(simaai::neat::nodes::groups::Infer(model));
  graph.add(simaai::neat::nodes::SimaBoxDecode(model, simaai::neat::BoxDecodeType::YoloV8,
                                               kMinScore, 0.5f, kTopK));
  graph.add(simaai::neat::nodes::Output());
  const auto wire = [&](const simaai::neat::PreprocessRoi& roi) {
    const simaai::neat::TensorList pre = simaai::neat::stages::Preproc(
        std::vector<cv::Mat>{frame}, model, std::vector<simaai::neat::PreprocessRoi>{roi});
    require(pre.size() == 1U, "Preproc must return one tensor");
    return simaai::neat::sample_from_tensors(pre);
  };
  auto run =
      graph.build_seeded_internal(simaai::neat::Sample{wire(first)}, simaai::neat::RunMode::Sync);
  int index = 0;
  for (const auto* roi : {&first, &second, &first}) {
    const std::string label = tag + " graph frame " + std::to_string(index++);
    const simaai::neat::Sample out = run.run(simaai::neat::Sample{wire(*roi)}, kTimeoutMs);
    require(out.size() == 1U, label + ": expected one output sample");
    check_roi(full, boxes_from(out.front(), frame, label), *roi, variant, label);
  }
  run.close_input();
  run.close();
  std::cout << "[OK] " << tag << " full=" << full.size() << " boxes\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    fs::path root = argc > 1 && argv[1][0] != '-' ? fs::path(argv[1]) : fs::current_path();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    const std::string tar_gz = sima_yolov8_test::resolve_yolov8s_tar_or_skip(root);
    const cv::Mat bgr = sima_yolov8_test::load_people_image_or_skip(root);
    require(bgr.cols >= 1152 && bgr.rows >= 720, "the people image is smaller than the test ROIs");

    // A letterboxed ROI keeps the detector's accuracy; a stretched one distorts the people, so the
    // stretch comparison is looser and needs fewer matches.
    for (const Variant& variant :
         {Variant{Color::Bgr, false, 0.7f, 2}, Variant{Color::Rgb, false, 0.7f, 2},
          Variant{Color::Bgr, true, 0.5f, 1}, Variant{Color::Rgb, true, 0.5f, 1}})
      run_variant(tar_gz, bgr, variant);
    std::cout << "[OK] preproc_roi_boxdecode_color_test\n";
    return 0;
  } catch (const SkipTest& e) {
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
