#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
/**
 * @example preproc_roi_boxdecode_moving_test.cpp
 * One built Preproc ROI -> MLA -> SimaBoxDecode graph over an ROI that moves between frames
 * (#1507).
 *
 * The same image patch sits at two positions of a canvas, and two same-sized ROIs select it.
 * The model sees identical pixels both times, so every detection must move by exactly the offset
 * between the ROIs (within rounding). Reusing the first ROI's transform for the second frame
 * would leave the boxes where they were.
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
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kTopK = 100;
constexpr int kTimeoutMs = 30000;
constexpr float kMinScore = 0.5f;
// Coordinate rounding of two ROI placements and the box decoder.
constexpr float kTolerancePx = 3.0f;

std::vector<objdet::Box> decode(simaai::neat::Run& run, const simaai::neat::Sample& wire,
                                const cv::Size& canvas, const std::string& label) {
  const simaai::neat::Sample out = run.run(simaai::neat::Sample{wire}, kTimeoutMs);
  require(out.size() == 1U, label + ": expected one output sample");
  std::vector<uint8_t> payload;
  std::string err;
  require(objdet::extract_bbox_payload(out.front(), payload, err), label + ": " + err);
  auto boxes = objdet::parse_boxes_lenient(payload, canvas.width, canvas.height, kTopK);
  boxes.erase(std::remove_if(boxes.begin(), boxes.end(),
                             [](const objdet::Box& b) { return b.score < kMinScore; }),
              boxes.end());
  return boxes;
}

bool near(const objdet::Box& a, const objdet::Box& b, float dx, float dy) {
  return a.class_id == b.class_id && std::abs(a.x1 + dx - b.x1) <= kTolerancePx &&
         std::abs(a.y1 + dy - b.y1) <= kTolerancePx && std::abs(a.x2 + dx - b.x2) <= kTolerancePx &&
         std::abs(a.y2 + dy - b.y2) <= kTolerancePx;
}

// Every box of `from` must have a counterpart in `to`, shifted by (dx, dy).
void require_shifted(const std::vector<objdet::Box>& from, const std::vector<objdet::Box>& to,
                     float dx, float dy, const std::string& label) {
  for (const auto& box : from) {
    const bool found = std::any_of(to.begin(), to.end(), [&](const objdet::Box& candidate) {
      return near(box, candidate, dx, dy);
    });
    require(found, label + ": no box at the expected position for " + objdet::format_box(box) +
                       " shifted by " + std::to_string(dx) + "," + std::to_string(dy));
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    fs::path root = argc > 1 && argv[1][0] != '-' ? fs::path(argv[1]) : fs::current_path();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    const std::string tar_gz = sima_yolov8_test::resolve_yolov8s_tar_or_skip(root);
    const cv::Mat patch = sima_yolov8_test::load_people_image_or_skip(root);

    simaai::neat::Model::Options opt;
    opt.preprocess.kind = simaai::neat::InputKind::Image;
    opt.preprocess.enable = simaai::neat::AutoFlag::On;
    opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
    opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
    opt.decode_type = simaai::neat::BoxDecodeType::YoloV8;
    opt.score_threshold = kMinScore;
    opt.nms_iou_threshold = 0.5f;
    opt.top_k = kTopK;
    opt.upstream_name = "decoder";
    const simaai::neat::Model model(tar_gz, opt);

    // The same patch at two offsets of one canvas.
    constexpr int kShiftX = 600, kShiftY = 300;
    const cv::Size canvas(kShiftX + patch.cols, kShiftY + patch.rows);
    require(canvas.width <= 1920 && canvas.height <= 1080,
            "the canvas exceeds the default dynamic image capacity");
    cv::Mat frame(canvas, patch.type(), cv::Scalar(114, 114, 114));
    patch.copyTo(frame(cv::Rect(0, 0, patch.cols, patch.rows)));
    cv::Mat moved_frame = frame.clone();
    moved_frame.setTo(cv::Scalar(114, 114, 114));
    patch.copyTo(moved_frame(cv::Rect(kShiftX, kShiftY, patch.cols, patch.rows)));
    const simaai::neat::PreprocessRoi first{0, 0, 0, patch.cols, patch.rows};
    const simaai::neat::PreprocessRoi second{0, kShiftX, kShiftY, patch.cols, patch.rows};

    // Real Preproc publication for each frame, decoded by one built Infer + BoxDecode graph.
    const auto wire = [&](const cv::Mat& image, const simaai::neat::PreprocessRoi& roi) {
      const simaai::neat::TensorList pre = simaai::neat::stages::Preproc(
          std::vector<cv::Mat>{image}, model, std::vector<simaai::neat::PreprocessRoi>{roi});
      require(pre.size() == 1U, "Preproc must return one tensor");
      return simaai::neat::sample_from_tensors(pre);
    };
    simaai::neat::Graph graph;
    graph.add(simaai::neat::nodes::Input());
    graph.add(simaai::neat::nodes::groups::Infer(model));
    graph.add(simaai::neat::nodes::SimaBoxDecode(model, simaai::neat::BoxDecodeType::YoloV8,
                                                 kMinScore, 0.5f, kTopK));
    graph.add(simaai::neat::nodes::Output());
    auto run = graph.build_seeded_internal(simaai::neat::Sample{wire(frame, first)},
                                           simaai::neat::RunMode::Sync);

    const auto at_first = decode(run, wire(frame, first), canvas, "first ROI");
    require(!at_first.empty(), "the first ROI produced no detections");
    const auto at_second = decode(run, wire(moved_frame, second), canvas, "second ROI");
    require_shifted(at_first, at_second, float(kShiftX), float(kShiftY), "second ROI");
    // Returning to the first ROI must not keep the second ROI's transform.
    const auto again = decode(run, wire(frame, first), canvas, "first ROI again");
    require_shifted(at_first, again, 0.0f, 0.0f, "first ROI again");
    run.close_input();
    run.close();
    std::cout << "[OK] preproc_roi_boxdecode_moving_test boxes=" << at_first.size() << "\n";
    return 0;
  } catch (const SkipTest& e) {
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
